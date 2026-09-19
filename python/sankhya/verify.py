"""Span verification: is every token of a decoded span independently
justified by the exported lexicon?

A span is **verified** when each of its (class, text) tokens is explained by
the lexicon or by a purely structural rule (separator / digits / punctuation /
range marker). The value of a verified span is then a pure function of the
lexicon plus the arithmetic core, both unit-tested -- so strict mode
("the right number or nothing") can drop everything else.

This mirrors `src/verify.ts` 1:1; see the verified-tier contract.

    from sankhya.verify import verify_tokens
    verify_tokens([("PFX_SAADHE", "sava"), ("SEP", " "), ("UNIT_LAKH", "lakh")])
"""
from __future__ import annotations

import json
import os
import unicodedata
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

__all__ = [
    "load_lexicon",
    "verify_tokens",
    "union_forms",
    "union_range_words",
    "union_bound_forms",
    "is_bound_form",
    "union_word_suffixes",
    "union_word_oblique_endings",
    "is_lexicon_o_only",
    "union_ambiguous_forms",
    "union_blocked_surfaces",
    "is_ambiguous_form",
    "is_blocked_surface",
    "is_known_form",
    "is_near_known_form",
    "levenshtein_le",
    "has_blocked_surface",
    "should_drop_ambiguous_form",
    "should_drop_loose_symbol_unit",
    "should_drop_unsupported_bare_cardinal",
    "should_drop_unknown_unit",
]

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LEXICON_PATH = os.path.join(_REPO_ROOT, "src", "data", "lexicon.json")

# structural range markers, in addition to the packs' alphabetic range words
RANGE_SYMBOLS = {"-", "–", "—", "/"}

# Indic digit glyphs a DIGITS token may legitimately be written with.
# Mirrors src/verify.ts isDigitsOnly and charset.NATIVE_DIGIT_BLOCKS; the
# arithmetic core normalises all of these to ASCII before evaluating.
_NATIVE_DIGITS = frozenset(
    chr(base + i) for base in (0x0966, 0x0AE6) for i in range(10)
)

# module-level caches of the union maps
_LEXICON: Optional[dict] = None
_FORMS: Optional[Dict[str, Set[str]]] = None
_RANGE_WORDS: Optional[Set[str]] = None
_BOUND: Optional[Dict[str, Dict[str, Set[str]]]] = None
_SUFFIXES: Optional[List[str]] = None
_OBLIQUES: Optional[List[str]] = None
_PACK_VIEWS: Optional[List[dict]] = None
_AMBIGUOUS: Optional[Set[str]] = None
_BLOCKED: Optional[Set[str]] = None

# R15: the minimum number of characters a suffix-stripped head must still
# have before it may be accepted as an inflected lexicon form. Without it
# a long case ending eats almost the whole word and any proper noun that
# happens to end in one ("हजारे" -> "हजार" is fine at 4, but a 2-char head
# is noise) verifies as a number.
MIN_SUFFIX_HEAD_LEN = 3

# R16: ambiguous SYMBOL unit surfaces at most this long ("k", "l", "m",
# "b") only count when written GLUED to the digits they scale ("20k"),
# never separated by a space -- the shipped gold has no spaced 1-2 char
# symbol unit, while real text is full of "1996 k" (a clitic) and
# "33 k. m." (kilometres).
MAX_GLUED_SYMBOL_LEN = 2


def _reset_caches():
    global _FORMS, _RANGE_WORDS, _BOUND, _SUFFIXES, _OBLIQUES
    global _PACK_VIEWS, _AMBIGUOUS, _BLOCKED
    _FORMS = _RANGE_WORDS = _BOUND = _SUFFIXES = _OBLIQUES = None
    _PACK_VIEWS = _AMBIGUOUS = _BLOCKED = None


def load_lexicon(path: str = None) -> dict:
    """Load (and cache) the exported lexicon JSON.

    Reads `src/data/lexicon.json` relative to the repo; if that file is
    missing, builds the same object in-process from the registered language
    packs via `export_lexicon.build()`.
    """
    global _LEXICON, _FORMS, _RANGE_WORDS, _BOUND, _SUFFIXES, _OBLIQUES
    if path is None and _LEXICON is not None:
        return _LEXICON
    p = path or LEXICON_PATH
    try:
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except FileNotFoundError:
        from .export_lexicon import build
        obj = build()
    if path is None:
        _LEXICON = obj
        _reset_caches()
    return obj


def _build_union(lex: dict):
    forms: Dict[str, Set[str]] = {}
    range_words: Set[str] = set()
    # surface -> class -> set of unit surfaces it may immediately precede
    bound: Dict[str, Dict[str, Set[str]]] = {}
    for pack in lex.get("packs", {}).values():
        for surface, cls in pack.get("forms", {}).items():
            forms.setdefault(_norm(surface), set()).add(cls)
        for w in pack.get("range_words", []):
            range_words.add(_norm(w))
        for surface, spec in (pack.get("bound_forms") or {}).items():
            slot = bound.setdefault(_norm(surface), {}).setdefault(spec["cls"], set())
            slot.update(_norm(u) for u in spec.get("before", ()))
    return forms, range_words, bound


def _build_endings(lex: dict):
    """(case endings, oblique stem endings) across every pack, longest first.

    Mirrors the pack fields `word_suffixes` / `word_oblique_endings` that
    llm_corpus._strip_word_suffixes uses when it resolves an inflected word
    ("लाखांचं") to its head class.
    """
    suffixes: Set[str] = set()
    obliques: Set[str] = set()
    for pack in lex.get("packs", {}).values():
        for s in pack.get("word_suffixes", []) or []:
            suffixes.add(_norm(s))
        for o in pack.get("word_oblique_endings", []) or []:
            obliques.add(_norm(o))
    return (sorted(suffixes, key=lambda x: (-len(x), x)),
            sorted(obliques, key=lambda x: (-len(x), x)))


def union_word_suffixes(lex: dict = None) -> List[str]:
    """Every pack's declared case endings, longest first."""
    global _SUFFIXES, _OBLIQUES
    if lex is not None:
        return _build_endings(lex)[0]
    if _SUFFIXES is None:
        _SUFFIXES, _OBLIQUES = _build_endings(load_lexicon())
    return _SUFFIXES


def union_word_oblique_endings(lex: dict = None) -> List[str]:
    """Every pack's declared oblique stem endings, longest first."""
    global _SUFFIXES, _OBLIQUES
    if lex is not None:
        return _build_endings(lex)[1]
    if _OBLIQUES is None:
        _SUFFIXES, _OBLIQUES = _build_endings(load_lexicon())
    return _OBLIQUES


def _strip_one_suffix_heads(surface: str, suffixes, obliques):
    """Yield the candidate head surfaces of `surface` after removing ONE
    declared case ending (longest first), and, under it, one declared
    oblique stem ending -- exactly the shapes llm_corpus._strip_word_suffixes
    yields on its first round. `surface` must already be _norm-ed.
    """
    for suf in suffixes:
        if not surface.endswith(suf) or len(surface) <= len(suf):
            continue
        stem = surface[: -len(suf)]
        yield stem
        for obl in obliques:
            if stem.endswith(obl) and len(stem) > len(obl):
                yield stem[: -len(obl)]


def union_forms(lex: dict = None) -> Dict[str, Set[str]]:
    """surface(lower, NFC) -> set of classes, across every pack."""
    global _FORMS, _RANGE_WORDS, _BOUND
    if lex is not None:
        return _build_union(lex)[0]
    if _FORMS is None:
        _FORMS, _RANGE_WORDS, _BOUND = _build_union(load_lexicon())
    return _FORMS


def union_range_words(lex: dict = None) -> Set[str]:
    global _FORMS, _RANGE_WORDS, _BOUND
    if lex is not None:
        return _build_union(lex)[1]
    if _RANGE_WORDS is None:
        _FORMS, _RANGE_WORDS, _BOUND = _build_union(load_lexicon())
    return _RANGE_WORDS


def union_bound_forms(lex: dict = None) -> Dict[str, Dict[str, Set[str]]]:
    """surface -> class -> the unit surfaces that surface may precede.

    Bound forms are deliberately absent from `union_forms()`: they are only
    ever justified in context (Gujarati "બ" is CARD_2 in બસો and nothing
    at all on its own), so `verify_tokens` consults this map only when the
    NEXT token's text is one of the listed unit surfaces.
    """
    global _FORMS, _RANGE_WORDS, _BOUND
    if lex is not None:
        return _build_union(lex)[2]
    if _BOUND is None:
        _FORMS, _RANGE_WORDS, _BOUND = _build_union(load_lexicon())
    return _BOUND


def is_bound_form(surface: str, cls: str, next_surface: Optional[str], lex: dict = None) -> bool:
    """True when `surface` is a declared BOUND number form of class `cls`
    that may attach to `next_surface` (Gujarati CARD_2 "બ" before UNIT_SAU
    "સો" = બસો). Shared by verify_tokens and the decoder's fused-word
    partition (decode._bound_keep_indices), so both runtimes agree on
    which 1-character sub-runs stand on lexicon evidence.
    """
    if next_surface is None:
        return False
    bound = union_bound_forms(lex)
    allowed = bound.get(_norm(surface), {}).get(cls)
    return bool(allowed) and _norm(next_surface) in allowed


def is_lexicon_o_only(surface: str, lex: dict = None) -> bool:
    """R9: True when `surface` appears in the lexicon union with class "O"
    and NO other class -- an ordinary word (an indefinite plural such as
    "karodon"/"करोडो") that some pack has explicitly declared a
    non-number. A surface that ALSO carries a real number class in some
    other pack (a cross-pack conflict) is NOT O-only and is left alone.
    """
    cls = union_forms(lex).get(_norm(surface))
    return cls is not None and cls == {"O"}


def _norm(s: str) -> str:
    return unicodedata.normalize("NFC", s).lower()


def _is_digit_char(c: str) -> bool:
    return ("0" <= c <= "9") or (c in _NATIVE_DIGITS)


def _verify_suffixed(cls: str, surface: str, forms, views) -> bool:
    """A PFX_*/CARD_*/UNIT_* token whose exact surface is unknown may still
    be an inflected form of a known head: Marathi "लाखांचं" is UNIT_LAKH
    (लाख + oblique "ां" + ending "चं"), Gujarati "કરોડનો" is UNIT_CRORE.

    Only ONE declared ending is removed, the head must carry the token's own
    class, and a surface that is itself a full lexicon form is never stripped
    (that is handled by the exact-match check before this is reached).

    R15 (real-text tightening, see data_wild/REPORT.md §4.1). The strip is
    PER PACK: the ending must be declared by the SAME pack that declares the
    head form, so Marathi's "-े" can no longer inflect a Hindi-only head (or
    vice versa), and the remaining head must still be at least
    MIN_SUFFIX_HEAD_LEN characters. A blocked surface is never stripped at
    all -- "हजारे" (the surname) stays unverified even though "हजार" + "-े"
    would otherwise explain it.
    """
    if surface in forms or surface in union_blocked_surfaces():
        return False
    for view in views:
        suffixes = view["suffixes"]
        if not suffixes:
            continue
        pack_forms = view["forms"]
        for head in _strip_one_suffix_heads(surface, suffixes, view["obliques"]):
            if len(head) < MIN_SUFFIX_HEAD_LEN:
                continue
            if cls in pack_forms.get(head, ()):
                return True
    return False


def _build_pack_views(lex: dict) -> List[dict]:
    """One entry per pack: its own forms/suffixes/obliques/symbol forms.

    Suffix stripping (R15) and the symbol-unit rule (R16) are per-pack
    questions -- a union of every pack's case endings lets one language's
    morphology justify another's word.
    """
    views = []
    for pack in lex.get("packs", {}).values():
        forms: Dict[str, Set[str]] = {}
        for surface, cls in pack.get("forms", {}).items():
            forms.setdefault(_norm(surface), set()).add(cls)
        views.append({
            "forms": forms,
            "suffixes": sorted({_norm(s) for s in pack.get("word_suffixes", []) or []},
                               key=lambda x: (-len(x), x)),
            "obliques": sorted({_norm(o) for o in pack.get("word_oblique_endings", []) or []},
                               key=lambda x: (-len(x), x)),
        })
    return views


def pack_views(lex: dict = None) -> List[dict]:
    global _PACK_VIEWS
    if lex is not None:
        return _build_pack_views(lex)
    if _PACK_VIEWS is None:
        _PACK_VIEWS = _build_pack_views(load_lexicon())
    return _PACK_VIEWS


def union_ambiguous_forms(lex: dict = None) -> Set[str]:
    """Every pack's `ambiguous_forms`: lexicon surfaces that are also
    ordinary words in that language (see LanguagePack.ambiguous_forms)."""
    global _AMBIGUOUS
    if lex is not None:
        return {_norm(a) for p in lex.get("packs", {}).values()
                for a in p.get("ambiguous_forms", []) or []}
    if _AMBIGUOUS is None:
        _AMBIGUOUS = union_ambiguous_forms(load_lexicon())
    return _AMBIGUOUS


def union_blocked_surfaces(lex: dict = None) -> Set[str]:
    """Every pack's `blocked_surfaces`: surfaces that must never verify or
    decode as a number word (proper nouns built on number words)."""
    global _BLOCKED
    if lex is not None:
        return {_norm(b) for p in lex.get("packs", {}).values()
                for b in p.get("blocked_surfaces", []) or []}
    if _BLOCKED is None:
        _BLOCKED = union_blocked_surfaces(load_lexicon())
    return _BLOCKED


def is_ambiguous_form(surface: str, lex: dict = None) -> bool:
    return _norm(surface) in union_ambiguous_forms(lex)


def is_blocked_surface(surface: str, lex: dict = None) -> bool:
    return _norm(surface) in union_blocked_surfaces(lex)


def levenshtein_le(a: str, b: str, k: int = 1) -> bool:
    """True when the edit distance between `a` and `b` is at most `k`.

    A tiny bounded implementation (the lexicon is small: a few hundred
    surfaces per class). Mirrors src/verify.ts levenshteinLE.
    """
    if abs(len(a) - len(b)) > k:
        return False
    if a == b:
        return True
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        best = i
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            v = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            cur.append(v)
            best = min(best, v)
        if best > k:
            return False
        prev = cur
    return prev[-1] <= k


def is_known_form(cls: str, surface: str, lex: dict = None) -> bool:
    """True when `surface` is an exact lexicon form of `cls`, or an
    inflected one (R15 per-pack suffix strip). Blocked surfaces never are."""
    s = _norm(surface)
    if s in union_blocked_surfaces(lex):
        return False
    forms = union_forms(lex)
    if cls in forms.get(s, ()):
        return True
    return _verify_suffixed(cls, s, forms, pack_views(lex))


def is_near_known_form(cls: str, surface: str, max_distance: int = 1, lex: dict = None) -> bool:
    """R17: `surface` is a known form of `cls` (see is_known_form) or within
    `max_distance` edits of one -- the typo tolerance that keeps a noised
    unit word ("hazzar", "करोड") after digits from being dropped."""
    if is_known_form(cls, surface, lex):
        return True
    s = _norm(surface)
    if s in union_blocked_surfaces(lex):
        return False
    for form, classes in union_forms(lex).items():
        if cls in classes and levenshtein_le(s, form, max_distance):
            return True
    return False


def _verify_token(cls: str, text: str, forms, range_words,
                  bound=None, next_text: str = None,
                  views=(), blocked=frozenset()) -> bool:
    if cls == "SEP":
        return len(text) > 0 and text.strip() == ""
    if cls == "DOT":
        return text == "."
    if cls == "COMMA":
        return text == ","
    if cls == "RANGE":
        t = text.strip()
        if t in RANGE_SYMBOLS:
            return True
        return _norm(t) in range_words
    if cls == "DIGITS":
        return len(text) > 0 and all(_is_digit_char(c) for c in text)
    if cls == "O":
        return "O" in forms.get(_norm(text), ())
    # PFX_* / CARD_* / UNIT_*
    # R14: a surface a pack declares BLOCKED (a proper noun built on a
    # number word: "हजारे", "अरबी", "સવાઈ") never verifies as a number,
    # whatever class the model gave it.
    if _norm(text) in blocked:
        return False
    if cls in forms.get(_norm(text), ()):
        return True
    # ... or the same class wearing one of the packs' declared case endings
    # ("लाखांचं" = UNIT_LAKH, "કરોડનો" = UNIT_CRORE).
    if _verify_suffixed(cls, _norm(text), forms, views):
        return True
    # ... or a BOUND number form, justified only by what follows it: the
    # very next token must be one of the unit surfaces it attaches to
    # (Gujarati CARD_2 "બ" before UNIT_SAU "સો" = બસો). A standalone "બ",
    # or "બ" before anything else, stays unverified.
    if bound and next_text is not None:
        allowed = bound.get(_norm(text), {}).get(cls)
        if allowed and _norm(next_text) in allowed:
            return True
    return False


def _is_content(cls: str) -> bool:
    return cls == "DIGITS" or cls.startswith("PFX_") or cls.startswith("CARD_") or cls.startswith("UNIT_")


def verify_tokens(tokens: Sequence[Tuple[str, str]], lex: dict = None) -> bool:
    """True when every token of the span is justified by the lexicon.

    `tokens` are the span's (class, text) pairs exactly as passed to
    `core.evaluate`. `lex` overrides the cached lexicon (tests).
    """
    if not tokens:
        return False
    if lex is None:
        forms, range_words, bound = union_forms(), union_range_words(), union_bound_forms()
    else:
        forms, range_words, bound = _build_union(lex)
    views = pack_views(lex)
    blocked = union_blocked_surfaces(lex)
    has_content = False
    for i, (cls, text) in enumerate(tokens):
        next_text = tokens[i + 1][1] if i + 1 < len(tokens) else None
        if not _verify_token(cls, text, forms, range_words, bound, next_text,
                             views, blocked):
            return False
        if _is_content(cls):
            has_content = True
    return has_content


# --- span-level drop rules that need the lexicon -------------------------
#
# Every rule here is mirrored 1:1 in src/verify.ts and applied by the same
# gate in both runtimes (src/index.ts decodeForward /
# eval_gold._apply_bare_digits_gate). They exist because the tagger, on real
# text, happily labels ordinary words as number words -- see
# python/data_wild/REPORT.md §4.

_GLUE_CLASSES = ("SEP", "RANGE", "DOT", "COMMA", "O")


def _meaningful(tokens) -> List[Tuple[int, str, str]]:
    """(index, cls, text) for the PFX_/CARD_/UNIT_/DIGITS tokens of a span."""
    return [(i, cls, text) for i, (cls, text) in enumerate(tokens)
            if cls == "DIGITS" or cls.startswith(("PFX_", "CARD_", "UNIT_"))]


def has_blocked_surface(tokens: Sequence[Tuple[str, str]], lex: dict = None) -> bool:
    """R14: the span carries a meaningful token whose surface a pack
    declares blocked -- a proper noun built on a number word
    ("अण्णा हजारे", "सवाई तुकोजीराव", "अरबी समुद्र"). Dropped outright."""
    blocked = union_blocked_surfaces(lex)
    return any(_norm(text) in blocked for _i, _cls, text in _meaningful(tokens))


def should_drop_ambiguous_form(tokens: Sequence[Tuple[str, str]], lex: dict = None) -> bool:
    """R12: the span rests entirely on surfaces that are also ordinary words.

    A meaningful token is *justified* when it is DIGITS or a known lexicon
    form of its own class (`is_known_form`). If the span has at least one
    justified token and EVERY justified token's surface is in the packs'
    `ambiguous_forms`, nothing but an ordinary-word reading is holding the
    span up: "so", "sath", "arab", "अरब", "peti" standing alone, or
    "सऊदी अरब" where the only lexicon-backed word is the ethnonym.

    A single non-ambiguous justified token anywhere in the span is enough to
    keep it: "ek arab", "100 अरब", "do peti", "bees lac pachees lac",
    "unnis sau sath" all survive. Callers exempt a span with a currency
    marker (see _apply_bare_digits_gate)."""
    justified = [(cls, text) for _i, cls, text in _meaningful(tokens)
                 if cls == "DIGITS" or is_known_form(cls, text, lex)]
    if not justified:
        return False
    return all(cls != "DIGITS" and is_ambiguous_form(text, lex) for cls, text in justified)


def should_drop_loose_symbol_unit(tokens: Sequence[Tuple[str, str]], lex: dict = None) -> bool:
    """R16: a 1-2 character ambiguous SYMBOL unit ("k", "l", "m", "b") only
    counts when it is GLUED to the digits it scales.

    "20k"/"2.5L" keep their unit; "1996 k" (the ke/ki clitic after a year)
    and "33 k. m." (kilometres) lose theirs and the span goes. Longer symbol
    forms ("lac", "cr", "bn", "LPA") are commonly written with a space and
    are deliberately untouched."""
    for i, cls, text in _meaningful(tokens):
        if not cls.startswith("UNIT_"):
            continue
        s = _norm(text)
        if len(s) > MAX_GLUED_SYMBOL_LEN or not is_ambiguous_form(s, lex):
            continue
        if not is_known_form(cls, s, lex):
            continue
        prev_cls = tokens[i - 1][0] if i > 0 else None
        if prev_cls in ("DIGITS", "DOT", "COMMA"):
            continue  # glued straight onto the number
        return True
    return False


def should_drop_unsupported_bare_cardinal(tokens: Sequence[Tuple[str, str]], lex: dict = None) -> bool:
    """R13b: a span whose ONLY meaningful token is a spelled cardinal is
    kept only when its surface is an EXACT lexicon form of that class.

    "पचास" (50) and "બાવીસ" (22) are real bare-cardinal gold spans; "aavesh",
    "chaahie", "barabar", "इ" are ordinary words the tagger guessed a
    cardinal for. No suffix stripping and no edit-distance tolerance here:
    a bare cardinal has nothing else in the span to corroborate it."""
    meaningful = _meaningful(tokens)
    if len(meaningful) != 1:
        return False
    _i, cls, text = meaningful[0]
    if not cls.startswith("CARD_"):
        return False
    return cls not in union_forms(lex).get(_norm(text), ())


def should_drop_unknown_unit(tokens: Sequence[Tuple[str, str]], lex: dict = None) -> bool:
    """R17: a unit word that is not a unit word, next to bare digits.

    "3 hours" -> 3,000, "25वे" (an ordinal) -> 2,500,000, "1980ના" ->
    1.98e8, "chori"/"thought"/"oooh"/"खोड" alone: the tagger invents a unit
    class for whatever letters sit next to (or instead of) a number. When a
    span's UNIT_* token's surface is neither a lexicon form of that class
    (after one declared case ending, R15) nor within edit distance 1 of one,
    and the span has no independently justified CARD_*/PFX_* coefficient to
    corroborate it, the unit is a guess and the span is dropped.

    The edit-distance-1 tolerance is what keeps the synthetic set's noised
    unit words ("hazzar", "lakhh", "करोड") working after digits."""
    meaningful = _meaningful(tokens)
    has_justified_coef = any(
        cls.startswith(("CARD_", "PFX_")) and is_known_form(cls, text, lex)
        for _i, cls, text in meaningful
    )
    if has_justified_coef:
        return False
    return any(
        cls.startswith("UNIT_") and not is_near_known_form(cls, text, 1, lex)
        for _i, cls, text in meaningful
    )
