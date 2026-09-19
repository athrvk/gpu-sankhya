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
    "is_lexicon_o_only",
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


def load_lexicon(path: str = None) -> dict:
    """Load (and cache) the exported lexicon JSON.

    Reads `src/data/lexicon.json` relative to the repo; if that file is
    missing, builds the same object in-process from the registered language
    packs via `export_lexicon.build()`.
    """
    global _LEXICON, _FORMS, _RANGE_WORDS, _BOUND
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
        _FORMS = None
        _RANGE_WORDS = None
        _BOUND = None
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


def _verify_token(cls: str, text: str, forms, range_words,
                  bound=None, next_text: str = None) -> bool:
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
    if cls in forms.get(_norm(text), ()):
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
    has_content = False
    for i, (cls, text) in enumerate(tokens):
        next_text = tokens[i + 1][1] if i + 1 < len(tokens) else None
        if not _verify_token(cls, text, forms, range_words, bound, next_text):
            return False
        if _is_content(cls):
            has_content = True
    return has_content
