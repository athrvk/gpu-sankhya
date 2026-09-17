"""Synthetic labelled data generator for gpu-sankhya.

Nothing here references a specific language beyond the pack it is given.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys

from . import classes as C
from . import core
from . import noise_latn as NL
from . import charset as CS
from .langs import base as langbase

MAX_LEN = 128

# Probability an example gets an out-of-vocab "unk noise" run inserted as
# O-labelled context (see _insert_unk_noise). Every character in the pool is
# guaranteed (by test) to be outside every registered pack's charset, so it
# always maps to the <unk> embedding at train/inference time - this mimics
# real production input (emoji, CJK, foreign symbols) the synthetic data
# otherwise never contains, and trains the <unk> embedding to be inert.
P_UNK = 0.12

_UNK_POOL = [
    # emoji
    "🙏", "😀", "🔥", "👍", "❤", "🎉", "💸", "✅",
    # arrows
    "→", "←",
    # box/dingbats
    "★", "✓", "•",
    # CJK
    "中", "文",
    # Cyrillic
    "д", "ж",
    # Greek
    "λ",
    # other currency/symbols
    "€", "£", "$", "¥", "§", "¶", "°",
    # fullwidth punctuation
    "，", "。", "！",
]

_CHAT_FRAGMENTS = [
    "arey", "bhai suno", "lol", "haan", "ok done", "kal milte hain", "waise",
    "hmm", "acha theek hai", "btw", "chal", "sahi hai", "kya baat hai",
]

_ENGLISH_WORD_SET = None


def _english_word_set(pack):
    global _ENGLISH_WORD_SET
    s = set()
    s.update({"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
              "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
              "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
              "eighty", "ninety", "hundred", "thousand", "million", "billion", "mil", "mn", "bn"})
    for forms in pack.symbol_units.values():
        s.update(f.lower() for f in forms)
    return s


def _noise_word(pack, rng, word):
    eng = _english_word_set(pack)
    if word.lower() in eng:
        return word
    if word.isdigit():
        return word
    # N1 variant pick handled by caller (choice among lexicon list); here just char noise
    return pack.noise(word, rng)


def _pick_form(pack, rng, cls, symbol=False):
    """N1: pick a curated variant (or symbol form)."""
    if symbol and cls in pack.symbol_units:
        return rng.choice(pack.symbol_units[cls]), True
    forms = pack.lexicon.get(cls, [None])
    return rng.choice(forms), False


def _all_forms_lower(pack):
    return pack.all_forms()


def _safe_word(pack, rng, cls, symbol=False, max_tries=6):
    """Pick + noise a word form for cls, rejecting collisions with other classes
    and with the pack's blocked_surfaces (proper-noun collisions etc)."""
    forms_map = _all_forms_lower(pack)
    blocked = set(w.lower() for w in getattr(pack, "blocked_surfaces", []))
    for _ in range(max_tries):
        base_form, is_symbol = _pick_form(pack, rng, cls, symbol=symbol)
        if base_form is None:
            return None, is_symbol
        if is_symbol:
            out = base_form  # symbols are not noised
        else:
            out = _noise_word(pack, rng, base_form)
        if out.lower() in blocked:
            continue
        other = forms_map.get(out.lower())
        if other is None or other == cls:
            return out, is_symbol
    return base_form, is_symbol


def _digits_str(rng, value, indian_grouping=False, decimal=False):
    if decimal:
        text = f"{value:g}"
        if "." not in text:
            text += ".0"
    else:
        text = str(int(value))
    if indian_grouping and "." not in text and len(text) > 3:
        text = _indian_group(text)
    return text


def _indian_group(digits: str) -> str:
    if len(digits) <= 3:
        return digits
    head, tail = digits[:-3], digits[-3:]
    parts = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return ",".join(parts) + "," + tail


def _num_tokens_from_digits(digit_str):
    """Turn a digit string like '1,25,000' or '2.5' into DIGITS/COMMA/DOT tokens."""
    toks = []
    for ch in digit_str:
        if ch == ",":
            toks.append(("COMMA", ch))
        elif ch == ".":
            toks.append(("DOT", ch))
        else:
            toks.append(("DIGITS", ch))
    return toks


def _sep():
    return ("SEP", " ")


def _num_tokens_from_digits_p(pack, rng, digit_str):
    """Like _num_tokens_from_digits, but for packs that declare a
    `deva_digit_prob` (hi_deva), renders the digit glyphs in Devanagari
    with that probability. The token CLASS stays DIGITS/DOT/COMMA either
    way -- only the surface text changes -- so label computation is
    unaffected (it happens after normalize_text() maps them back to ASCII)."""
    p = getattr(pack, "deva_digit_prob", 0.0)
    if p and rng.random() < p:
        from . import noise_deva as ND
        digit_str = ND.to_deva_digits(digit_str)
    return _num_tokens_from_digits(digit_str)


def build_term_tokens(pack, rng, structure, unit_cls=None):
    """Build tokens (list of (cls,text)) for a single Amount, per structure."""
    toks = []

    def unit_tok():
        u = unit_cls or rng.choices(
            ["UNIT_LAKH", "UNIT_HAZAAR", "UNIT_CRORE", "UNIT_SAU", "UNIT_MILLION", "UNIT_BILLION", "UNIT_ARAB", "UNIT_KHARAB"],
            weights=[35, 25, 20, 10, 3, 3, 2, 2],
        )[0]
        use_symbol = structure == "digits_symbol" and u in pack.symbol_units
        word, is_symbol = _safe_word(pack, rng, u, symbol=use_symbol)
        return u, word, is_symbol

    if structure == "prefix_unit":
        pfx = rng.choice(list(C.PREFIX_INFO.keys()))
        u, uword, _ = unit_tok()
        pword, _ = _safe_word(pack, rng, pfx)
        toks += [(pfx, pword), _sep(), (u, uword)]

    elif structure == "prefix_num_unit":
        pfx = rng.choice(["PFX_SAVA", "PFX_PAUNE", "PFX_SAADHE"])
        minn = C.PREFIX_INFO[pfx].get("min_n", 2)
        n = rng.randint(minn, 20)
        u, uword, _ = unit_tok()
        pword, _ = _safe_word(pack, rng, pfx)
        card = f"CARD_{n}"
        if rng.random() < 0.5 and card in pack.lexicon:
            nword, _ = _safe_word(pack, rng, card)
            ntok = (card, nword)
        else:
            ds = str(n)
            ntok = None
        # Hindi prefixes always precede their number ("saadhe teen", "sava do");
        # only the English fraction glue words come after the number.
        toks.append((pfx, pword))
        toks.append(_sep())
        if ntok:
            toks.append(ntok)
        else:
            toks += _num_tokens_from_digits_p(pack, rng, str(n))
        toks.append(_sep())
        toks.append((u, uword))

    elif structure == "card_unit":
        n = rng.randint(1, 99)
        u, uword, _ = unit_tok()
        card = f"CARD_{n}"
        nword, _ = _safe_word(pack, rng, card)
        toks += [(card, nword), _sep(), (u, uword)]

    elif structure == "digits_word_unit":
        u, uword, _ = unit_tok()
        if u in ("UNIT_LAKH", "UNIT_CRORE", "UNIT_MILLION", "UNIT_HAZAAR") and rng.random() < 0.4:
            val = round(rng.uniform(1, 20), 2)
            ds = _digits_str(rng, val, decimal=True)
        else:
            val = rng.randint(1, 99)
            ds = _digits_str(rng, val)
        toks += _num_tokens_from_digits_p(pack, rng, ds) + [_sep(), (u, uword)]

    elif structure == "digits_symbol":
        u, uword, is_symbol = unit_tok()
        # bias heavily toward very short forms ("2.5L", "5L", "20k", "12LPA")
        # since these are the most common real-world spellings.
        if rng.random() < 0.6:
            if rng.random() < 0.5:
                val = round(rng.uniform(1, 9.9), 1)  # one decimal place, 1-2 digit
                ds = _digits_str(rng, val, decimal=True)
            else:
                val = rng.randint(1, 99)
                ds = _digits_str(rng, val)
        else:
            val = round(rng.uniform(1, 50), 2) if rng.random() < 0.5 else rng.randint(1, 500)
            ds = _digits_str(rng, val, decimal=isinstance(val, float) and val != int(val))
        space = " " if rng.random() < 0.5 else ""
        toks += _num_tokens_from_digits_p(pack, rng, ds)
        if space:
            toks.append(_sep())
        toks.append((u, uword))

    elif structure == "digits_currency":
        val = rng.choice([500, 1000, 2000, 50000, 125000, 25000, 999, 15000])
        indian = rng.random() < 0.6
        ds = _digits_str(rng, val, indian_grouping=indian)
        toks += _num_tokens_from_digits_p(pack, rng, ds)

    elif structure == "chain":
        n_terms = rng.choice([2, 3])
        units = rng.sample(["UNIT_CRORE", "UNIT_LAKH", "UNIT_HAZAAR", "UNIT_SAU"], k=n_terms)
        units.sort(key=lambda u: -C.unit_value(u))
        for i, u in enumerate(units):
            if i > 0:
                toks.append(_sep())
            n = rng.randint(1, 99)
            card = f"CARD_{n}"
            nword, _ = _safe_word(pack, rng, card)
            uword, _ = _safe_word(pack, rng, u)
            toks += [(card, nword), _sep(), (u, uword)]

    elif structure == "english_fraction":
        u = rng.choice(["UNIT_LAKH", "UNIT_CRORE", "UNIT_HAZAAR"])
        uword, _ = _safe_word(pack, rng, u)
        choice = rng.choice(["half_unit", "one_unit", "num_and_half"])
        if choice == "half_unit":
            toks += [("PFX_AADHA", "half a"), _sep(), (u, uword)]
        elif choice == "one_unit":
            toks += [("CARD_1", "one"), _sep(), (u, uword)]
        else:
            n = rng.randint(2, 9)
            phrase = rng.choice(pack.english_fraction_phrases.get("PFX_SAADHE", ["and a half"]))
            toks += [(f"CARD_{n}", str(n)), _sep(), ("PFX_SAADHE", phrase), _sep(), (u, uword)]

    elif structure == "mult_chain":
        # ascending-unit multiplicative stack: "das hazaar crore", "sau crore",
        # "2 lakh crore" - a small unit's amount, then ONE larger unit on top.
        small_unit = rng.choices(["UNIT_SAU", "UNIT_HAZAAR", "UNIT_LAKH"], weights=[40, 35, 25])[0]
        larger_options = [u for u in ("UNIT_LAKH", "UNIT_CRORE") if C.unit_value(u) > C.unit_value(small_unit)]
        larger_unit = rng.choice(larger_options)
        first_structure = rng.choice(["prefix_unit", "prefix_num_unit", "card_unit", "digits_word_unit"])
        toks = build_term_tokens(pack, rng, first_structure, unit_cls=small_unit)
        uword, _ = _safe_word(pack, rng, larger_unit)
        toks += [_sep(), (larger_unit, uword)]

    elif structure == "bare_card_currency":
        n = rng.randint(1, 99)
        card = f"CARD_{n}"
        nword, _ = _safe_word(pack, rng, card)
        toks.append((card, nword))

    elif structure == "bare_unit_currency":
        # implicit-1 unit words: "hazar rupaye", "lakh rupaye", "sau rupaye"
        u = rng.choices(["UNIT_HAZAAR", "UNIT_LAKH", "UNIT_CRORE", "UNIT_SAU"], weights=[35, 30, 15, 20])[0]
        uword, _ = _safe_word(pack, rng, u)
        toks.append((u, uword))

    elif structure == "chain_trailing_bare":
        # (b) trailing/leading small-cardinal additive chains where the LAST
        # term has no unit of its own: "ek hazaar ek" (1001), "do lakh
        # paanch" (200005). Core already evaluates a bare trailing NUM as an
        # implicit +N (unit=1); this just adds the generator surface for it.
        n_units = rng.choice([1, 2])
        units = rng.sample(["UNIT_CRORE", "UNIT_LAKH", "UNIT_HAZAAR"], k=n_units)
        units.sort(key=lambda u: -C.unit_value(u))
        for i, u in enumerate(units):
            if i > 0:
                toks.append(_sep())
            n = rng.randint(1, 99)
            card = f"CARD_{n}"
            nword, _ = _safe_word(pack, rng, card)
            uword, _ = _safe_word(pack, rng, u)
            toks += [(card, nword), _sep(), (u, uword)]
        toks.append(_sep())
        tail_n = rng.randint(1, 9)
        tail_card = f"CARD_{tail_n}"
        tail_word, _ = _safe_word(pack, rng, tail_card)
        toks.append((tail_card, tail_word))

    elif structure == "unit_chain":
        # "unit + cardinal + unit" descending additive chain, e.g. "hazaar do sau" = 1200
        big, small = "UNIT_HAZAAR", "UNIT_SAU"
        bword, _ = _safe_word(pack, rng, big)
        n = rng.randint(1, 9)
        card = f"CARD_{n}"
        nword, _ = _safe_word(pack, rng, card)
        sword, _ = _safe_word(pack, rng, small)
        toks += [(big, bword), _sep(), (card, nword), _sep(), (small, sword)]

    else:
        n = rng.randint(1, 99)
        card = f"CARD_{n}"
        nword, _ = _safe_word(pack, rng, card)
        u, uword, _ = unit_tok()
        toks += [(card, nword), _sep(), (u, uword)]

    return toks


_STRUCTURE_WEIGHTS = [
    ("prefix_unit", 20), ("prefix_num_unit", 12), ("card_unit", 15),
    ("digits_word_unit", 15), ("digits_symbol", 15), ("digits_currency", 8),
    ("chain", 7), ("english_fraction", 4), ("bare_card_currency", 4),
    ("mult_chain", 5), ("bare_unit_currency", 3), ("unit_chain", 2),
    ("chain_trailing_bare", 6),
]


def _maybe_possessive(rng, toks, p=0.02):
    """(d) possessive/plural noise on units: "lakh's", "lakhs'", "crore's",
    "2lakh's" - the apostrophe+letters are O, the unit itself stays a unit
    (the value is unaffected). Only fires when the phrase actually ends on a
    unit token, and appends an O-labelled suffix glued to it."""
    if not toks or rng.random() >= p:
        return toks
    last_cls, _ = toks[-1]
    if not C.is_unit(last_cls):
        return toks
    suffix = rng.choice(["'s", "'"])
    return toks + [("O", suffix)]


def _maybe_join_words(pack, rng, toks, p=None):
    """No-space word-form noise: "sawalakh" (PFX_SAVA+UNIT_LAKH), "paanchlakh"
    (CARD_5+UNIT_LAKH), "dashazaar" (CARD_10+CARD_1000 in an additive chain).
    `p` defaults to env SANKHYA_JOIN_WORDS_P (0.04); set it to 0 for ablations.

    Removes a single SEP (" ") token that sits between two adjacent WORD
    tokens (PFX_*/CARD_*, or CARD_*/UNIT_*), gluing the two surface forms
    with no space. Each glued token keeps its own class id, so BIO/cls
    labels stay correct - this only deletes one O/I-ranged SEP character,
    never touches a DIGITS/DOT/COMMA token (so "2lakh" style glue, which
    already exists elsewhere, is untouched), and joins at most one pair per
    phrase.
    """
    if p is None:
        p = float(os.environ.get("SANKHYA_JOIN_WORDS_P", "0.04"))
    if not toks or rng.random() >= p:
        return toks
    candidates = []
    for i in range(1, len(toks) - 1):
        sep_cls, sep_text = toks[i]
        if sep_cls != "SEP" or sep_text != " ":
            continue
        left_cls, _ = toks[i - 1]
        right_cls, _ = toks[i + 1]
        pair_ok = (
            (C.is_prefix(left_cls) and C.is_unit(right_cls)) or
            (C.is_prefix(left_cls) and C.is_card(right_cls)) or
            (C.is_card(left_cls) and C.is_unit(right_cls)) or
            (C.is_card(left_cls) and C.is_card(right_cls))
        )
        if pair_ok:
            candidates.append(i)
    if not candidates:
        return toks
    i = rng.choice(candidates)
    return toks[:i] + toks[i + 1:]


def build_phrase(pack, rng, structure=None, exclude_currency_bare=False):
    if structure is None:
        pool = _STRUCTURE_WEIGHTS
        if exclude_currency_bare:
            pool = [(s, w) for s, w in pool if s not in ("digits_currency", "bare_card_currency")]
        if not pack.english_fraction_phrases:
            pool = [(s, w) for s, w in pool if s != "english_fraction"]
        names = [s for s, _ in pool]
        weights = [w for _, w in pool]
        structure = rng.choices(names, weights=weights)[0]
    toks = build_term_tokens(pack, rng, structure)
    toks = _maybe_possessive(rng, toks)
    toks = _maybe_join_words(pack, rng, toks)
    return toks, structure


_RANGE_CARD_PAIRS = [(2, 3), (5, 10), (10, 15), (20, 25), (50, 60), (3, 5), (15, 20), (25, 30)]
_RANGE_UNITS = ["UNIT_LAKH", "UNIT_HAZAAR", "UNIT_CRORE", "UNIT_SAU", "UNIT_MILLION"]
_RANGE_PREFIXES = ["PFX_DEDH", "PFX_DHAI"]


def _pick_range_unit(pack, rng, symbol=False):
    u = rng.choices(_RANGE_UNITS, weights=[35, 25, 20, 10, 5])[0]
    use_symbol = symbol and u in pack.symbol_units
    word, is_symbol = _safe_word(pack, rng, u, symbol=use_symbol)
    return u, word, is_symbol


def _range_connector(pack, rng, allow_juxtaposition=True):
    if allow_juxtaposition and rng.random() < 0.4:
        return ("RANGE", " ")
    return ("RANGE", rng.choice(pack.range_connectors))


def _maybe_repeat_unit(pack, rng, left, u, uword, repeat_prob=0.35):
    """Optionally repeat the shared unit on the left side too."""
    if rng.random() < repeat_prob:
        return left + [_sep(), (u, uword)]
    return left


def build_range_phrase(pack, rng):
    """Build a single-span Amount RANGE Amount token list.

    Both amounts always share the SAME unit (either the unit trails once, or
    it is repeated on both sides), the two sides are the same form family
    (both digits, both cardinal words, or a prefix-standalone + cardinal
    juxtaposition), the surface order is strictly ascending, and
    high <= 3 * low.
    """
    mode = rng.choices(["digits", "cardword", "prefix_juxt"], weights=[40, 40, 20])[0]

    if mode == "digits":
        use_symbol = rng.random() < 0.4
        u, uword, is_symbol = _pick_range_unit(pack, rng, symbol=use_symbol)
        # "45 sou / 121 sou" (2.7x) reads oddly for sau specifically; keep it tighter.
        max_ratio = 2.5 if u == "UNIT_SAU" else 3.0
        decimal = rng.random() < 0.3
        if decimal:
            low = round(rng.uniform(1.0, 20.0), 1)
            ratio = rng.uniform(1.2, max_ratio)
            high = round(low * ratio, 1)
            if high <= low:
                high = round(low + 0.5, 1)
            high = min(high, low * max_ratio)
            low_s = f"{low:g}"
            high_s = f"{high:g}"
            if "." not in low_s:
                low_s += ".0"
            if "." not in high_s:
                high_s += ".0"
        else:
            low = rng.randint(1, 90)
            ratio = rng.uniform(1.1, max_ratio)
            high = min(int(round(low * ratio)), int(low * max_ratio))
            if high <= low:
                high = low + 1
            low_s, high_s = str(low), str(high)
        assert float(high_s) <= 3 * float(low_s) + 1e-9
        left = _num_tokens_from_digits_p(pack, rng, low_s)
        left = _maybe_repeat_unit(pack, rng, left, u, uword)
        # digit pairs must always use an explicit connector - bare-space
        # juxtaposition ("58 157 hazzar", "1 2 LAKH") only reads naturally
        # between two WORD forms ("do teen lakh", "dedh do lakh").
        connector = _range_connector(pack, rng, allow_juxtaposition=False)
        right = _num_tokens_from_digits_p(pack, rng, high_s)
        space = _sep() if (rng.random() < 0.5 or not is_symbol) else None
        toks = left + [connector] + right + ([space] if space else []) + [(u, uword)]
        return toks

    if mode == "cardword":
        n1, n2 = rng.choice(_RANGE_CARD_PAIRS)
        if n2 < n1:
            n1, n2 = n2, n1
        u, uword, _ = _pick_range_unit(pack, rng)
        card1, card2 = f"CARD_{n1}", f"CARD_{n2}"
        w1, _ = _safe_word(pack, rng, card1)
        w2, _ = _safe_word(pack, rng, card2)
        left = [(card1, w1)]
        left = _maybe_repeat_unit(pack, rng, left, u, uword)
        connector = _range_connector(pack, rng)
        toks = left + [connector, (card2, w2), _sep(), (u, uword)]
        return toks

    # prefix_juxt: standalone prefix (dedh=1.5 / dhai=2.5) juxtaposed with a
    # nearby cardinal, e.g. "dedh do lakh" (1.5-2 lakh), "do dhai lakh" (2-2.5 lakh)
    pfx = rng.choice(_RANGE_PREFIXES)
    v = C.PREFIX_INFO[pfx]["standalone"]
    nmin = max(1, int((v / 3) + 0.999))
    nmax = max(nmin, int(v * 3))
    n = rng.randint(nmin, nmax)
    u, uword, _ = _pick_range_unit(pack, rng)
    pword, _ = _safe_word(pack, rng, pfx)
    card = f"CARD_{n}"
    nword, _ = _safe_word(pack, rng, card)

    pfx_tok = (pfx, pword)
    card_tok = (card, nword)
    if v <= n:
        left_tok, right_tok = pfx_tok, card_tok
    else:
        left_tok, right_tok = card_tok, pfx_tok
    connector = _range_connector(pack, rng)
    toks = [left_tok, connector, right_tok, _sep(), (u, uword)]
    return toks


def build_conjunction_phrase(pack, rng, ppack):
    """(a) TWO (sometimes three) independent quantity spans in one text
    joined by a conjunction, e.g. "sava lakh aur dedh lakh", "50k ya 60k".
    Each side is its own span (own B tag); the connector is O. NOT a range
    (that is a single span via build_range_phrase/RANGE class)."""
    n_spans = 3 if rng.random() < 0.15 else 2
    parts = [build_phrase(ppack, rng, exclude_currency_bare=True)[0] for _ in range(n_spans)]
    connectors = pack.conj_connectors or [" aur "]
    out = ""
    span_defs = []
    for i, toks in enumerate(parts):
        if i > 0:
            out += rng.choice(connectors)
        start = len(out)
        seg_text, _ = _tokens_to_text_and_labels(toks)
        out += seg_text
        end = len(out)
        span_defs.append((start, end, toks))
    return out, span_defs


def _resolve_currency_marker(pack, rng, force=False, next_is_digit=False):
    """A currency marker glued (no space) to what follows is only natural
    before DIGITS ("Rs500", "₹2.5L"); before a word form it always needs a
    space ("Rs bees", never "Rsbees"). "Re" only ever precedes digits."""
    if force or rng.random() < 0.35:
        markers = pack.currency_markers_before
        if not next_is_digit:
            markers = [m for m in markers if m != "Re"]
            if not markers:
                markers = pack.currency_markers_before
        marker = rng.choice(markers)
        if next_is_digit:
            sep = "" if marker in ("₹",) or rng.random() < 0.4 else " "
        else:
            sep = " "
        return marker + sep
    return ""


def _resolve_approximator(pack, rng):
    if rng.random() < 0.3:
        return rng.choice(pack.approximators) + " "
    return ""


def _tokens_to_text_and_labels(tokens):
    """tokens: list of (cls, text). Returns (text, cls_list_per_char)."""
    text = ""
    cls_per_char = []
    for cls, t in tokens:
        text += t
        cls_per_char += [cls] * len(t)
    return text, cls_per_char


def sample_example(pack, rng, max_len=MAX_LEN, phrase_pack=None, unk_p=P_UNK):
    for _attempt in range(50):
        try:
            ex = _sample_example_once(pack, rng, phrase_pack=phrase_pack, unk_p=unk_p)
        except AssertionError:
            raise
        except Exception:
            continue
        if ex is not None and len(ex["text"]) <= max_len:
            return ex
    # fallback: bare phrase, should always fit
    toks, _ = build_phrase(pack, rng, "card_unit")
    text, clsc = _tokens_to_text_and_labels(toks)
    return _finalize(pack, rng, text, [(0, len(text), toks)], is_negative=False)


def _assert_prefix_before_number(pack, span_tokens):
    """Any PFX_* token that is not an english_fraction_phrases surface form
    must appear before the NUMBER of its term (Hindi prefixes precede their
    number); only English glue words like "and a half" come after it."""
    packs = pack if isinstance(pack, (list, tuple)) else [pack]
    eng_forms = set()
    for p in packs:
        for forms in p.english_fraction_phrases.values():
            eng_forms.update(f.lower() for f in forms)

    num_seen = False
    for cls, text in span_tokens:
        if C.is_unit(cls) or cls == "RANGE":
            num_seen = False
            continue
        if cls == "DIGITS" or C.is_card(cls):
            num_seen = True
            continue
        if C.is_prefix(cls) and text.lower() not in eng_forms and num_seen:
            raise AssertionError(f"Hindi prefix after number: {span_tokens}")


def _finalize(pack, rng, text, span_defs, is_negative, currency_pack=None):
    _currency_pack = currency_pack if currency_pack is not None else pack
    # Labels are computed on the NORMALIZED text (NFC + Devanagari-digit ->
    # ASCII + lowercase); if NFC normalization changes the character length
    # the char-level offsets would desync, so bail out and let the caller
    # resample this example instead.
    normalized = CS.normalize_text(text)
    if len(normalized) != len(text):
        raise ValueError("NFC normalization changed text length; resample")

    bio = [0] * len(text)
    clsids = [C.CLASS_TO_ID["O"]] * len(text)
    spans = []
    for start, end, toks in span_defs:
        span_tokens = toks
        # core.evaluate() (and the prefix-order check) must see the same
        # NORMALIZED text the model will be trained/decoded on (NFC +
        # Devanagari-digit -> ASCII + lowercase) - e.g. a "७२" DIGITS token
        # must reach core as ASCII "72", never the raw Devanagari glyphs.
        norm_tokens = [(cls, CS.normalize_text(t)) for cls, t in span_tokens]
        _assert_prefix_before_number(_currency_pack, norm_tokens)
        result = core.evaluate(norm_tokens)
        currency = core.detect_currency(text, start, end, _currency_pack)
        classes_str = " ".join(c for c, _ in span_tokens if c != "SEP")
        spans.append({
            "start": start, "end": end,
            "value": result.value,
            "range": list(result.range) if result.range else None,
            "unit": result.unit,
            "currency": currency,
            "classes": classes_str,
        })
        for i in range(start, end):
            bio[i] = 1 if i == start else 2
        # per-char class labels
        pos = start
        for cls, t in span_tokens:
            for _ in t:
                if pos < end:
                    clsids[pos] = C.CLASS_TO_ID.get(cls, C.CLASS_TO_ID["O"])
                pos += 1

        # round-trip check
        rebuilt = [(cls, "x") for cls, _ in span_tokens]  # text irrelevant, only cls matters for eval
        check = core.evaluate(norm_tokens)
        if check.value != result.value or check.unit != result.unit or check.range != result.range:
            raise AssertionError("round-trip mismatch")

    if not is_negative and not spans:
        raise AssertionError("positive example produced no spans")
    if spans and all(c == C.CLASS_TO_ID["O"] for c in clsids):
        raise AssertionError("span text is all-O")

    return {
        "text": text,
        "lang": pack.id,
        "spans": spans,
        "bio": bio,
        "cls": clsids,
    }


def _make_filler_word(pack, rng):
    """A filler word, occasionally noised (elongation / vowel drop) so the
    model also sees never-before-seen O-labelled strings."""
    if not getattr(pack, "filler_words", None):
        return "hmm"
    w = rng.choice(pack.filler_words)
    if rng.random() < 0.10:
        fn = rng.choice([NL.n5_final_vowel, NL.n11_elongation])
        try:
            w = fn(w, rng)
        except Exception:
            pass
    if rng.random() < 0.85:
        w = w.lower()
    elif rng.random() < 0.5:
        w = w.title()
    return w


def _insert_fillers(pack, rng, text, span_defs):
    """(2) an O-labelled filler word landing immediately adjacent to a span
    15% of the time, and (1) 1-3 filler words dropped at a random word
    boundary outside every span 40% of the time. Uses fresh text.find() per
    span (positions get re-resolved downstream by _recompute_offsets).

    Spans with no UNIT_* token (bare digits/cardinals) depend on a currency
    marker/word staying adjacent (see §2) - filler insertion must never land
    between such a span and its currency context, so those spans (and a
    small buffer around them) are excluded from BOTH filler steps.
    """
    eligible = []  # (seg_text,) for spans safe to touch with filler
    protected_spans = []  # seg_text for bare currency-dependent spans
    for _, _, toks in span_defs:
        seg_text, _ = _tokens_to_text_and_labels(toks)
        if not seg_text:
            continue
        has_unit = any(C.is_unit(c) for c, _ in toks)
        if has_unit:
            eligible.append(seg_text)
        else:
            protected_spans.append(seg_text)

    def _span_occupied_positions():
        """Character positions to avoid: every span itself, plus an 8-char
        buffer on both sides of every bare (currency-dependent) span."""
        occupied = set()
        for seg in eligible + protected_spans:
            idx = text.find(seg)
            if idx == -1:
                continue
            occupied.update(range(idx, idx + len(seg)))
        for seg in protected_spans:
            idx = text.find(seg)
            if idx == -1:
                continue
            occupied.update(range(max(0, idx - 8), idx))
            occupied.update(range(idx + len(seg), min(len(text), idx + len(seg) + 8)))
        return occupied

    if eligible and rng.random() < 0.15:
        seg = rng.choice(eligible)
        pos = text.find(seg)
        if pos != -1:
            filler = _make_filler_word(pack, rng)
            if rng.random() < 0.5:
                text = text[:pos] + filler + " " + text[pos:]
            else:
                end = pos + len(seg)
                text = text[:end] + " " + filler + text[end:]

    if rng.random() < 0.40:
        n = rng.randint(1, 3)
        chunk = " ".join(_make_filler_word(pack, rng) for _ in range(n))
        for _try in range(10):
            occupied = _span_occupied_positions()
            positions = [i for i, ch in enumerate(text) if ch == " " and i not in occupied]
            if not positions:
                break
            pos = rng.choice(positions)
            text = text[:pos] + " " + chunk + text[pos:]
            break
    return text


def _insert_unk_noise(pack, rng, text, span_defs, p_unk=P_UNK):
    """With probability p_unk, insert a 1-3 char run drawn from _UNK_POOL
    (chars guaranteed outside every pack's charset -> always <unk>) as
    O-labelled context. Never touches a span or the 8-char buffer around a
    bare/currency-dependent span (same protection as _insert_fillers).
    Placement: start of text, end of text, or a random space position
    outside protected regions; ~25% of the time glued to the adjacent word
    with no space. ~15% of runs repeat the same char (mimics JS surrogate
    pairs, e.g. "🙏🙏")."""
    if rng.random() >= p_unk:
        return text

    eligible = []
    protected_spans = []
    for _, _, toks in span_defs:
        seg_text, _ = _tokens_to_text_and_labels(toks)
        if not seg_text:
            continue
        has_unit = any(C.is_unit(c) for c, _ in toks)
        if has_unit:
            eligible.append(seg_text)
        else:
            protected_spans.append(seg_text)

    def _occupied():
        occupied = set()
        for seg in eligible + protected_spans:
            idx = text.find(seg)
            if idx == -1:
                continue
            occupied.update(range(idx, idx + len(seg)))
        for seg in protected_spans:
            idx = text.find(seg)
            if idx == -1:
                continue
            occupied.update(range(max(0, idx - 8), idx))
            occupied.update(range(idx + len(seg), min(len(text), idx + len(seg) + 8)))
        return occupied

    n = rng.randint(1, 3)
    if rng.random() < 0.15:
        ch = rng.choice(_UNK_POOL)
        run = ch * n
    else:
        run = "".join(rng.choice(_UNK_POOL) for _ in range(n))

    glued = rng.random() < 0.25
    placement = rng.choice(["start", "end", "middle"])

    if placement == "start":
        sep = "" if glued else " "
        return run + sep + text
    if placement == "end":
        sep = "" if glued else " "
        return text + sep + run

    occupied = _occupied()
    positions = [i for i, ch in enumerate(text) if ch == " " and i not in occupied]
    if not positions:
        sep = "" if glued else " "
        return text + sep + run
    pos = rng.choice(positions)
    if glued:
        # glue to the word that starts right after this space
        return text[:pos + 1] + run + text[pos + 1:]
    return text[:pos] + " " + run + text[pos:]


def _apply_casing_and_wrap(pack, rng, text):
    if rng.random() < 0.10:
        frag = rng.choice(_CHAT_FRAGMENTS)
        if rng.random() < 0.5:
            text = frag + " " + text
        else:
            text = text + " " + frag
    return text


_SHORT_STRUCTURES = ["digits_symbol", "prefix_unit", "digits_word_unit", "mult_chain"]

_DURATION_SUFFIXES = ["baad", "mein", "se", "ko", ""]


def _build_negative_text(pack, rng):
    """Pick one of: a CARD/DIGITS + duration/time/count-noun negative
    (~35% of the "negatives" family, i.e. ~4% of all examples), a filler
    word-soup negative, or a fixed template negative."""
    r = rng.random()
    if getattr(pack, "duration_nouns", None) and r < 0.35:
        n = rng.randint(1, 99)
        if rng.random() < 0.4:
            numtext = str(n)
        else:
            card = f"CARD_{n}"
            numtext, _ = _safe_word(pack, rng, card)
        noun = rng.choice(pack.duration_nouns)
        text = f"{numtext} {noun}"
        suffix = rng.choice(_DURATION_SUFFIXES)
        if suffix:
            text += " " + suffix
        return text
    if r < 0.35 + 0.15:
        n = rng.randint(3, 10)
        return " ".join(_make_filler_word(pack, rng) for _ in range(n))
    return rng.choice(pack.templates["negatives"])


def _sample_example_once(pack, rng, phrase_pack=None, unk_p=P_UNK):
    """phrase_pack: when set (cross-script mode), the {P}/{P1}/{P2} phrase(s)
    are built from phrase_pack's lexicon while the template, currency
    markers, approximators and filler words still come from `pack`; currency
    detection scans the union of both packs' marker lists."""
    ppack = phrase_pack if phrase_pack is not None else pack
    currency_pack = pack if phrase_pack is None else [pack, phrase_pack]
    family_weights = [
        ("casual", 16), ("classifieds", 16), ("news", 13), ("salary", 9),
        ("ranges", 8), ("two_spans", 10), ("bare", 12), ("short_context", 8),
        ("negatives", 12), ("multi_conj", 7),
    ]
    names = [f for f, _ in family_weights]
    weights = [w for _, w in family_weights]
    family = rng.choices(names, weights=weights)[0]

    if family == "negatives":
        text = _build_negative_text(pack, rng)
        text = _apply_casing_and_wrap(pack, rng, text)
        return _finalize(pack, rng, text, [], is_negative=True, currency_pack=currency_pack)

    if family == "multi_conj":
        out, span_defs = build_conjunction_phrase(pack, rng, ppack)
        out = _insert_fillers(pack, rng, out, span_defs)
        out = _insert_unk_noise(pack, rng, out, span_defs, p_unk=unk_p)
        out = _apply_casing_and_wrap(pack, rng, out)
        return _finalize(pack, rng, out, _recompute_offsets(out, span_defs), is_negative=False, currency_pack=currency_pack)

    # "bare": at least half are TRULY bare - the phrase IS the entire text,
    # nothing else: no chat-fragment wrap, no filler, no punctuation. This is
    # the short-input regime ("2.5L", "20k", "sava lakh", "das hazaar crore").
    truly_bare = family == "bare" and rng.random() < 0.6
    if truly_bare:
        template = "{P}"
    else:
        template = rng.choice(pack.templates[family])

    force_currency = family in ("classifieds", "salary") and "{C}" in template and rng.random() < 0.5
    parts = re.split(r"(\{[A-Z0-9]+\})", template)
    out = ""
    span_defs = []
    phrase_cache = {}

    if family == "ranges":
        phrase_cache["P"] = build_range_phrase(ppack, rng)
    elif family == "two_spans":
        phrase_cache["P1"], _ = build_phrase(ppack, rng, exclude_currency_bare=True)
        phrase_cache["P2"], _ = build_phrase(ppack, rng, exclude_currency_bare=True)
    elif family in ("bare", "short_context"):
        # bias toward short, single-token-ish forms typical of short inputs
        struct = rng.choice(_SHORT_STRUCTURES) if rng.random() < 0.6 else None
        phrase_cache["P"], structure = build_phrase(ppack, rng, structure=struct)
        if "{C}" in template and structure in ("digits_currency", "bare_card_currency"):
            force_currency = True
    else:
        phrase_cache["P"], structure = build_phrase(ppack, rng)
        if "{C}" in template and structure in ("digits_currency", "bare_card_currency"):
            force_currency = True

    for pi, part in enumerate(parts):
        m = re.match(r"^\{([A-Z0-9]+)\}$", part)
        if not m:
            out += part
            continue
        key = m.group(1)
        if key == "C":
            next_is_digit = False
            for nxt in parts[pi + 1:]:
                nm = re.match(r"^\{(P1|P2|P)\}$", nxt)
                if nm and nm.group(1) in phrase_cache:
                    ntoks = phrase_cache[nm.group(1)]
                    next_is_digit = bool(ntoks) and ntoks[0][0] == "DIGITS"
                    break
                if nxt:
                    next_is_digit = bool(nxt) and nxt[0].isdigit()
                    break
            out += _resolve_currency_marker(pack, rng, force=force_currency, next_is_digit=next_is_digit)
        elif key == "A":
            out += _resolve_approximator(pack, rng)
        elif key in ("P", "P1", "P2"):
            toks = phrase_cache[key]
            start = len(out)
            seg_text, _ = _tokens_to_text_and_labels(toks)
            out += seg_text
            end = len(out)
            span_defs.append((start, end, toks))
        else:
            out += part

    # a bare cardinal (no unit) is only in-scope with adjacent currency (§2);
    # if the template gave it none, force one so it is a valid positive.
    if span_defs and family != "two_spans":
        struct_used = locals().get("structure")
        if struct_used in ("bare_card_currency", "digits_currency"):
            s, e, _ = span_defs[-1]
            if core.detect_currency(out, s, e, currency_pack) is None:
                out = out[:e] + " rupaye" + out[e:]

    if not truly_bare:
        out = _insert_fillers(pack, rng, out, span_defs)
        out = _insert_unk_noise(pack, rng, out, span_defs, p_unk=unk_p)
        out = _apply_casing_and_wrap(pack, rng, out)
    return _finalize(pack, rng, out, _recompute_offsets(out, span_defs), is_negative=False, currency_pack=currency_pack)


def _recompute_offsets(out, span_defs):
    """Re-find span text positions in `out` (handles prepended chat fragments)."""
    result = []
    search_from = 0
    for start, end, toks in span_defs:
        seg_text, _ = _tokens_to_text_and_labels(toks)
        idx = out.find(seg_text, search_from)
        if idx == -1:
            idx = out.find(seg_text)
        if idx == -1:
            idx = start
        result.append((idx, idx + len(seg_text), toks))
        search_from = idx + len(seg_text)
    return result


def generate(pack, n, seed=0, max_len=MAX_LEN, unk_p=P_UNK):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        out.append(sample_example(pack, rng, max_len=max_len, unk_p=unk_p))
    return out


def generate_multi(packs, weights=None, n=1000, seed=0, cross=0.0, max_len=MAX_LEN, unk_p=P_UNK):
    """Multi-pack generation: each example picks one pack by `weights` (default
    uniform); a `cross` share of examples instead take a template from one
    pack and build the {P}/{P1}/{P2} phrase(s) from a DIFFERENT pack (currency
    markers/approximators still come from the template's own pack). With a
    single pack this reduces to plain `generate()` behaviour."""
    if len(packs) == 1:
        return generate(packs[0], n, seed=seed, max_len=max_len, unk_p=unk_p)
    if weights is None:
        weights = [1.0] * len(packs)
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        pack = rng.choices(packs, weights=weights)[0]
        if rng.random() < cross:
            others = [p for p in packs if p is not pack]
            phrase_pack = rng.choice(others) if others else None
            ex = sample_example(pack, rng, max_len=max_len, phrase_pack=phrase_pack, unk_p=unk_p)
            ex["cross_script"] = True
            if phrase_pack is not None:
                ex["phrase_lang"] = phrase_pack.id
        else:
            ex = sample_example(pack, rng, max_len=max_len, unk_p=unk_p)
            ex["cross_script"] = False
        out.append(ex)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=str, default="data/train.jsonl")
    ap.add_argument("--lang", type=str, default="hi_latn", help="comma-separated pack ids, e.g. hi_latn,hi_deva")
    ap.add_argument("--mix", type=str, default=None, help="comma-separated weights matching --lang, e.g. 0.55,0.45")
    ap.add_argument("--cross", type=float, default=0.0, help="share of examples that mix template/phrase across packs")
    ap.add_argument("--unk-noise", type=float, default=P_UNK, help="probability of inserting an out-of-vocab 'unk noise' run per example (0 disables)")
    args = ap.parse_args(argv)

    lang_ids = [x.strip() for x in args.lang.split(",") if x.strip()]
    packs = [langbase.get_pack(l) for l in lang_ids]

    if len(packs) == 1:
        rng = random.Random(args.seed)
        with open(args.out, "w", encoding="utf-8") as f:
            for _ in range(args.n):
                ex = sample_example(packs[0], rng, unk_p=args.unk_noise)
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"wrote {args.n} examples to {args.out}", file=sys.stderr)
        return

    if args.mix:
        weights = [float(x) for x in args.mix.split(",")]
        assert len(weights) == len(packs), "--mix must match --lang count"
    else:
        weights = None

    examples = generate_multi(packs, weights=weights, n=args.n, seed=args.seed, cross=args.cross, unk_p=args.unk_noise)
    with open(args.out, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"wrote {args.n} examples to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
