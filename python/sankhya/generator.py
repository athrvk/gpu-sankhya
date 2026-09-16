"""Synthetic labelled data generator for gpu-sankhya.

Nothing here references a specific language beyond the pack it is given.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys

from . import classes as C
from . import core
from . import noise_latn as NL
from .langs import base as langbase

MAX_LEN = 128

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
            toks += _num_tokens_from_digits(str(n))
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
        toks += _num_tokens_from_digits(ds) + [_sep(), (u, uword)]

    elif structure == "digits_symbol":
        u, uword, is_symbol = unit_tok()
        val = round(rng.uniform(1, 50), 2) if rng.random() < 0.5 else rng.randint(1, 500)
        ds = _digits_str(rng, val, decimal=isinstance(val, float) and val != int(val))
        space = " " if rng.random() < 0.5 else ""
        toks += _num_tokens_from_digits(ds)
        if space:
            toks.append(_sep())
        toks.append((u, uword))

    elif structure == "digits_currency":
        val = rng.choice([500, 1000, 2000, 50000, 125000, 25000, 999, 15000])
        indian = rng.random() < 0.6
        ds = _digits_str(rng, val, indian_grouping=indian)
        toks += _num_tokens_from_digits(ds)

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
    ("mult_chain", 5),
]


def build_phrase(pack, rng, structure=None, exclude_currency_bare=False):
    if structure is None:
        pool = _STRUCTURE_WEIGHTS
        if exclude_currency_bare:
            pool = [(s, w) for s, w in pool if s not in ("digits_currency", "bare_card_currency")]
        names = [s for s, _ in pool]
        weights = [w for _, w in pool]
        structure = rng.choices(names, weights=weights)[0]
    toks = build_term_tokens(pack, rng, structure)
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
        left = _num_tokens_from_digits(low_s)
        left = _maybe_repeat_unit(pack, rng, left, u, uword)
        # digit pairs must always use an explicit connector - bare-space
        # juxtaposition ("58 157 hazzar", "1 2 LAKH") only reads naturally
        # between two WORD forms ("do teen lakh", "dedh do lakh").
        connector = _range_connector(pack, rng, allow_juxtaposition=False)
        right = _num_tokens_from_digits(high_s)
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


def sample_example(pack, rng, max_len=MAX_LEN):
    for _attempt in range(50):
        try:
            ex = _sample_example_once(pack, rng)
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
    eng_forms = set()
    for forms in pack.english_fraction_phrases.values():
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


def _finalize(pack, rng, text, span_defs, is_negative):
    bio = [0] * len(text)
    clsids = [C.CLASS_TO_ID["O"]] * len(text)
    spans = []
    for start, end, toks in span_defs:
        span_tokens = toks
        _assert_prefix_before_number(pack, span_tokens)
        result = core.evaluate(span_tokens)
        currency = core.detect_currency(text, start, end, pack)
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
        check = core.evaluate(span_tokens)
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


def _apply_casing_and_wrap(pack, rng, text):
    if rng.random() < 0.10:
        frag = rng.choice(_CHAT_FRAGMENTS)
        if rng.random() < 0.5:
            text = frag + " " + text
        else:
            text = text + " " + frag
    return text


def _sample_example_once(pack, rng):
    family_weights = [
        ("casual", 20), ("classifieds", 20), ("news", 15), ("salary", 10),
        ("ranges", 8), ("two_spans", 10), ("bare", 5), ("negatives", 12),
    ]
    names = [f for f, _ in family_weights]
    weights = [w for _, w in family_weights]
    family = rng.choices(names, weights=weights)[0]
    template = rng.choice(pack.templates[family])

    if family == "negatives":
        if rng.random() < 0.15:
            n = rng.randint(3, 10)
            text = " ".join(_make_filler_word(pack, rng) for _ in range(n))
        else:
            text = template
        text = _apply_casing_and_wrap(pack, rng, text)
        return _finalize(pack, rng, text, [], is_negative=True)

    force_currency = family in ("classifieds", "salary") and "{C}" in template and rng.random() < 0.5
    parts = re.split(r"(\{[A-Z0-9]+\})", template)
    out = ""
    span_defs = []
    phrase_cache = {}

    if family == "ranges":
        phrase_cache["P"] = build_range_phrase(pack, rng)
    elif family == "two_spans":
        phrase_cache["P1"], _ = build_phrase(pack, rng, exclude_currency_bare=True)
        phrase_cache["P2"], _ = build_phrase(pack, rng, exclude_currency_bare=True)
    else:
        phrase_cache["P"], structure = build_phrase(pack, rng)
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
            if core.detect_currency(out, s, e, pack) is None:
                out = out[:e] + " rupaye" + out[e:]

    out = _insert_fillers(pack, rng, out, span_defs)

    out = _apply_casing_and_wrap(pack, rng, out)
    return _finalize(pack, rng, out, _recompute_offsets(out, span_defs), is_negative=False)


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


def generate(pack, n, seed=0, max_len=MAX_LEN):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        out.append(sample_example(pack, rng, max_len=max_len))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=str, default="data/train.jsonl")
    ap.add_argument("--lang", type=str, default="hi_latn")
    args = ap.parse_args(argv)

    pack = langbase.get_pack(args.lang)
    rng = random.Random(args.seed)
    with open(args.out, "w", encoding="utf-8") as f:
        for _ in range(args.n):
            ex = sample_example(pack, rng)
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"wrote {args.n} examples to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
