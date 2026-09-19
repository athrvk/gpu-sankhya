"""Deterministic evaluation core.

evaluate(tokens) takes a list of (cls, text) pairs for ONE span and returns a
Result. Nothing here references a specific language.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from . import classes as C


@dataclass
class Result:
    value: float
    range: Optional[Tuple[float, float]]
    unit: Optional[str]
    classes: List[str] = field(default_factory=list)
    currency: Optional[str] = None


# Devanagari (U+0966-U+096F) and Gujarati (U+0AE6-U+0AEF) digits -> ASCII
# "0"-"9", 1:1. Callers normally run text through charset.normalize_text()
# before labelling, which already maps Devanagari digits to ASCII -- but
# core.evaluate() is also called directly with raw (unnormalised) token
# text in tests and some callers, so DIGITS text reaching _merge_numbers is
# normalised here too, defensively, rather than assumed to already be ASCII.
_DIGIT_MAP = {}
for _i in range(10):
    _DIGIT_MAP[chr(0x0966 + _i)] = str(_i)
    _DIGIT_MAP[chr(0x0AE6 + _i)] = str(_i)
del _i


def _normalize_digits(text: str) -> str:
    return "".join(_DIGIT_MAP.get(ch, ch) for ch in text)


def _merge_numbers(tokens):
    """Drop SEP, merge consecutive DIGITS/DOT/COMMA runs into ('NUM', float)."""
    out = []
    buf = ""
    have_num = False

    def flush():
        nonlocal buf, have_num
        if have_num:
            cleaned = buf.replace(",", "")
            try:
                val = float(cleaned)
            except ValueError:
                val = 0.0
            out.append(("NUM", val))
        buf = ""
        have_num = False

    for cls, text in tokens:
        if cls == "SEP":
            continue
        if cls in ("DIGITS", "DOT", "COMMA"):
            if cls == "DIGITS":
                text = _normalize_digits(text)
            buf += text
            have_num = True
        else:
            flush()
            out.append((cls, text))
    flush()
    return out


def _collapse_repeated_prefixes(merged):
    """Consecutive tokens with the SAME PFX_* class, separated only by SEP
    (already dropped by _merge_numbers, so they're adjacent here), collapse
    into a single occurrence of that prefix. This handles decode artifacts
    where a single-word prefix phrase like "half a" or "three n half" gets
    split into two separate same-class tokens by BIO/char decoding: without
    this, evaluate() would double-count the prefix's standalone value."""
    out = []
    for cls, text in merged:
        if C.is_prefix(cls) and out and out[-1][0] == cls:
            continue
        out.append((cls, text))
    return out


def _flush_coef(num, pfx):
    if pfx is not None:
        info = C.PREFIX_INFO.get(pfx)
        if info is None:
            return num if num is not None else 1
        if num is not None and info.get("op"):
            op = info["op"]
            if op == "sub25":
                return num - 0.25
            if op == "add25":
                return num + 0.25
            if op == "add5":
                return num + 0.5
        return info["standalone"]
    if num is not None:
        return num
    return 1


def _eval_amount(tokens):
    """tokens: list of (cls, text) with SEP already droppable, RANGE excluded.

    Units within one amount are normally descending and additive
    ("ek crore bees lakh" = 1e7 + 20*1e5). But Hindi/Indian-English also
    stacks an ASCENDING unit on top of everything accumulated so far
    ("das hazaar crore" = 10,000 * 1e7, "sau crore" = 100 * 1e7,
    "2 lakh crore" = 2e5 * 1e7): when a new unit is LARGER than the unit
    that closed the previous term, the running total is MULTIPLIED by the
    new unit's value instead of a new additive term being appended.
    """
    merged = _collapse_repeated_prefixes(_merge_numbers(tokens))
    accum = 0
    last_unit_val = None
    max_unit_cls = None
    pending_num = None
    pending_pfx = None

    def close_term(unit_cls=None):
        nonlocal accum, last_unit_val, max_unit_cls, pending_num, pending_pfx
        coef = _flush_coef(pending_num, pending_pfx)
        if unit_cls is None:
            accum += coef * 1
        else:
            new_val = C.unit_value(unit_cls)
            if last_unit_val is not None and new_val > last_unit_val:
                accum = accum * new_val
            else:
                accum = accum + coef * new_val
            last_unit_val = new_val
            if max_unit_cls is None or new_val > C.unit_value(max_unit_cls):
                max_unit_cls = unit_cls
        pending_num = None
        pending_pfx = None

    for cls, text in merged:
        if cls == "NUM":
            if pending_num is not None:
                close_term(None)
            pending_num = text
        elif C.is_card(cls):
            if pending_num is not None:
                close_term(None)
            try:
                pending_num = C.card_value(cls)
            except Exception:
                pass
        elif C.is_prefix(cls):
            if pending_pfx is not None:
                close_term(None)
            pending_pfx = cls
        elif C.is_unit(cls):
            close_term(cls)
        else:
            # unknown / stray class: ignore leniently
            continue

    if pending_num is not None or pending_pfx is not None:
        close_term(None)

    return accum, max_unit_cls


def _num(v):
    """Return an int when the value is (numerically, up to fp noise) integral,
    else a float rounded to kill binary-fp dust (e.g. 928999.9999999999,
    3509000000000.0005). Uses a relative tolerance since large scale-unit
    products (kharab = 1e11) amplify absolute fp error."""
    if isinstance(v, float):
        nearest = round(v)
        tol = max(1e-6, abs(v) * 1e-9)
        if abs(v - nearest) < tol:
            return int(nearest)
        return round(v, 6)
    return v


def _split_terms(merged):
    """Split a merged (RANGE-free) token stream into terms, each being
    zero-or-more coefficient tokens (NUM/CARD_*/PFX_*) followed by at most
    one UNIT token that closes it. Returns a list of
    (start, end, has_explicit_coef, unit_cls_or_None) over indices into
    `merged`; a trailing unit-less term (if any) is included with
    unit_cls=None.
    """
    terms = []
    term_start = 0
    has_coef = False
    for idx, (cls, text) in enumerate(merged):
        if cls == "NUM" or C.is_card(cls) or C.is_prefix(cls):
            has_coef = True
        elif C.is_unit(cls):
            terms.append((term_start, idx + 1, has_coef, cls))
            term_start = idx + 1
            has_coef = False
        else:
            continue
    if term_start < len(merged):
        terms.append((term_start, len(merged), has_coef, None))
    return terms


def _try_juxtaposition_range(tokens):
    """R7: juxtaposed ranges with no connector word, e.g. "teen hazaar
    paanch hazaar" (3000..5000) or "bees lac pachees lac" (2e6..2.5e6).
    The model emits no RANGE token here, so the normal additive evaluator
    would sum the terms; this rule detects the shape first.

    A RANGE-free token stream is split into terms (coefficient + UNIT).
    If term i has an EXPLICIT coefficient (a NUM/CARD_*/PFX_* token, not
    the implicit "1") AND term i+1 also has an explicit coefficient AND
    unit_value(i+1) >= unit_value(i), the boundary between them is a
    juxtaposition range: low = eval(terms[..i]), high = eval(terms[i+1..]).
    Requires low < high strictly; low == high ("paanch lakh paanch lakh")
    is deliberately NOT treated as a range -- it's ambiguous with plain
    repetition/emphasis, so it falls back to today's additive behaviour
    (2 * value, no range).

    This must NOT fire on:
      - multiplicative stacking ("das hazaar crore"): the second term has
        NO coefficient, so has_coef is False.
      - descending additive chains ("ek lakh dus hazaar"): unit_value of
        the second term (hazaar) is SMALLER than the first (lakh).
      - cardinal-only juxtaposition ("do teen lakh"): already turned into
        an explicit RANGE token upstream by decode-time repair, so it
        never reaches here RANGE-free.

    Returns a Result on a match, else None (caller falls back).
    """
    merged = _collapse_repeated_prefixes(_merge_numbers(tokens))
    terms = _split_terms(merged)
    for i in range(len(terms) - 1):
        _, end_i, coef_i, unit_i = terms[i]
        start_j, _, coef_j, unit_j = terms[i + 1]
        if not (coef_i and coef_j and unit_i is not None and unit_j is not None):
            continue
        if C.unit_value(unit_j) < C.unit_value(unit_i):
            continue
        low, _ = _eval_amount(merged[:end_i])
        high, _ = _eval_amount(merged[start_j:])
        if not (low < high):
            continue
        return Result(
            value=_num(low),
            range=(_num(low), _num(high)),
            unit=C.UNIT_NAME.get(unit_i),
            classes=[c for c, _ in tokens],
        )
    return None


def _try_cardinal_juxtaposition_range(tokens):
    """R8: two directly-juxtaposed SPELLED cardinals with nothing but SEP
    between them, e.g. "teen paanch hazaar" (CARD_3 SEP CARD_5 SEP
    UNIT_HAZAAR, 3000..5000) or "tees paintees hazaar" (30000..35000).

    Unlike R7 (which splits at a boundary between two UNIT-closed terms),
    this fires when two CARD_* tokens sit back-to-back -- no unit, prefix
    or digits token between them -- and the second is strictly larger than
    the first. The low reading drops the larger cardinal, the high reading
    drops the smaller one; both are then evaluated normally (so any
    trailing unit, e.g. UNIT_HAZAAR, applies to each side).

    Must NOT fire on:
      - "ek sau" (CARD_1 UNIT_SAU): sau is a UNIT, not a second cardinal.
      - "do hazaar paanch" (CARD_2 UNIT_HAZAAR CARD_5): the cardinals are
        separated by a unit token, not merely SEP -- additive, unchanged.
      - descending "bees paanch" (CARD_20 CARD_5): second < first, so the
        a < b guard fails and today's behaviour (additive) is kept.
      - DIGITS-based juxtaposition: restricted to CARD_* (spelled) tokens.

    Only used when R7 does not already match (checked first by the
    caller), so unit-bearing juxtaposition terms (R7) take precedence.

    Returns a Result on a match, else None.
    """
    card_idxs = [i for i, (cls, _) in enumerate(tokens) if C.is_card(cls)]
    for k in range(len(card_idxs) - 1):
        i, j = card_idxs[k], card_idxs[k + 1]
        if not all(tokens[m][0] == "SEP" for m in range(i + 1, j)):
            continue
        cls_a = tokens[i][0]
        cls_b = tokens[j][0]
        try:
            a_val = C.card_value(cls_a)
            b_val = C.card_value(cls_b)
        except Exception:
            continue
        if not (a_val < b_val):
            continue
        low_tokens = tokens[:j] + tokens[j + 1:]
        high_tokens = tokens[:i] + tokens[i + 1:]
        low_res = evaluate(low_tokens)
        high_res = evaluate(high_tokens)
        if low_res.range is not None or high_res.range is not None:
            continue
        if not (low_res.value < high_res.value):
            continue
        return Result(
            value=low_res.value,
            range=(low_res.value, high_res.value),
            unit=low_res.unit,
            classes=[c for c, _ in tokens],
        )
    return None


def evaluate(tokens: List[Tuple[str, str]]) -> Result:
    """Evaluate a class-token sequence for one span. Never raises."""
    try:
        all_classes = [c for c, _ in tokens]

        # split at RANGE into at most 2 amounts
        parts = [[]]
        for cls, text in tokens:
            if cls == "RANGE":
                if len(parts) < 2:
                    parts.append([])
                continue
            parts[-1].append((cls, text))

        amounts = [_eval_amount(p) for p in parts if p]
        if not amounts:
            return Result(value=0, range=None, unit=None, classes=all_classes)

        if len(amounts) == 1:
            # R7: no RANGE token present -- check for a juxtaposition range
            # (e.g. "teen hazaar paanch hazaar") before the plain additive
            # evaluation below.
            r7 = _try_juxtaposition_range(parts[0])
            if r7 is not None:
                return r7
            # R8: no unit-bearing juxtaposition term boundary (R7) matched
            # -- check for two directly-adjacent spelled cardinals instead.
            r8 = _try_cardinal_juxtaposition_range(parts[0])
            if r8 is not None:
                return r8
            value, unit = amounts[0]
            return Result(value=_num(value), range=None, unit=(C.UNIT_NAME.get(unit) if unit else None), classes=all_classes)

        (v1, u1), (v2, u2) = amounts[0], amounts[1]

        # R6: a RANGE connector that actually sits between a DESCENDING
        # additive chain (e.g. "ek lakh dus hazaar" mistagged RANGE on the
        # space) -- both sides already carry their own unit and the left
        # value is strictly greater than the right -- is evaluated as one
        # additive amount instead of a [low, high] range. A genuine range
        # ("2-3 lakh", "paanch se sadhe saat lakh") has only ONE side
        # carrying a unit (the other is a bare number/prefix scaled by it
        # below), so it never hits this branch.
        if u1 is not None and u2 is not None and v1 > v2:
            total = _num(v1 + v2)
            return Result(value=total, range=None, unit=C.UNIT_NAME.get(u1), classes=all_classes)

        if u1 is None and u2 is not None:
            v1 = v1 * C.unit_value(u2)
            eff_unit = u2
        else:
            eff_unit = u1 if u1 is not None else u2

        low, high = (v1, v2) if v1 <= v2 else (v2, v1)
        return Result(
            value=_num(low),
            range=(_num(low), _num(high)),
            unit=(C.UNIT_NAME.get(eff_unit) if eff_unit else None),
            classes=all_classes,
        )
    except Exception:
        return Result(value=0, range=None, unit=None, classes=[c for c, _ in tokens])


def is_bare_digits(tokens: List[Tuple[str, str]]) -> bool:
    """R3: true when a span's meaningful tokens are ONLY digits (with
    DOT/COMMA/SEP glue) -- no UNIT_/PFX_/CARD_ token at all."""
    has_digits = False
    for cls, _ in tokens:
        if cls == "DIGITS":
            has_digits = True
        elif cls.startswith(("UNIT_", "PFX_", "CARD_")):
            return False
    return has_digits


def should_drop_bare_digits(tokens: List[Tuple[str, str]]) -> bool:
    """R3 (narrowed): whether a bare-digits span (see is_bare_digits) with
    NO detected currency should be dropped. Most bare numbers ARE amounts
    (rent, prices, quantities: "15000", "2,50,00,000", "125,000") and must
    be kept; only the two genuinely ambiguous shapes are dropped:

      (a) short (<=4 digits) AND ungrouped (no comma) -- route numbers,
          OTPs, years, house numbers: "1", "66", "4521", "2024", "302".
      (b) very long (>=10 digits) -- phone numbers: "9876543210".

    A grouping comma (Indian or Western) is itself a strong signal of a
    real amount, so it exempts an otherwise-short span from rule (a):
    "1,00,000" (6 digits, comma) is kept, not dropped.
    """
    if not is_bare_digits(tokens):
        return False
    digit_count = 0
    has_comma = False
    for cls, text in tokens:
        if cls == "DIGITS":
            digit_count += len(text)
        elif cls == "COMMA":
            has_comma = True
    if digit_count <= 4 and not has_comma:
        return True
    if digit_count >= 10:
        return True
    return False



def has_leading_zero_coefficient(tokens: List[Tuple[str, str]]) -> bool:
    """R10: a DIGITS token that starts with '0' and has 2+ digits ("05",
    "007") is never an amount coefficient -- it is a vehicle plate, PIN,
    date or phone fragment ("GJ05 CD 4567"). Only the FIRST digits token
    of the span counts; a digits token right after a DOT ("1.05 lakh") is
    a fractional part and is exempt. Mirrored in src/core.ts."""
    prev = None
    for cls, text in tokens:
        if cls == "DIGITS":
            if prev != "DOT" and len(text) >= 2 and text[0] in "0\u0966\u0ae6":
                return True
            return False
        if cls not in ("SEP", "COMMA"):
            prev = cls
    return False

_IGNORED_GLUE_CLASSES = ("SEP", "RANGE", "DOT", "COMMA", "O")


def should_drop_lone_ambiguous_unit(tokens: List[Tuple[str, str]]) -> bool:
    """R4: whether a span consisting of a single ambiguous unit word (with
    no preceding number) should be dropped. "kharab" (Hindi: usually
    "broken") and "mil"/"million" (often just an English loanword/fragment)
    are only genuine amount units when a number precedes them ("das
    kharab", "2 mil"); alone they are almost always false positives
    ("washing machine kharab ho gaya", "mil ke rehna").

    True when the span's meaningful tokens (ignoring SEP/RANGE/DOT/COMMA/O
    glue) consist of exactly one token whose class is UNIT_KHARAB or
    UNIT_MILLION, and there is no CARD_*/DIGITS/PFX_* token anywhere in the
    span. This is class-level (not surface-level), so it needs no lexicon.
    """
    meaningful = [(cls, text) for cls, text in tokens if cls not in _IGNORED_GLUE_CLASSES]
    if len(meaningful) != 1:
        return False
    cls, _ = meaningful[0]
    if cls not in ("UNIT_KHARAB", "UNIT_MILLION"):
        return False
    for c, _ in tokens:
        if c == "DIGITS" or c.startswith("CARD_") or c.startswith("PFX_"):
            return False
    return True


def detect_currency(text: str, start: int, end: int, pack) -> Optional[str]:
    """Scan up to 8 chars before/after [start,end) for a currency marker.

    `pack` may be a single LanguagePack, or a list of packs (multi-lang runs)
    in which case the union of both packs' marker lists is scanned.

    The window has to be wider than the longest marker word (6 chars, e.g.
    "rupaye"/"rupees"): the span boundary sits right after the number, so
    the gap before the marker word starts (whitespace, and occasionally
    punctuation like ", ") eats into a same-sized window and truncates the
    word -- "1200 rupees" with a 6-char after-window is " rupee" (missing
    the final "s"), so "rupees" never matches with startswith(). 8 leaves
    slack for a couple of separator chars ahead of the longest marker.
    """
    packs = pack if isinstance(pack, (list, tuple)) else [pack]

    before = text[max(0, start - 8):start]
    after = text[end:end + 8]

    before_stripped = before.rstrip()
    after_stripped = after.lstrip()

    for p in packs:
        for marker in getattr(p, "currency_markers_before", []):
            m = marker.rstrip(".").lower()
            bs = before_stripped.lower()
            if bs.endswith(marker.lower()) or bs.endswith(m):
                return "INR"
        for word in getattr(p, "currency_words_after", []):
            w = word.lower()
            a = after_stripped.lower()
            if a.startswith(w):
                return "INR"

    return None


def detect_currency_multi(text: str, start: int, end: int, packs) -> Optional[str]:
    """Explicit multi-pack variant of detect_currency (same behaviour)."""
    return detect_currency(text, start, end, list(packs))
