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
            buf += text
            have_num = True
        else:
            flush()
            out.append((cls, text))
    flush()
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
    merged = _merge_numbers(tokens)
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
            value, unit = amounts[0]
            return Result(value=_num(value), range=None, unit=(C.UNIT_NAME.get(unit) if unit else None), classes=all_classes)

        (v1, u1), (v2, u2) = amounts[0], amounts[1]
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


def detect_currency(text: str, start: int, end: int, pack) -> Optional[str]:
    """Scan up to 4 chars before/after [start,end) for a currency marker from pack."""
    before = text[max(0, start - 4):start]
    after = text[end:end + 6]

    before_stripped = before.rstrip()
    for marker in getattr(pack, "currency_markers_before", []):
        m = marker.rstrip(".").lower()
        bs = before_stripped.lower()
        if bs.endswith(marker.lower()) or bs.endswith(m):
            return "INR"

    after_stripped = after.lstrip()
    for word in getattr(pack, "currency_words_after", []):
        w = word.lower()
        a = after_stripped.lower()
        if a.startswith(w):
            return "INR"

    return None
