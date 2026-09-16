"""Decode per-char BIO + class predictions into spans."""
from __future__ import annotations

import unicodedata
from typing import List, Optional, Sequence

from . import classes as C

_TRIM_CLASSES = {"SEP", "RANGE", "DOT", "COMMA"}
_MEANINGFUL_PREFIXES = ("PFX_", "CARD_", "UNIT_")
_RANGE_CONNECTORS = ("-", "–", "/")  # hyphen, en-dash, slash


def _is_meaningful(cls_name: str) -> bool:
    return cls_name.startswith(_MEANINGFUL_PREFIXES) or cls_name == "DIGITS"


def _is_vote_worthy(cls_name: str) -> bool:
    """Classes a letters-run majority vote counts (excludes O/SEP/RANGE/DIGITS/DOT/COMMA)."""
    return cls_name.startswith(_MEANINGFUL_PREFIXES) and cls_name != "DIGITS"


def _char_type(ch: str) -> str:
    # Devanagari combining marks (matras, nukta, virama/halant, anusvara,
    # chandrabindu -- Unicode category Mn/Mc/Me) are not alphabetic per
    # ch.isalpha(), but must count as "letter" for run-splitting or every
    # matra (e.g. the -f in dedh) would be forced into its own run,
    # splitting a single word mid-character. Match category L* or M*
    # (mirrors the JS side's \p{L}|\p{M}).
    cat = unicodedata.category(ch)
    if cat.startswith("L") or cat.startswith("M"):
        return "letter"
    if ch.isdigit():
        return "digit"
    if ch.isspace():
        return "space"
    return "other"


def _longest_run(indices: List[int]) -> int:
    """indices is a sorted list of positions; returns the length of the longest
    contiguous (consecutive-integer) sub-run."""
    best = cur = 1
    for i in range(1, len(indices)):
        if indices[i] == indices[i - 1] + 1:
            cur += 1
        else:
            cur = 1
        best = max(best, cur)
    return best


def _majority_vote_run(raw_ids: List[int]) -> int:
    """Whole-run majority vote over vote-worthy classes, tie-broken by the
    longest contiguous sub-run. Falls back to O if no char in the run voted."""
    votes = {}
    for idx, cid in enumerate(raw_ids):
        if _is_vote_worthy(C.CLASSES[cid]):
            votes.setdefault(cid, []).append(idx)
    if not votes:
        return C.CLASS_TO_ID["O"]
    max_count = max(len(v) for v in votes.values())
    candidates = [cid for cid, v in votes.items() if len(v) == max_count]
    if len(candidates) == 1:
        return candidates[0]
    best_cid, best_len = candidates[0], -1
    for cid in candidates:
        run_len = _longest_run(votes[cid])
        if run_len > best_len:
            best_len = run_len
            best_cid = cid
    return best_cid


def _repair_letters_run(raw_ids: List[int]) -> List[int]:
    """Sub-run smoothing within one letters run.

    Splits raw_ids into maximal sub-runs of identical class. Sub-runs of
    length >= 3 are kept as-is (this preserves joined forms like "dedhlakh"
    where "dedh"|"lakh" are both length-4 sub-runs). A shorter sub-run
    (length 1-2) is reassigned to the class of its longer adjacent sub-run
    that has length >= 3 (prefer the longer neighbour; tie -> prefer the
    left neighbour). If neither neighbour qualifies (length >= 3), the
    positions fall back to a whole-run majority vote (same as before).
    """
    n = len(raw_ids)
    sub_runs = []  # (cls_id, start, end) relative to the run
    i = 0
    while i < n:
        c = raw_ids[i]
        j = i + 1
        while j < n and raw_ids[j] == c:
            j += 1
        sub_runs.append((c, i, j))
        i = j

    result = list(raw_ids)
    fallback_positions = []

    for idx, (c, s, e) in enumerate(sub_runs):
        if e - s >= 3:
            continue  # keep as-is
        prev_run = sub_runs[idx - 1] if idx > 0 else None
        next_run = sub_runs[idx + 1] if idx < len(sub_runs) - 1 else None
        prev_ok = prev_run is not None and (prev_run[2] - prev_run[1]) >= 3
        next_ok = next_run is not None and (next_run[2] - next_run[1]) >= 3

        if prev_ok and next_ok:
            prev_len = prev_run[2] - prev_run[1]
            next_len = next_run[2] - next_run[1]
            new_c = prev_run[0] if prev_len >= next_len else next_run[0]
        elif prev_ok:
            new_c = prev_run[0]
        elif next_ok:
            new_c = next_run[0]
        else:
            new_c = None
            fallback_positions.extend(range(s, e))

        if new_c is not None:
            for k in range(s, e):
                result[k] = new_c

    if fallback_positions:
        maj = _majority_vote_run(raw_ids)
        for k in fallback_positions:
            result[k] = maj

    return result


def bridge_bio(text: str, bio_ids: List[int]) -> List[int]:
    """BIO repair (applied before class repair): a single O-tagged char that
    has B/I immediately before it AND I immediately after it becomes I --
    bridging a one-char gap inside what should be a single contiguous span.
    Fixes e.g. "five and a half" where the space right after "five" gets
    raw-tagged O, and "₹85000" where the "5" right after the currency
    symbol/first digit gets raw-tagged O.
    """
    out = list(bio_ids)
    n = len(out)
    for i in range(1, n - 1):
        if out[i] == 0 and out[i - 1] in (1, 2) and out[i + 1] == 2:
            out[i] = 2
    return out


def extend_digit_spans(text: str, bio_ids: List[int]) -> List[int]:
    """BIO repair (applied before class repair, after bridge_bio): if an
    in-span (B/I) run ends on a digit character and the immediately
    following character(s) are also digits but raw-tagged O, extend the
    span (tag them I) through that digit run. Fixes e.g. "15000/month"
    decoding as just "1500" (dropping the trailing digit).
    """
    out = list(bio_ids)
    n = len(out)
    i = 0
    while i < n:
        if out[i] in (1, 2):
            j = i
            while j + 1 < n and out[j + 1] in (1, 2):
                j += 1
            if text[j].isdigit():
                k = j + 1
                while k < n and out[k] == 0 and text[k].isdigit():
                    out[k] = 2
                    k += 1
            i = j + 1
        else:
            i += 1
    return out


def repair_classes(text: str, start: int, end: int, cls_ids: List[int]) -> List[int]:
    """Deterministically repair per-char class predictions within [start, end)
    of a decoded span, using character-type structure rather than the raw
    per-char argmax. Fixes cases like a stray "." inside a digit run being
    tagged O, or a letters-run splitting into multiple wrong classes.

    Returns a new list (same length as cls_ids) with only [start, end) changed.
    """
    out = list(cls_ids)
    if start >= end:
        return out

    # split [start, end) into maximal runs by char type
    runs = []  # (type, run_start, run_end)
    i = start
    while i < end:
        t = _char_type(text[i])
        j = i + 1
        while j < end and _char_type(text[j]) == t:
            j += 1
        runs.append((t, i, j))
        i = j

    o_id = C.CLASS_TO_ID["O"]
    sep_id = C.CLASS_TO_ID["SEP"]
    range_id = C.CLASS_TO_ID["RANGE"]
    digits_id = C.CLASS_TO_ID["DIGITS"]
    dot_id = C.CLASS_TO_ID["DOT"]
    comma_id = C.CLASS_TO_ID["COMMA"]

    for idx, (t, rs, re_) in enumerate(runs):
        if t == "digit":
            for k in range(rs, re_):
                out[k] = digits_id

        elif t == "space":
            any_range = any(C.CLASSES[cls_ids[k]] == "RANGE" for k in range(rs, re_))
            new_cls = range_id if any_range else sep_id
            for k in range(rs, re_):
                out[k] = new_cls

        elif t == "letter":
            raw_run = [cls_ids[k] for k in range(rs, re_)]
            new_run = _repair_letters_run(raw_run)
            for offset, k in enumerate(range(rs, re_)):
                out[k] = new_run[offset]

        else:  # "other" punctuation run
            ch = text[rs:re_]
            length = re_ - rs
            prev_type = runs[idx - 1][0] if idx > 0 else None
            next_type = runs[idx + 1][0] if idx < len(runs) - 1 else None

            if length == 1 and ch in (".", ",") and prev_type == "digit" and next_type == "digit":
                new_cls = dot_id if ch == "." else comma_id
            elif ch in _RANGE_CONNECTORS and prev_type == "digit" and next_type == "digit":
                new_cls = range_id
            elif ch == "-" and prev_type == "letter" and next_type == "letter":
                new_cls = sep_id
            else:
                new_cls = sep_id
            for k in range(rs, re_):
                out[k] = new_cls

    return out


def decode_spans(
    text: str,
    bio_ids: List[int],
    cls_ids: List[int],
    bio_probs: Optional[Sequence[Sequence[float]]] = None,
):
    """Strict BIO decode with filtering and deterministic class repair.

    A span starts ONLY at B (1). An I (2) following O is treated as O (ignored).
    A span ends right before the next O (0) or B (1).

    Before tokens are built, each span's per-char classes are passed through
    `repair_classes` (majority-vote over char-type runs), which fixes stray
    single-char misclassifications (a "." inside a digit run tagged O, a
    letters run split across two wrong classes, etc.) without touching BIO.
    Within a span, tokens are then runs of identical (repaired) cls id:
    (cls, text_substring).

    Post-processing:
    - spans whose token classes contain no PFX_*/CARD_*/UNIT_*/DIGITS are dropped.
    - leading/trailing SEP/RANGE/DOT/COMMA tokens are trimmed (start/end shrink).
    - if bio_probs is given (per-char list of [p_O, p_B, p_I]), a "confidence"
      field (mean over the span of the max BIO prob per char) is computed, and
      spans with confidence < 0.5 are dropped.

    Before spans are found, the raw BIO sequence itself is repaired in two
    passes (see their docstrings): `bridge_bio` closes single-char O gaps
    inside a span, then `extend_digit_spans` extends a span forward through
    a trailing digit run it was cut short of.

    Returns list of dicts: {start, end, text, tokens: [(cls_id, substr), ...], confidence}
    """
    bio_ids = bridge_bio(text, bio_ids)
    bio_ids = extend_digit_spans(text, bio_ids)

    n = len(text)
    spans = []
    start = None
    i = 0
    while i < n:
        b = bio_ids[i]
        if b == 1:
            if start is not None:
                spans.append((start, i))
            start = i
        elif b == 2:
            pass  # I is only valid inside an already-open span; otherwise ignored
        else:  # O
            if start is not None:
                spans.append((start, i))
                start = None
        i += 1
    if start is not None:
        spans.append((start, n))

    work_cls = list(cls_ids)

    out = []
    for s, e in spans:
        work_cls = repair_classes(text, s, e, work_cls)

        # token runs with (cls_id, start, end) offsets
        run_start = s
        orig_toks = []
        for j in range(s + 1, e + 1):
            if j == e or work_cls[j] != work_cls[run_start]:
                orig_toks.append((work_cls[run_start], run_start, j))
                run_start = j

        # trim leading/trailing trim-classes
        lo, hi = 0, len(orig_toks)
        while lo < hi and C.CLASSES[orig_toks[lo][0]] in _TRIM_CLASSES:
            lo += 1
        while hi > lo and C.CLASSES[orig_toks[hi - 1][0]] in _TRIM_CLASSES:
            hi -= 1
        kept = orig_toks[lo:hi]
        if not kept:
            continue
        new_start = kept[0][1]
        new_end = kept[-1][2]

        if not any(_is_meaningful(C.CLASSES[cid]) for cid, _, _ in kept):
            continue

        confidence = None
        if bio_probs is not None:
            probs = []
            for j in range(new_start, new_end):
                probs.append(max(bio_probs[j]))
            confidence = sum(probs) / len(probs) if probs else 0.0
            if confidence < 0.5:
                continue

        out.append({
            "start": new_start,
            "end": new_end,
            "text": text[new_start:new_end],
            "tokens": [(cid, text[a:b]) for cid, a, b in kept],
            "confidence": confidence,
        })
    return out
