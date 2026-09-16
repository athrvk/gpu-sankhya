"""Decode per-char BIO + class predictions into spans."""
from __future__ import annotations

from typing import List, Optional, Sequence

from . import classes as C

_TRIM_CLASSES = {"SEP", "RANGE", "DOT", "COMMA"}
_MEANINGFUL_PREFIXES = ("PFX_", "CARD_", "UNIT_")


def _is_meaningful(cls_name: str) -> bool:
    return cls_name.startswith(_MEANINGFUL_PREFIXES) or cls_name == "DIGITS"


def decode_spans(
    text: str,
    bio_ids: List[int],
    cls_ids: List[int],
    bio_probs: Optional[Sequence[Sequence[float]]] = None,
):
    """Strict BIO decode with filtering.

    A span starts ONLY at B (1). An I (2) following O is treated as O (ignored).
    A span ends right before the next O (0) or B (1).
    Within a span, tokens are runs of identical cls id: (cls, text_substring).

    Post-processing:
    - spans whose token classes contain no PFX_*/CARD_*/UNIT_*/DIGITS are dropped.
    - leading/trailing SEP/RANGE/DOT/COMMA tokens are trimmed (start/end shrink).
    - if bio_probs is given (per-char list of [p_O, p_B, p_I]), a "confidence"
      field (mean over the span of the max BIO prob per char) is computed, and
      spans with confidence < 0.5 are dropped.

    Returns list of dicts: {start, end, text, tokens: [(cls_id, substr), ...], confidence}
    """
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

    out = []
    for s, e in spans:
        toks = []
        run_start = s
        for j in range(s + 1, e + 1):
            if j == e or cls_ids[j] != cls_ids[run_start]:
                toks.append((cls_ids[run_start], text[run_start:j]))
                run_start = j

        # recompute token runs with (cls_id, start, end) offsets
        run_start = s
        orig_toks = []
        for j in range(s + 1, e + 1):
            if j == e or cls_ids[j] != cls_ids[run_start]:
                orig_toks.append((cls_ids[run_start], run_start, j))
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
