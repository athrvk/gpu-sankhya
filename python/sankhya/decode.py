"""Decode per-char BIO + class predictions into spans."""
from __future__ import annotations

from typing import List, Tuple


def decode_spans(text: str, bio_ids: List[int], cls_ids: List[int]):
    """Lenient BIO decode.

    A span starts at B (1), or at I (2) following an O (lenient recovery).
    A span ends right before the next O (0) or B (1).
    Within a span, tokens are runs of identical cls id: (cls, text_substring).

    Returns list of dicts: {start, end, text, tokens: [(cls_id, substr), ...]}
    """
    n = len(text)
    spans = []
    start = None
    i = 0
    while i < n:
        b = bio_ids[i]
        if b == 1:
            if start is None:
                start = i
            else:
                spans.append((start, i))
                start = i
        elif b == 2:
            if start is None:
                start = i  # lenient: I after O
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
        out.append({"start": s, "end": e, "text": text[s:e], "tokens": toks})
    return out
