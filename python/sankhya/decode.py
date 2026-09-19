"""Decode per-char BIO + class predictions into spans."""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import List, Optional, Sequence

from . import classes as C
from .verify import is_bound_form

_TRIM_CLASSES = {"SEP", "RANGE", "DOT", "COMMA"}
# R2: after word-integrity/possessive repair a partially-retagged word can
# leave an "O" token sitting at a span edge; trimming it away too (in
# addition to the original SEP/RANGE/DOT/COMMA) is how start/end get
# re-derived to the min/max of the remaining non-O chars.
_TRIM_CLASSES_FINAL = _TRIM_CLASSES | {"O"}
_MEANINGFUL_PREFIXES = ("PFX_", "CARD_", "UNIT_")
_RANGE_CONNECTORS = ("-", "–", "—", "/")  # hyphen, en-dash, em-dash, slash
_O_ID = C.CLASS_TO_ID["O"]
_POSSESSIVE_RE = re.compile(r"[\'’]([A-Za-zऀ-ॿ]{1,2})(?![A-Za-zऀ-ॿ])")
_SYMBOL_UNIT_CHARS = set("kKlL")


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


_BOUND_PARTITION_KINDS = ("PFX_", "CARD_", "UNIT_")


def _kind_index(cname: str) -> int:
    """Position of `cname` in the PFX -> CARD -> UNIT fused-word order, or
    -1 for anything else (O/SEP/DIGITS/...)."""
    for i, p in enumerate(_BOUND_PARTITION_KINDS):
        if cname.startswith(p):
            return i
    return -1


def _bound_keep_indices(sub_runs, run_text: Optional[str] = None) -> frozenset:
    """R2c: which sub-runs of ONE letters run are kept on their own class
    instead of being smoothed into a neighbour.

    A fused (written-solid) number word splits the letters run into two or
    three unanimous sub-runs following the order PFX? CARD? UNIT? -- at
    least two of the three present, each kind at most once, in that order:

      - CARD + UNIT  -- a bound unit suffix, Marathi "दोनशे" (2 + 100),
        Gujarati "બસો" (2 + 100), Hindi "दोसौ".
      - PFX + UNIT   -- "dedhlakh" (1.5 * 100000).
      - PFX + CARD   -- Marathi "पावणेचार" (0.75 + 4 = 3.75), "साडेआठ".
      - PFX + CARD + UNIT -- Gujarati "સાડાત્રણસો" (PFX_SAADHE + CARD_3 +
        UNIT_SAU = 350), the three-part case.

    Every sub-run must stand on its own evidence: length >= 2 (unanimous,
    not a stray 1-char misprediction). The ONE exception is a 1-char
    sub-run whose surface is a BOUND number form declared by a language
    pack for exactly the surface that follows it (`bound_forms` in the
    exported lexicon, the same map `verify.verify_tokens` consults) --
    Gujarati "બ" is CARD_2 only in "બસો", so "બસો" resolves to CARD_2 +
    UNIT_SAU = 200 while a stray 1-char sub-run anywhere else is still
    smoothed away. `run_text` is the run's surface (same indexing as
    `sub_runs`); without it no 1-char sub-run can be justified.

    Anything that does not match this shape returns an empty set, i.e.
    today's majority-vote / neighbour-adoption smoothing applies as before.
    """
    n = len(sub_runs)
    if n < 2 or n > 3:
        return frozenset()
    kinds = []
    for c, _s, _e in sub_runs:
        k = _kind_index(C.CLASSES[c])
        if k < 0:
            return frozenset()
        kinds.append(k)
    for i in range(n - 1):
        if kinds[i] >= kinds[i + 1]:  # strictly increasing: PFX < CARD < UNIT
            return frozenset()
    for idx, (c, s, e) in enumerate(sub_runs):
        if e - s >= 2:
            continue
        if e - s == 1 and run_text is not None and idx + 1 < n:
            _nc, ns, ne = sub_runs[idx + 1]
            if is_bound_form(run_text[s:e], C.CLASSES[c], run_text[ns:ne]):
                continue
        return frozenset()
    return frozenset(range(n))


def _repair_letters_run(raw_ids: List[int], run_text: Optional[str] = None) -> List[int]:
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
    keep = _bound_keep_indices(sub_runs, run_text)

    for idx, (c, s, e) in enumerate(sub_runs):
        if e - s >= 3 or idx in keep:
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


_MAX_WORD_RUN_LEN = 24


def _run_unanimous_class(raw_run: List[int], run_text: Optional[str] = None) -> Optional[int]:
    """Like `_repair_letters_run`'s sub-run smoothing, but returns a class
    id ONLY when every sub-run resolves from real, DIRECT evidence -- its
    own length >= 3, or adoption from an immediately adjacent sub-run of
    length >= 3 -- and all of those resolved classes agree. If any sub-run
    would need the whole-run majority-vote fallback (no qualifying
    neighbour on either side), that's diffuse/weak evidence -- this
    returns None rather than trust it, even if a fallback vote would
    happen to be unanimous (e.g. a 6-char run with 3 scattered chars of
    one class and 3 O's: majority-vote alone would call that unanimous,
    but no sub-run of it ever reaches 3 in a row on its own or via a
    qualifying neighbour, so it must stay untrusted)."""
    n = len(raw_run)
    sub_runs = []
    i = 0
    while i < n:
        c = raw_run[i]
        j = i + 1
        while j < n and raw_run[j] == c:
            j += 1
        sub_runs.append((c, i, j))
        i = j

    resolved = []
    keep = _bound_keep_indices(sub_runs, run_text)
    for idx, (c, s, e) in enumerate(sub_runs):
        if e - s >= 3 or idx in keep:
            resolved.append(c)
            continue
        prev_run = sub_runs[idx - 1] if idx > 0 else None
        next_run = sub_runs[idx + 1] if idx < len(sub_runs) - 1 else None
        prev_ok = prev_run is not None and (prev_run[2] - prev_run[1]) >= 3
        next_ok = next_run is not None and (next_run[2] - next_run[1]) >= 3
        if prev_ok and next_ok:
            prev_len = prev_run[2] - prev_run[1]
            next_len = next_run[2] - next_run[1]
            resolved.append(prev_run[0] if prev_len >= next_len else next_run[0])
        elif prev_ok:
            resolved.append(prev_run[0])
        elif next_ok:
            resolved.append(next_run[0])
        else:
            return None  # would need the whole-run fallback -- untrusted
    if len(set(resolved)) == 1:
        return resolved[0]
    return None


def extend_word_integrity_bio(text: str, bio_ids: List[int], cls_ids: List[int], letter_runs: List[tuple]) -> List[int]:
    """BIO repair (applied after bridge_bio/extend_digit_spans, before span
    construction): a word-class (UNIT_/PFX_/CARD_) token whose predicted
    span only PARTIALLY covers its letter word is not necessarily a bad
    match -- if the whole run's RAW per-char classes unanimously resolve
    (via `_run_unanimous_class`, direct-evidence-only sub-run smoothing)
    to a SINGLE word class, the partial coverage is just low-confidence
    BIO dropout on an otherwise-agreed word, not a genuinely mixed/
    ambiguous run. Extend the span's BIO to cover the whole run (I for
    every char, B at the run start) instead of leaving it for R2 to drop
    -- this also merges spans that were split by an internal O gap inside
    that run back into a single span (e.g. "unnasi" B I I O I O ->
    B I I I I I).

    A run whose raw classes do NOT resolve to one unanimous word class
    this way (e.g. only part of the run predicts a word class, or the
    agreement is only visible via `_repair_letters_run`'s diffuse
    whole-run majority-vote fallback rather than direct evidence) is left
    alone, so R2's drop still applies to it. Runs longer than 24 chars are
    skipped.
    """
    out = list(bio_ids)
    for rs, re_ in letter_runs:
        length = re_ - rs
        if length == 0 or length > _MAX_WORD_RUN_LEN:
            continue
        covered = [bio_ids[k] in (1, 2) for k in range(rs, re_)]
        if not any(covered) or all(covered):
            continue  # nothing to extend: no span touches it, or already full
        raw_run = [cls_ids[k] for k in range(rs, re_)]
        unanimous = _run_unanimous_class(raw_run, text[rs:re_])
        if unanimous is None or not _is_word_class(C.CLASSES[unanimous]):
            continue
        for k in range(rs, re_):
            out[k] = 2
        out[rs] = 1
    return out


def _run_repr_class(run_type: str, rs: int, re_: int, cls_ids: List[int], text: Optional[str] = None) -> Optional[str]:
    """Best-guess resulting class name for a run, without mutating cls_ids.
    Used by the R4 connector check to look at a letters-run neighbour that
    has not been processed yet (it comes after the connector in scan order).
    """
    if run_type == "digit":
        return "DIGITS"
    if run_type == "letter":
        raw_run = [cls_ids[k] for k in range(rs, re_)]
        new_run = _repair_letters_run(raw_run, text[rs:re_] if text is not None else None)
        cnt = Counter(new_run)
        best_id = max(cnt.items(), key=lambda kv: kv[1])[0]
        return C.CLASSES[best_id]
    return None


def _effective_neighbor(runs, idx: int, direction: int):
    """The adjacent digit/letter run in `direction`, skipping at most one
    intervening space run (so "2 - 3" and "2-3" are treated the same)."""
    j = idx + direction
    if j < 0 or j >= len(runs):
        return None
    if runs[j][0] == "space":
        j += direction
        if j < 0 or j >= len(runs):
            return None
    return runs[j] if runs[j][0] in ("digit", "letter") else None


def _connector_is_range(runs, idx: int, cls_ids: List[int], text: Optional[str] = None) -> bool:
    """R4: true when the connector genuinely sits between two SEPARATE
    numeric amounts (a real range), not inside one compound number.

    The left neighbour must be able to END an amount by itself (any
    meaningful class: it may be a bare number/prefix, e.g. "2-3", or a
    number that has already picked up a unit, e.g. "2 lakh/3 lakh"). The
    right neighbour must be able to START a fresh amount: DIGITS/CARD_/
    PFX_, but NOT a bare UNIT_ -- a unit can only ATTACH to a number that
    precedes it, so "dedh-lakh" (PFX_DHAI - UNIT_LAKH, one compound number
    "1.5 lakh") must stay SEP, while "2 lakh/3 lakh" (UNIT_LAKH - DIGITS,
    two separate amounts) becomes RANGE.
    """
    left = _effective_neighbor(runs, idx, -1)
    right = _effective_neighbor(runs, idx, 1)
    if left is None or right is None:
        return False
    left_cls = _run_repr_class(left[0], left[1], left[2], cls_ids, text)
    right_cls = _run_repr_class(right[0], right[1], right[2], cls_ids, text)
    if not left_cls or not right_cls or not _is_meaningful(left_cls):
        return False
    return right_cls == "DIGITS" or right_cls.startswith(("CARD_", "PFX_"))


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
            new_run = _repair_letters_run(raw_run, text[rs:re_])
            for offset, k in enumerate(range(rs, re_)):
                out[k] = new_run[offset]

        else:  # "other" punctuation run
            ch = text[rs:re_]
            length = re_ - rs
            prev_type = runs[idx - 1][0] if idx > 0 else None
            next_type = runs[idx + 1][0] if idx < len(runs) - 1 else None

            if length == 1 and ch in (".", ",") and prev_type == "digit" and next_type == "digit":
                new_cls = dot_id if ch == "." else comma_id
            elif length == 1 and ch in _RANGE_CONNECTORS and _connector_is_range(runs, idx, cls_ids, text):
                new_cls = range_id
            elif ch == "-" and prev_type == "letter" and next_type == "letter":
                new_cls = sep_id
            else:
                new_cls = sep_id
            for k in range(rs, re_):
                out[k] = new_cls

    return out


def _letter_runs(text: str) -> List[tuple]:
    """[(start, end), ...] maximal letter/mark runs over the WHOLE text (not
    just one span) -- word integrity needs the true word boundaries, which
    can extend outside a span that was cut short mid-word."""
    runs = []
    i, n = 0, len(text)
    while i < n:
        if _char_type(text[i]) == "letter":
            j = i + 1
            while j < n and _char_type(text[j]) == "letter":
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def _is_word_class(cname: str) -> bool:
    return cname.startswith(_MEANINGFUL_PREFIXES) and cname != "DIGITS"


def _repair_word_integrity(text: str, s: int, e: int, cls_ids: List[int], letter_runs: List[tuple]) -> List[int]:
    """R2: a word-class (UNIT_/PFX_/CARD_) token must cover its entire
    letter word. Checked per WHOLE letters-run rather than per token, so a
    run legitimately split into several back-to-back meaningful tokens
    (e.g. "dedhlakh" = PFX_DEDH|UNIT_LAKH, both length-4 sub-runs) is left
    alone; a run that is only PARTIALLY meaningful (some chars fall back to
    O, e.g. "km" tagged UNIT_.../O) has ALL its meaningful chars retagged
    O too -- a partial match without an explanation for the rest of the
    word is untrustworthy. A single-character symbol unit (k/K/l/L) is
    additionally only valid when the following character is not a letter,
    even when it fully (and only) covers its own run."""
    out = list(cls_ids)
    for rs, re_ in letter_runs:
        if re_ <= s or rs >= e:
            continue  # run doesn't touch this span at all
        crs, cre = max(rs, s), min(re_, e)
        fully_covered = rs >= s and re_ <= e and all(_is_word_class(C.CLASSES[out[k]]) for k in range(rs, re_))
        if fully_covered:
            if re_ - rs == 1 and C.CLASSES[out[rs]].startswith("UNIT_") and text[rs] in _SYMBOL_UNIT_CHARS:
                if re_ < len(text) and _char_type(text[re_]) == "letter":
                    out[rs] = _O_ID
            continue
        for k in range(crs, cre):
            if _is_word_class(C.CLASSES[out[k]]):
                out[k] = _O_ID
    return out


def _repair_possessive(text: str, s: int, e: int, cls_ids: List[int]) -> List[int]:
    """R5: an apostrophe followed by 1-2 letters at the end of a word (e.g.
    "lakh's") is a possessive/genitive suffix, not part of the amount --
    retag it (and those letters) O."""
    out = list(cls_ids)
    span_text = text[s:e]
    for m in _POSSESSIVE_RE.finditer(span_text):
        for k in range(s + m.start(), s + m.end()):
            out[k] = _O_ID
    return out


def _majority_class_any(raw_ids: List[int]) -> int:
    """Like `_majority_vote_run`, but counts votes over ALL classes (not just
    the word-class subset `_is_vote_worthy` cares about) -- used for a
    connector WORD run (e.g. "se"/"से"), whose relevant target class is
    RANGE itself, not a UNIT_/CARD_/PFX_ word class."""
    votes = {}
    for cid in raw_ids:
        votes.setdefault(cid, []).append(0)
    max_count = max(len(v) for v in votes.values())
    candidates = [cid for cid, v in votes.items() if len(v) == max_count]
    if len(candidates) == 1:
        return candidates[0]
    # tie-break by longest contiguous sub-run of that class, left-most wins
    best_cid, best_len = candidates[0], -1
    for cid in candidates:
        idxs = [i for i, c in enumerate(raw_ids) if c == cid]
        run_len = _longest_run(idxs)
        if run_len > best_len:
            best_len = run_len
            best_cid = cid
    return best_cid


def _merge_range_connector_spans(text: str, cls_ids: List[int], spans_out: List[dict]) -> List[dict]:
    """R4b: like `_connector_is_range` (punctuation connectors), but for a
    whole WORD connector (e.g. "se"/"से") that sits between two already-
    decoded spans. The class head can correctly tag such a connector word
    RANGE while the BIO head incorrectly emits its own B mid-word, splitting
    what should be one span into two (e.g. "do lakh se teen lakh" decoding
    as two separate amounts instead of one RANGE span). If the gap between
    two adjacent spans is exactly [optional whitespace] + one letters run
    (Latin or Devanagari, including combining marks) + [optional
    whitespace], and that letters run's repaired/majority class is RANGE,
    merge the two spans into one: the connector's letters become RANGE and
    its flanking whitespace becomes SEP, same as the punctuation-connector
    repair. Both spans are already guaranteed to carry a meaningful token
    (spans without one are dropped before this runs).
    """
    if len(spans_out) < 2:
        return spans_out
    sep_id = C.CLASS_TO_ID["SEP"]
    range_id = C.CLASS_TO_ID["RANGE"]

    merged: List[dict] = [spans_out[0]]
    for nxt in spans_out[1:]:
        prev = merged[-1]
        gap_start, gap_end = prev["end"], nxt["start"]
        gap = text[gap_start:gap_end]
        n = len(gap)
        i = 0
        while i < n and _char_type(gap[i]) == "space":
            i += 1
        lstart = i
        while i < n and _char_type(gap[i]) == "letter":
            i += 1
        lend = i
        while i < n and _char_type(gap[i]) == "space":
            i += 1

        if lstart == lend or i != n:
            merged.append(nxt)
            continue

        rs, re_ = gap_start + lstart, gap_start + lend
        conn_id = _majority_class_any([cls_ids[k] for k in range(rs, re_)])
        conn_cls = C.CLASSES[conn_id]
        if conn_cls != "RANGE":
            merged.append(nxt)
            continue

        conn_tokens = []
        if lstart > 0:
            conn_tokens.append((sep_id, gap[0:lstart]))
        conn_tokens.append((range_id, gap[lstart:lend]))
        if lend < n:
            conn_tokens.append((sep_id, gap[lend:n]))

        if prev["confidence"] is None or nxt["confidence"] is None:
            confidence = None
        else:
            confidence = (prev["confidence"] + nxt["confidence"]) / 2

        merged[-1] = {
            "start": prev["start"],
            "end": nxt["end"],
            "text": text[prev["start"]:nxt["end"]],
            "tokens": prev["tokens"] + conn_tokens + nxt["tokens"],
            "confidence": confidence,
        }
    return merged


def _last_meaningful_cls(tokens) -> Optional[str]:
    for cid, _txt in reversed(tokens):
        name = C.CLASSES[cid]
        if _is_meaningful(name):
            return name
    return None


def _first_meaningful_cls(tokens) -> Optional[str]:
    for cid, _txt in tokens:
        name = C.CLASSES[cid]
        if _is_meaningful(name):
            return name
    return None


def _last_meaningful(tokens):
    for cid, txt in reversed(tokens):
        name = C.CLASSES[cid]
        if _is_meaningful(name):
            return name, txt
    return None, None


def _first_meaningful(tokens):
    for cid, txt in tokens:
        name = C.CLASSES[cid]
        if _is_meaningful(name):
            return name, txt
    return None, None


_YEAR_LO, _YEAR_HI = 1900, 2099


def _looks_like_year(text: str) -> bool:
    """R18: a 4-digit run in 1900..2099 -- a year, not a coefficient.

    Real text puts a title and a year side by side ("दस हज़ार 1987 में बनी"),
    and the incomplete-amount merge below would otherwise read that as one
    11,987. Mirrors src/decode.ts looksLikeYear."""
    digits = "".join(_DIGIT_TO_ASCII.get(c, c) for c in text)
    if len(digits) != 4 or not digits.isdigit():
        return False
    return _YEAR_LO <= int(digits) <= _YEAR_HI


_DIGIT_TO_ASCII = {}
for _b in (0x0966, 0x0AE6):
    for _i in range(10):
        _DIGIT_TO_ASCII[chr(_b + _i)] = str(_i)
del _b, _i


def _same_value_restatement(left_cls, left_txt, right_cls, right_txt) -> bool:
    """R18: the right span merely RESTATES the left one in words --
    "5 panch hajar" (digits 5, then the same 5 spelled out). Merging would
    make it 5,005. True when the left span ends in a DIGITS token and the
    right span opens with a CARD_* of exactly that value."""
    if left_cls != "DIGITS" or right_cls is None or not right_cls.startswith("CARD_"):
        return False
    digits = "".join(_DIGIT_TO_ASCII.get(c, c) for c in left_txt)
    if not digits.isdigit():
        return False
    try:
        return int(digits) == C.card_value(right_cls)
    except Exception:
        return False


def _merge_incomplete_amount_spans(text: str, spans_out: List[dict]) -> List[dict]:
    """R4c: merge an INCOMPLETE amount with the span right after it.

    The BIO head sometimes cuts a single amount in two at a word boundary
    even with no connector between them: "ચોંસઠ લાખમાં" decodes as
    "ચોંસઠ" (64) + "લાખમાં" (100000) instead of 6400000, "पाव कोटीचा" as
    "पाव" + "कोटीचा", "दस बीस हज़ार" as "दस" + "बीस हज़ार".

    Span A is INCOMPLETE when its last meaningful token is a bare number
    (CARD_*/PFX_*/DIGITS) -- it has no UNIT_* token after it, so it cannot
    have closed an amount. Span B can CONTINUE it when its first
    meaningful token is a UNIT_* (the unit A is missing) or a CARD_* (a
    further number word of the same amount). When the two spans are
    separated by exactly one whitespace character -- so nothing, not even
    punctuation, stands between them -- they are one amount and get merged
    with a SEP token in between. Repeated until stable, so a chain
    ("दस" + "बीस" + "हज़ार") collapses into one span.

    Two COMPLETE amounts are untouched: "5 lakh 3 crore" has A ending in
    UNIT_LAKH, so A is not incomplete and the two stay separate spans (the
    arithmetic core, not the decoder, decides what a multi-term span
    means). After merging, core's R7/R8 juxtaposition rules apply to the
    merged token list as usual, so "दस बीस हज़ार" becomes a range.
    """
    if len(spans_out) < 2:
        return spans_out
    sep_id = C.CLASS_TO_ID["SEP"]
    spans = list(spans_out)
    while True:
        merged: List[dict] = [spans[0]]
        changed = False
        for nxt in spans[1:]:
            prev = merged[-1]
            gap = text[prev["end"]:nxt["start"]]
            last, last_txt = _last_meaningful(prev["tokens"])
            first, first_txt = _first_meaningful(nxt["tokens"])
            # R18: two real-text shapes that are NOT one interrupted amount
            # -- a following YEAR ("दस हज़ार 1987 में बनी") and a digits
            # amount restated in words ("5 panch hajar").
            blocked = (
                (first == "DIGITS" and _looks_like_year(first_txt))
                or _same_value_restatement(last, last_txt, first, first_txt)
            )
            if (
                not blocked
                and len(gap) == 1
                and gap.isspace()
                and last is not None
                and (last.startswith(("CARD_", "PFX_")) or last == "DIGITS")
                and first is not None
                and first.startswith(("UNIT_", "CARD_"))
            ):
                if prev["confidence"] is None or nxt["confidence"] is None:
                    confidence = None
                else:
                    confidence = (prev["confidence"] + nxt["confidence"]) / 2
                merged[-1] = {
                    "start": prev["start"],
                    "end": nxt["end"],
                    "text": text[prev["start"]:nxt["end"]],
                    "tokens": prev["tokens"] + [(sep_id, gap)] + nxt["tokens"],
                    "confidence": confidence,
                }
                changed = True
            else:
                merged.append(nxt)
        spans = merged
        if not changed:
            return spans


def _retrim(text: str, span: dict, toks) -> Optional[dict]:
    """Rebuild a span dict around a new token list, re-trimming the glue
    tokens at its edges and re-deriving start/end. None if nothing
    meaningful is left."""
    lo, hi = 0, len(toks)
    while lo < hi and C.CLASSES[toks[lo][0]] in _TRIM_CLASSES_FINAL:
        lo += 1
    while hi > lo and C.CLASSES[toks[hi - 1][0]] in _TRIM_CLASSES_FINAL:
        hi -= 1
    kept = toks[lo:hi]
    if not kept or not any(_is_meaningful(C.CLASSES[cid]) for cid, _ in kept):
        return None
    joined = "".join(t for _c, t in kept)
    start = text.index(joined, span["start"], span["end"]) if joined else span["start"]
    return {
        "start": start,
        "end": start + len(joined),
        "text": joined,
        "tokens": kept,
        "confidence": span["confidence"],
    }


def _trim_boundary_artifacts(text: str, spans_out: List[dict]) -> List[dict]:
    """R18: two real-text shapes the BIO head glues into ONE span although
    the writer wrote two things (data_wild/REPORT.md §4 shape 8).

    (a) a trailing YEAR: "इनाम दस हज़ार 1987 में बनी" is a film title plus
        its release year, decoded as one span worth 11,987. A 4-digit
        1900..2099 DIGITS token at the END of a span whose amount is
        already CLOSED (the meaningful token before it is a UNIT_*) is a
        year, and is trimmed off -- leaving "दस हज़ार" = 10,000.

    (b) a leading digits RESTATEMENT: "agni 5 panch hajar kilometer" says
        five thousand twice, decoded as 5,005. A leading DIGITS token whose
        value equals the CARD_* value of the very next meaningful token is
        dropped -- leaving "panch hajar" = 5,000.

    Both are also refused by the incomplete-amount merge below, so a span
    the BIO head splits at the same boundary is not re-joined."""
    out = []
    for span in spans_out:
        toks = list(span["tokens"])
        changed = True
        while changed and toks:
            changed = False
            meaningful = [(i, C.CLASSES[c], t) for i, (c, t) in enumerate(toks)
                          if _is_meaningful(C.CLASSES[c])]
            if len(meaningful) >= 2:
                li, lcls, ltxt = meaningful[-1]
                _pi, pcls, _ptxt = meaningful[-2]
                if lcls == "DIGITS" and _looks_like_year(ltxt) and pcls.startswith("UNIT_"):
                    toks = toks[:li]
                    changed = True
                    continue
                fi, fcls, ftxt = meaningful[0]
                _ni, ncls, _ntxt = meaningful[1]
                if _same_value_restatement(fcls, ftxt, ncls, _ntxt):
                    toks = toks[fi + 1:]
                    changed = True
        rebuilt = _retrim(text, span, toks) if len(toks) != len(span["tokens"]) else span
        if rebuilt is not None:
            out.append(rebuilt)
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

    Before spans are found, the raw BIO sequence itself is repaired in three
    passes (see their docstrings): `bridge_bio` closes single-char O gaps
    inside a span, `extend_digit_spans` extends a span forward through a
    trailing digit run it was cut short of, and `extend_word_integrity_bio`
    extends (and merges) a span across its whole letter word when every
    char in that word's raw class prediction unanimously agrees.

    Returns list of dicts: {start, end, text, tokens: [(cls_id, substr), ...], confidence}
    """
    n = len(text)
    letter_runs = _letter_runs(text)

    bio_ids = bridge_bio(text, bio_ids)
    bio_ids = extend_digit_spans(text, bio_ids)
    bio_ids = extend_word_integrity_bio(text, bio_ids, cls_ids, letter_runs)

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
        work_cls = _repair_word_integrity(text, s, e, work_cls, letter_runs)
        work_cls = _repair_possessive(text, s, e, work_cls)

        # token runs with (cls_id, start, end) offsets
        run_start = s
        orig_toks = []
        for j in range(s + 1, e + 1):
            if j == e or work_cls[j] != work_cls[run_start]:
                orig_toks.append((work_cls[run_start], run_start, j))
                run_start = j

        # trim leading/trailing trim-classes (SEP/RANGE/DOT/COMMA, plus O --
        # R2/R5 above can leave a now-meaningless O token at an edge, and
        # re-deriving start/end to the min/max of the remaining non-O chars
        # means trimming it away here too)
        lo, hi = 0, len(orig_toks)
        while lo < hi and C.CLASSES[orig_toks[lo][0]] in _TRIM_CLASSES_FINAL:
            lo += 1
        while hi > lo and C.CLASSES[orig_toks[hi - 1][0]] in _TRIM_CLASSES_FINAL:
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
    out = _merge_range_connector_spans(text, cls_ids, out)
    out = _merge_incomplete_amount_spans(text, out)
    return _trim_boundary_artifacts(text, out)
