"""Plain-assert tests for sankhya.decode, runnable via pytest or directly."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sankhya import classes as C
from sankhya.decode import decode_spans, repair_classes, _repair_letters_run

O = C.CLASS_TO_ID["O"]
SEP = C.CLASS_TO_ID["SEP"]
RANGE = C.CLASS_TO_ID["RANGE"]
DIGITS = C.CLASS_TO_ID["DIGITS"]
DOT = C.CLASS_TO_ID["DOT"]
COMMA = C.CLASS_TO_ID["COMMA"]
UNIT_CRORE = C.CLASS_TO_ID["UNIT_CRORE"]
UNIT_LAKH = C.CLASS_TO_ID["UNIT_LAKH"]
UNIT_HAZAAR = C.CLASS_TO_ID["UNIT_HAZAAR"]
CARD_10 = C.CLASS_TO_ID["CARD_10"]
PFX_DHAI = C.CLASS_TO_ID["PFX_DHAI"]
PFX_DEDH = C.CLASS_TO_ID["PFX_DEDH"]
PFX_SAADHE = C.CLASS_TO_ID["PFX_SAADHE"]
PFX_AADHA = C.CLASS_TO_ID["PFX_AADHA"]
CARD_5 = C.CLASS_TO_ID["CARD_5"]
CARD_8 = C.CLASS_TO_ID["CARD_8"]


def cnames(cls_ids, start, end):
    return [C.CLASSES[c] for c in cls_ids[start:end]]


def test_repair_dot_between_digits_mistagged_o():
    # "1.5cr" -> per-char raw pred: DIGITS O DIGITS UNIT_CRORE UNIT_CRORE
    text = "1.5cr"
    raw = [DIGITS, O, DIGITS, UNIT_CRORE, UNIT_CRORE]
    repaired = repair_classes(text, 0, len(text), raw)
    assert cnames(repaired, 0, len(text)) == ["DIGITS", "DOT", "DIGITS", "UNIT_CRORE", "UNIT_CRORE"]


def test_repair_hyphen_between_digits_mistagged_o():
    # "2-3 lakh" -> raw: DIGITS O DIGITS SEP UNIT_LAKH*4
    text = "2-3 lakh"
    raw = [DIGITS, O, DIGITS, SEP, UNIT_LAKH, UNIT_LAKH, UNIT_LAKH, UNIT_LAKH]
    repaired = repair_classes(text, 0, len(text), raw)
    assert cnames(repaired, 0, len(text)) == [
        "DIGITS", "RANGE", "DIGITS", "SEP", "UNIT_LAKH", "UNIT_LAKH", "UNIT_LAKH", "UNIT_LAKH",
    ]


def test_repair_letters_run_majority_vote_fallback():
    # "das hazaar crore" -> raw per-char has "d" tagged PFX_DHAI (sub-run len 1),
    # "as" tagged CARD_10 (sub-run len 2). Neither sub-run reaches length 3, and
    # neither has a qualifying (>=3) neighbour sub-run within the "das" run, so
    # this falls back to a whole-run majority vote: CARD_10 has 2 votes vs
    # PFX_DHAI's 1, so CARD_10 wins for the whole "das" run.
    text = "das hazaar crore"
    #        d          a          s           (space)     h..r (hazaar)      (space)    c..e (crore)
    raw = [PFX_DHAI, CARD_10, CARD_10] + [SEP] + [UNIT_HAZAAR] * 6 + [SEP] + [UNIT_CRORE] * 5
    assert len(raw) == len(text)
    repaired = repair_classes(text, 0, len(text), raw)
    assert cnames(repaired, 0, 3) == ["CARD_10", "CARD_10", "CARD_10"]
    assert cnames(repaired, 3, 4) == ["SEP"]
    assert cnames(repaired, 4, 10) == ["UNIT_HAZAAR"] * 6
    assert cnames(repaired, 10, 11) == ["SEP"]
    assert cnames(repaired, 11, 16) == ["UNIT_CRORE"] * 5


def test_repair_letters_run_preserves_long_subruns_dedhlakh():
    # "dedhlakh" (joined, no space): raw per-char sub-runs "dedh" (len 4,
    # PFX_DEDH) and "lakh" (len 4, UNIT_LAKH) are both >= 3, so sub-run
    # smoothing must keep them separate rather than collapsing the whole
    # 8-char letters run into a single (wrong) class via a tied majority vote.
    text = "dedhlakh"
    raw = [PFX_DEDH] * 4 + [UNIT_LAKH] * 4
    result = _repair_letters_run(raw)
    assert [C.CLASSES[c] for c in result] == ["PFX_DEDH"] * 4 + ["UNIT_LAKH"] * 4


def test_repair_letters_run_short_subrun_snaps_to_longer_neighbour():
    # a short (length-2) sub-run between two long (>=3) sub-runs of DIFFERENT
    # classes should snap to the longer neighbour; equal-length neighbours ->
    # prefer the left one.
    # neighbours of equal length (4 vs 4): prefer left
    raw_equal = [PFX_DEDH] * 4 + [CARD_10] * 2 + [UNIT_LAKH] * 4
    result_equal = _repair_letters_run(raw_equal)
    assert [C.CLASSES[c] for c in result_equal] == ["PFX_DEDH"] * 6 + ["UNIT_LAKH"] * 4

    # neighbours of unequal length: prefer the longer (right) one
    raw_unequal = [PFX_DEDH] * 3 + [CARD_10] * 2 + [UNIT_LAKH] * 5
    result_unequal = _repair_letters_run(raw_unequal)
    assert [C.CLASSES[c] for c in result_unequal] == ["PFX_DEDH"] * 3 + ["UNIT_LAKH"] * 7


def test_repair_letters_run_all_o_stays_o():
    text = "kya"
    raw = [O, O, O]
    repaired = repair_classes(text, 0, len(text), raw)
    assert cnames(repaired, 0, 3) == ["O", "O", "O"]


def test_repair_single_letter_word_majority_is_itself():
    # "five and a half" -- the standalone "a" is its own letters-run; with
    # only one char to vote on, the repair keeps that char's own prediction
    # (majority-of-one). This documents current, spec-faithful behavior.
    text = "a"
    raw = [CARD_8]
    repaired = repair_classes(text, 0, len(text), raw)
    assert cnames(repaired, 0, 1) == ["CARD_8"]


def test_repair_comma_between_digits():
    text = "1,25,000"
    raw = [DIGITS, O, DIGITS, DIGITS, O, DIGITS, DIGITS, DIGITS]
    repaired = repair_classes(text, 0, len(text), raw)
    assert cnames(repaired, 0, len(text)) == [
        "DIGITS", "COMMA", "DIGITS", "DIGITS", "COMMA", "DIGITS", "DIGITS", "DIGITS",
    ]


def test_repair_dot_not_between_digits_becomes_sep():
    # a "." that is NOT flanked by digit runs on both sides is just punctuation -> SEP
    text = "lakh.crore"
    raw = [UNIT_LAKH] * 4 + [O] + [UNIT_CRORE] * 5
    repaired = repair_classes(text, 0, len(text), raw)
    assert cnames(repaired, 4, 5) == ["SEP"]


def test_repair_hyphen_between_letters_is_sep():
    text = "dedh-lakh"
    raw = [PFX_DHAI] * 4 + [O] + [UNIT_LAKH] * 4
    repaired = repair_classes(text, 0, len(text), raw)
    assert cnames(repaired, 4, 5) == ["SEP"]


def test_repair_whitespace_keeps_range_if_any_char_was_range():
    text = "2 3"
    raw = [DIGITS, RANGE, DIGITS]
    repaired = repair_classes(text, 0, len(text), raw)
    assert cnames(repaired, 1, 2) == ["RANGE"]


def test_repair_whitespace_defaults_to_sep():
    text = "do lakh"
    raw = [CARD_10, CARD_10, O, UNIT_LAKH, UNIT_LAKH, UNIT_LAKH, UNIT_LAKH]
    repaired = repair_classes(text, 0, len(text), raw)
    assert cnames(repaired, 2, 3) == ["SEP"]


def test_decode_spans_applies_repair_before_filter():
    # end-to-end: strict BIO decode + repair should produce a single clean
    # DIGITS/DOT/DIGITS/UNIT_CRORE token run for "1.5cr" even though the "."
    # was raw-mistagged as O.
    text = "1.5cr"
    bio = [1, 2, 2, 2, 2]
    cls = [DIGITS, O, DIGITS, UNIT_CRORE, UNIT_CRORE]
    spans = decode_spans(text, bio, cls)
    assert len(spans) == 1
    sp = spans[0]
    assert sp["start"] == 0 and sp["end"] == 5
    tok_classes = [C.CLASSES[cid] for cid, _ in sp["tokens"]]
    assert tok_classes == ["DIGITS", "DOT", "DIGITS", "UNIT_CRORE"]


def test_decode_spans_repairs_range_hyphen():
    text = "2-3 lakh"
    bio = [1, 2, 2, 2, 2, 2, 2, 2]
    cls = [DIGITS, O, DIGITS, SEP, UNIT_LAKH, UNIT_LAKH, UNIT_LAKH, UNIT_LAKH]
    spans = decode_spans(text, bio, cls)
    assert len(spans) == 1
    tok_classes = [C.CLASSES[cid] for cid, _ in spans[0]["tokens"]]
    assert tok_classes == ["DIGITS", "RANGE", "DIGITS", "SEP", "UNIT_LAKH"]


if __name__ == "__main__":
    import types
    mod = types.ModuleType("m")
    failures = 0
    total = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            total += 1
            try:
                fn()
            except AssertionError as e:
                failures += 1
                print(f"FAIL {name}: {e}")
    print(f"{total - failures}/{total} passed")
    if failures:
        sys.exit(1)
