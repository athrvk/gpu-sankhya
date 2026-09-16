"""Plain-assert tests for sankhya.decode, runnable via pytest or directly."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sankhya import classes as C
from sankhya.decode import decode_spans, repair_classes, _repair_letters_run, bridge_bio, extend_digit_spans

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


def test_bridge_bio_closes_single_char_o_gap():
    # "five and a half" -- the space right after "five" is raw-tagged O
    # between two I's; bridge_bio should turn it into I.
    text = "five and a half"
    bio = [1, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    bridged = bridge_bio(text, bio)
    assert bridged[4] == 2
    # unaffected elsewhere
    assert bridged[0] == 1
    assert bridged[5:] == bio[5:]


def test_bridge_bio_does_not_bridge_at_span_edges():
    # a leading/trailing O should NOT be bridged (only a gap strictly
    # between a B/I and a following I is bridged).
    text = "xayz"
    bio = [0, 1, 2, 0]  # O at index 3 has no I after it -> stays O
    bridged = bridge_bio(text, bio)
    assert bridged == bio


def test_bridge_bio_currency_digit_gap():
    # "₹85000" -- the "5" right after the currency symbol/leading digit
    # raw-tagged O, flanked by I on both sides.
    text = "₹85000"
    #        0(sym) 1(8) 2(5) 3(0) 4(0) 5(0)
    bio = [1, 2, 0, 2, 2, 2]
    bridged = bridge_bio(text, bio)
    assert bridged == [1, 2, 2, 2, 2, 2]


def test_extend_digit_spans_extends_trailing_digit_run():
    # "15000/month" -- span currently ends at "1500" (index 3, a digit),
    # with the final "0" (index 4) raw-tagged O; extend_digit_spans should
    # pull it into the span since it's a digit immediately following a
    # digit the span already ends on.
    text = "15000/month"
    bio = [1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0]
    extended = extend_digit_spans(text, bio)
    assert extended[:5] == [1, 2, 2, 2, 2]
    # the "/" after the digit run is not a digit -> not extended further
    assert extended[5] == 0


def test_extend_digit_spans_stops_at_non_digit():
    # span ends on a digit but next char is not a digit -> no extension
    text = "20k logon"
    bio = [1, 2, 0] + [0] * 6
    extended = extend_digit_spans(text, bio)
    assert extended == bio


def test_extend_digit_spans_does_not_cross_into_new_b():
    # if the following digit char was already tagged B (start of a NEW
    # span), extend_digit_spans must not steal it.
    text = "1000200"
    bio = [1, 2, 2, 2, 1, 2, 2]  # "1000" then a new span "200" starting with B
    extended = extend_digit_spans(text, bio)
    assert extended == bio


def test_decode_spans_end_to_end_bridge_and_extend():
    # combines both BIO repairs end-to-end via decode_spans: "15000/month"
    # with the trailing "0" mistagged O should decode as a single span
    # covering the full "15000".
    text = "15000/month"
    bio = [1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0]
    cls = [DIGITS, DIGITS, DIGITS, DIGITS, DIGITS, O, O, O, O, O, O]
    spans = decode_spans(text, bio, cls)
    assert len(spans) == 1
    assert spans[0]["start"] == 0 and spans[0]["end"] == 5
    assert spans[0]["text"] == "15000"


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


def test_word_integrity_partial_unit_token_retagged_o():
    # "10 km" -- the letters-run "km" is only PARTIALLY meaningful (the "m"
    # fell back to O rather than joining "k"'s class), so the whole run
    # (including the "k" that WAS tagged UNIT_) gets retagged O -- a
    # partial match without an explanation for the rest of the word is
    # untrustworthy.
    from sankhya.decode import _repair_word_integrity, _letter_runs
    text = "10 km"
    cls = [DIGITS, DIGITS, SEP, UNIT_HAZAAR, O]
    out = _repair_word_integrity(text, 0, len(text), cls, _letter_runs(text))
    assert cnames(out, 3, 5) == ["O", "O"]


def test_word_integrity_dedhlakh_joined_not_split():
    # "dedhlakh" -- a letters-run legitimately split into two back-to-back
    # meaningful sub-words (PFX_DEDH|UNIT_LAKH) must be left untouched, even
    # though neither token covers the WHOLE run by itself.
    from sankhya.decode import _repair_word_integrity, _letter_runs
    text = "dedhlakh"
    cls = [PFX_DEDH] * 4 + [UNIT_LAKH] * 4
    out = _repair_word_integrity(text, 0, len(text), cls, _letter_runs(text))
    assert cnames(out, 0, 8) == ["PFX_DEDH"] * 4 + ["UNIT_LAKH"] * 4


def test_word_integrity_whole_word_unit_kept():
    # a UNIT_ token that exactly covers its own letters-run (the whole
    # glued number+symbol word "k" in "20k") is left alone.
    from sankhya.decode import _repair_word_integrity, _letter_runs
    text = "20k"
    cls = [DIGITS, DIGITS, UNIT_HAZAAR]
    out = _repair_word_integrity(text, 0, len(text), cls, _letter_runs(text))
    assert cnames(out, 2, 3) == ["UNIT_HAZAAR"]


def test_word_integrity_symbol_unit_followed_by_letter_dropped():
    # a one-char UNIT_ token ("k") that covers its whole (artificially
    # one-char) letters-run is still invalid if the very next char is a
    # letter -- isolates the extra single-char symbol-unit guard from the
    # (more common) "doesn't cover the whole run" path above.
    from sankhya.decode import _repair_word_integrity
    text = "5kg"
    cls = [DIGITS, UNIT_HAZAAR, O]
    out = _repair_word_integrity(text, 0, len(text), cls, [(1, 2), (2, 3)])
    assert cnames(out, 1, 2) == ["O"]


def test_word_integrity_symbol_unit_at_word_end_kept():
    # "20k logon" -- "k" is a whole one-char run followed by a space, not a
    # letter, so it stays valid.
    from sankhya.decode import _repair_word_integrity, _letter_runs
    text = "20k logon"
    cls = [DIGITS, DIGITS, UNIT_HAZAAR, O, O, O, O, O, O]
    out = _repair_word_integrity(text, 0, len(text), cls, _letter_runs(text))
    assert cnames(out, 2, 3) == ["UNIT_HAZAAR"]


def test_range_connector_unit_then_digits_is_range():
    # "2 lakh/3 lakh" -- connector between a closed UNIT_LAKH amount and a
    # fresh DIGITS amount is a real range.
    text = "2 lakh/3 lakh"
    bio = [1] + [2] * (len(text) - 1)
    cls = (
        [DIGITS, SEP, UNIT_LAKH, UNIT_LAKH, UNIT_LAKH, UNIT_LAKH]
        + [O]  # "/"
        + [DIGITS, SEP, UNIT_LAKH, UNIT_LAKH, UNIT_LAKH, UNIT_LAKH]
    )
    spans = decode_spans(text, bio, cls)
    assert len(spans) == 1
    tok_classes = [C.CLASSES[cid] for cid, _ in spans[0]["tokens"]]
    assert tok_classes == [
        "DIGITS", "SEP", "UNIT_LAKH", "RANGE", "DIGITS", "SEP", "UNIT_LAKH",
    ]


def test_range_connector_prefix_then_unit_stays_sep():
    # "dedh-lakh" is ONE compound number (PFX_DHAI directly glued to
    # UNIT_LAKH); a unit can never START a fresh amount, so the hyphen must
    # stay SEP, not become RANGE.
    text = "dedh-lakh"
    raw = [PFX_DHAI] * 4 + [O] + [UNIT_LAKH] * 4
    repaired = repair_classes(text, 0, len(text), raw)
    assert cnames(repaired, 4, 5) == ["SEP"]


def test_range_connector_deva_card_to_card_is_range():
    # "दो-तीन" -- Devanagari CARD_ words either side of a bare hyphen.
    text = "दो-तीन"
    CARD_2 = C.CLASS_TO_ID["CARD_2"]
    CARD_3 = C.CLASS_TO_ID["CARD_3"]
    raw = [CARD_2, CARD_2] + [O] + [CARD_3, CARD_3, CARD_3]
    repaired = repair_classes(text, 0, len(text), raw)
    assert cnames(repaired, 2, 3) == ["RANGE"]


def test_possessive_trim_apostrophe_s():
    # "2 lakh's ka scam" -- possessive "'s" is not part of the amount.
    text = "2 lakh's ka scam"
    bio = [1] + [2] * (len("2 lakh's") - 1) + [0] * (len(text) - len("2 lakh's"))
    cls = [DIGITS, SEP, UNIT_LAKH, UNIT_LAKH, UNIT_LAKH, UNIT_LAKH, O, O] + [O] * (len(text) - 8)
    spans = decode_spans(text, bio, cls)
    assert len(spans) == 1
    assert spans[0]["text"] == "2 lakh"


def test_possessive_trim_no_space():
    text = "2lakh's"
    bio = [1] + [2] * (len(text) - 1)
    cls = [DIGITS, UNIT_LAKH, UNIT_LAKH, UNIT_LAKH, UNIT_LAKH, O, O]
    spans = decode_spans(text, bio, cls)
    assert len(spans) == 1
    assert spans[0]["text"] == "2lakh"


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
