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


UNIT_KHARAB = C.CLASS_TO_ID["UNIT_KHARAB"]


def _apply_lone_ambiguous_unit_gate(spans):
    """Mirrors eval_gold._apply_bare_digits_gate's R4 half, for decode-level
    tests: drop a lone-ambiguous-unit span with no detected currency."""
    from sankhya import core
    kept = []
    for sp in spans:
        toks = [(C.CLASSES[cid], sub) for cid, sub in sp["tokens"]]
        if core.should_drop_lone_ambiguous_unit(toks):
            continue
        kept.append(sp)
    return kept


def test_decode_lone_kharab_dropped_by_gate():
    # "washing machine kharab ho gaya" -- bare "kharab" with no preceding
    # number should decode as a span but get dropped by the R4 gate.
    text = "washing machine kharab ho gaya"
    start, end = text.index("kharab"), text.index("kharab") + len("kharab")
    bio = [O] * len(text)
    cls = [O] * len(text)
    bio[start] = 1
    for i in range(start + 1, end):
        bio[i] = 2
    for i in range(start, end):
        cls[i] = UNIT_KHARAB
    spans = decode_spans(text, bio, cls)
    assert len(spans) == 1
    kept = _apply_lone_ambiguous_unit_gate(spans)
    assert kept == []


def test_decode_das_kharab_kept_by_gate():
    # "das kharab" -- a preceding number ("das") means the unit is a real
    # amount, so the R4 gate must NOT drop it.
    text = "das kharab"
    bio = [1, 2, 2] + [2] + [2, 2, 2, 2, 2, 2]
    cls = [CARD_10] * 3 + [SEP] + [UNIT_KHARAB] * 6
    spans = decode_spans(text, bio, cls)
    assert len(spans) == 1
    kept = _apply_lone_ambiguous_unit_gate(spans)
    assert len(kept) == 1


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
    # prefer the left one. (The middle sub-run is a UNIT_ here, so the
    # three sub-runs are PFX/UNIT/UNIT -- NOT the PFX? CARD? UNIT? shape
    # `_bound_keep_indices` keeps as a fused word; see the fused-word
    # tests below.)
    # neighbours of equal length (4 vs 4): prefer left
    raw_equal = [PFX_DEDH] * 4 + [UNIT_HAZAAR] * 2 + [UNIT_LAKH] * 4
    result_equal = _repair_letters_run(raw_equal)
    assert [C.CLASSES[c] for c in result_equal] == ["PFX_DEDH"] * 6 + ["UNIT_LAKH"] * 4

    # neighbours of unequal length: prefer the longer (right) one
    raw_unequal = [PFX_DEDH] * 3 + [UNIT_HAZAAR] * 2 + [UNIT_LAKH] * 5
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


def test_extend_word_integrity_bio_merges_unanimous_run():
    # synthetic 6-char word, all raw chars CARD_79, BIO B I I O I O (two
    # low-confidence O's split what should be one span mid-word) -- since
    # every char in the run raw-predicts the SAME word class, the whole
    # run's BIO is extended/merged into a single span.
    from sankhya.decode import extend_word_integrity_bio, _letter_runs
    text = "unnasi"
    CARD_79 = C.CLASS_TO_ID["CARD_79"]
    bio = [1, 2, 2, 0, 2, 0]
    cls = [CARD_79] * 6
    out = extend_word_integrity_bio(text, bio, cls, _letter_runs(text))
    assert out == [1, 2, 2, 2, 2, 2]

    spans = decode_spans(text, bio, cls)
    assert len(spans) == 1
    assert spans[0]["start"] == 0 and spans[0]["end"] == 6
    assert spans[0]["text"] == "unnasi"
    tok_classes = [C.CLASSES[cid] for cid, _ in spans[0]["tokens"]]
    assert tok_classes == ["CARD_79"]


def test_extend_word_integrity_bio_disagreeing_classes_not_extended():
    # counter-case: same shape (6-char run, partial BIO coverage) but the
    # run's RAW classes disagree -- first 3 chars CARD_79, rest O -- so no
    # unanimous word class exists and the run is left alone; the existing
    # R2 drop behaviour still applies (span drops entirely).
    from sankhya.decode import extend_word_integrity_bio, _letter_runs
    text = "unnasi"
    CARD_79 = C.CLASS_TO_ID["CARD_79"]
    bio = [1, 2, 2, 0, 2, 0]
    cls = [CARD_79, CARD_79, CARD_79, O, O, O]
    out = extend_word_integrity_bio(text, bio, cls, _letter_runs(text))
    assert out == bio  # unchanged

    spans = decode_spans(text, bio, cls)
    assert spans == []


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


def test_word_connector_range_deva_merges_spans():
    # "दो लाख से तीन लाख" -- the class head tags "से" RANGE, but the BIO
    # head splits it into two spans (its own B mid-word). The two spans
    # must be merged into one RANGE span, value 200000, range
    # (200000, 300000), instead of staying two separate amounts.
    from sankhya import core

    CARD_2 = C.CLASS_TO_ID["CARD_2"]
    CARD_3 = C.CLASS_TO_ID["CARD_3"]
    text = "दो लाख से तीन लाख"
    bio = [1, 2, 2, 2, 2, 2, 0, 0, 0, 0, 1, 2, 2, 2, 2, 2, 2]
    cls = [
        CARD_2, CARD_2, O, UNIT_LAKH, UNIT_LAKH, UNIT_LAKH,
        O, RANGE, RANGE, O,
        CARD_3, CARD_3, CARD_3, O, UNIT_LAKH, UNIT_LAKH, UNIT_LAKH,
    ]
    spans = decode_spans(text, bio, cls)
    assert len(spans) == 1
    assert spans[0]["text"] == text
    tok_classes = [C.CLASSES[cid] for cid, _ in spans[0]["tokens"]]
    assert tok_classes == [
        "CARD_2", "SEP", "UNIT_LAKH", "SEP", "RANGE", "SEP", "CARD_3", "SEP", "UNIT_LAKH",
    ]
    result = core.evaluate([(C.CLASSES[cid], sub) for cid, sub in spans[0]["tokens"]])
    assert result.value == 200000
    assert result.range == (200000, 300000)


def test_word_connector_range_latin_merges_spans():
    # "5 hazaar se 8 hazaar tak" -- Latin analogue of the Devanagari case
    # above; trailing "tak" is not part of either span or the connector run.
    from sankhya import core

    UNIT_HAZAAR = C.CLASS_TO_ID["UNIT_HAZAAR"]
    text = "5 hazaar se 8 hazaar tak"
    bio = (
        [1, 2, 2, 2, 2, 2, 2, 2]
        + [0, 0, 0, 0]
        + [1, 2, 2, 2, 2, 2, 2, 2]
        + [0, 0, 0, 0]
    )
    cls = (
        [DIGITS, SEP, UNIT_HAZAAR, UNIT_HAZAAR, UNIT_HAZAAR, UNIT_HAZAAR, UNIT_HAZAAR, UNIT_HAZAAR]
        + [O, RANGE, RANGE, O]
        + [DIGITS, SEP, UNIT_HAZAAR, UNIT_HAZAAR, UNIT_HAZAAR, UNIT_HAZAAR, UNIT_HAZAAR, UNIT_HAZAAR]
        + [O, O, O, O]
    )
    spans = decode_spans(text, bio, cls)
    assert len(spans) == 1
    assert spans[0]["text"] == "5 hazaar se 8 hazaar"
    result = core.evaluate([(C.CLASSES[cid], sub) for cid, sub in spans[0]["tokens"]])
    assert result.value == 5000
    assert result.range == (5000, 8000)


def test_word_connector_class_o_stays_two_spans():
    # "5 hazaar aur 8 hazaar" -- "aur" ("and") is class O, not RANGE, so the
    # two amounts must NOT be merged into one span.
    UNIT_HAZAAR = C.CLASS_TO_ID["UNIT_HAZAAR"]
    text = "5 hazaar aur 8 hazaar"
    bio = (
        [1, 2, 2, 2, 2, 2, 2, 2]
        + [0, 0, 0, 0, 0]
        + [1, 2, 2, 2, 2, 2, 2, 2]
    )
    cls = (
        [DIGITS, SEP, UNIT_HAZAAR, UNIT_HAZAAR, UNIT_HAZAAR, UNIT_HAZAAR, UNIT_HAZAAR, UNIT_HAZAAR]
        + [O, O, O, O, O]
        + [DIGITS, SEP, UNIT_HAZAAR, UNIT_HAZAAR, UNIT_HAZAAR, UNIT_HAZAAR, UNIT_HAZAAR, UNIT_HAZAAR]
    )
    spans = decode_spans(text, bio, cls)
    assert len(spans) == 2
    assert spans[0]["text"] == "5 hazaar"
    assert spans[1]["text"] == "8 hazaar"


CARD_4 = C.CLASS_TO_ID["CARD_4"]
PFX_PAUNE = C.CLASS_TO_ID["PFX_PAUNE"]


def test_repair_letters_run_keeps_fused_prefix_cardinal_paunechaar():
    # "पावणेचार" (paune + chaar fused, no space): raw per-char sub-runs
    # "पावणे" (5 chars, PFX_PAUNE) and "चार" (3 chars, CARD_4) are both >= 2
    # and unanimous. Without the two-sub-run bound-tail exception, sub-run
    # smoothing (both sub-runs already >= 3 here, so this exercises the
    # >=2-but-<3 case too) or a majority vote would collapse the whole word
    # to PFX_PAUNE, losing the fused cardinal entirely (750 instead of
    # 3.75 * 1000).
    raw = [PFX_PAUNE] * 5 + [CARD_4] * 3
    result = _repair_letters_run(raw)
    assert [C.CLASSES[c] for c in result] == ["PFX_PAUNE"] * 5 + ["CARD_4"] * 3


def test_repair_letters_run_keeps_fused_prefix_cardinal_short_subruns():
    # Same shape but with short (length-2) sub-runs on both sides -- neither
    # reaches the normal length->=3 "keep as-is" threshold, so this only
    # survives via the bound-tail exception (unanimous, length >= 2).
    raw = [PFX_PAUNE] * 2 + [CARD_4] * 2
    result = _repair_letters_run(raw)
    assert [C.CLASSES[c] for c in result] == ["PFX_PAUNE"] * 2 + ["CARD_4"] * 2


def test_repair_letters_run_still_smooths_non_prefix_cardinal_short_subrun():
    # A short CARD_* tail NOT preceded by a PFX_* head must still be
    # smoothed away as before -- the exception is narrowly scoped to
    # PFX_* -> CARD_*/UNIT_* and CARD_*/PFX_* -> UNIT_*.
    raw = [UNIT_HAZAAR] * 4 + [CARD_10] * 2
    result = _repair_letters_run(raw)
    assert [C.CLASSES[c] for c in result] == ["UNIT_HAZAAR"] * 6


def test_decode_paunechaar_hazaar_evaluates_3750():
    from sankhya import core
    text = "पावणेचार हजार भरले"
    # "पावणेचार" = PFX_PAUNE(5) + CARD_4(3), "हजार" = UNIT_HAZAAR(4), "भरले" = O(4)
    raw_cls = (
        [PFX_PAUNE] * 5 + [CARD_4] * 3
        + [SEP]
        + [UNIT_HAZAAR] * 4
        + [SEP]
        + [O] * 4
    )
    assert len(raw_cls) == len(text)
    bio = [1] + [2] * 7 + [0] + [2] * 4 + [0] * 5
    spans = decode_spans(text, bio, raw_cls)
    assert len(spans) == 1
    toks = [(C.CLASSES[cid], sub) for cid, sub in spans[0]["tokens"]]
    result = core.evaluate(toks)
    assert result.value == 3750


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


# ---------------------------------------------------------------------------
# R4c: merging an INCOMPLETE amount with the span that follows it
# ---------------------------------------------------------------------------

def _tok_names(span):
    return [(C.CLASSES[cid], sub) for cid, sub in span["tokens"]]


def test_merge_incomplete_amount_with_following_unit_span():
    # "ચોંસઠ લાખમાં": the BIO head cuts the amount in two at the space, so
    # "ચોંસઠ" (CARD_64) and "લાખમાં" (UNIT_LAKH) decode as separate spans.
    # The first has no UNIT after it (an INCOMPLETE amount) and the second
    # starts with the unit it is missing, so they are one amount.
    CARD_64 = C.CLASS_TO_ID["CARD_64"]
    text = "ચોંસઠ લાખમાં"
    cls = [CARD_64] * 5 + [SEP] + [UNIT_LAKH] * 6
    bio = [1, 2, 2, 2, 2, 0, 1, 2, 2, 2, 2, 2]
    spans = decode_spans(text, bio, cls)
    assert len(spans) == 1
    assert (spans[0]["start"], spans[0]["end"]) == (0, 12)
    assert _tok_names(spans[0]) == [
        ("CARD_64", "ચોંસઠ"), ("SEP", " "), ("UNIT_LAKH", "લાખમાં"),
    ]


def test_merge_incomplete_amount_with_following_cardinal_span():
    # "दस बीस हज़ार" -> "दस" + "बीस हज़ार"; A is a bare CARD, B starts with a
    # CARD, so they merge and core's R8 juxtaposition range then applies.
    from sankhya import core
    CARD_20 = C.CLASS_TO_ID["CARD_20"]
    text = "दस बीस हज़ार"
    cls = [CARD_10] * 2 + [SEP] + [CARD_20] * 3 + [SEP] + [UNIT_HAZAAR] * 5
    bio = [1, 2, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2]
    spans = decode_spans(text, bio, cls)
    assert len(spans) == 1
    assert (spans[0]["start"], spans[0]["end"]) == (0, len(text))
    res = core.evaluate([(C.CLASSES[cid], sub) for cid, sub in spans[0]["tokens"]])
    assert res.range == (10000, 20000)


def test_merge_incomplete_amount_leaves_two_complete_amounts_alone():
    # "5 lakh 3 crore": span A ends with UNIT_LAKH, so it is a COMPLETE
    # amount and must not swallow the next span.
    text = "5 lakh 3 crore"
    cls = [DIGITS] + [SEP] + [UNIT_LAKH] * 4 + [SEP] + [DIGITS] + [SEP] + [UNIT_CRORE] * 5
    bio = [1, 2, 2, 2, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2]
    spans = decode_spans(text, bio, cls)
    assert len(spans) == 2
    assert [s["text"] for s in spans] == ["5 lakh", "3 crore"]


def test_merge_incomplete_amount_does_not_cross_punctuation():
    # Anything but exactly one whitespace char between the two spans (here a
    # comma + space) keeps them separate.
    text = "das, hazaar"
    cls = [CARD_10] * 3 + [O] * 2 + [UNIT_HAZAAR] * 6
    bio = [1, 2, 2, 0, 0, 1, 2, 2, 2, 2, 2]
    spans = decode_spans(text, bio, cls)
    assert len(spans) == 2


def test_merge_incomplete_amount_does_not_cross_two_spaces():
    text = "das  hazaar"
    cls = [CARD_10] * 3 + [O] * 2 + [UNIT_HAZAAR] * 6
    bio = [1, 2, 2, 0, 0, 1, 2, 2, 2, 2, 2]
    spans = decode_spans(text, bio, cls)
    assert len(spans) == 2


def test_merge_incomplete_amount_chains_until_stable():
    # three spans: CARD + CARD + UNIT, each separated by one space
    from sankhya import core
    CARD_20 = C.CLASS_TO_ID["CARD_20"]
    text = "das bees hazaar"
    cls = [CARD_10] * 3 + [SEP] + [CARD_20] * 4 + [SEP] + [UNIT_HAZAAR] * 6
    bio = [1, 2, 2, 0, 1, 2, 2, 2, 0, 1, 2, 2, 2, 2, 2]
    spans = decode_spans(text, bio, cls)
    assert len(spans) == 1
    assert spans[0]["text"] == text
    res = core.evaluate([(C.CLASSES[cid], sub) for cid, sub in spans[0]["tokens"]])
    assert res.range == (10000, 20000)


# ---------------------------------------------------------------------------
# R2c: bound forms and three-part fused words
# ---------------------------------------------------------------------------

def test_bound_form_one_char_subrun_baso():
    # "બસો" = CARD_2 "બ" (a BOUND form declared in the lexicon, valid only
    # before "સો") + UNIT_SAU "સો" = 200. The 1-char CARD sub-run survives
    # only because of that lexicon declaration.
    from sankhya import core
    CARD_2 = C.CLASS_TO_ID["CARD_2"]
    UNIT_SAU = C.CLASS_TO_ID["UNIT_SAU"]
    text = "બસો"
    spans = decode_spans(text, [1, 2, 2], [CARD_2, UNIT_SAU, UNIT_SAU])
    assert _tok_names(spans[0]) == [("CARD_2", "બ"), ("UNIT_SAU", "સો")]
    assert core.evaluate(_tok_names(spans[0])).value == 200


def test_undeclared_one_char_subrun_is_still_smoothed():
    # The same shape with an UNDECLARED 1-char head ("છ" is not in the
    # pack's bound_forms) is still smoothed away by the majority vote.
    CARD_6 = C.CLASS_TO_ID["CARD_6"]
    UNIT_SAU = C.CLASS_TO_ID["UNIT_SAU"]
    text = "છસો"
    spans = decode_spans(text, [1, 2, 2], [CARD_6, UNIT_SAU, UNIT_SAU])
    assert _tok_names(spans[0]) == [("UNIT_SAU", "છસો")]


def test_three_part_fused_word_saadhatransau():
    # "સાડાત્રણસો" = PFX_SAADHE "સાડા" + CARD_3 "ત્રણ" + UNIT_SAU "સો" = 350.
    from sankhya import core
    CARD_3 = C.CLASS_TO_ID["CARD_3"]
    UNIT_SAU = C.CLASS_TO_ID["UNIT_SAU"]
    text = "સાડાત્રણસો"
    cls = [PFX_SAADHE] * 4 + [CARD_3] * 4 + [UNIT_SAU] * 2
    spans = decode_spans(text, [1] + [2] * (len(text) - 1), cls)
    assert _tok_names(spans[0]) == [
        ("PFX_SAADHE", "સાડા"), ("CARD_3", "ત્રણ"), ("UNIT_SAU", "સો"),
    ]
    assert core.evaluate(_tok_names(spans[0])).value == 350


def test_three_part_fused_word_rejects_out_of_order_kinds():
    # UNIT before CARD is not the PFX? CARD? UNIT? shape -> smoothed as before
    CARD_3 = C.CLASS_TO_ID["CARD_3"]
    UNIT_SAU = C.CLASS_TO_ID["UNIT_SAU"]
    raw = [UNIT_SAU] * 2 + [CARD_3] * 2
    result = _repair_letters_run(raw, "abcd")
    assert len(set(result)) == 1


def test_bound_forms_are_verified_in_context():
    from sankhya.verify import verify_tokens
    assert verify_tokens([("CARD_2", "બ"), ("UNIT_SAU", "સો")]) is True
    assert verify_tokens([("CARD_2", "બ")]) is False


# ---------------------------------------------------------------------------
# R9: lexicon-"O" gate
# ---------------------------------------------------------------------------

def test_lexicon_o_only_helper():
    from sankhya.verify import is_lexicon_o_only
    assert is_lexicon_o_only("karodon") is True
    assert is_lexicon_o_only("करोडो") is True
    assert is_lexicon_o_only("lakh") is False
    assert is_lexicon_o_only("zzzz-not-a-word") is False


def test_lexicon_o_gate_drops_indefinite_plural_span():
    from sankhya import eval_gold
    classes = C.CLASSES
    decoded = [{"start": 5, "end": 12, "tokens": [(UNIT_CRORE, "karodon")]}]
    assert eval_gold._apply_bare_digits_gate("usne karodon rupaye", decoded, classes) == []
    keep = [{"start": 0, "end": 8, "tokens": [(DIGITS, "5"), (SEP, " "), (UNIT_CRORE, "crore")]}]
    assert len(eval_gold._apply_bare_digits_gate("5 crore", keep, classes)) == 1
