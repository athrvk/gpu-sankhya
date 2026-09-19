"""Plain-assert tests for sankhya.core, runnable via pytest or directly."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sankhya import core
from sankhya.langs import base

pack = base.get_pack("hi_latn")


def ev(*toks):
    return core.evaluate(list(toks))


def test_paune_do_lakh():
    r = ev(("PFX_PAUNE", "paune"), ("SEP", " "), ("CARD_2", "do"), ("SEP", " "), ("UNIT_LAKH", "lakh"))
    assert r.value == 175000, r.value


def test_ek_crore_bees_lakh():
    r = ev(("CARD_1", "ek"), ("SEP", " "), ("UNIT_CRORE", "crore"), ("SEP", " "),
           ("CARD_20", "bees"), ("SEP", " "), ("UNIT_LAKH", "lakh"))
    assert r.value == 12000000, r.value


def test_dhai_sau():
    r = ev(("PFX_DHAI", "dhai"), ("SEP", " "), ("UNIT_SAU", "sau"))
    assert r.value == 250


def test_2_5L():
    r = ev(("DIGITS", "2"), ("DOT", "."), ("DIGITS", "5"), ("UNIT_LAKH", "L"))
    assert r.value == 250000


def test_12_lpa():
    r = ev(("DIGITS", "1"), ("DIGITS", "2"), ("SEP", " "), ("UNIT_LAKH", "LPA"))
    assert r.value == 1200000


def test_sadhe_3_lakh():
    r = ev(("DIGITS", "3"), ("SEP", " "), ("PFX_SAADHE", "sadhe"), ("SEP", " "), ("UNIT_LAKH", "lakh"))
    # order: prefix could come before/after; test with prefix after number too
    r2 = ev(("PFX_SAADHE", "sadhe"), ("SEP", " "), ("DIGITS", "3"), ("SEP", " "), ("UNIT_LAKH", "lakh"))
    assert r2.value == 350000, r2.value


def test_1_25_000():
    r = ev(("DIGITS", "1"), ("COMMA", ","), ("DIGITS", "2"), ("DIGITS", "5"), ("COMMA", ","),
           ("DIGITS", "0"), ("DIGITS", "0"), ("DIGITS", "0"))
    assert r.value == 125000, r.value


def test_sava_lakh():
    r = ev(("PFX_SAVA", "sava"), ("SEP", " "), ("UNIT_LAKH", "lakh"))
    assert r.value == 125000


def test_dedh_crore():
    r = ev(("PFX_DEDH", "dedh"), ("SEP", " "), ("UNIT_CRORE", "crore"))
    assert r.value == 15000000


def test_dhai_lakh():
    r = ev(("PFX_DHAI", "dhai"), ("SEP", " "), ("UNIT_LAKH", "lakh"))
    assert r.value == 250000


def test_20k():
    r = ev(("DIGITS", "2"), ("DIGITS", "0"), ("UNIT_HAZAAR", "k"))
    assert r.value == 20000


def test_dedh_do_lakh_range():
    r = ev(("PFX_DEDH", "dedh"), ("RANGE", " "), ("CARD_2", "do"), ("SEP", " "), ("UNIT_LAKH", "lakh"))
    assert r.range == (150000, 200000), r.range


def test_2_3_lakh_range():
    r = ev(("DIGITS", "2"), ("RANGE", "-"), ("DIGITS", "3"), ("SEP", " "), ("UNIT_LAKH", "lakh"))
    assert r.range == (200000, 300000), r.range
    assert r.unit == "lakh"


def test_two_and_a_half_lakh():
    r = ev(("CARD_2", "two"), ("SEP", " "), ("PFX_SAADHE", "and a half"), ("SEP", " "), ("UNIT_LAKH", "lakh"))
    assert r.value == 250000, r.value


def test_half_a_crore():
    r = ev(("PFX_AADHA", "half a"), ("SEP", " "), ("UNIT_CRORE", "crore"))
    assert r.value == 5000000


def test_pachas_rupaye_no_unit():
    r = ev(("CARD_50", "pachas"))
    assert r.value == 50
    assert r.unit is None


def test_rs_500():
    r = ev(("DIGITS", "5"), ("DIGITS", "0"), ("DIGITS", "0"))
    assert r.value == 500


def test_currency_detection():
    text = "Rs 500 only"
    cur = core.detect_currency(text, 3, 6, pack)
    assert cur == "INR", cur
    text2 = "pachas rupaye chahiye"
    cur2 = core.detect_currency(text2, 0, 6, pack)
    assert cur2 == "INR", cur2
    text3 = "do din baad"
    cur3 = core.detect_currency(text3, 0, 2, pack)
    assert cur3 is None


def test_das_hazaar_crore():
    r = ev(("CARD_10", "das"), ("SEP", " "), ("UNIT_HAZAAR", "hazaar"), ("SEP", " "), ("UNIT_CRORE", "crore"))
    assert r.value == 1e11, r.value
    assert r.unit == "crore"


def test_sau_crore():
    r = ev(("UNIT_SAU", "sau"), ("SEP", " "), ("UNIT_CRORE", "crore"))
    assert r.value == 1e9, r.value


def test_dedh_sau_crore():
    r = ev(("PFX_DEDH", "dedh"), ("SEP", " "), ("UNIT_SAU", "sau"), ("SEP", " "), ("UNIT_CRORE", "crore"))
    assert r.value == 1.5e9, r.value


def test_2_lakh_crore():
    r = ev(("DIGITS", "2"), ("SEP", " "), ("UNIT_LAKH", "lakh"), ("SEP", " "), ("UNIT_CRORE", "crore"))
    assert r.value == 2e12, r.value


def test_saadhe_teen_sau_crore():
    r = ev(("PFX_SAADHE", "saadhe"), ("SEP", " "), ("CARD_3", "teen"), ("SEP", " "),
           ("UNIT_SAU", "sau"), ("SEP", " "), ("UNIT_CRORE", "crore"))
    assert r.value == 3.5e9, r.value


def test_50_hazaar_crore():
    r = ev(("DIGITS", "5"), ("DIGITS", "0"), ("SEP", " "), ("UNIT_HAZAAR", "hazaar"), ("SEP", " "), ("UNIT_CRORE", "crore"))
    assert r.value == 5e11, r.value


def test_ek_lakh_bees_hazaar_crore():
    r = ev(("CARD_1", "ek"), ("SEP", " "), ("UNIT_LAKH", "lakh"), ("SEP", " "),
           ("CARD_20", "bees"), ("SEP", " "), ("UNIT_HAZAAR", "hazaar"), ("SEP", " "),
           ("UNIT_CRORE", "crore"))
    assert r.value == 1.2e12, r.value
    assert r.unit == "crore"


def test_collapse_repeated_prefix_half_a():
    # "half a lakh" decoded as PFX_AADHA SEP PFX_AADHA UNIT_LAKH (a decode
    # artifact where "half" and "a" both land on PFX_AADHA) must collapse to
    # a single PFX_AADHA before evaluation, giving 0.5 * lakh = 50000, not
    # 50000.5 (standalone 0.5 double-counted as an extra additive term).
    r = ev(("PFX_AADHA", "half"), ("SEP", " "), ("PFX_AADHA", "a"), ("SEP", " "), ("UNIT_LAKH", "lakh"))
    assert r.value == 50000, r.value


def test_collapse_repeated_prefix_three_n_half():
    # "three n half lakh" decoded as CARD_3 SEP PFX_SAADHE SEP PFX_SAADHE
    # UNIT_LAKH ("n" and "half" both land on PFX_SAADHE) must collapse the
    # duplicate PFX_SAADHE into one before evaluation: coef = 3 + 0.5 = 3.5,
    # value = 3.5 * lakh = 350000.
    r = ev(("CARD_3", "three"), ("SEP", " "), ("PFX_SAADHE", "n"), ("SEP", " "),
           ("PFX_SAADHE", "half"), ("SEP", " "), ("UNIT_LAKH", "lakh"))
    assert r.value == 350000, r.value


def test_collapse_repeated_prefix_does_not_merge_different_prefixes():
    # sanity: two DIFFERENT prefix classes in a row must NOT collapse --
    # only identical consecutive PFX_* classes do.
    r = ev(("PFX_SAVA", "sava"), ("SEP", " "), ("PFX_DEDH", "dedh"), ("SEP", " "), ("UNIT_LAKH", "lakh"))
    # sava (standalone 1.25) flushed as its own additive term (no unit) = 1.25,
    # then dedh (standalone 1.5) * lakh = 150000; total = 150001.25
    assert r.value == 150001.25, r.value


def test_descending_range_evaluated_as_additive_chain():
    # "ek lakh dus hazaar" mistagged RANGE on the space between: both sides
    # carry their own unit and the left (100000) > right (10000) -> additive
    # chain 110000, not a [10000, 100000] range.
    r = ev(("CARD_1", "ek"), ("SEP", " "), ("UNIT_LAKH", "lakh"), ("RANGE", " "),
           ("CARD_10", "dus"), ("SEP", " "), ("UNIT_HAZAAR", "hazaar"))
    assert r.value == 110000, r.value
    assert r.range is None
    assert r.unit == "lakh"


def test_ascending_range_stays_a_range():
    # "2-3 lakh" -- only the RIGHT side carries a unit, so this must stay a
    # genuine [200000, 300000] range.
    r = ev(("DIGITS", "2"), ("RANGE", "-"), ("DIGITS", "3"), ("SEP", " "), ("UNIT_LAKH", "lakh"))
    assert r.value == 200000, r.value
    assert r.range == (200000, 300000), r.range


def test_true_range_both_sides_units_ascending_stays_range():
    # "5 lakh-10 lakh" -- both sides carry units but left < right: a
    # genuine ascending range, not a descending chain.
    r = ev(("DIGITS", "5"), ("SEP", " "), ("UNIT_LAKH", "lakh"), ("RANGE", "-"),
           ("DIGITS", "10"), ("SEP", " "), ("UNIT_LAKH", "lakh"))
    assert r.value == 500000, r.value
    assert r.range == (500000, 1000000), r.range


def test_is_bare_digits_true_for_digits_only():
    assert core.is_bare_digits([("DIGITS", "1")]) is True
    assert core.is_bare_digits([("DIGITS", "1"), ("COMMA", ","), ("DIGITS", "00"), ("COMMA", ","), ("DIGITS", "000")]) is True


def test_is_bare_digits_false_with_unit_or_card_or_prefix():
    assert core.is_bare_digits([("DIGITS", "2"), ("SEP", " "), ("UNIT_LAKH", "lakh")]) is False
    assert core.is_bare_digits([("CARD_2", "do"), ("SEP", " "), ("UNIT_LAKH", "lakh")]) is False
    assert core.is_bare_digits([("PFX_SAVA", "sava"), ("SEP", " "), ("UNIT_LAKH", "lakh")]) is False


def test_is_bare_digits_false_when_no_digits_at_all():
    assert core.is_bare_digits([("SEP", " ")]) is False


def _run_all():
    fns = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"OK  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()


def test_should_drop_bare_digits_short_ungrouped_dropped():
    assert core.should_drop_bare_digits([("DIGITS", "1")]) is True
    assert core.should_drop_bare_digits([("DIGITS", "66")]) is True
    assert core.should_drop_bare_digits([("DIGITS", "4521")]) is True
    assert core.should_drop_bare_digits([("DIGITS", "2024")]) is True


def test_should_drop_bare_digits_phone_number_dropped():
    assert core.should_drop_bare_digits([("DIGITS", "9876543210")]) is True


def test_should_drop_bare_digits_mid_length_amounts_kept():
    assert core.should_drop_bare_digits([("DIGITS", "15000")]) is False
    assert core.should_drop_bare_digits([("DIGITS", "85000")]) is False
    assert core.should_drop_bare_digits([("DIGITS", "62000")]) is False


def test_should_drop_bare_digits_grouped_short_amount_kept():
    # "1,00,000" -- only 6 digits, but a grouping comma is a strong signal
    # of a real amount, so rule (a) (<=4 digits, no comma) does not apply.
    toks = [("DIGITS", "1"), ("COMMA", ","), ("DIGITS", "00"), ("COMMA", ","), ("DIGITS", "000")]
    assert core.should_drop_bare_digits(toks) is False


def test_should_drop_bare_digits_grouped_long_amount_kept():
    # "2,50,00,000" -- 8 digits with commas, below the 10-digit phone
    # threshold, kept.
    toks = [("DIGITS", "2"), ("COMMA", ","), ("DIGITS", "50"), ("COMMA", ","), ("DIGITS", "00"), ("COMMA", ","), ("DIGITS", "000")]
    assert core.should_drop_bare_digits(toks) is False


def test_should_drop_bare_digits_not_bare_digits_at_all():
    assert core.should_drop_bare_digits([("DIGITS", "2"), ("SEP", " "), ("UNIT_LAKH", "lakh")]) is False


def test_should_drop_lone_ambiguous_unit_kharab_alone_dropped():
    assert core.should_drop_lone_ambiguous_unit([("UNIT_KHARAB", "kharab")]) is True


def test_should_drop_lone_ambiguous_unit_million_alone_dropped():
    assert core.should_drop_lone_ambiguous_unit([("UNIT_MILLION", "mil")]) is True
    assert core.should_drop_lone_ambiguous_unit([("UNIT_MILLION", "million")]) is True


def test_should_drop_lone_ambiguous_unit_ignores_glue_tokens():
    # "kharab" surrounded by SEP/RANGE/DOT/COMMA/O glue is still "lone".
    toks = [("SEP", " "), ("UNIT_KHARAB", "kharab"), ("O", "")]
    assert core.should_drop_lone_ambiguous_unit(toks) is True


def test_should_drop_lone_ambiguous_unit_with_preceding_digits_kept():
    # "das kharab" / "2 mil" -- a number precedes the unit, so it's kept.
    assert core.should_drop_lone_ambiguous_unit([("DIGITS", "2"), ("SEP", " "), ("UNIT_MILLION", "mil")]) is False
    assert core.should_drop_lone_ambiguous_unit([("CARD_10", "das"), ("SEP", " "), ("UNIT_KHARAB", "kharab")]) is False


def test_should_drop_lone_ambiguous_unit_with_preceding_prefix_kept():
    assert core.should_drop_lone_ambiguous_unit([("PFX_DHAI", "dhai"), ("SEP", " "), ("UNIT_MILLION", "mil")]) is False


def test_should_drop_lone_ambiguous_unit_other_units_unaffected():
    # "lakh" / "hazaar" alone are NOT ambiguous in this sense -- only
    # UNIT_KHARAB/UNIT_MILLION are gated.
    assert core.should_drop_lone_ambiguous_unit([("UNIT_LAKH", "lakh")]) is False
    assert core.should_drop_lone_ambiguous_unit([("UNIT_HAZAAR", "hazaar")]) is False


def test_should_drop_lone_ambiguous_unit_multi_token_span_kept():
    # more than one meaningful token -- e.g. a unit plus another unit -- is
    # not a "lone" ambiguous unit.
    toks = [("UNIT_KHARAB", "kharab"), ("SEP", " "), ("UNIT_LAKH", "lakh")]
    assert core.should_drop_lone_ambiguous_unit(toks) is False


def test_detect_currency_window_wide_enough_for_rupees_after_a_short_number():
    # "1200 rupees" -- the "rupees" (6-char) marker sits right after a
    # single-space gap past the span end; the scan window must be wide
    # enough to see the whole word, not just its first few letters.
    text = "1200 rupees mein mil gaya"
    assert core.detect_currency(text, 0, 4, pack) == "INR"


def test_detect_currency_glued_symbol_before_digits():
    text = "₹85000"
    assert core.detect_currency(text, 1, 6, pack) == "INR"


def test_r7_teen_hazaar_paanch_hazaar_juxtaposition_range():
    r = ev(("CARD_3", "teen"), ("SEP", " "), ("UNIT_HAZAAR", "hazaar"), ("SEP", " "),
           ("CARD_5", "paanch"), ("SEP", " "), ("UNIT_HAZAAR", "hazaar"))
    assert r.value == 3000, r.value
    assert r.range == (3000, 5000), r.range
    assert r.unit == "hazaar"


def test_r7_bees_lakh_pachees_lakh_juxtaposition_range():
    r = ev(("CARD_20", "bees"), ("SEP", " "), ("UNIT_LAKH", "lakh"), ("SEP", " "),
           ("CARD_25", "pachees"), ("SEP", " "), ("UNIT_LAKH", "lakh"))
    assert r.value == 2000000, r.value
    assert r.range == (2000000, 2500000), r.range
    assert r.unit == "lakh"


def test_r7_3_hazaar_5_hazaar_digits_juxtaposition_range():
    r = ev(("DIGITS", "3"), ("SEP", " "), ("UNIT_HAZAAR", "hazaar"), ("SEP", " "),
           ("DIGITS", "5"), ("SEP", " "), ("UNIT_HAZAAR", "hazaar"))
    assert r.value == 3000, r.value
    assert r.range == (3000, 5000), r.range


def test_r7_paanch_lakh_paanch_lakh_equal_falls_back_to_additive():
    # Deliberate choice: equal juxtaposed terms are ambiguous with plain
    # repetition/emphasis, not a genuine [x, x] range, so R7 requires
    # low < high strictly and this falls back to today's additive sum.
    r = ev(("CARD_5", "paanch"), ("SEP", " "), ("UNIT_LAKH", "lakh"), ("SEP", " "),
           ("CARD_5", "paanch"), ("SEP", " "), ("UNIT_LAKH", "lakh"))
    assert r.value == 1000000, r.value
    assert r.range is None


def test_r7_das_hazaar_crore_multiplicative_stacking_unaffected():
    # Second term (crore) has NO explicit coefficient, so R7 must not fire.
    r = ev(("CARD_10", "das"), ("SEP", " "), ("UNIT_HAZAAR", "hazaar"), ("SEP", " "),
           ("UNIT_CRORE", "crore"))
    assert r.value == 1e11, r.value
    assert r.range is None


def test_r7_ek_lakh_dus_hazaar_descending_additive_chain_unaffected():
    # unit_value(hazaar) < unit_value(lakh), so R7 must not fire.
    r = ev(("CARD_1", "ek"), ("SEP", " "), ("UNIT_LAKH", "lakh"), ("SEP", " "),
           ("CARD_10", "dus"), ("SEP", " "), ("UNIT_HAZAAR", "hazaar"))
    assert r.value == 110000, r.value
    assert r.range is None


def test_r7_do_teen_lakh_cardinal_only_juxtaposition_unaffected():
    # By the time this reaches evaluate(), decode-time repair has already
    # inserted an explicit RANGE token between the two cardinals, so this
    # exercises the existing RANGE-based path (R6-adjacent), not R7.
    r = ev(("CARD_2", "do"), ("RANGE", " "),
           ("CARD_3", "teen"), ("SEP", " "), ("UNIT_LAKH", "lakh"))
    assert r.range == (200000, 300000), r.range
    assert r.unit == "lakh"


def test_devanagari_digits_pachees_hazaar():
    r = ev(("DIGITS", "२५"), ("SEP", " "), ("UNIT_HAZAAR", "हजार"))
    assert r.value == 25000, r.value


def test_devanagari_digits_bare():
    r = ev(("DIGITS", "७५००"))
    assert r.value == 7500, r.value


def test_devanagari_digits_grouped():
    r = ev(("DIGITS", "२"), ("COMMA", ","), ("DIGITS", "५०"), ("COMMA", ","), ("DIGITS", "०००"))
    assert r.value == 250000, r.value


def test_devanagari_digits_decimal():
    r = ev(("DIGITS", "१"), ("DOT", "."), ("DIGITS", "५"))
    assert r.value == 1.5, r.value


def test_devanagari_digits_mixed_ascii_and_deva():
    r = ev(("DIGITS", "1"), ("DOT", "."), ("DIGITS", "५"))
    assert r.value == 1.5, r.value


def test_gujarati_digits_bare():
    r = ev(("DIGITS", "૭૫૦૦"))
    assert r.value == 7500, r.value


def test_r8_tees_paintees_hazaar_juxtaposition_range():
    # "tees paintees hazaar" (CARD_30 SEP CARD_35 SEP UNIT_HAZAAR) means
    # the range 30,000-35,000, not 30 + 35*1000 = 35030.
    r = ev(("CARD_30", "tees"), ("SEP", " "), ("CARD_35", "paintees"), ("SEP", " "),
           ("UNIT_HAZAAR", "hazaar"))
    assert r.value == 30000, r.value
    assert r.range == (30000, 35000), r.range
    assert r.unit == "hazaar"


def test_r8_paach_ten_hazar_juxtaposition_range():
    r = ev(("CARD_5", "paach"), ("SEP", " "), ("CARD_10", "ten"), ("SEP", " "),
           ("UNIT_HAZAAR", "hazar"))
    assert r.value == 5000, r.value
    assert r.range == (5000, 10000), r.range
    assert r.unit == "hazaar"


def test_r8_do_teen_lakh_juxtaposition_range_at_core_level():
    # At core level (no RANGE token inserted, unlike the decoder path),
    # two adjacent spelled cardinals before a unit form a range.
    r = ev(("CARD_2", "do"), ("SEP", " "), ("CARD_3", "teen"), ("SEP", " "),
           ("UNIT_LAKH", "lakh"))
    assert r.value == 200000, r.value
    assert r.range == (200000, 300000), r.range
    assert r.unit == "lakh"


def test_r8_ek_sau_unaffected():
    # "sau" is a UNIT, not a second cardinal -- no adjacent CARD_ pair.
    r = ev(("CARD_1", "ek"), ("SEP", " "), ("UNIT_SAU", "sau"))
    assert r.value == 100, r.value
    assert r.range is None


def test_r8_do_hazaar_paanch_additive_unaffected():
    # Cardinals are separated by a UNIT token, not merely SEP -- additive.
    r = ev(("CARD_2", "do"), ("SEP", " "), ("UNIT_HAZAAR", "hazaar"), ("SEP", " "),
           ("CARD_5", "paanch"))
    assert r.value == 2005, r.value
    assert r.range is None


def test_r8_bees_paanch_descending_unaffected():
    # second < first, so the a < b guard fails and today's additive
    # behaviour is kept.
    r = ev(("CARD_20", "bees"), ("SEP", " "), ("CARD_5", "paanch"))
    assert r.value == 25, r.value
    assert r.range is None


def test_r7_wins_over_r8_precedence():
    # "teen hazaar paanch das lakh": R7 matches at the UNIT_HAZAAR /
    # UNIT_LAKH term boundary (both terms have explicit coefficients and
    # unit_value(lakh) >= unit_value(hazaar)). Within the second term,
    # CARD_5 SEP CARD_10 is ALSO an R8-eligible adjacent-cardinal pair, but
    # R7 is checked first and must win for the whole span.
    r = ev(("CARD_3", "teen"), ("SEP", " "), ("UNIT_HAZAAR", "hazaar"), ("SEP", " "),
           ("CARD_5", "paanch"), ("SEP", " "), ("CARD_10", "das"), ("SEP", " "),
           ("UNIT_LAKH", "lakh"))
    assert r.value == 3000, r.value
    assert r.range == (3000, 1000005), r.range
    assert r.unit == "hazaar"
