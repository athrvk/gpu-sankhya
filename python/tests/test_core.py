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


def _run_all():
    fns = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"OK  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
