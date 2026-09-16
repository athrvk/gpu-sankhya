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


def _run_all():
    fns = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"OK  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
