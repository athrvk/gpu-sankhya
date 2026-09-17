"""Tests for sankhya.llm_corpus.verify_line against an inline raw fixture."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sankhya import core
from sankhya.langs import base
from sankhya import llm_corpus as LC

pack = base.get_pack("hi_latn")


def _verify(obj, gold_texts=frozenset(), seen_texts=None):
    if seen_texts is None:
        seen_texts = set()
    return LC.verify_line(pack, obj, gold_texts, seen_texts)


def test_accepted_line_roundtrips():
    obj = {
        "text": "bhai sava lakh mein ho jayega kya",
        "lang": "hi_latn",
        "phrases": [{"phrase": "sava lakh", "value": 125000}],
    }
    ex = _verify(obj)
    assert ex["text"] == obj["text"]
    assert len(ex["bio"]) == len(ex["text"])
    assert len(ex["cls"]) == len(ex["text"])
    assert len(ex["spans"]) == 1
    sp = ex["spans"][0]
    assert ex["text"][sp["start"]:sp["end"]] == "sava lakh"
    toks = [(c, "x") for c in sp["classes"].split()]
    result = core.evaluate(toks)
    assert result.value == 125000 == sp["value"]


def test_value_mismatch_rejected():
    obj = {
        "text": "usne 20k mila bhai",
        "lang": "hi_latn",
        "phrases": [{"phrase": "20k", "value": 999999}],
    }
    try:
        _verify(obj)
        assert False, "expected RejectError"
    except LC.RejectError as e:
        assert e.args[0].startswith("value_mismatch"), e.args[0]


def test_unknown_token_rejected():
    obj = {
        "text": "usne zoop lakh maange",
        "lang": "hi_latn",
        "phrases": [{"phrase": "zoop lakh", "value": 500000}],
    }
    try:
        _verify(obj)
        assert False, "expected RejectError"
    except LC.RejectError as e:
        assert e.args[0] == "unknown_token:zoop", e.args[0]


def test_bad_phrase_substring_rejected():
    obj = {
        "text": "yeh sentence mein amount nahi hai",
        "lang": "hi_latn",
        "phrases": [{"phrase": "dedh lakh", "value": 150000}],
    }
    try:
        _verify(obj)
        assert False, "expected RejectError"
    except LC.RejectError as e:
        assert e.args[0] == "phrase_not_found", e.args[0]


def test_negative_with_hidden_quantity_rejected():
    obj = {
        "text": "usne 5 lakh kamaye lekin bataya nahi",
        "lang": "hi_latn",
        "phrases": [],
    }
    try:
        _verify(obj)
        assert False, "expected RejectError"
    except LC.RejectError as e:
        assert e.args[0] == "negative_contains_quantity", e.args[0]


def test_negative_without_quantity_accepted():
    obj = {
        "text": "9876543210 pe call karo",
        "lang": "hi_latn",
        "phrases": [],
    }
    ex = _verify(obj)
    assert ex["spans"] == []
    assert len(ex["bio"]) == len(ex["text"])
    assert len(ex["cls"]) == len(ex["text"])


def test_duplicate_rejected():
    obj = {
        "text": "usne dedh lakh maange yaar",
        "lang": "hi_latn",
        "phrases": [{"phrase": "dedh lakh", "value": 150000}],
    }
    seen = set()
    ex = _verify(obj, seen_texts=seen)
    from sankhya import charset as CS
    seen.add(CS.normalize_text(ex["text"]))
    try:
        _verify(obj, seen_texts=seen)
        assert False, "expected RejectError"
    except LC.RejectError as e:
        assert e.args[0] == "dup", e.args[0]


def test_dup_gold_rejected():
    from sankhya import charset as CS
    obj = {
        "text": "bhai sava lakh mein ho jayega kya",
        "lang": "hi_latn",
        "phrases": [{"phrase": "sava lakh", "value": 125000}],
    }
    gold = {CS.normalize_text(obj["text"])}
    try:
        _verify(obj, gold_texts=gold)
        assert False, "expected RejectError"
    except LC.RejectError as e:
        assert e.args[0] == "dup_gold", e.args[0]


def test_range_phrase_accepted():
    obj = {
        "text": "flat 2-3 lakh mein aa jayega",
        "lang": "hi_latn",
        "phrases": [{"phrase": "2-3 lakh", "value": 200000, "range": [200000, 300000]}],
    }
    ex = _verify(obj)
    sp = ex["spans"][0]
    assert sp["range"] == [200000, 300000]


def test_currency_marker_trimmed_from_span():
    """Rs.4,500 -> span excludes "Rs.", value is 4500 not 0.45; currency
    is still detected from the (now-adjacent) marker."""
    obj = {
        "text": "usne Rs.4,500 diye mujhe",
        "lang": "hi_latn",
        "phrases": [{"phrase": "Rs.4,500", "value": 4500}],
    }
    ex = _verify(obj)
    sp = ex["spans"][0]
    assert sp["value"] == 4500, sp
    assert ex["text"][sp["start"]:sp["end"]] == "4,500"
    assert sp["currency"] == "INR"


def test_currency_marker_trimmed_grouped_digits():
    obj = {
        "text": "flat Rs.1,20,000 mein diya",
        "lang": "hi_latn",
        "phrases": [{"phrase": "Rs.1,20,000", "value": 120000}],
    }
    ex = _verify(obj)
    sp = ex["spans"][0]
    assert sp["value"] == 120000, sp
    assert ex["text"][sp["start"]:sp["end"]] == "1,20,000"


def test_hyphen_between_digit_and_unit_word_is_not_range():
    """"35-lakh" is a compound number (digit glued to unit by hyphen), not
    a range -- the hyphen must stay SEP, not RANGE."""
    obj = {
        "text": "gaadi 35-lakh mein bik gayi",
        "lang": "hi_latn",
        "phrases": [{"phrase": "35-lakh", "value": 3500000}],
    }
    ex = _verify(obj)
    sp = ex["spans"][0]
    assert sp["value"] == 3500000, sp
    assert sp["range"] is None


def test_hyphen_between_two_digit_runs_is_range():
    obj = {
        "text": "flat 2-3 lakh mein aa jayega",
        "lang": "hi_latn",
        "phrases": [{"phrase": "2-3 lakh", "value": 200000, "range": [200000, 300000]}],
    }
    ex = _verify(obj)
    sp = ex["spans"][0]
    assert sp["range"] == [200000, 300000]


def test_ambiguous_phrase_respects_token_boundary():
    """"5000" is a substring of "45000" but not a boundary match, so the
    claimed "5000" phrase resolves unambiguously to its own occurrence."""
    obj = {
        "text": "sirf 5000 ka tha, ab 45000",
        "lang": "hi_latn",
        "phrases": [{"phrase": "5000", "value": 5000}],
    }
    ex = _verify(obj)
    sp = ex["spans"][0]
    assert ex["text"][sp["start"]:sp["end"]] == "5000"
    assert sp["value"] == 5000


def test_negative_duration_prefix_words_accepted():
    for text in ["saadhe teen baje", "dhai mahine ka advance", "chaar dost"]:
        obj = {"text": text, "lang": "hi_latn", "phrases": []}
        ex = _verify(obj)
        assert ex["spans"] == [], text


def test_negative_unit_word_still_rejected():
    """"unnees so pachasi" contains "so" (UNIT_SAU) -- still ambiguous
    enough to reject as a negative."""
    obj = {"text": "unnees so pachasi mein hua tha", "lang": "hi_latn", "phrases": []}
    try:
        _verify(obj)
        assert False, "expected RejectError"
    except LC.RejectError as e:
        assert e.args[0] == "negative_contains_quantity", e.args[0]


def test_char_maps_to_card_4():
    forms = pack.all_forms()
    assert forms.get("char") == "CARD_4"


def test_lang_mismatch_rejected():
    obj = {
        "text": "some devanagari lang line",
        "lang": "hi_deva",
        "phrases": [],
    }
    try:
        _verify(obj)
        assert False, "expected RejectError"
    except LC.RejectError as e:
        assert e.args[0] == "lang_mismatch", e.args[0]
