"""Unit tests for sankhya.verify (the strict-tier span predicate) and for the
exported lexicon staying in sync with the language packs."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sankhya import export_lexicon
from sankhya.verify import LEXICON_PATH, load_lexicon, verify_tokens


def test_sava_lakh_is_verified():
    assert verify_tokens([("PFX_SAVA", "sava"), ("SEP", " "), ("UNIT_LAKH", "lakh")])


def test_misspelt_unit_is_not_verified():
    assert not verify_tokens([("PFX_SAVA", "sava"), ("SEP", " "), ("UNIT_LAKH", "lakhhz")])


def test_digits_ascii_and_devanagari():
    assert verify_tokens([("DIGITS", "42"), ("SEP", " "), ("UNIT_LAKH", "lakh")])
    assert verify_tokens([("DIGITS", "४२"), ("SEP", " "), ("UNIT_LAKH", "लाख")])
    assert not verify_tokens([("DIGITS", "4a"), ("SEP", " "), ("UNIT_LAKH", "lakh")])
    assert not verify_tokens([("DIGITS", ""), ("UNIT_LAKH", "lakh")])


def test_sep_must_be_whitespace():
    assert not verify_tokens([("DIGITS", "5"), ("SEP", "x"), ("UNIT_LAKH", "lakh")])
    assert not verify_tokens([("DIGITS", "5"), ("SEP", ""), ("UNIT_LAKH", "lakh")])


def test_range_words_and_symbols():
    assert verify_tokens([("DIGITS", "5"), ("SEP", " "), ("RANGE", "se"), ("SEP", " "),
                          ("DIGITS", "7"), ("SEP", " "), ("UNIT_LAKH", "lakh")])
    assert verify_tokens([("DIGITS", "5"), ("RANGE", "-"), ("DIGITS", "7"),
                          ("SEP", " "), ("UNIT_LAKH", "lakh")])
    assert not verify_tokens([("DIGITS", "5"), ("SEP", " "), ("RANGE", "xyz"), ("SEP", " "),
                              ("DIGITS", "7"), ("SEP", " "), ("UNIT_LAKH", "lakh")])


def test_o_only_when_lexicon_says_o():
    forms = {}
    for pack in load_lexicon()["packs"].values():
        forms.update(pack["forms"])
    o_word = next(k for k, v in forms.items() if v == "O")
    assert verify_tokens([("DIGITS", "5"), ("SEP", " "), ("O", o_word)])
    assert not verify_tokens([("DIGITS", "5"), ("SEP", " "), ("O", "zzqq")])


def test_english_fraction_phrase_whole_token():
    assert verify_tokens([("CARD_8", "eight"), ("SEP", " "), ("PFX_SAADHE", "and a half"),
                          ("SEP", " "), ("UNIT_CRORE", "crore")])
    assert not verify_tokens([("PFX_SAADHE", "and a"), ("SEP", " "), ("UNIT_CRORE", "crore")])


def test_dot_and_comma():
    assert verify_tokens([("DIGITS", "1"), ("DOT", "."), ("DIGITS", "5"),
                          ("SEP", " "), ("UNIT_CRORE", "crore")])
    assert verify_tokens([("DIGITS", "1"), ("COMMA", ","), ("DIGITS", "500"),
                          ("SEP", " "), ("UNIT_CRORE", "crore")])
    assert not verify_tokens([("DOT", ".."), ("DIGITS", "5"), ("UNIT_CRORE", "crore")])


def test_empty_and_contentless_spans():
    assert not verify_tokens([])
    assert not verify_tokens([("SEP", " ")])


def test_class_must_match_the_lexicon_entry():
    # "lakh" exists, but not as a cardinal
    assert not verify_tokens([("CARD_5", "lakh")])


def test_exported_lexicon_matches_packs():
    built = export_lexicon.build()
    with open(LEXICON_PATH, "r", encoding="utf-8") as f:
        committed = json.load(f)
    assert committed == built, "src/data/lexicon.json is stale: rerun python -m sankhya.export_lexicon"
