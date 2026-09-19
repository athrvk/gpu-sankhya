"""The real-text precision rules (R11-R18), mirrored by test/wild-rules.test.ts.

Each rule here exists because of a concrete failure shape in
`python/data_wild/REPORT.md` §4; the docstrings name the sentence.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sankhya import core, verify
from sankhya.decode import decode_spans, _looks_like_year, _same_value_restatement
from sankhya import classes as C


# --- R11: lone prefix -----------------------------------------------------

def test_r11_drops_a_lone_prefix():
    # "ढाई साल" = two and a half YEARS, "sade hue tamatar" = rotten tomatoes
    assert core.should_drop_lone_prefix([("PFX_DHAI", "ढाई")])
    assert core.should_drop_lone_prefix([("PFX_SAADHE", "sade")])
    assert core.should_drop_lone_prefix([("SEP", " "), ("PFX_AADHA", "adha"), ("O", "x")])


def test_r11_keeps_a_prefix_with_a_number_or_unit():
    assert not core.should_drop_lone_prefix(
        [("PFX_DHAI", "ढाई"), ("SEP", " "), ("UNIT_LAKH", "लाख")])
    assert not core.should_drop_lone_prefix(
        [("PFX_SAADHE", "sadhe"), ("SEP", " "), ("CARD_4", "char"),
         ("SEP", " "), ("UNIT_LAKH", "lakh")])


# --- R12: ambiguous forms -------------------------------------------------

@pytest.mark.parametrize("tokens", [
    [("UNIT_SAU", "so")],                                    # English "so"
    [("CARD_60", "sath")],                                   # "with"
    [("CARD_60", "saath")],
    [("UNIT_LAKH", "peti")],                                 # "corn belt"
    [("UNIT_ARAB", "arab")],                                 # sanyukt arab amirat
    [("UNIT_ARAB", "अरब")],                                  # अरब संघ
    [("UNIT_ARAB", "અરબ")],                                  # અરબ સાગર
    [("CARD_1", "एक")],                                      # the article
    [("CARD_9", "सऊदी"), ("SEP", " "), ("UNIT_ARAB", "अरब")],  # only अरब verifies
])
def test_r12_drops_spans_held_up_only_by_ambiguous_surfaces(tokens):
    assert verify.should_drop_ambiguous_form(tokens)


@pytest.mark.parametrize("tokens", [
    [("CARD_1", "ek"), ("SEP", " "), ("UNIT_ARAB", "arab")],
    [("DIGITS", "100"), ("SEP", " "), ("UNIT_ARAB", "अरब")],
    [("CARD_2", "do"), ("SEP", " "), ("UNIT_LAKH", "peti")],
    [("CARD_10", "das"), ("SEP", " "), ("UNIT_KHARAB", "kharab")],
    # "bees lac pachees lac": bees/lac are ambiguous, pachees is not
    [("CARD_20", "bees"), ("SEP", " "), ("UNIT_LAKH", "lac"), ("SEP", " "),
     ("CARD_25", "pachees"), ("SEP", " "), ("UNIT_LAKH", "lac")],
    # "unnis sau sath" = 1960: "sau" corroborates the ambiguous "sath"
    [("CARD_19", "unnis"), ("SEP", " "), ("UNIT_SAU", "sau"), ("SEP", " "),
     ("CARD_60", "sath")],
    [("CARD_1", "एक"), ("SEP", " "), ("UNIT_LAKH", "लाख")],
])
def test_r12_keeps_a_span_with_one_unambiguous_number_word(tokens):
    assert not verify.should_drop_ambiguous_form(tokens)


# --- R13b: bare cardinals -------------------------------------------------

def test_r13b_keeps_an_exact_bare_cardinal_form():
    # gold_deva has a bare "पचास" = 50 span; gold_gu a bare "બાવીસ" = 22.
    assert not verify.should_drop_unsupported_bare_cardinal([("CARD_50", "पचास")])
    assert not verify.should_drop_unsupported_bare_cardinal([("CARD_22", "બાવીસ")])


@pytest.mark.parametrize("tokens", [
    [("CARD_89", "aavesh")],
    [("CARD_6", "chaahie")],
    [("CARD_12", "barabar")],
    [("CARD_1", "इ")],
])
def test_r13b_drops_a_bare_cardinal_that_is_not_a_lexicon_form(tokens):
    assert verify.should_drop_unsupported_bare_cardinal(tokens)


def test_r13b_only_fires_on_a_span_with_nothing_else_in_it():
    assert not verify.should_drop_unsupported_bare_cardinal(
        [("CARD_89", "aavesh"), ("SEP", " "), ("UNIT_LAKH", "lakh")])


# --- R14: blocked surfaces ------------------------------------------------

@pytest.mark.parametrize("surface,cls", [
    ("हजारे", "UNIT_HAZAAR"),   # अण्णा हजारे
    ("सवाई", "PFX_SAVA"),       # सवाई तुकोजीराव
    ("अरबी", "UNIT_ARAB"),      # अरबी समुद्र
    ("અરબી", "UNIT_ARAB"),      # અરબી સમુદ્ર
    ("હજારે", "UNIT_HAZAAR"),
])
def test_r14_a_blocked_surface_never_verifies_and_is_dropped(surface, cls):
    assert verify.is_blocked_surface(surface)
    assert not verify.verify_tokens([(cls, surface)])
    assert verify.has_blocked_surface([(cls, surface)])
    assert not verify.is_known_form(cls, surface)


def test_r14_leaves_the_head_word_alone():
    assert verify.verify_tokens([("UNIT_HAZAAR", "हजार")])
    assert not verify.has_blocked_surface([("UNIT_HAZAAR", "हजार")])


# --- R15: per-pack suffix stripping ---------------------------------------

def test_r15_an_inflected_form_still_verifies_within_its_own_pack():
    # Marathi "लाखांचं" = लाख + oblique "ां" + ending "चं"; Gujarati "કરોડનો".
    assert verify.verify_tokens([("UNIT_LAKH", "लाखांचं")])
    assert verify.verify_tokens([("UNIT_CRORE", "કરોડનો")])


def test_r15_a_stripped_head_must_survive_at_three_characters():
    assert verify.MIN_SUFFIX_HEAD_LEN == 3
    # "सोच" would otherwise strip Marathi's "-च" down to a 2-char head.
    assert not verify.is_known_form("UNIT_SAU", "सोच")


# --- R16: loose symbol units ----------------------------------------------

def test_r16_drops_a_symbol_unit_that_is_not_glued_to_its_digits():
    # "1996 k" (the ke/ki clitic after a year), "33 k. m." (kilometres)
    assert verify.should_drop_loose_symbol_unit(
        [("DIGITS", "1996"), ("SEP", " "), ("UNIT_HAZAAR", "k")])
    assert verify.should_drop_loose_symbol_unit(
        [("DIGITS", "33"), ("SEP", " "), ("UNIT_HAZAAR", "k"),
         ("SEP", ". "), ("UNIT_MILLION", "m")])


def test_r16_keeps_a_glued_symbol_unit_and_a_long_one():
    assert not verify.should_drop_loose_symbol_unit(
        [("DIGITS", "20"), ("UNIT_HAZAAR", "k")])
    assert not verify.should_drop_loose_symbol_unit(
        [("DIGITS", "2"), ("DOT", "."), ("DIGITS", "5"), ("UNIT_LAKH", "l")])
    # "9.5 LPA" / "8 lac" are routinely written with a space
    assert not verify.should_drop_loose_symbol_unit(
        [("DIGITS", "9"), ("DOT", "."), ("DIGITS", "5"), ("SEP", " "), ("UNIT_LAKH", "lpa")])
    assert not verify.should_drop_loose_symbol_unit(
        [("DIGITS", "8"), ("SEP", " "), ("UNIT_LAKH", "lac")])


# --- R17: unknown unit words ----------------------------------------------

@pytest.mark.parametrize("tokens", [
    [("DIGITS", "3"), ("SEP", " "), ("UNIT_HAZAAR", "hours")],   # -> 3,000
    [("DIGITS", "25"), ("UNIT_LAKH", "वे")],                      # an ordinal
    [("DIGITS", "1980"), ("UNIT_LAKH", "ના")],                    # a decade
    [("UNIT_CRORE", "chori")],
    [("UNIT_HAZAAR", "thought")],
    [("UNIT_SAU", "oooh")],
    [("UNIT_CRORE", "खोड")],
])
def test_r17_drops_a_unit_word_that_is_not_a_unit_word(tokens):
    assert verify.should_drop_unknown_unit(tokens)


@pytest.mark.parametrize("tokens", [
    [("DIGITS", "20"), ("SEP", " "), ("UNIT_HAZAAR", "hazaar")],
    [("DIGITS", "20"), ("SEP", " "), ("UNIT_HAZAAR", "hazzar")],   # one typo
    [("DIGITS", "2"), ("SEP", " "), ("UNIT_CRORE", "करोड")],
    [("DIGITS", "42"), ("SEP", " "), ("UNIT_CRORE", "કરોડનો")],     # inflected
])
def test_r17_tolerates_a_real_or_one_character_typod_unit(tokens):
    assert not verify.should_drop_unknown_unit(tokens)


def test_r17_does_not_fire_when_a_justified_cardinal_corroborates_the_unit():
    assert not verify.should_drop_unknown_unit(
        [("CARD_3", "teen"), ("SEP", " "), ("UNIT_HAZAAR", "hours")])


def test_levenshtein_le_is_bounded_and_symmetric():
    assert verify.levenshtein_le("lakh", "lakh", 1)
    assert verify.levenshtein_le("lakh", "lakhh", 1)
    assert verify.levenshtein_le("lakhh", "lakh", 1)
    assert not verify.levenshtein_le("lakh", "hours", 1)
    assert verify.levenshtein_le("lakh", "hours", 5)


# --- R18: boundary artifacts ----------------------------------------------

def test_r18_year_and_restatement_predicates():
    assert _looks_like_year("1987")
    assert _looks_like_year("२०११")       # Devanagari digits
    assert not _looks_like_year("1899")
    assert not _looks_like_year("198")
    assert _same_value_restatement("DIGITS", "5", "CARD_5", "panch")
    assert not _same_value_restatement("DIGITS", "5", "CARD_7", "saat")
    assert not _same_value_restatement("CARD_5", "panch", "CARD_5", "panch")


def _spans(text, tokens):
    """Decode `text` with a hand-built per-char BIO/class labelling from a
    list of (class, substring) tokens laid end to end."""
    bio, cls = [], []
    for i, (cname, sub) in enumerate(tokens):
        for j in range(len(sub)):
            bio.append(1 if (i == 0 and j == 0) else 2)
            cls.append(C.CLASS_TO_ID[cname])
    assert "".join(t for _c, t in tokens) == text
    return decode_spans(text, bio, cls)


def test_r18_trims_a_trailing_year():
    text = "दस हज़ार 1987"
    out = _spans(text, [("CARD_10", "दस"), ("SEP", " "), ("UNIT_HAZAAR", "हज़ार"),
                        ("SEP", " "), ("DIGITS", "1987")])
    assert len(out) == 1
    assert out[0]["text"] == "दस हज़ार"
    assert core.evaluate([(C.CLASSES[c], t) for c, t in out[0]["tokens"]]).value == 10000


def test_r18_drops_a_leading_digits_restatement():
    text = "5 panch hajar"
    out = _spans(text, [("DIGITS", "5"), ("SEP", " "), ("CARD_5", "panch"),
                        ("SEP", " "), ("UNIT_HAZAAR", "hajar")])
    assert len(out) == 1
    assert out[0]["text"] == "panch hajar"
    assert out[0]["start"] == 2
    assert core.evaluate([(C.CLASSES[c], t) for c, t in out[0]["tokens"]]).value == 5000


def test_r18_leaves_a_real_digits_coefficient_alone():
    text = "1987 hazaar"
    out = _spans(text, [("DIGITS", "1987"), ("SEP", " "), ("UNIT_HAZAAR", "hazaar")])
    assert len(out) == 1
    assert out[0]["text"] == text
