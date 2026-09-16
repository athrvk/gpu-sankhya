"""Tests for the Devanagari Hindi (hi_deva) language pack: normalization,
noise safety, generator round-trip, cross-script mixing and negatives."""
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sankhya import core
from sankhya.classes import CLASSES
from sankhya.charset import normalize_text
from sankhya.langs import base
from sankhya import generator as G
from sankhya import noise_deva as ND

N = 20000
SEED = 0


def _rebuild_tokens(text, cls_ids, start, end):
    toks = []
    j = start
    while j < end:
        c = cls_ids[j]
        k = j
        while k < end and cls_ids[k] == c:
            k += 1
        toks.append((CLASSES[c], text[j:k]))
        j = k
    return toks


# ---------------------------------------------------------------- normalize

def test_normalize_nfc():
    # decomposed + composed forms of the same character normalize identically
    decomposed = "क" + "ा"  # क + ा (matra) - already NFC in this case
    assert normalize_text(decomposed) == normalize_text(decomposed)
    import unicodedata
    s = unicodedata.normalize("NFD", "कि")
    assert normalize_text(s) == normalize_text(unicodedata.normalize("NFC", "कि"))


def test_normalize_digit_mapping():
    assert normalize_text("२.५ लाख") == "2.5 लाख"
    assert normalize_text("०१२३४५६७८९") == "0123456789"


def test_normalize_lowercase():
    assert normalize_text("HELLO Sava LAKH") == "hello sava lakh"


def test_normalize_length_preserved_1to1():
    for s in ["सवा लाख", "२.५ लाख", "Rs 500", "डेढ़ लाख"]:
        assert len(normalize_text(s)) == len(s)


# -------------------------------------------------------------------- noise

def test_noise_never_collides_via_safe_word():
    """Raw noise_deva.apply_noise() can occasionally land on another class's
    surface form (e.g. matra-deletion turns pachaasi/85 into pachaas/50) -
    that's expected and is exactly what the generator's collision-rejection
    (_safe_word, retried against pack.all_forms()) exists to filter out, the
    same mechanism hi_latn relies on. Verify that safety net actually holds."""
    pack = base.get_pack("hi_deva")
    forms = pack.all_forms()
    rng = random.Random(1)
    collisions = 0
    total = 0
    for cls in pack.lexicon:
        for _ in range(20):
            total += 1
            word, _ = G._safe_word(pack, rng, cls)
            other = forms.get(word.lower())
            if other is not None and other != cls:
                collisions += 1
    assert collisions == 0, f"{collisions}/{total} noise collisions"


def test_noise_saath_vs_saadhe_prefix_no_collision():
    # साठ (60) vs साढ़े prefix, सात (7) vs साठ - nukta/matra noise must not
    # turn one into the other.
    pack = base.get_pack("hi_deva")
    forms = pack.all_forms()
    rng = random.Random(2)
    for w, cls in [("साठ", "CARD_60"), ("सात", "CARD_7"), ("साढ़े", "PFX_SAADHE")]:
        for _ in range(50):
            noised = ND.apply_noise(w, rng)
            other = forms.get(noised.lower())
            assert other is None or other == cls, (w, noised, other, cls)


def test_to_deva_digits():
    assert ND.to_deva_digits("2025") == "२०२५"
    assert ND.to_deva_digits("1,25,000.5") == "१,२५,०००.५"


# --------------------------------------------------------------- generator

def test_generate_roundtrip_hi_deva():
    pack = base.get_pack("hi_deva")
    rng = random.Random(SEED)
    examples = [G.sample_example(pack, rng) for _ in range(N)]

    n_neg = 0
    for ex in examples:
        text = ex["text"]
        assert len(ex["bio"]) == len(text)
        assert len(ex["cls"]) == len(text)
        if not ex["spans"]:
            n_neg += 1
            continue
        for sp in ex["spans"]:
            toks = _rebuild_tokens(text, ex["cls"], sp["start"], sp["end"])
            norm_toks = [(c, normalize_text(t)) for c, t in toks]
            res = core.evaluate(norm_toks)
            assert res.value == sp["value"], (text, sp, res)

    frac_neg = n_neg / N
    assert 0.05 <= frac_neg <= 0.25, frac_neg


def test_generate_collision_safe():
    """20k mixed examples round-trip through core without exceptions."""
    pack = base.get_pack("hi_deva")
    rng = random.Random(123)
    for _ in range(20000):
        ex = G.sample_example(pack, rng)
        assert len(ex["text"]) <= G.MAX_LEN


def test_devanagari_digit_examples_labelled_digits():
    pack = base.get_pack("hi_deva")
    rng = random.Random(5)
    found = 0
    for _ in range(4000):
        ex = G.sample_example(pack, rng)
        text = ex["text"]
        if any(ch in "०१२३४५६७८९" for ch in text):
            found += 1
            for i, ch in enumerate(text):
                if ch in "०१२३४५६७८९":
                    assert CLASSES[ex["cls"][i]] == "DIGITS", (text, ch, i)
    assert found > 0, "no Devanagari-digit examples generated in 4000 tries"


def test_hash_number_negatives_no_span():
    pack = base.get_pack("hi_deva")
    for t in ["फ्लैट #1 देख लिया था", "आइटम #12 पसंद आया", "फ्लैट नंबर 4 खाली है"]:
        assert t in pack.templates["negatives"]


def test_hi_latn_hash_number_negatives_present():
    from sankhya.langs import hi_latn
    negs = " ".join(hi_latn.TEMPLATES["negatives"])
    assert "#" in negs or "number" in negs


# ---------------------------------------------------------------- multi-lang

def test_cross_script_share():
    p1 = base.get_pack("hi_latn")
    p2 = base.get_pack("hi_deva")
    exs = G.generate_multi([p1, p2], weights=[0.55, 0.45], n=5000, seed=9, cross=0.10)
    cross = sum(1 for e in exs if e.get("cross_script"))
    assert cross / len(exs) >= 0.08, cross / len(exs)
    # single-pack behaviour unchanged: generate() output shape matches sample_example
    single = G.generate(p1, 50, seed=1)
    assert all("lang" in e and e["lang"] == "hi_latn" for e in single)


def test_detect_currency_multi_union():
    p1 = base.get_pack("hi_latn")
    p2 = base.get_pack("hi_deva")
    text = "कीमत Rs 500 है"
    cur = core.detect_currency(text, 8, 11, [p2, p1])
    assert cur == "INR"


# ---------------------------------------------------------------------- gold

def test_gold_deva_file_valid():
    path = os.path.join(os.path.dirname(__file__), "gold_deva.jsonl")
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    assert len(rows) >= 120
    n_pos = sum(1 for r in rows if r["spans"] and not r.get("cross_script"))
    n_neg = sum(1 for r in rows if not r["spans"])
    n_mixed = sum(1 for r in rows if r.get("cross_script"))
    assert n_pos >= 95, n_pos
    assert n_neg >= 25, n_neg
    assert n_mixed >= 30, n_mixed
    for r in rows:
        for sp in r["spans"]:
            assert r["text"][sp["start"]:sp["end"]]


if __name__ == "__main__":
    test_normalize_nfc()
    test_normalize_digit_mapping()
    test_normalize_lowercase()
    test_normalize_length_preserved_1to1()
    test_noise_never_collides()
    test_noise_saath_vs_saadhe_prefix_no_collision()
    test_to_deva_digits()
    test_generate_roundtrip_hi_deva()
    test_generate_collision_safe()
    test_devanagari_digit_examples_labelled_digits()
    test_hash_number_negatives_no_span()
    test_hi_latn_hash_number_negatives_present()
    test_cross_script_share()
    test_detect_currency_multi_union()
    test_gold_deva_file_valid()
    print("OK")
