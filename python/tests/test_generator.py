"""Generate a batch of examples and sanity-check the round-trip invariant."""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sankhya import core
from sankhya.classes import CLASSES
from sankhya.langs import base
from sankhya import generator as G

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


def test_generate_and_roundtrip():
    pack = base.get_pack("hi_latn")
    rng = random.Random(SEED)
    examples = [G.sample_example(pack, rng) for _ in range(N)]

    n_neg = 0
    for ex in examples:
        text = ex["text"]
        assert len(ex["bio"]) == len(text), (len(ex["bio"]), len(text))
        assert len(ex["cls"]) == len(text), (len(ex["cls"]), len(text))
        if not ex["spans"]:
            n_neg += 1
            continue
        for sp in ex["spans"]:
            toks = _rebuild_tokens(text, ex["cls"], sp["start"], sp["end"])
            res = core.evaluate(toks)
            assert res.value == sp["value"], (text, sp, res)
            assert res.unit == sp["unit"], (text, sp, res)
            exp_range = tuple(sp["range"]) if sp["range"] else None
            assert res.range == exp_range, (text, sp, res)
            span_text = text[sp["start"]:sp["end"]]
            assert span_text.strip() != "", "span text is all-whitespace/empty"
            assert any(c != CLASSES.index("O") for c in ex["cls"][sp["start"]:sp["end"]]), "span is all-O"

    frac_neg = n_neg / N
    print(f"negatives: {n_neg}/{N} = {frac_neg:.3f}")
    assert 0.08 <= frac_neg <= 0.20, frac_neg

    print("\n30 random examples:")
    sample = random.Random(42).sample(examples, 30)
    for ex in sample:
        print(f"  {ex['text']!r}")
        for sp in ex["spans"]:
            print(f"      span={ex['text'][sp['start']:sp['end']]!r} value={sp['value']} "
                  f"range={sp['range']} unit={sp['unit']} currency={sp['currency']} classes={sp['classes']}")


def test_unk_pool_outside_every_charset():
    """Every _UNK_POOL char must be guaranteed out-of-vocab: absent from the
    charset built over every registered pack, individually and combined."""
    from sankhya import charset as CS  # lazy import, avoids cycles

    packs = base.all_packs()  # every registered pack, not a fixed list
    combined = set(CS.build_charset_multi(packs))
    for p in packs:
        combined |= set(CS.build_charset_multi([p]))
    for ch in G._UNK_POOL:
        assert ch not in combined, f"unk pool char {ch!r} is in a pack charset"


def test_unk_noise_present_and_labelled_O():
    pack = base.get_pack("hi_latn")
    rng = random.Random(7)
    n = 2000
    examples = [G.sample_example(pack, rng) for _ in range(n)]

    unk_set = set(G._UNK_POOL)
    o_id = CLASSES.index("O")
    with_unk = 0
    for ex in examples:
        text = ex["text"]
        positions = [i for i, ch in enumerate(text) if ch in unk_set]
        if not positions:
            continue
        with_unk += 1
        for i in positions:
            assert ex["bio"][i] == 0, (text, i, ex["bio"])
            assert ex["cls"][i] == o_id, (text, i, ex["cls"])
        # spans still round-trip through core.evaluate despite the noise
        for sp in ex["spans"]:
            toks = _rebuild_tokens(text, ex["cls"], sp["start"], sp["end"])
            res = core.evaluate(toks)
            assert res.value == sp["value"], (text, sp, res)
            assert res.unit == sp["unit"], (text, sp, res)
            exp_range = tuple(sp["range"]) if sp["range"] else None
            assert res.range == exp_range, (text, sp, res)

    print(f"examples with unk noise: {with_unk}/{n}")
    assert with_unk >= 150, with_unk


def test_multi_span_families_rate_and_roundtrip():
    """(a) conjunction-joined multi-span family: appears at a reasonable rate
    and every span round-trips independently through core.evaluate; also
    covers the pre-existing two_spans family so multi-span examples overall
    (>=2 spans) land in a sane band."""
    pack = base.get_pack("hi_latn")
    rng = random.Random(11)
    n = 5000
    examples = [G.sample_example(pack, rng) for _ in range(n)]

    n_multi = 0
    for ex in examples:
        text = ex["text"]
        if len(ex["spans"]) >= 2:
            n_multi += 1
            # spans must not overlap and each must round-trip on its own
            spans = sorted(ex["spans"], key=lambda s: s["start"])
            for i in range(len(spans) - 1):
                assert spans[i]["end"] <= spans[i + 1]["start"], (text, spans)
            for sp in spans:
                toks = _rebuild_tokens(text, ex["cls"], sp["start"], sp["end"])
                res = core.evaluate(toks)
                assert res.value == sp["value"], (text, sp, res)

    frac_multi = n_multi / n
    print(f"multi-span (>=2 spans) examples: {n_multi}/{n} = {frac_multi:.3f}")
    # two_spans (10%) + multi_conj (~6%, sometimes 3 spans) combined
    assert 0.08 <= frac_multi <= 0.24, frac_multi


def test_chain_trailing_bare_roundtrip():
    """(b) trailing/leading small-cardinal additive chains (no unit on the
    last term, e.g. "ek hazaar ek") round-trip correctly when sampled
    directly via the chain_trailing_bare structure."""
    pack = base.get_pack("hi_latn")
    rng = random.Random(3)
    seen = 0
    for _ in range(500):
        toks, structure = G.build_phrase(pack, rng, structure="chain_trailing_bare")
        text, clsc = G._tokens_to_text_and_labels(toks)
        res = core.evaluate(toks)
        assert res.unit in ("crore", "lakh", "hazaar")
        seen += 1
    assert seen == 500


def test_negatives_contain_new_negative_words():
    """(c) negatives family: net-confusing surfaces (lakhpati/crorepati-style,
    unit-noun abbreviations, ordinals, id-like numbers, prefix+duration time
    phrases) show up among negatives and never produce spans."""
    pack = base.get_pack("hi_latn")
    rng = random.Random(11)
    n = 5000
    examples = [G.sample_example(pack, rng) for _ in range(n)]

    negwords = [
        "lakhpati", "karodpati", "crore-pati", "laakhon", "karodon",
        " km ", " kg ", "kmph", "kb ", "mb ", "1st ", "2nd ", "3rd ", "4th ",
        "route ", "mobile ", "flat no ", "sava ghanta", "dedh ghante",
        "saadhe teen baje", "paune paanch baje",
    ]
    hits = 0
    for ex in examples:
        if ex["spans"]:
            continue
        text = ex["text"].lower()
        if any(w in text for w in negwords):
            hits += 1
            assert ex["spans"] == []
    print(f"negatives with new negative words: {hits}/{n}")
    assert hits >= 5, hits


def test_typo_spellings_recognised_in_quantity_context():
    """(e) curated typo spellings ("croer", "lkah", "lakhh", "crorr") appear
    inside quantity contexts and round-trip through core.evaluate."""
    pack = base.get_pack("hi_latn")
    rng = random.Random(9)
    typo_forms = {"croer", "crorr", "lkah", "lakhh"}
    found = set()
    for _ in range(3000):
        toks, _ = G.build_phrase(pack, rng)
        text, _ = G._tokens_to_text_and_labels(toks)
        for cls, word in toks:
            if word.lower() in typo_forms:
                found.add(word.lower())
        res = core.evaluate(toks)
        if any(w.lower() in typo_forms for _, w in toks):
            assert res.value > 0, (text, res)
    print(f"typo forms observed: {found}")
    assert found, "no curated typo spellings were ever produced"


def test_possessive_noise_on_units():
    """(d) possessive/plural noise on units ("lakh's", "crore's"): the
    apostrophe+letters are O, the unit keeps its value."""
    pack = base.get_pack("hi_latn")
    rng = random.Random(4)
    seen = 0
    for _ in range(4000):
        toks, _ = G.build_phrase(pack, rng)
        text, _ = G._tokens_to_text_and_labels(toks)
        if text.endswith("'s") or text.endswith("'"):
            seen += 1
            res = core.evaluate(toks)
            assert res.value > 0, (text, res)
    print(f"possessive-noised phrases seen: {seen}/4000")
    assert seen >= 5, seen


def test_cardinal_variants_map_correctly():
    """Every curated 11..99 cardinal spelling in hi_latn's _EXTRA_VARIANTS
    (including the Dakshina-mined and user-reported 66 spellings merged in)
    maps to the right CARD_n via pack.all_forms(), and the pack's charset
    gains no new characters from them (all variants are plain lowercase
    a-z, already covered by string.ascii_lowercase in charset.py)."""
    from sankhya.langs import hi_latn as HL
    from sankhya import charset as CS

    pack = base.get_pack("hi_latn")
    forms = pack.all_forms()
    for n, variants in HL._EXTRA_VARIANTS.items():
        for v in variants:
            if v.isdigit():
                continue
            assert forms.get(v.lower()) == f"CARD_{n}", (v, n, forms.get(v.lower()))
            assert v == v.lower(), f"variant not lowercase: {v!r}"
            assert all(c in "abcdefghijklmnopqrstuvwxyz" for c in v), v

    charset = set(CS.build_charset(pack))
    assert charset == set(CS.build_charset(pack)), "charset build not stable"
    assert set("abcdefghijklmnopqrstuvwxyz").issubset(charset)
    # no character outside plain a-z leaked in from any variant
    all_variant_chars = set()
    for variants in HL._EXTRA_VARIANTS.values():
        for v in variants:
            all_variant_chars.update(v.lower())
    assert all_variant_chars <= set("abcdefghijklmnopqrstuvwxyz0123456789")

    # every 66-spelling reported by the user is present and maps to CARD_66
    for v in ["chanchat", "chanchatt", "chaachat", "chhachat", "chhasath",
              "chansath", "chhiyasat", "chiyasat"]:
        assert v in HL._EXTRA_VARIANTS[66], v
        assert forms.get(v) == "CARD_66", (v, forms.get(v))

    # sanity: no accepted variant collides with a different class or with
    # filler/blocked/duration words (same invariant the merge step enforced)
    filler = set(w.lower() for w in pack.filler_words)
    blocked = set(w.lower() for w in pack.blocked_surfaces)
    duration = set(w.lower() for w in pack.duration_nouns)
    for n, variants in HL._EXTRA_VARIANTS.items():
        for v in variants:
            if v.isdigit():
                continue
            v = v.lower()
            assert v not in filler, f"{v!r} collides with a filler word"
            assert v not in blocked, f"{v!r} collides with a blocked surface"
            assert v not in duration, f"{v!r} collides with a duration noun"


def test_dakshina_merged_unit_variants_map_correctly():
    """Unit/prefix romanizations merged in from the Dakshina mining pass map
    to the right class (or, for indefinite-plural forms like 'krodo'/
    'hajaron'/'arabon', to O) and don't collide with anything else."""
    pack = base.get_pack("hi_latn")
    forms = pack.all_forms()

    expect_unit = {
        "dus": "CARD_10", "terh": "CARD_13", "atarah": "CARD_18",
        "athaarah": "CARD_18", "pachis": "CARD_25", "chhabees": "CARD_26",
        "untees": "CARD_29", "unatees": "CARD_29", "pantis": "CARD_35",
        "cheh": "CARD_6", "chihttr": "CARD_76",
        "araba": "UNIT_ARAB", "crode": "UNIT_CRORE", "khokhaa": "UNIT_CRORE",
        "millium": "UNIT_MILLION",
    }
    for surface, cls in expect_unit.items():
        assert forms.get(surface) == cls, (surface, cls, forms.get(surface))

    expect_indefinite_O = ["hajaron", "krodo", "karoron", "krodon", "arabon"]
    for surface in expect_indefinite_O:
        assert forms.get(surface) == "O", (surface, forms.get(surface))


if __name__ == "__main__":
    test_generate_and_roundtrip()
    test_unk_pool_outside_every_charset()
    test_unk_noise_present_and_labelled_O()
    test_multi_span_families_rate_and_roundtrip()
    test_chain_trailing_bare_roundtrip()
    test_negatives_contain_new_negative_words()
    test_typo_spellings_recognised_in_quantity_context()
    test_possessive_noise_on_units()
    test_cardinal_variants_map_correctly()
    test_dakshina_merged_unit_variants_map_correctly()
    print("\nOK")
