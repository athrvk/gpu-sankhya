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

    ids = ["hi_latn", "hi_deva"]
    packs = [base.get_pack(i) for i in ids]
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


if __name__ == "__main__":
    test_generate_and_roundtrip()
    test_unk_pool_outside_every_charset()
    test_unk_noise_present_and_labelled_O()
    print("\nOK")
