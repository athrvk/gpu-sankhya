"""Non-gating checks on the REAL-TEXT gold files (`gold_wild_<lang>.jsonl`).

These lines are sampled from openly licensed corpora (see
`python/data_wild/REPORT.md`) and hand-labelled, unlike the rest of
`tests/gold*.jsonl`, which we wrote ourselves. They are deliberately NOT part
of `tests/test_gates.py`: real text is where the model is weakest, so pinning
a floor on it would either be trivially loose or block unrelated work. What
this file asserts is only that the files stay well-formed and that the shipped
weights still evaluate over them -- the numbers themselves live in REPORT.md.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, "..", "..", "models", "default", "sankhya.weights.int8.json")
LANGS = ("hi_latn", "hi_deva", "mr_deva", "gu_gujr")
WILD_GOLD = [os.path.join(HERE, f"gold_wild_{l}.jsonl") for l in LANGS]

_PRESENT = [p for p in WILD_GOLD if os.path.exists(p)]


def _load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.mark.parametrize("path", WILD_GOLD, ids=LANGS)
def test_wild_gold_is_well_formed(path):
    if not os.path.exists(path):
        pytest.skip(f"{os.path.basename(path)} not present")
    rows = _load(path)
    assert rows, "empty gold file"
    lang = os.path.basename(path)[len("gold_wild_"):-len(".jsonl")]
    for ex in rows:
        assert ex["lang"] == lang
        assert isinstance(ex["text"], str) and ex["text"].strip()
        for sp in ex["spans"]:
            s, e = sp["start"], sp["end"]
            assert 0 <= s < e <= len(ex["text"]), (ex["text"], sp)
            assert isinstance(sp["value"], (int, float))
            if sp.get("range"):
                assert len(sp["range"]) == 2 and sp["range"][0] <= sp["range"][1]
    assert any(ex["spans"] for ex in rows), "no positives"
    assert any(not ex["spans"] for ex in rows), "no negatives"


@pytest.mark.parametrize("path", WILD_GOLD, ids=LANGS)
def test_wild_gold_values_agree_with_the_arithmetic_core(path):
    """Every labelled value must be reproducible by `core.evaluate` on the
    token classes the pack lexicon assigns to the span -- the same check the
    Marathi/Gujarati gold authors ran, so a typo'd label cannot sit in the
    file unnoticed."""
    if not os.path.exists(path):
        pytest.skip(f"{os.path.basename(path)} not present")
    from sankhya import core
    from sankhya.wild import tokens_from_lexicon

    bad = []
    for ex in _load(path):
        for sp in ex["spans"]:
            toks = tokens_from_lexicon(ex["text"][sp["start"]:sp["end"]], ex["lang"])
            if toks is None:
                continue  # span the lexicon cannot tokenize on its own; label stands
            res = core.evaluate(toks)
            if res.value != sp["value"] or (
                sp.get("range") and (not res.range or list(res.range) != list(sp["range"]))
            ):
                bad.append((ex["text"][sp["start"]:sp["end"]], sp, res.value, res.range))
    assert not bad, "gold values the core does not reproduce:\n" + "\n".join(map(str, bad))


@pytest.mark.skipif(not _PRESENT, reason="no wild gold files present")
def test_shipped_weights_evaluate_over_wild_gold():
    if not os.path.exists(WEIGHTS):
        pytest.skip("shipped int8 weights not found")
    from sankhya.eval_gold import run_eval

    out = run_eval(_PRESENT, weights_json=WEIGHTS, int8=True, quiet=True)
    combined = out["combined"]
    assert combined["spans"] > 0
    # not a quality gate -- just that the pipeline produced a full metrics dict
    assert 0.0 <= combined["value_acc"] <= 1.0
    assert set(combined["strict"]) >= {"coverage", "value_acc", "negatives"}
