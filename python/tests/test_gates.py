"""Release gates on the SHIPPED int8 weights (models/default/sankhya.weights.int8.json).

These pin the promises the strict ("verified") tier makes to users:
  a) no false positive on any gold negative example,
  b) every verified prediction on a gold span has the right value,
  c) the strict tier still covers most gold spans.

One int8 numpy pass over the 748-example gold set (~5 s).
"""
import functools
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sankhya.eval_gold import run_eval

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, "..", "..", "models", "default", "sankhya.weights.int8.json")
GOLD = [os.path.join(HERE, f) for f in ("gold.jsonl", "gold_deva.jsonl", "gold_mr.jsonl", "gold_gu.jsonl")]

STRICT_COVERAGE_FLOOR = 0.85


@functools.lru_cache(maxsize=1)
def _metrics():
    if not os.path.exists(WEIGHTS):
        pytest.skip(f"shipped int8 weights not found: {WEIGHTS}")
    out = run_eval(GOLD, weights_json=WEIGHTS, int8=True, quiet=True)
    return out["combined"]


def test_no_false_positives_on_gold_negatives():
    negatives = _metrics()["negatives"]
    assert negatives["examples"] > 0
    assert negatives["false_positives"] == 0, (
        f"{negatives['false_positives']} of {negatives['examples']} negative examples "
        f"produced a span"
    )
    strict_neg = _metrics()["strict"]["negatives"]
    assert strict_neg["false_positives"] == 0


def test_strict_value_accuracy_is_perfect():
    strict = _metrics()["strict"]
    assert strict["covered"] > 0
    bad = strict["wrong_value_examples"]
    assert strict["value_acc"] == 1.0, (
        "verified predictions with a wrong value:\n"
        + "\n".join(f"  {b['gold_text']!r} gold={b['gold_span']['value']} {b['pred']}" for b in bad)
    )


def test_strict_coverage_floor():
    strict = _metrics()["strict"]
    assert strict["coverage"] >= STRICT_COVERAGE_FLOOR, (
        f"strict coverage {strict['coverage']:.4f} "
        f"({strict['covered']}/{strict['gold_spans']}) below {STRICT_COVERAGE_FLOOR}"
    )
