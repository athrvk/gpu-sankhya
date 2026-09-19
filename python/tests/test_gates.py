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
WILD_LANGS = ("hi_latn", "hi_deva", "mr_deva", "gu_gujr")
WILD_GOLD = [os.path.join(HERE, f"gold_wild_{l}.jsonl") for l in WILD_LANGS]

STRICT_COVERAGE_FLOOR = 0.85

# Strict ("verified") false-positive rate on the REAL-TEXT gold negatives:
# the share of hand-labelled real sentences with no amount in them that still
# produce a VERIFIED span. The shipped int8 weights sit at 4/240 = 0.0167
# with the R11-R18 gates in place (0.0875 before them, see
# python/data_wild/REPORT.md). This is deliberately a real gate, not a
# formality: real-text precision is what the strict tier is bought on, and
# nothing may regress it without a retrain saying so.
WILD_STRICT_FP_CEILING = 0.02
# Overall (non-strict) real-text FP rate, currently 10/240 = 0.0417, pinned
# a little looser so ordinary decoder work has room.
WILD_FP_CEILING = 0.05


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


@functools.lru_cache(maxsize=1)
def _wild_metrics():
    missing = [p for p in WILD_GOLD if not os.path.exists(p)]
    if missing or not os.path.exists(WEIGHTS):
        pytest.skip("wild gold files or shipped int8 weights not found")
    out = run_eval(WILD_GOLD, weights_json=WEIGHTS, int8=True, quiet=True)
    return out["combined"]


def test_strict_false_positive_rate_on_real_text():
    """R11-R18 exist for this number -- see python/data_wild/REPORT.md."""
    strict_neg = _wild_metrics()["strict"]["negatives"]
    assert strict_neg["examples"] > 0
    assert strict_neg["fp_rate"] <= WILD_STRICT_FP_CEILING, (
        f"strict FP rate on the wild gold {strict_neg['fp_rate']:.4f} "
        f"({strict_neg['false_positives']}/{strict_neg['examples']}) "
        f"above {WILD_STRICT_FP_CEILING}"
    )


def test_false_positive_rate_on_real_text():
    negatives = _wild_metrics()["negatives"]
    assert negatives["examples"] > 0
    assert negatives["fp_rate"] <= WILD_FP_CEILING, (
        f"FP rate on the wild gold {negatives['fp_rate']:.4f} "
        f"({negatives['false_positives']}/{negatives['examples']}) "
        f"above {WILD_FP_CEILING}"
    )


def test_strict_value_accuracy_is_perfect_on_real_text():
    """The strict tier's central promise -- "verified implies the right
    value" -- has to hold on text nobody wrote for this repo, too."""
    strict = _wild_metrics()["strict"]
    assert strict["covered"] > 0
    bad = strict["wrong_value_examples"]
    assert strict["value_acc"] == 1.0, (
        "verified predictions with a wrong value on real text:\n"
        + "\n".join(f"  {b['gold_text']!r} gold={b['gold_span']['value']} {b['pred']}" for b in bad)
    )
