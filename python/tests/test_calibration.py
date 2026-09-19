"""Confidence calibration: the PAV fit, the piecewise-linear map, and the
shipped src/data/calibration.json.

Regenerating the committed file means running the checkpoint over 100k
freshly generated examples (~2-3 minutes), which is far too slow for the
unit suite -- so the shipped file is pinned on its SCHEMA and its
MONOTONICITY, not on a re-fit. `python -m sankhya.calibrate` is the thing
that regenerates it.
"""
import json
import math

import numpy as np
import pytest

from sankhya.calibration import (
    CALIBRATION_PATH,
    apply,
    calibrate_confidence,
    fit_breakpoints,
    fit_isotonic,
    load_calibration,
    pav,
)

# The hand example the TS side pins against too (test/calibration.test.ts).
HAND_X = [0.5, 0.7, 0.9]
HAND_Y = [0.1, 0.6, 0.9]


def test_pav_hand_example():
    # classic PAV: the one violator (4 before 3) pools with its neighbour
    assert list(pav([1, 2, 4, 3, 5])) == [1.0, 2.0, 3.5, 3.5, 5.0]


def test_pav_leaves_a_monotone_sequence_alone():
    y = [0.1, 0.2, 0.2, 0.9]
    assert list(pav(y)) == y


def test_pav_pools_everything_when_strictly_decreasing():
    assert list(pav([3, 2, 1])) == [2.0, 2.0, 2.0]


def test_pav_is_weighted():
    # one point at 0 with weight 9 and one at 1 with weight 1 pool to 0.1
    assert list(pav([1.0, 0.0], [1.0, 9.0])) == pytest.approx([0.1, 0.1])


def test_pav_output_is_non_decreasing_on_random_input():
    rng = np.random.default_rng(0)
    for _ in range(20):
        y = rng.normal(size=50)
        out = pav(y)
        assert all(out[i] <= out[i + 1] + 1e-12 for i in range(len(out) - 1))
        # PAV is a projection: it preserves the total (unweighted) sum
        assert float(out.sum()) == pytest.approx(float(y.sum()))


def test_pav_empty():
    assert len(pav([])) == 0


def test_fit_isotonic_hand_example():
    xs, ys = fit_isotonic([0.1, 0.2, 0.3, 0.4, 0.5], [0, 1, 0, 1, 1])
    assert list(xs) == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5])
    assert list(ys) == pytest.approx([0.0, 0.5, 0.5, 1.0, 1.0])


def test_fit_isotonic_pools_ties_in_the_score():
    # the same raw score with both outcomes averages, it does not depend on
    # the order the two rows happen to arrive in
    xs, ys = fit_isotonic([0.8, 0.8, 0.9], [1, 0, 1])
    assert list(xs) == pytest.approx([0.8, 0.9])
    assert list(ys) == pytest.approx([0.5, 1.0])


def test_fit_breakpoints_is_monotone_and_bounded():
    rng = np.random.default_rng(3)
    raw = rng.uniform(0.5, 1.0, size=2000)
    correct = (rng.uniform(size=2000) < raw).astype(float)
    xs, ys = fit_breakpoints(raw, correct, max_points=32)
    assert 2 <= len(xs) <= 32
    assert len(xs) == len(ys)
    assert all(xs[i] < xs[i + 1] for i in range(len(xs) - 1))
    assert all(ys[i] <= ys[i + 1] for i in range(len(ys) - 1))
    assert all(0.0 <= y <= 1.0 for y in ys)


def test_fit_breakpoints_respects_max_points():
    rng = np.random.default_rng(4)
    raw = rng.uniform(0.5, 1.0, size=5000)
    correct = (rng.uniform(size=5000) < raw).astype(float)
    xs, ys = fit_breakpoints(raw, correct, max_points=8)
    assert len(xs) <= 8
    assert all(ys[i] <= ys[i + 1] for i in range(len(ys) - 1))


def test_apply_interpolates_the_hand_example():
    assert apply(0.5, HAND_X, HAND_Y) == pytest.approx(0.1)
    assert apply(0.6, HAND_X, HAND_Y) == pytest.approx(0.35)
    assert apply(0.7, HAND_X, HAND_Y) == pytest.approx(0.6)
    assert apply(0.8, HAND_X, HAND_Y) == pytest.approx(0.75)
    assert apply(0.9, HAND_X, HAND_Y) == pytest.approx(0.9)


def test_apply_clamps_outside_the_breakpoint_range():
    assert apply(0.0, HAND_X, HAND_Y) == pytest.approx(0.1)
    assert apply(0.4, HAND_X, HAND_Y) == pytest.approx(0.1)
    assert apply(1.0, HAND_X, HAND_Y) == pytest.approx(0.9)
    # values outside [0, 1] are clamped into it
    assert apply(0.5, [0.0, 1.0], [-2.0, 5.0]) == pytest.approx(1.0)
    assert apply(0.0, [0.0, 1.0], [-2.0, 5.0]) == pytest.approx(0.0)


def test_apply_with_no_breakpoints_is_the_identity():
    assert apply(0.42, [], []) == pytest.approx(0.42)


def test_apply_is_non_decreasing_over_the_shipped_map():
    calib = load_calibration()
    prev = -1.0
    for i in range(0, 1001):
        v = apply(i / 1000.0, calib["raw"], calib["calibrated"])
        assert v >= prev - 1e-12
        prev = v


# --- the committed src/data/calibration.json ---

def test_shipped_calibration_schema():
    with open(CALIBRATION_PATH, "r", encoding="utf-8") as f:
        obj = json.load(f)
    assert obj["version"] == 1
    assert isinstance(obj["raw"], list) and isinstance(obj["calibrated"], list)
    assert len(obj["raw"]) == len(obj["calibrated"])
    assert 2 <= len(obj["raw"]) <= 32
    assert isinstance(obj["n_samples"], int) and obj["n_samples"] > 0
    assert isinstance(obj["seed"], int)
    assert all(isinstance(x, (int, float)) and math.isfinite(x) for x in obj["raw"])
    assert all(0.0 <= y <= 1.0 for y in obj["calibrated"])


def test_shipped_calibration_is_monotone():
    with open(CALIBRATION_PATH, "r", encoding="utf-8") as f:
        obj = json.load(f)
    xs, ys = obj["raw"], obj["calibrated"]
    assert all(xs[i] < xs[i + 1] for i in range(len(xs) - 1)), "raw breakpoints must be increasing"
    assert all(ys[i] <= ys[i + 1] for i in range(len(ys) - 1)), "calibrated values must be non-decreasing"


def test_calibrate_confidence_passes_none_through():
    assert calibrate_confidence(None) is None


def test_calibrate_confidence_uses_the_shipped_map():
    calib = load_calibration()
    for raw in (0.0, 0.55, 0.85, 0.95, 1.0):
        assert calibrate_confidence(raw) == pytest.approx(apply(raw, calib["raw"], calib["calibrated"]))
