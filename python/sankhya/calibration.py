"""Confidence calibration: isotonic (PAV) fit, breakpoint export, and the
piecewise-linear map applied at decode time.

The model's raw span confidence (mean over the span's characters of
max(p_B, p_I)) is a well-ordered but badly-scaled score: almost every span
scores above 0.9, so "0.95" says nothing about how often the value is
right. `python -m sankhya.calibrate` measures the empirical precision of
each raw score on fresh synthetic data and fits a monotone map from raw
score to precision with pool-adjacent-violators (PAV) isotonic regression;
the fitted map is exported as `src/data/calibration.json` and applied by
both runtimes, so a reported confidence of 0.8 means "about 80% of spans
scoring this have the right value".

This module holds the maths (fit + apply). `calibrate.py` is the CLI that
runs the model and writes the JSON; `src/calibration.ts` mirrors `apply`
1:1 for the JS runtime.
"""
from __future__ import annotations

import json
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "pav",
    "fit_isotonic",
    "fit_breakpoints",
    "apply",
    "load_calibration",
    "calibrate_confidence",
    "CALIBRATION_PATH",
]

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CALIBRATION_PATH = os.path.join(_REPO_ROOT, "src", "data", "calibration.json")

_CALIBRATION: Optional[dict] = None


def pav(y: Sequence[float], w: Sequence[float] | None = None) -> np.ndarray:
    """Pool-adjacent-violators: the least-squares non-decreasing fit to `y`.

    `y` must already be ordered by the (increasing) predictor. Returns an
    array the same length as `y`, non-decreasing, where each entry is the
    weighted mean of the block it was pooled into. Plain O(n) PAV: walk
    left to right keeping a stack of blocks, and whenever the new block's
    mean is not above the previous block's, merge them and re-check.
    """
    y = np.asarray(y, dtype=np.float64)
    if w is None:
        w = np.ones_like(y)
    w = np.asarray(w, dtype=np.float64)
    if y.shape != w.shape:
        raise ValueError("pav: y and w must have the same shape")
    n = len(y)
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    # stacks of block (sum of w*y, sum of w, number of original points)
    sums: List[float] = []
    weights: List[float] = []
    counts: List[int] = []
    for i in range(n):
        s, wt, c = y[i] * w[i], w[i], 1
        while sums and (sums[-1] / weights[-1]) >= (s / wt if wt else 0.0):
            s += sums.pop()
            wt += weights.pop()
            c += counts.pop()
        sums.append(s)
        weights.append(wt)
        counts.append(c)

    out = np.empty(n, dtype=np.float64)
    pos = 0
    for s, wt, c in zip(sums, weights, counts):
        out[pos:pos + c] = (s / wt) if wt else 0.0
        pos += c
    return out


def fit_isotonic(raw: Sequence[float], correct: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Isotonic regression of a 0/1 outcome on a raw score.

    Ties in `raw` are pooled first (so the fit is a function of the score,
    not of the sort order within a tie), then PAV runs over the distinct
    scores weighted by their counts. Returns (xs, ys): the distinct raw
    scores in increasing order and the fitted precision at each.
    """
    raw = np.asarray(raw, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    if len(raw) != len(correct):
        raise ValueError("fit_isotonic: raw and correct must be the same length")
    if len(raw) == 0:
        return np.zeros(0), np.zeros(0)
    order = np.argsort(raw, kind="mergesort")
    raw, correct = raw[order], correct[order]
    xs, idx, counts = np.unique(raw, return_index=True, return_counts=True)
    means = np.add.reduceat(correct, idx) / counts
    ys = pav(means, counts)
    return xs, ys


def fit_breakpoints(raw: Sequence[float], correct: Sequence[float],
                    max_points: int = 32) -> Tuple[List[float], List[float]]:
    """Fit the isotonic map and reduce it to at most `max_points` knots.

    The isotonic fit is a step function; between two consecutive distinct
    levels the true map is unknown, so the exported form is piecewise
    LINEAR through knots placed at the centre of each constant block (plus
    the two endpoints), which is smooth, still monotone, and cheap to
    evaluate in both runtimes. If there are more blocks than `max_points`,
    knots are dropped by keeping the ones that change the map most (the
    largest jumps in fitted precision), endpoints always kept.
    """
    xs, ys = fit_isotonic(raw, correct)
    if len(xs) == 0:
        return [0.0, 1.0], [0.0, 1.0]
    if len(xs) == 1:
        return [0.0, 1.0], [float(ys[0]), float(ys[0])]

    # one knot per constant block of the step function, at the block's
    # midpoint in x, plus the two endpoints of the observed range
    knots_x: List[float] = [float(xs[0])]
    knots_y: List[float] = [float(ys[0])]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and ys[j + 1] == ys[i]:
            j += 1
        mid = float((xs[i] + xs[j]) / 2.0)
        if mid > knots_x[-1]:
            knots_x.append(mid)
            knots_y.append(float(ys[i]))
        else:
            knots_y[-1] = float(ys[i])
        i = j + 1
    if float(xs[-1]) > knots_x[-1]:
        knots_x.append(float(xs[-1]))
        knots_y.append(float(ys[-1]))

    while len(knots_x) > max_points:
        # drop the interior knot whose removal changes the interpolated map
        # least (smallest deviation from the line through its neighbours)
        best_i, best_err = 1, float("inf")
        for k in range(1, len(knots_x) - 1):
            x0, y0 = knots_x[k - 1], knots_y[k - 1]
            x1, y1 = knots_x[k + 1], knots_y[k + 1]
            t = (knots_x[k] - x0) / (x1 - x0) if x1 > x0 else 0.0
            err = abs(knots_y[k] - (y0 + t * (y1 - y0)))
            if err < best_err:
                best_err, best_i = err, k
        knots_x.pop(best_i)
        knots_y.pop(best_i)

    # enforce exact monotonicity after rounding-free construction
    for k in range(1, len(knots_y)):
        if knots_y[k] < knots_y[k - 1]:
            knots_y[k] = knots_y[k - 1]
    return knots_x, knots_y


def apply(raw: float, knots_x: Sequence[float], knots_y: Sequence[float]) -> float:
    """Piecewise-linear interpolation of `raw` through the knots, clamped.

    Below the first knot the first knot's value is used, above the last the
    last knot's; the result is clamped to [0, 1]. Mirrored exactly by
    src/calibration.ts::calibrateConfidence.
    """
    n = len(knots_x)
    if n == 0:
        return min(1.0, max(0.0, float(raw)))
    if n == 1 or raw <= knots_x[0]:
        return min(1.0, max(0.0, float(knots_y[0])))
    if raw >= knots_x[n - 1]:
        return min(1.0, max(0.0, float(knots_y[n - 1])))
    lo, hi = 0, n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if knots_x[mid] <= raw:
            lo = mid
        else:
            hi = mid
    x0, x1 = knots_x[lo], knots_x[lo + 1]
    y0, y1 = knots_y[lo], knots_y[lo + 1]
    if x1 <= x0:
        return min(1.0, max(0.0, float(y1)))
    t = (raw - x0) / (x1 - x0)
    return min(1.0, max(0.0, float(y0 + t * (y1 - y0))))


def load_calibration(path: str = None) -> Optional[dict]:
    """Load (and cache) `src/data/calibration.json`. None if absent."""
    global _CALIBRATION
    if path is None and _CALIBRATION is not None:
        return _CALIBRATION
    p = path or CALIBRATION_PATH
    try:
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except FileNotFoundError:
        obj = None
    if path is None:
        _CALIBRATION = obj
    return obj


def calibrate_confidence(raw: Optional[float], calib: dict = None) -> Optional[float]:
    """Map a raw span confidence through the shipped calibration.

    Returns None for None (a decode with no bio_probs has no confidence)
    and the raw value unchanged when no calibration file is available.
    """
    if raw is None:
        return None
    if calib is None:
        calib = load_calibration()
    if not calib:
        return min(1.0, max(0.0, float(raw)))
    return apply(float(raw), calib["raw"], calib["calibrated"])
