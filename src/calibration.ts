// Confidence calibration: maps the model's RAW span confidence (mean over
// the span's characters of max(p_B, p_I)) to a calibrated confidence that
// reads as a precision -- "spans reported at 0.8 have the right value about
// 80% of the time". The breakpoints in ./data/calibration.json come from an
// isotonic (pool-adjacent-violators) fit on fresh synthetic data, run by
// `python -m sankhya.calibrate`; this file mirrors
// python/sankhya/calibration.py::apply 1:1.
import calibrationJson from "./data/calibration.json" with { type: "json" };

export interface CalibrationJson {
  version: number;
  /** raw-confidence breakpoints, strictly increasing */
  raw: number[];
  /** calibrated value at each breakpoint, non-decreasing */
  calibrated: number[];
  n_samples?: number;
  seed?: number;
}

export const CALIBRATION = calibrationJson as unknown as CalibrationJson;

/** Piecewise-linear interpolation of `raw` through (xs, ys), clamped to
 * [0, 1]. Outside the breakpoint range the nearest endpoint's value is
 * used. Exported separately from `calibrateConfidence` so it can be tested
 * against the same hand example as the Python side. */
export function interpolate(raw: number, xs: number[], ys: number[]): number {
  const n = xs.length;
  if (n === 0) return Math.min(1, Math.max(0, raw));
  if (n === 1 || raw <= xs[0]) return Math.min(1, Math.max(0, ys[0]));
  if (raw >= xs[n - 1]) return Math.min(1, Math.max(0, ys[n - 1]));
  let lo = 0;
  let hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] <= raw) lo = mid;
    else hi = mid;
  }
  const x0 = xs[lo];
  const x1 = xs[lo + 1];
  const y0 = ys[lo];
  const y1 = ys[lo + 1];
  if (x1 <= x0) return Math.min(1, Math.max(0, y1));
  const t = (raw - x0) / (x1 - x0);
  return Math.min(1, Math.max(0, y0 + t * (y1 - y0)));
}

/** Calibrated confidence for a raw span confidence, using the bundled map. */
export function calibrateConfidence(raw: number): number {
  return interpolate(raw, CALIBRATION.raw, CALIBRATION.calibrated);
}
