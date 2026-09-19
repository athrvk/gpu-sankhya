import { test } from "node:test";
import assert from "node:assert/strict";
import { interpolate, calibrateConfidence, CALIBRATION } from "../src/calibration.ts";
import { parse } from "../src/index.ts";

// The same hand example python/tests/test_calibration.py pins against, so
// the two interpolations are demonstrably the same function.
const HAND_X = [0.5, 0.7, 0.9];
const HAND_Y = [0.1, 0.6, 0.9];

const close = (a: number, b: number, eps = 1e-12) =>
  assert.ok(Math.abs(a - b) <= eps, `${a} !== ${b}`);

test("interpolate() matches the hand example (same as python apply())", () => {
  close(interpolate(0.5, HAND_X, HAND_Y), 0.1);
  close(interpolate(0.6, HAND_X, HAND_Y), 0.35);
  close(interpolate(0.7, HAND_X, HAND_Y), 0.6);
  close(interpolate(0.8, HAND_X, HAND_Y), 0.75);
  close(interpolate(0.9, HAND_X, HAND_Y), 0.9);
});

test("interpolate() clamps outside the breakpoint range and to [0, 1]", () => {
  close(interpolate(0.0, HAND_X, HAND_Y), 0.1);
  close(interpolate(0.4, HAND_X, HAND_Y), 0.1);
  close(interpolate(1.0, HAND_X, HAND_Y), 0.9);
  close(interpolate(0.5, [0, 1], [-2, 5]), 1);
  close(interpolate(0.0, [0, 1], [-2, 5]), 0);
});

test("interpolate() with no breakpoints is the (clamped) identity", () => {
  close(interpolate(0.42, [], []), 0.42);
  close(interpolate(-1, [], []), 0);
});

test("the bundled calibration map is well-formed and monotone", () => {
  assert.equal(CALIBRATION.version, 1);
  assert.equal(CALIBRATION.raw.length, CALIBRATION.calibrated.length);
  assert.ok(CALIBRATION.raw.length >= 2 && CALIBRATION.raw.length <= 32);
  for (let i = 1; i < CALIBRATION.raw.length; i++) {
    assert.ok(CALIBRATION.raw[i] > CALIBRATION.raw[i - 1], "raw breakpoints must increase");
    assert.ok(CALIBRATION.calibrated[i] >= CALIBRATION.calibrated[i - 1], "calibrated must not decrease");
  }
  let prev = -1;
  for (let i = 0; i <= 1000; i++) {
    const v = calibrateConfidence(i / 1000);
    assert.ok(v >= prev - 1e-12 && v >= 0 && v <= 1);
    prev = v;
  }
});

test("parse() reports both a calibrated confidence and the raw score", () => {
  const [s] = parse("sava lakh");
  assert.ok(s, "expected a span for 'sava lakh'");
  assert.equal(typeof s.rawConfidence, "number");
  assert.equal(typeof s.confidence, "number");
  close(s.confidence, calibrateConfidence(s.rawConfidence));
  assert.ok(s.confidence >= 0 && s.confidence <= 1);
});
