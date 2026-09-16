import { test } from "node:test";
import assert from "node:assert/strict";
import { parse } from "../src/index.ts";

// These exercise the bundled dev weights end-to-end. The current training
// run is early, so some spec examples may not decode/evaluate correctly yet;
// those are marked `expected-failure` with a comment rather than deleted, so
// this file stays a live record of model quality as training improves.

test("sava lakh", () => {
  const r = parse("sava lakh");
  assert.equal(r.length, 1);
  assert.equal(r[0].value, 125000);
});

test("dedh crore", () => {
  const r = parse("dedh crore");
  assert.equal(r.length, 1);
  assert.equal(r[0].value, 15000000);
});

test("paune do lakh", () => {
  const r = parse("paune do lakh");
  assert.equal(r.length, 1);
  assert.equal(r[0].value, 175000);
});

test("2.5L", () => {
  const r = parse("2.5L");
  assert.equal(r.length, 1);
  assert.equal(r[0].value, 250000);
});

test("20k logon ne attend kiya", () => {
  const r = parse("20k logon ne attend kiya");
  assert.ok(r.length >= 1);
  assert.equal(r[0].value, 20000);
});

test("2-3 lakh", () => {
  // The dev checkpoint tags '-' as O in isolation; decode.ts's class-repair
  // step recovers RANGE from context (digit run on both sides), fixing this.
  const r = parse("2-3 lakh");
  assert.equal(r.length, 1);
  assert.deepEqual(r[0].range, [200000, 300000]);
});

test("das hazaar crore", () => {
  // Similarly recovered by class-repair's letter-run majority vote/fallback.
  const r = parse("das hazaar crore");
  assert.equal(r.length, 1);
  assert.equal(r[0].value, 1e11);
});
