import { test } from "node:test";
import assert from "node:assert/strict";
import { viterbi, argmaxRow } from "../src/infer-cpu.ts";
import { loadWeights } from "../src/weights.ts";
import type { CrfParams } from "../src/weights.ts";
import type { WeightsJsonV2Float } from "../src/types.ts";
import { createParser } from "../src/index.ts";

function zeroCrf(): CrfParams {
  return { trans: new Float32Array(9), start: new Float32Array(3), end: new Float32Array(3) };
}

function argmaxPath(bioLogits: Float32Array, L: number): number[] {
  const out: number[] = [];
  for (let t = 0; t < L; t++) out.push(argmaxRow(bioLogits, t * 3, 3));
  return out;
}

test("viterbi with all-zero trans/start/end matches per-position argmax", () => {
  const L = 40;
  const logits = new Float32Array(L * 3);
  for (let i = 0; i < logits.length; i++) logits[i] = Math.random() * 4 - 2;
  const crf = zeroCrf();
  const out = new Int32Array(L);
  viterbi(logits, L, crf, out);
  const want = argmaxPath(logits, L);
  assert.deepEqual(Array.from(out.subarray(0, L)), want);
});

test("viterbi: exact tie at a position picks the FIRST (lowest) index, matching argmax's tie rule", () => {
  const L = 3;
  const logits = new Float32Array(L * 3);
  // position 0: all three labels tied -> index 0 wins.
  logits.set([1, 1, 1], 0);
  // position 1: labels 1 and 2 tied, 0 lower -> index 1 wins.
  logits.set([0, 5, 5], 3);
  // position 2: normal case.
  logits.set([0, 0, 9], 6);
  const crf = zeroCrf();
  const out = new Int32Array(L);
  viterbi(logits, L, crf, out);
  assert.deepEqual(Array.from(out.subarray(0, L)), [0, 1, 2]);
  assert.deepEqual(argmaxPath(logits, L), [0, 1, 2]);
});

test("viterbi: a strongly forbidden O->I transition is avoided on a case where argmax would take it", () => {
  const L = 200;
  const logits = new Float32Array(L * 3);
  for (let i = 0; i < logits.length; i++) logits[i] = Math.random() * 4 - 2;
  // Force a deterministic O->I hop at t=10->11 in the argmax path: make O
  // clearly best at t=10 and I clearly best at t=11.
  logits.set([10, 0, 0], 10 * 3);
  logits.set([0, 0, 10], 11 * 3);

  const argmax = argmaxPath(logits, L);
  assert.equal(argmax[10], 0);
  assert.equal(argmax[11], 2); // O -> I, confirming the setup produces the transition we're forbidding

  const crf: CrfParams = { trans: new Float32Array(9), start: new Float32Array(3), end: new Float32Array(3) };
  crf.trans[0 * 3 + 2] = -1e4; // O -> I forbidden

  const out = new Int32Array(L);
  viterbi(logits, L, crf, out);
  for (let t = 0; t + 1 < L; t++) {
    assert.ok(!(out[t] === 0 && out[t + 1] === 2), `found forbidden O->I at t=${t}`);
  }
});

test("viterbi: hand-computed 4-position example with non-trivial trans/start/end differs from argmax, verified by brute force over all 81 paths", () => {
  const L = 4;
  // emissions favor O at every position under plain argmax.
  const logits = new Float32Array([
    5, 4.9, 0, // t0: O slightly best
    5, 4.9, 0, // t1
    5, 4.9, 0, // t2
    5, 4.9, 0, // t3
  ]);
  const trans = new Float32Array([
    // O->O, O->B, O->I
    -10, -10, -10,
    // B->O, B->B, B->I
    0, -10, 5,
    // I->O, I->B, I->I
    0, -10, 5,
  ]);
  const start = new Float32Array([-10, 5, -10]); // strongly prefer starting at B
  const end = new Float32Array([0, 0, 5]); // prefer ending at I
  const crf: CrfParams = { trans, start, end };

  // Brute force over all 3^4 = 81 label sequences using the contract's
  // score formula (float32 accumulation order matches, but plain JS number
  // arithmetic here is fine for finding the best score; we then check
  // viterbi's result matches this externally-computed best path).
  let bestScore = -Infinity;
  let bestPath: number[] = [];
  for (let a = 0; a < 3; a++) {
    for (let b = 0; b < 3; b++) {
      for (let c = 0; c < 3; c++) {
        for (let d = 0; d < 3; d++) {
          const path = [a, b, c, d];
          let score = start[path[0]];
          for (let t = 0; t < L; t++) score += logits[t * 3 + path[t]];
          for (let t = 1; t < L; t++) score += trans[path[t - 1] * 3 + path[t]];
          score += end[path[L - 1]];
          if (score > bestScore) {
            bestScore = score;
            bestPath = path;
          }
        }
      }
    }
  }

  const out = new Int32Array(L);
  viterbi(logits, L, crf, out);
  assert.deepEqual(Array.from(out.subarray(0, L)), bestPath);

  // Confirm this really differs from plain per-position argmax (otherwise
  // this test would not be exercising the CRF path meaningfully).
  const argmax = argmaxPath(logits, L);
  assert.notDeepEqual(bestPath, argmax);
});

test("viterbi: L=0 is a no-op (empty path)", () => {
  const out = new Int32Array(0);
  viterbi(new Float32Array(0), 0, zeroCrf(), out);
  assert.equal(out.length, 0);
});

function randArr(n: number): number[] {
  return Array.from({ length: n }, () => Math.random() * 2 - 1);
}

function makeV2WeightsWithCrf(includeCrf: boolean): WeightsJsonV2Float {
  const E = 4, C = 6, V = 10, NCLS = 3;
  const base: WeightsJsonV2Float = {
    version: 2,
    charset: Array.from({ length: V }, (_, i) => String.fromCharCode(97 + i)),
    classes: Array.from({ length: NCLS }, (_, i) => `CLS_${i}`),
    embed_dim: E,
    channels: C,
    embed: { shape: [V, E], data: randArr(V * E) },
    conv: [{ k: 3, dilation: 1, residual: false, w: { shape: [C, E, 3], data: randArr(C * E * 3) }, b: { shape: [C], data: randArr(C) } }],
    bio: { w: { shape: [3, C], data: randArr(3 * C) }, b: { shape: [3], data: randArr(3) } },
    cls: { w: { shape: [NCLS, C], data: randArr(NCLS * C) }, b: { shape: [NCLS], data: randArr(NCLS) } },
  };
  if (includeCrf) {
    base.crf = {
      trans: { shape: [3, 3], data: randArr(9) },
      start: { shape: [3], data: randArr(3) },
      end: { shape: [3], data: randArr(3) },
    };
  }
  return base;
}

test("loadWeights: exposes crf params from a synthetic v2 JSON with a crf block", () => {
  const json = makeV2WeightsWithCrf(true);
  const w = loadWeights(json as any);
  assert.ok(w.crf);
  assert.equal(w.crf!.trans.length, 9);
  assert.equal(w.crf!.start.length, 3);
  assert.equal(w.crf!.end.length, 3);
  const got = Array.from(w.crf!.trans);
  const want = Float32Array.from(json.crf!.trans.data);
  for (let i = 0; i < got.length; i++) assert.equal(got[i], want[i]);
});

test("loadWeights: crf is undefined when the JSON has no crf block (legacy)", () => {
  const json = makeV2WeightsWithCrf(false);
  const w = loadWeights(json as any);
  assert.equal(w.crf, undefined);
});

test("modelInfo(): crf is true and counts 15 params when weights have a crf block, false/no extra params otherwise", () => {
  const withCrf = createParser({ weights: makeV2WeightsWithCrf(true) as any });
  const withoutCrf = createParser({ weights: makeV2WeightsWithCrf(false) as any });
  const infoWith = withCrf.modelInfo();
  const infoWithout = withoutCrf.modelInfo();
  assert.equal(infoWith.crf, true);
  assert.equal(infoWithout.crf, false);
  assert.equal(infoWith.params - infoWithout.params, 15);
});

test("Parser.parse()/inspect() run without throwing on synthetic weights with a crf block (Viterbi path exercised end-to-end)", () => {
  const p = createParser({ weights: makeV2WeightsWithCrf(true) as any });
  assert.doesNotThrow(() => p.parse("2.5 lakh rupees"));
  const insp = p.inspect("2.5 lakh rupees");
  assert.equal(insp.chars.length, "2.5 lakh rupees".length);
});
