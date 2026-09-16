import { test } from "node:test";
import assert from "node:assert/strict";
import { loadWeights, type ConvLayer, type LoadedWeights } from "../src/weights.ts";
import { forward, Scratch } from "../src/infer-cpu.ts";
import type { WeightsJsonV2Float } from "../src/types.ts";

const E = 4;
const C = 6;
const V = 10; // vocab
const NCLS = 3;

function randArr(n: number): number[] {
  return Array.from({ length: n }, () => Math.random() * 2 - 1);
}

function makeV2Weights(): WeightsJsonV2Float {
  return {
    version: 2,
    charset: Array.from({ length: V }, (_, i) => String.fromCharCode(97 + i)),
    classes: Array.from({ length: NCLS }, (_, i) => `CLS_${i}`),
    embed_dim: E,
    channels: C,
    embed: { shape: [V, E], data: randArr(V * E) },
    conv: [
      { k: 3, dilation: 1, residual: false, w: { shape: [C, E, 3], data: randArr(C * E * 3) }, b: { shape: [C], data: randArr(C) } },
      { k: 3, dilation: 2, residual: true, w: { shape: [C, C, 3], data: randArr(C * C * 3) }, b: { shape: [C], data: randArr(C) } },
    ],
    bio: { w: { shape: [3, C], data: randArr(3 * C) }, b: { shape: [3], data: randArr(3) } },
    cls: { w: { shape: [NCLS, C], data: randArr(NCLS * C) }, b: { shape: [NCLS], data: randArr(NCLS) } },
  };
}

/** Naive reference conv1d+relu(+residual) implementation, independent of
 * infer-cpu.ts's unrolled/scratch-buffer implementation. x: (cIn, L). */
function naiveConv(x: number[][], layer: { k: number; dilation: number; residual: boolean; w: number[][][]; b: number[] }): number[][] {
  const cIn = x.length;
  const L = x[0].length;
  const cOut = layer.w.length;
  const padding = (layer.dilation * (layer.k - 1)) / 2;
  const out: number[][] = Array.from({ length: cOut }, () => new Array(L).fill(0));
  for (let oc = 0; oc < cOut; oc++) {
    for (let t = 0; t < L; t++) {
      let acc = layer.b[oc];
      for (let ic = 0; ic < cIn; ic++) {
        for (let tap = 0; tap < layer.k; tap++) {
          const srcT = t + tap * layer.dilation - padding;
          if (srcT >= 0 && srcT < L) acc += layer.w[oc][ic][tap] * x[ic][srcT];
        }
      }
      let v = acc > 0 ? acc : 0;
      if (layer.residual) v += x[oc][t];
      out[oc][t] = v;
    }
  }
  return out;
}

function tensorToNested(w: { shape: number[]; data: Float32Array }): number[][][] {
  const [cOut, cIn, k] = w.shape;
  const out: number[][][] = [];
  for (let oc = 0; oc < cOut; oc++) {
    const row: number[][] = [];
    for (let ic = 0; ic < cIn; ic++) {
      const tap: number[] = [];
      for (let t = 0; t < k; t++) tap.push(w.data[(oc * cIn + ic) * k + t]);
      row.push(tap);
    }
    out.push(row);
  }
  return out;
}

test("loadWeights: normalizes a synthetic v2 float weights JSON", () => {
  const json = makeV2Weights();
  const w = loadWeights(json as any);
  assert.equal(w.version, 2);
  assert.equal(w.conv.length, 2);
  assert.equal(w.conv[0].k, 3);
  assert.equal(w.conv[0].dilation, 1);
  assert.equal(w.conv[0].residual, false);
  assert.equal(w.conv[0].padding, 1);
  assert.equal(w.conv[1].dilation, 2);
  assert.equal(w.conv[1].residual, true);
  assert.equal(w.conv[1].padding, 2);
});

test("forward(): v2 synthetic weights produce correctly-shaped output", () => {
  const json = makeV2Weights();
  const w = loadWeights(json as any);
  const scratch = new Scratch(w);
  const L = 12;
  const charIds = Int32Array.from({ length: L }, () => Math.floor(Math.random() * V));
  const fw = forward(w, charIds, scratch);
  assert.equal(fw.length, L);
  assert.equal(fw.nCls, NCLS);
  assert.equal(fw.bioLogits.length >= L * 3, true);
  assert.equal(fw.clsLogits.length >= L * NCLS, true);
});

test("forward(): residual layer equals relu(conv(x)) + x, checked against a naive JS reference", () => {
  const json = makeV2Weights();
  const w = loadWeights(json as any);
  const L = 9;
  const charIds = Int32Array.from({ length: L }, () => Math.floor(Math.random() * V));

  // Reference: embed -> layer0 (no residual) -> layer1 (residual)
  const embed: number[][] = Array.from({ length: E }, () => new Array(L).fill(0));
  for (let t = 0; t < L; t++) {
    for (let c = 0; c < E; c++) embed[c][t] = json.embed.data[charIds[t] * E + c];
  }
  const layer0 = { k: w.conv[0].k, dilation: w.conv[0].dilation, residual: w.conv[0].residual, w: tensorToNested(w.conv[0].w), b: Array.from(w.conv[0].b) };
  const layer1 = { k: w.conv[1].k, dilation: w.conv[1].dilation, residual: w.conv[1].residual, w: tensorToNested(w.conv[1].w), b: Array.from(w.conv[1].b) };
  const after0 = naiveConv(embed, layer0);
  const after1 = naiveConv(after0, layer1);

  const scratch = new Scratch(w);
  const fw = forward(w, charIds, scratch);

  // fw only exposes head outputs, so recompute head0 (bio) from the naive
  // reference's final activations and compare to fw.bioLogits directly.
  for (let t = 0; t < L; t++) {
    for (let o = 0; o < 3; o++) {
      let acc = w.bio.b[o];
      for (let c = 0; c < C; c++) acc += w.bio.w.data[o * C + c] * after1[c][t];
      const got = fw.bioLogits[t * 3 + o];
      assert.ok(Math.abs(got - acc) < 1e-3, `bio logit mismatch at t=${t} o=${o}: got ${got}, want ${acc}`);
    }
  }
});
