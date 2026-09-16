// Plain typed-array forward pass mirroring python/sankhya/np_infer.py.

import type { ConvLayer, LoadedWeights, Tensor } from "./weights.ts";

export interface ForwardResult {
  bioLogits: Float32Array; // (L, 3) row-major
  clsLogits: Float32Array; // (L, nCls) row-major
  length: number;
  nCls: number;
}

/** x: (C_in, L) row-major flat. w: (C_out, C_in, K). Returns (C_out, L) row-major flat, with ReLU applied. */
function conv1dRelu(
  x: Float32Array,
  cIn: number,
  L: number,
  layer: ConvLayer,
  padding: number,
  dilation: number,
): { out: Float32Array; cOut: number } {
  const [cOut, cInW, k] = layer.w.shape;
  if (cInW !== cIn) throw new Error(`channel mismatch: expected ${cIn}, got ${cInW}`);
  const w = layer.w.data;
  const b = layer.b;

  const padded = padding * 2 + L;
  const xp = new Float32Array(cIn * padded);
  for (let c = 0; c < cIn; c++) {
    xp.set(x.subarray(c * L, c * L + L), c * padded + padding);
  }

  const out = new Float32Array(cOut * L);
  // channels-outer loop order
  for (let oc = 0; oc < cOut; oc++) {
    const outBase = oc * L;
    const bias = b[oc];
    for (let t = 0; t < L; t++) out[outBase + t] = bias;
    for (let ic = 0; ic < cIn; ic++) {
      const xpBase = ic * padded;
      const wBase = (oc * cIn + ic) * k;
      for (let tap = 0; tap < k; tap++) {
        const wv = w[wBase + tap];
        if (wv === 0) continue;
        const offset = tap * dilation;
        const xBase = xpBase + offset;
        for (let t = 0; t < L; t++) {
          out[outBase + t] += wv * xp[xBase + t];
        }
      }
    }
    for (let t = 0; t < L; t++) {
      const v = out[outBase + t];
      out[outBase + t] = v > 0 ? v : 0;
    }
  }
  return { out, cOut };
}

/** Dense matmul: xt (L, C) @ w^T (C, nOut) + b -> (L, nOut), row-major flat. w is (nOut, C). */
function headMatmul(xChannelsMajor: Float32Array, cIn: number, L: number, layer: ConvLayer): { out: Float32Array; nOut: number } {
  const [nOut, cInW] = layer.w.shape;
  if (cInW !== cIn) throw new Error(`head channel mismatch: expected ${cIn}, got ${cInW}`);
  const w = layer.w.data;
  const b = layer.b;
  const out = new Float32Array(L * nOut);
  for (let t = 0; t < L; t++) {
    const outBase = t * nOut;
    for (let o = 0; o < nOut; o++) out[outBase + o] = b[o];
  }
  for (let o = 0; o < nOut; o++) {
    const wBase = o * cIn;
    for (let ic = 0; ic < cIn; ic++) {
      const wv = w[wBase + ic];
      if (wv === 0) continue;
      const xBase = ic * L;
      for (let t = 0; t < L; t++) {
        out[t * nOut + o] += wv * xChannelsMajor[xBase + t];
      }
    }
  }
  return { out, nOut };
}

export function forward(weights: LoadedWeights, charIds: Int32Array | number[]): ForwardResult {
  const L = charIds.length;
  const embed: Tensor = weights.embed;
  const embedDim = embed.shape[1];

  // x: (C, L) channels-major, matching np_infer's embed[char_ids].T
  let x: Float32Array<ArrayBufferLike> = new Float32Array(embedDim * L);
  for (let t = 0; t < L; t++) {
    const cid = charIds[t] as number;
    const eBase = cid * embedDim;
    for (let c = 0; c < embedDim; c++) {
      x[c * L + t] = embed.data[eBase + c];
    }
  }
  let cIn = embedDim;

  let r = conv1dRelu(x, cIn, L, weights.conv1, 1, 1);
  x = r.out;
  cIn = r.cOut;

  r = conv1dRelu(x, cIn, L, weights.conv2, 2, 1);
  x = r.out;
  cIn = r.cOut;

  const dilation = weights.dilation;
  const pad3 = Math.floor((dilation * (3 - 1)) / 2);
  r = conv1dRelu(x, cIn, L, weights.conv3, pad3, dilation);
  x = r.out;
  cIn = r.cOut;

  if (weights.layers === 4 && weights.conv4) {
    const dilation4 = 4;
    const pad4 = Math.floor((dilation4 * (3 - 1)) / 2);
    r = conv1dRelu(x, cIn, L, weights.conv4, pad4, dilation4);
    x = r.out;
    cIn = r.cOut;
  }

  const bioOut = headMatmul(x, cIn, L, weights.bio);
  const clsOut = headMatmul(x, cIn, L, weights.cls);

  return { bioLogits: bioOut.out, clsLogits: clsOut.out, length: L, nCls: clsOut.nOut };
}

export function softmaxRow(logits: Float32Array, rowStart: number, n: number): Float32Array {
  let max = -Infinity;
  for (let i = 0; i < n; i++) max = Math.max(max, logits[rowStart + i]);
  let sum = 0;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const e = Math.exp(logits[rowStart + i] - max);
    out[i] = e;
    sum += e;
  }
  for (let i = 0; i < n; i++) out[i] /= sum;
  return out;
}

export function argmaxRow(logits: Float32Array, rowStart: number, n: number): number {
  let best = 0;
  let bestV = -Infinity;
  for (let i = 0; i < n; i++) {
    const v = logits[rowStart + i];
    if (v > bestV) {
      bestV = v;
      best = i;
    }
  }
  return best;
}
