// Plain typed-array forward pass mirroring python/sankhya/np_infer.py.
//
// Performance notes: all working buffers (including the zero-padded input
// copy each conv layer needs) are preallocated once per Scratch (sized to
// PADDED_MAX) and reused across calls -- forward() itself never allocates on
// the hot path. The conv1d inner loop is loop-interchanged (t innermost,
// tap coefficients hoisted into locals and unrolled for the k=3/k=5 kernel
// sizes this model actually uses) instead of a separate tap loop per
// output position, which lets V8 keep the whole accumulation in registers
// and roughly triples throughput over the naive "for tap { for t }" form.

import type { ConvLayer, LoadedWeights, Tensor } from "./weights.ts";
import { PADDED_MAX } from "./charset.ts";

export interface ForwardResult {
  bioLogits: Float32Array; // (L, 3) row-major, view into scratch (valid until next forward() call on the same scratch)
  clsLogits: Float32Array; // (L, nCls) row-major, view into scratch
  length: number;
  nCls: number;
}

/** Preallocated buffers for one Parser's forward passes, sized once from the
 * weights' channel counts and PADDED_MAX so forward() does no per-call allocation. */
export class Scratch {
  private buffers: Float32Array[]; // ping-pong pair, sized [maxChannels * PADDED_MAX]
  private padded: Float32Array; // scratch for the zero-padded conv input, sized [maxChannels * (PADDED_MAX + 2*maxPad)]
  readonly bioBuf: Float32Array;
  readonly clsBuf: Float32Array;

  constructor(weights: LoadedWeights) {
    const layers = [weights.conv1, weights.conv2, weights.conv3, ...(weights.layers === 4 && weights.conv4 ? [weights.conv4] : [])];
    const maxC = Math.max(weights.embed.shape[1], ...layers.map((l) => l.w.shape[0]));
    this.buffers = [new Float32Array(maxC * PADDED_MAX), new Float32Array(maxC * PADDED_MAX)];
    // Max padding across layers this model uses is small (dilation*(k-1)/2, dilation<=4, k<=5) -- 8 is a safe ceiling.
    const maxPad = 8;
    this.padded = new Float32Array(maxC * (PADDED_MAX + 2 * maxPad));
    this.bioBuf = new Float32Array(weights.bio.w.shape[0] * PADDED_MAX);
    this.clsBuf = new Float32Array(weights.cls.w.shape[0] * PADDED_MAX);
  }

  buf(i: number): Float32Array {
    return this.buffers[i % 2];
  }

  paddedBuf(): Float32Array {
    return this.padded;
  }
}

/** x: (cIn, L) row-major flat. w: (C_out, C_in, K). Writes ReLU(conv1d(x)) into
 * `out` (cOut*L flat prefix of a larger scratch buffer). Copies x into a
 * reused zero-padded scratch buffer, then runs an unrolled per-tap
 * accumulation with t innermost (k=3 and k=5 are the only kernel sizes this
 * model uses; anything else falls back to a generic unrolled-free loop). */
function conv1dRelu(
  x: Float32Array,
  cIn: number,
  L: number,
  layer: ConvLayer,
  padding: number,
  dilation: number,
  out: Float32Array,
  padBuf: Float32Array,
): number {
  const [cOut, cInW, k] = layer.w.shape;
  if (cInW !== cIn) throw new Error(`channel mismatch: expected ${cIn}, got ${cInW}`);
  const w = layer.w.data;
  const b = layer.b;

  const padded = L + 2 * padding;
  for (let c = 0; c < cIn; c++) {
    padBuf.fill(0, c * padded, c * padded + padding);
    padBuf.set(x.subarray(c * L, c * L + L), c * padded + padding);
    padBuf.fill(0, c * padded + padding + L, c * padded + padded);
  }

  for (let oc = 0; oc < cOut; oc++) {
    const outBase = oc * L;
    const bias = b[oc];
    const wBaseOc = oc * cIn * k;

    if (k === 3) {
      for (let t = 0; t < L; t++) out[outBase + t] = bias;
      for (let ic = 0; ic < cIn; ic++) {
        const xpBase = ic * padded;
        const wBase = wBaseOc + ic * 3;
        const w0 = w[wBase];
        const w1 = w[wBase + 1];
        const w2 = w[wBase + 2];
        for (let t = 0; t < L; t++) {
          const xb = xpBase + t;
          out[outBase + t] += w0 * padBuf[xb] + w1 * padBuf[xb + dilation] + w2 * padBuf[xb + 2 * dilation];
        }
      }
    } else if (k === 5) {
      for (let t = 0; t < L; t++) out[outBase + t] = bias;
      for (let ic = 0; ic < cIn; ic++) {
        const xpBase = ic * padded;
        const wBase = wBaseOc + ic * 5;
        const w0 = w[wBase],
          w1 = w[wBase + 1],
          w2 = w[wBase + 2],
          w3 = w[wBase + 3],
          w4 = w[wBase + 4];
        for (let t = 0; t < L; t++) {
          const xb = xpBase + t;
          out[outBase + t] += w0 * padBuf[xb] + w1 * padBuf[xb + 1] + w2 * padBuf[xb + 2] + w3 * padBuf[xb + 3] + w4 * padBuf[xb + 4];
        }
      }
    } else {
      for (let t = 0; t < L; t++) out[outBase + t] = bias;
      for (let ic = 0; ic < cIn; ic++) {
        const xpBase = ic * padded;
        const wBase = wBaseOc + ic * k;
        for (let tap = 0; tap < k; tap++) {
          const wv = w[wBase + tap];
          const off = tap * dilation;
          const xBase = xpBase + off;
          for (let t = 0; t < L; t++) out[outBase + t] += wv * padBuf[xBase + t];
        }
      }
    }

    for (let t = 0; t < L; t++) {
      const v = out[outBase + t];
      out[outBase + t] = v > 0 ? v : 0;
    }
  }
  return cOut;
}

/** Dense matmul: xt (L, C) @ w^T (C, nOut) + b -> (L, nOut), row-major flat into `out`. w is (nOut, C). */
function headMatmul(xChannelsMajor: Float32Array, cIn: number, L: number, layer: ConvLayer, out: Float32Array): number {
  const [nOut, cInW] = layer.w.shape;
  if (cInW !== cIn) throw new Error(`head channel mismatch: expected ${cIn}, got ${cInW}`);
  const w = layer.w.data;
  const b = layer.b;
  const cIn4 = cIn - (cIn % 4);
  for (let t = 0; t < L; t++) {
    const ob = t * nOut;
    for (let o = 0; o < nOut; o++) {
      let acc = b[o];
      const wBase = o * cIn;
      let ic = 0;
      for (; ic < cIn4; ic += 4) {
        acc +=
          w[wBase + ic] * xChannelsMajor[ic * L + t] +
          w[wBase + ic + 1] * xChannelsMajor[(ic + 1) * L + t] +
          w[wBase + ic + 2] * xChannelsMajor[(ic + 2) * L + t] +
          w[wBase + ic + 3] * xChannelsMajor[(ic + 3) * L + t];
      }
      for (; ic < cIn; ic++) acc += w[wBase + ic] * xChannelsMajor[ic * L + t];
      out[ob + o] = acc;
    }
  }
  return nOut;
}

export function forward(weights: LoadedWeights, charIds: Int32Array | number[], scratch: Scratch): ForwardResult {
  const L = charIds.length;
  const embed: Tensor = weights.embed;
  const embedDim = embed.shape[1];

  // x: (C, L) channels-major, matching np_infer's embed[char_ids].T
  let x = scratch.buf(0);
  for (let t = 0; t < L; t++) {
    const cid = charIds[t] as number;
    const eBase = cid * embedDim;
    for (let c = 0; c < embedDim; c++) {
      x[c * L + t] = embed.data[eBase + c];
    }
  }
  let cIn = embedDim;
  const padBuf = scratch.paddedBuf();

  let out = scratch.buf(1);
  cIn = conv1dRelu(x, cIn, L, weights.conv1, 1, 1, out, padBuf);
  x = out;

  out = scratch.buf(2);
  cIn = conv1dRelu(x, cIn, L, weights.conv2, 2, 1, out, padBuf);
  x = out;

  const dilation = weights.dilation;
  const pad3 = Math.floor((dilation * (3 - 1)) / 2);
  out = scratch.buf(3);
  cIn = conv1dRelu(x, cIn, L, weights.conv3, pad3, dilation, out, padBuf);
  x = out;

  if (weights.layers === 4 && weights.conv4) {
    const dilation4 = 4;
    const pad4 = Math.floor((dilation4 * (3 - 1)) / 2);
    out = scratch.buf(4);
    cIn = conv1dRelu(x, cIn, L, weights.conv4, pad4, dilation4, out, padBuf);
    x = out;
  }

  headMatmul(x, cIn, L, weights.bio, scratch.bioBuf);
  const nCls = headMatmul(x, cIn, L, weights.cls, scratch.clsBuf);

  return { bioLogits: scratch.bioBuf, clsLogits: scratch.clsBuf, length: L, nCls };
}

export function softmaxRow(logits: Float32Array, rowStart: number, n: number, out: Float32Array): Float32Array {
  let max = -Infinity;
  for (let i = 0; i < n; i++) max = Math.max(max, logits[rowStart + i]);
  let sum = 0;
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
