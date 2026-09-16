import type { WeightsJson, WeightsJsonFloat, WeightsJsonInt8 } from "./types.ts";

export interface Tensor {
  shape: number[];
  data: Float32Array;
}

export interface ConvLayer {
  w: Tensor; // (C_out, C_in, K)
  b: Float32Array;
}

export interface LoadedWeights {
  layers: number;
  dilation: number;
  charset: string[];
  classes: string[];
  embed: Tensor; // (V, E)
  conv1: ConvLayer;
  conv2: ConvLayer;
  conv3: ConvLayer;
  conv4?: ConvLayer;
  bio: ConvLayer;
  cls: ConvLayer;
}

function isInt8(obj: WeightsJson): obj is WeightsJsonInt8 {
  return (obj as WeightsJsonInt8).quantized !== undefined;
}

function b64ToInt8(b64: string): Int8Array {
  const bin = atob(b64);
  const arr = new Int8Array(bin.length);
  for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i) << 24 >> 24;
  return arr;
}

function dequantTensor(t: { shape: number[]; scale: number; data_b64: string }): Tensor {
  const raw = b64ToInt8(t.data_b64);
  const data = new Float32Array(raw.length);
  for (let i = 0; i < raw.length; i++) data[i] = raw[i] * t.scale;
  return { shape: t.shape, data };
}

function floatTensor(t: { shape: number[]; data: number[] }): Tensor {
  return { shape: t.shape, data: Float32Array.from(t.data) };
}

/** Load either the float or int8 weights JSON (as written by
 * python/sankhya/export.py) into flat Float32Array tensors. */
export function loadWeights(json: WeightsJson): LoadedWeights {
  const layers = json.layers ?? 3;
  const dilation = json.dilation ?? 2;
  const names = ["conv1", "conv2", "conv3", ...(layers === 4 ? ["conv4"] : [])] as const;

  if (isInt8(json)) {
    const out: LoadedWeights = {
      layers,
      dilation,
      charset: json.charset,
      classes: json.classes,
      embed: dequantTensor(json.embed),
      conv1: undefined as unknown as ConvLayer,
      conv2: undefined as unknown as ConvLayer,
      conv3: undefined as unknown as ConvLayer,
      bio: undefined as unknown as ConvLayer,
      cls: undefined as unknown as ConvLayer,
    };
    for (const name of [...names, "bio", "cls"] as const) {
      const layer = (json as unknown as Record<string, { w: { shape: number[]; scale: number; data_b64: string }; b: number[] }>)[name];
      (out as any)[name] = { w: dequantTensor(layer.w), b: Float32Array.from(layer.b) };
    }
    return out;
  }

  const f = json as WeightsJsonFloat;
  const out: LoadedWeights = {
    layers,
    dilation,
    charset: f.charset,
    classes: f.classes,
    embed: floatTensor(f.embed),
    conv1: undefined as unknown as ConvLayer,
    conv2: undefined as unknown as ConvLayer,
    conv3: undefined as unknown as ConvLayer,
    bio: undefined as unknown as ConvLayer,
    cls: undefined as unknown as ConvLayer,
  };
  for (const name of [...names, "bio", "cls"] as const) {
    const layer = (f as unknown as Record<string, { w: { shape: number[]; data: number[] }; b: { shape: number[]; data: number[] } }>)[name];
    (out as any)[name] = { w: floatTensor(layer.w), b: Float32Array.from(layer.b.data) };
  }
  return out;
}
