import type { WeightsJson, WeightsJsonFloat, WeightsJsonInt8, WeightsJsonV2Float, WeightsJsonV2Int8 } from "./types.ts";

export interface Tensor {
  shape: number[];
  data: Float32Array;
}

export interface ConvLayer {
  k: number;
  dilation: number;
  residual: boolean;
  padding: number;
  w: Tensor; // (C_out, C_in, K)
  b: Float32Array;
}

export interface HeadLayer {
  w: Tensor;
  b: Float32Array;
}

export interface LoadedWeights {
  version: number;
  charset: string[];
  classes: string[];
  embed: Tensor; // (V, E)
  conv: ConvLayer[];
  bio: HeadLayer;
  cls: HeadLayer;
}

function isV2(obj: WeightsJson): obj is WeightsJsonV2Float | WeightsJsonV2Int8 {
  return (obj as { version?: number }).version === 2 && Array.isArray((obj as WeightsJsonV2Float).conv);
}

function isInt8(obj: WeightsJson): obj is WeightsJsonInt8 | WeightsJsonV2Int8 {
  return (obj as { quantized?: string }).quantized !== undefined;
}

function b64ToInt8(b64: string): Int8Array {
  const bin = atob(b64);
  const arr = new Int8Array(bin.length);
  for (let i = 0; i < bin.length; i++) arr[i] = (bin.charCodeAt(i) << 24) >> 24;
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

function padOf(dilation: number, k: number): number {
  return (dilation * (k - 1)) / 2;
}

function headFromFloat(layer: { w: { shape: number[]; data: number[] }; b: { shape: number[]; data: number[] } }): HeadLayer {
  return { w: floatTensor(layer.w), b: Float32Array.from(layer.b.data) };
}

function headFromInt8(layer: { w: { shape: number[]; scale: number; data_b64: string }; b: number[] }): HeadLayer {
  return { w: dequantTensor(layer.w), b: Float32Array.from(layer.b) };
}

/** Load a v1 or v2, float or int8, weights JSON (as written by
 * python/sankhya/export.py) into a normalised LoadedWeights with an ordered
 * `conv` layer list. See CONTRACT.md section 2. */
export function loadWeights(json: WeightsJson): LoadedWeights {
  const int8 = isInt8(json);

  if (isV2(json)) {
    const conv: ConvLayer[] = json.conv.map((l) => {
      const padding = padOf(l.dilation, l.k);
      if (int8) {
        const li = l as unknown as { k: number; dilation: number; residual: boolean; w: { shape: number[]; scale: number; data_b64: string }; b: number[] };
        return { k: l.k, dilation: l.dilation, residual: l.residual, padding, w: dequantTensor(li.w), b: Float32Array.from(li.b) };
      }
      const lf = l as unknown as { k: number; dilation: number; residual: boolean; w: { shape: number[]; data: number[] }; b: { shape: number[]; data: number[] } };
      return { k: l.k, dilation: l.dilation, residual: l.residual, padding, w: floatTensor(lf.w), b: Float32Array.from(lf.b.data) };
    });
    return {
      version: 2,
      charset: json.charset,
      classes: json.classes,
      embed: int8 ? dequantTensor((json as WeightsJsonV2Int8).embed) : floatTensor((json as WeightsJsonV2Float).embed),
      conv,
      bio: int8 ? headFromInt8((json as WeightsJsonV2Int8).bio) : headFromFloat((json as WeightsJsonV2Float).bio),
      cls: int8 ? headFromInt8((json as WeightsJsonV2Int8).cls) : headFromFloat((json as WeightsJsonV2Float).cls),
    };
  }

  // v1: conv1..conv4, optional layers (default 3), optional dilation (default 2).
  const layers = json.layers ?? 3;
  const dilation = json.dilation ?? 2;
  const specs: Array<{ name: "conv1" | "conv2" | "conv3" | "conv4"; k: number; dilation: number }> = [
    { name: "conv1", k: 3, dilation: 1 },
    { name: "conv2", k: 5, dilation: 1 },
    { name: "conv3", k: 3, dilation },
  ];
  if (layers === 4) specs.push({ name: "conv4", k: 3, dilation: 4 });

  const conv: ConvLayer[] = specs.map((spec) => {
    const padding = padOf(spec.dilation, spec.k);
    if (int8) {
      const j = json as WeightsJsonInt8;
      const layer = j[spec.name] as { w: { shape: number[]; scale: number; data_b64: string }; b: number[] } | undefined;
      if (!layer) throw new Error(`missing ${spec.name} in weights JSON`);
      return { k: spec.k, dilation: spec.dilation, residual: false, padding, w: dequantTensor(layer.w), b: Float32Array.from(layer.b) };
    }
    const j = json as WeightsJsonFloat;
    const layer = j[spec.name] as { w: { shape: number[]; data: number[] }; b: { shape: number[]; data: number[] } } | undefined;
    if (!layer) throw new Error(`missing ${spec.name} in weights JSON`);
    return { k: spec.k, dilation: spec.dilation, residual: false, padding, w: floatTensor(layer.w), b: Float32Array.from(layer.b.data) };
  });

  return {
    version: json.version ?? 1,
    charset: json.charset,
    classes: json.classes,
    embed: int8 ? dequantTensor((json as WeightsJsonInt8).embed) : floatTensor((json as WeightsJsonFloat).embed),
    conv,
    bio: int8 ? headFromInt8((json as WeightsJsonInt8).bio) : headFromFloat((json as WeightsJsonFloat).bio),
    cls: int8 ? headFromInt8((json as WeightsJsonInt8).cls) : headFromFloat((json as WeightsJsonFloat).cls),
  };
}
