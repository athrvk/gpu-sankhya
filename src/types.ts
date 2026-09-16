export interface Sankhya {
  span: string;
  start: number;
  end: number;
  value: number;
  range?: [number, number];
  unit:
    | "sau"
    | "hazaar"
    | "lakh"
    | "crore"
    | "million"
    | "billion"
    | "arab"
    | "kharab"
    | null;
  currency: "INR" | null;
  confidence: number;
  classes: string[];
}

export interface ParseOptions {
  backend?: "cpu" | "webgpu" | "auto";
}

/** Raw float weights JSON, as written by python/sankhya/export.py (float form). */
export interface WeightsJsonFloat {
  version: number;
  layers?: number;
  dilation?: number;
  charset: string[];
  classes: string[];
  embed: { shape: number[]; data: number[] };
  conv1: { w: { shape: number[]; data: number[] }; b: { shape: number[]; data: number[] } };
  conv2: { w: { shape: number[]; data: number[] }; b: { shape: number[]; data: number[] } };
  conv3: { w: { shape: number[]; data: number[] }; b: { shape: number[]; data: number[] } };
  conv4?: { w: { shape: number[]; data: number[] }; b: { shape: number[]; data: number[] } };
  bio: { w: { shape: number[]; data: number[] }; b: { shape: number[]; data: number[] } };
  cls: { w: { shape: number[]; data: number[] }; b: { shape: number[]; data: number[] } };
}

/** Raw int8-quantized weights JSON, as written by python/sankhya/export.py (int8 form). */
export interface WeightsJsonInt8 {
  version: number;
  quantized: string;
  layers?: number;
  dilation?: number;
  charset: string[];
  classes: string[];
  embed: { shape: number[]; scale: number; data_b64: string };
  conv1: { w: { shape: number[]; scale: number; data_b64: string }; b: number[] };
  conv2: { w: { shape: number[]; scale: number; data_b64: string }; b: number[] };
  conv3: { w: { shape: number[]; scale: number; data_b64: string }; b: number[] };
  conv4?: { w: { shape: number[]; scale: number; data_b64: string }; b: number[] };
  bio: { w: { shape: number[]; scale: number; data_b64: string }; b: number[] };
  cls: { w: { shape: number[]; scale: number; data_b64: string }; b: number[] };
}

export type WeightsJson = WeightsJsonFloat | WeightsJsonInt8;
