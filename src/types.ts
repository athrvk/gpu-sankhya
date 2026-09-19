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
  /** True when every decoded token is independently justified by the
   * lexicon (or a structural rule for SEP/DOT/COMMA/RANGE/DIGITS) -- see
   * verifyTokens() in verify.ts. A verified span's value is a pure
   * function of the lexicon + arithmetic core. */
  verified: boolean;
  /** The (class, text) tokens the span decoded to, in order -- for
   * explainability and for verifyTokens(). */
  tokens: Array<[string, string]>;
}

export interface ParseOptions {
  backend?: "cpu" | "webgpu" | "auto";
  /** When true, spans that are not verified() are dropped from the
   * result. Default false. */
  strict?: boolean;
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

/** v2 conv layer entry, float form. */
export interface ConvLayerJsonFloat {
  k: number;
  dilation: number;
  residual: boolean;
  w: { shape: number[]; data: number[] };
  b: { shape: number[]; data: number[] };
}

/** v2 conv layer entry, int8 form. */
export interface ConvLayerJsonInt8 {
  k: number;
  dilation: number;
  residual: boolean;
  w: { shape: number[]; scale: number; data_b64: string };
  b: number[];
}

/** Weights JSON version 2, float form: an explicit ordered `conv` layer list
 * instead of the hard-coded conv1..conv4 fields. See CONTRACT.md section 2. */
export interface WeightsJsonV2Float {
  version: 2;
  charset: string[];
  classes: string[];
  embed_dim?: number;
  channels?: number;
  embed: { shape: number[]; data: number[] };
  conv: ConvLayerJsonFloat[];
  bio: { w: { shape: number[]; data: number[] }; b: { shape: number[]; data: number[] } };
  cls: { w: { shape: number[]; data: number[] }; b: { shape: number[]; data: number[] } };
  /** Optional CRF params over the BIO head (3 labels: O/B/I). Stored as
   * float tensors even in the int8 weights file -- 15 numbers, not worth
   * quantizing. Absent -> legacy argmax BIO decoding. */
  crf?: {
    trans: { shape: number[]; data: number[] }; // (3,3)
    start: { shape: number[]; data: number[] }; // (3,)
    end: { shape: number[]; data: number[] }; // (3,)
  };
}

/** Weights JSON version 2, int8 form. */
export interface WeightsJsonV2Int8 {
  version: 2;
  quantized: string;
  charset: string[];
  classes: string[];
  embed_dim?: number;
  channels?: number;
  embed: { shape: number[]; scale: number; data_b64: string };
  conv: ConvLayerJsonInt8[];
  bio: { w: { shape: number[]; scale: number; data_b64: string }; b: number[] };
  cls: { w: { shape: number[]; scale: number; data_b64: string }; b: number[] };
  /** Optional CRF params -- stored in FLOAT even in the int8 file (see
   * WeightsJsonV2Float.crf). */
  crf?: {
    trans: { shape: number[]; data: number[] };
    start: { shape: number[]; data: number[] };
    end: { shape: number[]; data: number[] };
  };
}

export type WeightsJson = WeightsJsonFloat | WeightsJsonInt8 | WeightsJsonV2Float | WeightsJsonV2Int8;

export interface CharTag {
  ch: string;
  bio: 0 | 1 | 2;
  bioProb: [number, number, number];
  cls: string;
  clsId: number;
}

export interface Inspection {
  /** normalized text (see normalizeText()) */
  text: string;
  chars: CharTag[];
  spans: Sankhya[];
  ms: number;
}

export interface ModelInfo {
  version: number;
  params: number;
  channels: number;
  embedDim: number;
  vocab: number;
  classes: number;
  layers: Array<{ k: number; dilation: number; residual: boolean }>;
  /** Whether the loaded weights include a CRF head (Viterbi decoding used
   * for BIO instead of legacy per-position argmax). */
  crf: boolean;
}
