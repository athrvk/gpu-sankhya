import type { Sankhya, ParseOptions, WeightsJson } from "./types.ts";
import { loadWeights, type LoadedWeights } from "./weights.ts";
import { buildCharToId, encodeChars, makeWindows } from "./charset.ts";
import { forward, softmaxRow, argmaxRow } from "./infer-cpu.ts";
import { decodeSpans } from "./decode.ts";
import { evaluate, detectCurrency } from "./core.ts";
import { HI_LATN } from "./lang-hi-latn.ts";
import { CLASSES } from "./classes.ts";
import { WebGPUBackend, isWebGPUAvailable as gpuAvailable } from "./infer-webgpu.ts";
import defaultWeightsJson from "./data/default-weights.json" with { type: "json" };

export type { Sankhya, ParseOptions } from "./types.ts";
export { isWebGPUAvailable } from "./infer-webgpu.ts";

export interface CreateParserOptions {
  weights?: WeightsJson;
  backend?: "cpu" | "webgpu" | "auto";
}

export class Parser {
  private weights: LoadedWeights;
  private charToId: Map<string, number>;
  private gpu: WebGPUBackend | null = null;
  private defaultBackend: "cpu" | "webgpu" | "auto";

  constructor(opts: CreateParserOptions = {}) {
    this.weights = loadWeights(opts.weights ?? (defaultWeightsJson as unknown as WeightsJson));
    this.charToId = buildCharToId(this.weights.charset);
    this.defaultBackend = opts.backend ?? "cpu";
  }

  /** Synchronous CPU parse of a single string. */
  parse(text: string, opts: ParseOptions = {}): Sankhya[] {
    const windows = makeWindows(text.length);
    const merged: Sankhya[] = [];
    const seen = new Set<string>();
    for (const win of windows) {
      const sub = text.slice(win.offset, win.offset + win.length);
      const ids = encodeChars(sub, this.charToId);
      const fw = forward(this.weights, ids);
      const results = this.decodeForward(sub, fw, win.offset);
      for (const r of results) {
        const key = `${r.start}:${r.end}`;
        if (!seen.has(key)) {
          seen.add(key);
          merged.push(r);
        }
      }
    }
    merged.sort((a, b) => a.start - b.start);
    return merged;
  }

  private decodeForward(sub: string, fw: ReturnType<typeof forward>, offset: number): Sankhya[] {
    const L = fw.length;
    const bioIds = new Int32Array(L);
    const clsIds = new Int32Array(L);
    const bioProbs: Float32Array[] = new Array(L);
    for (let t = 0; t < L; t++) {
      bioIds[t] = argmaxRow(fw.bioLogits, t * 3, 3);
      clsIds[t] = argmaxRow(fw.clsLogits, t * fw.nCls, fw.nCls);
      bioProbs[t] = softmaxRow(fw.bioLogits, t * 3, 3);
    }
    const spans = decodeSpans(sub, bioIds, clsIds, bioProbs);
    const out: Sankhya[] = [];
    for (const span of spans) {
      const tokens: Array<[string, string]> = span.tokens.map(([cid, txt]) => [CLASSES[cid], txt]);
      const res = evaluate(tokens);
      const currency = detectCurrency(sub, span.start, span.end, HI_LATN);
      out.push({
        span: span.text,
        start: span.start + offset,
        end: span.end + offset,
        value: res.value,
        ...(res.range ? { range: res.range as [number, number] } : {}),
        unit: (res.unit as Sankhya["unit"]) ?? null,
        currency,
        confidence: span.confidence ?? 0,
        classes: res.classes,
      });
    }
    return out;
  }

  /** Batched parse. Uses WebGPU when requested/available and the batch is
   * large enough to be worthwhile; falls back to CPU otherwise. */
  async parseBatch(texts: string[], opts: ParseOptions = {}): Promise<Sankhya[][]> {
    const backend = opts.backend ?? this.defaultBackend;
    const useGpu =
      backend === "webgpu" || (backend === "auto" && gpuAvailable() && texts.length >= 32);

    if (!useGpu) {
      return texts.map((t) => this.parse(t, { backend: "cpu" }));
    }

    // WebGPU path: batch texts that fit a single window; texts needing a
    // sliding window fall back to CPU (kept correct, still simple).
    const results: Sankhya[][] = new Array(texts.length);
    const gpuIdx: number[] = [];
    const cpuIdx: number[] = [];
    for (let i = 0; i < texts.length; i++) {
      if (texts[i].length <= 128) gpuIdx.push(i);
      else cpuIdx.push(i);
    }
    for (const i of cpuIdx) results[i] = this.parse(texts[i], { backend: "cpu" });

    if (gpuIdx.length > 0) {
      if (!this.gpu) this.gpu = new WebGPUBackend(this.weights);
      const length = Math.max(...gpuIdx.map((i) => texts[i].length), 1);
      const batch = gpuIdx.length;
      const charIds = new Int32Array(batch * length);
      const padId = this.charToId.get("<pad>") ?? 0;
      charIds.fill(padId);
      for (let bi = 0; bi < batch; bi++) {
        const text = texts[gpuIdx[bi]];
        const ids = encodeChars(text, this.charToId);
        charIds.set(ids, bi * length);
      }
      const { bio, cls } = await this.gpu.run(charIds, batch, length);
      const nCls = this.weights.cls.w.shape[0];
      for (let bi = 0; bi < batch; bi++) {
        const text = texts[gpuIdx[bi]];
        const L = text.length;
        const bioLogits = new Float32Array(L * 3);
        const clsLogits = new Float32Array(L * nCls);
        for (let t = 0; t < L; t++) {
          bioLogits.set(bio.subarray((bi * length + t) * 3, (bi * length + t) * 3 + 3), t * 3);
          clsLogits.set(cls.subarray((bi * length + t) * nCls, (bi * length + t) * nCls + nCls), t * nCls);
        }
        const fw = { bioLogits, clsLogits, length: L, nCls };
        results[gpuIdx[bi]] = this.decodeForward(text, fw, 0);
      }
    }
    return results;
  }
}

let defaultParser: Parser | null = null;
function getDefault(): Parser {
  if (!defaultParser) defaultParser = new Parser();
  return defaultParser;
}

/** Parse a single string synchronously, using the CPU backend and the
 * bundled default weights. */
export function parse(text: string, opts: ParseOptions = {}): Sankhya[] {
  return getDefault().parse(text, opts);
}

/** Parse many strings. Uses WebGPU when `opts.backend === "webgpu"`, or
 * when `opts.backend === "auto"` and WebGPU is available and the batch is
 * large (>=32); otherwise runs on CPU. Defaults to CPU. */
export function parseBatch(texts: string[], opts: ParseOptions = {}): Promise<Sankhya[][]> {
  return getDefault().parseBatch(texts, opts);
}

export function createParser(opts: CreateParserOptions = {}): Parser {
  return new Parser(opts);
}
