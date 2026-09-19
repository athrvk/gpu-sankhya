import type { Sankhya, ParseOptions, WeightsJson, CharTag, Inspection, ModelInfo } from "./types.ts";
import { loadWeights, type LoadedWeights } from "./weights.ts";
import { buildCharToId, encodeChars, makeWindows, normalizeText, MAX_LEN, PADDED_MAX, paddedLength } from "./charset.ts";
import { forward, softmaxRow, argmaxRow, viterbi, Scratch } from "./infer-cpu.ts";
import { decodeSpans } from "./decode.ts";
import { evaluate, detectCurrency, mergeLangPacks, shouldDropBareDigits, shouldDropLoneAmbiguousUnit } from "./core.ts";
import { HI_LATN } from "./lang-hi-latn.ts";
import { HI_DEVA } from "./lang-hi-deva.ts";
import { MR_DEVA } from "./lang-mr-deva.ts";
import { CLASSES } from "./classes.ts";
import { WebGPUBackend, probeWebGPU } from "./infer-webgpu.ts";
import defaultWeightsJson from "./data/default-weights.json" with { type: "json" };

export type { Sankhya, ParseOptions, CharTag, Inspection, ModelInfo } from "./types.ts";
export { isWebGPUAvailable, probeWebGPU } from "./infer-webgpu.ts";
export { normalizeText } from "./charset.ts";
export { CLASSES } from "./classes.ts";

export interface CreateParserOptions {
  weights?: WeightsJson;
  backend?: "cpu" | "webgpu" | "auto";
}

export class Parser {
  private weights: LoadedWeights;
  private charToId: Map<string, number>;
  private gpu: WebGPUBackend | null = null;
  private defaultBackend: "cpu" | "webgpu" | "auto";

  // Reused scratch buffers for the hot (single-window) CPU path -- forward()
  // and its callers do no per-call allocation beyond small per-span objects.
  private scratch: Scratch;
  private idsScratch = new Int32Array(PADDED_MAX);
  private bioIdsScratch = new Int32Array(MAX_LEN);
  private clsIdsScratch = new Int32Array(MAX_LEN);
  private bioProbsScratch: Float32Array[] = Array.from({ length: MAX_LEN }, () => new Float32Array(3));

  constructor(opts: CreateParserOptions = {}) {
    this.weights = loadWeights(opts.weights ?? (defaultWeightsJson as unknown as WeightsJson));
    this.charToId = buildCharToId(this.weights.charset);
    this.defaultBackend = opts.backend ?? "cpu";
    this.scratch = new Scratch(this.weights);
  }

  /** Synchronous CPU parse of a single string.
   *
   * The returned spans/offsets (`start`, `end`, and `span` when the
   * normalized string's length differs from the input's) are indices into
   * `normalizeText(text)`, not the raw `text` argument -- see
   * normalizeText() in charset.ts. For the overwhelming majority of real
   * input, which already arrives in NFC form, normalization never changes
   * string length, so offsets and `span` are identical either way. */
  parse(text: string, opts: ParseOptions = {}): Sankhya[] {
    // R1: non-string input (null/undefined/number/object/...) is treated
    // as "" rather than throwing -- callers passing through loosely-typed
    // data (form fields, JSON with an optional key, etc.) get [] back.
    if (typeof text !== "string") return [];
    const original = text;
    text = normalizeText(text);
    const useOriginalSpan = original.length === text.length;
    let results: Sankhya[];
    const windows = makeWindows(text.length);
    if (windows.length === 1 && windows[0].length === text.length) {
      // fast path: no window offset bookkeeping / merge-dedup needed
      const padLen = paddedLength(text.length);
      const ids = encodeChars(text, this.charToId, this.idsScratch, padLen);
      const fw = forward(this.weights, ids, this.scratch);
      results = this.decodeForward(text, fw, 0, text.length);
    } else {
      const merged: Sankhya[] = [];
      const seen = new Set<string>();
      for (const win of windows) {
        const sub = text.slice(win.offset, win.offset + win.length);
        const padLen = paddedLength(sub.length);
        const ids = encodeChars(sub, this.charToId, this.idsScratch, padLen);
        const fw = forward(this.weights, ids, this.scratch);
        const winResults = this.decodeForward(sub, fw, win.offset, sub.length);
        for (const r of winResults) {
          const key = `${r.start}:${r.end}`;
          if (!seen.has(key)) {
            seen.add(key);
            merged.push(r);
          }
        }
      }
      merged.sort((a, b) => a.start - b.start);
      results = merged;
    }
    // `span` is a substring of the ORIGINAL input when normalization did
    // not change the string's length (the overwhelming majority case,
    // including all pure lowercasing/digit-mapping with no NFC
    // recomposition), so callers see the real casing/original characters
    // rather than the lowercased/normalized form; otherwise (a rare NFC
    // length change) it falls back to the normalized string, since
    // `start`/`end` only index consistently into that string.
    if (useOriginalSpan) {
      for (const r of results) r.span = original.slice(r.start, r.end);
    }
    return results;
  }

  /** `fw` may cover a padded length (PAD_TAIL right-padding, see charset.ts);
   * `realLen` (defaults to fw.length, i.e. no padding) bounds decoding to the
   * actual text -- the padded tail's logits are only used to give the real
   * tokens correct right-context during the conv stack, never decoded. */
  private decodeForward(sub: string, fw: ReturnType<typeof forward>, offset: number, realLen: number = fw.length): Sankhya[] {
    const L = realLen;
    const useScratchBufs = L <= MAX_LEN;
    const bioIds = useScratchBufs ? this.bioIdsScratch : new Int32Array(L);
    const clsIds = useScratchBufs ? this.clsIdsScratch : new Int32Array(L);
    const bioProbs: Float32Array[] = useScratchBufs ? this.bioProbsScratch : Array.from({ length: L }, () => new Float32Array(3));
    if (this.weights.crf) {
      viterbi(fw.bioLogits, L, this.weights.crf, bioIds);
    }
    for (let t = 0; t < L; t++) {
      if (!this.weights.crf) bioIds[t] = argmaxRow(fw.bioLogits, t * 3, 3);
      clsIds[t] = argmaxRow(fw.clsLogits, t * fw.nCls, fw.nCls);
      softmaxRow(fw.bioLogits, t * 3, 3, bioProbs[t]);
    }
    const spans = decodeSpans(sub, bioIds.subarray(0, L), clsIds.subarray(0, L), bioProbs);
    const out: Sankhya[] = [];
    for (const span of spans) {
      const tokens: Array<[string, string]> = span.tokens.map(([cid, txt]) => [CLASSES[cid], txt]);
      const currency = detectCurrency(sub, span.start, span.end, CURRENCY_PACK);
      // R3: drop a digits-only span (no UNIT_/PFX_/CARD_) unless a
      // currency marker was found for it, narrowed to short-ungrouped
      // (<=4 digits, no comma) or very-long (>=10 digits) spans -- see
      // shouldDropBareDigits().
      if (shouldDropBareDigits(tokens) && currency === null) continue;
      // R4: drop a lone ambiguous unit span ("kharab"/"mil" with no
      // preceding number) unless a currency marker was found for it -- see
      // shouldDropLoneAmbiguousUnit().
      if (shouldDropLoneAmbiguousUnit(tokens) && currency === null) continue;
      const res = evaluate(tokens);
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

  /** Synchronous CPU inspection: raw argmax BIO/class tags for every
   * character of the normalized text (before decode repair), plus the same
   * spans `parse()` would return. Uses the same window/padding logic as
   * parse(): for texts longer than MAX_LEN, each character is filled from
   * the first window that covers it. */
  inspect(text: string): Inspection {
    const t0 = performance.now();
    if (typeof text !== "string") text = ""; // R1: non-string -> "" (empty chars/spans)
    const normalized = normalizeText(text);
    const windows = makeWindows(normalized.length);
    const chars: CharTag[] = new Array(normalized.length);
    const filled = new Array<boolean>(normalized.length).fill(false);
    for (const win of windows) {
      const sub = normalized.slice(win.offset, win.offset + win.length);
      const padLen = paddedLength(sub.length);
      const ids = encodeChars(sub, this.charToId, this.idsScratch, padLen);
      const fw = forward(this.weights, ids, this.scratch);
      // `bio` reflects the Viterbi path (not per-position argmax) when the
      // loaded weights have a CRF head -- decoded once per window over its
      // real length, matching decodeForward()'s behaviour for parse().
      if (this.weights.crf) viterbi(fw.bioLogits, sub.length, this.weights.crf, this.bioIdsScratch);
      for (let t = 0; t < sub.length; t++) {
        const gi = win.offset + t;
        if (filled[gi]) continue;
        filled[gi] = true;
        const bioId = (this.weights.crf ? this.bioIdsScratch[t] : argmaxRow(fw.bioLogits, t * 3, 3)) as 0 | 1 | 2;
        const probs = new Float32Array(3);
        softmaxRow(fw.bioLogits, t * 3, 3, probs);
        const clsId = argmaxRow(fw.clsLogits, t * fw.nCls, fw.nCls);
        chars[gi] = {
          ch: sub[t],
          bio: bioId,
          bioProb: [probs[0], probs[1], probs[2]],
          cls: CLASSES[clsId],
          clsId,
        };
      }
    }
    const spans = this.parse(text);
    const ms = performance.now() - t0;
    return { text: normalized, chars, spans, ms };
  }

  /** Static info about the loaded model: architecture, parameter count and
   * vocab/class sizes. */
  modelInfo(): ModelInfo {
    const w = this.weights;
    let params = w.embed.data.length;
    for (const l of w.conv) params += l.w.data.length + l.b.length;
    params += w.bio.w.data.length + w.bio.b.length;
    params += w.cls.w.data.length + w.cls.b.length;
    if (w.crf) params += w.crf.trans.length + w.crf.start.length + w.crf.end.length; // 9 + 3 + 3 = 15
    const channels = w.conv.length ? Math.max(...w.conv.map((l) => l.w.shape[0])) : w.embed.shape[1];
    return {
      version: w.version,
      params,
      channels,
      embedDim: w.embed.shape[1],
      vocab: w.embed.shape[0],
      classes: w.classes.length,
      layers: w.conv.map((l) => ({ k: l.k, dilation: l.dilation, residual: l.residual })),
      crf: !!w.crf,
    };
  }

  /** Batched parse. Uses WebGPU when requested/available and the batch is
   * large enough to be worthwhile; falls back to CPU otherwise. */
  async parseBatch(texts: string[], opts: ParseOptions = {}): Promise<Sankhya[][]> {
    // R1: non-string entries (null/undefined/...) are treated as "" rather
    // than throwing, same as parse().
    texts = texts.map((t) => (typeof t === "string" ? t : ""));
    const backend = opts.backend ?? this.defaultBackend;
    const useGpu =
      backend === "webgpu" ||
      (backend === "auto" && texts.length >= 32 && (await probeWebGPU()));

    if (!useGpu) {
      return texts.map((t) => this.parse(t, { backend: "cpu" }));
    }

    const originals = texts;
    const norm = texts.map(normalizeText);

    // WebGPU path: batch texts that fit a single window; texts needing a
    // sliding window fall back to CPU (kept correct, still simple).
    const results: Sankhya[][] = new Array(texts.length);
    const gpuIdx: number[] = [];
    const cpuIdx: number[] = [];
    for (let i = 0; i < norm.length; i++) {
      if (norm[i].length <= 128) gpuIdx.push(i);
      else cpuIdx.push(i);
    }
    for (const i of cpuIdx) results[i] = this.parse(originals[i], { backend: "cpu" });

    if (gpuIdx.length > 0) {
      if (!this.gpu) this.gpu = new WebGPUBackend(this.weights);
      const maxRealLen = Math.max(...gpuIdx.map((i) => norm[i].length), 1);
      // row length includes PAD_TAIL right-padding beyond the longest real
      // text in the batch (see charset.ts) -- without it the longest row
      // would have zero pad context, same distribution-mismatch bug as an
      // unpadded single-text parse().
      const rowLen = paddedLength(maxRealLen);
      const batch = gpuIdx.length;
      const charIds = new Int32Array(batch * rowLen);
      const padId = this.charToId.get("<pad>") ?? 0;
      charIds.fill(padId);
      for (let bi = 0; bi < batch; bi++) {
        const text = norm[gpuIdx[bi]];
        const ids = encodeChars(text, this.charToId);
        charIds.set(ids, bi * rowLen);
      }
      const { bio, cls } = await this.gpu.run(charIds, batch, rowLen);
      const nCls = this.weights.cls.w.shape[0];
      for (let bi = 0; bi < batch; bi++) {
        const idx = gpuIdx[bi];
        const text = norm[idx];
        const original = originals[idx];
        const useOriginalSpan = original.length === text.length;
        const L = text.length;
        const bioLogits = new Float32Array(L * 3);
        const clsLogits = new Float32Array(L * nCls);
        for (let t = 0; t < L; t++) {
          bioLogits.set(bio.subarray((bi * rowLen + t) * 3, (bi * rowLen + t) * 3 + 3), t * 3);
          clsLogits.set(cls.subarray((bi * rowLen + t) * nCls, (bi * rowLen + t) * nCls + nCls), t * nCls);
        }
        const fw = { bioLogits, clsLogits, length: L, nCls };
        const decoded = this.decodeForward(text, fw, 0, L);
        if (useOriginalSpan) {
          for (const r of decoded) r.span = original.slice(r.start, r.end);
        }
        results[idx] = decoded;
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

/** Inspect a single string on the default parser -- see Parser.inspect(). */
export function inspect(text: string): Inspection {
  return getDefault().inspect(text);
}

/** Info about the default parser's loaded model -- see Parser.modelInfo(). */
export function modelInfo(): ModelInfo {
  return getDefault().modelInfo();
}

export function createParser(opts: CreateParserOptions = {}): Parser {
  return new Parser(opts);
}

// Union of every shipped language pack's currency marker list,
// longest-match first (see core.mergeLangPacks) -- used for currency
// detection so a mixed-script input ("₹2 lakh", "2 लाख रुपये", "दीड लाख
// रुपयांचा") is handled the same way regardless of which script/language
// wrote the currency marker. Adding a language pack means adding it here.
const CURRENCY_PACK = mergeLangPacks(HI_LATN, HI_DEVA, MR_DEVA);
