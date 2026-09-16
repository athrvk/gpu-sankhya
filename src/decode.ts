// Port of python/sankhya/decode.py — strict BIO decoding + filtering + confidence.

import { CLASSES } from "./classes.ts";

const TRIM_CLASSES = new Set(["SEP", "RANGE", "DOT", "COMMA"]);
const MEANINGFUL_PREFIXES = ["PFX_", "CARD_", "UNIT_"];

function isMeaningful(clsName: string): boolean {
  return MEANINGFUL_PREFIXES.some((p) => clsName.startsWith(p)) || clsName === "DIGITS";
}

export interface DecodedSpan {
  start: number;
  end: number;
  text: string;
  tokens: Array<[number, string]>;
  confidence: number | null;
}

export function decodeSpans(
  text: string,
  bioIds: number[] | Int32Array,
  clsIds: number[] | Int32Array,
  bioProbs?: Float32Array[] | number[][] | null,
): DecodedSpan[] {
  const n = text.length;
  const spans: Array<[number, number]> = [];
  let start: number | null = null;
  for (let i = 0; i < n; i++) {
    const b = bioIds[i];
    if (b === 1) {
      if (start !== null) spans.push([start, i]);
      start = i;
    } else if (b === 2) {
      // I only valid inside an already-open span
    } else {
      if (start !== null) {
        spans.push([start, i]);
        start = null;
      }
    }
  }
  if (start !== null) spans.push([start, n]);

  const out: DecodedSpan[] = [];
  for (const [s, e] of spans) {
    let runStart = s;
    const origToks: Array<[number, number, number]> = [];
    for (let j = s + 1; j <= e; j++) {
      if (j === e || clsIds[j] !== clsIds[runStart]) {
        origToks.push([clsIds[runStart], runStart, j]);
        runStart = j;
      }
    }

    let lo = 0;
    let hi = origToks.length;
    while (lo < hi && TRIM_CLASSES.has(CLASSES[origToks[lo][0]])) lo++;
    while (hi > lo && TRIM_CLASSES.has(CLASSES[origToks[hi - 1][0]])) hi--;
    const kept = origToks.slice(lo, hi);
    if (kept.length === 0) continue;
    const newStart = kept[0][1];
    const newEnd = kept[kept.length - 1][2];

    if (!kept.some(([cid]) => isMeaningful(CLASSES[cid]))) continue;

    let confidence: number | null = null;
    if (bioProbs) {
      const probs: number[] = [];
      for (let j = newStart; j < newEnd; j++) {
        probs.push(Math.max(...bioProbs[j]));
      }
      confidence = probs.length ? probs.reduce((a, b) => a + b, 0) / probs.length : 0.0;
      if (confidence < 0.5) continue;
    }

    out.push({
      start: newStart,
      end: newEnd,
      text: text.slice(newStart, newEnd),
      tokens: kept.map(([cid, a, b]): [number, string] => [cid, text.slice(a, b)]),
      confidence: confidence as number,
    });
  }
  return out;
}
