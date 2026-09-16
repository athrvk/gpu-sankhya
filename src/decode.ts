// Port of python/sankhya/decode.py — strict BIO decoding + filtering + confidence.

import { CLASSES, CLASS_TO_ID } from "./classes.ts";

const TRIM_CLASSES = new Set(["SEP", "RANGE", "DOT", "COMMA"]);
const MEANINGFUL_PREFIXES = ["PFX_", "CARD_", "UNIT_"];
const NUMERIC_PREFIXES = ["PFX_", "CARD_", "UNIT_"];

function isMeaningful(clsName: string): boolean {
  return MEANINGFUL_PREFIXES.some((p) => clsName.startsWith(p)) || clsName === "DIGITS";
}

function isNumericClass(clsName: string): boolean {
  return NUMERIC_PREFIXES.some((p) => clsName.startsWith(p));
}

const ID_O = CLASS_TO_ID["O"];
const ID_SEP = CLASS_TO_ID["SEP"];
const ID_RANGE = CLASS_TO_ID["RANGE"];
const ID_DIGITS = CLASS_TO_ID["DIGITS"];
const ID_DOT = CLASS_TO_ID["DOT"];
const ID_COMMA = CLASS_TO_ID["COMMA"];

type CharType = "letter" | "digit" | "space" | "punct";

function charType(c: string): CharType {
  if (/[a-z]/i.test(c)) return "letter";
  if (/[0-9]/.test(c)) return "digit";
  if (/\s/.test(c)) return "space";
  return "punct";
}

interface Run {
  type: CharType;
  start: number; // local index into the span
  end: number;
}

function splitRuns(spanText: string): Run[] {
  const runs: Run[] = [];
  let i = 0;
  const n = spanText.length;
  while (i < n) {
    const t = charType(spanText[i]);
    let j = i + 1;
    while (j < n && charType(spanText[j]) === t) j++;
    runs.push({ type: t, start: i, end: j });
    i = j;
  }
  return runs;
}

/** Whole-run majority vote over numeric classes (PFX_/CARD_/UNIT_), ignoring
 * O/SEP/RANGE/DIGITS/DOT/COMMA votes; ties broken by the class of the
 * longest contiguous same-class sub-run within [start,end); no numeric
 * votes at all -> O. Used as the fallback when sub-run smoothing (below)
 * finds no qualifying (length>=3) neighbour to adopt a class from. */
function runMajority(ids: number[] | Int32Array, start: number, end: number): number {
  const counts = new Map<number, number>();
  for (let i = start; i < end; i++) {
    const cid = ids[i] as number;
    if (isNumericClass(CLASSES[cid])) counts.set(cid, (counts.get(cid) ?? 0) + 1);
  }
  if (counts.size === 0) return ID_O;
  let maxCount = -1;
  for (const c of counts.values()) if (c > maxCount) maxCount = c;
  const tied = [...counts.entries()].filter(([, c]) => c === maxCount).map(([cid]) => cid);
  if (tied.length === 1) return tied[0];
  let bestLen = -1;
  let bestId = tied[0];
  for (const cid of tied) {
    let curLen = 0;
    let maxLen = 0;
    for (let i = start; i < end; i++) {
      if (ids[i] === cid) {
        curLen++;
        if (curLen > maxLen) maxLen = curLen;
      } else {
        curLen = 0;
      }
    }
    if (maxLen > bestLen || (maxLen === bestLen && cid < bestId)) {
      bestLen = maxLen;
      bestId = cid;
    }
  }
  return bestId;
}

interface SubRun {
  start: number;
  end: number;
  cls: number;
}

/** Sub-run smoothing within one letters run [start,end): find maximal
 * sub-runs of identical raw class; a sub-run of length >= 3 keeps its own
 * class; a shorter (1-2 char) sub-run adopts the class of whichever
 * immediately adjacent sub-run (left/right) qualifies with length >= 3
 * (prefer the longer if both qualify, tie -> left); if neither neighbour
 * qualifies, that sub-run falls back to the whole-run majority vote. This
 * preserves joined forms like "dedhlakh" (dedh|lakh, both length 4) while
 * still fixing a short mistagged run like "d|as" (neither sub-run reaches
 * length 3, so both fall back to majority -> CARD_10). Returns
 * [start, end, classId) triples covering the whole run. */
function repairLetterRun(ids: number[] | Int32Array, start: number, end: number): Array<[number, number, number]> {
  const subRuns: SubRun[] = [];
  let i = start;
  while (i < end) {
    let j = i + 1;
    while (j < end && ids[j] === ids[i]) j++;
    subRuns.push({ start: i, end: j, cls: ids[i] as number });
    i = j;
  }

  const fallback = runMajority(ids, start, end);
  const out: Array<[number, number, number]> = [];
  for (let si = 0; si < subRuns.length; si++) {
    const sr = subRuns[si];
    const len = sr.end - sr.start;
    if (len >= 3) {
      out.push([sr.start, sr.end, sr.cls]);
      continue;
    }
    const left = si > 0 ? subRuns[si - 1] : null;
    const right = si < subRuns.length - 1 ? subRuns[si + 1] : null;
    const leftQualifies = left !== null && left.end - left.start >= 3;
    const rightQualifies = right !== null && right.end - right.start >= 3;
    let winner: number;
    if (leftQualifies && rightQualifies) {
      const leftLen = left!.end - left!.start;
      const rightLen = right!.end - right!.start;
      winner = rightLen > leftLen ? right!.cls : left!.cls; // tie -> left
    } else if (leftQualifies) {
      winner = left!.cls;
    } else if (rightQualifies) {
      winner = right!.cls;
    } else {
      winner = fallback;
    }
    out.push([sr.start, sr.end, winner]);
  }
  return out;
}

/** Class-repair: split a span into maximal runs by char type (letters / digits /
 * whitespace / other punctuation) and normalise the class tag of each run
 * deterministically from that structure, rather than trusting the raw
 * per-char model output. Operates on local (span-relative) indices;
 * `ids` is the model's per-char class ids for the span, `spanText` the
 * corresponding substring. Returns a new array of the same length. */
export function repairClasses(spanText: string, ids: number[] | Int32Array): number[] {
  const n = spanText.length;
  const out = new Array<number>(n);
  const runs = splitRuns(spanText);

  for (let ri = 0; ri < runs.length; ri++) {
    const run = runs[ri];
    const { type, start, end } = run;

    if (type === "digit") {
      for (let i = start; i < end; i++) out[i] = ID_DIGITS;
      continue;
    }

    if (type === "space") {
      let anyRange = false;
      for (let i = start; i < end; i++) {
        if (ids[i] === ID_RANGE) {
          anyRange = true;
          break;
        }
      }
      const id = anyRange ? ID_RANGE : ID_SEP;
      for (let i = start; i < end; i++) out[i] = id;
      continue;
    }

    if (type === "letter") {
      for (const [i, j, winner] of repairLetterRun(ids, start, end)) {
        for (let p = i; p < j; p++) out[p] = winner;
      }
      continue;
    }

    // punctuation
    const prevType = ri > 0 ? runs[ri - 1].type : null;
    const nextType = ri < runs.length - 1 ? runs[ri + 1].type : null;
    let id = ID_SEP;
    if (end - start === 1) {
      const ch = spanText[start];
      if ((ch === "." || ch === ",") && prevType === "digit" && nextType === "digit") {
        id = ch === "." ? ID_DOT : ID_COMMA;
      } else if ((ch === "-" || ch === "–" || ch === "/") && prevType === "digit" && nextType === "digit") {
        id = ID_RANGE;
      } else if (ch === "-" && prevType === "letter" && nextType === "letter") {
        id = ID_SEP;
      } else {
        id = ID_SEP;
      }
    }
    for (let i = start; i < end; i++) out[i] = id;
  }

  return out;
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
    const spanText = text.slice(s, e);
    const localIds = Array.from({ length: e - s }, (_, i) => clsIds[s + i]);
    const repaired = repairClasses(spanText, localIds);

    let runStart = s;
    const origToks: Array<[number, number, number]> = [];
    for (let j = s + 1; j <= e; j++) {
      if (j === e || repaired[j - s] !== repaired[runStart - s]) {
        origToks.push([repaired[runStart - s], runStart, j]);
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
