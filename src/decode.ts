// Port of python/sankhya/decode.py — strict BIO decoding + filtering + confidence.

import { CLASSES, CLASS_TO_ID } from "./classes.ts";

const TRIM_CLASSES = new Set(["SEP", "RANGE", "DOT", "COMMA"]);
// R2/R5: after word-integrity/possessive repair a partially-retagged word
// can leave an "O" token sitting at a span edge; trimming it away too (in
// addition to the original SEP/RANGE/DOT/COMMA) is how start/end get
// re-derived to the min/max of the remaining non-O chars.
const TRIM_CLASSES_FINAL = new Set([...TRIM_CLASSES, "O"]);
const MEANINGFUL_PREFIXES = ["PFX_", "CARD_", "UNIT_"];
const NUMERIC_PREFIXES = ["PFX_", "CARD_", "UNIT_"];
const RANGE_CONNECTORS = new Set(["-", "–", "—", "/"]); // hyphen, en-dash, em-dash, slash
const SYMBOL_UNIT_CHARS = new Set(["k", "K", "l", "L"]);
const POSSESSIVE_RE = /['’]([A-Za-zऀ-ॿ]{1,2})(?![A-Za-zऀ-ॿ])/g;

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

// Devanagari letters (base consonants/vowels) and combining marks (matras,
// nukta, virama/halant, anusvara, chandrabindu -- U+0900-U+097F) must count
// as "letter" for run-splitting, or every matra (e.g. the ि in डेढ़) would
// be classified as punctuation and forced into its own SEP run, splitting
// what should be one letters-run mid-word. `\p{L}` (Unicode "Letter"
// category) alone excludes combining marks -- they're category `\p{M}`
// ("Mark: nonspacing"/"spacing combining"/"enclosing") -- so both are
// required. This mirrors the Python side, which cannot use `str.isalpha()`
// for the same reason (combining marks are not alphabetic per Python
// either) and instead checks unicodedata category startswith("L") or
// startswith("M"). Devanagari digits (U+0966-U+096F) are excluded from
// both categories already (Unicode category Nd), and normalizeText() maps
// them to ASCII digits before this ever runs, so they fall into the
// "digit" branch below regardless. Danda (।) and double danda (॥) are
// punctuation (category Po), not \p{L}/\p{M}, so they correctly fall
// through to "punct".
const LETTER_OR_MARK_RE = /\p{L}|\p{M}/u;

function charType(c: string): CharType {
  if (LETTER_OR_MARK_RE.test(c)) return "letter";
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
  let bestFirstPos = Infinity;
  let bestId = tied[0];
  for (const cid of tied) {
    let curLen = 0;
    let maxLen = 0;
    let firstPos = Infinity;
    for (let i = start; i < end; i++) {
      if (ids[i] === cid) {
        if (firstPos === Infinity) firstPos = i;
        curLen++;
        if (curLen > maxLen) maxLen = curLen;
      } else {
        curLen = 0;
      }
    }
    // longest contiguous sub-run wins; a further tie prefers whichever
    // class occurs first (leftmost) in the run.
    if (maxLen > bestLen || (maxLen === bestLen && firstPos < bestFirstPos)) {
      bestLen = maxLen;
      bestFirstPos = firstPos;
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

function isMeaningfulName(clsName: string): boolean {
  return MEANINGFUL_PREFIXES.some((p) => clsName.startsWith(p)) || clsName === "DIGITS";
}

/** Best-guess resulting class name for a run, without mutating the working
 * class array. Used by the R4 connector check to look at a letters-run
 * neighbour that has not been processed yet (it comes after the connector
 * in scan order). */
function runReprClass(runType: CharType, rs: number, re: number, ids: number[] | Int32Array): string | null {
  if (runType === "digit") return "DIGITS";
  if (runType === "letter") {
    const winners = repairLetterRun(ids, rs, re);
    const counts = new Map<number, number>();
    for (const [i, j, cls] of winners) {
      counts.set(cls, (counts.get(cls) ?? 0) + (j - i));
    }
    let bestId = -1;
    let bestCount = -1;
    for (const [cls, count] of counts) {
      if (count > bestCount) {
        bestCount = count;
        bestId = cls;
      }
    }
    return bestId >= 0 ? CLASSES[bestId] : null;
  }
  return null;
}

/** The adjacent digit/letter run in `direction`, skipping at most one
 * intervening space run (so "2 - 3" and "2-3" are treated the same). */
function effectiveNeighbor(runs: Run[], idx: number, direction: 1 | -1): Run | null {
  let j = idx + direction;
  if (j < 0 || j >= runs.length) return null;
  if (runs[j].type === "space") {
    j += direction;
    if (j < 0 || j >= runs.length) return null;
  }
  const r = runs[j];
  return r.type === "digit" || r.type === "letter" ? r : null;
}

/** R4: true when the connector genuinely sits between two SEPARATE numeric
 * amounts (a real range), not inside one compound number.
 *
 * The left neighbour must be able to END an amount by itself (any
 * meaningful class: it may be a bare number/prefix, e.g. "2-3", or a
 * number that has already picked up a unit, e.g. "2 lakh/3 lakh"). The
 * right neighbour must be able to START a fresh amount: DIGITS/CARD_/PFX_,
 * but NOT a bare UNIT_ -- a unit can only ATTACH to a number that precedes
 * it, so "dedh-lakh" (PFX_DHAI - UNIT_LAKH, one compound number "1.5
 * lakh") must stay SEP, while "2 lakh/3 lakh" (UNIT_LAKH - DIGITS, two
 * separate amounts) becomes RANGE. */
function connectorIsRange(runs: Run[], idx: number, ids: number[] | Int32Array): boolean {
  const left = effectiveNeighbor(runs, idx, -1);
  const right = effectiveNeighbor(runs, idx, 1);
  if (!left || !right) return false;
  const leftCls = runReprClass(left.type, left.start, left.end, ids);
  const rightCls = runReprClass(right.type, right.start, right.end, ids);
  if (!leftCls || !rightCls || !isMeaningfulName(leftCls)) return false;
  return rightCls === "DIGITS" || rightCls.startsWith("CARD_") || rightCls.startsWith("PFX_");
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
      } else if (RANGE_CONNECTORS.has(ch) && connectorIsRange(runs, ri, ids)) {
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

/** BIO repair (a): bridge a single stray O back into I when it sits between
 * an already-open span (B or I immediately before) and a continuing I
 * immediately after -- a lone one-char dropout shouldn't split the span.
 * Decisions are all made from the ORIGINAL array (not the array being
 * built), so this only ever bridges a single-char gap, never a run of
 * multiple consecutive O's. */
function bridgeBio(bioIds: number[] | Int32Array, n: number): number[] {
  const out = Array.from({ length: n }, (_, i) => bioIds[i] as number);
  for (let i = 1; i < n - 1; i++) {
    if (bioIds[i] === 0 && (bioIds[i - 1] === 1 || bioIds[i - 1] === 2) && bioIds[i + 1] === 2) {
      out[i] = 2;
    }
  }
  return out;
}

function isDigitChar(c: string | undefined): boolean {
  return c !== undefined && c >= "0" && c <= "9";
}

/** BIO repair (b): if a span ends right on a digit and the following
 * char(s) are also digits, extend the span through the rest of that digit
 * run (a stray O mid-digit-run shouldn't truncate a number). */
function extendDigitRun(text: string, s: number, e: number): number {
  if (e <= s || !isDigitChar(text[e - 1])) return e;
  let ee = e;
  while (ee < text.length && isDigitChar(text[ee])) ee++;
  return ee;
}

const MAX_WORD_RUN_LEN = 24;

/** Like `repairLetterRun`'s sub-run smoothing, but returns a class id ONLY
 * when every sub-run resolves from real, DIRECT evidence -- its own
 * length >= 3, or adoption from an immediately adjacent sub-run of length
 * >= 3 -- and all of those resolved classes agree. If any sub-run would
 * need the whole-run majority-vote fallback (no qualifying neighbour on
 * either side), that's diffuse/weak evidence -- this returns null rather
 * than trust it, even if a fallback vote would happen to be unanimous
 * (e.g. a 6-char run with 3 scattered chars of one class and 3 O's:
 * majority-vote alone would call that unanimous, but no sub-run of it
 * ever reaches 3 in a row on its own or via a qualifying neighbour, so it
 * must stay untrusted). */
function runUnanimousClass(clsIds: number[] | Int32Array, rs: number, re: number): number | null {
  const subRuns: SubRun[] = [];
  let i = rs;
  while (i < re) {
    let j = i + 1;
    while (j < re && clsIds[j] === clsIds[i]) j++;
    subRuns.push({ start: i, end: j, cls: clsIds[i] as number });
    i = j;
  }

  const resolved: number[] = [];
  for (let si = 0; si < subRuns.length; si++) {
    const sr = subRuns[si];
    if (sr.end - sr.start >= 3) {
      resolved.push(sr.cls);
      continue;
    }
    const left = si > 0 ? subRuns[si - 1] : null;
    const right = si < subRuns.length - 1 ? subRuns[si + 1] : null;
    const leftQualifies = left !== null && left.end - left.start >= 3;
    const rightQualifies = right !== null && right.end - right.start >= 3;
    if (leftQualifies && rightQualifies) {
      const leftLen = left!.end - left!.start;
      const rightLen = right!.end - right!.start;
      resolved.push(rightLen > leftLen ? right!.cls : left!.cls); // tie -> left
    } else if (leftQualifies) {
      resolved.push(left!.cls);
    } else if (rightQualifies) {
      resolved.push(right!.cls);
    } else {
      return null; // would need the whole-run fallback -- untrusted
    }
  }
  const uniq = new Set(resolved);
  return uniq.size === 1 ? resolved[0] : null;
}

/** BIO repair (c): a word-class (UNIT_/PFX_/CARD_) token whose predicted
 * span only PARTIALLY covers its letter word is not necessarily a bad
 * match -- if the whole run's RAW per-char classes unanimously resolve
 * (via `runUnanimousClass`, direct-evidence-only sub-run smoothing) to a
 * SINGLE word class, the partial coverage is just low-confidence BIO
 * dropout on an otherwise-agreed word, not a genuinely mixed/ambiguous
 * run. Extend the span's BIO to cover the whole run (I for every char, B
 * at the run start) instead of leaving it for R2 to drop -- this also
 * merges spans that were split by an internal O gap inside that run back
 * into a single span (e.g. "unnasi" B I I O I O -> B I I I I I).
 *
 * A run whose raw classes do NOT resolve to one unanimous word class this
 * way (e.g. only part of the run predicts a word class, or the agreement
 * is only visible via `repairLetterRun`'s diffuse whole-run majority-vote
 * fallback rather than direct evidence) is left alone, so R2's drop still
 * applies to it. Runs longer than 24 chars are skipped. */
export function extendWordIntegrityBio(
  bioIds: number[] | Int32Array,
  clsIds: number[] | Int32Array,
  runs: Array<[number, number]>,
): number[] {
  const n = bioIds.length;
  const out = Array.from({ length: n }, (_, i) => bioIds[i] as number);
  for (const [rs, re] of runs) {
    const length = re - rs;
    if (length === 0 || length > MAX_WORD_RUN_LEN) continue;
    let anyCovered = false;
    let allCovered = true;
    for (let k = rs; k < re; k++) {
      const covered = bioIds[k] === 1 || bioIds[k] === 2;
      if (covered) anyCovered = true;
      else allCovered = false;
    }
    if (!anyCovered || allCovered) continue; // nothing to extend
    const unanimous = runUnanimousClass(clsIds, rs, re);
    if (unanimous === null || !isWordClass(CLASSES[unanimous])) continue;
    for (let k = rs; k < re; k++) out[k] = 2;
    out[rs] = 1;
  }
  return out;
}

/** [start, end) maximal letter/mark runs over the WHOLE text (not just one
 * span) -- word integrity needs the true word boundaries, which can extend
 * outside a span that was cut short mid-word. */
export function letterRuns(text: string): Array<[number, number]> {
  const runs: Array<[number, number]> = [];
  let i = 0;
  const n = text.length;
  while (i < n) {
    if (charType(text[i]) === "letter") {
      let j = i + 1;
      while (j < n && charType(text[j]) === "letter") j++;
      runs.push([i, j]);
      i = j;
    } else {
      i++;
    }
  }
  return runs;
}

function isWordClass(cname: string): boolean {
  return MEANINGFUL_PREFIXES.some((p) => cname.startsWith(p)) && cname !== "DIGITS";
}

/** R2: a word-class (UNIT_/PFX_/CARD_) token must cover its entire letter
 * word. Checked per WHOLE letters-run rather than per token, so a run
 * legitimately split into several back-to-back meaningful tokens (e.g.
 * "dedhlakh" = PFX_DEDH|UNIT_LAKH, both length-4 sub-runs) is left alone; a
 * run that is only PARTIALLY meaningful (some chars fall back to O, e.g.
 * "km" tagged UNIT_.../O) has ALL its meaningful chars retagged O too -- a
 * partial match without an explanation for the rest of the word is
 * untrustworthy. A single-character symbol unit (k/K/l/L) is additionally
 * only valid when the following character is not a letter, even when it
 * fully (and only) covers its own run. */
export function repairWordIntegrity(
  text: string,
  s: number,
  e: number,
  ids: number[] | Int32Array,
  runs: Array<[number, number]>,
): number[] {
  const out = Array.from({ length: ids.length }, (_, i) => ids[i] as number);
  for (const [rs, re] of runs) {
    if (re <= s || rs >= e) continue; // run doesn't touch this span at all
    const crs = Math.max(rs, s);
    const cre = Math.min(re, e);
    let fullyCovered = rs >= s && re <= e;
    if (fullyCovered) {
      for (let k = rs; k < re; k++) {
        if (!isWordClass(CLASSES[out[k]])) {
          fullyCovered = false;
          break;
        }
      }
    }
    if (fullyCovered) {
      if (re - rs === 1 && CLASSES[out[rs]].startsWith("UNIT_") && SYMBOL_UNIT_CHARS.has(text[rs])) {
        if (re < text.length && charType(text[re]) === "letter") out[rs] = ID_O;
      }
      continue;
    }
    for (let k = crs; k < cre; k++) {
      if (isWordClass(CLASSES[out[k]])) out[k] = ID_O;
    }
  }
  return out;
}

/** R5: an apostrophe followed by 1-2 letters at the end of a word (e.g.
 * "lakh's") is a possessive/genitive suffix, not part of the amount --
 * retag it (and those letters) O. */
function repairPossessive(text: string, s: number, e: number, ids: number[] | Int32Array): number[] {
  const out = Array.from({ length: ids.length }, (_, i) => ids[i] as number);
  const spanText = text.slice(s, e);
  POSSESSIVE_RE.lastIndex = 0;
  let m: RegExpExecArray | null;
  while ((m = POSSESSIVE_RE.exec(spanText))) {
    for (let k = s + m.index; k < s + m.index + m[0].length; k++) out[k] = ID_O;
  }
  return out;
}

export function decodeSpans(
  text: string,
  bioIds: number[] | Int32Array,
  clsIds: number[] | Int32Array,
  bioProbs?: Float32Array[] | number[][] | null,
): DecodedSpan[] {
  const n = text.length;
  const allLetterRuns = letterRuns(text);
  let bridged = bridgeBio(bioIds, n);
  bridged = extendWordIntegrityBio(bridged, clsIds, allLetterRuns);
  const spans: Array<[number, number]> = [];
  let start: number | null = null;
  for (let i = 0; i < n; i++) {
    const b = bridged[i];
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
  for (let k = 0; k < spans.length; k++) {
    spans[k] = [spans[k][0], extendDigitRun(text, spans[k][0], spans[k][1])];
  }

  let workCls = Array.from({ length: n }, (_, i) => clsIds[i] as number);

  const out: DecodedSpan[] = [];
  for (const [s, e] of spans) {
    const spanText = text.slice(s, e);
    const localIds = Array.from({ length: e - s }, (_, i) => workCls[s + i]);
    const repaired = repairClasses(spanText, localIds);
    for (let i = s; i < e; i++) workCls[i] = repaired[i - s];
    workCls = repairWordIntegrity(text, s, e, workCls, allLetterRuns);
    workCls = repairPossessive(text, s, e, workCls);

    let runStart = s;
    const origToks: Array<[number, number, number]> = [];
    for (let j = s + 1; j <= e; j++) {
      if (j === e || workCls[j] !== workCls[runStart]) {
        origToks.push([workCls[runStart], runStart, j]);
        runStart = j;
      }
    }

    // trim leading/trailing trim-classes (SEP/RANGE/DOT/COMMA, plus O --
    // R2/R5 above can leave a now-meaningless O token at an edge, and
    // re-deriving start/end to the min/max of the remaining non-O chars
    // means trimming it away here too)
    let lo = 0;
    let hi = origToks.length;
    while (lo < hi && TRIM_CLASSES_FINAL.has(CLASSES[origToks[lo][0]])) lo++;
    while (hi > lo && TRIM_CLASSES_FINAL.has(CLASSES[origToks[hi - 1][0]])) hi--;
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
