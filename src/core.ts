// Port of python/sankhya/core.py — deterministic evaluation core.

import * as C from "./classes.ts";

export interface Result {
  value: number;
  range: [number, number] | null;
  unit: string | null;
  classes: string[];
  currency: string | null;
}

export interface LangPack {
  currency_markers_before: string[];
  currency_words_after: string[];
}

type Tok = [string, string];

function mergeNumbers(tokens: Tok[]): Array<[string, string | number]> {
  const out: Array<[string, string | number]> = [];
  let buf = "";
  let haveNum = false;
  const flush = () => {
    if (haveNum) {
      const cleaned = buf.replace(/,/g, "");
      const val = parseFloat(cleaned);
      out.push(["NUM", Number.isNaN(val) ? 0.0 : val]);
    }
    buf = "";
    haveNum = false;
  };
  for (const [cls, text] of tokens) {
    if (cls === "SEP") continue;
    if (cls === "DIGITS" || cls === "DOT" || cls === "COMMA") {
      buf += text;
      haveNum = true;
    } else {
      flush();
      out.push([cls, text]);
    }
  }
  flush();
  return out;
}

function flushCoef(num: number | null, pfx: string | null): number {
  if (pfx !== null) {
    const info = C.PREFIX_INFO[pfx];
    if (info === undefined) return num !== null ? num : 1;
    if (num !== null && info.op) {
      if (info.op === "sub25") return num - 0.25;
      if (info.op === "add25") return num + 0.25;
      if (info.op === "add5") return num + 0.5;
    }
    return info.standalone;
  }
  if (num !== null) return num;
  return 1;
}

function evalAmount(tokens: Tok[]): [number, string | null] {
  const merged = mergeNumbers(tokens);
  let accum = 0;
  let lastUnitVal: number | null = null;
  let maxUnitCls: string | null = null;
  let pendingNum: number | null = null;
  let pendingPfx: string | null = null;

  function closeTerm(unitCls: string | null) {
    const coef = flushCoef(pendingNum, pendingPfx);
    if (unitCls === null) {
      accum += coef * 1;
    } else {
      const newVal = C.unitValue(unitCls);
      if (lastUnitVal !== null && newVal > lastUnitVal) {
        accum = accum * newVal;
      } else {
        accum = accum + coef * newVal;
      }
      lastUnitVal = newVal;
      if (maxUnitCls === null || newVal > C.unitValue(maxUnitCls)) {
        maxUnitCls = unitCls;
      }
    }
    pendingNum = null;
    pendingPfx = null;
  }

  for (const [cls, text] of merged) {
    if (cls === "NUM") {
      if (pendingNum !== null) closeTerm(null);
      pendingNum = text as number;
    } else if (C.isCard(cls)) {
      if (pendingNum !== null) closeTerm(null);
      try {
        pendingNum = C.cardValue(cls);
      } catch {
        // ignore
      }
    } else if (C.isPrefix(cls)) {
      if (pendingPfx !== null) closeTerm(null);
      pendingPfx = cls;
    } else if (C.isUnit(cls)) {
      closeTerm(cls);
    } else {
      continue;
    }
  }

  if (pendingNum !== null || pendingPfx !== null) closeTerm(null);

  return [accum, maxUnitCls];
}

function num(v: number): number {
  const nearest = Math.round(v);
  const tol = Math.max(1e-6, Math.abs(v) * 1e-9);
  if (Math.abs(v - nearest) < tol) return nearest;
  return Math.round(v * 1e6) / 1e6;
}

/** Consecutive tokens with the same PFX_* class, separated only by SEP
 * tokens, collapse into one prefix token (e.g. "half a" = PFX_AADHA SEP
 * PFX_AADHA -> one PFX_AADHA; "three n half" = CARD_3 SEP PFX_SAADHE SEP
 * PFX_SAADHE -> CARD_3 SEP PFX_SAADHE). Without this, each repeated prefix
 * closes and re-opens a term on its own, double-counting the prefix's
 * standalone/combining value. */
function collapsePfxRuns(tokens: Tok[]): Tok[] {
  const out: Tok[] = [];
  let i = 0;
  while (i < tokens.length) {
    const [cls, text] = tokens[i];
    if (cls.startsWith("PFX_")) {
      let mergedText = text;
      let j = i + 1;
      for (;;) {
        let k = j;
        if (k < tokens.length && tokens[k][0] === "SEP") k++;
        if (k < tokens.length && tokens[k][0] === cls) {
          for (let m = j; m <= k; m++) mergedText += tokens[m][1];
          j = k + 1;
          continue;
        }
        break;
      }
      out.push([cls, mergedText]);
      i = j;
    } else {
      out.push(tokens[i]);
      i++;
    }
  }
  return out;
}

export function evaluate(rawTokens: Tok[]): Result {
  try {
    // `classes` in the Result reports the original (uncollapsed) token
    // sequence -- collapsing is purely an arithmetic-evaluation detail, not
    // part of the normalised class listing callers see.
    const allClasses = rawTokens.map((t) => t[0]);
    const tokens = collapsePfxRuns(rawTokens);

    const parts: Tok[][] = [[]];
    for (const [cls, text] of tokens) {
      if (cls === "RANGE") {
        if (parts.length < 2) parts.push([]);
        continue;
      }
      parts[parts.length - 1].push([cls, text]);
    }

    const amounts = parts.filter((p) => p.length > 0).map((p) => evalAmount(p));
    if (amounts.length === 0) {
      return { value: 0, range: null, unit: null, classes: allClasses, currency: null };
    }

    if (amounts.length === 1) {
      const [value, unit] = amounts[0];
      return {
        value: num(value),
        range: null,
        unit: unit ? C.UNIT_NAME[unit] ?? null : null,
        classes: allClasses,
        currency: null,
      };
    }

    let [v1, u1] = amounts[0];
    const [v2, u2] = amounts[1];

    // R6: a RANGE connector that actually sits between a DESCENDING
    // additive chain (e.g. "ek lakh dus hazaar" mistagged RANGE on the
    // space) -- both sides already carry their own unit and the left
    // value is strictly greater than the right -- is evaluated as one
    // additive amount instead of a [low, high] range. A genuine range
    // ("2-3 lakh", "paanch se sadhe saat lakh") has only ONE side carrying
    // a unit (the other is a bare number/prefix scaled by it below), so it
    // never hits this branch.
    if (u1 !== null && u2 !== null && v1 > v2) {
      const total = num(v1 + v2);
      return { value: total, range: null, unit: C.UNIT_NAME[u1] ?? null, classes: allClasses, currency: null };
    }

    let effUnit: string | null;
    if (u1 === null && u2 !== null) {
      v1 = v1 * C.unitValue(u2);
      effUnit = u2;
    } else {
      effUnit = u1 !== null ? u1 : u2;
    }

    const [low, high] = v1 <= v2 ? [v1, v2] : [v2, v1];
    return {
      value: num(low),
      range: [num(low), num(high)],
      unit: effUnit ? C.UNIT_NAME[effUnit] ?? null : null,
      classes: allClasses,
      currency: null,
    };
  } catch {
    return { value: 0, range: null, unit: null, classes: rawTokens.map((t) => t[0]), currency: null };
  }
}

/** Merge one or more LangPacks' marker lists into a single pack, de-duped,
 * sorted longest-first so `detectCurrency` tries the longest (most
 * specific) marker before a shorter one that might be its prefix (e.g.
 * "Rs." before "Rs", "रुपये" before a shorter overlapping marker). */
/** R3: true when a span's meaningful tokens are ONLY digits (with
 * DOT/COMMA/SEP glue) -- no UNIT_/PFX_/CARD_ token at all. Such a span is
 * dropped by the caller unless a currency marker is detected for it. */
export function isBareDigits(tokens: Tok[]): boolean {
  let hasDigits = false;
  for (const [cls] of tokens) {
    if (cls === "DIGITS") hasDigits = true;
    else if (cls.startsWith("UNIT_") || cls.startsWith("PFX_") || cls.startsWith("CARD_")) return false;
  }
  return hasDigits;
}

/** R3 (narrowed): whether a bare-digits span (see isBareDigits) with NO
 * detected currency should be dropped. Most bare numbers ARE amounts
 * (rent, prices, quantities: "15000", "2,50,00,000", "125,000") and must
 * be kept; only the two genuinely ambiguous shapes are dropped:
 *
 *   (a) short (<=4 digits) AND ungrouped (no comma) -- route numbers,
 *       OTPs, years, house numbers: "1", "66", "4521", "2024", "302".
 *   (b) very long (>=10 digits) -- phone numbers: "9876543210".
 *
 * A grouping comma (Indian or Western) is itself a strong signal of a
 * real amount, so it exempts an otherwise-short span from rule (a):
 * "1,00,000" (6 digits, comma) is kept, not dropped. */
export function shouldDropBareDigits(tokens: Tok[]): boolean {
  if (!isBareDigits(tokens)) return false;
  let digitCount = 0;
  let hasComma = false;
  for (const [cls, text] of tokens) {
    if (cls === "DIGITS") digitCount += text.length;
    else if (cls === "COMMA") hasComma = true;
  }
  if (digitCount <= 4 && !hasComma) return true;
  if (digitCount >= 10) return true;
  return false;
}

const IGNORED_GLUE_CLASSES = new Set(["SEP", "RANGE", "DOT", "COMMA", "O"]);

/** R4: whether a span consisting of a single ambiguous unit word (with no
 * preceding number) should be dropped. "kharab" (Hindi: usually "broken")
 * and "mil"/"million" (often just an English loanword/fragment) are only
 * genuine amount units when a number precedes them ("das kharab", "2 mil");
 * alone they are almost always false positives ("washing machine kharab ho
 * gaya", "mil ke rehna").
 *
 * True when the span's meaningful tokens (ignoring SEP/RANGE/DOT/COMMA/O
 * glue) consist of exactly one token whose class is UNIT_KHARAB or
 * UNIT_MILLION, and there is no CARD_-prefixed/DIGITS/PFX_-prefixed token
 * anywhere in the span. This is class-level (not surface-level), so it
 * needs no lexicon. */
export function shouldDropLoneAmbiguousUnit(tokens: Tok[]): boolean {
  const meaningful = tokens.filter(([cls]) => !IGNORED_GLUE_CLASSES.has(cls));
  if (meaningful.length !== 1) return false;
  const [cls] = meaningful[0];
  if (cls !== "UNIT_KHARAB" && cls !== "UNIT_MILLION") return false;
  for (const [c] of tokens) {
    if (c === "DIGITS" || c.startsWith("CARD_") || c.startsWith("PFX_")) return false;
  }
  return true;
}

export function mergeLangPacks(...packs: LangPack[]): LangPack {
  const byLenDesc = (a: string, b: string) => b.length - a.length;
  const dedupe = (lists: string[][]) => [...new Set(lists.flat())].sort(byLenDesc);
  return {
    currency_markers_before: dedupe(packs.map((p) => p.currency_markers_before ?? [])),
    currency_words_after: dedupe(packs.map((p) => p.currency_words_after ?? [])),
  };
}

/** Scan up to 8 chars before/after [start,end) for a currency marker. The
 * window has to be wider than the longest marker word (6 chars, e.g.
 * "rupaye"/"rupees"): the span boundary sits right after the number, so
 * the gap before the marker word starts (whitespace, and occasionally
 * punctuation like ", ") eats into a same-sized window and truncates the
 * word -- "1200 rupees" with a 6-char after-window is " rupee" (missing
 * the final "s"), so "rupees" never matches with startsWith(). 8 leaves
 * slack for a couple of separator chars ahead of the longest marker. */
export function detectCurrency(text: string, start: number, end: number, pack: LangPack): "INR" | null {
  const before = text.slice(Math.max(0, start - 8), start);
  const after = text.slice(end, end + 8);

  const beforeStripped = before.replace(/\s+$/, "");
  const bs = beforeStripped.toLowerCase();
  const markersBefore = [...(pack.currency_markers_before ?? [])].sort((a, b) => b.length - a.length);
  for (const marker of markersBefore) {
    const m = marker.replace(/\.+$/, "").toLowerCase();
    if (bs.endsWith(marker.toLowerCase()) || bs.endsWith(m)) return "INR";
  }

  const afterStripped = after.replace(/^\s+/, "");
  const a = afterStripped.toLowerCase();
  const wordsAfter = [...(pack.currency_words_after ?? [])].sort((x, y) => y.length - x.length);
  for (const word of wordsAfter) {
    const w = word.toLowerCase();
    if (a.startsWith(w)) return "INR";
  }

  return null;
}
