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

export function detectCurrency(text: string, start: number, end: number, pack: LangPack): "INR" | null {
  const before = text.slice(Math.max(0, start - 4), start);
  const after = text.slice(end, end + 6);

  const beforeStripped = before.replace(/\s+$/, "");
  for (const marker of pack.currency_markers_before ?? []) {
    const m = marker.replace(/\.+$/, "").toLowerCase();
    const bs = beforeStripped.toLowerCase();
    if (bs.endsWith(marker.toLowerCase()) || bs.endsWith(m)) return "INR";
  }

  const afterStripped = after.replace(/^\s+/, "");
  for (const word of pack.currency_words_after ?? []) {
    const w = word.toLowerCase();
    const a = afterStripped.toLowerCase();
    if (a.startsWith(w)) return "INR";
  }

  return null;
}
