import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { parse } from "../src/index.ts";

interface FixtureSpan {
  start: number;
  end: number;
  value: number;
  range: [number, number] | null;
  unit: string | null;
  currency: string | null;
  classes: string; // space-joined
  // Added alongside the verified tier; may be absent in fixtures generated
  // before make_fixtures.py started writing them -- checked only when
  // present (see VERIFY_CONTRACT.md).
  verified?: boolean;
  tokens?: Array<[string, string]>;
  // Added alongside calibrated confidence; compared within 1e-6 (floats
  // crossing a JSON round-trip) rather than by exact equality, and only
  // when the fixture row carries them.
  confidence?: number;
  rawConfidence?: number;
}

interface FixtureRow {
  text: string;
  spans: FixtureSpan[];
}

const fixturePath = fileURLToPath(new URL("./fixtures/decoded.jsonl", import.meta.url));
const lines = readFileSync(fixturePath, "utf-8").trim().split("\n").filter(Boolean);

function normalize(
  span: {
    start: number;
    end: number;
    value: number;
    range?: [number, number];
    unit: string | null;
    currency: string | null;
    classes: string[];
    verified: boolean;
    tokens: Array<[string, string]>;
  },
  includeVerified: boolean,
) {
  const base: Record<string, unknown> = {
    start: span.start,
    end: span.end,
    value: span.value,
    range: span.range ?? null,
    unit: span.unit,
    currency: span.currency,
    classes: span.classes.join(" "),
  };
  if (includeVerified) {
    base.verified = span.verified;
    base.tokens = span.tokens;
  }
  return base;
}

test("parse() matches the python end-to-end decoded fixture for all texts", () => {
  let identical = 0;
  const mismatches: Array<{ text: string; got: unknown; want: unknown }> = [];
  let confChecked = 0;

  let skipped = 0;
  for (const line of lines) {
    const row = JSON.parse(line) as FixtureRow;
    // Astral characters (emoji) are ONE code point in Python but TWO UTF-16
    // code units (two <unk> ids) in JS, so the two sides do not see the same
    // input for such texts -- a documented limitation (see charset.test.ts).
    if (Array.from(row.text).length !== row.text.length) { skipped++; continue; }
    // Fixtures regenerated after VERIFY_CONTRACT.md carry `verified`/`tokens`
    // per span; older fixtures do not -- only assert those fields when the
    // fixture row actually has them.
    const includeVerified = row.spans.length > 0 && row.spans[0].verified !== undefined;
    const live = parse(row.text);
    const got = live.map((s) => normalize(s, includeVerified));
    const want = row.spans.map((s) => {
      const base: Record<string, unknown> = {
        start: s.start,
        end: s.end,
        value: s.value,
        range: s.range,
        unit: s.unit,
        currency: s.currency,
        classes: s.classes,
      };
      if (includeVerified) {
        base.verified = s.verified;
        base.tokens = s.tokens;
      }
      return base;
    });
    const same = JSON.stringify(got) === JSON.stringify(want);
    if (same) identical++;
    else mismatches.push({ text: row.text, got, want });

    // Confidences are floats, so they are compared with a tolerance
    // instead of going through the JSON.stringify equality above.
    if (same && row.spans.length && row.spans[0].confidence !== undefined) {
      for (let i = 0; i < row.spans.length; i++) {
        assert.ok(
          Math.abs(live[i].confidence - (row.spans[i].confidence as number)) < 1e-6,
          `confidence mismatch on ${JSON.stringify(row.text)}: ${live[i].confidence} vs ${row.spans[i].confidence}`,
        );
        assert.ok(
          Math.abs(live[i].rawConfidence - (row.spans[i].rawConfidence as number)) < 1e-6,
          `rawConfidence mismatch on ${JSON.stringify(row.text)}: ${live[i].rawConfidence} vs ${row.spans[i].rawConfidence}`,
        );
        confChecked++;
      }
    }
  }
  console.log(`e2e parity: ${confChecked} span confidences compared within 1e-6`);

  console.log(`e2e parity: ${identical}/${lines.length - skipped} texts identical to python end-to-end decode (${skipped} astral-char rows skipped)`);
  if (mismatches.length) {
    console.log("mismatches (first 5):", JSON.stringify(mismatches.slice(0, 5), null, 2));
  }
  assert.equal(identical, lines.length - skipped, `expected all ${lines.length} texts to match; ${mismatches.length} mismatched`);
});
