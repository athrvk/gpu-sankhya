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
}

interface FixtureRow {
  text: string;
  spans: FixtureSpan[];
}

const fixturePath = fileURLToPath(new URL("./fixtures/decoded.jsonl", import.meta.url));
const lines = readFileSync(fixturePath, "utf-8").trim().split("\n").filter(Boolean);

function normalize(span: {
  start: number;
  end: number;
  value: number;
  range?: [number, number];
  unit: string | null;
  currency: string | null;
  classes: string[];
}) {
  return {
    start: span.start,
    end: span.end,
    value: span.value,
    range: span.range ?? null,
    unit: span.unit,
    currency: span.currency,
    classes: span.classes.join(" "),
  };
}

test("parse() matches the python end-to-end decoded fixture for all texts", () => {
  let identical = 0;
  const mismatches: Array<{ text: string; got: unknown; want: unknown }> = [];

  for (const line of lines) {
    const row = JSON.parse(line) as FixtureRow;
    const got = parse(row.text).map(normalize);
    const want = row.spans.map((s) => ({
      start: s.start,
      end: s.end,
      value: s.value,
      range: s.range,
      unit: s.unit,
      currency: s.currency,
      classes: s.classes,
    }));
    const same = JSON.stringify(got) === JSON.stringify(want);
    if (same) identical++;
    else mismatches.push({ text: row.text, got, want });
  }

  console.log(`e2e parity: ${identical}/${lines.length} texts identical to python end-to-end decode`);
  if (mismatches.length) {
    console.log("mismatches (first 5):", JSON.stringify(mismatches.slice(0, 5), null, 2));
  }
  assert.equal(identical, lines.length, `expected all ${lines.length} texts to match; ${mismatches.length} mismatched`);
});
