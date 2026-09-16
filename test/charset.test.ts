import { test } from "node:test";
import assert from "node:assert/strict";
import { normalizeText } from "../src/charset.ts";
import { parse } from "../src/index.ts";

test("normalizeText: lowercases ASCII", () => {
  assert.equal(normalizeText("Rs 500 ONLY"), "rs 500 only");
});

test("normalizeText: maps Devanagari digits to ASCII 1:1", () => {
  assert.equal(normalizeText("२.५ लाख"), "2.5 लाख");
  assert.equal(normalizeText("०१२३४५६७८९"), "0123456789");
});

test("normalizeText: NFC-normalizes decomposed input", () => {
  // "é" (U+00E9) has a canonical NFC composition but is also representable,
  // non-canonically, as the decomposed pair "e" (U+0065) + combining acute
  // accent (U+0301). NFC folds the decomposed form back to the single
  // precomposed code point, so both spellings must normalize identically
  // and to the same (shorter) length -- exercising the same NFC codepath
  // normalizeText() applies to any Devanagari input with non-canonically-
  // ordered or decomposed combining marks.
  const composed = "café"; // café
  const decomposed = "café"; // cafe + combining acute
  assert.notEqual(composed, decomposed, "sanity: the two spellings differ in code units");
  assert.equal(composed.length, 4);
  assert.equal(decomposed.length, 5);
  assert.equal(normalizeText(decomposed), normalizeText(composed));
  assert.equal(normalizeText(decomposed).length, 4);
});

test("normalizeText: is idempotent on already-normalized ASCII (no-op for fixtures)", () => {
  const s = "sava lakh ka budget hai";
  assert.equal(normalizeText(s), s);
});

test("parse(): span is a substring of the ORIGINAL input when NFC is a length no-op", () => {
  const text = "Mera Budget 2.5L hai";
  const results = parse(text);
  assert.ok(results.length > 0);
  for (const r of results) {
    assert.equal(r.span, text.slice(r.start, r.end));
  }
});

test("parse(): offsets are into normalizeText(input); span text matches lowercased form when case differs", () => {
  const text = "MRP Rs.2.5L only";
  const results = parse(text);
  assert.ok(results.length > 0);
  const r = results[0];
  // start/end index normalizeText(text), which (for this ASCII input) has
  // the same length as `text`, so both raw and normalized slicing agree
  // positionally -- only casing differs.
  assert.equal(normalizeText(text).slice(r.start, r.end), r.span.toLowerCase());
});

test("astral character (documented limitation): offsets after it are shifted vs. Python code-point offsets", () => {
  // "😀" (U+1F600) is astral -- JS sees it as a surrogate PAIR (2 UTF-16
  // code units), Python (and this package's documented target semantics)
  // sees it as ONE code point. gpu-sankhya deliberately keeps UTF-16
  // code-unit indexing (see charset.ts encodeChars doc comment), so the
  // "2 lakh" span here lands 1 UTF-16 unit later than a code-point-indexed
  // (Python-equivalent) run would report it at.
  const emoji = "\u{1F600}";
  assert.equal(emoji.length, 2, "sanity: astral char is 2 UTF-16 code units in JS");
  const text = `${emoji} 2 lakh`;
  const results = parse(text);
  assert.ok(results.length > 0);
  const r = results[0];
  // Python code-point offset would be 2 (1 code point for emoji + 1 space);
  // JS UTF-16 offset is 3 (2 code units for emoji + 1 space) -- one higher.
  assert.equal(r.start, 3, "documented: UTF-16 offset is +1 vs. Python's code-point offset for this input");
  assert.equal(r.span, "2 lakh");
});
