import { test } from "node:test";
import assert from "node:assert/strict";
import { decodeSpans } from "../src/decode.ts";
import { CLASS_TO_ID, CLASSES } from "../src/classes.ts";
import { evaluate } from "../src/core.ts";

function cid(name: string): number {
  return CLASS_TO_ID[name];
}

/** Run decodeSpans (which applies class-repair internally) on a whole-text
 * single span and evaluate the first resulting span's tokens via core, to
 * check repair + evaluate produce the expected final value. */
function repairAndEvaluate(text: string, rawCls: number[]) {
  const bio = new Array(text.length).fill(2);
  bio[0] = 1;
  const spans = decodeSpans(text, bio, rawCls);
  assert.equal(spans.length, 1, `expected exactly one span, got ${spans.length}`);
  const tokens: Array<[string, string]> = spans[0].tokens.map(([c, t]) => [CLASSES[c], t]);
  return evaluate(tokens);
}

test("trims leading/trailing SEP tokens", () => {
  // "  do lakh  " with a span tagged over the whole thing incl. spaces
  const text = " do lakh ";
  //             0123456789
  const bio = [1, 2, 2, 2, 2, 2, 2, 2, 2]; // B then I I I...
  const cls = [
    cid("SEP"),
    cid("CARD_2"),
    cid("CARD_2"),
    cid("SEP"),
    cid("UNIT_LAKH"),
    cid("UNIT_LAKH"),
    cid("UNIT_LAKH"),
    cid("UNIT_LAKH"),
    cid("SEP"),
  ];
  const spans = decodeSpans(text, bio, cls);
  assert.equal(spans.length, 1);
  assert.equal(spans[0].text, "do lakh");
  assert.equal(text.slice(spans[0].start, spans[0].end), "do lakh");
});

test("drops spans with no meaningful tokens", () => {
  const text = "  ";
  const bio = [1, 2];
  const cls = [cid("SEP"), cid("SEP")];
  const spans = decodeSpans(text, bio, cls);
  assert.equal(spans.length, 0);
});

test("I after O is ignored (treated as O)", () => {
  const text = "xdo";
  const bio = [0, 2, 1]; // O, I(ignored), B
  const cls = [cid("O"), cid("CARD_2"), cid("CARD_2")];
  const spans = decodeSpans(text, bio, cls);
  assert.equal(spans.length, 1);
  assert.equal(spans[0].text, "o");
});

test("confidence filtering drops low-confidence spans", () => {
  const text = "do";
  const bio = [1, 2];
  const cls = [cid("CARD_2"), cid("CARD_2")];
  const lowProbs = [
    [0.34, 0.33, 0.33],
    [0.34, 0.33, 0.33],
  ];
  const spansLow = decodeSpans(text, bio, cls, lowProbs);
  assert.equal(spansLow.length, 0);

  const highProbs = [
    [0.05, 0.9, 0.05],
    [0.05, 0.05, 0.9],
  ];
  const spansHigh = decodeSpans(text, bio, cls, highProbs);
  assert.equal(spansHigh.length, 1);
  assert.ok(spansHigh[0].confidence! >= 0.5);
});

test("multiple spans are all returned", () => {
  const text = "do lakh trin";
  //             0         1
  //             0123456789012
  const bio = new Array(text.length).fill(0);
  const cls = new Array(text.length).fill(cid("O"));
  // "do" span [0,2)
  bio[0] = 1;
  cls[0] = cid("CARD_2");
  cls[1] = cid("CARD_2");
  bio[1] = 2;
  // "lakh" span [3,7)
  bio[3] = 1;
  cls[3] = cid("UNIT_LAKH");
  for (let i = 4; i < 7; i++) {
    bio[i] = 2;
    cls[i] = cid("UNIT_LAKH");
  }
  const spans = decodeSpans(text, bio, cls);
  assert.equal(spans.length, 2);
  assert.equal(spans[0].text, "do");
  assert.equal(spans[1].text, "lakh");
});

// --- class-repair ----------------------------------------------------

test("class-repair: '1.5cr' with '.' mistagged O -> 15000000", () => {
  const text = "1.5cr";
  const raw = [cid("DIGITS"), cid("O"), cid("DIGITS"), cid("UNIT_CRORE"), cid("UNIT_CRORE")];
  const r = repairAndEvaluate(text, raw);
  assert.equal(r.value, 15000000);
});

test("class-repair: '2-3 lakh' with '-' mistagged O -> range [200000,300000]", () => {
  const text = "2-3 lakh";
  const raw = [
    cid("DIGITS"),
    cid("O"),
    cid("DIGITS"),
    cid("SEP"),
    cid("UNIT_LAKH"),
    cid("UNIT_LAKH"),
    cid("UNIT_LAKH"),
    cid("UNIT_LAKH"),
  ];
  const r = repairAndEvaluate(text, raw);
  assert.deepEqual(r.range, [200000, 300000]);
});

test("class-repair: 'das hazaar crore' with 'd' mistagged PFX_DHAI -> 1e11", () => {
  const text = "das hazaar crore";
  const raw = [
    cid("PFX_DHAI"), // d (wrong)
    cid("CARD_10"), // a
    cid("CARD_10"), // s
    cid("SEP"),
    cid("UNIT_HAZAAR"),
    cid("UNIT_HAZAAR"),
    cid("UNIT_HAZAAR"),
    cid("UNIT_HAZAAR"),
    cid("UNIT_HAZAAR"),
    cid("UNIT_HAZAAR"),
    cid("SEP"),
    cid("UNIT_CRORE"),
    cid("UNIT_CRORE"),
    cid("UNIT_CRORE"),
    cid("UNIT_CRORE"),
    cid("UNIT_CRORE"),
  ];
  const r = repairAndEvaluate(text, raw);
  assert.equal(r.value, 1e11);
  assert.equal(r.unit, "crore");
});

test("class-repair: 'five and a half crore' with the 'a' in 'half' mistagged CARD_8 -> 55000000", () => {
  // "five"=CARD_5, "and"/"a"/"half" all vote/repair to PFX_SAADHE (the
  // multi-word "and a half" idiom), then core's PFX-collapse-across-SEP
  // rule merges the three separated-by-SEP PFX_SAADHE tokens into one.
  // The injected error sits on the 'a' *inside* "half" (h-a-l-f): none of
  // "half"'s sub-runs reach length 3, so the whole run falls back to
  // majority vote (3x SAADHE vs 1x CARD_8) and self-corrects.
  const text = "five and a half crore";
  //             f i v e   a n d   a   h a l f   c r o r e
  const raw = [
    cid("CARD_5"),
    cid("CARD_5"),
    cid("CARD_5"),
    cid("CARD_5"), // five
    cid("SEP"), // space
    cid("PFX_SAADHE"),
    cid("PFX_SAADHE"),
    cid("PFX_SAADHE"), // and
    cid("SEP"), // space
    cid("PFX_SAADHE"), // a
    cid("SEP"), // space
    cid("PFX_SAADHE"), // h
    cid("CARD_8"), // a (wrong)
    cid("PFX_SAADHE"), // l
    cid("PFX_SAADHE"), // f
    cid("SEP"), // space
    cid("UNIT_CRORE"),
    cid("UNIT_CRORE"),
    cid("UNIT_CRORE"),
    cid("UNIT_CRORE"),
    cid("UNIT_CRORE"), // crore
  ];
  const r = repairAndEvaluate(text, raw);
  assert.equal(r.value, 55000000);
});

test("class-repair: 'dedhlakh' (joined, no space) tagged dedh(4)+lakh(4) -> 150000", () => {
  const text = "dedhlakh";
  const raw = [
    cid("PFX_DEDH"),
    cid("PFX_DEDH"),
    cid("PFX_DEDH"),
    cid("PFX_DEDH"), // dedh
    cid("UNIT_LAKH"),
    cid("UNIT_LAKH"),
    cid("UNIT_LAKH"),
    cid("UNIT_LAKH"), // lakh
  ];
  const r = repairAndEvaluate(text, raw);
  assert.equal(r.value, 150000);
});
