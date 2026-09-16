import { test } from "node:test";
import assert from "node:assert/strict";
import { decodeSpans } from "../src/decode.ts";
import { CLASS_TO_ID } from "../src/classes.ts";

function cid(name: string): number {
  return CLASS_TO_ID[name];
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
