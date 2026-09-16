import { test } from "node:test";
import assert from "node:assert/strict";
import { decodeSpans, repairClasses, repairWordIntegrity, letterRuns } from "../src/decode.ts";
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
  // "a-o" -- "-" separates "a" and "o" into their own one-char letters-runs,
  // so the CARD_ token on "o" alone still fully covers its own run and
  // survives R2's word-integrity check.
  const text = "a-o";
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

// --- BIO repairs (applied before class-repair) ------------------------

test("BIO bridge: space after 'five' mistagged O -> 55000000", () => {
  const text = "five and a half crore";
  const bio = new Array(text.length).fill(2);
  bio[0] = 1;
  bio[text.indexOf(" ")] = 0; // single stray O right after "five"

  const cls = new Array(text.length).fill(cid("O"));
  const tag = (word: string, from: number, clsName: string): number => {
    const start = text.indexOf(word, from);
    for (let i = start; i < start + word.length; i++) cls[i] = cid(clsName);
    return start + word.length;
  };
  let pos = 0;
  pos = tag("five", pos, "CARD_5");
  pos = tag("and", pos, "PFX_SAADHE");
  pos = tag("a", pos, "PFX_SAADHE");
  pos = tag("half", pos, "PFX_SAADHE");
  tag("crore", pos, "UNIT_CRORE");

  const spans = decodeSpans(text, bio, cls);
  assert.equal(spans.length, 1, `expected the bridged O to keep this as one span, got ${spans.length}`);
  const tokens: Array<[string, string]> = spans[0].tokens.map(([c, t]) => [CLASSES[c], t]);
  const r = evaluate(tokens);
  assert.equal(r.value, 55000000);
});

test("digit-extension: last '0' of '15000' mistagged O -> 15000", () => {
  const text = "15000/month";
  const bio = new Array(text.length).fill(0);
  bio[0] = 1;
  bio[1] = 2;
  bio[2] = 2;
  bio[3] = 2; // span so far covers "1500" (indices 0-3); index 4 ('0') stays O below

  const cls = new Array(text.length).fill(cid("O"));
  for (let i = 0; i < 5; i++) cls[i] = cid("DIGITS"); // all five digit chars, including the mistagged one

  const spans = decodeSpans(text, bio, cls);
  assert.equal(spans.length, 1);
  assert.equal(spans[0].text, "15000", "digit-extension should pull the trailing '0' back into the span");
  const tokens: Array<[string, string]> = spans[0].tokens.map(([c, t]) => [CLASSES[c], t]);
  const r = evaluate(tokens);
  assert.equal(r.value, 15000);
});

test("Devanagari letters-run: matras/nukta are treated as letters, not punctuation", () => {
  // "डेढ़" = ड (base) + े (vowel sign, \p{M}) + ढ (base) + ़ (nukta, \p{M}).
  // Combining marks are not `\p{L}`, only `\p{M}` -- if charType() only
  // checked `\p{L}`, each mark would fall into the punctuation branch and
  // force a SEP mid-word.
  const text = "डेढ़";
  assert.equal(text.length, 4, "sanity: 4 UTF-16 code units");
  const ids = new Array(text.length).fill(cid("PFX_DEDH"));
  const repaired = repairClasses(text, ids);
  for (const id of repaired) {
    assert.notEqual(CLASSES[id], "SEP", `expected no SEP inside "${text}", got repaired=${repaired.map((r) => CLASSES[r])}`);
  }
});

test("Devanagari: डेढ़ लाख (\"one and a half lakh\") -> 150000", () => {
  const text = "डेढ़ लाख";
  const bio = new Array(text.length).fill(2);
  bio[0] = 1;
  const cls = new Array(text.length);
  let pos = 0;
  const tag = (chars: string, cls_: string) => {
    for (let i = 0; i < chars.length; i++) cls[pos + i] = cid(cls_);
    pos += chars.length;
  };
  tag("डेढ़", "PFX_DEDH");
  tag(" ", "SEP");
  tag("लाख", "UNIT_LAKH");

  const spans = decodeSpans(text, bio, cls);
  assert.equal(spans.length, 1, `expected exactly one span, got ${spans.length}`);
  const tokens: Array<[string, string]> = spans[0].tokens.map(([c, t]) => [CLASSES[c], t]);
  const r = evaluate(tokens);
  assert.equal(r.value, 150000);
  assert.equal(r.unit, "lakh");
});

test("R2: word integrity -- a letters-run that is only PARTIALLY meaningful (some chars fell back to O) has ALL its meaningful chars wiped too", () => {
  const text = "10 km";
  const cls = [cid("DIGITS"), cid("DIGITS"), cid("SEP"), cid("UNIT_HAZAAR"), cid("O")];
  const out = repairWordIntegrity(text, 0, text.length, cls, letterRuns(text));
  assert.deepEqual(out.slice(3, 5).map((c) => CLASSES[c]), ["O", "O"]);
});

test("R2: word integrity -- a whole-run single-char symbol unit is kept", () => {
  const text = "20k";
  const cls = [cid("DIGITS"), cid("DIGITS"), cid("UNIT_HAZAAR")];
  const out = repairWordIntegrity(text, 0, text.length, cls, letterRuns(text));
  assert.equal(CLASSES[out[2]], "UNIT_HAZAAR");
});

test("R2: word integrity -- 'dedhlakh' joined compound (both sub-runs meaningful) stays untouched", () => {
  const text = "dedhlakh";
  const cls = [
    ...new Array(4).fill(cid("PFX_DEDH")),
    ...new Array(4).fill(cid("UNIT_LAKH")),
  ];
  const out = repairWordIntegrity(text, 0, text.length, cls, letterRuns(text));
  assert.deepEqual(
    out.map((c) => CLASSES[c]),
    [...new Array(4).fill("PFX_DEDH"), ...new Array(4).fill("UNIT_LAKH")],
  );
});

test("R4: connector between a closed UNIT_ amount and a fresh DIGITS amount is a real range", () => {
  const text = "2 lakh/3 lakh";
  const bio = [1, ...new Array(text.length - 1).fill(2)];
  const cls = [
    cid("DIGITS"), cid("SEP"), cid("UNIT_LAKH"), cid("UNIT_LAKH"), cid("UNIT_LAKH"), cid("UNIT_LAKH"),
    cid("O"), // "/"
    cid("DIGITS"), cid("SEP"), cid("UNIT_LAKH"), cid("UNIT_LAKH"), cid("UNIT_LAKH"), cid("UNIT_LAKH"),
  ];
  const spans = decodeSpans(text, bio, cls);
  assert.equal(spans.length, 1);
  const tokClasses = spans[0].tokens.map(([c]) => CLASSES[c]);
  assert.deepEqual(tokClasses, ["DIGITS", "SEP", "UNIT_LAKH", "RANGE", "DIGITS", "SEP", "UNIT_LAKH"]);
});

test("R4: PFX_ directly glued to UNIT_ (one compound number) keeps the hyphen SEP", () => {
  const text = "dedh-lakh";
  const raw = [
    ...new Array(4).fill(cid("PFX_DHAI")),
    cid("O"),
    ...new Array(4).fill(cid("UNIT_LAKH")),
  ];
  const repaired = repairClasses(text, raw);
  assert.deepEqual(repaired.slice(4, 5).map((c) => CLASSES[c]), ["SEP"]);
});

test("R5: possessive apostrophe+letters trimmed from the end of the amount", () => {
  const text = "2 lakh's ka scam";
  const bio = [1, ...new Array("2 lakh's".length - 1).fill(2), ...new Array(text.length - "2 lakh's".length).fill(0)];
  const cls = [
    cid("DIGITS"), cid("SEP"), cid("UNIT_LAKH"), cid("UNIT_LAKH"), cid("UNIT_LAKH"), cid("UNIT_LAKH"), cid("O"), cid("O"),
    ...new Array(text.length - 8).fill(cid("O")),
  ];
  const spans = decodeSpans(text, bio, cls);
  assert.equal(spans.length, 1);
  assert.equal(spans[0].text, "2 lakh");
});

test("R5: possessive with no space before the apostrophe", () => {
  const text = "2lakh's";
  const bio = [1, ...new Array(text.length - 1).fill(2)];
  const cls = [cid("DIGITS"), cid("UNIT_LAKH"), cid("UNIT_LAKH"), cid("UNIT_LAKH"), cid("UNIT_LAKH"), cid("O"), cid("O")];
  const spans = decodeSpans(text, bio, cls);
  assert.equal(spans.length, 1);
  assert.equal(spans[0].text, "2lakh");
});
