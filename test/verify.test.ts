import { test } from "node:test";
import assert from "node:assert/strict";
import { verifyTokens } from "../src/verify.ts";

test("sava + lakh is verified", () => {
  assert.equal(
    verifyTokens([
      ["PFX_SAVA", "sava"],
      ["SEP", " "],
      ["UNIT_LAKH", "lakh"],
    ]),
    true,
  );
});

test("misspelt unit is not verified", () => {
  assert.equal(
    verifyTokens([
      ["PFX_SAVA", "sava"],
      ["SEP", " "],
      ["UNIT_LAKH", "lakhhh"],
    ]),
    false,
  );
});

test("Devanagari digits are verified as DIGITS", () => {
  assert.equal(verifyTokens([["DIGITS", "१२३"]]), true);
});

test("DIGITS token with a non-digit char is not verified", () => {
  assert.equal(verifyTokens([["DIGITS", "12a"]]), false);
});

test("SEP must be whitespace only", () => {
  assert.equal(
    verifyTokens([
      ["PFX_SAVA", "sava"],
      ["SEP", " "],
      ["UNIT_LAKH", "lakh"],
    ]),
    true,
  );
  assert.equal(
    verifyTokens([
      ["PFX_SAVA", "sava"],
      ["SEP", "x"],
      ["UNIT_LAKH", "lakh"],
    ]),
    false,
  );
});

test("RANGE accepts symbols and lexicon range words", () => {
  assert.equal(
    verifyTokens([
      ["DIGITS", "2"],
      ["RANGE", "-"],
      ["DIGITS", "3"],
    ]),
    true,
  );
  assert.equal(
    verifyTokens([
      ["DIGITS", "2"],
      ["RANGE", "se"],
      ["DIGITS", "3"],
    ]),
    true,
  );
  assert.equal(
    verifyTokens([
      ["DIGITS", "2"],
      ["RANGE", "xyz"],
      ["DIGITS", "3"],
    ]),
    false,
  );
});

test("O is verified only if the lexicon knows it as class O", () => {
  // "लाखों" (indefinite plural) is a lexicon word mapped to class O.
  assert.equal(
    verifyTokens([
      ["DIGITS", "2"],
      ["O", "लाखों"],
      ["DIGITS", "3"],
    ]),
    true,
  );
  assert.equal(
    verifyTokens([
      ["DIGITS", "2"],
      ["O", "notaword"],
      ["DIGITS", "3"],
    ]),
    false,
  );
});

test("DOT and COMMA must be exactly '.' and ','", () => {
  assert.equal(verifyTokens([["DOT", "."]]), false); // no content token -> false regardless
  assert.equal(
    verifyTokens([
      ["DIGITS", "2"],
      ["DOT", "."],
      ["DIGITS", "5"],
    ]),
    true,
  );
  assert.equal(
    verifyTokens([
      ["DIGITS", "2"],
      ["DOT", ","],
      ["DIGITS", "5"],
    ]),
    false,
  );
  assert.equal(
    verifyTokens([
      ["DIGITS", "2"],
      ["COMMA", ","],
      ["DIGITS", "5"],
    ]),
    true,
  );
  assert.equal(
    verifyTokens([
      ["DIGITS", "2"],
      ["COMMA", "x"],
      ["DIGITS", "5"],
    ]),
    false,
  );
});

test("multi-word english fraction phrase is looked up whole", () => {
  assert.equal(verifyTokens([["PFX_SAADHE", "and a half"]]), true);
});

test("empty token list is not verified", () => {
  assert.equal(verifyTokens([]), false);
});

test("a span with only SEP/RANGE/DOT (no content token) is not verified", () => {
  assert.equal(verifyTokens([["SEP", " "]]), false);
});

test("unknown surface for a PFX/CARD/UNIT class is not verified", () => {
  assert.equal(verifyTokens([["UNIT_LAKH", "notaword"]]), false);
});

test("RANGE word tokens carry their surrounding spaces (decoder emits ' किंवा ' as one token)", () => {
  assert.equal(verifyTokens([["CARD_3", "तीन"], ["RANGE", " किंवा "], ["CARD_4", "चार"], ["SEP", " "], ["UNIT_LAKH", "लाख"]]), true);
  assert.equal(verifyTokens([["CARD_3", "teen"], ["RANGE", " se "], ["CARD_4", "chaar"], ["SEP", " "], ["UNIT_LAKH", "lakh"]]), true);
  assert.equal(verifyTokens([["CARD_3", "teen"], ["RANGE", " xyz "], ["CARD_4", "chaar"], ["SEP", " "], ["UNIT_LAKH", "lakh"]]), false);
});

test("Gujarati digits count as DIGITS, mirroring python verify._NATIVE_DIGITS", () => {
  assert.equal(verifyTokens([["DIGITS", "૭૫૦૦"]]), true);
  assert.equal(verifyTokens([["DIGITS", "૧૨"], ["SEP", " "], ["UNIT_LAKH", "લાખ"]]), true);
  // Devanagari digits keep working, and a non-digit glyph still fails
  assert.equal(verifyTokens([["DIGITS", "२५"]]), true);
  assert.equal(verifyTokens([["DIGITS", "૭x"]]), false);
});

test("a bound number form verifies only immediately before its unit", () => {
  // બસો = 200: CARD_2 "બ" is justified by the "સો" that follows it
  assert.equal(verifyTokens([["CARD_2", "બ"], ["UNIT_SAU", "સો"]]), true);
  assert.equal(verifyTokens([["CARD_6", "છસ્"], ["UNIT_SAU", "સો"]]), true);
  // ... and never on its own, spaced, or before any other unit
  assert.equal(verifyTokens([["CARD_2", "બ"]]), false);
  assert.equal(verifyTokens([["CARD_2", "બ"], ["SEP", " "], ["UNIT_SAU", "સો"]]), false);
  assert.equal(verifyTokens([["CARD_2", "બ"], ["UNIT_HAZAAR", "હજાર"]]), false);
  // the free form is unaffected
  assert.equal(verifyTokens([["CARD_2", "બે"], ["SEP", " "], ["UNIT_LAKH", "લાખ"]]), true);
});
