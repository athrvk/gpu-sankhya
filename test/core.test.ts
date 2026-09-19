import { test } from "node:test";
import assert from "node:assert/strict";
import { evaluate, detectCurrency, mergeLangPacks, isBareDigits, shouldDropBareDigits, shouldDropLoneAmbiguousUnit, hasLeadingZeroCoefficient } from "../src/core.ts";
import { HI_LATN } from "../src/lang-hi-latn.ts";
import { HI_DEVA } from "../src/lang-hi-deva.ts";
import { MR_DEVA } from "../src/lang-mr-deva.ts";
import { GU_GUJR } from "../src/lang-gu-gujr.ts";

type Tok = [string, string];
function ev(...toks: Tok[]) {
  return evaluate(toks);
}

test("paune do lakh", () => {
  const r = ev(["PFX_PAUNE", "paune"], ["SEP", " "], ["CARD_2", "do"], ["SEP", " "], ["UNIT_LAKH", "lakh"]);
  assert.equal(r.value, 175000);
});

test("ek crore bees lakh", () => {
  const r = ev(["CARD_1", "ek"], ["SEP", " "], ["UNIT_CRORE", "crore"], ["SEP", " "], ["CARD_20", "bees"], ["SEP", " "], ["UNIT_LAKH", "lakh"]);
  assert.equal(r.value, 12000000);
});

test("dhai sau", () => {
  const r = ev(["PFX_DHAI", "dhai"], ["SEP", " "], ["UNIT_SAU", "sau"]);
  assert.equal(r.value, 250);
});

test("2.5L", () => {
  const r = ev(["DIGITS", "2"], ["DOT", "."], ["DIGITS", "5"], ["UNIT_LAKH", "L"]);
  assert.equal(r.value, 250000);
});

test("12 lpa", () => {
  const r = ev(["DIGITS", "1"], ["DIGITS", "2"], ["SEP", " "], ["UNIT_LAKH", "LPA"]);
  assert.equal(r.value, 1200000);
});

test("sadhe 3 lakh", () => {
  const r2 = ev(["PFX_SAADHE", "sadhe"], ["SEP", " "], ["DIGITS", "3"], ["SEP", " "], ["UNIT_LAKH", "lakh"]);
  assert.equal(r2.value, 350000);
});

test("1,25,000", () => {
  const r = ev(
    ["DIGITS", "1"],
    ["COMMA", ","],
    ["DIGITS", "2"],
    ["DIGITS", "5"],
    ["COMMA", ","],
    ["DIGITS", "0"],
    ["DIGITS", "0"],
    ["DIGITS", "0"],
  );
  assert.equal(r.value, 125000);
});

test("sava lakh", () => {
  const r = ev(["PFX_SAVA", "sava"], ["SEP", " "], ["UNIT_LAKH", "lakh"]);
  assert.equal(r.value, 125000);
});

test("dedh crore", () => {
  const r = ev(["PFX_DEDH", "dedh"], ["SEP", " "], ["UNIT_CRORE", "crore"]);
  assert.equal(r.value, 15000000);
});

test("dhai lakh", () => {
  const r = ev(["PFX_DHAI", "dhai"], ["SEP", " "], ["UNIT_LAKH", "lakh"]);
  assert.equal(r.value, 250000);
});

test("20k", () => {
  const r = ev(["DIGITS", "2"], ["DIGITS", "0"], ["UNIT_HAZAAR", "k"]);
  assert.equal(r.value, 20000);
});

test("dedh-do lakh range", () => {
  const r = ev(["PFX_DEDH", "dedh"], ["RANGE", " "], ["CARD_2", "do"], ["SEP", " "], ["UNIT_LAKH", "lakh"]);
  assert.deepEqual(r.range, [150000, 200000]);
});

test("2-3 lakh range", () => {
  const r = ev(["DIGITS", "2"], ["RANGE", "-"], ["DIGITS", "3"], ["SEP", " "], ["UNIT_LAKH", "lakh"]);
  assert.deepEqual(r.range, [200000, 300000]);
  assert.equal(r.unit, "lakh");
});

test("two and a half lakh", () => {
  const r = ev(["CARD_2", "two"], ["SEP", " "], ["PFX_SAADHE", "and a half"], ["SEP", " "], ["UNIT_LAKH", "lakh"]);
  assert.equal(r.value, 250000);
});

test("half a crore", () => {
  const r = ev(["PFX_AADHA", "half a"], ["SEP", " "], ["UNIT_CRORE", "crore"]);
  assert.equal(r.value, 5000000);
});

test("pachas rupaye no unit", () => {
  const r = ev(["CARD_50", "pachas"]);
  assert.equal(r.value, 50);
  assert.equal(r.unit, null);
});

test("rs 500", () => {
  const r = ev(["DIGITS", "5"], ["DIGITS", "0"], ["DIGITS", "0"]);
  assert.equal(r.value, 500);
});

test("currency detection", () => {
  const text = "Rs 500 only";
  assert.equal(detectCurrency(text, 3, 6, HI_LATN), "INR");
  const text2 = "pachas rupaye chahiye";
  assert.equal(detectCurrency(text2, 0, 6, HI_LATN), "INR");
  const text3 = "do din baad";
  assert.equal(detectCurrency(text3, 0, 2, HI_LATN), null);
});

test("hi_deva currency detection: markers before", () => {
  const text = "₹500 only";
  assert.equal(detectCurrency(text, 1, 4, HI_DEVA), "INR");
  const text2 = "रु 500 only";
  assert.equal(detectCurrency(text2, 3, 6, HI_DEVA), "INR");
  const text3 = "रु. 500 only";
  assert.equal(detectCurrency(text3, 4, 7, HI_DEVA), "INR");
});

test("hi_deva currency detection: words after", () => {
  const text = "पचास रुपये चाहिए";
  assert.equal(detectCurrency(text, 0, 4, HI_DEVA), "INR");
  const text2 = "पचास रुपए चाहिए";
  assert.equal(detectCurrency(text2, 0, 4, HI_DEVA), "INR");
  const text3 = "दो दिन बाद";
  assert.equal(detectCurrency(text3, 0, 2, HI_DEVA), null);
});

test("mergeLangPacks: union of hi_latn + hi_deva, longest-match first", () => {
  const pack = mergeLangPacks(HI_LATN, HI_DEVA);
  // hi_latn behaviour preserved
  assert.equal(detectCurrency("Rs 500 only", 3, 6, pack), "INR");
  assert.equal(detectCurrency("pachas rupaye chahiye", 0, 6, pack), "INR");
  // hi_deva markers also recognised through the merged pack
  assert.equal(detectCurrency("₹500 only", 1, 4, pack), "INR");
  assert.equal(detectCurrency("पचास रुपये चाहिए", 0, 4, pack), "INR");
  // "Rs." (longer) must win over a bare "Rs" prefix match
  assert.equal(detectCurrency("Rs.500 only", 3, 6, pack), "INR");
  // negative case still negative through the merged pack
  assert.equal(detectCurrency("do din baad", 0, 2, pack), null);
});

test("mr_deva currency detection: markers before and words after", () => {
  assert.equal(detectCurrency("₹500 फक्त", 1, 4, MR_DEVA), "INR");
  assert.equal(detectCurrency("रु. 1200 झाले", 4, 8, MR_DEVA), "INR");
  assert.equal(detectCurrency("पन्नास रुपये हवेत", 0, 6, MR_DEVA), "INR");
  // the oblique stem covers every case-inflected form
  assert.equal(detectCurrency("सहा लाख रुपयांचा तोटा", 0, 7, MR_DEVA), "INR");
  assert.equal(detectCurrency("दोन दिवसांत येतो", 0, 3, MR_DEVA), null);
});

test("gu_gujr currency detection: markers before and words after", () => {
  assert.equal(detectCurrency("₹500 ફાઇનલ", 1, 4, GU_GUJR), "INR");
  assert.equal(detectCurrency("રૂ. 1200 થયા", 4, 8, GU_GUJR), "INR");
  assert.equal(detectCurrency("પચાસ રૂપિયા જોઈએ", 0, 4, GU_GUJR), "INR");
  // singular/colloquial and case-inflected forms
  assert.equal(detectCurrency("બાવીસ રૂપિયો કિલો", 0, 6, GU_GUJR), "INR");
  assert.equal(detectCurrency("છ લાખ રૂપિયાનું નુકસાન", 0, 5, GU_GUJR), "INR");
  assert.equal(detectCurrency("બે દિવસમાં આવું", 0, 2, GU_GUJR), null);
});

test("mergeLangPacks: adding mr_deva keeps the hi packs working", () => {
  // this is exactly the CURRENCY_PACK the parser builds in index.ts
  const pack = mergeLangPacks(HI_LATN, HI_DEVA, MR_DEVA);
  // every pack's markers are still recognised through the merged pack
  assert.equal(detectCurrency("Rs 500 only", 3, 6, pack), "INR");
  assert.equal(detectCurrency("pachas rupaye chahiye", 0, 6, pack), "INR");
  assert.equal(detectCurrency("पचास रुपये चाहिए", 0, 4, pack), "INR");
  assert.equal(detectCurrency("दोनशे रुपयांचे बिल", 0, 5, pack), "INR");
  assert.equal(detectCurrency("₹2,50,000 भरले", 1, 9, pack), "INR");
  // longest-match ordering survives the third pack
  assert.equal(detectCurrency("Rs.500 only", 3, 6, pack), "INR");
  // and a non-currency context is still negative
  assert.equal(detectCurrency("do din baad", 0, 2, pack), null);
  assert.equal(detectCurrency("दोन दिवसांत येतो", 0, 3, pack), null);
});

test("mergeLangPacks: the four-pack CURRENCY_PACK keeps every pack working", () => {
  // this is exactly the CURRENCY_PACK the parser builds in index.ts today
  const pack = mergeLangPacks(HI_LATN, HI_DEVA, MR_DEVA, GU_GUJR);
  assert.equal(detectCurrency("Rs 500 only", 3, 6, pack), "INR");
  assert.equal(detectCurrency("pachas rupaye chahiye", 0, 6, pack), "INR");
  assert.equal(detectCurrency("पचास रुपये चाहिए", 0, 4, pack), "INR");
  assert.equal(detectCurrency("दोनशे रुपयांचे बिल", 0, 5, pack), "INR");
  assert.equal(detectCurrency("બસો રૂપિયા આપ", 0, 3, pack), "INR");
  assert.equal(detectCurrency("₹2,50,000 ભર્યા", 1, 9, pack), "INR");
  // "રૂ" must not swallow the Marathi/Hindi markers or vice versa
  assert.equal(detectCurrency("રૂ. 1200 થયા", 4, 8, pack), "INR");
  assert.equal(detectCurrency("रु. 1200 झाले", 4, 8, pack), "INR");
  // and non-currency contexts are still negative in every language
  assert.equal(detectCurrency("do din baad", 0, 2, pack), null);
  assert.equal(detectCurrency("दोन दिवसांत येतो", 0, 3, pack), null);
  assert.equal(detectCurrency("બે દિવસમાં આવું", 0, 2, pack), null);
});

test("das hazaar crore", () => {
  const r = ev(["CARD_10", "das"], ["SEP", " "], ["UNIT_HAZAAR", "hazaar"], ["SEP", " "], ["UNIT_CRORE", "crore"]);
  assert.equal(r.value, 1e11);
  assert.equal(r.unit, "crore");
});

test("sau crore", () => {
  const r = ev(["UNIT_SAU", "sau"], ["SEP", " "], ["UNIT_CRORE", "crore"]);
  assert.equal(r.value, 1e9);
});

test("dedh sau crore", () => {
  const r = ev(["PFX_DEDH", "dedh"], ["SEP", " "], ["UNIT_SAU", "sau"], ["SEP", " "], ["UNIT_CRORE", "crore"]);
  assert.equal(r.value, 1.5e9);
});

test("2 lakh crore", () => {
  const r = ev(["DIGITS", "2"], ["SEP", " "], ["UNIT_LAKH", "lakh"], ["SEP", " "], ["UNIT_CRORE", "crore"]);
  assert.equal(r.value, 2e12);
});

test("saadhe teen sau crore", () => {
  const r = ev(
    ["PFX_SAADHE", "saadhe"],
    ["SEP", " "],
    ["CARD_3", "teen"],
    ["SEP", " "],
    ["UNIT_SAU", "sau"],
    ["SEP", " "],
    ["UNIT_CRORE", "crore"],
  );
  assert.equal(r.value, 3.5e9);
});

test("50 hazaar crore", () => {
  const r = ev(["DIGITS", "5"], ["DIGITS", "0"], ["SEP", " "], ["UNIT_HAZAAR", "hazaar"], ["SEP", " "], ["UNIT_CRORE", "crore"]);
  assert.equal(r.value, 5e11);
});

test("ek lakh bees hazaar crore", () => {
  const r = ev(
    ["CARD_1", "ek"],
    ["SEP", " "],
    ["UNIT_LAKH", "lakh"],
    ["SEP", " "],
    ["CARD_20", "bees"],
    ["SEP", " "],
    ["UNIT_HAZAAR", "hazaar"],
    ["SEP", " "],
    ["UNIT_CRORE", "crore"],
  );
  assert.equal(r.value, 1.2e12);
  assert.equal(r.unit, "crore");
});

test("half a lakh (PFX collapse: repeated PFX_AADHA separated only by SEP)", () => {
  const r = ev(["PFX_AADHA", "half"], ["SEP", " "], ["PFX_AADHA", "a"], ["SEP", " "], ["UNIT_LAKH", "lakh"]);
  assert.equal(r.value, 50000);
});

test("three n half lakh (PFX collapse: CARD_3 + collapsed PFX_SAADHE)", () => {
  const r = ev(
    ["CARD_3", "three"],
    ["SEP", " "],
    ["PFX_SAADHE", "n"],
    ["SEP", " "],
    ["PFX_SAADHE", "half"],
    ["SEP", " "],
    ["UNIT_LAKH", "lakh"],
  );
  assert.equal(r.value, 350000);
});

test("R6: descending RANGE chain (both sides carry units, left > right) evaluates additively", () => {
  // "ek lakh dus hazaar" mistagged RANGE on the space between.
  const r = ev(
    ["CARD_1", "ek"], ["SEP", " "], ["UNIT_LAKH", "lakh"],
    ["RANGE", " "],
    ["CARD_10", "dus"], ["SEP", " "], ["UNIT_HAZAAR", "hazaar"],
  );
  assert.equal(r.value, 110000);
  assert.equal(r.range, null);
  assert.equal(r.unit, "lakh");
});

test("R6: ascending range with only ONE side carrying a unit stays a true range", () => {
  const r = ev(["DIGITS", "2"], ["RANGE", "-"], ["DIGITS", "3"], ["SEP", " "], ["UNIT_LAKH", "lakh"]);
  assert.equal(r.value, 200000);
  assert.deepEqual(r.range, [200000, 300000]);
});

test("R6: true range with BOTH sides carrying units, ascending, stays a range", () => {
  const r = ev(
    ["DIGITS", "5"], ["SEP", " "], ["UNIT_LAKH", "lakh"],
    ["RANGE", "-"],
    ["DIGITS", "10"], ["SEP", " "], ["UNIT_LAKH", "lakh"],
  );
  assert.equal(r.value, 500000);
  assert.deepEqual(r.range, [500000, 1000000]);
});

test("R3: isBareDigits true for digits-only token sequences", () => {
  assert.equal(isBareDigits([["DIGITS", "1"]]), true);
  assert.equal(
    isBareDigits([["DIGITS", "1"], ["COMMA", ","], ["DIGITS", "00"], ["COMMA", ","], ["DIGITS", "000"]]),
    true,
  );
});

test("R3: isBareDigits false with a UNIT_/CARD_/PFX_ token present", () => {
  assert.equal(isBareDigits([["DIGITS", "2"], ["SEP", " "], ["UNIT_LAKH", "lakh"]]), false);
  assert.equal(isBareDigits([["CARD_2", "do"], ["SEP", " "], ["UNIT_LAKH", "lakh"]]), false);
  assert.equal(isBareDigits([["PFX_SAVA", "sava"], ["SEP", " "], ["UNIT_LAKH", "lakh"]]), false);
});

test("R3: isBareDigits false when there are no digits at all", () => {
  assert.equal(isBareDigits([["SEP", " "]]), false);
});

test("R3 (narrowed): should_drop_bare_digits drops short ungrouped digit-only spans", () => {
  assert.equal(shouldDropBareDigits([["DIGITS", "1"]]), true);
  assert.equal(shouldDropBareDigits([["DIGITS", "66"]]), true);
  assert.equal(shouldDropBareDigits([["DIGITS", "4521"]]), true);
  assert.equal(shouldDropBareDigits([["DIGITS", "2024"]]), true);
});

test("R3 (narrowed): should_drop_bare_digits drops phone-number-length spans", () => {
  assert.equal(shouldDropBareDigits([["DIGITS", "9876543210"]]), true);
});

test("R3 (narrowed): should_drop_bare_digits keeps mid-length bare amounts", () => {
  assert.equal(shouldDropBareDigits([["DIGITS", "15000"]]), false);
  assert.equal(shouldDropBareDigits([["DIGITS", "85000"]]), false);
  assert.equal(shouldDropBareDigits([["DIGITS", "62000"]]), false);
});

test("R3 (narrowed): should_drop_bare_digits keeps a grouped short amount ('1,00,000')", () => {
  const toks: Tok[] = [["DIGITS", "1"], ["COMMA", ","], ["DIGITS", "00"], ["COMMA", ","], ["DIGITS", "000"]];
  assert.equal(shouldDropBareDigits(toks), false);
});

test("R3 (narrowed): should_drop_bare_digits keeps a grouped 8-digit amount ('2,50,00,000')", () => {
  const toks: Tok[] = [
    ["DIGITS", "2"], ["COMMA", ","], ["DIGITS", "50"], ["COMMA", ","], ["DIGITS", "00"], ["COMMA", ","], ["DIGITS", "000"],
  ];
  assert.equal(shouldDropBareDigits(toks), false);
});

test("R3 (narrowed): should_drop_bare_digits is false when not bare digits at all", () => {
  assert.equal(shouldDropBareDigits([["DIGITS", "2"], ["SEP", " "], ["UNIT_LAKH", "lakh"]]), false);
});

test("R4: should_drop_lone_ambiguous_unit drops kharab/mil alone", () => {
  assert.equal(shouldDropLoneAmbiguousUnit([["UNIT_KHARAB", "kharab"]]), true);
  assert.equal(shouldDropLoneAmbiguousUnit([["UNIT_MILLION", "mil"]]), true);
  assert.equal(shouldDropLoneAmbiguousUnit([["UNIT_MILLION", "million"]]), true);
});

test("R4: should_drop_lone_ambiguous_unit ignores glue tokens", () => {
  const toks: Tok[] = [["SEP", " "], ["UNIT_KHARAB", "kharab"], ["O", ""]];
  assert.equal(shouldDropLoneAmbiguousUnit(toks), true);
});

test("R4: should_drop_lone_ambiguous_unit keeps a unit preceded by a number", () => {
  assert.equal(shouldDropLoneAmbiguousUnit([["DIGITS", "2"], ["SEP", " "], ["UNIT_MILLION", "mil"]]), false);
  assert.equal(shouldDropLoneAmbiguousUnit([["CARD_10", "das"], ["SEP", " "], ["UNIT_KHARAB", "kharab"]]), false);
});

test("R4: should_drop_lone_ambiguous_unit keeps a unit preceded by a prefix", () => {
  assert.equal(shouldDropLoneAmbiguousUnit([["PFX_DHAI", "dhai"], ["SEP", " "], ["UNIT_MILLION", "mil"]]), false);
});

test("R4: should_drop_lone_ambiguous_unit does not affect other unit classes", () => {
  assert.equal(shouldDropLoneAmbiguousUnit([["UNIT_LAKH", "lakh"]]), false);
  assert.equal(shouldDropLoneAmbiguousUnit([["UNIT_HAZAAR", "hazaar"]]), false);
});

test("R4: should_drop_lone_ambiguous_unit keeps multi-token spans", () => {
  const toks: Tok[] = [["UNIT_KHARAB", "kharab"], ["SEP", " "], ["UNIT_LAKH", "lakh"]];
  assert.equal(shouldDropLoneAmbiguousUnit(toks), false);
});

test("detectCurrency: window wide enough for 'rupees' after a short number", () => {
  const text = "1200 rupees mein mil gaya";
  assert.equal(detectCurrency(text, 0, 4, HI_LATN), "INR");
});

test("detectCurrency: glued symbol before digits ('₹85000')", () => {
  const text = "₹85000";
  assert.equal(detectCurrency(text, 1, 6, HI_LATN), "INR");
});

test("R7: 'teen hazaar paanch hazaar' juxtaposition range (no connector)", () => {
  const r = ev(
    ["CARD_3", "teen"], ["SEP", " "], ["UNIT_HAZAAR", "hazaar"],
    ["SEP", " "],
    ["CARD_5", "paanch"], ["SEP", " "], ["UNIT_HAZAAR", "hazaar"],
  );
  assert.equal(r.value, 3000);
  assert.deepEqual(r.range, [3000, 5000]);
  assert.equal(r.unit, "hazaar");
});

test("R7: 'bees lakh pachees lakh' juxtaposition range (no connector)", () => {
  const r = ev(
    ["CARD_20", "bees"], ["SEP", " "], ["UNIT_LAKH", "lakh"],
    ["SEP", " "],
    ["CARD_25", "pachees"], ["SEP", " "], ["UNIT_LAKH", "lakh"],
  );
  assert.equal(r.value, 2000000);
  assert.deepEqual(r.range, [2000000, 2500000]);
  assert.equal(r.unit, "lakh");
});

test("R7: '3 hazaar 5 hazaar' juxtaposition range with digits", () => {
  const r = ev(
    ["DIGITS", "3"], ["SEP", " "], ["UNIT_HAZAAR", "hazaar"],
    ["SEP", " "],
    ["DIGITS", "5"], ["SEP", " "], ["UNIT_HAZAAR", "hazaar"],
  );
  assert.equal(r.value, 3000);
  assert.deepEqual(r.range, [3000, 5000]);
});

test("R7: 'paanch lakh paanch lakh' (low == high) falls back to additive, no range", () => {
  // Deliberate choice: equal juxtaposed terms are ambiguous with plain
  // repetition/emphasis, not a genuine [x, x] range, so R7 requires
  // low < high strictly and this falls back to today's additive sum.
  const r = ev(
    ["CARD_5", "paanch"], ["SEP", " "], ["UNIT_LAKH", "lakh"],
    ["SEP", " "],
    ["CARD_5", "paanch"], ["SEP", " "], ["UNIT_LAKH", "lakh"],
  );
  assert.equal(r.value, 1000000);
  assert.equal(r.range, null);
});

test("R7: 'das hazaar crore' multiplicative stacking is unaffected (no coefficient on crore)", () => {
  const r = ev(
    ["CARD_10", "das"], ["SEP", " "], ["UNIT_HAZAAR", "hazaar"],
    ["SEP", " "],
    ["UNIT_CRORE", "crore"],
  );
  assert.equal(r.value, 1e11);
  assert.equal(r.range, null);
});

test("R7: 'ek lakh dus hazaar' descending additive chain is unaffected", () => {
  const r = ev(
    ["CARD_1", "ek"], ["SEP", " "], ["UNIT_LAKH", "lakh"],
    ["SEP", " "],
    ["CARD_10", "dus"], ["SEP", " "], ["UNIT_HAZAAR", "hazaar"],
  );
  assert.equal(r.value, 110000);
  assert.equal(r.range, null);
});

test("R7: 'do teen lakh' cardinal-only juxtaposition is unaffected (already an explicit RANGE upstream)", () => {
  // By the time this reaches evaluate(), decode-time repair has already
  // inserted an explicit RANGE token between the two cardinals, so this
  // exercises the existing RANGE-based path, not R7.
  const r = ev(
    ["CARD_2", "do"],
    ["RANGE", " "],
    ["CARD_3", "teen"], ["SEP", " "], ["UNIT_LAKH", "lakh"],
  );
  assert.deepEqual(r.range, [200000, 300000]);
  assert.equal(r.unit, "lakh");
});

test("Devanagari digits: pachees hazaar", () => {
  const r = ev(["DIGITS", "२५"], ["SEP", " "], ["UNIT_HAZAAR", "हजार"]);
  assert.equal(r.value, 25000);
});

test("Devanagari digits: bare", () => {
  const r = ev(["DIGITS", "७५००"]);
  assert.equal(r.value, 7500);
});

test("Devanagari digits: grouped", () => {
  const r = ev(["DIGITS", "२"], ["COMMA", ","], ["DIGITS", "५०"], ["COMMA", ","], ["DIGITS", "०००"]);
  assert.equal(r.value, 250000);
});

test("Devanagari digits: decimal", () => {
  const r = ev(["DIGITS", "१"], ["DOT", "."], ["DIGITS", "५"]);
  assert.equal(r.value, 1.5);
});

test("Devanagari digits: mixed ASCII and Devanagari", () => {
  const r = ev(["DIGITS", "1"], ["DOT", "."], ["DIGITS", "५"]);
  assert.equal(r.value, 1.5);
});

test("Gujarati digits: bare", () => {
  const r = ev(["DIGITS", "૭૫૦૦"]);
  assert.equal(r.value, 7500);
});

test("R8: 'tees paintees hazaar' juxtaposition range (no connector)", () => {
  // CARD_30 SEP CARD_35 SEP UNIT_HAZAAR means 30,000-35,000, not
  // 30 + 35*1000 = 35030.
  const r = ev(
    ["CARD_30", "tees"], ["SEP", " "], ["CARD_35", "paintees"], ["SEP", " "],
    ["UNIT_HAZAAR", "hazaar"],
  );
  assert.equal(r.value, 30000);
  assert.deepEqual(r.range, [30000, 35000]);
  assert.equal(r.unit, "hazaar");
});

test("R8: 'paach ten hazar' juxtaposition range", () => {
  const r = ev(
    ["CARD_5", "paach"], ["SEP", " "], ["CARD_10", "ten"], ["SEP", " "],
    ["UNIT_HAZAAR", "hazar"],
  );
  assert.equal(r.value, 5000);
  assert.deepEqual(r.range, [5000, 10000]);
  assert.equal(r.unit, "hazaar");
});

test("R8: 'do teen lakh' juxtaposition range at core level", () => {
  // At core level (no RANGE token inserted, unlike the decoder path), two
  // adjacent spelled cardinals before a unit form a range.
  const r = ev(
    ["CARD_2", "do"], ["SEP", " "], ["CARD_3", "teen"], ["SEP", " "],
    ["UNIT_LAKH", "lakh"],
  );
  assert.equal(r.value, 200000);
  assert.deepEqual(r.range, [200000, 300000]);
  assert.equal(r.unit, "lakh");
});

test("R8: 'ek sau' is unaffected (sau is a UNIT, not a second cardinal)", () => {
  const r = ev(["CARD_1", "ek"], ["SEP", " "], ["UNIT_SAU", "sau"]);
  assert.equal(r.value, 100);
  assert.equal(r.range, null);
});

test("R8: 'do hazaar paanch' additive is unaffected (cardinals separated by a unit)", () => {
  const r = ev(
    ["CARD_2", "do"], ["SEP", " "], ["UNIT_HAZAAR", "hazaar"], ["SEP", " "],
    ["CARD_5", "paanch"],
  );
  assert.equal(r.value, 2005);
  assert.equal(r.range, null);
});

test("R8: 'bees paanch' descending is unaffected", () => {
  const r = ev(["CARD_20", "bees"], ["SEP", " "], ["CARD_5", "paanch"]);
  assert.equal(r.value, 25);
  assert.equal(r.range, null);
});

test("R7 wins over R8 precedence: 'teen hazaar paanch das lakh'", () => {
  // R7 matches at the UNIT_HAZAAR / UNIT_LAKH term boundary (both terms
  // have explicit coefficients and unitValue(lakh) >= unitValue(hazaar)).
  // Within the second term, CARD_5 SEP CARD_10 is ALSO an R8-eligible
  // adjacent-cardinal pair, but R7 is checked first and must win for the
  // whole span.
  const r = ev(
    ["CARD_3", "teen"], ["SEP", " "], ["UNIT_HAZAAR", "hazaar"],
    ["SEP", " "],
    ["CARD_5", "paanch"], ["SEP", " "], ["CARD_10", "das"], ["SEP", " "],
    ["UNIT_LAKH", "lakh"],
  );
  assert.equal(r.value, 3000);
  assert.deepEqual(r.range, [3000, 1000005]);
  assert.equal(r.unit, "hazaar");
});

// --- R2c / R9 lexicon helpers --------------------------------------------

test("bound forms verify only in context", async () => {
  const { verifyTokens, isBoundForm, isLexiconOOnly } = await import("../src/verify.ts");
  assert.equal(verifyTokens([["CARD_2", "બ"], ["UNIT_SAU", "સો"]]), true);
  assert.equal(verifyTokens([["CARD_2", "બ"]]), false);
  assert.equal(isBoundForm("CARD_2", "બ", "સો"), true);
  assert.equal(isBoundForm("CARD_6", "છ", "સો"), false);
  assert.equal(isBoundForm("CARD_2", "બ", null), false);
});

test("R9: isLexiconOOnly recognises lexicon-declared ordinary words", async () => {
  const { isLexiconOOnly } = await import("../src/verify.ts");
  assert.equal(isLexiconOOnly("karodon"), true);
  assert.equal(isLexiconOOnly("करोडो"), true);
  assert.equal(isLexiconOOnly("KARODON"), true); // NFC-lowercased
  assert.equal(isLexiconOOnly("lakh"), false);
  assert.equal(isLexiconOOnly("zzzz-not-a-word"), false);
});

test("R10: leading-zero digits coefficient is never an amount", () => {
  assert.equal(hasLeadingZeroCoefficient([["DIGITS", "05"], ["SEP", " "], ["UNIT_CRORE", "CD"]]), true);
  assert.equal(hasLeadingZeroCoefficient([["DIGITS", "007"], ["SEP", " "], ["UNIT_LAKH", "lakh"]]), true);
  assert.equal(hasLeadingZeroCoefficient([["DIGITS", "0"], ["DOT", "."], ["DIGITS", "5"], ["SEP", " "], ["UNIT_LAKH", "lakh"]]), false);
  assert.equal(hasLeadingZeroCoefficient([["DIGITS", "1"], ["DOT", "."], ["DIGITS", "05"], ["SEP", " "], ["UNIT_LAKH", "lakh"]]), false);
  assert.equal(hasLeadingZeroCoefficient([["DIGITS", "50000"]]), false);
  assert.equal(hasLeadingZeroCoefficient([["DIGITS", "०५"], ["SEP", " "], ["UNIT_HAZAAR", "हज़ार"]]), true);
});
