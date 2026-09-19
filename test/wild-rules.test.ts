// The real-text precision rules (R11-R18), mirroring
// python/tests/test_wild_rules.py case for case. Each rule exists because of
// a concrete failure shape in python/data_wild/REPORT.md section 4.
import { test } from "node:test";
import assert from "node:assert/strict";
import { shouldDropLonePrefix, evaluate } from "../src/core.ts";
import {
  hasBlockedSurface,
  isBlockedSurface,
  isKnownForm,
  levenshteinLE,
  shouldDropAmbiguousForm,
  shouldDropLooseSymbolUnit,
  shouldDropUnknownUnit,
  shouldDropUnsupportedBareCardinal,
  verifyTokens,
} from "../src/verify.ts";
import { decodeSpans } from "../src/decode.ts";
import { CLASSES, CLASS_TO_ID } from "../src/classes.ts";

type Tok = [string, string];

// --- R11: lone prefix ------------------------------------------------------

test("R11 drops a lone prefix", () => {
  // "ढाई साल" = two and a half YEARS, "sade hue tamatar" = rotten tomatoes
  assert.ok(shouldDropLonePrefix([["PFX_DHAI", "ढाई"]]));
  assert.ok(shouldDropLonePrefix([["PFX_SAADHE", "sade"]]));
  assert.ok(shouldDropLonePrefix([["SEP", " "], ["PFX_AADHA", "adha"], ["O", "x"]]));
});

test("R11 keeps a prefix with a number or unit", () => {
  assert.ok(!shouldDropLonePrefix([["PFX_DHAI", "ढाई"], ["SEP", " "], ["UNIT_LAKH", "लाख"]]));
  assert.ok(
    !shouldDropLonePrefix([
      ["PFX_SAADHE", "sadhe"], ["SEP", " "], ["CARD_4", "char"], ["SEP", " "], ["UNIT_LAKH", "lakh"],
    ]),
  );
});

// --- R12: ambiguous forms --------------------------------------------------

const AMBIGUOUS_ONLY: Tok[][] = [
  [["UNIT_SAU", "so"]],
  [["CARD_60", "sath"]],
  [["CARD_60", "saath"]],
  [["UNIT_LAKH", "peti"]],
  [["UNIT_ARAB", "arab"]],
  [["UNIT_ARAB", "अरब"]],
  [["UNIT_ARAB", "અરબ"]],
  [["CARD_1", "एक"]],
  [["CARD_9", "सऊदी"], ["SEP", " "], ["UNIT_ARAB", "अरब"]],
];

for (const tokens of AMBIGUOUS_ONLY) {
  test(`R12 drops a span held up only by ambiguous surfaces: ${tokens.map(([, t]) => t).join(" ")}`, () => {
    assert.ok(shouldDropAmbiguousForm(tokens));
  });
}

const AMBIGUOUS_SUPPORTED: Tok[][] = [
  [["CARD_1", "ek"], ["SEP", " "], ["UNIT_ARAB", "arab"]],
  [["DIGITS", "100"], ["SEP", " "], ["UNIT_ARAB", "अरब"]],
  [["CARD_2", "do"], ["SEP", " "], ["UNIT_LAKH", "peti"]],
  [["CARD_10", "das"], ["SEP", " "], ["UNIT_KHARAB", "kharab"]],
  [["CARD_20", "bees"], ["SEP", " "], ["UNIT_LAKH", "lac"], ["SEP", " "],
    ["CARD_25", "pachees"], ["SEP", " "], ["UNIT_LAKH", "lac"]],
  [["CARD_19", "unnis"], ["SEP", " "], ["UNIT_SAU", "sau"], ["SEP", " "], ["CARD_60", "sath"]],
  [["CARD_1", "एक"], ["SEP", " "], ["UNIT_LAKH", "लाख"]],
];

for (const tokens of AMBIGUOUS_SUPPORTED) {
  test(`R12 keeps a span with one unambiguous number word: ${tokens.map(([, t]) => t).join(" ")}`, () => {
    assert.ok(!shouldDropAmbiguousForm(tokens));
  });
}

// --- R13b: bare cardinals --------------------------------------------------

test("R13b keeps an exact bare cardinal form", () => {
  // gold_deva has a bare "पचास" = 50 span; gold_gu a bare "બાવીસ" = 22.
  assert.ok(!shouldDropUnsupportedBareCardinal([["CARD_50", "पचास"]]));
  assert.ok(!shouldDropUnsupportedBareCardinal([["CARD_22", "બાવીસ"]]));
});

test("R13b drops a bare cardinal that is not a lexicon form", () => {
  for (const t of [["CARD_89", "aavesh"], ["CARD_6", "chaahie"], ["CARD_12", "barabar"], ["CARD_1", "इ"]] as Tok[]) {
    assert.ok(shouldDropUnsupportedBareCardinal([t]), t[1]);
  }
});

test("R13b only fires on a span with nothing else in it", () => {
  assert.ok(!shouldDropUnsupportedBareCardinal([["CARD_89", "aavesh"], ["SEP", " "], ["UNIT_LAKH", "lakh"]]));
});

// --- R14: blocked surfaces -------------------------------------------------

test("R14 a blocked surface never verifies and is dropped", () => {
  const cases: Tok[] = [
    ["UNIT_HAZAAR", "हजारे"],   // अण्णा हजारे
    ["PFX_SAVA", "सवाई"],       // सवाई तुकोजीराव
    ["UNIT_ARAB", "अरबी"],      // अरबी समुद्र
    ["UNIT_ARAB", "અરબી"],      // અરબી સમુદ્ર
    ["UNIT_HAZAAR", "હજારે"],
  ];
  for (const [cls, surface] of cases) {
    assert.ok(isBlockedSurface(surface), surface);
    assert.ok(!verifyTokens([[cls, surface]]), surface);
    assert.ok(hasBlockedSurface([[cls, surface]]), surface);
    assert.ok(!isKnownForm(cls, surface), surface);
  }
});

test("R14 leaves the head word alone", () => {
  assert.ok(verifyTokens([["UNIT_HAZAAR", "हजार"]]));
  assert.ok(!hasBlockedSurface([["UNIT_HAZAAR", "हजार"]]));
});

// --- R15: per-pack suffix stripping ----------------------------------------

test("R15 an inflected form still verifies within its own pack", () => {
  assert.ok(verifyTokens([["UNIT_LAKH", "लाखांचं"]]));
  assert.ok(verifyTokens([["UNIT_CRORE", "કરોડનો"]]));
});

test("R15 a stripped head must survive at three characters", () => {
  assert.ok(!isKnownForm("UNIT_SAU", "सोच"));
});

// --- R16: loose symbol units -----------------------------------------------

test("R16 drops a symbol unit that is not glued to its digits", () => {
  assert.ok(shouldDropLooseSymbolUnit([["DIGITS", "1996"], ["SEP", " "], ["UNIT_HAZAAR", "k"]]));
  assert.ok(
    shouldDropLooseSymbolUnit([
      ["DIGITS", "33"], ["SEP", " "], ["UNIT_HAZAAR", "k"], ["SEP", ". "], ["UNIT_MILLION", "m"],
    ]),
  );
});

test("R16 keeps a glued symbol unit and a long one", () => {
  assert.ok(!shouldDropLooseSymbolUnit([["DIGITS", "20"], ["UNIT_HAZAAR", "k"]]));
  assert.ok(!shouldDropLooseSymbolUnit([["DIGITS", "2"], ["DOT", "."], ["DIGITS", "5"], ["UNIT_LAKH", "l"]]));
  assert.ok(
    !shouldDropLooseSymbolUnit([
      ["DIGITS", "9"], ["DOT", "."], ["DIGITS", "5"], ["SEP", " "], ["UNIT_LAKH", "lpa"],
    ]),
  );
  assert.ok(!shouldDropLooseSymbolUnit([["DIGITS", "8"], ["SEP", " "], ["UNIT_LAKH", "lac"]]));
});

// --- R17: unknown unit words -----------------------------------------------

test("R17 drops a unit word that is not a unit word", () => {
  const cases: Tok[][] = [
    [["DIGITS", "3"], ["SEP", " "], ["UNIT_HAZAAR", "hours"]],
    [["DIGITS", "25"], ["UNIT_LAKH", "वे"]],
    [["DIGITS", "1980"], ["UNIT_LAKH", "ના"]],
    [["UNIT_CRORE", "chori"]],
    [["UNIT_HAZAAR", "thought"]],
    [["UNIT_SAU", "oooh"]],
    [["UNIT_CRORE", "खोड"]],
  ];
  for (const tokens of cases) assert.ok(shouldDropUnknownUnit(tokens), tokens.map(([, t]) => t).join(" "));
});

test("R17 tolerates a real or one-character typo'd unit", () => {
  const cases: Tok[][] = [
    [["DIGITS", "20"], ["SEP", " "], ["UNIT_HAZAAR", "hazaar"]],
    [["DIGITS", "20"], ["SEP", " "], ["UNIT_HAZAAR", "hazzar"]],
    [["DIGITS", "2"], ["SEP", " "], ["UNIT_CRORE", "करोड"]],
    [["DIGITS", "42"], ["SEP", " "], ["UNIT_CRORE", "કરોડનો"]],
  ];
  for (const tokens of cases) assert.ok(!shouldDropUnknownUnit(tokens), tokens.map(([, t]) => t).join(" "));
});

test("R17 does not fire when a justified cardinal corroborates the unit", () => {
  assert.ok(!shouldDropUnknownUnit([["CARD_3", "teen"], ["SEP", " "], ["UNIT_HAZAAR", "hours"]]));
});

test("levenshteinLE is bounded and symmetric", () => {
  assert.ok(levenshteinLE("lakh", "lakh", 1));
  assert.ok(levenshteinLE("lakh", "lakhh", 1));
  assert.ok(levenshteinLE("lakhh", "lakh", 1));
  assert.ok(!levenshteinLE("lakh", "hours", 1));
  assert.ok(levenshteinLE("lakh", "hours", 5));
});

// --- R18: boundary artifacts -----------------------------------------------

/** Decode `text` from a hand-built per-char BIO/class labelling. */
function spansOf(text: string, tokens: Tok[]) {
  const bio: number[] = [];
  const cls: number[] = [];
  let k = 0;
  for (const [cname, sub] of tokens) {
    for (let j = 0; j < sub.length; j++) {
      bio.push(k === 0 && j === 0 ? 1 : 2);
      cls.push(CLASS_TO_ID[cname]);
    }
    k++;
  }
  assert.equal(tokens.map(([, t]) => t).join(""), text);
  return decodeSpans(text, bio, cls);
}

test("R18 trims a trailing year", () => {
  const text = "दस हज़ार 1987";
  const out = spansOf(text, [
    ["CARD_10", "दस"], ["SEP", " "], ["UNIT_HAZAAR", "हज़ार"], ["SEP", " "], ["DIGITS", "1987"],
  ]);
  assert.equal(out.length, 1);
  assert.equal(out[0].text, "दस हज़ार");
  assert.equal(evaluate(out[0].tokens.map(([c, t]) => [CLASSES[c], t] as Tok)).value, 10000);
});

test("R18 drops a leading digits restatement", () => {
  const text = "5 panch hajar";
  const out = spansOf(text, [
    ["DIGITS", "5"], ["SEP", " "], ["CARD_5", "panch"], ["SEP", " "], ["UNIT_HAZAAR", "hajar"],
  ]);
  assert.equal(out.length, 1);
  assert.equal(out[0].text, "panch hajar");
  assert.equal(out[0].start, 2);
  assert.equal(evaluate(out[0].tokens.map(([c, t]) => [CLASSES[c], t] as Tok)).value, 5000);
});

test("R18 leaves a real digits coefficient alone", () => {
  const text = "1987 hazaar";
  const out = spansOf(text, [["DIGITS", "1987"], ["SEP", " "], ["UNIT_HAZAAR", "hazaar"]]);
  assert.equal(out.length, 1);
  assert.equal(out[0].text, text);
});
