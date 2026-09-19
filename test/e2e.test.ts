import { test } from "node:test";
import assert from "node:assert/strict";
import { parse, parseBatch, inspect } from "../src/index.ts";

// These exercise the bundled dev weights end-to-end. The current training
// run is early, so some spec examples may not decode/evaluate correctly yet;
// those are marked `expected-failure` with a comment rather than deleted, so
// this file stays a live record of model quality as training improves.

test("sava lakh", () => {
  const r = parse("sava lakh");
  assert.equal(r.length, 1);
  assert.equal(r[0].value, 125000);
});

test("sava lakh is verified, with its tokens", () => {
  const r = parse("sava lakh");
  assert.equal(r.length, 1);
  assert.equal(r[0].verified, true);
  assert.deepEqual(r[0].tokens, [
    ["PFX_SAVA", "sava"],
    ["SEP", " "],
    ["UNIT_LAKH", "lakh"],
  ]);
});

test("a made-up spelling the model still tags is unverified, and dropped in strict mode", () => {
  const r = parse("savaa lakhhh");
  assert.equal(r.length, 1);
  assert.equal(r[0].value, 125000);
  assert.equal(r[0].verified, false);
  assert.ok(r[0].tokens.length > 0);

  const strict = parse("savaa lakhhh", { strict: true });
  assert.equal(strict.length, 0);

  // strict mode keeps a genuinely verified span
  const strictOk = parse("sava lakh", { strict: true });
  assert.equal(strictOk.length, 1);
  assert.equal(strictOk[0].verified, true);
});

test("dedh crore", () => {
  const r = parse("dedh crore");
  assert.equal(r.length, 1);
  assert.equal(r[0].value, 15000000);
});

test("paune do lakh", () => {
  const r = parse("paune do lakh");
  assert.equal(r.length, 1);
  assert.equal(r[0].value, 175000);
});

test("2.5L", () => {
  const r = parse("2.5L");
  assert.equal(r.length, 1);
  assert.equal(r[0].value, 250000);
});

test("20k logon ne attend kiya", () => {
  const r = parse("20k logon ne attend kiya");
  assert.ok(r.length >= 1);
  assert.equal(r[0].value, 20000);
});

test("2-3 lakh", () => {
  // The dev checkpoint tags '-' as O in isolation; decode.ts's class-repair
  // step recovers RANGE from context (digit run on both sides), fixing this.
  const r = parse("2-3 lakh");
  assert.equal(r.length, 1);
  assert.deepEqual(r[0].range, [200000, 300000]);
});

test("unnasi hazaar (word integrity)", () => {
  // The model's BIO output on rare bare cardinals like "unnasi" is patchy
  // (B/O flips mid-word); extendWordIntegrityBio merges the whole letter
  // run back into one span when every char raw-predicts the same CARD_
  // class. With the 0.6.0 four-language weights the bare word alone sits
  // just under the 0.5 confidence gate (an honest abstain); with a unit it
  // is confidently one span.
  const r = parse("unnasi hazaar");
  assert.equal(r.length, 1);
  assert.equal(r[0].span, "unnasi hazaar");
  assert.equal(r[0].value, 79000);
});

test("das hazaar crore", () => {
  // Similarly recovered by class-repair's letter-run majority vote/fallback.
  const r = parse("das hazaar crore");
  assert.equal(r.length, 1);
  assert.equal(r[0].value, 1e11);
});

test("unknown characters around a span do not break it", () => {
  // Out-of-vocab characters (uncommon punctuation, emoji, other currency
  // symbols) map every affected char to <unk>. These specific cases already
  // decode correctly with the currently shipped weights. Retraining the
  // generator to inject unk noise (see python/sankhya/generator.py) is meant
  // to make more such cases robust (e.g. emoji directly before a span);
  // those aren't asserted here yet since they depend on the retrained model.
  let r = parse("» sava lakh");
  assert.equal(r.length, 1);
  assert.equal(r[0].value, 125000);

  r = parse("sava lakh 🙏");
  assert.equal(r.length, 1);
  assert.equal(r[0].value, 125000);

  r = parse("€ 2 lakh");
  assert.equal(r.length, 1);
  assert.equal(r[0].value, 200000);
});

test("R1: parse() treats non-string input as empty rather than throwing", () => {
  assert.deepEqual(parse(null as unknown as string), []);
  assert.deepEqual(parse(undefined as unknown as string), []);
  assert.deepEqual(parse(12345 as unknown as string), []);
  assert.deepEqual(parse({} as unknown as string), []);
});

test("R1: inspect() treats non-string input as empty", () => {
  const r = inspect(null as unknown as string);
  assert.equal(r.text, "");
  assert.deepEqual(r.chars, []);
  assert.deepEqual(r.spans, []);
});

test("R1: parseBatch() tolerates non-string entries in the batch", async () => {
  const r = await parseBatch([null as unknown as string, "sava lakh"]);
  assert.deepEqual(r[0], []);
  assert.equal(r[1].length, 1);
  assert.equal(r[1][0].value, 125000);
});

test("minConfidence drops low-confidence spans; the default keeps a clear one", () => {
  // Default (minConfidence 0): a clean, unambiguous phrase parses, and its
  // calibrated confidence sits well above the decoder's own raw 0.5 gate.
  const clear = parse("sava lakh");
  assert.equal(clear.length, 1);
  assert.equal(clear[0].value, 125000);
  assert.ok(clear[0].confidence > 0.9, `expected a high calibrated confidence, got ${clear[0].confidence}`);
  assert.ok(clear[0].rawConfidence > 0.5);

  // A ragged mixed-numeral construct the model is much less sure about
  // still parses by default...
  const shaky = parse("teen n half lakh");
  assert.equal(shaky.length, 1);
  assert.ok(
    shaky[0].confidence < clear[0].confidence,
    `expected 'teen n half lakh' (${shaky[0].confidence}) below 'sava lakh' (${clear[0].confidence})`,
  );
  // ...and is dropped once the caller asks for near-certainty.
  assert.deepEqual(parse("teen n half lakh", { minConfidence: 0.99 }), []);

  // The gate is on the CALIBRATED score, so a threshold just under a span's
  // reported confidence keeps it and one just over drops it.
  assert.equal(parse("sava lakh", { minConfidence: clear[0].confidence - 1e-9 }).length, 1);
  assert.equal(parse("sava lakh", { minConfidence: clear[0].confidence + 1e-9 }).length, 0);
});

test("minConfidence composes with strict and flows through parseBatch", async () => {
  assert.equal(parse("sava lakh", { strict: true, minConfidence: 0.9 }).length, 1);
  assert.equal(parse("sava lakh", { strict: true, minConfidence: 0.999 }).length, 0);

  // parseBatch passes the option down the CPU path: the clear phrase clears
  // a 0.96 bar, the ragged one does not.
  const batch = await parseBatch(["sava lakh", "teen n half lakh"], { minConfidence: 0.96 });
  assert.equal(batch[0].length, 1);
  assert.deepEqual(batch[1], []);
  // 0.99 is above everything this model reports, so nothing survives it.
  assert.deepEqual(await parseBatch(["sava lakh"], { minConfidence: 0.99 }), [[]]);
});

test("R9: indefinite plurals ('karodon'/'करोडो') are not amounts", () => {
  // Both surfaces are declared lexicon class "O" (and nothing else), so the
  // lexicon-"O" gate drops any span the model tags UNIT_CRORE over them.
  assert.deepEqual(parse("usne karodon rupaye kamaye"), []);
  assert.deepEqual(parse("त्याने करोडो रुपये कमावले"), []);
});
