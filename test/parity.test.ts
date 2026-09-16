import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { loadWeights } from "../src/weights.ts";
import { buildCharToId, encodeChars, normalizeText, paddedLength } from "../src/charset.ts";
import { forward, argmaxRow, Scratch } from "../src/infer-cpu.ts";
import weightsJson from "../src/data/default-weights.json" with { type: "json" };

const fixturePath = fileURLToPath(new URL("./fixtures/parity.jsonl", import.meta.url));
const lines = readFileSync(fixturePath, "utf-8").trim().split("\n").filter(Boolean);

test("CPU backend argmax matches python np_infer reference (parity fixture)", () => {
  const weights = loadWeights(weightsJson as any);
  const charToId = buildCharToId(weights.charset);
  const scratch = new Scratch(weights);

  let identical = 0;
  const mismatches: string[] = [];
  for (const line of lines) {
    const row = JSON.parse(line) as { text: string; bio: number[]; cls: number[] };
    // Right-pad to match training-time padding (see charset.ts PAD_TAIL /
    // python np_infer.py's pad_ids()); run forward() on the padded array but
    // only argmax the real (unpadded) text length -- the padded tail's
    // logits are discarded, only used to give the real tokens correct
    // right-context.
    // encodeChars no longer lowercases internally (see charset.ts
    // normalizeText) -- the fixtures were generated on raw (un-normalized)
    // text, so normalize here exactly as Parser.parse() does before
    // encoding.
    const text = normalizeText(row.text);
    const padLen = paddedLength(text.length);
    const ids = encodeChars(text, charToId, undefined, padLen);
    const fw = forward(weights, ids, scratch);
    const bio: number[] = [];
    const cls: number[] = [];
    for (let t = 0; t < text.length; t++) {
      bio.push(argmaxRow(fw.bioLogits, t * 3, 3));
      cls.push(argmaxRow(fw.clsLogits, t * fw.nCls, fw.nCls));
    }
    const same = JSON.stringify(bio) === JSON.stringify(row.bio) && JSON.stringify(cls) === JSON.stringify(row.cls);
    if (same) identical++;
    else mismatches.push(row.text);
  }

  console.log(`parity: ${identical}/${lines.length} texts identical to python np_infer`);
  if (mismatches.length) console.log("mismatches:", mismatches.slice(0, 5));
  assert.equal(identical, lines.length, `expected all ${lines.length} texts to match; ${mismatches.length} mismatched`);
});
