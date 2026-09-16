import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { loadWeights } from "../src/weights.ts";
import { buildCharToId, encodeChars } from "../src/charset.ts";
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
    const ids = encodeChars(row.text, charToId);
    const fw = forward(weights, ids, scratch);
    const bio: number[] = [];
    const cls: number[] = [];
    for (let t = 0; t < fw.length; t++) {
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
