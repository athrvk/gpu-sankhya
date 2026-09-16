import { test } from "node:test";
import assert from "node:assert/strict";
import weightsJson from "../src/data/default-weights.json" with { type: "json" };
import { inspect, parse, modelInfo, CLASSES } from "../src/index.ts";

test("inspect(): chars covers every char of the normalized text, spans match parse()", () => {
  const text = "bhai sava lakh mein ho jayega kya";
  const insp = inspect(text);
  assert.equal(insp.chars.length, insp.text.length);
  assert.equal(insp.text.length, text.length); // this text is already normalized-length-stable

  const spans = parse(text);
  assert.deepEqual(
    insp.spans.map((s) => [s.start, s.end, s.value, s.span]),
    spans.map((s) => [s.start, s.end, s.value, s.span]),
  );
  assert.ok(typeof insp.ms === "number" && insp.ms >= 0);

  for (const span of insp.spans) {
    for (let i = span.start; i < span.end; i++) {
      const tag = insp.chars[i];
      assert.notEqual(tag.bio, 0, `char at ${i} ('${tag.ch}') inside span should be tagged B/I, got O`);
      assert.ok(CLASSES.includes(tag.cls), `cls '${tag.cls}' at ${i} should be a known class name`);
      assert.equal(CLASSES[tag.clsId], tag.cls);
      assert.equal(tag.bioProb.length, 3);
      const sum = tag.bioProb[0] + tag.bioProb[1] + tag.bioProb[2];
      assert.ok(Math.abs(sum - 1) < 1e-3, `bioProb should sum to ~1, got ${sum}`);
    }
  }
});

test("modelInfo(): default model reports its arch consistently with the weights file", () => {
  const info = modelInfo();
  const layers = (weightsJson as any).conv ?? [];
  assert.ok(info.layers.length >= 3);
  if (layers.length) {
    // v2 weights: the JSON carries the arch explicitly
    assert.deepEqual(info.layers.map((l) => l.k), layers.map((l: any) => l.k));
    assert.deepEqual(info.layers.map((l) => l.dilation), layers.map((l: any) => l.dilation));
    assert.deepEqual(info.layers.map((l) => l.residual), layers.map((l: any) => l.residual));
  } else {
    // v1 weights: fixed 4-layer stack
    assert.deepEqual(info.layers.map((l) => l.k), [3, 5, 3, 3]);
    assert.deepEqual(info.layers.map((l) => l.dilation), [1, 1, 2, 4]);
  }
  assert.equal(info.vocab, (weightsJson as any).charset.length);
  assert.equal(info.classes, (weightsJson as any).classes.length);
  assert.ok(info.params > 10000);
});
