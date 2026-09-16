import { test } from "node:test";
import assert from "node:assert/strict";
import { probeWebGPU, isWebGPUAvailable } from "../src/infer-webgpu.ts";

test("isWebGPUAvailable is false in Node", () => {
  assert.equal(isWebGPUAvailable(), false);
});

test("probeWebGPU resolves false in Node", async () => {
  const result = await probeWebGPU();
  assert.equal(result, false);
});

test("probeWebGPU caches its result", async () => {
  const a = await probeWebGPU();
  const b = await probeWebGPU();
  assert.equal(a, false);
  assert.equal(b, false);
});
