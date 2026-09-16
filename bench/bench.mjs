import { parse, parseBatch } from "../dist/index.js";

function percentile(sorted, p) {
  const idx = Math.min(sorted.length - 1, Math.floor((p / 100) * sorted.length));
  return sorted[idx];
}

// --- parse() latency: 60-char string, 1000 iterations ---
const text = "bhai mujhe paune do lakh chahiye is mahine ke andar warna problem ho jayega";
const sample = text.slice(0, 60);

// warmup
for (let i = 0; i < 20; i++) parse(sample);

const times = [];
for (let i = 0; i < 1000; i++) {
  const t0 = performance.now();
  parse(sample);
  times.push(performance.now() - t0);
}
times.sort((a, b) => a - b);
const p50 = percentile(times, 50);
const p95 = percentile(times, 95);
console.log(`parse() 60-char string, 1000 iters: p50=${p50.toFixed(4)}ms p95=${p95.toFixed(4)}ms`);

// --- parseBatch() CPU: 500 strings ---
const texts = Array.from({ length: 500 }, (_, i) => `${sample} #${i}`);
const tb0 = performance.now();
await parseBatch(texts, { backend: "cpu" });
const tb1 = performance.now();
console.log(`parseBatch() CPU, 500 strings: total=${(tb1 - tb0).toFixed(2)}ms (${((tb1 - tb0) / 500).toFixed(4)}ms/string)`);
