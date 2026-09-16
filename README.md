# gpu-sankhya

Parses Indian informal number/currency shorthand — Hinglish (romanised
Hindi) and Indian-English amount phrases like `sava lakh`, `dedh crore`,
`2.5L`, `20k`, `2-3 lakh` — into a clean numeric value, with the span,
unit, currency, and confidence that produced it.

A small char-level CNN tags each character of the input with a BIO span
label and a semantic token class (digit, prefix word like "sava"/"dedh",
cardinal number word, scale unit like lakh/crore/hazaar). A deterministic
arithmetic core then turns that class sequence into a number — the model
never predicts the value directly, so the arithmetic can't drift from the
grammar (prefix semantics, additive descending units, multiplicative
ascending units like `das hazaar crore` = 10,000 × 1 crore = 1e11, etc).

Ships with a small (~22 KB gzipped) int8-quantized default model inlined
in the package — `import { parse } from "gpu-sankhya"` works with no
network fetch. Zero runtime dependencies.

Model training lives in `python/` (a separate, actively-trained
component); this package is the runtime that loads its exported weights.

## Install

```bash
npm install gpu-sankhya
```

## Usage

```ts
import { parse, parseBatch, createParser } from "gpu-sankhya";

parse("sava lakh");
// [{
//   span: "sava lakh", start: 0, end: 9,
//   value: 125000, unit: "lakh", currency: null,
//   confidence: 0.97, classes: ["PFX_SAVA", "SEP", "UNIT_LAKH"]
// }]

parse("mera budget paune do lakh tak ka hai");
// [{ span: "paune do lakh", value: 175000, unit: "lakh", ... }]

parse("2.5L");
// [{ span: "2.5L", value: 250000, unit: "lakh", ... }]

parse("20k logon ne attend kiya");
// [{ span: "20k", value: 20000, unit: "hazaar", ... }]

parse("2-3 lakh");
// [{ span: "2-3 lakh", value: 200000, range: [200000, 300000], unit: "lakh", ... }]

parse("das hazaar crore");
// [{ span: "das hazaar crore", value: 100000000000, unit: "crore", ... }]

// batch (uses WebGPU automatically for large batches in a browser, else CPU)
const results = await parseBatch(["sava lakh", "dedh crore", "..."]);

// custom / newer trained weights
const parser = createParser({ weights: myWeightsJson, backend: "cpu" });
parser.parse("paune do lakh");
```

### Output shape

```ts
interface Sankhya {
  span: string;            // exact source substring
  start: number; end: number;
  value: number;           // resolved value; low end for ranges
  range?: [number, number];// present only for ranges
  unit: "sau"|"hazaar"|"lakh"|"crore"|"million"|"billion"|"arab"|"kharab"|null;
  currency: "INR"|null;    // adjacent marker detected outside the span
  confidence: number;      // mean of span-tag softmax probs over the span
  classes: string[];       // normalised token classes, e.g. ["PFX_DHAI","UNIT_LAKH"]
}
```

### API

- **`parse(text, opts?) => Sankhya[]`** — synchronous, always CPU. Use
  this for one-off strings; there's no async overhead.
- **`parseBatch(texts, opts?) => Promise<Sankhya[][]>`** — batched parse.
  Runs on CPU by default. Pass `{ backend: "webgpu" }` to force WebGPU, or
  `{ backend: "auto" }` to use WebGPU automatically when it's available
  *and* the batch has at least 32 texts (otherwise CPU, since GPU
  dispatch overhead dominates for small batches).
- **`createParser({ weights?, backend? })`** — build a `Parser` instance
  around a custom weights JSON (float or int8 form, as written by
  `python/sankhya/export.py`), instead of the bundled default model.
- **`isWebGPUAvailable()`** — true if `navigator.gpu` exists in the
  current environment.

## Backends

- **CPU** (default everywhere): a plain typed-array forward pass mirroring
  the Python reference implementation exactly, tuned for throughput
  (preallocated buffers, channels-outer loop order, no per-character
  allocation). Runs in Node and any browser.
- **WebGPU**: the same forward pass as WGSL compute shaders (embedding
  gather, one dispatch per conv1d+ReLU layer, two head matmuls), used only
  through `parseBatch` for large batches. Device/shader setup is lazy —
  importing the package never touches `navigator.gpu`, so it's safe to
  import in Node or SSR. Falls back to CPU per-text for inputs longer than
  the model's 128-char window (the sliding-window path used by `parse` is
  CPU-only for now).

## Limitations

- **Latin-script Hindi only.** Only Hinglish / romanised Hindi and Indian
  English amount phrases are supported today. Devanagari script and other
  Indian languages are planned via additional language packs (the
  arithmetic core is already language-independent; only the class
  vocabulary and currency-marker lists are per-language).
- Text longer than 128 characters is processed with a sliding window
  (128-char windows, 16-char overlap) and results are merged/deduplicated
  by span; extremely long inputs may still miss a span that straddles a
  window boundary in an unlucky way.
- The bundled default weights are from an early training checkpoint (see
  `python/`) and will be replaced as training progresses; some spec
  examples (multi-unit multiplicative phrases like `das hazaar crore`,
  dash-separated ranges like `2-3 lakh`) may not resolve correctly yet —
  this is a model-quality issue, not a bug in the arithmetic core (which
  is unit-tested directly in `test/core.test.ts`).

## Development

```bash
npm install
npm run build   # esbuild -> dist/index.js (ESM), tsc -> dist/*.d.ts
npm test        # node --test over test/*.test.ts
npm run bench   # parse() and parseBatch() latency
npm run size    # gzipped dist/index.js size
```

See `demo/index.html` for a minimal textarea + live-results demo that
imports `dist/index.js` directly (no build step needed beyond `npm run
build`).
