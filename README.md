# gpu-sankhya

[![npm version](https://img.shields.io/npm/v/gpu-sankhya.svg)](https://www.npmjs.com/package/gpu-sankhya)
[![bundle size](https://img.shields.io/badge/bundle-47.2%20KB%20gzipped-blue)](https://www.npmjs.com/package/gpu-sankhya?activeTab=code)
[![license](https://img.shields.io/npm/l/gpu-sankhya.svg)](./LICENSE)
[![demo](https://img.shields.io/badge/demo-live-brightgreen)](https://athrvk.github.io/gpu-sankhya/)

Parses Indian informal number/currency shorthand — Hinglish (romanised
Hindi), Devanagari Hindi, and Indian-English amount phrases like `sava
lakh`, `dedh crore`, `डेढ़ लाख`, `सवा करोड़`, `2.5L`, `20k`, `2-3 lakh` —
into a clean numeric value, with the span, unit, currency, and confidence
that produced it. Mixed-script input (Latin and Devanagari in the same
string, e.g. `"budget 2 लाख hai"`) is supported: currency and unit
detection scan both scripts' marker lists.

A small char-level CNN tags each character of the input with a BIO span
label and a semantic token class (digit, prefix word like "sava"/"dedh",
cardinal number word, scale unit like lakh/crore/hazaar). A deterministic
arithmetic core then turns that class sequence into a number — the model
never predicts the value directly, so the arithmetic can't drift from the
grammar (prefix semantics, additive descending units, multiplicative
ascending units like `das hazaar crore` = 10,000 × 1 crore = 1e11, etc).

Ships with a small (47.2 KB gzipped, 80.8 KB raw `dist/index.js`)
int8-quantized default model inlined in the package — `import { parse }
from "gpu-sankhya"` works with no network fetch. Zero runtime
dependencies.

Model training lives in `python/` (a separate, actively-trained
component); this package is the runtime that loads its exported weights.
See `python/README.md` if you want to train your own weights and load
them via `createParser({ weights })` instead of the bundled default.

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
//   confidence: 0.95, classes: ["PFX_SAVA", "SEP", "UNIT_LAKH"]
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

parse("sawaa laakh ka budget hai");
// [{ span: "sawaa laakh", value: 125000, unit: "lakh", ... }] -- spelling
// variance ("sawaa"/"sava", "laakh"/"lakh") is part of the training data,
// not special-cased.

// Devanagari input follows the same grammar/arithmetic as Hinglish (the
// core is language-independent -- see How it works); the expected output
// shape is identical, e.g.:
parse("डेढ़ लाख");
// [{ span: "डेढ़ लाख", value: 150000, unit: "lakh", ... }]

parse("सवा दो करोड़ का फ्लैट");
// [{ span: "सवा दो करोड़", value: 22500000, unit: "crore", ... }]

parse("बजट साढ़े तीन लाख है");
// [{ span: "साढ़े तीन लाख", value: 350000, unit: "lakh", ... }]
// NOTE: the correctness of these Devanagari examples depends on the
// bundled weights being trained on the hi_deva pack (see python/README.md);
// the runtime-side normalization/decoding/currency support is in place
// independent of which weights are loaded (createParser({ weights })).

// batch (uses WebGPU automatically for large batches in a browser, else CPU)
const results = await parseBatch(["sava lakh", "dedh crore", "..."]);

// custom / newer trained weights (see python/README.md to train your own)
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
  `python/sankhya/export.py`), instead of the bundled default model. See
  `python/README.md` for how to train and export your own weights.
- **`isWebGPUAvailable()`** — true if `navigator.gpu` exists in the
  current environment. Cheap and synchronous, but doesn't guarantee a
  usable adapter (e.g. headless browsers without GPU access).
- **`probeWebGPU() => Promise<boolean>`** — authoritative async check:
  resolves false immediately if `navigator.gpu` is missing, otherwise
  awaits `requestAdapter()` and resolves to whether an adapter was
  actually obtained. Result is cached, so repeated calls only probe once.
  `parseBatch`'s `"auto"` backend uses this (not `isWebGPUAvailable()`)
  to decide whether to try the GPU path.
- **Non-string input**: `parse`, `parseBatch`, and `inspect` all tolerate
  a non-string argument (`null`, `undefined`, a number, object, array,
  boolean, ...) by treating it as no input rather than throwing —
  `parse(null)` returns `[]` instead of crashing.
- **`inspect(text) => Inspection`** — synchronous CPU inspection of the
  raw model output before decode/repair: `{ text, chars: [{ch, bio,
  bioProb, cls, clsId}], spans, ms }`, one entry per character with its
  argmax BIO tag, BIO softmax probability, and predicted class. Useful
  for debugging misclassifications or building a "what did the model
  actually see" view (this is what powers the demo's per-character
  panel); `parse()`'s decoded spans are the thing to use for normal
  consumption.
- **`modelInfo() => ModelInfo`** — info about the currently loaded
  default model: `{ version, params, channels, embedDim, vocab, classes,
  layers: [{k, dilation, residual}] }`.
- **`CLASSES`** — the exported array of semantic token class names (the
  same strings that appear in `Sankhya.classes` and `Inspection.chars[].cls`).
- **`normalizeText(text) => string`** — the exact normalization
  `parse`/`parseBatch` apply before char-encoding (Unicode NFC, then
  Devanagari digits U+0966-U+096F mapped 1:1 to ASCII `"0"`-`"9"`, then
  lowercasing), exposed so callers can reproduce the same offsets. `parse`
  returns `start`/`end` as indices into `normalizeText(text)`, not the raw
  input — for the overwhelming majority of input, which already arrives in
  NFC form, normalization never changes the string's length, so offsets
  are unaffected; call `normalizeText` yourself first if you need to be
  sure for non-NFC input. The `span` field is a substring of your original
  input when lengths match, and of the normalized string otherwise.

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

### Why CPU by default

One inference is small — a 5-layer, 48-channel dilated/residual char
CNN, 39,579 parameters — and `parse()` runs in about 3.6 ms p50 (60-char
input) in plain JS; `parseBatch` on CPU averages about 4.0 ms/string over
500 strings. A WebGPU dispatch has fixed overhead of a few milliseconds
(device/pipeline setup, buffer upload, queue submit, readback), which
dwarfs that per-string cost. So the GPU only pays off once you're
amortizing that overhead over a batch — hundreds of strings at once —
which is exactly what `parseBatch` does with `backend: "auto"`/`"webgpu"`;
`parse()` stays CPU-only and synchronous on purpose.

The WebGPU path has been verified on a real GPU in desktop Chrome with
the shipped `v2` model (residual layers included): on the demo page's
batch benchmark (500 varied strings) it matches the CPU backend
span-for-span (500/500) and runs about 1.9x faster (CPU 1.66 ms/string,
WebGPU 0.88 ms/string). The demo's benchmark re-runs the CPU/WebGPU
comparison on every click, so any regression shows up as a mismatch
count rather than a silent wrong answer.

## Accuracy

The bundled default model (arch `v2`): 39,579 parameters, 5 conv layers
over 16-dim char embeddings — a plain kernel-5 layer, then four residual
kernel-3 layers with dilations 1/2/4/8 (`y = relu(conv(x)) + x`), 48
channels, ±17-character receptive field — over a 114-character vocab
(union of the `hi_latn` and `hi_deva` packs), 120 output classes. Trained
20 epochs on 200,000 synthetic examples generated from both language
packs' grammars, mixed 0.55/0.45 with a 10% cross-pack share, plus
out-of-vocab "unk noise" augmentation so the model has actually seen
`<unk>` characters (emoji, CJK, Cyrillic, other symbols) during training
(see `python/README.md`). An older `v1` preset (the previously shipped
4-layer, non-residual, 32-channel stack) is still loadable by both the
Python and JS runtimes for anyone using older exported weights.

On synthetic validation data (drawn from the same generator/templates as
training): 0.925 value accuracy. This number is optimistic — it's testing
the model on its own distribution.

On two hand-written gold sets, written independently of the generator —
`python/tests/gold.jsonl` (romanised Hindi, 220 sentences / 194 spans)
and `python/tests/gold_deva.jsonl` (Devanagari Hindi, 182 sentences / 151
spans) — evaluated against the shipped int8-quantized weights:

| gold set | examples | spans | value accuracy | span precision | span recall | span F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gold.jsonl (romanised) | 220 | 194 | 0.9536 | 0.9585 | 0.9536 | 0.9561 |
| gold_deva.jsonl (Devanagari) | 182 | 151 | 0.9801 | 0.9867 | 0.9801 | 0.9834 |
| combined | 402 | 345 | 0.9652 | 0.9708 | 0.9652 | 0.9680 |

Negatives (zero-gold-span examples, 73 total): 0 false positives.
Miss summary: missed 3, spurious 0, wrong value 0, wrong boundary 9.
Per-category value accuracy: digits 0.967, words 0.940, prefix 0.926,
range 0.917, currency 1.0, multi_unit 0.952, symbol_unit 0.970,
mixed_script 1.0, long 0.80.

For comparison, the previous shipped weights scored, on this same
enlarged gold set: 0.9096 (romanised) / 0.9603 (Devanagari) / 0.9322
combined value acc, F1 0.9258, 4/73 negatives false positives (5.5%).
(Their numbers on the older, smaller 353-example gold set were 0.952 /
0.986 / 0.968 — the new gold set is deliberately harder: it adds
conjunction-joined multi-span sentences, trailing-cardinal chains,
possessive noise, in-context typos, cardinal spelling variants, and 18
more negative examples.) This round also ran a 198-case hand-written
edge-case probe across 12 categories (whitespace/short input, long
input, Unicode, numeric forms, prefix semantics, compound units,
currency, negatives, multi-span strings, noise/typos, Devanagari
equivalents, API contract) targeting decoder and boundary bugs rather
than the gold distribution itself; the fixes below took it from 146
pass / 17 spurious / 8 wrong value / 7 crashes to 179 pass / 0 crashes /
5 spurious / 1 wrong value (remaining misses are mostly boundary/range
edge cases, tracked below).

**The gold numbers are the ones to trust.** Known miss categories, in rough
order of frequency:

- wrong span boundaries on multi-span/range/connector phrases (the
  largest single category this round)
- wrong value on a handful of prefix/compound constructs
- unusual typos the noise model doesn't cover (e.g. "croer" for "crore")
- long multi-term/mixed-numeral constructs ("three n half lakh", "50M")
- occasional spurious spans triggered by unfamiliar words near number-ish
  context

Reproduce these numbers yourself with
`python -m sankhya.eval_gold --gold tests/gold.jsonl tests/gold_deva.jsonl --weights-json src/data/default-weights.json --int8`
from `python/` (see `python/README.md`).

## How it works

1. The input string is normalized (`normalizeText`: Unicode NFC, then
   Devanagari digits U+0966-U+096F mapped to ASCII `"0"`-`"9"`, then
   lowercased — byte-identical to the Python reference) and char-encoded
   against the model's vocab (unknown chars map to `<unk>`).
2. A dilated/residual conv1d stack (see Accuracy above) produces, per
   character, a 3-way BIO logit (O/B/I) and a class logit over the
   semantic token vocabulary (prefix words, cardinals, units, digits,
   separators, misc).
3. Decoding turns those per-character predictions into spans and tokens
   (see Decoding below).
4. The deterministic arithmetic core (`src/core.ts`, mirrored 1:1 from
   `python/sankhya/core.py` and unit-tested directly in
   `test/core.test.ts`) evaluates each span's token sequence into a
   value: prefix semantics (sava = ×1.25, dedh = ×1.5, paune = subtract
   1/4 from the next cardinal, ...), additive combination of descending
   units, multiplicative combination of ascending units, and range
   handling for `X-Y unit` / `X se Y unit` phrases.

### Decoding

Raw per-character BIO/class predictions are cleaned up before evaluation:

- **Strict BIO decode**: a span starts only at a `B` tag; an `I` that
  isn't preceded by an open span is treated as `O`.
- **BIO bridging**: a single-character `O` gap inside what's otherwise a
  contiguous span is closed (handles a stray misclassified character
  without splitting the span in two).
- **Digit-run extension**: a span is extended forward through a trailing
  run of digit characters it was cut short of.
- **Class repair**: within a span, per-character classes are smoothed by
  majority vote over character-type sub-runs (fixes a stray misclassified
  character inside an otherwise-consistent digit or letter run), plus a
  few punctuation-specific rules.
- **Word integrity**: a `UNIT_`/`PFX_`/`CARD_` token must line up with the
  letter-word it sits in — a letters-run that's only partially meaningful
  (some characters fell back to `O`) has all of it retagged `O` (e.g.
  `"10 km"`), while a run legitimately split into several back-to-back
  meaningful sub-words is left alone (e.g. `"dedhlakh"` = `PFX_DEDH` +
  `UNIT_LAKH`). A one-character symbol unit (`k`/`K`/`l`/`L`) is additionally
  only valid when the next character isn't a letter, so `"20k logon"`
  keeps its `k` but `"10 km"`'s `k` doesn't survive alone either.
  Before that drop check runs, a BIO repair pass looks at the letter-word's
  *raw* per-character classes: if they unambiguously agree on one word
  class (trusting only direct evidence — a same-class run of 3+ characters,
  or a short stray adopting a qualifying neighbour's class, never a diffuse
  whole-run majority vote alone), the span's BIO is extended to cover the
  whole word instead of being left partial — this also re-merges a span
  that a low-confidence internal `O` gap had split in two (e.g. `"unnasi"`
  raw-tagged `B I I O I O`, all six characters `CARD_79`, becomes one span
  `"unnasi"` = 79 instead of dropping).
- **Bare-digits gate**: a span whose only meaningful tokens are digits (no
  unit/prefix/cardinal word at all) with no detected currency marker is
  dropped only when it's short and ungrouped (≤4 digits, no comma — route
  numbers, OTPs, years: `"1"`, `"route 66"`, `"OTP 4521"`, `"2024"`) or very
  long (≥10 digits — phone numbers: `"9876543210"`). Everything else is
  kept as a plain (unitless) amount — `"15000"`, `"1,00,000"`,
  `"2,50,00,000"` all parse; `"₹75"` / `"Rs 2,50,000/-"` / `"1200 rupees"`
  still parse with `currency: "INR"`. The currency-marker look-ahead
  scans 8 characters past the amount for a trailing marker word (up from
  a shorter window that missed `"1200 rupees"` — `"rupees"` alone runs to
  6 characters plus the leading space).
- **Range-connector repair**: a bare `-`/`–`/`—`/`/` between two amounts
  becomes a `RANGE` tag (spaces around it stay `SEP`) when the left side
  can end an amount by itself and the right side can start a fresh one
  (`"2 lakh/3 lakh"`, `"दो-तीन लाख"`) — but not when it's really one
  compound number, e.g. a prefix glued straight to a unit (`"dedh-lakh"`
  stays one span, not a range).
- **Possessive trim**: a trailing `'s`/`’s` (1-2 letters) is stripped from
  the end of a word and excluded from the span (`"2 lakh's"` → `"2 lakh"`).
- **Confidence filter**: a span's confidence is the mean of the max BIO
  softmax probability per character; spans below 0.5 are dropped.
- Only after all of the above does the deterministic arithmetic core run
  on the resulting token sequence, which also treats a `RANGE` connector
  between two amounts that BOTH already carry a unit and are strictly
  *descending* (e.g. `"ek lakh dus hazaar"` mistagged `RANGE` on the
  space) as one additive amount rather than a `[low, high]` range —
  genuine ranges (only one side has a unit, or both do but ascending) are
  unaffected.

One more detail that matters more than it looks like it should: the
runtime right-pads the character-id array with 24 pad tokens before
running the forward pass (mirroring `python/sankhya/np_infer.py`'s
`pad_ids`/`PAD_TAIL=24`), because training always right-pads every
example to the model's max length the same way. Running a short, tightly
cropped input (e.g. the bare 4 characters of `"2.5L"`) without that
padding measurably corrupts predictions — the model was never trained on
inputs that end at the literal edge of the array. The padded tail's
outputs are discarded; only the real characters' predictions are used.

## Limitations

- **Hinglish and Devanagari Hindi.** Romanised Hindi / Indian-English
  amount phrases and Devanagari-script Hindi (`डेढ़ लाख`, `सवा करोड़`) are
  both supported, including mixed-script input in the same string.
  Currency and unit detection scan the Latin (`hi_latn`) and Devanagari
  (`hi_deva`) marker lists together. Other Indian languages (Marathi,
  Gujarati, Bengali, Tamil/Telugu/Kannada) are planned via additional
  language packs (the arithmetic core is already language-independent;
  only the class vocabulary and currency-marker lists are per-language) —
  see Roadmap. Devanagari accuracy depends on the bundled weights being
  trained on the `hi_deva` pack; see Devanagari gold-set numbers above.
- **Offsets are into the normalized string.** `parse()`'s `start`/`end`
  index `normalizeText(text)`, not the raw input, in the rare case NFC
  normalization changes the string's length (see `normalizeText` in the
  API section above); in JS, an astral (surrogate-pair) character such as
  an emoji also counts as two UTF-16 units in those offsets, same as any
  other JS string indexing.
- Text longer than 128 characters is processed with a sliding window
  (128-char windows, 16-char overlap) and results are merged/deduplicated
  by span; extremely long inputs may still miss a span that straddles a
  window boundary in an unlucky way.
- **Bare single cardinal words without a unit are not reliably
  extracted.** `"unnasi"` alone produces no span, while `"unnasi hazaar"`
  correctly resolves to 79000 — a bare number word needs a unit, prefix,
  or digit context to be tagged.
- **Long multi-term ranges still split incorrectly.** Constructs like
  `"paanch se sadhe saat lakh"` (a range where one side itself carries a
  prefix word) are not reliably decoded as a single range.
- **Fullwidth and Arabic-Indic digits are not normalised.** Only
  Devanagari digits (U+0966-U+096F) are mapped by `normalizeText`;
  fullwidth (`２`) and Arabic-Indic (`٢`) digit forms are left as unknown
  characters, so a phrase like `"２ lakh"` loses the digit and falls back
  to the unit's default value.
- Model quality: see Accuracy above. The known miss categories there
  (wrong span boundaries on multi-span/range phrases, unusual typos,
  long multi-term ranges, occasional spurious spans) are model-quality
  issues, not bugs in the arithmetic core, which is unit-tested directly
  and independently of the model in `test/core.test.ts`.

## Repository layout

- `src/` — the JS/TS runtime: char encoding, CPU forward pass
  (`infer-cpu.ts`), WebGPU forward pass, decode, the arithmetic core
  (`core.ts`), the public API (`index.ts`), and the bundled default
  weights (`src/data/default-weights.json`).
- `test/` — Node test files (`node --test`), including parity fixtures
  generated from the Python reference implementation
  (`test/fixtures/parity.jsonl`, `decoded.jsonl` — see
  `python/README.md`'s "Ship to the npm package" section).
- `bench/` — `parse()`/`parseBatch()` latency benchmarks.
- `demo/` — an interactive demo: live parse as you type, a per-character
  model-internals view (`inspect()`), batch benchmark controls (CPU vs
  WebGPU), a raw-JSON view of results, a theme toggle, and shareable
  state via URL hash.
- `python/sankhya/` — `classes.py` (shared class vocabulary), `core.py`
  (the deterministic arithmetic core), `langs/` (language packs, e.g.
  `hi_latn.py`), `noise_latn.py` (typo/spelling-variance injection),
  `generator.py` (synthetic labelled-data generator), `model.py` (the
  char CNN), `train.py`, `export.py`, `np_infer.py` (numpy reference
  forward pass, what the JS port mirrors), `decode.py`, `eval_gold.py`,
  `make_fixtures.py`.
- `python/tests/` — `test_core.py`, `test_decode.py`, `test_generator.py`,
  and `gold.jsonl` (the hand-written gold set).
- `docs/DATA_GRAMMAR.md` — the data/grammar spec the generator and
  language packs implement.

## Roadmap

Everything here is scoped to Indian languages — there's no plan to
support non-Indian numbering/currency shorthand.

1. ~~**Devanagari Hindi pack** (डेढ़ लाख).~~ **Done.** The `hi_deva`
   language pack, Devanagari currency markers, Unicode-aware (`\p{L}`/
   `\p{M}`) letters-run repair for matras/nukta/virama, and input
   normalization (NFC + Devanagari-digit mapping) are all in place, and
   mixed Latin/Devanagari input is supported. See `python/README.md` for
   the training-side status and gold-set numbers.
2. **Other Indian languages as packs**: Marathi (साडे, सव्वा), Gujarati
   (સવા, દોઢ), Bengali (দেড়, আড়াই), and Tamil/Telugu/Kannada number
   words. Same shape as (1) — a new pack, a new noise function, a charset
   rebuild.
3. **A WASM SIMD kernel**, if sub-millisecond latency is ever needed
   beyond what the plain-JS CPU path already gives.

## Development

```bash
npm install
npm run build   # esbuild -> dist/index.js (ESM), tsc -> dist/*.d.ts
npm test        # node --test over test/*.test.ts
npm run bench   # parse() and parseBatch() latency
npm run size    # gzipped dist/index.js size
```

See `demo/index.html` for the full interactive demo (live parse,
per-character model view, benchmark controls, JSON view, theme toggle,
shareable hash) — it imports `dist/index.js` directly (no build step
needed beyond `npm run build`). Live at
https://athrvk.github.io/gpu-sankhya/.

## Releasing

The normal flow: bump the version and push to master.

```bash
npm version patch|minor|major   # bumps package.json and commits + tags locally
git push --follow-tags origin master
```

You can also just edit the `version` field in `package.json` in a PR — no
need to run `npm version` or create a tag yourself. Either way, once the
new version lands on master, the `publish` workflow detects that
`package.json`'s version isn't on npm yet, builds, tests, publishes (with
provenance), and creates the matching `vX.Y.Z` git tag if it does not
exist yet. Running the workflow manually via `workflow_dispatch` does the
same check and publishes only if the version is not on npm.

Trusted publishing (OIDC, no `NPM_TOKEN`) must be configured once on
npmjs.com for this to work: package page -> Settings -> Trusted publisher,
with Organization/user `athrvk`, Repository `gpu-sankhya`, Workflow
filename `publish.yml`. The first release was published manually with
`npm publish`.

### Publishing to Hugging Face

The `.github/workflows/huggingface.yml` workflow mirrors the shipped model,
the demo, and the gold evaluation sets to Hugging Face:

- model: https://huggingface.co/athrvk/gpu-sankhya
- demo Space: https://huggingface.co/spaces/athrvk/gpu-sankhya-demo
- dataset: https://huggingface.co/datasets/athrvk/gpu-sankhya-gold

It runs on `workflow_dispatch` (choose `all`/`model`/`space`/`dataset`) and
on push to `master` when `package.json`, `models/default/**`,
`python/tests/gold*.jsonl`, or `demo/**` change. One-time setup: add an
`HF_TOKEN` repo secret with a write-scoped Hugging Face token for the
`athrvk` namespace — without it the workflow prints a notice and skips.
See `python/README.md`'s "Publishing to Hugging Face" section for the
underlying `python -m sankhya.hf_push` CLI (including `--dry-run`).
