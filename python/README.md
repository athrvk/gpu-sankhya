# python/sankhya — training guide

This is the training/export side of gpu-sankhya: a synthetic labelled-data
generator, a deterministic evaluation core, a char-CNN model, and the
tooling to train it and export weights that the JS runtime (`../src/`)
loads. See `../docs/DATA_GRAMMAR.md` for the underlying grammar/data spec.

Everything below is run from inside `python/`.

## Setup

```bash
pip install torch numpy onnx onnxruntime
```

CPU-only torch is fine — training the current multi-pack default model
(18,811 params) takes about 12 minutes on 4 CPU cores for 200k examples.

## Generate data

Single-pack (Latin-script/romanised Hindi only):

```bash
python -m sankhya.generator --n 150000 --seed 3 --out data/train.jsonl --lang hi_latn
python -m sankhya.generator --n 5000 --seed 4 --out data/val.jsonl --lang hi_latn
```

Multi-pack (romanised + Devanagari Hindi, the recipe used for the bundled
default weights):

```bash
python -m sankhya.generator --n 200000 --seed 5 --out data/train.jsonl \
  --lang hi_latn,hi_deva --mix 0.55,0.45 --cross 0.10
python -m sankhya.generator --n 6000 --seed 6 --out data/val.jsonl \
  --lang hi_latn,hi_deva --mix 0.55,0.45 --cross 0.10
```

Paths are relative to `python/`. `--lang` takes a comma-separated list of
registered `LanguagePack` ids (`hi_latn` Latin-script/romanised Hindi,
`hi_deva` Devanagari Hindi). `--mix` gives per-pack sampling weights
(defaults to uniform); `--cross` is the share of examples that mix a
template/phrase across packs (0 disables cross-mixing).

You can also regenerate the charset file directly if you need it standalone
— pass the same comma-separated `--lang` list used for the data so the
vocab is the union of every pack present (`build_charset_multi`):

```bash
python -m sankhya.charset --lang hi_latn,hi_deva --out data/charset.json
```

Every generated example also has a `P_UNK` (default 0.12) chance of getting
a 1-3 character out-of-vocab "unk noise" run inserted as O-labelled context
(emoji, CJK, Cyrillic, Greek, arrows, other currency symbols, fullwidth
punctuation — chars guaranteed absent from every pack's charset, so they
always map to `<unk>`), placed at the start/end of the text or at a random
space, sometimes glued to the adjacent word with no space, and occasionally
repeated (`🙏🙏`) to mimic JS's UTF-16 surrogate-pair splitting; this trains
the `<unk>` embedding, which clean synthetic data otherwise never exercises
even though production input (real emoji, foreign scripts, uncommon
symbols) hits it constantly. Tune or disable it with `--unk-noise FLOAT`
passed to `sankhya.generator` (0 disables).

## Train

```bash
python -m sankhya.train \
  --train data/train.jsonl --val data/val.jsonl \
  --epochs 20 --batch 128 --lr 3e-3 \
  --channels 32 --layers 4 --dilation 2 \
  --lang hi_latn,hi_deva \
  --out models/
```

This is the exact recipe used for the bundled default weights: 4 conv
layers (kernel sizes 3/5/3/3, dilations 1/1/2/4), 32 channels, 16-dim char
embeddings over the 114-character `hi_latn,hi_deva` union vocab — 18,811
parameters total. Writes `models/sankhya.pt` (a torch checkpoint carrying
`vocab`, `classes`, `dilation`, `channels`, `layers`, `state_dict`).

`--lang` accepts a comma-separated pack list; when more than one pack is
given, `train.py` builds the charset as the multi-pack union
(`build_charset_multi`) so no character from any pack maps to `<unk>` —
this matters most for `hi_deva`, whose Devanagari characters would
otherwise all collapse to `<unk>` if the charset were built from a single
(e.g. `hi_latn`-only) pack.

A channels=48 variant was also trained and compared on gold: it scored
marginally higher val value_acc (0.929 vs 0.919) but a *lower* combined
gold span F1 (0.942 vs 0.953) at ~1.9x the parameters (35,691), so
channels=32 remains the shipped default.

Other flags: `--label-smoothing`, `--grad-clip`, `--num-train` (subsample
the training set).

## Export

```bash
python -m sankhya.export --ckpt models/sankhya.pt --val data/val.jsonl --out models/
```

Reads the checkpoint (architecture params come from the checkpoint itself,
not from export flags) and writes to `models/`:

- `sankhya.onnx` — ONNX graph, for interop/inspection
- `sankhya.weights.json` — float32 weights, human-readable JSON (charset,
  classes, per-layer `w`/`b` tensors as nested lists)
- `sankhya.weights.int8.json` — int8-quantized weights (symmetric per-tensor,
  base64-packed), what actually ships in the npm package
- `charset.json`, `classes.json` — the vocab/class tables the checkpoint was
  trained with, as standalone files

`export.py` also runs a val-set sanity check (numpy reference forward pass
vs. the torch model) before writing, so a broken export fails loudly.

## Evaluate on gold

The hand-written gold sets (`tests/gold.jsonl`, 180 sentences / 161 spans,
romanised; `tests/gold_deva.jsonl`, 166 sentences / 141 spans, Devanagari)
are the only numbers to trust for real-world quality — synthetic validation
accuracy is optimistic because it's drawn from the same generator/templates
the model was trained on. `--gold` accepts multiple files; per-file and
combined metrics are printed.

```bash
python -m sankhya.eval_gold --gold tests/gold.jsonl tests/gold_deva.jsonl --ckpt models/sankhya.pt
```

or against the exported JSON weights directly (what the JS runtime actually
runs):

```bash
python -m sankhya.eval_gold --gold tests/gold.jsonl tests/gold_deva.jsonl \
  --weights-json models/sankhya.weights.json
# add --int8 with --weights-json models/sankhya.weights.int8.json
# to check the quantized weights specifically
```

Besides the overall precision/recall/F1/value_acc, `eval_gold.py` also
prints a per-category breakdown (digits vs. word-numerals, prefix forms,
ranges, currency, multi-unit, symbol units like "20k"/"1.5cr", mixed-script
spans, long spans) and negatives (zero-gold-span examples) with their
false-positive rate. `--json-out` includes `categories` and `negatives` for
both per-file and combined results. Pass `--quiet` to suppress the
missed/spurious/wrong-value/wrong-boundary listings and keep just the
tables.

### Comparing multiple models/configs (`eval_matrix.py`)

`python -m sankhya.eval_matrix run` evaluates several checkpoints/weights
files against the gold sets in one process and prints a single comparison
table (value_acc/F1 per gold file, combined, negatives FP rate) plus a
per-category table:

```bash
python -m sankhya.eval_matrix run \
  --models v1=models_v1_32_s0/sankhya.weights.int8.json \
           v2=models_v2_32_s0/sankhya.weights.int8.json \
  --json-out output/eval_matrix.json
# --gold defaults to every tests/gold*.jsonl file found (language inferred
# from the filename, e.g. a future tests/gold_mr_deva.jsonl "just works")
```

A model path ending `.pt` runs the torch checkpoint, `.int8.json` runs the
quantized JSON weights, and any other `.json` runs the float32 JSON weights.

`python -m sankhya.eval_matrix compare` groups a set of metrics-json files
(from `eval_gold.py --json-out`, `eval_matrix.py run --json-out`, or a
Kaggle `matrix.json`/`metrics.json`) by a config label and prints mean ±
std for combined int8 value_acc/F1 and per-file value_acc, to see seed
variance:

```bash
python -m sankhya.eval_matrix compare --runs output/run_seed0.json output/run_seed1.json \
  --group-key config   # dotted path into each JSON; default "config"
```

### Gold results (bundled channels=32 model, torch checkpoint)

| gold set          | examples | spans | precision | recall | F1     | value_acc |
|--------------------|---------:|------:|----------:|-------:|-------:|----------:|
| gold.jsonl          |      180 |   161 |    0.9222 | 0.9565 | 0.9390 |    0.9441 |
| gold_deva.jsonl      |      166 |   141 |    0.9583 | 0.9787 | 0.9684 |    0.9645 |
| combined             |      346 |   302 |    0.9389 | 0.9669 | 0.9527 |    0.9536 |

int8-quantized JSON weights (what the JS runtime actually ships):

| gold set          | examples | spans | precision | recall | F1     | value_acc |
|--------------------|---------:|------:|----------:|-------:|-------:|----------:|
| gold.jsonl          |      180 |   161 |    0.9222 | 0.9565 | 0.9390 |    0.9441 |
| gold_deva.jsonl      |      166 |   141 |    0.9580 | 0.9716 | 0.9648 |    0.9645 |
| combined             |      346 |   302 |    0.9387 | 0.9636 | 0.9510 |    0.9536 |

Known miss categories (24 misses, torch, combined gold): unusual typos
("croer", "five and a half crore" without a unit-noun boundary marker),
possessive apostrophes ("do lakh's"), long multi-term/mixed-numeral
constructs ("three n half lakh", "50M"), multi-number range phrases
("तीस पैंतीस हज़ार", "paanch se sadhe saat lakh"), and occasional spurious
spans or off-by-a-word boundaries on unfamiliar surrounding words
("mil", "poora", "raato raat").

## Ship to the npm package

The JS package embeds the int8 weights directly, plus a fixed set of
fixtures that pin the JS forward pass and decoder to this exact model.

1. Copy the quantized weights into the package:

   ```bash
   cp models/sankhya.weights.int8.json ../src/data/default-weights.json
   ```

2. Regenerate the two parity fixtures the JS test suite checks against:

   ```bash
   python -m sankhya.make_fixtures \
     --weights ../src/data/default-weights.json \
     --gold tests/gold.jsonl \
     --out-parity ../test/fixtures/parity.jsonl \
     --out-decoded ../test/fixtures/decoded.jsonl
   ```

   - `test/fixtures/parity.jsonl`: one line per gold text, `{text, bio, cls}`
     — the per-character int8 numpy forward pass's argmax BIO/class ids
     (padded with `PAD_TAIL=16` pad tokens via `np_infer.pad_ids`, matching
     training-time padding, then sliced back to the real length). The JS
     `test/parity.test.ts` runs its own CPU forward pass on the same texts
     and checks it produces identical argmax ids — this is what pins the JS
     port's arithmetic to the Python reference bit-for-bit.
   - `test/fixtures/decoded.jsonl`: one line per gold text, `{text, spans}`
     — the fully decoded output (`decode.decode_spans` with softmax
     `bio_probs` for the confidence gate, then `core.evaluate` on the
     decoded tokens, plus `core.detect_currency` against the `hi_latn`
     pack), each span normalised to `{start, end, value, range, unit,
     currency, classes}`. The JS `test/e2e-parity.test.ts` runs `parse()`
     on the same texts and checks it produces the same spans — this pins
     end-to-end behaviour (decode + arithmetic core), not just the raw
     model output.

   `make_fixtures.py` uses `np_infer.load_weights_int8_json` +
   `np_infer.forward` (int8-dequantized, matching what the JS runtime
   actually loads), `np_infer.pad_ids`/`softmax`, `decode.decode_spans`,
   and `core.evaluate`/`core.detect_currency` — i.e. the same reference
   pipeline `eval_gold.py` uses, factored out so it can write fixture files
   instead of computing a score.

3. Verify from the repo root:

   ```bash
   npm test
   ```

   Must show `48/48` passing (`parity: 180/180 texts identical to python
   np_infer` and the e2e-parity fixture check both included). Re-running
   `make_fixtures.py` against the same weights + gold set is deterministic
   (no RNG involved), so this reproduces the existing fixtures exactly.

## Tests

```bash
python tests/test_core.py
python tests/test_decode.py
python tests/test_generator.py
# or, if you installed pytest:
python -m pytest tests/
```

`test_core.py` covers the deterministic arithmetic core (prefix semantics,
additive/multiplicative unit combination, ranges) directly on class-token
sequences, independent of the model. `test_decode.py` covers `decode_spans`
(BIO decode, gap bridging, digit-run extension, class repair, confidence
filtering). `test_generator.py` round-trips the synthetic generator's own
labels through the core evaluator. `tests/gold.jsonl` is the hand-written
gold set used by `eval_gold.py`, not a generator round-trip test.

## Colab

See `colab.md` for a copy-pasteable notebook recipe (same commands as
above, runs fine on Colab's CPU runtime).

## Training on Kaggle (GPU)

`python/kaggle_train/` wraps a Kaggle "script" kernel that runs the full
multi-pack recipe above (`--lang hi_latn,hi_deva --mix 0.55,0.45 --cross
0.10`, 200k train / 6k val, `--epochs 20 --channels 32 --layers 4
--dilation 2`) on a Kaggle GPU, then evaluates on both gold sets and
stages `models/` + `metrics.json` for download.

### Prerequisites

- A Kaggle account with **phone verification** — required for internet
  access and GPU quota on kernels. Without it, `enable_internet`/GPU
  kernels are rejected server-side even with valid credentials.
- Kaggle CLI credentials. `kaggle` (2.2.4+) accepts, in priority order:
  1. `KAGGLE_API_TOKEN=<token>` env var (generate at
     https://www.kaggle.com/settings/api)
  2. a token file at `~/.kaggle/access_token`
  3. `kaggle auth login` (OAuth, cached under `~/.kaggle/`)
  4. legacy: `~/.kaggle/kaggle.json` (or `KAGGLE_USERNAME` +
     `KAGGLE_KEY` env vars), optionally relocated via `KAGGLE_CONFIG_DIR`

  `python -m kaggle_train.run` checks for one of these and fails fast
  with the same list if none is set — it never invents its own env var
  names.

### Commands (run from `python/`)

```bash
python -m kaggle_train.run push              # upload the kernel, start the GPU run
python -m kaggle_train.run status             # poll every 30s until complete/error
python -m kaggle_train.run pull               # download output, update weights + fixtures
python -m kaggle_train.run all                # push, then status, then pull
```

Override a training default with `--set KEY=VALUE` (repeatable) before
`push`/`all`, e.g. `--set EPOCHS=30 --set GIT_REF=my-branch`. Valid keys:
`REPO_URL`, `GIT_REF`, `N_TRAIN`, `N_VAL`, `LANGS`, `MIX`, `CROSS`,
`EPOCHS`, `CHANNELS`, `LAYERS`, `DILATION`, `TRAIN_SEED`, `VAL_SEED`,
`MATRIX`.

### Training a matrix of configs

`MATRIX` (default `"v1:32:0"`) is a comma-separated list of
`arch:channels:seed` entries. The kernel generates the data once, then for
each entry trains (`--arch --channels --seed --device auto`), exports, and
evaluates on gold (float32 + int8), writing `output/matrix.json` (every
entry's val + gold metrics) and `output/matrix.md` (a markdown table,
printed at the end of the kernel log too).

Winner selection (one implementation, `sankhya.eval_matrix.select_matrix_winner`,
shared by `train_kernel.py` and `eval_matrix.py`):
1. Winning **config** (arch:channels) = highest mean int8 combined gold
   value_acc across its seeds.
2. Within that config, every seed within **0.005 val value_acc** of the
   best seed counts as tied (near-tied seeds, e.g. 0.9343 vs. 0.9351,
   aren't decided by noise); among them, higher int8 combined gold F1
   wins, then lower negatives false-positive rate, then input order.

That winning run's `models/` and metrics are staged at `output/models/` /
`output/metrics.json` exactly as a single-config run would be (plus a
`matrix_winner` field), so `run.py pull` needs no changes. Every entry
(not just the winner) also gets its exported weights + gold metrics staged
at `output/entries/<arch>_<channels>_s<seed>/` (`sankhya.weights.int8.json`,
`sankhya.weights.json`, `gold_metrics_torch.json`,
`gold_metrics_json_float32.json`, `gold_metrics_json_int8.json` -- no `.pt`
or `.onnx`, to keep the download small), so picking a different seed
afterwards doesn't require a full re-run:

```bash
python -m kaggle_train.run push --set MATRIX=v1:32:0,v2:32:0,v2:32:1
```

`SMOKE=1` (tiny local dry run against a filesystem `REPO_URL`) still works
and now copies your **working tree** (including uncommitted changes) on
top of the git clone, not just the committed history — useful when testing
against in-progress local edits.

Expected wall time: roughly 15-30 minutes on a Kaggle T4/P100 for the
default 200k-example recipe (vs. ~12 minutes on 4 CPU cores at the same
size — GPU mainly helps at larger `N_TRAIN`/`EPOCHS`), plus queueing time
for the kernel to be scheduled.

**After `pull`, run `npm test` from the repo root before committing.**
`pull` overwrites `src/data/default-weights.json` and regenerates
`test/fixtures/parity.jsonl` / `decoded.jsonl` from the new weights —
`npm test` must pass (`63/63`) before those changes are committed.

See the module docstrings in `python/kaggle_train/run.py` and
`python/kaggle_train/train_kernel.py` for how the kernel is structured,
and `python/kaggle_train/kernel-metadata.json` for the exact Kaggle
kernel settings (GPU + internet enabled, private script kernel).

## Adding a language pack

Only Indian languages are in scope. To add one, construct a
`sankhya.langs.base.LanguagePack` (see `sankhya/langs/hi_latn.py` for a
full example) with:

- `lexicon` — per-class word lists (prefixes like "sava"/"dedh", cardinal
  number words, unit words like lakh/crore/hazaar, misc/filler classes)
- `symbol_units` — symbol forms of units (`"L"`, `"Cr"`, `"k"`, ...)
- `english_fraction_phrases` — phrases like "one and a half"
- `indefinite_plurals` — words that must be tagged O even though they look
  number-ish (collision guards)
- `currency_markers_before` / `currency_words_after` — Rs./₹/rupaye-style
  markers detected adjacent to a span
- `approximators` — "around", "lagbhag", etc.
- `range_connectors` — "-", "se", "to", etc., for range phrases
- `templates` — sentence templates per family (bare amount, currency
  phrase, range, duration negative, ...) used to synthesize examples
- `filler_words` — generic O-labelled words used around spans as negatives
- `duration_nouns` — "din"/"minute"/etc., so bare numbers next to a
  duration noun are *not* tagged as a span
- `blocked_surfaces` — surface strings the noise function must never
  produce (collisions with real words/names)
- `noise_fn(word, rng) -> str` — typo/spelling-variance injection for that
  script/language

then `sankhya.langs.base.register(pack)` it and give it a unique `id`
(register in `sankhya/langs/__init__.py` alongside the existing packs so
`--lang <id>` resolves it). Regenerate the charset for the new pack
(`python -m sankhya.charset --lang <id> --out data/charset.json`) — the
model's char vocab is per-language. `sankhya/classes.py`'s class inventory
(the BIO/semantic token classes) is shared across all languages and should
not need to change for a new pack; `core.py` (the arithmetic core) is
fully language-independent and never needs to change either — only the
class *vocabulary* a pack uses and its currency-marker lists are
per-language.

A Devanagari pack (e.g. Hindi in Devanagari script, डेढ़ लाख) additionally
needs a new noise module for script-specific variance (matra/nukta
elision or substitution) since `noise_latn.py`'s typo model is
Latin-script-specific; the class inventory and core still don't change.
