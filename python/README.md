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

CPU-only torch is fine — training the current shipped multi-pack default
model (arch `v2`, channels 48, 39,579 params) takes roughly 2x as long as
the previous `v1:32` model's ~12 minutes on 4 CPU cores for 200k examples
(a Kaggle GPU run of the same config is ~3 minutes).

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
  --arch v2 --channels 48 --seed 0 \
  --lang hi_latn,hi_deva \
  --out models/
```

This is the exact recipe used for the bundled default weights: arch `v2`
(`sankhya.model.ARCHS["v2"]`) is 5 conv layers over 16-dim char
embeddings — a plain kernel-5 layer, then four residual kernel-3 layers
(`y = relu(conv(x)) + x`) with dilations 1/2/4/8 — at 48 channels, over
the 114-character `hi_latn,hi_deva` union vocab: 39,579 parameters total.
Writes `models/sankhya.pt` (a torch checkpoint carrying `vocab`,
`classes`, `arch`, `channels`, `state_dict`).

`--arch` selects a preset from `sankhya.model.ARCHS` (`v1`: the older
4-layer, non-residual, kernel 3/5/3/3, dilation 1/1/2/4 stack; `v2`: the
5-layer residual stack above). Both remain loadable by every runtime
(torch, numpy reference, JS CPU, JS WebGPU) regardless of which is
shipped — weights JSON version 2 carries the arch spec, so old `v1`
weights keep working. The legacy `--layers`/`--dilation` flags still work
and reconstruct the `v1` shape for old call sites.

`--lang` accepts a comma-separated pack list; when more than one pack is
given, `train.py` builds the charset as the multi-pack union
(`build_charset_multi`) so no character from any pack maps to `<unk>` —
this matters most for `hi_deva`, whose Devanagari characters would
otherwise all collapse to `<unk>` if the charset were built from a single
(e.g. `hi_latn`-only) pack.

`--unk-noise FLOAT` (also read by `sankhya.generator`, see above) tunes
the out-of-vocab noise augmentation rate; the shipped model was trained
with the generator's default `P_UNK=0.12` so the `<unk>` embedding
actually gets gradient signal.

A sweep across `arch`/`channels`/`seed` (see "Comparing multiple
models/configs" and the Kaggle matrix below) is how `v2`/48 channels was
chosen over `v1`/32 and other configs — always compare by **mean int8
combined gold value_acc over >= 3 seeds**, not a single run: seed-to-seed
spread is +/-1-2 points, so a single lucky or unlucky seed is not a
reliable signal. `--seed` sets the training seed (with deterministic
cuDNN, a given seed reproduces the same result).

Other flags: `--label-smoothing`, `--grad-clip`, `--num-train` (subsample
the training set).

### Mixing in an extra corpus (`--extra`)

```bash
python -m sankhya.train \
  --train data/train.jsonl --val data/val.jsonl \
  --extra data_llm/hi_latn.jsonl --extra-ratio 0.2 \
  --lang hi_latn --out models/
```

`--extra` takes one or more jsonl files in the same schema as `--train`
(text/lang/bio/cls/spans — e.g. the output of `sankhya.llm_corpus verify`,
see "LLM-written corpus" below) and mixes them into training alongside the
synthetic `--train` set. Mixing works per-epoch, not as a one-time
concatenation: let `n_main` be the size of `--train` and `n_extra_pool` the
size of the `--extra` pool. Each epoch, `n_extra_per_epoch =
round(n_main * ratio / (1 - ratio))` extra examples are freshly drawn from
the pool (`ratio` = `--extra-ratio`, so `n_extra_per_epoch / (n_main +
n_extra_per_epoch) ~= ratio`) — **without** replacement if the pool is at
least that big that epoch (a subsample), **with** replacement otherwise (an
oversample) — then shuffled together with the full `--train` set for that
epoch's batches. Because the sample is redrawn every epoch, even a small
extra pool gets seen in different combinations rather than being replayed
identically. `--extra-ratio 0` (or omitting `--extra`) disables mixing
entirely — unchanged behaviour.

The extra corpus is tensorized against the **same charset** built from
`--lang` (never rebuilt to include extra-only characters), so any character
in it that isn't in a pack's charset maps to `<unk>` exactly like unknown
production input would — `train.py` prints a warning listing the 20 most
frequent such characters (and how many extra examples they came from) so
you can see whether an unexpected script slipped in.

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

The hand-written gold sets (`tests/gold.jsonl`, 220 sentences / 194 spans,
romanised; `tests/gold_deva.jsonl`, 182 sentences / 151 spans, Devanagari
— this round added conjunction-joined multi-span sentences,
trailing-cardinal chains, new negative families, possessive noise,
in-context typos, and curated romanised cardinal variants 11-99) are the
only numbers to trust for real-world quality — synthetic validation
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

### Gold results (bundled `v2`/48-channel model, int8 JSON weights — what the JS runtime ships)

```
python -m sankhya.eval_gold --gold tests/gold.jsonl tests/gold_deva.jsonl \
  --weights-json ../src/data/default-weights.json --int8
```

| gold set          | examples | spans | precision | recall | F1     | value_acc |
|--------------------|---------:|------:|----------:|-------:|-------:|----------:|
| gold.jsonl          |      214 |   188 |    0.9293 | 0.9485 | 0.9388 |    0.9381 |
| gold_deva.jsonl      |      182 |   151 |    0.9867 | 0.9801 | 0.9834 |    0.9735 |
| combined             |      396 |   339 |    0.9540 | 0.9623 | 0.9582 |    0.9536 |

Negatives: 73 examples, 0 false positives. Miss summary: missed 2,
spurious 7, wrong value 3, wrong boundary 13. Per-category value_acc:
digits 0.967, words 0.940, prefix 0.926, range 0.917, currency 1.0,
multi_unit 0.952, symbol_unit 0.970, mixed_script 1.0, long 0.80.

For comparison, the previous shipped weights scored, on this same
enlarged gold set, 0.9096 (romanised) / 0.9603 (Devanagari) / 0.9322
(combined) value_acc, 0.9258 combined F1, 4/73 (5.5%) negatives false
positives. Their numbers on the older, smaller 353-example gold set were
0.9515 / 0.9861 / 0.9676 combined value_acc — the new gold set is
deliberately harder: it adds conjunction-joined multi-span sentences,
trailing-cardinal chains, possessives, in-context typos, cardinal
spelling variants, and 18 more negative examples. See the sweep evidence
below for why `v2`/48 seed 2 is the currently shipped run.

Known miss categories: wrong span boundaries on multi-span/range/
connector phrases (the largest category this round), wrong value on a
handful of prefix/compound constructs, unusual typos ("croer", "five and
a half crore" without a unit-noun boundary marker), long
multi-term/mixed-numeral constructs ("three n half lakh", "50M",
"paanch se sadhe saat lakh"), and occasional spurious spans on
unfamiliar surrounding words ("mil", "poora", "raato raat").

### Sweep evidence: why `v2`/48 channels is shipped

Config/seed sweeps were run on the Kaggle GPU matrix kernel (deterministic
cuDNN, int8 combined gold value_acc on the older 346-example gold set,
before the `<unk>`-noise generator change):

| config      | seed 0 | seed 1 | seed 2 | mean  |
|-------------|-------:|-------:|-------:|------:|
| v1:32       | 0.940  | 0.934  | 0.934  | 0.936 |
| v2:32       | 0.947  | 0.957  | 0.944  | 0.949 |

(the previously *shipped* `v1:32` weights, at 0.954, were a lucky seed —
0.936 is the honest mean for that config.) `v2:48` across five seeds:
0.950 / 0.957 / 0.964 / 0.944 / 0.940 (mean 0.951). After adding
`<unk>`-noise augmentation, `v2:48` seeds 0-2 on the newer 353-example
gold set scored 0.968 / 0.948 / 0.951.

**Currently shipped:** a 3-seed `v2:48` sweep run on the enriched
generator (conjunction multi-span family, trailing-cardinal chains, new
negative families, possessive noise, in-context typos, curated cardinal
variants), scored int8 combined gold value_acc / F1 / negatives
false-positive rate:

| seed | value_acc | F1    | negatives FP |
|------|----------:|------:|--------------:|
| 0    | 0.950     | 0.943 | 2.7%          |
| 1    | 0.950     | 0.948 | 2.7%          |
| 2    | 0.947     | 0.949 | 0.0%          |

Seeds 0/1 tie on val value_acc within the 0.005 band ahead of seed 2, so
the winner is picked by the tie-break rule below: seed 2 has the lowest
negatives false-positive rate (0.0%) of the tied/near-tied set and a
competitive F1, so **seed 2 is the shipped weights**.

**The rule this implies, and what `eval_matrix.select_matrix_winner` /
the Kaggle MATRIX kernel actually do:** pick the winning **config**
(arch:channels) by highest **mean** int8 combined gold value_acc across
its seeds (seed-to-seed spread is +/-1-2 points, so never compare configs
by a single seed); within the winning config, pick the **seed** by val
value_acc within a 0.005 tie band, then higher int8 combined gold F1,
then lower negatives false-positive rate.

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

   Must show every test passing (`parity: 391/391 texts identical to python
   np_infer` and the e2e-parity fixture check both included). Re-running
   `make_fixtures.py` against the same weights + gold set is deterministic
   (no RNG involved), so this reproduces the existing fixtures exactly.

## LLM-written corpus

`python -m sankhya.llm_corpus` verifies free-text lines written by an LLM
(following `data_llm/prompts/GENERATOR_BRIEF.md`) against this repo's own
lexicon + `core.evaluate`, and turns accepted lines into the same
text/lang/bio/cls/spans schema `sankhya.generator` produces, so they load
through `train.py`'s `load_jsonl`/`tensorize` unchanged.

**Format.** Other agents write one JSON object per line to
`data_llm/raw/<lang>.<agent>.jsonl`:

```json
{"text": "bhai sava lakh mein ho jayega kya", "lang": "hi_latn", "phrases": [{"phrase": "sava lakh", "value": 125000}], "scenario": "casual", "gen": "sonnet"}
```

`phrases` is `[]` for negatives (no quantity in the line); `phrase` must be
an exact substring of `text`; `range` is included alongside `value` for
range phrases (`"phrase": "2-3 lakh", "value": 200000, "range": [200000, 300000]`).
See the prompt file for the full spec (registers, forms, coverage targets).

**Verify:**

```bash
python -m sankhya.llm_corpus verify --lang hi_latn \
  --in data_llm/raw/*.jsonl --out data_llm/hi_latn.jsonl \
  --rejects data_llm/rejects/hi_latn.jsonl --manifest data_llm/manifest.json
```

For each raw line: text is located via `charset.normalize_text` substring
matching, the phrase is tokenized with the SAME greedy lexicon lookup
`sankhya.generator` uses to assign per-char classes, and the resulting
token stream is run through `core.evaluate` and compared against the
claimed `value`/`range`. Rejected reasons include `phrase_not_found`,
`ambiguous_phrase`, `overlapping_phrase`, `unknown_token:<token>` (a
letters-run not in the pack's lexicon), `value_mismatch: ours=X theirs=Y`,
`negative_contains_quantity` (a claimed negative actually contains a
`UNIT_`/`PFX_` lexicon form — including bare symbol units like the `k` in
`50k`), `dup` (duplicate normalized text within this run), and `dup_gold`
(duplicate of a line already in `tests/gold*.jsonl`). A line whose `lang`
field doesn't match `--lang` is skipped as `lang_mismatch`, so `--in
data_llm/raw/*.jsonl` can safely glob every agent's raw files across
languages in one call. `verify` prints a summary table (totals, reject
histogram, top 15 unknown tokens, negatives share, mean length) and writes/
merges a per-lang entry into `--manifest` (counts, timestamp, the prompt
file's sha256, input files, reject histogram).

**Inspect coverage:**

```bash
python -m sankhya.llm_corpus stats --in data_llm/hi_latn.jsonl
python -m sankhya.llm_corpus sample --in data_llm/hi_latn.jsonl --n 20 --seed 1
```

`stats` prints the class histogram (which `CARD_`/`UNIT_`/`PFX_` classes
appear and how often), the share of lines with 2+ spans, the negatives
share, and unique surface forms per class — compare this against
`sankhya.generator`'s own coverage to see what the LLM corpus adds.
`sample` prints a random sample of lines with their phrase/value pairs for
eyeballing.

**Rejects workflow.** `--rejects` writes every rejected line (its
`text`/`phrases`, the reject reason, and its source file/line number) to a
separate jsonl file — hand this back to whichever agent generated it so
bad lines can be fixed or dropped, without re-deriving why each one
failed. `data_llm/raw/` and `data_llm/rejects/` are both tracked in git
(review artefacts, not build output) — see `.gitignore`.

**Mixing into training.** The verified output is a normal `--extra` file
for `sankhya.train` (see "Mixing in an extra corpus" above):

```bash
python -m sankhya.train --train data/train.jsonl --val data/val.jsonl \
  --extra data_llm/hi_latn.jsonl --extra-ratio 0.2 --lang hi_latn --out models/
```

On Kaggle, set `EXTRA` (comma-separated repo-relative paths under
`python/`) and `EXTRA_RATIO` via `run.py push --set
EXTRA=data_llm/hi_latn.jsonl --set EXTRA_RATIO=0.2` — see "Training on
Kaggle" below.

## Tests

```bash
python tests/test_core.py
python tests/test_decode.py
python tests/test_generator.py
python tests/test_llm_corpus.py
# or, if you installed pytest:
python -m pytest tests/
```

`test_core.py` covers the deterministic arithmetic core (prefix semantics,
additive/multiplicative unit combination, ranges) directly on class-token
sequences, independent of the model. `test_decode.py` covers `decode_spans`
(BIO decode, gap bridging, digit-run extension, class repair, confidence
filtering). `test_generator.py` round-trips the synthetic generator's own
labels through the core evaluator. `test_llm_corpus.py` covers
`sankhya.llm_corpus.verify_line`'s accept/reject decisions (value
mismatches, unknown tokens, bad phrase substrings, hidden-quantity
negatives, duplicates) against inline fixtures. `tests/gold.jsonl` is the
hand-written gold set used by `eval_gold.py`, not a generator round-trip
test.

## Colab

See `colab.md` for a copy-pasteable notebook recipe (same commands as
above, runs fine on Colab's CPU runtime).

## Training on Kaggle (GPU)

`python/kaggle_train/` wraps a Kaggle "script" kernel that runs the full
multi-pack recipe above (`--lang hi_latn,hi_deva --mix 0.55,0.45 --cross
0.10`, 200k train / 6k val, 20 epochs, batch 128, lr 3e-3) on a Kaggle
GPU, then evaluates on both gold sets and stages `models/` +
`metrics.json` for download. The default `MATRIX` (below) trains
`arch:channels:seed` configs — the shipped default is `v2:48`; legacy
`CHANNELS`/`LAYERS`/`DILATION` overrides still exist for the non-MATRIX
single-run path (which defaults to the `v1` arch shape). Training on the
GPU takes about 3 minutes per model for this recipe (vs. roughly 12
minutes on 4 CPU cores for `v1:32`, about 2x that for `v2:48` on CPU).

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
python -m kaggle_train.run pull               # download output, update weights + fixtures + models/default/
python -m kaggle_train.run all                # push, then status, then pull
```

Override a training default with `--set KEY=VALUE` (repeatable) before
`push`/`all`, e.g. `--set EPOCHS=30 --set GIT_REF=my-branch`. Valid keys:
`REPO_URL`, `GIT_REF`, `N_TRAIN`, `N_VAL`, `LANGS`, `MIX`, `CROSS`,
`EPOCHS`, `CHANNELS`, `LAYERS`, `DILATION`, `TRAIN_SEED`, `VAL_SEED`,
`MATRIX`, `EXTRA`, `EXTRA_RATIO`.

`EXTRA` (default `""`, disabled) is a comma-separated list of
repo-relative paths under `python/` — e.g. a verified LLM corpus from
`sankhya.llm_corpus verify` — mixed into every matrix entry's training run
via `sankhya.train --extra ... --extra-ratio EXTRA_RATIO` (see "Mixing in
an extra corpus" above); `EXTRA_RATIO` defaults to `0.2`:

```bash
python -m kaggle_train.run push --set EXTRA=data_llm/hi_latn.jsonl,data_llm/hi_deva.jsonl --set EXTRA_RATIO=0.2
```

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
`npm test` must pass before those changes are committed. `pull`
also copies the winning run's checkpoint + exports (`sankhya.pt`,
`sankhya.onnx`, `sankhya.weights.json`, `sankhya.weights.int8.json`,
`charset.json`, `classes.json`, `gold_metrics_*.json`) plus
`output/matrix.md` and `output/metrics.json` into `models/default/` (repo
root, tracked in git) and prints what it copied — this is the tree the
`huggingface` workflow publishes from (see "Publishing to Hugging Face"
below).

## Publishing to Hugging Face

`python -m sankhya.hf_push model|space|dataset|all [--dry-run]` (from
`python/`, needs `pip install -r requirements-hf.txt`) pushes:

- `model` — `models/default/*` plus a generated model card, to
  https://huggingface.co/athrvk/gpu-sankhya (and tags the revision
  `vX.Y.Z` from `package.json`'s version)
- `space` — the built `site/` directory (run `npm run build && npm run
  site` from the repo root first), to
  https://huggingface.co/spaces/athrvk/gpu-sankhya-demo
- `dataset` — `python/tests/gold.jsonl` / `gold_deva.jsonl`, to
  https://huggingface.co/datasets/athrvk/gpu-sankhya-gold

Requires an `HF_TOKEN` env var (a write-scoped Hugging Face token); repos
are created automatically on first push. `--dry-run` writes the generated
cards + file lists to `python/hf_out/` without any network access, so you
can review them before running for real.

In CI, the `.github/workflows/huggingface.yml` workflow does this
automatically: on `workflow_dispatch` (pick `all`/`model`/`space`/`dataset`
via the `target` input), and on push to `master` when `package.json`,
`models/default/**`, `python/tests/gold*.jsonl`, or `demo/**` change. It
needs an `HF_TOKEN` repo secret (Settings -> Secrets and variables ->
Actions -> New repository secret) with write access to the `athrvk`
namespace on huggingface.co; without it, the jobs print a notice and skip.

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
