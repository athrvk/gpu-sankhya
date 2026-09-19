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
model (arch `v2`, channels 48, 40,503 params) takes roughly 2x as long as
the previous `v1:32` model's ~12 minutes on 4 CPU cores for 200k examples
(a Kaggle GPU run of the same config is ~3 minutes).

## Generate data

Single-pack (Latin-script/romanised Hindi only):

```bash
python -m sankhya.generator --n 150000 --seed 3 --out data/train.jsonl --lang hi_latn
python -m sankhya.generator --n 5000 --seed 4 --out data/val.jsonl --lang hi_latn
```

Multi-pack (romanised + Devanagari Hindi + Devanagari Marathi + Gujarati,
the recipe used for the bundled default weights):

```bash
python -m sankhya.generator --n 200000 --seed 5 --out data/train.jsonl \
  --lang hi_latn,hi_deva,mr_deva,gu_gujr --mix 0.32,0.26,0.21,0.21 --cross 0.10
python -m sankhya.generator --n 6000 --seed 6 --out data/val.jsonl \
  --lang hi_latn,hi_deva,mr_deva,gu_gujr --mix 0.32,0.26,0.21,0.21 --cross 0.10
```

Paths are relative to `python/`. `--lang` takes a comma-separated list of
registered `LanguagePack` ids; the shipped packs are

| id | language / script | in the bundled weights? |
| --- | --- | --- |
| `hi_latn` | romanised Hindi / Hinglish | yes |
| `hi_deva` | Devanagari Hindi | yes |
| `mr_deva` | Devanagari Marathi | yes |
| `gu_gujr` | Gujarati | yes |

(`sankhya.langs.base.KNOWN_PACKS` is the authoritative list; every tool
that needs "all packs" iterates `base.all_packs()` rather than hard-coding
ids). `--mix` gives per-pack sampling weights (defaults to uniform);
`--cross` is the share of examples that mix a template/phrase across
packs (0 disables cross-mixing).

You can also regenerate the charset file directly if you need it standalone
— pass the same comma-separated `--lang` list used for the data so the
vocab is the union of every pack present (`build_charset_multi`):

```bash
python -m sankhya.charset --lang hi_latn,hi_deva,mr_deva,gu_gujr --out data/charset.json
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
  --arch v2 --channels 48 --seed 2 \
  --lang hi_latn,hi_deva,mr_deva,gu_gujr \
  --out models/
```

This is the exact recipe used for the bundled default weights: arch `v2`
(`sankhya.model.ARCHS["v2"]`) is 5 conv layers over 16-dim char
embeddings — a plain kernel-5 layer, then four residual kernel-3 layers
(`y = relu(conv(x)) + x`) with dilations 1/2/4/8 — at 48 channels, over
the 170-character `hi_latn,hi_deva,mr_deva,gu_gujr` union vocab: 40,503
parameters total.
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

### Real negatives (`--extra-negatives`)

```bash
# regenerate the pool (gitignored: it is derived from CC BY-SA corpus text)
python -m sankhya.wild negatives --out data_wild/negatives.jsonl --n 20000

python -m sankhya.train \
  --train data/train.jsonl --val data/val.jsonl \
  --extra data_llm/hi_latn.jsonl data_llm/hi_deva.jsonl \
          data_llm/mr_deva.jsonl data_llm/gu_gujr.jsonl --extra-ratio 0.2 \
  --extra-negatives data_wild/negatives.jsonl --extra-negative-ratio 0.15 \
  --lang hi_latn,hi_deva,mr_deva,gu_gujr --out models/
```

`data_wild/REPORT.md` measured a **20% false-positive rate on real sentences
with no amount in them** (8.8% of them produce a *verified* span) against 0%
on the hand-written negatives — the model has never seen real non-amount text,
so names (`अण्णा हजारे`), ethnonyms (`अरब`, `અરબ`), measurements and years look
like amounts to it. `--extra-negatives` is the data-side answer: real
sentences, labelled all-`O`.

`--extra-negatives` takes jsonl file(s) of `{"text": ..., "lang": ...}` lines
(no `bio`/`cls`/`spans` — they are *derived*: BIO all 0, class all `O`, for the
normalized, `MAX_LEN`-truncated text). They are mixed per epoch exactly like
`--extra`: `n_neg_per_epoch = round(base * r / (1 - r))` where `base` is
`--train` plus that epoch's `--extra` draw, so `--extra-negative-ratio 0.15`
makes real negatives ~15% of the combined epoch and the two ratios compose.
The pool is redrawn each epoch (without replacement when it is big enough,
with replacement otherwise). `--extra-negative-ratio 0` disables it.

**Building the pool** — `python -m sankhya.wild negatives`:

- draws a uniform reservoir sample of real lines per language from the same
  corpora and filters `sankhya.wild sample` uses (3–40 words, ≤ 128 chars,
  deduplicated);
- keeps only lines the **lexicon scorer** gives no amount signal at all
  (score 0: no unit word, prefix, cardinal, currency marker or digit);
- *and* only lines the **currently shipped int8 weights leave alone** — a line
  the model spans might be a real amount the lexicon missed, and training on
  it as all-`O` would teach the wrong thing. About 1–4% of no-signal lines are
  dropped this way (worst in `hi_latn`);
- excludes every text in `tests/gold_wild_*.jsonl` and asserts the result is
  disjoint from it, so the evaluation set cannot leak into training;
- balances the kept lines across the four languages (`--n 20000` → 5,000 each).

The output is **derived from CC BY-SA corpus text**, so like `data_wild/raw/`
and `data_wild/samples/` it is **gitignored**; only the tool is committed.
Regenerate it with the command at the top of this section (seeded, so it is
reproducible).

## Masked-character pretraining (`pretrain.py`)

```bash
python -m sankhya.pretrain --out models/pretrain --n 500000 --epochs 6 \
  --arch v2 --channels 48 --lang hi_latn,hi_deva,mr_deva,gu_gujr \
  --cache data_wild/pretrain_corpus.jsonl

python -m sankhya.train ... --init-from models/pretrain/pretrain.pt
```

The same "the model has never seen real text" problem as above, attacked
unsupervised instead: take the **same `SankhyaCNN` trunk** (char embedding +
conv stack), put a temporary `channels -> vocab_size` head on it, mask 15% of
the characters of real sentences and train it to reconstruct **only the masked
positions** (`CrossEntropyLoss(ignore_index=-100)`, so padding and unmasked
characters contribute nothing to the loss). The head is then thrown away.

- Corpus: `--n` real sentences pulled through `sankhya.wild`'s loaders and
  filters (3–40 words, ≤ 128 chars, deduplicated), balanced across the four
  languages by water-filling (a language with fewer lines than its equal share
  hands the remainder to the ones that have more — `hi_latn` only has ~12k
  usable lines, `hi_deva` has 716k). Every `tests/gold_wild_*.jsonl` text is
  excluded, so pretraining cannot memorise the evaluation set. `--cache PATH`
  writes/reads the collected sentences so a re-run skips the corpus scan.
- Charset: the **shipped** one, built from `--lang` exactly as `train.py`
  builds it. Real characters outside it map to `<unk>`, as production input
  does. The mask token is `<unk>` too, which keeps the embedding table the
  exact shape the tagger expects.
- Budget: ~40k parameters; on 4 CPU cores one epoch over 500k sentences is a
  few minutes, so pick `--epochs` after timing the first one.

`train.py --init-from CKPT` then loads **only the trunk** tensors
(`embed.*`, `conv*.*`) out of that checkpoint and leaves the BIO and class
heads at their fresh torch init; a shape or key mismatch is fatal rather than
silently partial. It composes with everything else (`--extra`,
`--extra-negatives`, `--arch`, `--channels`), which is how experiment B in
`data_wild/REPORT.md` was run.

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

## CRF head for BIO decoding (optional, not used by shipped weights)

Pass `--crf` to `sankhya.train` to add a linear-chain CRF over the BIO
emissions (`O`/`B`/`I`), in addition to the per-character class head:

```bash
python -m sankhya.train --train data/train.jsonl --val data/val.jsonl --crf --out models/
```

- Training loss is `crf_nll + label-smoothed class-weighted BIO CE + 1e-3
  L2 on trans/start/end` (the class head is unchanged).
- The checkpoint gets `ckpt["use_crf"] = True` and `ckpt["crf"] =
  {"trans", "start", "end"}` (3x3 / 3 / 3 lists); `export.py` writes the
  same tensors, in float (never quantized — it's only 15 numbers), as a
  top-level `"crf"` block into *both* `sankhya.weights.json` and
  `sankhya.weights.int8.json`. The ONNX graph is unaffected — `forward()`
  always returns raw emissions; the CRF only changes how those emissions
  get decoded into a BIO path (Viterbi, `sankhya/crf.py`).
- Weights files without a `crf` block (every file exported before this,
  or trained without `--crf`) keep decoding BIO via plain per-position
  argmax — this is fully backward compatible. `np_infer.bio_path(...)`
  is the single choke point that picks Viterbi vs. argmax based on
  whether `weights["crf"]` is present, and is used by `eval_gold.py`,
  `make_fixtures.py`, and `export.py`'s int8 value-accuracy check.
- The Kaggle `MATRIX` env var accepts an optional 4th `arch:channels:seed:crf`
  field (`crf` ∈ {0,1}, default 0) — see `python/kaggle_train/train_kernel.py`.
- End-to-end parity is tested: Python train/export/`np_infer`, the JS
  runtime, and the WebGPU results path all handle a `crf` block
  identically to a weights file without one.

### Findings (T4, v2 arch, 48ch, 20 epochs, generator + verified LLM
corpora at 20%, deterministic seeds)

| run | combined value acc | Hinglish | Devanagari | F1 | spurious | FP/73 neg | s/epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| plain (no CRF), 3 seeds | 0.959 / 0.957 / 0.962 | — | — | — | — | — | 17 |
| plain, shipped 0.3.3 | 0.965 | 0.954 | 0.980 | 0.968 | 0 | 0/73 | 17 |
| plain, shipped 0.3.4 (seed 0, CPU, 410-example gold) | 0.963 | 0.954 | 0.974 | 0.966 | 0 | 0/75 | 75 (CPU) |
| CRF, pure NLL objective | 0.968 (best epoch; diverged to NaN by ep 20) | — | — | — | — | — | 118 |
| CRF, fixed objective (above) | 0.957 (best ckpt; loss still exploded to 1.4e6 by ep 20) | 0.949 | 0.967 | — | 0 | 0/73 | 118 |

Both CRF runs used a single seed. Neither beat the plain model's combined
value accuracy, and the CRF's sequential 128-step forward/Viterbi loop is
launch-bound on the T4 — ~7x slower per epoch (118 s vs 17 s, ~41 min vs
~6 min per run) — and truncating to the batch's max length doesn't help,
since almost every batch has a near-128-character example. Emissions were
verified bounded on a local repro of the fixed objective (max|emissions|
plateaus around 13, vs. unbounded growth under the pure-NLL objective),
but training loss still blew up late under the fixed objective too, with
OneCycle LR already near zero by then — not reproduced in a 6-epoch local
run, so the cause is still open (suspects: Adam optimizer state on the
tiny 15-parameter CRF head, or float32 forward-algorithm (alpha)
accumulation over long sequences).

The likely reason the CRF doesn't help here: the structural decoding
rules above (single-`O` bridging, word integrity, connector repair,
unanimous-class extension) already give the label sequence most of the
label-sequence coherence a CRF would add, leaving little for a learned
transition structure to win. `--crf` is kept as a tested, parity-verified,
opt-in flag for further experimentation, but the shipped weights do not
use it. If revisited: try a lower learning rate for the CRF parameters,
float64 alpha accumulation, or — simplest — a fixed (non-learned)
transition constraint matrix, which needs no training at all and would
remove both the instability and the extra training cost.

## Evaluate on gold

The hand-written gold sets (`tests/gold.jsonl`, 227 sentences / 199 spans,
romanised; `tests/gold_deva.jsonl`, 186 sentences / 155 spans, Devanagari
Hindi; `tests/gold_mr.jsonl`, 173 sentences / 135 spans, Devanagari
Marathi; `tests/gold_gu.jsonl`, 162 sentences / 118 spans, Gujarati) are
the only numbers to trust for real-world quality — synthetic validation
accuracy is optimistic because it's drawn from the same generator/templates
the model was trained on. `--gold` accepts multiple files; per-file and
combined metrics are printed.

```bash
python -m sankhya.eval_gold --gold tests/gold.jsonl tests/gold_deva.jsonl tests/gold_mr.jsonl tests/gold_gu.jsonl --ckpt models/sankhya.pt
```

or against the exported JSON weights directly (what the JS runtime actually
runs):

```bash
python -m sankhya.eval_gold --gold tests/gold.jsonl tests/gold_deva.jsonl tests/gold_mr.jsonl tests/gold_gu.jsonl \
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

Lines longer than `train.MAX_LEN` (128 characters) are handled with the same
sliding window the JS runtime's `parse()` uses (`charset.make_windows`,
mirroring `src/charset.ts::makeWindows`: 128-character windows overlapping
by 16, each character taken from the first window that covers it).
`eval_gold` used to predict over `normalize_text(text)[:MAX_LEN]` while
returning the original text, so any longer line made `decode_spans` index
past the end of the prediction and raise `IndexError` — never triggered by
the short hand-written gold, but 11–27% of real sentences hit it (see
`data_wild/REPORT.md` failure #10). Span offsets index into
`normalize_text(text)`, which is 1:1 per character for every script these
packs support, so gold offsets recorded against the raw text carry over
unchanged — `tests/test_eval_long_lines.py` pins both facts and covers a
200-character gold line.

### What counts as a span

The gold sets and the decoder now agree on one convention (the wild gold
forced the question; see `data_wild/REPORT.md` §9):

- an **amount expression** contains a scale unit (word or symbol), or a
  prefix word together with a number, or digits next to a currency marker;
- a **lone prefix** is not a span (`आधा`, `ढाई साल`, `દોઢ સદી`) — R11;
- a **bare cardinal word** is a span only when its surface is an exact
  lexicon form and not an `ambiguous_forms` entry: `पचास` (50) and `બાવીસ`
  (22) are spans, `sath`, `so` and a lone `एक` are not — R13b and R12;
- an **indefinite plural** is not a span (`हजारों`, `करोड़ों`, `લાખો`) — R9;
- currency words stay **outside** the span; pack case endings stay
  **inside** it (`32 करोडचा`, `15 લાખનું`).

The full rule list, with the real sentence behind each one, is in the root
`README.md`; the rules themselves live in `core.py` (structural) and
`verify.py` (lexicon-driven), are applied by
`eval_gold._apply_bare_digits_gate`, and are mirrored 1:1 in `src/core.ts` /
`src/verify.ts` / `src/index.ts`.

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
python -m sankhya.eval_gold --gold tests/gold.jsonl tests/gold_deva.jsonl tests/gold_mr.jsonl tests/gold_gu.jsonl \
  --weights-json ../src/data/default-weights.json --int8
```

| gold set          | examples | spans | precision | recall | F1     | value_acc |
|--------------------|---------:|------:|----------:|-------:|-------:|----------:|
| gold.jsonl          |      227 |   199 |    0.9550 | 0.9598 | 0.9574 |    0.9598 |
| gold_deva.jsonl      |      186 |   155 |    0.9868 | 0.9677 | 0.9772 |    0.9677 |
| gold_mr.jsonl        |      173 |   135 |    0.9852 | 0.9852 | 0.9852 |    0.9852 |
| gold_gu.jsonl        |      162 |   118 |    0.9658 | 0.9576 | 0.9617 |    0.9322 |
| combined             |      748 |   607 |    0.9719 | 0.9671 | 0.9694 |    0.9621 |
| strict (verified only) |    748 |   506 covered (0.9094) | — | — | — | 1.0000 |

Negatives: 170 examples, 0 false positives. Miss summary: missed 6,
spurious 0, wrong value 3, wrong boundary 14. Per-category value_acc:
digits 0.972, words 0.959, prefix 0.981, range 0.959, currency 0.966,
multi_unit 0.964, symbol_unit 1.0, mixed_script 1.0, long 0.875.

For comparison, the previous shipped weights (0.5.0, three languages, 586
examples) scored 0.9548 (romanised) / 0.9613 (Devanagari) / 0.9778
(Marathi) / 0.9632 (combined) value_acc, strict coverage 0.873 at 1.000
value accuracy. See `models/default/matrix.md` for the seed comparison
behind this release's `v2:48` seed 2 weights (trained locally on CPU,
200k synthetic + the four verified LLM corpora at 20%).

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

**As of 0.4.0 (Hindi only):** a 3-seed `v2:48` sweep run on the enriched
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
the winner is picked by the tie-break rule below: seed 2 had the lowest
negatives false-positive rate (0.0%) of the tied/near-tied set and a
competitive F1, so **seed 2 was the 0.4.0 shipped weights**. The 0.5.0
three-language retrain (`v2:48`, seed 0 vs. seed 2) and the 0.6.0
four-language retrain (`v2:48`, seed 0 vs. seed 2) are separate sweeps —
see `models/default/matrix.md`.

**The rule this implies, and what `eval_matrix.select_matrix_winner` /
the Kaggle MATRIX kernel actually do:** pick the winning **config**
(arch:channels) by highest **mean** int8 combined gold value_acc across
its seeds (seed-to-seed spread is +/-1-2 points, so never compare configs
by a single seed); within the winning config, pick the **seed** by val
value_acc within a 0.005 tie band, then higher int8 combined gold F1,
then lower negatives false-positive rate.

### Handling a wrong-parse report

Reports filed with the "Wrong parse" GitHub issue template (or the
demo's "Report" links, which prefill that template) land as an issue
labelled `wrong-parse` with the input text, the language pack, what the
parser returned, and what the reporter says the correct value/range/span
should be. To turn one into a gold-set fix:

1. **Reproduce it against the shipped weights** with
   `sankhya.triage_issue`:

   ```bash
   cd python
   python -m sankhya.triage_issue --text "<text from the report>" \
     --lang hi_latn --expect-value 125000
   # or: --expect-range LOW HIGH   for a range
   # or: --negative                for a false-positive report (no amount)
   ```

   It prints what the shipped `models/default/sankhya.weights.int8.json`
   currently returns for that text. If the output already matches what the
   reporter says is correct, the report does not reproduce against the
   shipped weights (stale report, already-fixed, or a misunderstanding) --
   reply on the issue and close it rather than touching the gold set.

2. **If it reproduces**, the tool prints the exact JSONL line to append to
   the matching `tests/gold_<lang>.jsonl` (or `tests/gold.jsonl` for
   `hi_latn`), with span offsets computed from the text. Append it, then
   check it against the surrounding examples in that file for style
   (`false` positives use `"spans": []`).

3. **Run the gates** to make sure the new gold line doesn't reveal a wider
   regression and passes with the current weights (it likely won't yet --
   that's expected for a genuine bug):

   ```bash
   python -m pytest tests/test_gates.py -q
   ```

4. **Retrain if needed.** If the gold line is now failing (the usual case
   for a real bug), fix the language pack, decoding rule, or training data
   that caused it (see "Generate data" / "Adding a language pack" above),
   regenerate synthetic data if the fix touches the generator, and retrain
   (see "Train" and "Export" above) so the new gold example passes. Re-run
   `python -m sankhya.eval_gold` and `pytest tests/test_gates.py -q` before
   shipping the new weights.

## Verified spans and strict metrics

The CNN only proposes spans; the value comes from the deterministic core.
A span is **verified** when every one of its `(class, text)` tokens is
independently justified by the exported lexicon
(`src/data/lexicon.json`, regenerated with `python -m sankhya.export_lexicon`):

```python
from sankhya.verify import verify_tokens
verify_tokens([("PFX_SAVA", "sava"), ("SEP", " "), ("UNIT_LAKH", "lakh")])   # True
verify_tokens([("PFX_SAVA", "sava"), ("SEP", " "), ("UNIT_LAKH", "lakhhz")])  # False
```

Rules (mirrored 1:1 by `src/verify.ts`): `SEP` must be whitespace, `DOT`/`COMMA`
exact, `RANGE` is `-`/`–`/`—`/`/` or a pack range word, `DIGITS` is ASCII or
Devanagari digits, `O` must be a lexicon `O` form, and every `PFX_*`/`CARD_*`/
`UNIT_*` token must map to exactly that class in the union of all packs -- or,
when its exact surface is unknown, map to that class after ONE declared case
ending is stripped (the packs' `word_suffixes` / `word_oblique_endings`, also
exported into the lexicon JSON: Marathi "लाखांचं" verifies as `UNIT_LAKH`,
Gujarati "કરોડનો" as `UNIT_CRORE`). A surface that is itself a full lexicon
form is never stripped, so "છનું" stays `CARD_96` and never becomes
`CARD_6`. An empty span, or one with no content token, is unverified. The union maps are
cached at module level; if `src/data/lexicon.json` is missing, `load_lexicon()`
falls back to `export_lexicon.build()`. `tests/test_verify.py` also asserts the
committed JSON equals a fresh `build()`.

Strict mode = "the right number or nothing": return only verified spans.
`eval_gold.py` therefore prints (and writes under `"strict"` in `--json-out`,
alongside every pre-existing key) a strict block: **coverage** (gold spans that
got a verified prediction), **value accuracy** among those, **spurious**
verified spans, and **false positives on negatives**. Shipped int8 weights,
`tests/gold.jsonl tests/gold_deva.jsonl`:

```
coverage:       317/351 = 0.9031
value accuracy: 317/317 = 1.0000
spurious:       1
negatives FP:   0/75 = 0.0000
```

`tests/test_gates.py` are the release gates on those shipped int8 weights:
zero false positives on gold negatives, strict value accuracy == 1.0, and
strict coverage >= 0.85. It runs one int8 numpy pass (~2 s).

`make_fixtures.py` writes `verified` and `tokens` per span into
`test/fixtures/decoded.jsonl`, so the JS parity test pins the same predicate.

## Property test

`proptest.py` runs the same pipeline over freshly generated examples instead of
the 748 hand-written gold ones - the generator's labels are the ground truth,
the torch checkpoint runs batched, and decoding uses exactly `eval_gold`'s gates:

```bash
python -m sankhya.proptest --n 50000 --seed 11 --ckpt ../models/default/sankhya.pt \
    --lang hi_latn,hi_deva --mix 0.55,0.45 --cross 0.10
```

50k examples (52,495 positive spans, 5,447 negatives), ~30 s:

```
exact-value rate (all spans): 0.9703
strict coverage:              35717/52495 = 0.6804
strict value accuracy:        35695/35717 = 0.9994
strict spurious spans:        0
negatives fp_rate=0.0000 | strict fp_rate=0.0000
```

Strict coverage is much lower than on gold because the generator deliberately
injects heavy misspelling noise ("lack", "hazzari", "lkah"): those surfaces are
not in the lexicon, so the span is correctly left unverified. The 22 verified
wrong values are all **juxtaposition ranges** with no connector word
("teen hazaar paanch hazaar" = 3000-5000): the model emits no `RANGE` token, so
the core sums the two terms. Every token is genuinely lexicon-justified, so the
verifier cannot catch it - it is a decoder/model gap, not a lexicon gap.
The report also lists the top 20 misses grouped by gold class pattern.

## Calibrate span confidence (`calibrate.py`)

The decoder's raw span confidence (mean over the span's characters of
max(p_B, p_I)) orders spans well but is squashed — 99% of predicted spans
score above 0.9 — so a caller thresholding it gets no predictable
precision. `calibrate.py` fixes that by measuring, then fitting:

```bash
python -m sankhya.calibrate --n 100000 --seed 13 --ckpt ../models/default/sankhya.pt \
    --mix 0.32,0.26,0.21,0.21 --cross 0.10 \
    --gold tests/gold.jsonl tests/gold_deva.jsonl tests/gold_mr.jsonl tests/gold_gu.jsonl
```

It generates fresh examples (a seed training never saw), runs the torch
checkpoint batched like `proptest.py`, decodes with exactly `eval_gold`'s
gates, and records for every PREDICTED span its raw confidence, whether it
is `verified`, whether its value is right, and whether it is spurious. It
then prints a reliability table (10 raw-confidence bins x count x
empirical precision, separately for verified and unverified spans, for
synthetic and for gold), fits a monotone isotonic
(pool-adjacent-violators, implemented in `calibration.py` in plain numpy —
no sklearn) map from raw confidence to empirical precision on the
**synthetic** spans only, and writes it to `../src/data/calibration.json`:

```json
{"version": 1, "raw": [...], "calibrated": [...], "n_samples": 104805, "seed": 13}
```

At most 32 breakpoints, for piecewise-linear interpolation. Gold is the
held-out check and is never fitted on. `python -m sankhya.eval_gold`
prints (and writes under `"confidence_curve"` in `--json-out`) what each
`minConfidence` threshold buys on gold; both runtimes load the same JSON
and report the calibrated number as `confidence`, keeping the raw score as
`rawConfidence`. Re-run this whenever the shipped weights change — the map
is specific to a checkpoint. Useful flags: `--no-write` (print the tables
only), `--json-out` (dump the measured tables), `--max-points`.

The maths lives in `calibration.py` (`pav`, `fit_isotonic`,
`fit_breakpoints`, `apply`), unit-tested in `tests/test_calibration.py`
and mirrored 1:1 by `src/calibration.ts`. Regenerating the committed JSON
takes a couple of minutes, so the test suite pins the shipped file's
schema and monotonicity rather than re-fitting it.

## Ship to the npm package

The JS package embeds the int8 weights directly, plus a fixed set of
fixtures that pin the JS forward pass and decoder to this exact model.

1. Copy the quantized weights into the package:

   ```bash
   cp models/sankhya.weights.int8.json ../src/data/default-weights.json
   ```

2. Re-fit the confidence calibration for the NEW weights (the map is
   specific to a checkpoint — see "Calibrate span confidence" above), so
   the fixtures below are generated against the map the runtimes will use:

   ```bash
   python -m sankhya.calibrate --n 100000 --seed 13 --ckpt models/sankhya.pt
   ```

3. Regenerate the two parity fixtures the JS test suite checks against:

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
     currency, rawConfidence, confidence, classes, verified, tokens}` —
     `rawConfidence` is the decoder's uncalibrated score and `confidence`
     is that score through `src/data/calibration.json`, the two numbers
     the JS side compares within 1e-6. The JS `test/e2e-parity.test.ts` runs `parse()`
     on the same texts and checks it produces the same spans — this pins
     end-to-end behaviour (decode + arithmetic core), not just the raw
     model output.

   `make_fixtures.py` uses `np_infer.load_weights_int8_json` +
   `np_infer.forward` (int8-dequantized, matching what the JS runtime
   actually loads), `np_infer.pad_ids`/`softmax`, `decode.decode_spans`,
   and `core.evaluate`/`core.detect_currency` — i.e. the same reference
   pipeline `eval_gold.py` uses, factored out so it can write fixture files
   instead of computing a score.

4. Verify from the repo root:

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

## Real-text measurement (`sankhya.wild`)

Everything above measures the model on data we made: the generator, or gold
lines we wrote. `sankhya.wild` measures it on **sentences other people wrote**
— openly licensed corpora (Dakshina romanised/native-script Wikipedia, CMU DoG
Hinglish), filtered to 3–40 words and ≤128 chars, scored for amount-likelihood
with the pack lexicon (never the model), and sampled seeded/deduplicated:

```bash
python -m sankhya.wild sources   # registry + URLs + licences (fetch by hand)
python -m sankhya.wild sample    # -> data_wild/samples/, seed 17
python -m sankhya.wild run       # shipped int8 weights + eval_gold's gates
```

Raw corpora and samples live under `python/data_wild/` and are gitignored (the
sources are share-alike). 100 lines per language were hand-labelled into
`tests/gold_wild_<lang>.jsonl` — same schema as the other gold files, with
attribution in `tests/GOLD_WILD_SOURCES.md`.

`tests/test_gold_wild.py` checks that they load, that `core.evaluate`
reproduces every labelled value, and prints the per-language real-text table
(value accuracy, strict coverage, FP, strict FP) under `pytest -s`. Accuracy
floors on real text stay out of the gates — that is where the model is
weakest and a floor would either be trivially loose or block unrelated work
— but **precision is gated**: `tests/test_gates.py` pins the strict
(verified) false-positive rate on the wild negatives at ≤ 0.02 (currently
4/240 = 0.0167), the overall wild FP rate at ≤ 0.05 (10/240 = 0.0417), and
strict value accuracy on real text at 1.0. Those are the numbers the R11–R18
gates were added for; before them they were 0.0875 and 0.200.

The measured numbers, the failure taxonomy and what real text contains that the
generator never produces are in **`python/data_wild/REPORT.md`**.

## Tests

```bash
python tests/test_core.py
python tests/test_decode.py
python tests/test_generator.py
python tests/test_llm_corpus.py
python -m pytest tests/test_verify.py tests/test_gates.py
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
negatives, duplicates) against inline fixtures. `test_verify.py` covers the
strict-tier predicate and the freshness of `src/data/lexicon.json`;
`test_gates.py` is the release gate on the shipped int8 weights (see
"Verified spans and strict metrics") — including the real-text
false-positive ceilings. `tests/test_wild_rules.py` covers the R11–R18
precision rules case by case (mirrored by `test/wild-rules.test.ts` on the
JS side) and `tests/test_eval_long_lines.py` the >128-character windowing.
`test_pretrain_negatives.py` covers the data plumbing of the two model-side
experiments: `--extra-negatives` lines become all-`O` examples and are mixed
at the requested ratio, `sankhya.pretrain`'s masking hits 15% of real
characters with the loss ignoring every unmasked position, and `--init-from`
loads the trunk while leaving the heads fresh. `tests/gold.jsonl` is the
hand-written gold set used by `eval_gold.py`, not a generator round-trip
test.

## Colab

See `colab.md` for a copy-pasteable notebook recipe (same commands as
above, runs fine on Colab's CPU runtime).

## Training on Kaggle (GPU)

`python/kaggle_train/` wraps a Kaggle "script" kernel that runs the full
multi-pack recipe above (`--lang hi_latn,hi_deva,mr_deva,gu_gujr --mix
0.32,0.26,0.21,0.21 --cross 0.10`, 200k train / 6k val, 20 epochs, batch 128,
lr 3e-3) on a Kaggle
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
- `dataset` — `python/tests/gold.jsonl` / `gold_deva.jsonl` / `gold_mr.jsonl` /
  `gold_gu.jsonl`, to
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
  produce (collisions with real words/names). Also honoured at runtime
  (R14): a blocked surface never verifies as a number word and a decoded
  span carrying one is dropped, so this is where proper nouns built on
  number words go (`हजारे`, `सवाई`, `अरबी`)
- `ambiguous_forms` — lexicon **surfaces** that are also ordinary words in
  the language (`so`, `sath`, `mil`, `arab`, `अरब`, `એક`). They stay real
  lexicon forms, but R12 drops a span whose every lexicon-justified token
  is one of them; see "What counts as a span" above
- `noise_fn(word, rng) -> str` — typo/spelling-variance injection for that
  script/language

plus, if the language needs them, the morphology hooks
`glue_unit_forms` / `fused_prefix_forms` / `word_suffixes` +
`word_oblique_endings` (see the Marathi notes in the checklist below).

then `sankhya.langs.base.register(pack)` it and give it a unique `id`
(add that id to `sankhya/langs/base.py`'s `KNOWN_PACKS` so `--lang <id>`,
`base.all_packs()` and every tool built on them resolve it). Regenerate the charset for the new pack
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
(A second Devanagari language can reuse `noise_deva.py`, as `mr_deva`
does.)

### Checklist: what adding `mr_deva` (Marathi) actually took

In order, with the commands used:

1. **Write the pack** — `sankhya/langs/mr_deva.py`, mirroring
   `hi_deva.py`: cardinals 1-99 with the spelling variants people
   actually type (तेहेतीस/तेहतीस, शहात्तर/सहात्तर, ऐंशी/अंशी,
   चौऱ्याऐंशी/चौर्याऐंशी), units (शंभर/शे, हजार, लाख, कोटी + करोड, अब्ज),
   prefixes (सव्वा, दीड, अडीच, साडे, पावणे, अर्धा, पाव), currency markers,
   `" ते "`/`" किंवा "` range connectors, `" आणि "` conjunctions,
   approximators, duration nouns, filler words, ~25 templates per register
   and ~50 negatives, `deva_digit_prob = 0.25`.
2. **Register it** in `base.KNOWN_PACKS`.
3. **Model the language's morphology through the generic hooks**, never
   with per-language code in the generator/tokenizer:
   - `glue_unit_forms = ["शे"]` — a BOUND unit form that only ever appears
     glued to the number before it (दोनशे = `CARD_2 UNIT_SAU`). The
     generator glues it (and swaps in शंभर when the shape doesn't allow
     gluing); `llm_corpus._word_tokens` splits such a word back into two
     tokens; `decode._is_bound_unit_tail` keeps the 2-char unit tail from
     being smoothed into the cardinal at decode time.
   - `fused_prefix_forms = {"PFX_SAADHE": 0.75, "PFX_PAUNE": 0.45}` —
     prefixes usually typed as one word with their cardinal (साडेतीन), keyed
     by class so noised spellings fuse too. Both spellings parse.
   - `word_suffixes` + `word_oblique_endings` — case endings that attach
     to a unit/cardinal (हजारात, लाखांचा, कोटींची, तीसच). The whole word
     keeps the head word's class.
4. **Check cross-pack collisions.** A surface shared by two packs must map
   to the same class (the charset/training data unions every pack).
   `tests/test_mr.py::test_no_cross_pack_conflicts` enforces this; adding
   Marathi turned up exactly one — Hindi's O-labelled filler "तेरा" is
   CARD_13 in Marathi — resolved by dropping it from `hi_deva`'s fillers.
5. **Verify the LLM corpus**:

   ```bash
   python -m sankhya.llm_corpus verify --lang mr_deva \
     --in data_llm/raw/mr_deva.sonnet.jsonl --out data_llm/mr_deva.jsonl \
     --rejects data_llm/rejects/mr_deva.jsonl --manifest data_llm/manifest.json
   ```

   Iterate on the lexicon until the rejects are only genuine LLM
   arithmetic errors or truly unusual phrasing (Marathi landed at 591/600
   accepted: 4 wrong values, 5 "negatives" that really do contain a unit
   word). Rejects that reveal real spellings belong in the lexicon.
6. **Write a gold set by hand** — `tests/gold_mr.jsonl`, same schema as
   `gold_deva.jsonl`, ~150+ examples, ~30% negatives, covering every
   prefix, the fused hundreds, ranges, currency, Latin-mixed lines and
   digits. Check every positive value with `core.evaluate` on the tokens
   your lexicon implies, and keep it disjoint from the corpus.
7. **Mirror the currency markers on the JS side** — `src/lang-<id>.ts`,
   merged into `CURRENCY_PACK` in `src/index.ts`.
8. **Tests** — `tests/test_mr.py` (lexicon sanity, cross-pack collisions,
   generator round-trips at n=2000, fused-hundreds label boundaries,
   corpus verification, gold re-evaluation) and a JS currency-pack test.
9. **Retrain** with the new pack in `--lang`/`--mix` and regenerate the
   charset; only then do the bundled weights understand the language.
   `test/fixtures/*.jsonl` and `models/default` stay untouched until that
   retrain happens.

### What adding `gu_gujr` (Gujarati) added on top

Gujarati followed the same nine steps (583/600 corpus accepted, a
162-example gold set, `tests/test_gu.py`). Four things were new, and all
four were done as *generic* hooks so the next pack inherits them:

- **A non-Devanagari script.** Gujarati needed its own `noise_gujr.py`
  (anusvara drop, ળ/સ and ડ/ઢ confusions, matra swaps, geminate
  simplification) — `noise_deva.py`'s transforms are Devanagari code
  points. It also needed its digits (U+0AE6-U+0AEF) added to the
  normalization map in `charset.py` / `src/charset.ts` and to the DIGITS
  predicate in `verify.py` / `src/verify.ts`; `core.py` already had them.
  The charset itself needs nothing: `build_charset()` derives the vocab
  from the pack's own forms, templates and fillers, so registering the
  pack is what puts the Gujarati block in.
- **Per-pack native digits.** `deva_digit_prob` became
  `native_digits` (a ten-glyph table) + `native_digit_prob`, so the
  generator renders DIGITS tokens in whatever script a pack declares.
  hi_deva/mr_deva keep identical behaviour.
- **A glue unit that is also a free word.** Marathi's `शे` is purely
  bound; Gujarati's `સો` glues (બારસો) *and* stands alone
  (`સો રૂપિયા` = 100). `glue_unit_forms_standalone` names that
  subset; `pack.bound_units()` is what the tokenizer and the negative
  scanner now consult.
- **Bound number forms.** Gujarati's 200 is `બસો`, never *`બેસો`: in
  the glued position CARD_2 is `બ`, which is not a word at all on its
  own (600 is likewise `છસો`/`છસ્સો`). `bound_number_forms` maps
  such a surface to its class *and the unit it must precede*. These
  forms are kept out of `all_forms()`, so they never enter the
  cross-pack collision map; `export_lexicon` writes them to a separate
  `bound_forms` section, and both verifiers accept one only when the
  next token is that unit — a standalone `("CARD_2", "બ")` never
  verifies. Modelling it this way (rather than as extra UNIT_SAU
  surfaces) is what keeps બ and છસ્ from leaking into any other
  position.

  One decode-time caveat is deliberately left open:
  `decode._is_bound_unit_tail` needs >= 2 characters of CARD evidence
  before a unit tail, and `બ` is one character, so `બસો` may smooth to
  a single UNIT_SAU run (100, not 200) once weights exist. Loosening
  that bound also changes how the three shipped languages smooth stray
  one-char mispredictions, so it belongs with the Gujarati retrain, not
  before it.
