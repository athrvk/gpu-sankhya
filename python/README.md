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

CPU-only torch is fine — training the default model (17,883 params) takes
about 9 minutes on 4 CPU cores.

## Generate data

```bash
python -m sankhya.generator --n 150000 --seed 3 --out data/train.jsonl --lang hi_latn
python -m sankhya.generator --n 5000 --seed 4 --out data/val.jsonl --lang hi_latn
```

Paths are relative to `python/`. `--lang` selects the registered
`LanguagePack` (currently only `hi_latn`, Latin-script/romanised Hindi).

You can also regenerate the charset file directly if you need it standalone:

```bash
python -m sankhya.charset --lang hi_latn --out data/charset.json
```

## Train

```bash
python -m sankhya.train \
  --train data/train.jsonl --val data/val.jsonl \
  --epochs 20 --batch 128 --lr 3e-3 \
  --channels 32 --layers 4 --dilation 2 \
  --out models/
```

This is the exact recipe used for the bundled default weights: 4 conv
layers (kernel sizes 3/5/3/3, dilations 1/1/2/4), 32 channels, 16-dim char
embeddings over the 56-character `hi_latn` vocab — 17,883 parameters total.
Writes `models/sankhya.pt` (a torch checkpoint carrying `vocab`, `classes`,
`dilation`, `channels`, `layers`, `state_dict`).

Other flags: `--label-smoothing`, `--grad-clip`, `--num-train` (subsample
the training set), `--lang` (default matches the data's language pack).

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

The hand-written gold set (`tests/gold.jsonl`, 180 sentences / 161 spans)
is the only number to trust for real-world quality — synthetic validation
accuracy is optimistic because it's drawn from the same generator/templates
the model was trained on.

```bash
python -m sankhya.eval_gold --gold tests/gold.jsonl --ckpt models/sankhya.pt
```

or against the exported JSON weights directly (what the JS runtime actually
runs):

```bash
python -m sankhya.eval_gold --gold tests/gold.jsonl \
  --weights-json models/sankhya.weights.json
# add --int8 with --weights-json models/sankhya.weights.int8.json
# to check the quantized weights specifically
```

Current numbers for the bundled model: span precision 0.90, recall 0.96,
F1 0.93, value accuracy 0.95 (153/161). Known miss categories: unusual
typos ("croer"), possessive apostrophes ("do lakh's"), long multi-term
ranges ("paanch se sadhe saat lakh"), and occasional spurious spans on
unfamiliar words.

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
