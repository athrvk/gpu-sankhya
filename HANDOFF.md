# Handoff — gpu-sankhya

State as of 2026-09-16. Everything below master is shipped; the only
unmerged work is on branch `claude/busy-tesla-nt8ytr` (this file and
`python/kaggle_train/`).

## What exists and works

- **npm `gpu-sankhya@0.2.0`** (with provenance): Hinglish + Devanagari
  Hindi, mixed-script input, CPU default, WebGPU for batches.
  Auto-publishes when `package.json` version changes on `master`
  (`.github/workflows/publish.yml`, npm trusted publishing, no token).
- **Demo**: https://athrvk.github.io/gpu-sankhya/ (deploys on push to
  master via `.github/workflows/pages.yml`), rewritten for live parse, a
  per-character model-internals view (`inspect()`), batch benchmark
  controls, a JSON view, a theme toggle, and shareable state via URL
  hash. Its batch benchmark compares WebGPU vs CPU span-for-span; the
  residual layers in arch `v2` (now shipped) have not been re-verified
  on a real GPU adapter since the model changed — see Known misses.
- **Model**: arch `v2`, 5 conv layers over 16-dim char embeddings — a
  plain k5 layer, then four residual k3 layers (dilations 1/2/4/8,
  `y = relu(conv(x)) + x`) — 48 channels, 39,579 params, vocab 114, 120
  classes, ±17-char receptive field. int8 bundle: `dist/index.js` 80.8 KB
  raw / 47.2 KB gzipped. An older `v1` preset (4-layer, non-residual, 32
  ch, 18,811 params) is still loadable by every runtime; weights JSON v2
  carries the arch spec so both load unchanged. Gold (int8, what ships):
  romanised value acc 0.9515 / span F1 0.9235; Devanagari 0.9861 /
  0.9795; combined 0.9676 / 0.9494 (353 examples / 309 spans — grew by 7
  foreign-character examples this round). Negatives: 4/55 false
  positives. JS and Python agree on all gold texts (`test/fixtures/
  parity.jsonl`, `decoded.jsonl`). Trained with a new `--unk-noise`
  generator augmentation (out-of-vocab chars as O context) so `<unk>` is
  actually trained, fixing cases like "🙏 sava lakh" returning nothing.
  Latency: `parse()` p50 (60-char input) ~3.6 ms; `parseBatch` CPU ~4.0
  ms/string over 500 strings (both roughly 2x the prior `v1` model's,
  the cost of the larger stack). `PAD_TAIL` is 24 (was 16) on both
  sides. See `python/README.md`'s sweep-evidence section for why `v2`/48
  was chosen over `v1`/32 and other configs (mean gold value_acc over
  >= 3 seeds, not a single seed).
- **Tests**: `npm test` (69) at root; `python -m pytest tests/` (86) in
  `python/`.

## Architecture in one paragraph

Language-independent semantic classes (`python/sankhya/classes.py`):
`PFX_*`, `CARD_1..99`, `UNIT_*`, `DIGITS`, `RANGE`, ... The model tags
each character with BIO + class. A deterministic core (`core.py`,
mirrored in `src/core.ts`) evaluates the class sequence: prefixes,
additive descending units, multiplicative ascending units
(`das hazaar crore` = 1e11), ranges, currency. Each language is a
`LanguagePack` (`python/sankhya/langs/`) with lexicon, noise fn,
templates, fillers, negatives. Adding a language = new pack + noise
module + charset rebuild + retrain; core, model, runtime unchanged.
Text normalization (NFC, Devanagari digits -> ASCII, lowercase) is
identical in `charset.py` and `src/charset.ts`; keep them in sync.

## Decoder rules that matter (both sides)

Strict BIO (span starts only at B), bridge single O gaps, extend digit
runs, class repair (sub-run smoothing >= 3 chars, punctuation rules,
Unicode letters+marks count as letters), confidence gate 0.5, and the
runtime right-pads input with 24 pad tokens to match training padding.

## Kaggle training

`python/kaggle_train/`: `kernel-metadata.json`, `train_kernel.py`
(clones repo at GIT_REF, generates data once, then trains/exports/evals a
MATRIX of `arch:channels:seed` configs on GPU, picks a winner, writes
`output/`), `run.py` (push/status/pull/all). Needs `KAGGLE_API_TOKEN` in
the session env (new sessions see env vars; a running session does not).
From `python/`:

    python -m kaggle_train.run push --set GIT_REF=<branch> \
      --set MATRIX=v2:48:0,v2:48:1,v2:48:2
    python -m kaggle_train.run status --timeout 3600
    python -m kaggle_train.run pull      # weights -> src/data, regen fixtures
    cd .. && npm test                    # must pass before committing weights

Winner selection (`sankhya.eval_matrix.select_matrix_winner`, shared by
the kernel and `eval_matrix.py`): config (arch:channels) by highest mean
int8 combined gold value_acc across its seeds; seed within that config by
val value_acc inside a 0.005 tie band, then int8 combined gold F1, then
negatives FP rate. Every matrix entry (not just the winner) is staged at
`output/entries/<arch>_<channels>_s<seed>/`, so picking a different seed
afterwards doesn't need a full re-run. See `python/README.md`'s "Sweep
evidence" section for the actual numbers behind the `v2`/48-channel
choice — seed-to-seed spread is ~1-2 points, so never trust a single
seed's result.

Gotchas learned: the CLI validates the token by POSTing it in a request
body, so proxy header injection cannot replace it — use a freshly
generated token. `--device auto` and cuDNN-deterministic training
(reproducible seeds) both required fixes in `train.py` that only landed
on this branch; the bio loss class weight has to be created on the
training device, not the host device. Wall time for the current default
recipe (200k/6k examples, 20 epochs) is roughly 3 min/model on a Kaggle
GPU vs. ~12 min on CPU for `v1:32`, ~2x that on CPU for `v2:48`.

## Known misses / small follow-ups

- Hinglish: `croer` (typo), `do lakh's` (apostrophe), long ranges like
  `paanch se sadhe saat lakh`, spurious spans on unfamiliar words (the
  "long" category is the weakest at 0.75 value acc; 13 spurious spans and
  9 wrong-boundary misses combined dominate the miss count on this round's
  gold eval).
- `v2` trades a little span precision for its accuracy/robustness gain:
  4/55 negatives are now false positives (was 2/55 under `v1`).

## Next

- Add languages: Marathi (साडे, सव्वा) and Gujarati (સવા, દોઢ) next
  (cheapest, share prefix semantics); then Bengali, Tamil/Telugu/Kannada.
- Consider reducing spurious spans: more negative examples in training
  data, and/or a precision-aware tweak to matrix winner selection (it
  currently optimizes value_acc first, FP rate only as a late tie-break).
- WebGPU is verified on a real GPU (desktop Chrome) only for the `v1`
  arch's plain conv layers. `v2`'s residual path (`y = relu(conv(x)) +
  x`) in WGSL is unit-tested by structure but has not been checked on an
  actual GPU adapter — run the demo's batch benchmark and confirm 0
  mismatches before trusting it in production.
- WASM SIMD only if sub-millisecond single-string latency is ever needed.

## Process notes

- Sonnet subagents implemented; the orchestrator reviewed and integrated.
- Release: bump version on master (PR or `npm version patch` +
  `git push --follow-tags`), workflow publishes and tags. No GitHub
  releases by choice (v0.1.1 release can be deleted; keep the tag).
