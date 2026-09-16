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
  master via `.github/workflows/pages.yml`). Its batch benchmark compares
  WebGPU vs CPU span-for-span; last run 500/500 match, ~1.5x speedup.
- **Model**: 4-layer dilated char CNN, 32 ch, 18,811 params, vocab 114.
  Gold (int8, what ships): Hinglish value acc 0.944 / span F1 0.939;
  Devanagari 0.965 / 0.965. JS and Python agree on all 346 gold texts
  (`test/fixtures/parity.jsonl`, `decoded.jsonl`).
- **Tests**: `npm test` (63) at root; `python -m pytest tests/` (65) in
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
runtime right-pads input with 16 pad tokens to match training padding.

## Kaggle training (verified on Kaggle 2026-09-16)

`python/kaggle_train/`: `kernel-metadata.json`, `train_kernel.py`
(clones repo at GIT_REF, generates data, trains on GPU, exports, evals
both gold files, writes `output/`), `run.py` (push/status/pull/all).
Smoke-tested locally on CPU only. Needs `KAGGLE_API_TOKEN` in the
session env (new sessions see env vars; a running session does not).
From `python/`:

    python -m kaggle_train.run push --set GIT_REF=<branch with --device flag>
    # GIT_REF=master fails until this branch merges: master's train.py
    # has no --device flag (Kaggle run v1, 2026-09-16, exit 2).
    python -m kaggle_train.run status --timeout 3600
    python -m kaggle_train.run pull      # weights -> src/data, regen fixtures
    cd .. && npm test                    # must pass before committing weights

Gotchas learned: the CLI validates the token by POSTing it in a request
body, so proxy header injection cannot replace it; the previously
pasted token returned 401 and is exposed — use a freshly generated one.

Verified run (kernel v3, this branch, default config = shipped recipe):
clone + pip 10s, data gen 50s, training 172s on the Kaggle GPU (vs ~12
min on CPU), export + gold eval ~25s; ~4.5 min wall total. Result:
int8 gold Hinglish 0.938 value acc / 0.927 F1, Devanagari 0.943 /
0.944, val value acc 0.917 — same recipe as shipped, so this is
run-to-run variance (a hair below the shipped 0.944 / 0.965). Shipped
weights were kept. Two fixes were needed to get there: `--device auto`
exists only on this branch (v1 cloned master and failed), and the bio
loss class weight had to be created on the training device (v2).

## Known misses / small follow-ups

- Hinglish: `croer` (typo), `do lakh's` (apostrophe), long ranges like
  `paanch se sadhe saat lakh`, spurious spans on unfamiliar words.
- Demo text still says "13 examples" for the benchmark set (22 chips now).
- Roadmap: Marathi (साडे, सव्वा) and Gujarati (સવા, દોઢ) packs next
  (cheapest, share prefix semantics); then Bengali, Tamil/Telugu/Kannada.
  WASM SIMD only if sub-millisecond single-string latency is ever needed.

## Process notes

- Sonnet subagents implemented; the orchestrator reviewed and integrated.
- Release: bump version on master (PR or `npm version patch` +
  `git push --follow-tags`), workflow publishes and tags. No GitHub
  releases by choice (v0.1.1 release can be deleted; keep the tag).
