# Changelog

All notable changes to `gpu-sankhya`, summarized release by release. See the
[git history](https://github.com/athrvk/gpu-sankhya/commits/master) for full
detail.

## Unreleased

- Demo site SEO and social sharing: an og.png preview image, favicon.svg
  and apple-touch-icon.png, robots.txt/sitemap.xml, JSON-LD
  (SoftwareApplication + FAQPage) structured data, and Open Graph/Twitter
  card meta tags; `scripts/site.mjs` now copies every file in `demo/`.
- The demo site (`demo/index.html`) is redesigned as a single centred column:
  a "Try it out" card that highlights the typed phrase in place (pastel
  colours per token class — fraction word, number, unit, currency, range),
  a clean Result panel with value, unit caption, confidence and the
  `verified` badge, plus a tappable grid of example phrases in Hinglish,
  Hindi, Marathi and Gujarati.
- "Inside the Model" now explains the current phrase directly: the
  per-character BIO/class strip from `inspect()`, the decoded token chips,
  and the arithmetic line, captioned with the live `modelInfo()` parameter
  count.
- Everything the old demo could do is still there — shareable URL hash,
  local parse timing, wrong-parse reporting — with strict mode, the
  min-confidence slider, raw JSON, WebGPU status and the batch benchmark
  moved into a collapsed "Developer options" disclosure. Light by default,
  with `prefers-color-scheme: dark` support.

## 0.7.0 — Calibrated confidence

- Confidence scores are now calibrated: an isotonic (PAV) map is fit on
  ~100k fresh synthetic spans plus the four gold sets and exported as
  `src/data/calibration.json`, so a reported confidence better reflects the
  actual probability of being correct.
- Each parsed span now carries both the calibrated `confidence` and the raw
  model `rawConfidence`, and `ParseOptions.minConfidence` lets callers gate
  spans below a threshold.
- Verification now strips declared case endings (Python + TS), raising
  strict coverage to 552/607 = 0.909 at 1.000 value accuracy.
- `eval_gold` reports a confidence-reliability curve; the demo gained a
  min-confidence slider explaining what calibration means.

## 0.6.0 — Gujarati (gu_gujr)

- Fourth language pack: Gujarati (`gu_gujr`), with its own lexicon, noise
  module, and a verified LLM-written corpus (583/600 accepted).
- Decoder: incomplete-amount span merging, a bound-form/fused-word
  partition (up to prefix+cardinal+unit with 1-char bound heads), and a
  lexicon-"O" gate (R9) to stop ordinary words like "karodon" being tagged
  as a unit; R10 rejects leading-zero digit runs (vehicle plates, PINs) in
  both Python and TS.
- Gold set grows to 748 examples across four languages (162 new Gujarati
  examples); the four-language shipped model (v2, seed 2, CPU) is
  documented with per-language metrics, matrix provenance, and an updated
  model card.
- Gujarati digits (U+0AE6–U+0AEF) accepted by the charset and verifiers;
  the fixture generator applies the shared post-decode gates.

## 0.5.0 — Marathi (mr_deva)

- Third language pack: Marathi (`mr_deva`), sharing the Devanagari charset
  with Hindi but with its own lexicon, bound unit forms, fused prefixes,
  and case-ending morphology hooks generalized onto `LanguagePack`.
  Verified LLM-written corpus: 591/600 accepted.
- R8: juxtaposed spelled-cardinal ranges decoded in both Python and TS.
- Fixed Devanagari/Gujarati digit normalization that was evaluating some
  spans to 0, and preserved fused prefix+cardinal sub-runs (e.g.
  "पावणेचार") through decoding — including 8 pre-existing Hindi gold
  values baked with the digit bug.
- Gold set grows to 586 examples (three languages); tools and tests are
  now registry-driven instead of hard-coding the hi_latn/hi_deva pair.
- `hf_push` gained the numpy dependency needed for `gold_sets`, fixing a
  0.5.0 Hugging Face dataset publish failure.

## 0.4.0 — Verified tier / strict mode

- New "verified" tier: every token in a span is independently checked
  against the language pack's lexicon before that span counts as trusted;
  `strict` mode returns only verified spans. Strict coverage on the gold
  set reaches 0.904 at 1.000 value accuracy with 0 false positives.
- R7: word-connector ranges (e.g. "kharab"/"mil", "se"-ranges) merge across
  a `RANGE`-tagged connector in both the Python and TS decoders.
- Lexicon gaps closed: Devanagari connectors, a lone kharab/mil gate, and
  no-space word forms; a `SANKHYA_JOIN_WORDS_P` env toggle controls the
  no-space glue noise in the generator for ablations.
- Property test and CI gates added: JS and Python suites plus the release
  gates now run on every PR and master push. Gold set grows to 413
  examples.

## 0.3.0 — CRF head, LLM-written corpora, edge-case decoding

*(0.3.0 through 0.3.4 shipped as a fast sequence of retrains on the same
decoder/data work; grouped here as one release line.)*

- Optional CRF layer over the BIO tags with Viterbi decoding (Python
  train/export/np_infer and a parity-verified JS runtime), vectorised and
  batched ~44x faster than the naive implementation — kept as a tested,
  opt-in `--crf` flag; the shipped weights don't use it (see the README's
  CRF section for why).
- Large LLM-written corpora added: hi_latn (1,187/1,200 verified) and
  hi_deva (1,049/1,111 verified), plus raw mr_deva/gu_gujr drafts for
  later language packs; a `llm_corpus` pipeline (verify/stats/sample) and
  `train --extra` mixing support them.
- Decoder/core edge-case rules: word integrity, a bare-digit size gate,
  connector repair, possessive-noise trimming, descending-range-as-chain,
  and a widened currency look-ahead (fixes "1200 rupees"). Non-string
  input is now tolerated.
- Dakshina-mined romanised spellings (CC BY-SA 4.0) plus colloquial forms
  folded into the lexicon.
- Hugging Face publishing added: a workflow that pushes the model, a
  Space, and a dataset, with generated model cards.
- Across the 0.3.x retrains, shipped int8 gold accuracy moved from 0.926
  Hinglish / 0.974 Devanagari / 0.947 combined up to 0.954 / 0.980 / 0.965
  combined (F1 0.968), 0 false positives, on a gold set that grew from ~73
  negatives to 410 examples.
