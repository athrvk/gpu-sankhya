# gpu-sankhya on real text

Every accuracy number this repo has published so far comes from
`sankhya.generator` or from gold lines we wrote ourselves. This report is the
first measurement on **sentences written by other people for other purposes**:
openly licensed corpora, filtered and sampled by `sankhya.wild`, run through
the shipped int8 weights (`models/default/sankhya.weights.int8.json`) with the
same gates `eval_gold` applies, plus a 400-line hand-labelled real-text gold
set (`python/tests/gold_wild_<lang>.jsonl`).

Reproduce:

```bash
python -m sankhya.wild sources     # registry, URLs, licences
python -m sankhya.wild sample      # seed 17 -> data_wild/samples/
python -m sankhya.wild run --dump-preds data_wild/wild_preds.jsonl
python -m sankhya.eval_gold --gold tests/gold_wild_*.jsonl \
    --weights-json ../models/default/sankhya.weights.int8.json --int8 --quiet
```

## 1. Sources

| id | lang | what | licence | raw text committable? |
| --- | --- | --- | --- | --- |
| `dakshina_hi_roman` | hi_latn | Google **Dakshina v1.0**, `hi/romanized/hi.romanized.rejoined.tsv` - 10k Hindi Wikipedia sentences romanised by native speakers | CC BY-SA 4.0 | yes, with attribution + share-alike |
| `dakshina_hi_wiki` | hi_deva | Dakshina v1.0 `hi/native_script_wikipedia` (filtered, shuffled Hindi Wikipedia sentences) | CC BY-SA 4.0 (underlying text CC BY-SA 3.0) | yes, as above |
| `dakshina_mr_wiki` | mr_deva | Dakshina v1.0 `mr/native_script_wikipedia` | CC BY-SA 4.0 | yes |
| `dakshina_gu_wiki` | gu_gujr | Dakshina v1.0 `gu/native_script_wikipedia` | CC BY-SA 4.0 | yes |
| `cmu_hinglish_dog` | hi_latn | **CMU DoG Hinglish** (`festvox/cmu_hinglish_dog`) - 8k human-typed romanised-Hindi chat turns | CC BY-SA 3.0 / GFDL | yes |

Dakshina tarball: <https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar>
(2.0 GB; `sankhya.wild sources` prints the member paths it needs).
CMU DoG Hinglish parquet: HF `festvox/cmu_hinglish_dog`.

Considered and rejected: **Samanantar** (CC BY-NC 4.0 - non-commercial, so its
text cannot be redistributed in an MIT repo), **AI4Bharat IndicCorp v2** (no
licence declared on the HF card), **Sangraha** (CC BY-4.0 but 375 MB/shard,
not worth the download at this sample size), **LinCE** (registration-gated).

Because both accepted licences are share-alike, the bulk of what we downloaded
stays out of git: `python/data_wild/raw/`, `python/data_wild/samples/` and the
prediction dumps are gitignored. Only the 400 hand-labelled lines are
committed, with attribution in `python/tests/GOLD_WILD_SOURCES.md`.

### Source stats (seed 17)

Lines are kept at 3-40 words **and** <=128 characters - `train.MAX_LEN` is 128,
so a longer line is silently truncated before the model ever sees it
(see failure #10).

| source | lines in 3-40 word band | of those, >128 chars | unique kept | amount-signal candidates | no-signal candidates |
| --- | ---: | ---: | ---: | ---: | ---: |
| dakshina_hi_roman | 8,719 | 2,318 (27%) | 6,375 | 104 | 5,034 |
| cmu_hinglish_dog | 7,026 | 586 (8%) | 6,151 | 170 | 5,258 |
| dakshina_hi_wiki | 929,032 | 186,436 (20%) | 716,069 | 8,119 | 578,357 |
| dakshina_mr_wiki | 324,393 | 35,600 (11%) | 279,033 | 2,849 | 241,401 |
| dakshina_gu_wiki | 133,875 | 30,506 (23%) | 101,974 | 977 | 42,560 |

Amount-signal rate in real text is **0.9-1.6%** of sentences (hi_latn: 2.2% of
the romanised set, mostly from the chat corpus). Sampled per language:
300 candidate positives (hi_latn: 274 - the whole pool) + 150 negatives.

## 2. Behaviour on the wild samples (no gold needed)

`python -m sankhya.wild run`, 1,774 lines total.

| lang | lines | any span (all) | any span (signal lines) | any span (no-signal lines) | spans | strict (verified) share of spans | strict span on a no-signal line | mean calibrated conf |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hi_latn | 424 | 27.1% | 39.4% | 4.7% | 127 | 66.9% | 0.0% | 0.918 |
| hi_deva | 450 | 47.6% | 71.3% | 0.0% | 244 | 84.0% | 0.0% | 0.947 |
| mr_deva | 450 | 47.3% | 68.3% | 5.3% | 229 | 80.4% | 0.0% | 0.945 |
| gu_gujr | 450 | 45.6% | 67.0% | 2.7% | 237 | 90.3% | 0.7% | 0.954 |

Calibrated-confidence distribution over all 837 predicted spans: **0** below
0.7, 136 in 0.7-0.9, 701 at >=0.9. On real text the calibration is badly
over-confident - the confidence curve on the hand-labelled subset (below)
shows 68% precision at `minConfidence` 0 and only 82% at 0.9, yet nothing the
model emits ever scores below 0.7, so `minConfidence` buys far less than the
synthetic reliability table implies.

Most common token patterns (combined): `DIGITS+SEP+UNIT_LAKH` (77),
`DIGITS+SEP+UNIT_CRORE` (76), `DIGITS+SEP+UNIT_MILLION` (54),
`DIGITS+SEP+UNIT_HAZAAR` (42), bare `UNIT_SAU` (30), bare `UNIT_HAZAAR` (32),
`DIGITS+DOT+DIGITS+SEP+UNIT_*` (40), `UNIT_ARAB` (23).
Real text is overwhelmingly **digits + unit word**; the prefix grammar
(`sava`/`dedh`/`saadhe`) that most of the synthetic set exercises is rare
(8 of 168 gold spans), and symbol units (`20k`, `2.5L`, `12 LPA`) - a headline
feature - appear **zero** times in 1,774 real lines.

## 3. Hand-labelled real-text gold (100 lines per language)

Each file is 100 lines drawn from the samples: 50 the model predicted a span
on, 25 lexicon-positive lines it stayed silent on, 25 no-signal negatives.
Every positive value was checked with `core.evaluate` on the tokens the pack
lexicon implies (`sankhya.wild.tokens_from_lexicon`); 5 spans the lexicon
cannot tokenize on its own (`pacchis karod`, `bahar hajar`, `2.5 लाख से 20 लाख`,
`6થી 8 કરોડ`, `પાઁચ હજાર`) were checked by hand.

**Labelling convention used** (and see failure #9): a span is an *amount
expression* - it contains a scale unit (word or symbol), or a prefix word
together with a number, or digits adjacent to a currency marker. A bare
cardinal or bare digit run with no unit and no currency is **not** a span
(`109 वोट`, `1956`, `teen movie`); nor is a lone prefix with no number and no
unit (`आधा`, `ढाई साल`, `દોઢ સદી`); nor an indefinite plural (`हजारों`,
`करोड़ों`, `લાખો`). Currency words stay outside the span; pack case endings
stay inside it (`32 करोडचा`, `4806.312 કરોડનો`, `15 લાખનું`).

### Wild gold vs. synthetic gold, shipped int8 weights

| lang | set | examples | gold spans | value acc | strict coverage | strict value acc | negatives | FP rate | strict FP rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hi_latn | synthetic | 227 | 199 | 0.960 | 0.889 | 1.000 | 37 | 0.000 | 0.000 |
| hi_latn | **wild** | 100 | 21 | **0.857** | **0.762** | **1.000** | 79 | **0.367** | **0.177** |
| hi_deva | synthetic | 186 | 155 | 0.968 | 0.923 | 1.000 | 38 | 0.000 | 0.000 |
| hi_deva | **wild** | 100 | 44 | **0.932** | **0.909** | **1.000** | 57 | **0.140** | **0.070** |
| mr_deva | synthetic | 173 | 135 | 0.985 | 0.933 | 1.000 | 45 | 0.000 | 0.000 |
| mr_deva | **wild** | 100 | 49 | **0.837** | **0.837** | **1.000** | 52 | **0.135** | **0.019** |
| gu_gujr | synthetic | 162 | 118 | 0.932 | 0.898 | 1.000 | 50 | 0.000 | 0.000 |
| gu_gujr | **wild** | 100 | 54 | **0.944** | **0.907** | **1.000** | 52 | **0.077** | **0.038** |
| **combined** | synthetic | 748 | 607 | 0.962 | 0.909 | 1.000 | 170 | 0.000 | 0.000 |
| **combined** | **wild** | 400 | 168 | **0.899** | **0.869** | **1.000** | 240 | **0.200** | **0.088** |

Combined wild miss breakdown: 9 missed spans, 7 wrong boundaries, 1 wrong
value, **63 spurious spans** (span precision 0.685 vs. 0.972 synthetic).

**The headline:** recall and value accuracy hold up surprisingly well
(0.90 value accuracy on real spans vs. 0.96 synthetic), and the strict tier's
central promise still holds - **every one of the 146 verified spans on real
text has the right value (1.000, same as synthetic)**. What collapses is
*precision*: one in five real negatives produces a span, and strict mode only
halves that (8.8% of real negatives still produce a *verified* span, against
0% on hand-written negatives). Strict mode on real text means "the right
number, or a confident number for something that was never an amount".

Per-category value accuracy on wild gold: digits 0.961 (102 spans), words
0.803 (66), prefix 0.500 (8), range 1.000 (3), currency 1.000 (10),
multi_unit 0.750 (4).

## 4. Top 10 failure shapes on real text

Tagged: **lexicon gap** (a surface the packs get wrong) / **decoder gap**
(spans, merging, gates) / **model gap** (tagger) / **convention** (what gold
should even say). "Triage" = whether the `eval_gold` wrong-parse workflow
turns it straight into a gold line.

1. **Proper nouns that are unit words** - *lexicon gap*, triage: yes.
   `अण्णा हजारे` / `हजारे 43` -> `UNIT_HAZAAR` 1000 (hi_deva #40/#45, mr_deva #19);
   `सवाई तुकोजीराव` -> `PFX_SAVA` (mr_deva #15); `शे स्वीट` (mr_deva #72).
   Person and place names built on number words are everywhere in real text
   and nowhere in the generator. Wants a blocked-surface list (the packs
   already have the mechanism: `blocked_surfaces` currently holds 1-2 city
   names each) or a "unit word immediately after a given name" gate.
2. **Geographic/ethnonym `arab` / `अरब` / `અરબ` / `अरबी`** - *lexicon gap*,
   triage: yes. `sanyukt arab amirat`, `अरब संघ`, `सऊदी अरब`, `अरबी समुद्र`,
   `અરબ સાગર` all parse as 1,000,000,000. This single surface is the largest
   source of verified false positives in hi_deva/mr_deva/gu_gujr.
   `अरबी`/`અરબી` (the adjective) is not itself a unit form - the model produces
   it unverified, so strict mode already drops those; the bare `अरब`/`અરબ` is
   the one that gets through.
3. **Hinglish clitics and English words colliding with the lexicon** -
   *lexicon gap*, triage: partly.
   `k` (= "ke/ki") is the `UNIT_HAZAAR` symbol; `so` (= English "so") is
   `UNIT_SAU`; `sath` (= "with") is `CARD_60`; `sade`/`sad` (= "rotten",
   "sad") is `PFX_SAADHE`; `peti` (a real slang term for a lakh, here "corn
   belt") is `UNIT_LAKH`; `mil` (= "mile"/"meet") is `UNIT_MILLION`;
   plus `adha`/`adhik`, `teen`, `hours`, `chori`, `thought`, `favreau`, `oooh`.
   This is why hi_latn's FP rate on real negatives is 37% - by far the worst
   of the four packs.
4. **Bare digits given a unit that is not there** - *model gap*, triage: yes.
   `3 hours` -> 3,000 (`DIGITS+UNIT_HAZAAR`); `1405 mil` -> 1.405e9;
   `33 k. m.` -> 3.3e10; `25वे` (an ordinal) -> 2,500,000; `1980ના` -> 1.98e8;
   `55 से 70l` (litres) -> a 5.5-7.0 lakh range. The tagger invents a unit
   class for letters adjacent to digits, and the R3 bare-digits gate cannot
   help because the span is no longer digits-only.
5. **A unit word used for a measurement, not an amount** - *convention*
   (+ lexicon), triage: needs a decision first.
   `gyarah mil dur` (11 miles), `डेढ़ मील`, `એક થી દોઢ મીટર`, `20 કરોડ કિમી`,
   `85 મિલિયન ટન`. We label the ones with a real scale unit as spans
   (`20 કરોડ` = 2e8 is a correct number) but drop `mil`/`मील` = mile; the
   packs list `mil` under `ambiguous_units`, yet R4 only fires for a *lone*
   ambiguous unit, so `gyarah mil` / `1405 mil` sail through.
6. **Indefinite plurals and "several X"** - *decoder/convention*, triage: yes.
   `हजारों-हजार`, `करोड़ों रूपये`, `લાખો રૂપિયા`, `कित्येक हजार`, `કેટલાક સો`.
   R9 drops a token whose surface is lexicon-"O", which catches `करोड़ों`, but
   `हजारों-हजार` still yields a `UNIT_HAZAAR` span, and `कित्येक हजार वर्षे`
   ("several thousand years") is labelled 1,000 here - defensible but
   arbitrary.
7. **Boundary loss on prefix + cardinal** - *model gap*, triage: yes.
   `साढ़े तीन साल` -> the model spans only `साढ़े` (0.5); `sadhe terah`,
   `साडे तीन`, `સાડા ત્રણ`, `साढ़े सात` are all missed or truncated. Prefix
   value accuracy on wild gold is **0.500 on 8 spans**, the worst category,
   even though prefixes are the feature the synthetic set drills hardest.
8. **Span merging across a boundary the writer did not intend** - *decoder
   gap*, triage: yes. `दस हज़ार 1987 में बनी` (film title + year) -> one span
   worth 11,987; `42 કરોડ એકત્ર` -> 4.3e8 (a following word retagged
   `UNIT_CRORE`); `5 panch hajar` -> 5,005; `तब्बल तीन लाख चाळीस हजार` ->
   a spurious range because `तब्बल` was tagged `CARD_15`;
   `226.70 करोड़ का करोड़` -> two spans out of one typo'd amount.
9. **What counts as a span at all** - *convention*, triage: no, needs a call.
   Real text is full of bare cardinals and lone prefixes: `पचास लोग`,
   `अडीच महिने` (2.5 months), `आधा दिन`, `एक या डेढ मिनट`, `दीड तास`.
   The existing gold files disagree with the convention used here -
   `gold_deva.jsonl` has a bare `पचास` = 50 span, while this report labels
   bare cardinals as non-spans. Also unresolved: an elided unit
   (`125 અને 250 અબજ` - is `125` a span worth 1.25e11?), digits with a
   currency word (`30 रुपयांमध्ये` - labelled a span, value 30) and grouped
   digits with no unit (`67,000 किलोग्राम` - labelled *not* a span, which the
   shipped model contradicts).
10. **Real sentences are longer than the model's window** - *decoder/harness
    gap*, triage: no, it is a plumbing bug.
    11-27% of real 3-40-word sentences exceed `train.MAX_LEN` = 128
    characters. `eval_gold.run_json_weights` predicts over
    `normalize_text(text)[:MAX_LEN]` but returns the **original** text, so
    `decode_spans` indexes past the end of the prediction and raises
    `IndexError` on any longer line (never triggered by the short
    hand-written gold). `sankhya.wild` works around it by normalising and
    truncating before predicting, and by sampling only <=128-char lines - but
    that also means everything in this report is measured on the short end of
    real text.

## 5. What real text has that the generator never produces

- **Amounts are rare and unevenly distributed.** ~1% of real sentences mention
  one; the generator's world is mostly amounts. Precision, not recall, is what
  real deployment is bought on, and the synthetic set barely tests it (170
  hand-written negatives, all of which the model gets right).
- **Number words inside names, places and titles** - `हजारे`, `सवाई`,
  `अरब अमीरात`, `हज़ार राहें` (a song), `इनाम दस हज़ार` (a film),
  `द बिलियन डॉलर प्रिन्सेस`. The generator only ever puts a unit word in a
  quantity slot.
- **Measurement and scientific registers** - `4.1-8.2 अरब सालों`,
  `53.3 लाख सालों`, `32 મિલિયન ઘનમિટર`, `85 મિલિયન ટન`, `620 હજાર` people,
  `10 અબજ બિટ્સ પ્રતિ સેકન્ડ`. Units of *things*, not rupees; `million` /
  `billion` / `અબજ` are far more common than the money-register `लाख/करोड़` in
  encyclopedic text, and the generator's currency-heavy templates never show
  the model a plain count in the billions.
- **Years, ordinals, scores, percentages and digit runs** sitting next to
  amount words in the same sentence (`2011 साली 1.05 लाख`,
  `84% sade tamatar 6.9/10`, `7vin kaksa`, `25वे`, `राष्ट्रीय महामार्ग 51`).
  The generator's negatives are much cleaner than this.
- **Romanisation the generator's noise model does not reach** -
  `hajar/hajaar/hazar/hazaar`, `karod/crore/croer`, `bahar` (= बारह),
  `pacchis`, `unnis sau sath`, chandrabindu spellings (`પાઁચ`), plus
  code-switched English clauses in the middle of a Hindi sentence.
- **Spoken-out years and quantities** - `do hajar char` (2004),
  `unnis sau sath` (1960), `do hajar dus` (2010). Arithmetically these are
  numbers, but they are dates; the generator never writes a year in words.
- **Broken and typo'd amounts** - `226.70 करोड़ का करोड़ रूपए`, `12,000 મિલિયન`
  (12 billion written as 12,000 million), `$3 મિલિયને` with an inflection
  glued on, `રુ.15 લાખનું` with the currency marker glued to the digits.
- **Zero symbol units.** `20k`, `2.5L`, `1.5cr`, `12 LPA` - a headline feature
  of the library and a large share of the synthetic data - did not occur once
  in 1,774 sampled real lines. They live in chat and classifieds, which these
  corpora do not cover; a wild sample from that register is the obvious next
  source to add.
