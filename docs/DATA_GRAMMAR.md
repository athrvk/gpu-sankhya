# gpu-sankhya — synthetic data grammar (proposal, v0)

This document defines the closed vocabulary, the arithmetic, the noise model and
the sentence templates used to generate training data. Nothing here is model code.

Design rule that shapes everything below: **the model never emits strings the
deterministic core has to fuzzy-match.** The CNN emits, per character, a span
tag (BIO) and a *token class* (e.g. `PFX_SAVA`, `UNIT_LAKH`, `CARD_23`,
`DIGITS`). The deterministic core does arithmetic over the class sequence only.
That is what makes "spelling generalisation" the model's job and keeps the
core a pure table.

---

## 1. Closed vocabulary

### 1.1 Fractional prefixes (class `PFX_*`)

| class        | canonical | standalone value | with following number N | curated spellings (seed list, noise is applied on top)                    |
|--------------|-----------|------------------|-------------------------|---------------------------------------------------------------------------|
| `PFX_PAAV`   | paav      | 0.25             | —                       | paav, pav, pau, paw, pauv                                                 |
| `PFX_AADHA`  | aadha     | 0.5              | —                       | aadha, adha, aadhaa, adhaa, aadh, aadhe, adhe, aadhi, adhi                |
| `PFX_PAUNE`  | paune     | 0.75 × unit      | N − 0.25 (N ≥ 2)        | paune, pone, pauna, paun, pona, pawne, powne, paunay                      |
| `PFX_SAVA`   | sava      | 1.25             | N + 0.25 (N ≥ 2)        | sava, sawa, savaa, sawaa, swa, savva, sava'                               |
| `PFX_DEDH`   | dedh      | 1.5              | — (never takes N)       | dedh, derh, ded, dhed, dhedh, dedhh, dedhi, dhai? **no** (see note)       |
| `PFX_DHAI`   | dhai      | 2.5              | — (never takes N)       | dhai, dhaai, dhaee, dhayi, dhay, adhai, adhaai, arhai, arhaai, dhaii      |
| `PFX_SAADHE` | saadhe    | —                | N + 0.5 (N ≥ 3)         | saadhe, sadhe, sade, saade, sarhe, sadhey, saadhey, sarhe                 |

Notes
- `saadhe ek` / `saadhe do` are ungrammatical (Hindi uses dedh / dhai). The
  generator never emits them; the core still evaluates them leniently.
- `paune` and `sava` with no number and a unit: `paune lakh` = 75 000,
  `sava sau` = 125. Missing N is treated as 1.
- `dhai` and `dedh` never take a following number: `dedh do lakh` is invalid.
- Prefixes compose with `sau` as a unit: `dedh sau` = 150, `saadhe teen sau` = 350.

### 1.2 Cardinals (class `CARD_1` … `CARD_99`)

Full irregular Hindi table 1–99 romanised (ek, do, teen, chaar, paanch, chhe,
saat, aath, nau, das, gyarah, baarah, terah, chaudah, pandrah, solah, satrah,
atharah, unnees, bees, ikkis, bais, teis, chaubis, pachchis, chhabbis, sattais,
atthais, untis, tees, … , navve, ninyanve). Stored as a 99-row table.

`hi_latn`'s `_EXTRA_VARIANTS` (`sankhya/langs/hi_latn.py`) curates 2–4 real
Hinglish-chat spellings for **every** cardinal 11–99, not just a handful of
frequent ones, so common non-canonical spellings are recognised directly
instead of depending on the ~4%-rate N7 char-noise to happen to produce them
(the original gap: "unnasi" for 79 parsed as nothing because only the
canonical "unyasi" was in the lexicon). E.g. 79: `unyasi, unasi, unnasi,
unnyasi, unnaasi`; 81: `ikyasi, ikkyasi, ikyaasi, ekyasi`; 91: `ikyanve,
ikkyanve, ikyaanve, ekyanve`; 21: `ikkis, ikkees, ekkis`; and so on through
99. All variants are checked (see `test_cardinal_variants_map_correctly`)
against cross-cardinal collisions and against every other class's forms
(prefixes, units, symbols, English-fraction phrases) before being added, and
use only plain lowercase a-z (so the charset — already the full ASCII
lowercase alphabet — gains no new characters from them). 1–10 and the
frequent round tens keep their earlier curated sets too
(`chhe/che/chhah/chah`, `paanch/panch/paach`, `pachas/pachaas`,
`bees/bis/biss`, `tees/tis`, etc.).

English cardinals `one` … `twenty`, `thirty` … `ninety`, and `hundred` are also
in the table (code-mixed text: `two lakh`, `ten hazaar`, `one and a half crore`).

Sampling weight: 1–10 and round tens get 70 % of the cardinal mass; 11–99 the
rest. Cardinals are also emitted as digits (`3 lakh`, `sadhe 3 lakh`) 50 % of
the time.

### 1.3 Scale units (class `UNIT_*`)

| class          | value | word forms                                                   | symbol / abbreviation forms (attach directly to digits)  |
|----------------|-------|--------------------------------------------------------------|----------------------------------------------------------|
| `UNIT_SAU`     | 1e2   | sau, so, sao, sou, hundred                                    | —                                                        |
| `UNIT_HAZAAR`  | 1e3   | hazaar, hazar, hajaar, hajar, hazzar, thousand, thou          | k, K, hzr                                                |
| `UNIT_LAKH`    | 1e5   | lakh, lac, lack, laakh, lakhs, lacs, lakh's, peti (slang)     | L, l, lk, lkh, LPA / lpa (lakh per annum)                |
| `UNIT_MILLION` | 1e6   | million, mil                                                  | M, mn                                                    |
| `UNIT_CRORE`   | 1e7   | crore, karod, karor, karore, crores, karodh, khokha (slang)   | cr, Cr, CR, crs                                          |
| `UNIT_BILLION` | 1e9   | billion                                                       | B, bn                                                    |
| `UNIT_ARAB`    | 1e9   | arab, arab                                                    | —                                                        |
| `UNIT_KHARAB`  | 1e11  | kharab, kharab                                                | —                                                        |

Sampling weights: lakh 35 %, hazaar/k 25 %, crore 20 %, sau 10 %, million/billion 6 %, arab/kharab/slang 4 %.

Symbol forms are only generated after a digit form (`2.5L`, `20k`, `1.5cr`,
`12 LPA`); word forms after either digits or words.

### 1.4 Indefinite plurals — hard negatives, **no span**

`lakhon`, `lakho`, `laakhon`, `hazaaron`, `hazaron`, `karodon`, `crodon`,
`croron`, `sainkdon` ("lakhs of people"). These carry no value and are labelled
`O`. They are the most important negative class because they share 5+ characters
with the positive unit words.

### 1.5 Digit forms (class `DIGITS`, `DOT`, `COMMA`)

- plain integer: `20000`, `125000`
- Indian grouping: `1,25,000`, `2,50,00,000` (and occasionally wrong/western grouping `125,000`)
- decimal: `2.5`, `1.25`, `12.5`, `.5` (rare)
- decimal + word unit: `1.25 lakh`, `2.5 crore`
- decimal + symbol unit, with/without space: `2.5L`, `2.5 L`, `20k`, `1.5 cr`, `12LPA`
- prefix + digit: `sadhe 3 lakh`, `sawa 2 crore`, `paune 2 lakh`

### 1.5b Curated typo spellings (`UNIT_*`, hi_latn only)

Beyond the char-level N7 typo noise (§4), a handful of very common misspellings
are curated directly into the `hi_latn` lexicon so they are *guaranteed* to
appear inside a quantity context (preceded by a number), not just at N7's 4%
rate: `UNIT_CRORE` gains `croer`, `crorr`; `UNIT_LAKH` gains `lkah`, `lakhh`;
`UNIT_HAZAAR` already covered `hajaar`, `hazzar`. These are ordinary lexicon
entries, so `_safe_word` may pick them directly and N-noise may still stack on
top of them.

### 1.5c Possessive/plural apostrophe noise on units (hard negative for the
span *boundary*, not the unit)

`lakh's`, `lakhs'`, `crore's`, `2lakh's`: an apostrophe+letters suffix glued
directly onto a unit word. The unit itself keeps its class/value; the
apostrophe run is `O`. Fires ~2% of the time a phrase ends on a unit token
(`generator._maybe_possessive`), on top of whichever surface form/casing/N-noise
was already chosen for that unit.

### 1.5d Dakshina-mined romanization variants (`CARD_*`/`UNIT_*`, hi_latn only)

`python/scripts/mine_dakshina.py` mines additional attested romanizations of
Hindi cardinal/unit/prefix words out of the Google Dakshina v1.0 Hindi
romanization lexicon (`hi.translit.sampled.{train,dev,test}.tsv`,
https://github.com/google-research-datasets/dakshina, **CC BY-SA 4.0** —
this derived spellings list is a redistribution of that dataset under the
same licence; attribute Google Research / the Dakshina dataset when
redistributing `_EXTRA_VARIANTS`/`LEXICON` entries sourced from it).

For every Devanagari surface form used by `hi_deva`'s cardinal (1–99), unit
and prefix tables (plus common inflections — plural unit suffixes and
with/without-nukta spellings of ज़/ज, ड़/ड, ढ़/ढ), the script looks up every
attested Latin romanization + its count across the three TSVs, drops
anything already in `hi_latn`'s lexicon, and writes
`python/scripts/dakshina_variants.json` (`{class: {devanagari: [[roman,
count], ...]}}`).

Merge rule applied by hand from that report: a variant is added to
`_EXTRA_VARIANTS[n]` (cardinals) or a unit/prefix's form list if its count is
≥2, or ≥1 when the Devanagari word has fewer than 5 total attestations in
Dakshina — then it must clear a collision filter: not already mapped to a
different class/number via `pack.all_forms()`, not a filler word / blocked
surface / duration noun, longer than 2 characters, and not a common
Hindi/English word with another meaning (hand list: `sat`, `char`, `ath`,
`sath`, `saath`, `kharab`, `kharaab`, `das`, `so`, `me`, `to`, `petty`).
Devanagari plural forms of a scale unit (करोड़ों, हज़ारों, अरबों → `krodo`,
`karoron`, `krodon`, `hajaron`, `arabon`, …) are indefinite-plural usages
("thousands of people"), not the "N thousand" multiplier, so they were added
to `INDEFINITE_PLURALS` (label `O`), matching the existing `lakhon`/
`hazaron`/`karodon` treatment, not to the unit's form list.

The user-reported colloquial spellings of 66 (`chanchat`, `chanchatt`,
`chaachat`, `chhachat`, `chhasath`, `chansath`, `chhiyasat`, `chiyasat`) are
not attested in Dakshina and were added to `_EXTRA_VARIANTS[66]` directly.

Re-run: `python python/scripts/mine_dakshina.py --lexicon-dir <dir containing
hi.translit.sampled.{train,dev,test}.tsv>` (defaults to a scratchpad path;
pass `--lexicon-dir` explicitly elsewhere). It only reports candidates — the
merge into `hi_latn.py`'s lexicon tables and `test_generator.py`'s
`test_dakshina_merged_unit_variants_map_correctly` is manual, so the
collision/exclusion decisions above stay auditable.

### 1.6 Connectors inside a span (class `SEP`)

- additive chain glue: whitespace only (`do lakh pachas hazaar`), occasionally `aur`
- English fraction glue: `and a half`, `and half`, `n half`, `and a quarter` → mapped to `PFX_SAADHE` / `PFX_SAVA` semantics
- range glue (class `RANGE`): `-`, `–`, `to`, `se`, `ya`, `/`, `or`, plain juxtaposition → `2-3 lakh`, `2 se 3 lakh`, `do teen lakh`, `20-25k`, `1.5 to 2 crore`, `dedh do lakh`
- **multi-span conjunctions** (`O`, class `conj_connectors` on the pack — NOT
  `RANGE`): join TWO (sometimes three) *independent* quantity phrases, each
  its own span with its own `B` tag — hi_latn: `aur`, `ya`, `ya phir`, `, `,
  `nahi to`; hi_deva: `और`, `या`, `या फिर`, `, `, `नहीं तो`. Distinguish
  carefully from a range: `"2-3 lakh"` / `"dedh do lakh"` is ONE span (an
  amount range); `"sava lakh aur dedh lakh"` / `"50k ya 60k"` is TWO spans (two
  separate amounts) — `"se lekar ... tak"` specifically is always a range, never
  built as a conjunction. Generator: `build_conjunction_phrase` /family
  `multi_conj` (~6% of examples), plus a couple of literal `two_spans`
  templates for the "X mein Y flat, har flat Z" unrelated-amounts style.

### 1.7 Context words — **outside the span**, labelled `O`

- currency markers: `₹`, `Rs`, `Rs.`, `rs`, `INR`, `rupaye`, `rupay`, `rupees`, `rupiya`, `rupya`, `/-`, `Re`
  (the core records `currency: "INR"` if one is adjacent to the span; the span itself excludes it)
- approximators: `karib`, `kareeb`, `lagbhag`, `takreeban`, `around`, `approx`, `about`, `roughly`, `~`, `koi`, `kuch`
- quantity nouns after the phrase: `rupaye`, `log`, `logon`, `views`, `units`, `pieces`, `sq ft`, `km`
- rate suffixes: `per month`, `/month`, `per annum`, `p.a.`, `mahine ka`, `saal ka`

---

## 2. Span grammar (what is extracted)

```
Span     := Amount (RANGE Amount)?       -- range: both amounts, unit may be shared (`2-3 lakh`)
Amount   := Term (SEP? Term)*            -- descending units, max 3 terms
                                          -- (the LAST term may be a bare
                                          -- cardinal with no unit of its own:
                                          -- "ek hazaar ek" = 1001, "do lakh
                                          -- paanch" = 200005 -- generator
                                          -- structure `chain_trailing_bare`)
Term     := Prefix? Number? Unit?        -- at least one of Prefix/Unit present, or Number+currency
Number   := CARD_n | DIGITS(.DIGITS)?
Prefix   := PFX_*
Unit     := UNIT_* (word or symbol)
```

Extraction policy (this decides what counts as a positive vs a negative):

| text                          | extracted? | why                                             |
|-------------------------------|------------|-------------------------------------------------|
| `sava lakh`                   | yes        | prefix + unit                                   |
| `20k logon ne`                | yes        | digits + unit                                   |
| `paanch sau rupaye`           | yes        | cardinal + unit                                 |
| `pachas rupaye`               | yes        | bare cardinal **with** adjacent currency marker |
| `do din baad`                 | no         | bare cardinal, no unit, no currency             |
| `9876543210`, `pin 400001`    | no         | bare digits, no unit, no currency               |
| `15 August`, `2.30 baje`      | no         | bare digits                                     |
| `lakhon log aaye`             | no         | indefinite plural                               |
| `Lakhan`, `crorepati`, `sauda`| no         | substrings of unit words inside other words     |
| `Rs 500`                      | yes        | bare digits **with** currency marker            |

Bare numbers without unit or currency are out of scope on purpose. They are
trivially regexable, highly ambiguous (dates, phones, OTPs) and would dominate
the label distribution.

## 3. Evaluation semantics (deterministic core)

Given a class sequence for one span:

1. Split into terms at unit boundaries (each `UNIT_*` closes a term; a trailing term without a unit gets unit = 1).
2. For each term compute `coef`:
   - Number only → N
   - Prefix only → standalone value from §1.1 (`paune` → 0.75, i.e. N = 1)
   - Prefix + Number → `sava`: N + 0.25, `saadhe`: N + 0.5, `paune`: N − 0.25
   - neither → 1 (bare `lakh`, `crore`)
   - English `and a half` / `and a quarter` behave as `saadhe` / `sava`
3. `term = coef × unit`; `value = Σ terms` for units in DESCENDING order
   (`ek crore bees lakh` = 1e7 + 20×1e5). Hindi/Indian-English also stacks a
   *larger* unit on top of everything accumulated so far instead of adding it
   ("das hazaar crore" = 10 000 crore, "2 lakh crore" = 2 lakh crore, "sau
   crore" = 100 crore): while walking terms within one amount, if the unit
   that closes the current term is LARGER than the unit that closed the
   previous term, the running total accumulated so far is MULTIPLIED by the
   new unit's value instead of a new additive term being appended, and the
   reported `unit` becomes that larger unit. Descending units keep adding as
   before; only a run of `sau`/`hazaar`/`lakh` may precede a larger unit
   (`crore` or `lakh`), and the generator caps this at one multiplicative
   step per amount. Worked: `das hazaar crore` = (10×1e3)×1e7 = 1e11;
   `sau crore` = (1×1e2)×1e7 = 1e9; `dedh sau crore` = (1.5×1e2)×1e7 = 1.5e9;
   `2 lakh crore` = (2×1e5)×1e7 = 2e12; `saadhe teen sau crore` =
   (3.5×1e2)×1e7 = 3.5e9; `ek lakh bees hazaar crore` = (1e5 + 20×1e3)×1e7 =
   120 000×1e7 = 1.2e12.
3b. Ranges: split at `RANGE`; if the left amount has no unit it inherits the
   right amount's unit (`2-3 lakh` → 200 000–300 000, `dedh do lakh` → 150 000–200 000).
   Output `value` = low end, `range = [low, high]`. Juxtaposed cardinals
   (`do teen`, `paanch das`) are a range only when both are cardinals and
   ascending; the generator only emits ascending pairs.
4. Validation used only by the generator: units strictly descending, `saadhe` needs N ≥ 3,
   `dedh`/`dhai` take no N, at most one prefix per term.

Worked examples: `paune do lakh` → (2 − 0.25) × 1e5 = 175 000; `ek crore bees lakh` → 1e7 + 20 × 1e5 = 12 000 000;
`dhai sau` → 250; `2.5L` → 250 000; `12 LPA` → 1 200 000; `sadhe 3 lakh` → 350 000; `1,25,000` → 125 000.

## 4. Noise model

Applied only to romanised Hindi word tokens (never to digits, symbols or English
words). Each transformation fires independently with the listed probability;
the generator checks the result is still not equal to another vocabulary word.

| id  | transformation                                                     | p    | example                     |
|-----|--------------------------------------------------------------------|------|-----------------------------|
| N1  | pick a curated variant instead of canonical (§1)                    | 0.55 | dedh → derh                 |
| N2  | vowel length swap: a↔aa, i↔ee↔ii, u↔oo, e↔ai, o↔au                  | 0.15 | lakh → laakh, paune → pone  |
| N3  | aspiration toggle: dh↔d, bh↔b, kh↔k, th↔t, chh↔ch                   | 0.08 | dhai → dai, hazaar → hazar  |
| N4  | consonant alternation: z↔j, w↔v, f↔ph, sh↔s, ṛ-cluster rh↔dh↔d     | 0.10 | hazaar → hajaar, sava → sawa|
| N5  | final vowel drop / add (-a, -e, -i)                                 | 0.08 | saadhe → sadh, paun → pauna |
| N6  | letter doubling / de-doubling                                      | 0.05 | chhe → che, ikkis → ikis    |
| N7  | keyboard typo: delete / insert / transpose / adjacent-key (len ≥ 4) | 0.04 | crore → croer               |
| N8  | casing: lower 70 %, Title 15 %, UPPER 5 %, random 10 %              | 1.0  | Lakh, LAKH, lAkh            |
| N9  | joining / hyphenation of prefix+unit or number+unit                | 0.06 | dedhlakh, 2.5-lakh, dedh-lakh|
| N10 | spacing noise around symbols: `2.5 L`, `2.5L`, `Rs.2.5L`, `Rs 2,50,000/-` | 0.3 | —                      |
| N11 | chat elongation: repeat last letter 2–4×                            | 0.02 | lakhhh                      |
| N12 | Hinglish plural: lakh → lakhs, crore → crores (value unchanged)     | 0.15 | 5 lakhs                     |

Everything after N1 is stacked at most twice per token so words stay recognisable.
Devanagari script input is out of scope for v1 (would double the char vocab).

## 5. Sentence templates

Each example = one template with 0, 1 or 2 `{P}` slots, plus optional `{C}`
(currency marker) and `{A}` (approximator) slots that resolve to a word or to
the empty string.

| family                  | share | examples                                                                                           |
|-------------------------|-------|----------------------------------------------------------------------------------------------------|
| casual chat / WhatsApp  | 20 %  | `bhai {A} {P} mein ho jayega kya`, `usne {C}{P} maange yaar`, `{P} toh bahut zyada hai`, `mera budget {P} tak ka hai` |
| classifieds / e-commerce| 20 %  | `Price: {C}{P} negotiable`, `2019 model, {P} final`, `rent {P}/month, 2bhk`, `MRP {C}{P} only`     |
| news / social feed      | 15 %  | `{P} logon ne attend kiya`, `startup ne {P} ka funding raise kiya`, `{P} se zyada views aa gaye`   |
| salary / finance        | 10 %  | `CTC {P} per annum`, `package {P}`, `EMI {C}{P} monthly`, `loan {P} ka liya tha`                   |
| ranges                  | 8 %   | `{P1}-{P2} lakh mein aa jayega`, `{P1} se {P2} crore ki deal`, `do teen hazaar lagenge`, `20-25k ka phone` |
| two spans               | 10 %  | `{P1} se badhkar {P2} ho gaya`, `pehle {P1} tha ab {P2}`, `{P1} + {P2} = ?`, `{P1} mein {P2} flat, har flat alag price` |
| multi-span conjunction  | ~6 %  | `sava lakh aur dedh lakh`, `50k ya 60k`, `teen lakh nahi to paanch lakh` (2, sometimes 3, independent spans — see §1.6) |
| bare phrase             | 5 %   | `{P}`                                                                                              |
| negatives (no span)     | 12 %  | `Lakhan bhai ka number 9876543210 hai`, `KBC crorepati`, `lakhon log aaye`, `do din mein aa jaunga`, `pin 400001`, `2.30 baje milte hain`, `so what`, `chaar dost`, `lakhpati bann gaya`, `karodpati keh diya`, `laakhon mein bik gaya`, `10 km door hai`, `5 kg aata`, `100 kmph se`, `1st prize`, `2nd floor`, `route 66`, `mobile 98765 43210`, `flat no 302`, `sava ghanta lag gaya`, `dedh ghante mein`, `saadhe teen baje`, `paune paanch baje` (prefix-word duration/time phrases, deliberately confusable with quantity prefixes but O throughout) |

Templates carry random casing, trailing emoji / punctuation (`!!`, `..`, `😂`),
and 10 % of the time a random-length prefix/suffix of unrelated chat text so the
span is not always near the sentence boundary. Sentence length is capped at 128
characters (model context window).

## 6. Labels

Per example the generator writes:

```json
{"text": "bhai sawa 2 lakh mein ho jayega kya",
 "spans": [{"start": 5, "end": 16, "value": 225000, "currency": null,
            "classes": "PFX_SAVA DIGITS UNIT_LAKH"}]}
```

Per-character targets derived from it:
- `bio`: 0 = O, 1 = B, 2 = I (B on the first char of each span)
- `cls`: one of ~126 classes (7 prefixes, 99 cardinals, 8 units, DIGITS, DOT,
  COMMA, SEP, RANGE, O). Whitespace inside a span is `SEP`. Both heads share the CNN trunk.

## 6b. Inference output shape

```ts
interface Sankhya {
  span: string;            // exact source substring
  start: number; end: number;
  value: number;           // resolved value; low end for ranges
  range?: [number, number];// present only for ranges
  unit: "sau"|"hazaar"|"lakh"|"crore"|"million"|"billion"|"arab"|"kharab"|null;
                           // largest scale unit in the phrase, canonical spelling;
                           // null for `Rs 500` style bare amounts
  currency: "INR"|null;    // adjacent marker detected by the core, outside the span
  confidence: number;      // mean of span-tag softmax probs over the span
  classes: string[];       // normalised token classes, e.g. ["PFX_DHAI","UNIT_LAKH"]
}
```

`unit` and `classes` come for free from the class head, so `2.5L` and
`dhai lakh` both return `unit: "lakh"`, `value: 250000`. Callers who want
"2.5 lakh" back as display text can rebuild it from `value` and `unit`.

`value` is not a model target. It is recomputed by the core from the class
sequence, and the generator asserts round-trip equality (`core(classes) == value`)
so the arithmetic table and the generator can never drift apart.

## 7. Sizes and splits

- train 80 000, val 5 000 (same generator, different seed)
- a hand-written gold set of ~200 real-style sentences, never generated, used
  as the only number we report. Synthetic val is optimistic by construction.
- Char vocab: `a-z 0-9 space . , - / + ( ) % : ' " ! ? @ # ₹ ~` + `<pad>` + `<unk>`,
  input lower-cased (casing carries no information here). ≈ 45 symbols.

## 8. Known ambiguities, decided up front

- `2.5L` is always lakh (it is just `dhai lakh` written with a symbol); `5k` is
  always thousand. No litre/km negatives are generated, so the model never
  learns to suppress these.
- English `so` vs Hindi `sau`: only a unit when preceded by a number word/prefix; the
  negatives include free-standing `so`.
- `L` in `LPA`, `M` in `5M views`: treated as units (lakh, million).

## 9. Open questions before writing the generator

1. Should `sau`-only phrases (`paanch sau`, no currency) count? Proposal: yes.
2. Should bare `Rs 500` be in scope? Proposal: yes (digits + currency).
3. ~~Ranges in v1 or v2?~~ Decided: v1.
