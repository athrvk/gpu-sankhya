"""Wild-text sampling and measurement: how does the shipped model behave on
REAL sentences, as opposed to the synthetic generator or hand-written gold?

Every number the repo reports today comes from `sankhya.generator` output or
from gold lines we wrote ourselves. This module pulls sentences out of openly
licensed corpora, scores them for amount-likelihood with the *pack lexicon*
(never with the model), samples a seeded/deduplicated set of candidate
positives and no-signal negatives per language, and runs the shipped weights
over them with exactly the gates `eval_gold` applies.

    python -m sankhya.wild sources                 # registry + licences
    python -m sankhya.wild sample --out data_wild/samples
    python -m sankhya.wild run  --samples data_wild/samples \\
        --weights-json ../models/default/sankhya.weights.int8.json --int8 \\
        --json-out data_wild/wild_metrics.json

Raw corpus files live under `python/data_wild/raw/` (gitignored: some sources
are share-alike, and none of them need to be in this repo to be re-fetched).
`fetch` downloads them; see SOURCES for the URL/licence of each.
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import random
import re
import sys
import unicodedata
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
_PY_ROOT = os.path.dirname(_HERE)
DATA_WILD = os.path.join(_PY_ROOT, "data_wild")
RAW_DIR = os.path.join(DATA_WILD, "raw")

LANGS = ("hi_latn", "hi_deva", "mr_deva", "gu_gujr")


# --------------------------------------------------------------------------
# source registry
# --------------------------------------------------------------------------

class Source:
    def __init__(self, sid, lang, path, kind, url, licence, redistributable, note=""):
        self.id = sid
        self.lang = lang
        self.path = path
        self.kind = kind
        self.url = url
        self.licence = licence
        self.redistributable = redistributable
        self.note = note

    @property
    def abspath(self):
        return os.path.join(RAW_DIR, self.path)


_DAKSHINA_TAR = "https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar"

SOURCES = [
    Source(
        "dakshina_hi_roman", "hi_latn", "hi.romanized.rejoined.tsv", "tsv_col2",
        _DAKSHINA_TAR + " :: dakshina_dataset_v1.0/hi/romanized/hi.romanized.rejoined.tsv",
        "CC BY-SA 4.0", True,
        "Google Dakshina v1.0: 10k Hindi Wikipedia sentences romanised by native speakers "
        "-- real human romanisation, the closest public thing to Hinglish orthography.",
    ),
    Source(
        "dakshina_hi_wiki", "hi_deva", "hi.wiki-filt.train.text.shuf.txt.gz", "gz_text",
        _DAKSHINA_TAR + " :: dakshina_dataset_v1.0/hi/native_script_wikipedia/",
        "CC BY-SA 4.0 (text CC BY-SA 3.0 from Wikipedia)", True,
        "Devanagari Hindi Wikipedia sentences, filtered + shuffled by Dakshina.",
    ),
    Source(
        "dakshina_mr_wiki", "mr_deva", "mr.wiki-filt.train.text.shuf.txt.gz", "gz_text",
        _DAKSHINA_TAR + " :: dakshina_dataset_v1.0/mr/native_script_wikipedia/",
        "CC BY-SA 4.0 (text CC BY-SA 3.0 from Wikipedia)", True,
        "Marathi Wikipedia sentences.",
    ),
    Source(
        "dakshina_gu_wiki", "gu_gujr", "gu.wiki-filt.train.text.shuf.txt.gz", "gz_text",
        _DAKSHINA_TAR + " :: dakshina_dataset_v1.0/gu/native_script_wikipedia/",
        "CC BY-SA 4.0 (text CC BY-SA 3.0 from Wikipedia)", True,
        "Gujarati Wikipedia sentences.",
    ),
    Source(
        "cmu_hinglish_dog", "hi_latn", "cmu_hinglish_dog.train.parquet", "cmu_parquet",
        "https://huggingface.co/datasets/festvox/cmu_hinglish_dog/resolve/main/data/train-00000-of-00001.parquet",
        "CC BY-SA 3.0 / GFDL", True,
        "CMU DoG Hinglish: 8k chat utterances typed by humans in romanised Hindi. "
        "Genuinely informal register (spelling variance, code-switching); amounts are "
        "rare in it, so it mostly contributes real negatives.",
    ),
]

SOURCES_BY_ID = {s.id: s for s in SOURCES}


def sources_for(lang):
    return [s for s in SOURCES if s.lang == lang]


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def _iter_tsv_col2(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                yield parts[1]


def _iter_gz_text(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            yield line.rstrip("\n")


def _iter_cmu_parquet(path):
    import pyarrow.parquet as pq
    t = pq.read_table(path, columns=["translation"])
    for row in t.column("translation").to_pylist():
        if row and row.get("hi_en"):
            yield row["hi_en"]


_LOADERS = {
    "tsv_col2": _iter_tsv_col2,
    "gz_text": _iter_gz_text,
    "cmu_parquet": _iter_cmu_parquet,
}


def load_source(source, limit=None):
    """Yield raw lines from one source (no filtering)."""
    if not os.path.exists(source.abspath):
        raise FileNotFoundError(
            f"{source.abspath} missing -- see `python -m sankhya.wild sources` "
            f"for where to fetch {source.id} from"
        )
    it = _LOADERS[source.kind](source.abspath)
    for i, line in enumerate(it):
        if limit is not None and i >= limit:
            break
        yield line


# --------------------------------------------------------------------------
# amount-likelihood scoring (lexicon only -- never the model)
# --------------------------------------------------------------------------

_WORD_RE = re.compile(r"[\wऀ-ॿ઀-૿]+", re.UNICODE)
_NUM_GLUED_RE = re.compile(r"^[\d.,०-९૦-૯]+([A-Za-z]{1,4})$")
_DIGIT_RE = re.compile(r"[\d०-९૦-૯]")

_LEX_CACHE = {}


def _pack_tables(lang):
    """(word class map, symbol forms, currency markers, suffixes) for a pack."""
    if lang in _LEX_CACHE:
        return _LEX_CACHE[lang]
    from .langs import base as langs_base
    from .verify import union_word_suffixes, union_word_oblique_endings

    pack = langs_base.resolve_pack(lang)
    forms = dict(pack.all_forms())  # surface_lower -> class
    symbols = set()
    for fl in pack.symbol_units.values():
        symbols.update(fl)
    currency = set(m.lower() for m in pack.currency_markers_before)
    currency |= set(w.lower() for w in pack.currency_words_after)
    suffixes = sorted(
        set(union_word_suffixes()) | set(union_word_oblique_endings()),
        key=len, reverse=True,
    )
    out = (pack, forms, symbols, currency, suffixes)
    _LEX_CACHE[lang] = out
    return out


def _lookup(word, forms, suffixes):
    """Class for a surface, tolerating a pack case ending / oblique stem
    ("लाखांचा" -> UNIT_LAKH) the way llm_corpus resolves inflected words."""
    w = unicodedata.normalize("NFC", word).lower()
    if w in forms:
        return forms[w]
    for suf in suffixes:
        if suf and len(w) > len(suf) and w.endswith(suf):
            stem = w[: -len(suf)]
            if stem in forms:
                return forms[stem]
    return None


# weights per signal kind; tuned only to rank, never used as ground truth
_W = {"UNIT": 4, "PFX": 4, "CARD": 1, "CUR": 3, "SYMBOL": 4}

# surfaces this short ("so"/"sau", "lac") collide with ordinary words often
# enough that a lone one is not, by itself, evidence of an amount.
_SHORT_SURFACE = 3


def score_line(text, lang):
    """(score, signals) -- how likely this real line mentions an amount.

    Pure lexicon: scale-unit words, prefix words ("sava"/"डेढ़"), cardinal
    number words, currency markers, and digit+unit shapes ("20k", "2.5L").
    """
    pack, forms, symbols, currency, suffixes = _pack_tables(lang)
    signals = Counter()
    toks = _WORD_RE.findall(text)
    raw_toks = text.split()
    short_unit_only = True  # every UNIT hit came from a short/ambiguous surface

    for t in toks:
        # a symbol unit standing alone ("k", "l", "cr") is not an amount cue:
        # in real Hinglish "k" is overwhelmingly the clitic "ke"/"ki". Symbol
        # units only count glued to digits, handled below.
        if t in symbols:
            continue
        cls = _lookup(t, forms, suffixes)
        if cls is None:
            continue
        if cls.startswith("UNIT_"):
            signals["UNIT"] += 1
            if len(t) > _SHORT_SURFACE:
                short_unit_only = False
        elif cls.startswith("PFX_"):
            signals["PFX"] += 1
        elif cls.startswith("CARD_"):
            signals["CARD"] += 1

    low = [t.lower().strip(".,:;!?") for t in raw_toks]
    for t in low:
        if t in currency:
            signals["CUR"] += 1
    if "₹" in text:
        signals["CUR"] += 1

    if _DIGIT_RE.search(text):
        signals["DIGITS"] += 1

    for t in raw_toks:
        if t in symbols:
            signals["SYMBOL"] += 1
            continue
        m = _NUM_GLUED_RE.match(t)
        if m and m.group(1) in symbols:
            signals["SYMBOL"] += 1

    if signals.get("UNIT") and short_unit_only:
        signals["UNIT_SHORT_ONLY"] = 1
    score = sum(_W[k] * v for k, v in signals.items() if k in _W)
    return score, dict(signals)


def _is_amount_signal(signals):
    """A "positive candidate" needs a real amount cue, not just a number word:
    a scale unit, a prefix word, a currency marker or a digit+unit symbol."""
    if signals.get("PFX") or signals.get("CUR") or signals.get("SYMBOL"):
        return True
    if not signals.get("UNIT"):
        return False
    # a lone short/ambiguous unit surface ("so") needs company -- a digit, a
    # number word or a currency marker -- before it counts as an amount cue.
    if signals.get("UNIT_SHORT_ONLY"):
        return bool(signals.get("CARD") or signals.get("DIGITS"))
    return True


# --------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------

MIN_WORDS = 3
MAX_WORDS = 40
# the char-CNN training/eval path tensorizes at `train.MAX_LEN` characters, so
# a longer line is silently truncated (and `eval_gold.run_json_weights` then
# hands decode_spans a prediction shorter than the text). Real sentences are
# routinely longer than this -- we count how many, and sample only lines the
# model can actually see end to end.
MAX_CHARS = 128


def _clean(line):
    line = unicodedata.normalize("NFC", line).strip()
    line = re.sub(r"\s+", " ", line)
    return line


def _dedup_key(line):
    return re.sub(r"[^\wऀ-ॿ઀-૿]+", "", line.lower())


def scan_source(source, limit=None, counter=None):
    """Yield (text, score, signals) for every length-filtered line."""
    for raw in load_source(source, limit=limit):
        text = _clean(raw)
        n = len(text.split())
        if n < MIN_WORDS or n > MAX_WORDS:
            continue
        if counter is not None:
            counter["in_word_band"] += 1
        if len(text) > MAX_CHARS:
            if counter is not None:
                counter["over_max_chars"] += 1
            continue
        score, signals = score_line(text, source.lang)
        yield text, score, signals


def sample_lang(lang, n_pos=300, n_neg=150, seed=17, limit=None, verbose=True):
    """Seeded, deduplicated sample for one language across its sources.

    Positives are candidates by LEXICON signal only (the model never sees the
    line before it is sampled), drawn uniformly from the candidate pool rather
    than top-scored, so the sample is not biased toward long/rich lines.
    """
    rng = random.Random(f"{seed}:{lang}")
    pos_pool, neg_pool = [], []
    stats = {}
    seen = set()
    for src in sources_for(lang):
        n_lines = n_kept = n_pos_c = n_neg_c = 0
        counter = Counter()
        for text, score, signals in scan_source(src, limit=limit, counter=counter):
            n_lines += 1
            key = _dedup_key(text)
            if not key or key in seen:
                continue
            seen.add(key)
            n_kept += 1
            row = {"text": text, "lang": lang, "source": src.id,
                   "score": score, "signals": signals}
            if _is_amount_signal(signals):
                pos_pool.append(row)
                n_pos_c += 1
            elif score == 0:
                neg_pool.append(row)
                n_neg_c += 1
        stats[src.id] = {
            "lines_in_word_band": counter["in_word_band"],
            "over_max_chars": counter["over_max_chars"],
            "lines_in_length_band": n_lines, "unique": n_kept,
            "positive_candidates": n_pos_c, "negative_candidates": n_neg_c,
        }
        if verbose:
            print(f"  [{src.id}] word-band lines={counter['in_word_band']} "
                  f"(over {MAX_CHARS} chars: {counter['over_max_chars']}) "
                  f"kept={n_lines} unique={n_kept} pos_cand={n_pos_c} neg_cand={n_neg_c}")

    rng.shuffle(pos_pool)
    rng.shuffle(neg_pool)
    pos = pos_pool[:n_pos]
    neg = neg_pool[:n_neg]
    if verbose:
        print(f"  -> sampled {len(pos)} positives / {len(neg)} negatives for {lang}")
    return pos, neg, stats


def write_samples(out_dir, n_pos=300, n_neg=150, seed=17, limit=None, langs=LANGS):
    os.makedirs(out_dir, exist_ok=True)
    all_stats = {}
    for lang in langs:
        print(f"[{lang}]")
        pos, neg, stats = sample_lang(lang, n_pos=n_pos, n_neg=n_neg, seed=seed, limit=limit)
        path = os.path.join(out_dir, f"wild_{lang}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for row in pos:
                row = dict(row, kind="pos_candidate")
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            for row in neg:
                row = dict(row, kind="negative")
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        all_stats[lang] = {"sources": stats, "sampled_positive_candidates": len(pos),
                           "sampled_negatives": len(neg), "path": path}
        print(f"  wrote {path}")
    return all_stats


def load_samples(sample_dir, langs=LANGS):
    out = {}
    for lang in langs:
        path = os.path.join(sample_dir, f"wild_{lang}.jsonl")
        if not os.path.exists(path):
            continue
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        out[lang] = rows
    return out


# --------------------------------------------------------------------------
# running the shipped weights, with eval_gold's gates
# --------------------------------------------------------------------------

def predict(rows, weights_json, int8=True):
    """Run weights over rows and return, per row, the gated decoded spans.

    Uses `eval_gold.run_json_weights` + `eval_gold._apply_bare_digits_gate` so
    what we measure here is exactly what eval_gold (and therefore the shipped
    runtime's gates) would produce.
    """
    from . import core
    from .eval_gold import run_json_weights, _apply_bare_digits_gate
    from .decode import decode_spans
    from .verify import verify_tokens
    from .calibration import apply as apply_calibration, load_calibration

    calib = load_calibration()
    knots_x = calib["raw"] if calib else []
    knots_y = calib["calibrated"] if calib else []

    from .charset import normalize_text
    from .train import MAX_LEN

    # `run_json_weights` predicts over `normalize_text(text)[:MAX_LEN]` but
    # hands back the ORIGINAL text, so anything longer would make decode_spans
    # index past the prediction. Sampling already caps line length; normalize
    # here too so offsets line up with what the model saw.
    examples = [{"text": normalize_text(r["text"])[:MAX_LEN], "lang": r["lang"], "spans": []}
                for r in rows]
    preds, classes = run_json_weights(weights_json, examples, int8=int8)

    out = []
    for row, (text, bio_pred, cls_pred, bio_probs) in zip(rows, preds):
        decoded = decode_spans(text, bio_pred, cls_pred, bio_probs=bio_probs)
        decoded = _apply_bare_digits_gate(text, decoded, classes)
        spans = []
        for d in decoded:
            toks = [(classes[cid], sub) for cid, sub in d["tokens"]]
            res = core.evaluate(toks)
            raw = float(d["confidence"]) if d["confidence"] is not None else 0.0
            spans.append({
                "start": d["start"], "end": d["end"], "span": text[d["start"]:d["end"]],
                "tokens": toks,
                "classes": [c for c, _ in toks],
                "value": res.value,
                "range": list(res.range) if res.range else None,
                "verified": bool(verify_tokens(toks)),
                "raw_confidence": raw,
                "confidence": apply_calibration(raw, knots_x, knots_y),
            })
        out.append({"row": row, "text": text, "spans": spans})
    return out


_CONF_BUCKETS = ((0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01))


def summarize(results):
    """Per-language behaviour summary over wild text (no gold needed)."""
    pos = [r for r in results if r["row"]["kind"] == "pos_candidate"]
    neg = [r for r in results if r["row"]["kind"] == "negative"]

    all_spans = [s for r in results for s in r["spans"]]
    pos_spans = [s for r in pos for s in r["spans"]]

    buckets = {f"{lo:.1f}-{hi:.1f}": 0 for lo, hi in _CONF_BUCKETS}
    for s in all_spans:
        for lo, hi in _CONF_BUCKETS:
            if lo <= s["confidence"] < hi:
                buckets[f"{lo:.1f}-{hi:.1f}"] += 1
                break

    patterns = Counter("+".join(s["classes"]) for s in all_spans)
    verified_patterns = Counter("+".join(s["classes"]) for s in all_spans if s["verified"])
    unverified_patterns = Counter("+".join(s["classes"]) for s in all_spans if not s["verified"])

    def _share(xs):
        return sum(1 for r in xs if r["spans"]) / len(xs) if xs else 0.0

    return {
        "lines": len(results),
        "pos_candidates": len(pos),
        "negatives": len(neg),
        "any_span_share_all": _share(results),
        "any_span_share_pos_candidates": _share(pos),
        "any_span_share_negatives": _share(neg),
        "spans": len(all_spans),
        "spans_on_pos_candidates": len(pos_spans),
        "verified_spans": sum(1 for s in all_spans if s["verified"]),
        "strict_coverage_of_spans": (
            sum(1 for s in all_spans if s["verified"]) / len(all_spans) if all_spans else 0.0
        ),
        "strict_any_span_share_negatives": (
            sum(1 for r in neg if any(s["verified"] for s in r["spans"])) / len(neg) if neg else 0.0
        ),
        "confidence_buckets": buckets,
        "mean_confidence": (sum(s["confidence"] for s in all_spans) / len(all_spans)) if all_spans else 0.0,
        "top_patterns": patterns.most_common(12),
        "top_verified_patterns": verified_patterns.most_common(8),
        "top_unverified_patterns": unverified_patterns.most_common(8),
    }


def _print_summary(lang, summ):
    print(f"\n=== {lang} ===")
    print(f"lines={summ['lines']} (pos_cand={summ['pos_candidates']} neg={summ['negatives']}) "
          f"spans={summ['spans']}")
    print(f"any-span share:   all={summ['any_span_share_all']:.4f} "
          f"pos_cand={summ['any_span_share_pos_candidates']:.4f} "
          f"neg={summ['any_span_share_negatives']:.4f}")
    print(f"strict (verified) coverage of predicted spans: "
          f"{summ['verified_spans']}/{summ['spans']} = {summ['strict_coverage_of_spans']:.4f}")
    print(f"strict any-span share on negatives: {summ['strict_any_span_share_negatives']:.4f}")
    print(f"calibrated confidence buckets: {summ['confidence_buckets']} "
          f"(mean {summ['mean_confidence']:.3f})")
    print("top token patterns:")
    for pat, n in summ["top_patterns"]:
        print(f"  {n:>5}  {pat}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m sankhya.wild")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("sources", help="print the source registry + licences")

    sp = sub.add_parser("sample", help="filter + sample real lines per language")
    sp.add_argument("--out", default=os.path.join(DATA_WILD, "samples"))
    sp.add_argument("--n-pos", type=int, default=300)
    sp.add_argument("--n-neg", type=int, default=150)
    sp.add_argument("--seed", type=int, default=17)
    sp.add_argument("--limit", type=int, default=None, help="cap lines read per source")
    sp.add_argument("--stats-out", default=os.path.join(DATA_WILD, "source_stats.json"))

    rp = sub.add_parser("run", help="run shipped weights over the samples")
    rp.add_argument("--samples", default=os.path.join(DATA_WILD, "samples"))
    rp.add_argument("--weights-json",
                    default=os.path.join(_PY_ROOT, "..", "models", "default", "sankhya.weights.int8.json"))
    rp.add_argument("--int8", action="store_true", default=True)
    rp.add_argument("--float32", dest="int8", action="store_false")
    rp.add_argument("--json-out", default=os.path.join(DATA_WILD, "wild_metrics.json"))
    rp.add_argument("--dump-preds", default=None, help="write per-line predictions as jsonl")

    args = ap.parse_args(argv)

    if args.cmd == "sources":
        for s in SOURCES:
            print(f"{s.id}  [{s.lang}]")
            print(f"  url:              {s.url}")
            print(f"  licence:          {s.licence}")
            print(f"  redistributable:  {s.redistributable}")
            print(f"  local:            {s.abspath} "
                  f"({'present' if os.path.exists(s.abspath) else 'MISSING'})")
            print(f"  note:             {s.note}")
        return

    if args.cmd == "sample":
        stats = write_samples(args.out, n_pos=args.n_pos, n_neg=args.n_neg,
                              seed=args.seed, limit=args.limit)
        if args.stats_out:
            os.makedirs(os.path.dirname(args.stats_out) or ".", exist_ok=True)
            with open(args.stats_out, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            print(f"wrote {args.stats_out}")
        return

    if args.cmd == "run":
        samples = load_samples(args.samples)
        if not samples:
            print(f"no sample files under {args.samples}; run `sample` first", file=sys.stderr)
            return 1
        out = {}
        dump = open(args.dump_preds, "w", encoding="utf-8") if args.dump_preds else None
        for lang, rows in samples.items():
            results = predict(rows, args.weights_json, int8=args.int8)
            if dump:
                for r in results:
                    dump.write(json.dumps({
                        "lang": lang, "text": r["text"], "kind": r["row"]["kind"],
                        "source": r["row"]["source"], "signals": r["row"]["signals"],
                        "spans": r["spans"],
                    }, ensure_ascii=False) + "\n")
            summ = summarize(results)
            out[lang] = summ
            _print_summary(lang, summ)
        if dump:
            dump.close()
            print(f"\nwrote {args.dump_preds}")
        if args.json_out:
            os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
            with open(args.json_out, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2, ensure_ascii=False)
            print(f"wrote {args.json_out}")
        return


# --------------------------------------------------------------------------
# lexicon tokenization (used to VERIFY a hand-written label, never to make one)
# --------------------------------------------------------------------------

_SYM_CLASS_CACHE = {}


def _symbol_class(surface, lang):
    """UNIT_* class for a symbol unit surface ("k", "cr", "L"), case-sensitive."""
    if lang not in _SYM_CLASS_CACHE:
        from .langs import base as langs_base
        pack = langs_base.resolve_pack(lang)
        m = {}
        for cls, forms in pack.symbol_units.items():
            for f in forms:
                m[f] = cls
        _SYM_CLASS_CACHE[lang] = m
    return _SYM_CLASS_CACHE[lang].get(surface)


def tokens_from_lexicon(span_text, lang):
    """Split a span into `core.evaluate` tokens using only the pack lexicon.

    Returns None when some letter run has no lexicon class -- the caller then
    has no independent way to check the label and should leave it alone (this
    is a verification aid for hand-labelling, not a parser).
    """
    _, forms, symbols, _currency, suffixes = _pack_tables(lang)
    text = unicodedata.normalize("NFC", span_text)
    tokens = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch.isspace():
            j = i
            while j < n and text[j].isspace():
                j += 1
            tokens.append(("SEP", text[i:j]))
            i = j
            continue
        if _DIGIT_RE.match(ch):
            j = i
            while j < n and _DIGIT_RE.match(text[j]):
                j += 1
            tokens.append(("DIGITS", text[i:j]))
            i = j
            continue
        if unicodedata.category(ch).startswith(("L", "M")):
            j = i
            while j < n and unicodedata.category(text[j]).startswith(("L", "M")):
                j += 1
            word = text[i:j]
            cls = _symbol_class(word, lang) if word in symbols else None
            if cls is None:
                cls = _lookup(word, forms, suffixes)
            if cls is None or cls == "O":
                return None
            tokens.append((cls, word))
            i = j
            continue
        if ch in ("-", "–", "—", "/"):
            tokens.append(("RANGE", ch))
        elif ch == ".":
            tokens.append(("DOT", ch))
        elif ch == ",":
            tokens.append(("COMMA", ch))
        else:
            return None
        i += 1
    return tokens or None


if __name__ == "__main__":
    raise SystemExit(main())
