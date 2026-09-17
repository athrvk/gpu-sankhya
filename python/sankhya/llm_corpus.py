"""Verify, inspect and sample LLM-written raw corpus lines for gpu-sankhya.

Raw lines are written by other agents following
`data_llm/prompts/GENERATOR_BRIEF.md`: one JSON object per line with
`text`, `lang`, `phrases` (list of {phrase, value, range?}), and free-form
`scenario`/`gen` tags. `verify` re-derives labels (bio/cls/spans) for each
accepted line using the SAME lexicon + `core.evaluate` the synthetic
generator (`sankhya.generator`) uses, so accepted output round-trips
through `sankhya.train.load_jsonl`/`tensorize` unchanged.

    python -m sankhya.llm_corpus verify --lang hi_latn \
        --in data_llm/raw/*.jsonl --out data_llm/hi_latn.jsonl \
        --rejects data_llm/rejects/hi_latn.jsonl --manifest data_llm/manifest.json
    python -m sankhya.llm_corpus stats --in data_llm/hi_latn.jsonl
    python -m sankhya.llm_corpus sample --in data_llm/hi_latn.jsonl --n 20 --seed 1
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import random
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from . import classes as C
from . import core
from . import charset as CS
from .langs import base as langbase

HERE = Path(__file__).resolve().parent
PY_ROOT = HERE.parent
DEFAULT_PROMPT_FILE = PY_ROOT / "data_llm" / "prompts" / "GENERATOR_BRIEF.md"
DEFAULT_GOLD_GLOB = str(PY_ROOT / "tests" / "gold*.jsonl")


# --------------------------------------------------------------------------
# tokenization: mirrors generator.py's per-char class assignment closely
# enough to reproduce the same (cls, text) token stream for a free-text
# phrase, so core.evaluate() sees exactly what it would see for
# generator-built data.
# --------------------------------------------------------------------------

def _char_kind(ch: str) -> str:
    if ch.isspace():
        return "space"
    if ch in ".":
        return "dot"
    if ch in ",":
        return "comma"
    if ch.isdigit():
        return "digit"
    cat = unicodedata.category(ch)
    if cat.startswith("L") or cat.startswith("M"):
        return "letter"
    return "other"


def _iter_runs(s: str):
    """Yield (kind, text) runs: 'space', 'numeric' (digit/dot/comma mixed,
    kept together so '2,50,000' / '1.5' stay one run), 'letter', 'other'
    (one char at a time)."""
    i = 0
    n = len(s)
    while i < n:
        kind = _char_kind(s[i])
        if kind == "space":
            j = i
            while j < n and s[j].isspace():
                j += 1
            yield "space", s[i:j]
            i = j
        elif kind in ("digit", "dot", "comma"):
            j = i
            while j < n and _char_kind(s[j]) in ("digit", "dot", "comma"):
                j += 1
            yield "numeric", s[i:j]
            i = j
        elif kind == "letter":
            j = i
            while j < n and _char_kind(s[j]) == "letter":
                j += 1
            yield "letter", s[i:j]
            i = j
        else:
            yield "other", s[i]
            i += 1


class TokenizeError(Exception):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def _word_class(pack, word_lower):
    """Classify a single letters-run. Returns a class string, or None if
    unrecognised (caller decides whether that is fatal)."""
    forms = pack.all_forms()
    if word_lower in forms:
        return forms[word_lower]
    if word_lower in {w.strip().lower() for w in pack.range_connectors if w.strip().isalpha()}:
        return "RANGE"
    if word_lower in {w.lower() for w in getattr(pack, "conj_connectors", [])}:
        return "O"
    if word_lower in {w.lower() for w in getattr(pack, "approximators", []) if w.isalpha()}:
        return "O"
    if word_lower in {w.lower() for w in getattr(pack, "currency_words_after", []) if w.isalpha()}:
        return "O"
    if word_lower in {w.strip(".").lower() for w in getattr(pack, "currency_markers_before", []) if w.strip(".").isalpha()}:
        return "O"
    return None


def tokenize_span_text(pack, text):
    """Tokenize a normalized phrase/text segment into (cls, text) tokens,
    the same shape generator.py's `build_term_tokens` produces. Raises
    TokenizeError('unknown_token:<token>') for a letters-run this pack's
    lexicon does not recognise (a lone possessive suffix like "'s" is
    tolerated as O, mirroring generator._maybe_possessive)."""
    tokens = []
    prev_other = None
    for kind, run in _iter_runs(text):
        if kind == "space":
            tokens.append(("SEP", run))
            prev_other = None
        elif kind == "numeric":
            for ch in run:
                if ch == ".":
                    tokens.append(("DOT", ch))
                elif ch == ",":
                    tokens.append(("COMMA", ch))
                else:
                    tokens.append(("DIGITS", ch))
            prev_other = None
        elif kind == "letter":
            cls = _word_class(pack, run.lower())
            if cls is None:
                # possessive suffix directly after an apostrophe ("'s") -> O
                if prev_other in ("'", "’") and len(run) <= 2:
                    cls = "O"
                else:
                    raise TokenizeError(f"unknown_token:{run}")
            tokens.append((cls, run))
            prev_other = None
        else:  # 'other': single punctuation/symbol char
            if run in ("-", "–", "—", "/"):
                tokens.append(("RANGE", run))
            else:
                tokens.append(("O", run))
            prev_other = run
    return tokens


def _negative_has_quantity(pack, normalized_text):
    """True if a UNIT_*/PFX_* lexicon form (word or symbol, e.g. bare "k"
    glued to digits) appears anywhere in the text -- disqualifies it as a
    negative."""
    forms = pack.all_forms()
    for kind, run in _iter_runs(normalized_text):
        if kind != "letter":
            continue
        cls = forms.get(run.lower())
        if cls and (cls.startswith("UNIT_") or cls.startswith("PFX_")):
            return True
    return False


# --------------------------------------------------------------------------
# per-line verification
# --------------------------------------------------------------------------

def _values_match(a, b, tol=1e-6):
    try:
        a = float(a)
        b = float(b)
    except (TypeError, ValueError):
        return False
    return abs(a - b) <= max(tol, abs(b) * 1e-6)


def _range_match(result_range, claimed_range):
    if claimed_range is None:
        return True
    if result_range is None:
        return False
    if len(claimed_range) != 2 or len(result_range) != 2:
        return False
    return _values_match(result_range[0], claimed_range[0]) and _values_match(result_range[1], claimed_range[1])


class RejectError(Exception):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def verify_line(pack, obj, gold_texts, seen_texts):
    """Verify one parsed raw JSON object. Returns the accepted training-schema
    dict, or raises RejectError(reason)."""
    if not isinstance(obj, dict):
        raise RejectError("malformed")
    line_lang = obj.get("lang")
    if line_lang is not None and line_lang != pack.id:
        raise RejectError("lang_mismatch")
    raw_text = obj.get("text")
    phrases = obj.get("phrases")
    if not isinstance(raw_text, str) or not raw_text:
        raise RejectError("missing_text")
    if phrases is None or not isinstance(phrases, list):
        raise RejectError("missing_phrases")

    normalized = CS.normalize_text(raw_text)
    if len(normalized) != len(raw_text):
        raise RejectError("nfc_length_mismatch")

    if normalized in seen_texts:
        raise RejectError("dup")
    if normalized in gold_texts:
        raise RejectError("dup_gold")

    bio = [0] * len(raw_text)
    clsids = [C.CLASS_TO_ID["O"]] * len(raw_text)
    spans = []

    if not phrases:
        if _negative_has_quantity(pack, normalized):
            raise RejectError("negative_contains_quantity")
        return {
            "text": raw_text, "lang": pack.id, "spans": [], "bio": bio, "cls": clsids,
            "source": "llm", "gen": obj.get("gen"),
        }

    # count how many times each distinct phrase string is claimed vs.
    # actually occurs, to reject genuinely ambiguous placements while still
    # allowing a legitimately-repeated phrase ("50k ya 50k" twice).
    claimed_counts = Counter()
    for p in phrases:
        if not isinstance(p, dict) or not isinstance(p.get("phrase"), str):
            raise RejectError("bad_phrase_entry")
        claimed_counts[CS.normalize_text(p["phrase"])] += 1

    used_spans = []  # (start, end)
    search_cursor = defaultdict(int)  # phrase_norm -> next search-from index

    for p in phrases:
        phrase = p["phrase"]
        value = p.get("value")
        rng = p.get("range")
        if value is None:
            raise RejectError("missing_value")
        phrase_norm = CS.normalize_text(phrase)
        if len(phrase_norm) != len(phrase):
            raise RejectError("nfc_length_mismatch")
        occurrences = normalized.count(phrase_norm)
        if occurrences == 0:
            raise RejectError("phrase_not_found")
        if occurrences != claimed_counts[phrase_norm]:
            raise RejectError("ambiguous_phrase")

        start = normalized.find(phrase_norm, search_cursor[phrase_norm])
        if start == -1:
            raise RejectError("phrase_not_found")
        end = start + len(phrase_norm)
        search_cursor[phrase_norm] = end

        for (us, ue) in used_spans:
            if start < ue and us < end:
                raise RejectError("overlapping_phrase")
        used_spans.append((start, end))

        span_text = normalized[start:end]
        try:
            toks = tokenize_span_text(pack, span_text)
        except TokenizeError as e:
            raise RejectError(e.reason)

        result = core.evaluate(toks)
        if not _values_match(result.value, value):
            raise RejectError(f"value_mismatch: ours={result.value} theirs={value}")
        if not _range_match(result.range, rng):
            ours = list(result.range) if result.range else None
            raise RejectError(f"value_mismatch: ours_range={ours} theirs_range={rng}")

        currency = core.detect_currency(normalized, start, end, pack)
        classes_str = " ".join(c for c, _ in toks if c != "SEP")
        spans.append({
            "start": start, "end": end,
            "value": result.value,
            "range": list(result.range) if result.range else None,
            "unit": result.unit,
            "currency": currency,
            "classes": classes_str,
        })
        for i in range(start, end):
            bio[i] = 1 if i == start else 2
        pos = start
        for cls, t in toks:
            for _ in t:
                if pos < end:
                    clsids[pos] = C.CLASS_TO_ID.get(cls, C.CLASS_TO_ID["O"])
                pos += 1

    if not spans:
        raise RejectError("phrase_not_found")

    return {
        "text": raw_text, "lang": pack.id, "spans": spans, "bio": bio, "cls": clsids,
        "source": "llm", "gen": obj.get("gen"),
    }


# --------------------------------------------------------------------------
# verify command
# --------------------------------------------------------------------------

def _load_gold_texts(patterns):
    texts = set()
    files = []
    for pat in patterns:
        files.extend(sorted(glob.glob(pat)))
    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    t = obj.get("text")
                    if isinstance(t, str):
                        texts.add(CS.normalize_text(t))
        except OSError:
            continue
    return texts


def _sha256_file(path):
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            h.update(f.read())
        return h.hexdigest()
    except OSError:
        return None


def cmd_verify(args):
    pack = langbase.get_pack(args.lang)

    in_files = []
    for pat in args.inp:
        matched = sorted(glob.glob(pat))
        in_files.extend(matched if matched else [pat])
    in_files = [f for f in in_files if Path(f).is_file()]

    gold_texts = _load_gold_texts([args.gold] if isinstance(args.gold, str) else args.gold)

    total = 0
    malformed = 0
    accepted = []
    reject_rows = []
    reject_reasons = Counter()
    unknown_tokens = Counter()
    seen_texts = set()
    lengths = []
    n_negatives = 0
    n_multi = 0

    for path in in_files:
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                raw_line = line.strip()
                if not raw_line:
                    continue
                total += 1
                try:
                    obj = json.loads(raw_line)
                except json.JSONDecodeError:
                    malformed += 1
                    reject_reasons["malformed_json"] += 1
                    continue
                try:
                    ex = verify_line(pack, obj, gold_texts, seen_texts)
                except RejectError as e:
                    reject_reasons[e.args[0].split(":")[0]] += 1
                    if e.args[0].startswith("unknown_token:"):
                        unknown_tokens[e.args[0].split(":", 1)[1]] += 1
                    reject_rows.append({
                        "text": obj.get("text") if isinstance(obj, dict) else None,
                        "phrases": obj.get("phrases") if isinstance(obj, dict) else None,
                        "reason": e.args[0], "src_file": path, "src_line": line_no,
                    })
                    continue
                normalized = CS.normalize_text(ex["text"])
                seen_texts.add(normalized)
                accepted.append(ex)
                lengths.append(len(ex["text"]))
                if not ex["spans"]:
                    n_negatives += 1
                elif len(ex["spans"]) >= 2:
                    n_multi += 1

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for ex in accepted:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    if args.rejects:
        Path(args.rejects).parent.mkdir(parents=True, exist_ok=True)
        with open(args.rejects, "w", encoding="utf-8") as f:
            for row in reject_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_accepted = len(accepted)
    n_rejected = total - n_accepted
    print(f"=== verify summary ({args.lang}) ===")
    print(f"input files: {len(in_files)}")
    print(f"total lines: {total}")
    print(f"accepted:    {n_accepted}")
    print(f"rejected:    {n_rejected}")
    if n_accepted:
        print(f"negatives share: {n_negatives / n_accepted:.1%}")
        print(f"2+ span share:   {n_multi / n_accepted:.1%}")
        print(f"mean length:     {sum(lengths) / len(lengths):.1f} chars")
    print("\nrejects by reason:")
    for reason, cnt in reject_reasons.most_common():
        print(f"  {reason:30s} {cnt}")
    if unknown_tokens:
        print("\ntop unknown tokens:")
        for tok, cnt in unknown_tokens.most_common(15):
            print(f"  {tok!r:20s} {cnt}")

    if args.manifest:
        manifest_path = Path(args.manifest)
        manifest = {}
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                manifest = {}
        manifest.setdefault("langs", {})
        manifest["langs"][args.lang] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total": total,
            "accepted": n_accepted,
            "rejected": n_rejected,
            "malformed": malformed,
            "prompt_file": str(args.prompt),
            "prompt_sha256": _sha256_file(args.prompt),
            "input_files": in_files,
            "reject_histogram": dict(reject_reasons),
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nwrote manifest to {manifest_path}")


# --------------------------------------------------------------------------
# stats command
# --------------------------------------------------------------------------

def _load_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def cmd_stats(args):
    examples = _load_jsonl(args.inp)
    n = len(examples)
    class_hist = Counter()
    surface_by_class = defaultdict(set)
    n_multi = 0
    n_negative = 0

    for ex in examples:
        spans = ex.get("spans", [])
        if not spans:
            n_negative += 1
        elif len(spans) >= 2:
            n_multi += 1
        for sp in spans:
            for tok in sp.get("classes", "").split():
                class_hist[tok] += 1
        text = ex.get("text", "")
        clsids = ex.get("cls", [])
        i = 0
        L = min(len(text), len(clsids))
        while i < L:
            cid = clsids[i]
            j = i
            while j < L and clsids[j] == cid:
                j += 1
            name = C.CLASSES[cid] if cid < len(C.CLASSES) else "O"
            if name != "O":
                surface_by_class[name].add(text[i:j].lower())
            i = j

    print(f"=== stats: {args.inp} ===")
    print(f"total lines: {n}")
    if n:
        print(f"negatives share: {n_negative / n:.1%}")
        print(f"2+ span share:   {n_multi / n:.1%}")
    print("\nclass histogram (occurrences in spans):")
    for cls, cnt in class_hist.most_common(40):
        print(f"  {cls:14s} {cnt}")
    print("\nunique surface forms per class (top 30 by count):")
    counts = sorted(((cls, len(s)) for cls, s in surface_by_class.items()), key=lambda x: -x[1])
    for cls, cnt in counts[:30]:
        print(f"  {cls:14s} {cnt} unique forms")


# --------------------------------------------------------------------------
# sample command
# --------------------------------------------------------------------------

def cmd_sample(args):
    examples = _load_jsonl(args.inp)
    rng = random.Random(args.seed)
    n = min(args.n, len(examples))
    chosen = rng.sample(examples, n) if n else []
    for ex in chosen:
        text = ex.get("text", "")
        spans = ex.get("spans", [])
        phrase_desc = "; ".join(
            f"{text[sp['start']:sp['end']]!r}=>{sp['value']}" for sp in spans
        ) or "(negative)"
        print(f"{text!r}\n  {phrase_desc}\n")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m sankhya.llm_corpus")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_verify = sub.add_parser("verify", help="verify raw LLM lines and write the training-schema corpus")
    p_verify.add_argument("--lang", required=True)
    p_verify.add_argument("--in", dest="inp", nargs="+", required=True, help="raw jsonl file(s)/glob(s)")
    p_verify.add_argument("--out", required=True)
    p_verify.add_argument("--rejects", default=None)
    p_verify.add_argument("--manifest", default=None)
    p_verify.add_argument("--prompt", default=str(DEFAULT_PROMPT_FILE))
    p_verify.add_argument("--gold", nargs="+", default=[DEFAULT_GOLD_GLOB])
    p_verify.set_defaults(func=cmd_verify)

    p_stats = sub.add_parser("stats", help="print class/coverage stats for a verified corpus file")
    p_stats.add_argument("--in", dest="inp", required=True)
    p_stats.set_defaults(func=cmd_stats)

    p_sample = sub.add_parser("sample", help="print a random sample of lines with their phrases")
    p_sample.add_argument("--in", dest="inp", required=True)
    p_sample.add_argument("--n", type=int, default=20)
    p_sample.add_argument("--seed", type=int, default=0)
    p_sample.set_defaults(func=cmd_sample)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
