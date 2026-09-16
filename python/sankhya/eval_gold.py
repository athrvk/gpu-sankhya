"""Evaluate a checkpoint (torch or JSON weights) against the hand-written gold set."""
from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np
import torch

from . import classes as C
from . import core
from .decode import decode_spans
from .model import SankhyaCNN
from .train import load_jsonl, build_char_to_id, MAX_LEN
from . import np_infer
from .charset import normalize_text
from .langs import base as langs_base

_DEVA_RE = re.compile(r"[ऀ-ॿ]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_WORD_RE = re.compile(r"[\wऀ-ॿ]+", re.UNICODE)
_NUM_SUFFIX_RE = re.compile(r"^[\d.,]+([A-Za-z]+)$")

# categories that describe a single gold span
CATEGORY_NAMES = [
    "digits", "words", "prefix", "range", "currency", "multi_unit",
    "symbol_unit", "mixed_script", "long",
]


def _get_pack(lang_id):
    if lang_id in langs_base.registry:
        return langs_base.registry[lang_id]
    try:
        return langs_base.get_pack(lang_id)
    except KeyError:
        for fallback in ("hi_latn", "hi_deva"):
            try:
                return langs_base.get_pack(fallback)
            except KeyError:
                continue
    return None


def categorize_span(text, span, pack):
    """Return a set of category tags for one gold span.

    `text` is the (already normalized) example text, `span` is the gold
    span dict (with `start`/`end`/`value`/optional `range`/`currency`/
    `unit`), and `pack` is a LanguagePack (may be None if unresolvable).
    """
    start, end = span["start"], span["end"]
    span_text = text[start:end]
    cats = set()

    if any(ch.isdigit() for ch in span_text):
        cats.add("digits")
    else:
        cats.add("words")

    if span.get("range"):
        cats.add("range")

    if span.get("currency"):
        cats.add("currency")
    elif pack is not None:
        before = text[:start].rstrip()
        after = text[end:].lstrip()
        markers_before = [m.lower() for m in pack.currency_markers_before]
        words_after = [w.lower() for w in pack.currency_words_after]
        last_before_word = before.split()[-1].lower() if before.split() else ""
        first_after_word = after.split()[0].lower() if after.split() else ""
        if last_before_word in markers_before or first_after_word in words_after:
            cats.add("currency")

    if len(span_text) >= 20:
        cats.add("long")

    if _DEVA_RE.search(span_text) and _LATIN_RE.search(span_text):
        cats.add("mixed_script")

    if pack is not None:
        forms = pack.all_forms()
        tokens = [t.lower() for t in _WORD_RE.findall(span_text)]
        classes_seen = [forms[t] for t in tokens if t in forms]
        if any(c.startswith("PFX_") for c in classes_seen):
            cats.add("prefix")
        unit_hits = [c for c in classes_seen if c.startswith("UNIT_")]
        if len(unit_hits) >= 2:
            cats.add("multi_unit")
        symbol_forms = set()
        for forms_list in pack.symbol_units.values():
            symbol_forms.update(forms_list)  # symbol case matters (k vs K)
        raw_tokens = span_text.split()
        has_symbol = any(t in symbol_forms for t in raw_tokens)
        if not has_symbol:
            # symbols are usually glued to a number with no space (e.g. "20k", "1.5cr")
            for t in raw_tokens:
                m = _NUM_SUFFIX_RE.match(t)
                if m and m.group(1) in symbol_forms:
                    has_symbol = True
                    break
        if has_symbol:
            cats.add("symbol_unit")

    return cats


def run_torch(ckpt_path, examples):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    vocab = ckpt["vocab"]
    classes = ckpt["classes"]
    channels = ckpt.get("channels", 32)
    embed_dim = ckpt.get("embed_dim", 16)
    arch = ckpt.get("arch")
    if arch is not None:
        model = SankhyaCNN(vocab_size=len(vocab), n_cls=len(classes), arch=arch, channels=channels, embed_dim=embed_dim)
    else:
        dilation = ckpt.get("dilation", 1)
        layers = ckpt.get("layers", 3)
        model = SankhyaCNN(vocab_size=len(vocab), n_cls=len(classes), dilation=dilation, channels=channels, layers=layers)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    char_to_id = build_char_to_id(vocab)
    unk = char_to_id.get("<unk>", 1)

    preds = []
    with torch.no_grad():
        for ex in examples:
            text = normalize_text(ex["text"])[:MAX_LEN]
            L = len(text)
            ids = np_infer.pad_ids([char_to_id.get(c, unk) for c in text])
            t = torch.tensor([ids], dtype=torch.int64)
            bio_logits, cls_logits = model(t)
            bio_logits, cls_logits = bio_logits[0, :L], cls_logits[0, :L]
            bio_pred = bio_logits.argmax(-1).tolist()
            cls_pred = cls_logits.argmax(-1).tolist()
            bio_probs = torch.softmax(bio_logits, dim=-1).tolist()
            preds.append((ex["text"], bio_pred, cls_pred, bio_probs))
    return preds, classes


def run_json_weights(weights_path, examples, int8=False):
    with open(weights_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    vocab = obj["charset"]
    classes = obj["classes"]
    weights = np_infer.load_weights_int8_json(obj) if int8 else np_infer.load_weights_json(obj)
    char_to_id = build_char_to_id(vocab)
    unk = char_to_id.get("<unk>", 1)

    preds = []
    for ex in examples:
        text = normalize_text(ex["text"])[:MAX_LEN]
        L = len(text)
        ids = np_infer.pad_ids(np.array([char_to_id.get(c, unk) for c in text], dtype=np.int64))
        bio_logits, cls_logits = np_infer.forward(weights, ids)
        bio_logits, cls_logits = bio_logits[:L], cls_logits[:L]
        bio_pred = bio_logits.argmax(-1).tolist()
        cls_pred = cls_logits.argmax(-1).tolist()
        bio_probs = np_infer.softmax(bio_logits, axis=-1).tolist()
        preds.append((ex["text"], bio_pred, cls_pred, bio_probs))
    return preds, classes


def _print_metrics(examples, preds, classes, header=""):
    tp = fp = fn = 0
    val_correct = val_total = 0
    for ex, (text, bio_pred, cls_pred, bio_probs) in zip(examples, preds):
        gold_spans = [(s["start"], s["end"]) for s in ex["spans"]]
        decoded = decode_spans(text, bio_pred, cls_pred, bio_probs=bio_probs)
        pred_set = {(d["start"], d["end"]) for d in decoded}
        pred_by_span = {(d["start"], d["end"]): d for d in decoded}
        gold_set = set(gold_spans)
        tp += len(gold_set & pred_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)
        for sp in ex["spans"]:
            val_total += 1
            key = (sp["start"], sp["end"])
            if key in pred_by_span:
                toks = [(classes[cid], sub) for cid, sub in pred_by_span[key]["tokens"]]
                res = core.evaluate(toks)
                if res.value == sp["value"]:
                    if "range" in sp and sp.get("range"):
                        if res.range and list(res.range) == sp["range"]:
                            val_correct += 1
                    else:
                        val_correct += 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    val_acc = val_correct / val_total if val_total else 0.0
    print(f"{header} examples={len(examples)} precision={prec:.4f} recall={rec:.4f} f1={f1:.4f} value_acc={val_acc:.4f}")
    return {
        "examples": len(examples), "spans": val_total,
        "precision": prec, "recall": rec, "f1": f1, "value_acc": val_acc,
    }


def _category_and_negative_metrics(examples, preds, classes):
    """Compute per-category span metrics + negatives/false-positive stats
    for one set of (examples, preds). Returns (categories_dict, negatives_dict).
    """
    cat_stats = {c: {"spans": 0, "value_correct": 0, "exact_match": 0} for c in CATEGORY_NAMES}
    neg_total = 0
    neg_fp = 0

    for ex, (text, bio_pred, cls_pred, bio_probs) in zip(examples, preds):
        norm_text = normalize_text(ex["text"])[:MAX_LEN]
        pack = _get_pack(ex.get("lang"))
        decoded = decode_spans(text, bio_pred, cls_pred, bio_probs=bio_probs)
        pred_set = {(d["start"], d["end"]) for d in decoded}
        pred_by_span = {(d["start"], d["end"]): d for d in decoded}

        if not ex["spans"]:
            neg_total += 1
            if pred_set:
                neg_fp += 1
            continue

        for sp in ex["spans"]:
            cats = categorize_span(norm_text, sp, pack)
            key = (sp["start"], sp["end"])
            exact = key in pred_set
            value_ok = False
            if key in pred_by_span:
                toks = [(classes[cid], sub) for cid, sub in pred_by_span[key]["tokens"]]
                res = core.evaluate(toks)
                if res.value == sp["value"]:
                    if sp.get("range"):
                        if res.range and list(res.range) == sp["range"]:
                            value_ok = True
                    else:
                        value_ok = True
            for c in cats:
                cat_stats[c]["spans"] += 1
                if exact:
                    cat_stats[c]["exact_match"] += 1
                if value_ok:
                    cat_stats[c]["value_correct"] += 1

    categories = {}
    for c, st in cat_stats.items():
        if st["spans"] == 0:
            continue
        categories[c] = {
            "spans": st["spans"],
            "value_acc": st["value_correct"] / st["spans"],
            "exact_match_rate": st["exact_match"] / st["spans"],
        }
    negatives = {
        "examples": neg_total,
        "false_positives": neg_fp,
        "fp_rate": (neg_fp / neg_total) if neg_total else 0.0,
    }
    return categories, negatives


def _print_category_table(categories, negatives, header=""):
    if header:
        print(f"\n{header}")
    print(f"{'category':<14}{'spans':>7}{'value_acc':>12}{'exact_match':>13}")
    for c in CATEGORY_NAMES:
        if c not in categories:
            continue
        st = categories[c]
        print(f"{c:<14}{st['spans']:>7}{st['value_acc']:>12.4f}{st['exact_match_rate']:>13.4f}")
    print(f"\nnegatives: examples={negatives['examples']} false_positives={negatives['false_positives']} "
          f"fp_rate={negatives['fp_rate']:.4f}")


def _evaluate_all(gold_files, examples, preds, classes, file_bounds, quiet=False):
    per_file_metrics = {}
    per_file_categories = {}
    if len(gold_files) > 1:
        for path, s, e in file_bounds:
            per_file_metrics[path] = _print_metrics(examples[s:e], preds[s:e], classes, header=f"[{path}]")
            cats, negs = _category_and_negative_metrics(examples[s:e], preds[s:e], classes)
            per_file_metrics[path]["categories"] = cats
            per_file_metrics[path]["negatives"] = negs
            per_file_categories[path] = (cats, negs)
        print("\n=== combined ===")

    tp = fp = fn = 0
    val_correct = val_total = 0

    missed_spans = []
    wrong_boundary = []
    wrong_value = []
    spurious_spans = []

    def overlaps(a, b):
        return a[0] < b[1] and b[0] < a[1]

    for ex, (text, bio_pred, cls_pred, bio_probs) in zip(examples, preds):
        gold_spans = [(s["start"], s["end"]) for s in ex["spans"]]
        decoded = decode_spans(text, bio_pred, cls_pred, bio_probs=bio_probs)
        pred_set = {(d["start"], d["end"]) for d in decoded}
        pred_by_span = {(d["start"], d["end"]): d for d in decoded}
        gold_set = set(gold_spans)

        tp += len(gold_set & pred_set)
        fp_here = pred_set - gold_set
        fn_here = gold_set - pred_set
        fp += len(fp_here)
        fn += len(fn_here)

        for sp in ex["spans"]:
            val_total += 1
            key = (sp["start"], sp["end"])
            ok = False
            if key in pred_by_span:
                toks = [(classes[cid], sub) for cid, sub in pred_by_span[key]["tokens"]]
                res = core.evaluate(toks)
                pred_desc = f"value={res.value} range={res.range}"
                if res.value == sp["value"]:
                    if "range" in sp:
                        if res.range and list(res.range) == sp["range"]:
                            ok = True
                    else:
                        ok = True
                if ok:
                    val_correct += 1
                else:
                    wrong_value.append({
                        "text": text, "gold_span": sp, "gold_text": text[sp["start"]:sp["end"]],
                        "pred": pred_desc,
                    })
            else:
                overlap = next((p for p in pred_set if overlaps(p, key)), None)
                if overlap is not None:
                    wrong_boundary.append({
                        "text": text, "gold_span": sp, "gold_text": text[sp["start"]:sp["end"]],
                        "pred_span": list(overlap), "pred_text": text[overlap[0]:overlap[1]],
                    })
                else:
                    missed_spans.append({
                        "text": text, "gold_span": sp, "gold_text": text[sp["start"]:sp["end"]],
                    })

        for s, e in fp_here:
            if not any(overlaps((s, e), g) for g in gold_spans):
                spurious_spans.append({"text": text, "extra_pred_span": [s, e], "extra_text": text[s:e]})

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    val_acc = val_correct / val_total if val_total else 0.0

    print(f"gold examples: {len(examples)}")
    print(f"span precision={prec:.4f} recall={rec:.4f} f1={f1:.4f}")
    print(f"value accuracy: {val_correct}/{val_total} = {val_acc:.4f}")

    total_misses = len(missed_spans) + len(spurious_spans) + len(wrong_value) + len(wrong_boundary)
    print("\nmiss summary:")
    print(f"  {'category':<16}{'count':>6}")
    print(f"  {'missed span':<16}{len(missed_spans):>6}")
    print(f"  {'spurious span':<16}{len(spurious_spans):>6}")
    print(f"  {'wrong value':<16}{len(wrong_value):>6}")
    print(f"  {'wrong boundary':<16}{len(wrong_boundary):>6}")
    print(f"  {'total':<16}{total_misses:>6}")

    combined_categories, combined_negatives = _category_and_negative_metrics(examples, preds, classes)
    _print_category_table(combined_categories, combined_negatives, header="per-category metrics (combined):")

    if not quiet:
        print(f"\nmissed spans ({len(missed_spans)}):")
        for m in missed_spans:
            print(f"  {m}")
        print(f"\nspurious spans ({len(spurious_spans)}):")
        for m in spurious_spans:
            print(f"  {m}")
        print(f"\nwrong value ({len(wrong_value)}):")
        for m in wrong_value:
            print(f"  {m}")
        print(f"\nwrong boundary ({len(wrong_boundary)}):")
        for m in wrong_boundary:
            print(f"  {m}")

    combined_metrics = {
        "examples": len(examples), "spans": val_total,
        "precision": prec, "recall": rec, "f1": f1, "value_acc": val_acc,
        "miss_summary": {
            "missed_span": len(missed_spans), "spurious_span": len(spurious_spans),
            "wrong_value": len(wrong_value), "wrong_boundary": len(wrong_boundary),
            "total": total_misses,
        },
        "categories": combined_categories,
        "negatives": combined_negatives,
    }
    return {"per_file": per_file_metrics, "combined": combined_metrics}


def run_eval(gold_files, ckpt=None, weights_json=None, int8=False, quiet=False):
    """Programmatic entry point: run inference + full evaluation, return the
    metrics dict (same shape written to --json-out). Used by eval_matrix.py.
    """
    if isinstance(gold_files, str):
        gold_files = [gold_files]
    examples = []
    file_bounds = []
    for path in gold_files:
        exs = load_jsonl(path)
        file_bounds.append((path, len(examples), len(examples) + len(exs)))
        examples.extend(exs)

    if weights_json:
        preds, classes = run_json_weights(weights_json, examples, int8=int8)
    else:
        preds, classes = run_torch(ckpt, examples)

    return _evaluate_all(gold_files, examples, preds, classes, file_bounds, quiet=quiet)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=["tests/gold.jsonl"], nargs="+",
                     help="one or more gold jsonl files; per-file metrics are printed too")
    ap.add_argument("--ckpt", default="models/sankhya.pt")
    ap.add_argument("--weights-json", default=None, help="use JSON weights (float) instead of torch ckpt")
    ap.add_argument("--int8", action="store_true", help="use int8 JSON weights (requires --weights-json)")
    ap.add_argument("--json-out", default=None, help="write per-file + combined metrics as JSON to this path")
    ap.add_argument("--quiet", action="store_true", help="suppress the miss lists (still prints tables/summaries)")
    args = ap.parse_args(argv)

    gold_files = args.gold if isinstance(args.gold, list) else [args.gold]

    out = run_eval(
        gold_files, ckpt=args.ckpt, weights_json=args.weights_json, int8=args.int8,
        quiet=args.quiet,
    )

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote metrics json: {args.json_out}")


if __name__ == "__main__":
    main()
