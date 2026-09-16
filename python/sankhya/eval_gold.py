"""Evaluate a checkpoint (torch or JSON weights) against the hand-written gold set."""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from . import classes as C
from . import core
from .decode import decode_spans
from .model import SankhyaCNN
from .train import load_jsonl, build_char_to_id, MAX_LEN
from . import np_infer
from .charset import normalize_text


def run_torch(ckpt_path, examples):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    vocab = ckpt["vocab"]
    classes = ckpt["classes"]
    dilation = ckpt.get("dilation", 1)
    channels = ckpt.get("channels", 32)
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
    layers = obj.get("layers", 3)
    weights = np_infer.load_weights_int8_json(obj) if int8 else np_infer.load_weights_json(obj)
    char_to_id = build_char_to_id(vocab)
    unk = char_to_id.get("<unk>", 1)

    preds = []
    for ex in examples:
        text = normalize_text(ex["text"])[:MAX_LEN]
        L = len(text)
        ids = np_infer.pad_ids(np.array([char_to_id.get(c, unk) for c in text], dtype=np.int64))
        bio_logits, cls_logits = np_infer.forward(weights, ids, dilation=2, layers=layers)
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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=["tests/gold.jsonl"], nargs="+",
                     help="one or more gold jsonl files; per-file metrics are printed too")
    ap.add_argument("--ckpt", default="models/sankhya.pt")
    ap.add_argument("--weights-json", default=None, help="use JSON weights (float) instead of torch ckpt")
    ap.add_argument("--int8", action="store_true", help="use int8 JSON weights (requires --weights-json)")
    args = ap.parse_args(argv)

    gold_files = args.gold if isinstance(args.gold, list) else [args.gold]
    examples = []
    file_bounds = []  # (path, start_idx, end_idx) into `examples`
    for path in gold_files:
        exs = load_jsonl(path)
        file_bounds.append((path, len(examples), len(examples) + len(exs)))
        examples.extend(exs)

    if args.weights_json:
        preds, classes = run_json_weights(args.weights_json, examples, int8=args.int8)
    else:
        preds, classes = run_torch(args.ckpt, examples)

    if len(gold_files) > 1:
        for path, s, e in file_bounds:
            _print_metrics(examples[s:e], preds[s:e], classes, header=f"[{path}]")
        print("\n=== combined ===")

    tp = fp = fn = 0
    val_correct = val_total = 0

    missed_spans = []      # gold span with no overlapping predicted span at all
    wrong_boundary = []    # predicted span overlaps gold but start/end differ
    wrong_value = []       # predicted span matches boundary exactly but decoded value/range is wrong
    spurious_spans = []    # predicted span with no overlapping gold span

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
                # any overlapping predicted span at all?
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


if __name__ == "__main__":
    main()
