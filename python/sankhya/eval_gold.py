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


def run_torch(ckpt_path, examples):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    vocab = ckpt["vocab"]
    classes = ckpt["classes"]
    dilation = ckpt.get("dilation", 1)
    model = SankhyaCNN(vocab_size=len(vocab), n_cls=len(classes), dilation=dilation)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    char_to_id = build_char_to_id(vocab)
    unk = char_to_id.get("<unk>", 1)

    preds = []
    with torch.no_grad():
        for ex in examples:
            text = ex["text"].lower()[:MAX_LEN]
            ids = [char_to_id.get(c, unk) for c in text]
            t = torch.tensor([ids], dtype=torch.int64)
            bio_logits, cls_logits = model(t)
            bio_pred = bio_logits[0].argmax(-1).tolist()
            cls_pred = cls_logits[0].argmax(-1).tolist()
            preds.append((ex["text"], bio_pred, cls_pred))
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
        text = ex["text"].lower()[:MAX_LEN]
        ids = np.array([char_to_id.get(c, unk) for c in text], dtype=np.int64)
        bio_logits, cls_logits = np_infer.forward(weights, ids, dilation=2)
        bio_pred = bio_logits.argmax(-1).tolist()
        cls_pred = cls_logits.argmax(-1).tolist()
        preds.append((ex["text"], bio_pred, cls_pred))
    return preds, classes


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="tests/gold.jsonl")
    ap.add_argument("--ckpt", default="models/sankhya.pt")
    ap.add_argument("--weights-json", default=None, help="use JSON weights (float) instead of torch ckpt")
    ap.add_argument("--int8", action="store_true", help="use int8 JSON weights (requires --weights-json)")
    args = ap.parse_args(argv)

    examples = load_jsonl(args.gold)

    if args.weights_json:
        preds, classes = run_json_weights(args.weights_json, examples, int8=args.int8)
    else:
        preds, classes = run_torch(args.ckpt, examples)

    tp = fp = fn = 0
    val_correct = val_total = 0
    misses = []

    for ex, (text, bio_pred, cls_pred) in zip(examples, preds):
        gold_set = {(s["start"], s["end"]) for s in ex["spans"]}
        decoded = decode_spans(text, bio_pred, cls_pred)
        pred_set = {(d["start"], d["end"]) for d in decoded}
        pred_by_span = {(d["start"], d["end"]): d for d in decoded}

        tp += len(gold_set & pred_set)
        fp_here = pred_set - gold_set
        fn_here = gold_set - pred_set
        fp += len(fp_here)
        fn += len(fn_here)

        for sp in ex["spans"]:
            val_total += 1
            key = (sp["start"], sp["end"])
            ok = False
            pred_desc = "NO SPAN"
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
                misses.append({
                    "text": text, "gold_span": sp, "gold_text": text[sp["start"]:sp["end"]],
                    "pred": pred_desc,
                })
        if fp_here:
            for s, e in fp_here:
                misses.append({"text": text, "extra_pred_span": [s, e], "extra_text": text[s:e]})

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    val_acc = val_correct / val_total if val_total else 0.0

    print(f"gold examples: {len(examples)}")
    print(f"span precision={prec:.4f} recall={rec:.4f} f1={f1:.4f}")
    print(f"value accuracy: {val_correct}/{val_total} = {val_acc:.4f}")
    print(f"\nmisses ({len(misses)}):")
    for m in misses:
        print(f"  {m}")


if __name__ == "__main__":
    main()
