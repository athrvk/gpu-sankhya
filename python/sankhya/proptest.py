"""Property test: generate fresh examples and check the model + decoder +
verifier against the generator's own ground truth.

Unlike `eval_gold` (410 hand-written examples) this draws an arbitrary number
of fresh synthetic examples from the same generator used for training data,
runs the torch checkpoint in batches, decodes with exactly the gates
`eval_gold` uses, and reports the plain and the strict (verified-only) tier.

    python -m sankhya.proptest --n 50000 --seed 11 --ckpt ../models/default/sankhya.pt \
        --lang hi_latn,hi_deva --mix 0.55,0.45 --cross 0.10
"""
from __future__ import annotations

import argparse
import collections
import json

import torch

from . import core
from .decode import decode_spans
from .eval_gold import _apply_bare_digits_gate
from .generator import generate, generate_multi
from .langs import base as langs_base
from .langs.base import load_all

load_all()  # registers every known pack
from .model import SankhyaCNN
from .np_infer import PAD_TAIL
from .train import MAX_LEN, build_char_to_id
from .charset import normalize_text
from .verify import verify_tokens


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    vocab = ckpt["vocab"]
    classes = ckpt["classes"]
    channels = ckpt.get("channels", 32)
    embed_dim = ckpt.get("embed_dim", 16)
    arch = ckpt.get("arch")
    use_crf = ckpt.get("use_crf", False)
    if arch is not None:
        model = SankhyaCNN(vocab_size=len(vocab), n_cls=len(classes), arch=arch,
                           channels=channels, embed_dim=embed_dim, crf=use_crf)
    else:
        model = SankhyaCNN(vocab_size=len(vocab), n_cls=len(classes),
                           dilation=ckpt.get("dilation", 1), channels=channels,
                           layers=ckpt.get("layers", 3), crf=use_crf)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, vocab, classes


def run_torch_batched(model, vocab, examples, batch_size=256):
    """Same predictions as eval_gold.run_torch, computed in batches."""
    char_to_id = build_char_to_id(vocab)
    unk = char_to_id.get("<unk>", 1)
    texts = [normalize_text(ex["text"])[:MAX_LEN] for ex in examples]
    preds = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            width = max((len(t) for t in chunk), default=0) + PAD_TAIL
            ids = torch.zeros((len(chunk), width), dtype=torch.int64)
            for r, t in enumerate(chunk):
                for c, ch in enumerate(t):
                    ids[r, c] = char_to_id.get(ch, unk)
            bio_logits, cls_logits = model(ids)
            cls_pred_all = cls_logits.argmax(-1)
            for r, t in enumerate(chunk):
                L = len(t)
                bl = bio_logits[r, :L]
                if model.crf is not None:
                    bio_pred = model.crf.viterbi(bl) if L else []
                else:
                    bio_pred = bl.argmax(-1).tolist()
                preds.append((
                    t,
                    bio_pred,
                    cls_pred_all[r, :L].tolist(),
                    torch.softmax(bl, dim=-1).tolist(),
                ))
    return preds


def evaluate(examples, preds, classes, top_k=20):
    gold_total = 0
    exact_value = 0
    strict_covered = 0
    strict_value_correct = 0
    strict_spurious = 0
    neg_examples = 0
    neg_fp = 0
    strict_neg_fp = 0
    miss_patterns = collections.Counter()
    miss_samples = {}

    def overlaps(a, b):
        return a[0] < b[1] and b[0] < a[1]

    for ex, (text, bio_pred, cls_pred, bio_probs) in zip(examples, preds):
        decoded = decode_spans(text, bio_pred, cls_pred, bio_probs=bio_probs)
        decoded = _apply_bare_digits_gate(text, decoded, classes)
        pred_by_span = {}
        verified_by_span = {}
        for d in decoded:
            toks = [(classes[cid], sub) for cid, sub in d["tokens"]]
            pred_by_span[(d["start"], d["end"])] = toks
            verified_by_span[(d["start"], d["end"])] = verify_tokens(toks)

        if not ex["spans"]:
            neg_examples += 1
            if pred_by_span:
                neg_fp += 1
            if any(verified_by_span.values()):
                strict_neg_fp += 1
            continue

        gold_spans = [(s["start"], s["end"]) for s in ex["spans"]]
        for key, v in verified_by_span.items():
            if v and not any(overlaps(key, g) for g in gold_spans):
                strict_spurious += 1

        for sp in ex["spans"]:
            gold_total += 1
            key = (sp["start"], sp["end"])
            ok = False
            toks = pred_by_span.get(key)
            if toks is not None:
                res = core.evaluate(toks)
                if res.value == sp["value"]:
                    if sp.get("range"):
                        ok = bool(res.range) and list(res.range) == list(sp["range"])
                    else:
                        ok = True
            if ok:
                exact_value += 1
            else:
                pat = sp.get("classes") or "?"
                miss_patterns[pat] += 1
                miss_samples.setdefault(pat, {
                    "text": ex["text"],
                    "gold_text": text[sp["start"]:sp["end"]],
                    "gold_value": sp["value"],
                    "pred_tokens": toks,
                    "pred_value": (core.evaluate(toks).value if toks else None),
                })
            if toks is not None and verified_by_span.get(key):
                strict_covered += 1
                if ok:
                    strict_value_correct += 1

    out = {
        "examples": len(examples),
        "gold_spans": gold_total,
        "exact_value_rate": exact_value / gold_total if gold_total else 0.0,
        "strict": {
            "covered": strict_covered,
            "coverage": strict_covered / gold_total if gold_total else 0.0,
            "value_correct": strict_value_correct,
            "value_acc": strict_value_correct / strict_covered if strict_covered else 0.0,
            "spurious": strict_spurious,
            "neg_fp": strict_neg_fp,
            "neg_fp_rate": strict_neg_fp / neg_examples if neg_examples else 0.0,
        },
        "negatives": {
            "examples": neg_examples,
            "false_positives": neg_fp,
            "fp_rate": neg_fp / neg_examples if neg_examples else 0.0,
        },
        "top_misses": [
            {"pattern": p, "count": c, "example": miss_samples[p]}
            for p, c in miss_patterns.most_common(top_k)
        ],
    }
    return out


def print_report(out):
    s = out["strict"]
    print(f"examples={out['examples']} positive spans={out['gold_spans']}")
    print(f"exact-value rate (all spans): {out['exact_value_rate']:.4f}")
    print(f"strict coverage:              {s['covered']}/{out['gold_spans']} = {s['coverage']:.4f}")
    print(f"strict value accuracy:        {s['value_correct']}/{s['covered']} = {s['value_acc']:.4f}")
    print(f"strict spurious spans:        {s['spurious']}")
    print(f"negatives: examples={out['negatives']['examples']} "
          f"fp={out['negatives']['false_positives']} fp_rate={out['negatives']['fp_rate']:.4f} "
          f"| strict fp={s['neg_fp']} strict_fp_rate={s['neg_fp_rate']:.4f}")
    print(f"\ntop {len(out['top_misses'])} misses by gold class pattern:")
    print(f"  {'count':>6}  pattern")
    for m in out["top_misses"]:
        print(f"  {m['count']:>6}  {m['pattern']}")
        ex = m["example"]
        print(f"          e.g. {ex['gold_text']!r} gold={ex['gold_value']} pred={ex['pred_value']}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m sankhya.proptest")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--ckpt", default="../models/default/sankhya.pt")
    ap.add_argument("--lang", default=",".join(langs_base.KNOWN_PACKS))
    ap.add_argument("--mix", default=None, help="comma-separated weights matching --lang")
    ap.add_argument("--cross", type=float, default=0.0)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)

    lang_ids = [x.strip() for x in args.lang.split(",") if x.strip()]
    packs = [langs_base.get_pack(l) for l in lang_ids]
    weights = [float(x) for x in args.mix.split(",")] if args.mix else None
    if weights is not None:
        assert len(weights) == len(packs), "--mix must match --lang count"
    if len(packs) == 1:
        examples = generate(packs[0], args.n, seed=args.seed)
    else:
        examples = generate_multi(packs, weights=weights, n=args.n, seed=args.seed,
                                  cross=args.cross)

    model, vocab, classes = load_model(args.ckpt)
    preds = run_torch_batched(model, vocab, examples, batch_size=args.batch_size)
    out = evaluate(examples, preds, classes, top_k=args.top)
    print_report(out)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\nwrote {args.json_out}")
    return out


if __name__ == "__main__":
    main()
