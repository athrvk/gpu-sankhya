"""Train SankhyaCNN on generated data."""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

from . import classes as C
from .charset import build_charset, build_charset_multi, normalize_text
from .langs import base as langbase
from .model import SankhyaCNN, count_params, ARCHS
from .decode import decode_spans
from . import core

MAX_LEN = 128


def load_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def build_char_to_id(vocab):
    return {c: i for i, c in enumerate(vocab)}


def tensorize(examples, char_to_id, max_len=MAX_LEN):
    n = len(examples)
    chars = np.zeros((n, max_len), dtype=np.int64)
    bio = np.zeros((n, max_len), dtype=np.int64)
    cls = np.zeros((n, max_len), dtype=np.int64)
    mask = np.zeros((n, max_len), dtype=np.float32)
    unk = char_to_id.get("<unk>", 1)
    for i, ex in enumerate(examples):
        text = normalize_text(ex["text"])
        L = min(len(text), max_len)
        for j in range(L):
            chars[i, j] = char_to_id.get(text[j], unk)
        bio_ex = ex["bio"][:L]
        cls_ex = ex["cls"][:L]
        bio[i, :L] = bio_ex
        cls[i, :L] = cls_ex
        mask[i, :L] = 1.0
    return (
        torch.from_numpy(chars),
        torch.from_numpy(bio),
        torch.from_numpy(cls),
        torch.from_numpy(mask),
    )


def span_f1(gold_spans_list, pred_spans_list):
    tp = fp = fn = 0
    for gold, pred in zip(gold_spans_list, pred_spans_list):
        gold_set = {(s, e) for s, e in gold}
        pred_set = {(s, e) for s, e in pred}
        tp += len(gold_set & pred_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def value_accuracy(examples, all_bio_pred, all_cls_pred, all_bio_probs):
    """For each gold span, find exact-matching predicted span and check evaluate() value match."""
    correct = 0
    total = 0
    misses = []
    for ex, bio_pred, cls_pred, bio_probs in zip(examples, all_bio_pred, all_cls_pred, all_bio_probs):
        text = ex["text"]
        L = min(len(text), MAX_LEN)
        decoded = decode_spans(text, bio_pred[:L].tolist(), cls_pred[:L].tolist(), bio_probs=bio_probs[:L].tolist())
        pred_by_span = {(d["start"], d["end"]): d for d in decoded}
        for sp in ex["spans"]:
            total += 1
            key = (sp["start"], sp["end"])
            gold_result = core.evaluate([(t, "") for t in sp["classes"].split()]) if False else None
            gold_value = sp["value"]
            ok = False
            if key in pred_by_span:
                toks = [(C.CLASSES[cid], sub) for cid, sub in pred_by_span[key]["tokens"]]
                res = core.evaluate(toks)
                if res.value == gold_value:
                    if sp.get("range"):
                        if res.range == tuple(sp["range"]) or (res.range and list(res.range) == sp["range"]):
                            ok = True
                    else:
                        ok = True
            if ok:
                correct += 1
            else:
                misses.append((text, sp))
    acc = correct / total if total else 0.0
    return acc, misses


@torch.no_grad()
def evaluate_model(model, chars, bio, cls, mask, examples, batch=256, device="cpu"):
    model.eval()
    n = chars.shape[0]
    all_bio_pred = []
    all_cls_pred = []
    all_bio_probs = []
    bio_correct = bio_total = 0
    cls_correct = cls_total = 0
    for i in range(0, n, batch):
        cb = chars[i:i + batch].to(device)
        mb = mask[i:i + batch].to(device)
        bb = bio[i:i + batch].to(device)
        clb = cls[i:i + batch].to(device)
        bio_logits, cls_logits = model(cb)
        bio_pred = bio_logits.argmax(-1)
        cls_pred = cls_logits.argmax(-1)
        bio_probs = torch.softmax(bio_logits, dim=-1)
        m = mb.bool()
        bio_correct += ((bio_pred == bb) & m).sum().item()
        bio_total += m.sum().item()
        cls_correct += ((cls_pred == clb) & m).sum().item()
        cls_total += m.sum().item()
        for j in range(cb.shape[0]):
            all_bio_pred.append(bio_pred[j].cpu().numpy())
            all_cls_pred.append(cls_pred[j].cpu().numpy())
            all_bio_probs.append(bio_probs[j].cpu().numpy())
    bio_acc = bio_correct / bio_total if bio_total else 0.0
    cls_acc = cls_correct / cls_total if cls_total else 0.0

    gold_spans_list = [[(s["start"], s["end"]) for s in ex["spans"]] for ex in examples]
    pred_spans_list = []
    for ex, bp, cp, bpr in zip(examples, all_bio_pred, all_cls_pred, all_bio_probs):
        L = min(len(ex["text"]), MAX_LEN)
        decoded = decode_spans(ex["text"], bp[:L].tolist(), cp[:L].tolist(), bio_probs=bpr[:L].tolist())
        pred_spans_list.append([(d["start"], d["end"]) for d in decoded])
    prec, rec, f1 = span_f1(gold_spans_list, pred_spans_list)

    vacc, misses = value_accuracy(examples, all_bio_pred, all_cls_pred, all_bio_probs)
    return {
        "bio_acc": bio_acc,
        "cls_acc": cls_acc,
        "span_prec": prec,
        "span_rec": rec,
        "span_f1": f1,
        "value_acc": vacc,
        "misses": misses,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--val", default="data/val.jsonl")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--out", default="models/")
    ap.add_argument("--lang", default="hi_latn", help="comma-separated pack ids, e.g. hi_latn,hi_deva")
    ap.add_argument("--dilation", type=int, default=1)
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--layers", type=int, default=3, choices=[3, 4])
    ap.add_argument("--arch", default=None, choices=list(ARCHS.keys()),
                     help="named conv-stack preset; overrides --layers/--dilation when given")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--embed-dim", type=int, default=16)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--num-train", type=int, default=None, help="subsample training set to this many examples")
    ap.add_argument("--device", default="cpu", choices=["cpu", "auto"],
                     help="'cpu' (default, unchanged behaviour) or 'auto' to use CUDA when available")
    ap.add_argument("--extra", nargs="+", default=None,
                     help="extra jsonl file(s) (e.g. verified LLM corpus) mixed into training at --extra-ratio")
    ap.add_argument("--extra-ratio", type=float, default=0.2,
                     help="target share of each epoch's examples drawn from --extra (0 disables mixing); "
                          "--extra is oversampled (with replacement) or subsampled each epoch to hit it")
    args = ap.parse_args(argv)

    import random
    torch.manual_seed(args.seed)
    # Reproducible GPU runs: cuDNN's default autotuned conv kernels are
    # nondeterministic, which made the same seed land ~1 gold point apart
    # across Kaggle runs. The model is tiny, so the speed cost is nil.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    t0 = time.time()
    lang_ids = [x.strip() for x in args.lang.split(",") if x.strip()]
    packs = [langbase.get_pack(l) for l in lang_ids]
    vocab = build_charset_multi(packs) if len(packs) > 1 else build_charset(packs[0])
    char_to_id = build_char_to_id(vocab)

    train_ex = load_jsonl(args.train)
    val_ex = load_jsonl(args.val)
    if args.num_train is not None and args.num_train < len(train_ex):
        rng = np.random.RandomState(0)
        idx = rng.choice(len(train_ex), size=args.num_train, replace=False)
        train_ex = [train_ex[i] for i in idx]
    print(f"loaded {len(train_ex)} train, {len(val_ex)} val examples in {time.time()-t0:.1f}s")

    extra_ex = []
    if args.extra:
        for p in args.extra:
            extra_ex.extend(load_jsonl(p))
        print(f"loaded {len(extra_ex)} extra examples from {len(args.extra)} file(s)")
        if extra_ex:
            unk_chars = Counter()
            for ex in extra_ex:
                text = normalize_text(ex["text"])
                for ch in text:
                    if ch not in char_to_id:
                        unk_chars[ch] += 1
            if unk_chars:
                total_unk = sum(unk_chars.values())
                top = unk_chars.most_common(20)
                print(
                    f"WARNING: --extra corpus has {total_unk} chars outside the "
                    f"{len(vocab)}-char pack charset ({len(unk_chars)} distinct); "
                    f"they map to <unk>. Top 20: " + " ".join(f"{c!r}={n}" for c, n in top)
                )

    t0 = time.time()
    tr_chars, tr_bio, tr_cls, tr_mask = tensorize(train_ex, char_to_id)
    va_chars, va_bio, va_cls, va_mask = tensorize(val_ex, char_to_id)
    ex_chars = ex_bio = ex_cls = ex_mask = None
    if extra_ex:
        ex_chars, ex_bio, ex_cls, ex_mask = tensorize(extra_ex, char_to_id)
    print(f"tensorized in {time.time()-t0:.1f}s")

    if args.device == "auto" and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    torch.set_num_threads(4)
    print(f"device={device}")

    arch = ARCHS[args.arch] if args.arch else None
    model = SankhyaCNN(
        vocab_size=len(vocab), n_cls=len(C.CLASSES), arch=arch, dilation=args.dilation,
        channels=args.channels, layers=args.layers, embed_dim=args.embed_dim,
    ).to(device)
    n_params = count_params(model)
    print(f"param count: {n_params} (arch={args.arch or 'legacy'} channels={args.channels} layers={len(model.arch)})")

    # --extra mixing: each epoch, n_main train examples are combined with a
    # freshly-drawn sample of n_extra_per_epoch extra examples such that the
    # extra share of the COMBINED epoch is approximately --extra-ratio
    # (n_extra/(n_main+n_extra) ~= extra_ratio). The extra pool is sampled
    # WITHOUT replacement when it is at least as large as what's needed that
    # epoch, and WITH replacement (oversampled) otherwise; either way the
    # sample is re-drawn every epoch so a small extra pool isn't just
    # replayed identically each time.
    n_main = tr_chars.shape[0]
    n_extra_pool = ex_chars.shape[0] if ex_chars is not None else 0
    n_extra_per_epoch = 0
    if n_extra_pool and args.extra_ratio > 0:
        ratio = min(max(args.extra_ratio, 0.0), 0.95)
        n_extra_per_epoch = int(round(n_main * ratio / (1 - ratio)))
        print(
            f"mixing in {n_extra_per_epoch} extra examples/epoch from a pool of "
            f"{n_extra_pool} (target ratio={ratio:.2f}, "
            f"{'oversampled' if n_extra_per_epoch > n_extra_pool else 'subsampled'})"
        )
    n = n_main + n_extra_per_epoch

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_batches_per_epoch = math.ceil(n / args.batch)
    total_steps = n_batches_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total_steps)

    # B (class 1) is rare relative to O/I; upweight it 2x in the bio loss.
    bio_class_weight = torch.tensor([1.0, 2.0, 1.0], dtype=torch.float32, device=device)
    ce_bio = nn.CrossEntropyLoss(reduction="none", weight=bio_class_weight, label_smoothing=args.label_smoothing)
    ce_cls = nn.CrossEntropyLoss(reduction="none", label_smoothing=args.label_smoothing)

    best_vacc = -1.0
    best_state = None

    train_t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        if n_extra_per_epoch:
            replace = n_extra_per_epoch > n_extra_pool
            extra_idx = np.random.choice(n_extra_pool, size=n_extra_per_epoch, replace=replace)
            extra_idx = torch.from_numpy(extra_idx)
            ep_chars = torch.cat([tr_chars, ex_chars[extra_idx]], dim=0)
            ep_bio = torch.cat([tr_bio, ex_bio[extra_idx]], dim=0)
            ep_cls = torch.cat([tr_cls, ex_cls[extra_idx]], dim=0)
            ep_mask = torch.cat([tr_mask, ex_mask[extra_idx]], dim=0)
        else:
            ep_chars, ep_bio, ep_cls, ep_mask = tr_chars, tr_bio, tr_cls, tr_mask
        perm = torch.randperm(n)
        epoch_loss = 0.0
        nb = 0
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            cb = ep_chars[idx].to(device)
            bb = ep_bio[idx].to(device)
            clb = ep_cls[idx].to(device)
            mb = ep_mask[idx].to(device)

            bio_logits, cls_logits = model(cb)
            B, L, _ = bio_logits.shape
            loss_bio = ce_bio(bio_logits.reshape(B * L, -1), bb.reshape(B * L))
            loss_cls = ce_cls(cls_logits.reshape(B * L, -1), clb.reshape(B * L))
            m = mb.reshape(B * L)
            loss = ((loss_bio + loss_cls) * m).sum() / m.sum().clamp_min(1)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            sched.step()
            epoch_loss += loss.item()
            nb += 1

        metrics = evaluate_model(model, va_chars, va_bio, va_cls, va_mask, val_ex, device=device)
        print(
            f"epoch {epoch+1}/{args.epochs} loss={epoch_loss/nb:.4f} "
            f"bio_acc={metrics['bio_acc']:.4f} cls_acc={metrics['cls_acc']:.4f} "
            f"span_prec={metrics['span_prec']:.4f} span_rec={metrics['span_rec']:.4f} "
            f"span_f1={metrics['span_f1']:.4f} value_acc={metrics['value_acc']:.4f} "
            f"time={time.time()-train_t0:.1f}s"
        )
        if metrics["value_acc"] > best_vacc:
            best_vacc = metrics["value_acc"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_metrics = metrics

    train_time = time.time() - train_t0
    print(f"total training time: {train_time:.1f}s")
    print(f"best val value_acc={best_vacc:.4f} span_f1={best_metrics['span_f1']:.4f}")

    model.load_state_dict(best_state)
    ckpt_path = os.path.join(args.out, "sankhya.pt")
    torch.save({
        "state_dict": model.state_dict(),
        "vocab": vocab,
        "classes": C.CLASSES,
        "dilation": args.dilation,
        "channels": args.channels,
        "layers": args.layers,
        "arch": model.arch,
        "embed_dim": args.embed_dim,
        "seed": args.seed,
        "val_metrics": {k: v for k, v in best_metrics.items() if k != "misses"},
    }, ckpt_path)
    print(f"saved checkpoint to {ckpt_path}")

    with open(os.path.join(args.out, "charset.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out, "classes.json"), "w", encoding="utf-8") as f:
        json.dump(C.CLASSES, f, ensure_ascii=False, indent=2)

    print(f"sample misses ({min(10, len(best_metrics['misses']))} of {len(best_metrics['misses'])}):")
    for text, sp in best_metrics["misses"][:10]:
        print(f"  {text!r} expected={sp}")


if __name__ == "__main__":
    main()
