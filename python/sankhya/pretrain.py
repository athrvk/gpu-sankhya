"""Masked-character pretraining for the SankhyaCNN trunk.

The tagger only ever sees text the generator wrote (plus a small verified LLM
corpus). Real sentences -- names, ethnonyms, measurements, years, the ordinary
prose amounts are embedded in -- are what it has never had any signal about,
and `data_wild/REPORT.md` shows the cost: 20% of real no-amount sentences get
a span.

This module gives the *trunk* (char embedding + conv stack, the same
`SankhyaCNN` the tagger uses) a cheap unsupervised look at that text first:
mask a fraction of the characters of a real sentence, put a temporary
per-character vocab head on the conv stack, and train it to reconstruct only
the masked positions. The head is thrown away; `sankhya.train --init-from`
loads just the trunk and re-initialises the BIO/class heads.

    python -m sankhya.pretrain --out /tmp/pretrain --n 500000 --epochs 8

The charset is the SHIPPED one (built from the `--lang` packs exactly as
`train.py` builds it), never rebuilt from the real text: a real character
outside it maps to `<unk>`, which is what production input does too. The mask
token is `<unk>` as well, so the embedding table keeps the exact shape the
tagger expects and `--init-from` is a straight tensor copy.

Wild-gold text (`tests/gold_wild_*.jsonl`) is excluded, so pretraining cannot
memorise the evaluation set.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

from . import classes as C
from .charset import build_charset, build_charset_multi, normalize_text
from .langs import base as langbase
from .model import SankhyaCNN, count_params, ARCHS

MAX_LEN = 128
DEFAULT_MASK_RATE = 0.15
IGNORE_INDEX = -100


class PretrainModel(nn.Module):
    """`SankhyaCNN` trunk + a temporary `channels -> vocab_size` head.

    The trunk is the real thing (same module, same state_dict keys), so its
    weights drop straight into a tagger via `train.py --init-from`. The
    tagger's own BIO/class heads exist here too but are never trained -- only
    `self.head` gets gradient from the reconstruction loss.
    """

    def __init__(self, vocab_size, arch=None, channels=48, embed_dim=16):
        super().__init__()
        self.trunk = SankhyaCNN(
            vocab_size=vocab_size, n_cls=len(C.CLASSES), arch=arch,
            channels=channels, embed_dim=embed_dim,
        )
        self.head = nn.Linear(channels, vocab_size)

    def features(self, chars):
        t = self.trunk
        x = t.embed(chars).transpose(1, 2)
        for layer, conv in zip(t.arch, t.convs):
            y = t.act(conv(x))
            if layer["residual"]:
                y = y + x
            x = y
        return x.transpose(1, 2)

    def forward(self, chars):
        return self.head(self.features(chars))

    def trunk_state_dict(self):
        """Only the tensors `train.py --init-from` consumes."""
        return {k: v for k, v in self.trunk.state_dict().items()
                if k.startswith("embed.") or k.startswith("conv")}


def apply_mask(chars, mask, mask_id, rate=DEFAULT_MASK_RATE, rng=None):
    """Mask `rate` of the real (unpadded) characters of each row.

    Returns `(masked_chars, targets)`. `targets` is the original id at every
    masked position and `IGNORE_INDEX` everywhere else, so a plain
    `CrossEntropyLoss(ignore_index=IGNORE_INDEX)` sees loss ONLY on masked
    positions -- padding and unmasked characters contribute nothing.
    """
    if rng is None:
        rng = torch.Generator()
        rng.manual_seed(0)
    real = mask > 0
    draw = torch.rand(chars.shape, generator=rng)
    chosen = real & (draw < rate)
    targets = torch.full_like(chars, IGNORE_INDEX)
    targets[chosen] = chars[chosen]
    out = chars.clone()
    out[chosen] = mask_id
    return out, targets


# --------------------------------------------------------------------------
# corpus
# --------------------------------------------------------------------------

def collect_sentences(n=500000, seed=23, langs=None, limit=None, verbose=True):
    """Up to `n` real sentences, balanced across languages as far as each
    corpus allows (water-filling: a language with fewer lines than its equal
    share gives the remainder to the ones that have more).

    Reuses `sankhya.wild`'s loaders and filters (3-40 words, <= 128 chars,
    deduplicated) and drops anything that appears in the wild gold.
    """
    from . import wild

    langs = list(langs or wild.LANGS)
    gold_keys = wild.gold_wild_keys(langs=langs)
    pools = {}
    for lang in langs:
        rng = random.Random(f"pretrain:{seed}:{lang}")
        seen = set()
        res = []
        n_seen = 0
        cap = n  # never keep more than the whole budget for one language
        for src in wild.sources_for(lang):
            for text, _score, _signals in wild.scan_source(src, limit=limit):
                key = wild._dedup_key(text)
                if not key or key in seen or key in gold_keys:
                    continue
                seen.add(key)
                n_seen += 1
                if len(res) < cap:
                    res.append(text)
                else:
                    j = rng.randrange(n_seen)
                    if j < cap:
                        res[j] = text
        rng.shuffle(res)
        pools[lang] = res
        if verbose:
            print(f"  [{lang}] {len(res)} usable lines")

    # water-fill: equal shares, shortfalls redistributed
    remaining = n
    quotas = {}
    for lang in sorted(langs, key=lambda l: len(pools[l])):
        share = remaining // max(1, (len(langs) - len(quotas)))
        quotas[lang] = min(share, len(pools[lang]))
        remaining -= quotas[lang]

    out = []
    for lang in langs:
        out.extend({"text": t, "lang": lang} for t in pools[lang][:quotas[lang]])
    random.Random(seed).shuffle(out)
    if verbose:
        print(f"  -> {len(out)} sentences " +
              " ".join(f"{l}={quotas[l]}" for l in langs))
    return out


def tensorize(texts, char_to_id, max_len=MAX_LEN):
    n = len(texts)
    chars = np.zeros((n, max_len), dtype=np.int64)
    mask = np.zeros((n, max_len), dtype=np.float32)
    unk = char_to_id.get("<unk>", 1)
    for i, t in enumerate(texts):
        s = normalize_text(t)[:max_len]
        for j, ch in enumerate(s):
            chars[i, j] = char_to_id.get(ch, unk)
        mask[i, :len(s)] = 1.0
    return torch.from_numpy(chars), torch.from_numpy(mask)


# --------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m sankhya.pretrain")
    ap.add_argument("--out", default="models/pretrain")
    ap.add_argument("--n", type=int, default=500000)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--mask-rate", type=float, default=DEFAULT_MASK_RATE)
    ap.add_argument("--lang", default="hi_latn,hi_deva,mr_deva,gu_gujr")
    ap.add_argument("--arch", default="v2", choices=list(ARCHS.keys()))
    ap.add_argument("--channels", type=int, default=48)
    ap.add_argument("--embed-dim", type=int, default=16)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=None, help="cap lines read per source")
    ap.add_argument("--cache", default=None,
                     help="jsonl of collected sentences; read if present, written if not")
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(4)
    os.makedirs(args.out, exist_ok=True)

    lang_ids = [x.strip() for x in args.lang.split(",") if x.strip()]
    packs = [langbase.get_pack(l) for l in lang_ids]
    vocab = build_charset_multi(packs) if len(packs) > 1 else build_charset(packs[0])
    char_to_id = {c: i for i, c in enumerate(vocab)}
    mask_id = char_to_id["<unk>"]
    print(f"charset: {len(vocab)} chars (mask token = <unk>, id {mask_id})")

    t0 = time.time()
    rows = None
    if args.cache and os.path.exists(args.cache):
        rows = [json.loads(l) for l in open(args.cache, encoding="utf-8") if l.strip()]
        print(f"loaded {len(rows)} cached sentences from {args.cache}")
    if rows is None:
        rows = collect_sentences(n=args.n, seed=args.seed, langs=lang_ids, limit=args.limit)
        if args.cache:
            os.makedirs(os.path.dirname(args.cache) or ".", exist_ok=True)
            with open(args.cache, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"wrote {args.cache}")
    print(f"corpus ready in {time.time()-t0:.1f}s")

    chars, cmask = tensorize([r["text"] for r in rows], char_to_id)
    n = chars.shape[0]
    print(f"tensorized {n} sentences")

    model = PretrainModel(vocab_size=len(vocab), arch=ARCHS[args.arch],
                          channels=args.channels, embed_dim=args.embed_dim)
    print(f"trunk params: {count_params(model.trunk)} "
          f"(+{count_params(model.head)} throwaway head)")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    steps = math.ceil(n / args.batch) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)
    ce = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    gen = torch.Generator()
    gen.manual_seed(args.seed)

    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n)
        tot, nb = 0.0, 0
        correct = masked_total = 0
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            cb, mb = chars[idx], cmask[idx]
            inp, tgt = apply_mask(cb, mb, mask_id, rate=args.mask_rate, rng=gen)
            logits = model(inp)
            V = logits.shape[-1]
            loss = ce(logits.reshape(-1, V), tgt.reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            sched.step()
            tot += loss.item()
            nb += 1
            with torch.no_grad():
                sel = tgt != IGNORE_INDEX
                if sel.any():
                    correct += (logits.argmax(-1)[sel] == tgt[sel]).sum().item()
                    masked_total += int(sel.sum().item())
        acc = correct / masked_total if masked_total else 0.0
        print(f"epoch {epoch+1}/{args.epochs} loss={tot/max(nb,1):.4f} "
              f"masked_char_acc={acc:.4f} time={time.time()-t0:.1f}s")

    ckpt_path = os.path.join(args.out, "pretrain.pt")
    torch.save({
        "state_dict": model.trunk_state_dict(),
        "vocab": vocab,
        "arch": model.trunk.arch,
        "channels": args.channels,
        "embed_dim": args.embed_dim,
        "seed": args.seed,
        "mask_rate": args.mask_rate,
        "n_sentences": n,
        "epochs": args.epochs,
        "pretrain": True,
    }, ckpt_path)
    print(f"saved trunk to {ckpt_path} ({time.time()-t0:.1f}s total)")


if __name__ == "__main__":
    main()
