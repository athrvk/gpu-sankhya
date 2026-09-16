"""Regenerate the JS test suite's parity fixtures from the exported model.

Writes two fixture files consumed by test/parity.test.ts and
test/e2e-parity.test.ts at the repo root:

  parity.jsonl  -- {text, bio, cls}: raw per-character argmax ids from the
                   int8 numpy forward pass (padding-corrected per np_infer's
                   PAD_TAIL scheme), for pinning the JS forward pass's
                   arithmetic to this reference bit-for-bit.

  decoded.jsonl -- {text, spans}: decode_spans() + core.evaluate() output,
                   normalised to {start, end, value, range, unit, currency,
                   classes}, for pinning end-to-end parse() behaviour.

Deterministic given the same weights JSON and gold set -- no RNG involved.

Usage (from python/):
    python -m sankhya.make_fixtures \
        --weights ../src/data/default-weights.json \
        --gold tests/gold.jsonl \
        --out-parity ../test/fixtures/parity.jsonl \
        --out-decoded ../test/fixtures/decoded.jsonl
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from . import classes as C
from . import core
from .decode import decode_spans
from .langs.base import get_pack
from .train import MAX_LEN, build_char_to_id, load_jsonl
from . import np_infer
from .charset import normalize_text


def run(weights_path: str, examples: list, lang: str = "hi_latn"):
    with open(weights_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    vocab = obj["charset"]
    class_names = obj["classes"]
    layers = obj.get("layers", 3)
    weights = np_infer.load_weights_int8_json(obj)
    char_to_id = build_char_to_id(vocab)
    unk = char_to_id.get("<unk>", 1)
    pack = get_pack(lang)

    parity_rows = []
    decoded_rows = []

    for ex in examples:
        text = normalize_text(ex["text"])[:MAX_LEN]
        L = len(text)
        ids = np_infer.pad_ids(np.array([char_to_id.get(c, unk) for c in text], dtype=np.int64))
        bio_logits, cls_logits = np_infer.forward(weights, ids, dilation=2, layers=layers)
        bio_logits, cls_logits = bio_logits[:L], cls_logits[:L]
        bio_pred = bio_logits.argmax(-1).tolist()
        cls_pred = cls_logits.argmax(-1).tolist()
        bio_probs = np_infer.softmax(bio_logits, axis=-1).tolist()

        parity_rows.append({"text": ex["text"], "bio": bio_pred, "cls": cls_pred})

        decoded = decode_spans(text, bio_pred, cls_pred, bio_probs=bio_probs)
        spans = []
        for d in decoded:
            toks = [(class_names[cid], sub) for cid, sub in d["tokens"]]
            res = core.evaluate(toks)
            currency = core.detect_currency(text, d["start"], d["end"], pack)
            spans.append({
                "start": d["start"],
                "end": d["end"],
                "value": res.value,
                "range": list(res.range) if res.range else None,
                "unit": res.unit,
                "currency": currency,
                "classes": " ".join(res.classes),
            })
        decoded_rows.append({"text": ex["text"], "spans": spans})

    return parity_rows, decoded_rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="../src/data/default-weights.json")
    ap.add_argument("--gold", default="tests/gold.jsonl")
    ap.add_argument("--out-parity", default="../test/fixtures/parity.jsonl")
    ap.add_argument("--out-decoded", default="../test/fixtures/decoded.jsonl")
    ap.add_argument("--lang", default="hi_latn")
    args = ap.parse_args(argv)

    examples = load_jsonl(args.gold)
    parity_rows, decoded_rows = run(args.weights, examples, lang=args.lang)

    with open(args.out_parity, "w", encoding="utf-8") as f:
        for row in parity_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(args.out_decoded, "w", encoding="utf-8") as f:
        for row in decoded_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote {len(parity_rows)} rows to {args.out_parity}")
    print(f"wrote {len(decoded_rows)} rows to {args.out_decoded}")


if __name__ == "__main__":
    main()
