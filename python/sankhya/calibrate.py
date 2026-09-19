"""Measure and calibrate span confidence.

Runs the torch checkpoint over freshly generated synthetic examples (the
same generator/mix used for training, at a seed training never saw) plus
the hand-written gold sets, decodes with exactly the gates `eval_gold`
applies, and records for every PREDICTED span:

  raw confidence | verified | value correct | spurious

From that it prints a reliability table (10 raw-confidence bins x count x
empirical precision, split verified/unverified, for synthetic and for
gold) and fits a monotone isotonic (PAV) map from raw confidence to
empirical precision on the SYNTHETIC spans -- gold is the held-out check,
never fitted on. The map is exported as `src/data/calibration.json` for
both runtimes.

    python -m sankhya.calibrate --n 100000 --seed 13 \
        --mix 0.32,0.26,0.21,0.21 --cross 0.10 \
        --gold tests/gold.jsonl tests/gold_deva.jsonl tests/gold_mr.jsonl tests/gold_gu.jsonl
"""
from __future__ import annotations

import argparse
import json

from .calibration import fit_breakpoints
from .eval_gold import collect_confidence_records, confidence_curve
from .generator import generate, generate_multi
from .langs import base as langs_base
from .langs.base import load_all

load_all()
from .proptest import load_model, run_torch_batched  # noqa: E402
from .train import load_jsonl  # noqa: E402

N_BINS = 10
DEFAULT_MIX = "0.32,0.26,0.21,0.21"


def reliability_table(rows, n_bins=N_BINS):
    """[(lo, hi, count, precision, spurious_rate), ...] over raw-confidence bins."""
    out = []
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        if b == n_bins - 1:
            sel = [r for r in rows if lo <= r["raw"] <= hi]
        else:
            sel = [r for r in rows if lo <= r["raw"] < hi]
        n = len(sel)
        prec = sum(r["correct"] for r in sel) / n if n else 0.0
        spur = sum(r["spurious"] for r in sel) / n if n else 0.0
        out.append((lo, hi, n, prec, spur))
    return out


def print_reliability(rows, header):
    print(f"\n=== {header} ===")
    groups = [
        ("all", rows),
        ("verified", [r for r in rows if r["verified"]]),
        ("unverified", [r for r in rows if not r["verified"]]),
    ]
    for name, sub in groups:
        n = len(sub)
        prec = sum(r["correct"] for r in sub) / n if n else 0.0
        print(f"\n{name}: {n} spans, precision {prec:.4f}")
        print(f"  {'raw bin':<14}{'count':>9}{'precision':>12}{'spurious':>10}")
        for lo, hi, cnt, p, spur in reliability_table(sub):
            if cnt == 0:
                print(f"  [{lo:.1f}, {hi:.1f}){'':>9}{0:>9}{'--':>12}{'--':>10}")
            else:
                print(f"  [{lo:.1f}, {hi:.1f}){'':>9}{cnt:>9}{p:>12.4f}{spur:>10.4f}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m sankhya.calibrate")
    ap.add_argument("--n", type=int, default=100000, help="synthetic examples to generate")
    ap.add_argument("--seed", type=int, default=13, help="generator seed (must differ from training)")
    ap.add_argument("--ckpt", default="../models/default/sankhya.pt")
    ap.add_argument("--lang", default=",".join(langs_base.KNOWN_PACKS))
    ap.add_argument("--mix", default=DEFAULT_MIX, help="comma-separated weights matching --lang")
    ap.add_argument("--cross", type=float, default=0.10)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--gold", nargs="+", default=[
        "tests/gold.jsonl", "tests/gold_deva.jsonl", "tests/gold_mr.jsonl", "tests/gold_gu.jsonl",
    ])
    ap.add_argument("--out", default="../src/data/calibration.json")
    ap.add_argument("--max-points", type=int, default=32)
    ap.add_argument("--no-write", action="store_true", help="print the tables, do not write the JSON")
    ap.add_argument("--json-out", default=None, help="also dump the measured tables as JSON here")
    args = ap.parse_args(argv)

    lang_ids = [x.strip() for x in args.lang.split(",") if x.strip()]
    packs = [langs_base.get_pack(l) for l in lang_ids]
    weights = [float(x) for x in args.mix.split(",")] if args.mix else None
    if weights is not None:
        assert len(weights) == len(packs), "--mix must match --lang count"

    print(f"generating {args.n} synthetic examples (seed {args.seed}, packs {','.join(lang_ids)}, "
          f"mix {args.mix}, cross {args.cross}) ...")
    if len(packs) == 1:
        syn_examples = generate(packs[0], args.n, seed=args.seed)
    else:
        syn_examples = generate_multi(packs, weights=weights, n=args.n, seed=args.seed, cross=args.cross)

    model, vocab, classes = load_model(args.ckpt)

    print("running the checkpoint over the synthetic set ...")
    syn_preds = run_torch_batched(model, vocab, syn_examples, batch_size=args.batch_size)
    syn_rows = collect_confidence_records(syn_examples, syn_preds, classes)

    gold_examples = []
    for path in args.gold:
        gold_examples.extend(load_jsonl(path))
    print(f"running the checkpoint over {len(gold_examples)} gold examples ...")
    gold_preds = run_torch_batched(model, vocab, gold_examples, batch_size=args.batch_size)
    gold_rows = collect_confidence_records(gold_examples, gold_preds, classes)
    n_gold_spans = sum(len(ex["spans"]) for ex in gold_examples)

    print_reliability(syn_rows, f"synthetic (n={args.n}, seed={args.seed})")
    print_reliability(gold_rows, f"gold ({len(gold_examples)} examples, {n_gold_spans} spans)")

    knots_x, knots_y = fit_breakpoints(
        [r["raw"] for r in syn_rows], [1.0 if r["correct"] else 0.0 for r in syn_rows],
        max_points=args.max_points,
    )
    print(f"\nisotonic fit: {len(knots_x)} breakpoints")
    print(f"  {'raw':>10}{'calibrated':>14}")
    for x, y in zip(knots_x, knots_y):
        print(f"  {x:>10.6f}{y:>14.6f}")

    print("\ngold confidence curve (calibrated threshold -> coverage / value accuracy):")
    print(f"  {'minConfidence':<16}{'kept':>8}{'coverage':>11}{'value_acc':>11}")
    curve = confidence_curve(gold_rows, n_gold_spans, knots_x=knots_x, knots_y=knots_y)
    for c in curve:
        print(f"  {c['min_confidence']:<16.2f}{c['kept']:>8}{c['coverage']:>11.4f}{c['value_acc']:>11.4f}")

    obj = {
        "version": 1,
        "raw": [round(float(x), 6) for x in knots_x],
        "calibrated": [round(float(y), 6) for y in knots_y],
        "n_samples": len(syn_rows),
        "n_examples": args.n,
        "seed": args.seed,
        "mix": args.mix,
        "cross": args.cross,
        "langs": lang_ids,
    }
    if not args.no_write:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
            f.write("\n")
        print(f"\nwrote {args.out} ({len(knots_x)} breakpoints from {len(syn_rows)} synthetic spans)")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({
                "calibration": obj,
                "synthetic": reliability_table(syn_rows),
                "gold": reliability_table(gold_rows),
                "gold_curve": curve,
            }, f, indent=2)
        print(f"wrote {args.json_out}")

    return obj


if __name__ == "__main__":
    main()
