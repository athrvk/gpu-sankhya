"""Run eval_gold across several models (torch / float JSON / int8 JSON) and
gold files, print a markdown comparison table, and compare metrics-json runs
across seeds/configs for variance.

    python -m sankhya.eval_matrix run --models v1=models/sankhya.pt v2=models2/sankhya.pt \
        --gold tests/gold.jsonl tests/gold_deva.jsonl --json-out output/matrix_eval.json

    python -m sankhya.eval_matrix compare --runs run1.json run2.json run3.json \
        --group-key config

`--gold` defaults to every `tests/gold*.jsonl` file found, deriving the
language label from the filename (`gold_<lang>.jsonl`, or `gold.jsonl` ->
hi_latn) purely for display -- a future `tests/gold_mr_deva.jsonl` is picked
up automatically with no code changes.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as stats

from . import eval_gold


def _default_gold_files():
    here = os.path.dirname(os.path.abspath(__file__))
    py_root = os.path.dirname(here)
    files = sorted(glob.glob(os.path.join(py_root, "tests", "gold*.jsonl")))
    if not files:
        files = sorted(glob.glob("tests/gold*.jsonl"))
    return files


def _lang_label(path):
    base = os.path.basename(path)
    name = base[:-len(".jsonl")] if base.endswith(".jsonl") else base
    if name == "gold":
        return "hi_latn"
    if name.startswith("gold_"):
        return name[len("gold_"):]
    return name


def _model_kind(path):
    if path.endswith(".int8.json"):
        return "json_int8"
    if path.endswith(".json"):
        return "json_float32"
    if path.endswith(".pt"):
        return "torch"
    raise SystemExit(f"cannot infer model kind from path (expected .pt/.json/.int8.json): {path}")


def _parse_models(items):
    models = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--models expects NAME=PATH, got: {item!r}")
        name, path = item.split("=", 1)
        models[name.strip()] = path.strip()
    return models


def run_matrix(models, gold_files, quiet=True):
    """Run eval_gold's evaluation for each named model, in-process.

    Returns {model_name: {"path":..., "kind":..., "metrics": <eval_gold metrics dict>}}
    """
    results = {}
    for name, path in models.items():
        kind = _model_kind(path)
        if kind == "torch":
            metrics = eval_gold.run_eval(gold_files, ckpt=path, quiet=quiet)
        elif kind == "json_float32":
            metrics = eval_gold.run_eval(gold_files, weights_json=path, int8=False, quiet=quiet)
        else:
            metrics = eval_gold.run_eval(gold_files, weights_json=path, int8=True, quiet=quiet)
        results[name] = {"path": path, "kind": kind, "metrics": metrics}
    return results


def _fmt(x, digits=4):
    return f"{x:.{digits}f}" if isinstance(x, float) else str(x)


def render_main_table(results, gold_files):
    labels = [_lang_label(p) for p in gold_files]
    header = ["model"] + [f"{lbl} value_acc" for lbl in labels] + [f"{lbl} f1" for lbl in labels] + \
        ["combined value_acc", "combined f1", "negatives fp_rate"]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for name, r in results.items():
        m = r["metrics"]
        per_file = m.get("per_file", {})
        row = [f"{name} ({r['kind']})"]
        for path in gold_files:
            pf = per_file.get(path)
            row.append(_fmt(pf["value_acc"]) if pf else "-")
        for path in gold_files:
            pf = per_file.get(path)
            row.append(_fmt(pf["f1"]) if pf else "-")
        combined = m["combined"]
        row.append(_fmt(combined["value_acc"]))
        row.append(_fmt(combined["f1"]))
        row.append(_fmt(combined.get("negatives", {}).get("fp_rate", 0.0)))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def render_category_table(results):
    all_cats = list(eval_gold.CATEGORY_NAMES)
    header = ["model"] + all_cats
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for name, r in results.items():
        cats = r["metrics"]["combined"].get("categories", {})
        row = [f"{name} ({r['kind']})"]
        for c in all_cats:
            st = cats.get(c)
            row.append(_fmt(st["value_acc"]) if st else "-")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def cmd_run(args):
    models = _parse_models(args.models)
    if not models:
        raise SystemExit("--models NAME=PATH required (at least one)")
    gold_files = args.gold or _default_gold_files()
    if not gold_files:
        raise SystemExit("no gold files found; pass --gold explicitly")

    results = run_matrix(models, gold_files, quiet=not args.verbose)

    main_table = render_main_table(results, gold_files)
    cat_table = render_category_table(results)
    print("\n" + main_table)
    print("\nper-category value_acc:\n")
    print(cat_table)

    if args.json_out:
        out = {
            "gold_files": gold_files,
            "results": results,
        }
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {args.json_out}")


def _extract_group_key(run, group_key):
    """Pull a group label out of one metrics-json blob. Supports a plain
    top-level field name, or dotted path like config.arch."""
    cur = run
    for part in group_key.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _find_nested(run, *path):
    cur = run
    for p in path:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return None
    return cur


def cmd_compare(args):
    runs = []
    for path in args.runs:
        with open(path, "r", encoding="utf-8") as f:
            runs.append((path, json.load(f)))

    groups = {}
    for path, run in runs:
        key = _extract_group_key(run, args.group_key)
        if key is None:
            key = f"<no {args.group_key}>:{path}"
        groups.setdefault(key, []).append((path, run))

    def mean_std(values):
        values = [v for v in values if v is not None]
        if not values:
            return None, None
        if len(values) == 1:
            return values[0], 0.0
        return stats.mean(values), stats.stdev(values)

    print(f"{'group':<30}{'n':>3}  {'int8 value_acc':>20}  {'int8 f1':>20}")
    per_file_labels = set()
    rows = []
    for key, items in groups.items():
        int8_va = []
        int8_f1 = []
        per_file_va = {}
        for path, run in items:
            # accept eval_gold's own json-out shape ({"combined": {...}}),
            # eval_matrix's run-mode shape ({"results": {name: {"metrics": {...}}}}),
            # or a kaggle matrix.json entry ({"gold": {"json_int8": {"combined": {...}}}}).
            combined = _find_nested(run, "combined")
            per_file = _find_nested(run, "per_file") or {}
            if combined is None:
                gold = _find_nested(run, "gold", "json_int8")
                if gold:
                    combined = gold.get("combined")
                    per_file = gold.get("per_file", {})
            if combined is None:
                results = _find_nested(run, "results")
                if results:
                    for _, r in results.items():
                        if r.get("kind") == "json_int8":
                            combined = r["metrics"]["combined"]
                            per_file = r["metrics"].get("per_file", {})
                            break
            if combined is None:
                continue
            int8_va.append(combined.get("value_acc"))
            int8_f1.append(combined.get("f1"))
            for fpath, pf in per_file.items():
                lbl = eval_matrix_lang_label(fpath)
                per_file_labels.add(lbl)
                per_file_va.setdefault(lbl, []).append(pf.get("value_acc"))

        m_va, s_va = mean_std(int8_va)
        m_f1, s_f1 = mean_std(int8_f1)
        va_str = f"{m_va:.4f}+-{s_va:.4f}" if m_va is not None else "-"
        f1_str = f"{m_f1:.4f}+-{s_f1:.4f}" if m_f1 is not None else "-"
        print(f"{str(key):<30}{len(items):>3}  {va_str:>20}  {f1_str:>20}")
        rows.append((key, len(items), per_file_va))

    if per_file_labels:
        print("\nper-file value_acc (mean +- std):")
        for key, n, per_file_va in rows:
            parts = []
            for lbl in sorted(per_file_labels):
                vals = per_file_va.get(lbl)
                if vals:
                    m, s = mean_std(vals)
                    parts.append(f"{lbl}={m:.4f}+-{s:.4f}")
            print(f"  {key}: " + ", ".join(parts))


def eval_matrix_lang_label(path):
    return _lang_label(path)


def select_matrix_winner(entries):
    """Pick the winning matrix entry from a list of dicts, each with at
    least `arch`, `channels`, `seed`, `val_value_acc`, `gold_int8_value_acc`,
    and optionally `gold_int8_f1` (default 0.0) and `negatives_fp_rate`
    (default 0.0), used only to break near-ties.

    This is the single canonical implementation of the matrix winner rule;
    `kaggle_train/train_kernel.py`'s `select_winner` imports and wraps this
    function (with a richer `results` shape) rather than duplicating the
    logic, so there is exactly one place the rule is defined.

    Selection:
      1. Winning **config** (arch:channels) = highest mean
         `gold_int8_value_acc` across its seeds. Ties broken by first
         config encountered, in input order.
      2. Within that config, the winning **seed** is ranked by
         `val_value_acc` ROUNDED to 2 decimals (so seeds that are
         practically tied on val accuracy, e.g. 0.9343 vs 0.9338 vs 0.9346,
         are treated as equal there); ties broken by higher
         `gold_int8_f1`; further ties broken by lower `negatives_fp_rate`;
         remaining ties broken by first entry encountered, in input order.

    Returns (winner_entry, config_label) where config_label is "arch:channels".
    """
    if not entries:
        raise ValueError("no matrix entries to select a winner from")

    config_order = []
    by_config = {}
    for e in entries:
        key = (e["arch"], e["channels"])
        if key not in by_config:
            by_config[key] = []
            config_order.append(key)
        by_config[key].append(e)

    best_key = None
    best_mean = None
    for key in config_order:
        vals = [e["gold_int8_value_acc"] for e in by_config[key]]
        m = sum(vals) / len(vals)
        if best_mean is None or m > best_mean:
            best_mean = m
            best_key = key

    winners = by_config[best_key]

    def rank(e):
        # sort DESCENDING on (rounded val_value_acc, gold_int8_f1) and
        # ASCENDING on negatives_fp_rate -- negate the ascending term.
        return (
            round(e["val_value_acc"], 2),
            e.get("gold_int8_f1", 0.0),
            -e.get("negatives_fp_rate", 0.0),
        )

    best_entry = None
    best_rank = None
    for e in winners:
        r = rank(e)
        if best_rank is None or r > best_rank:
            best_rank = r
            best_entry = e

    config_label = f"{best_key[0]}:{best_key[1]}"
    return best_entry, config_label


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m sankhya.eval_matrix")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="evaluate several models against gold, print a comparison table")
    p_run.add_argument("--models", nargs="+", required=True, help="NAME=PATH (.pt / .json / .int8.json), repeatable")
    p_run.add_argument("--gold", nargs="+", default=None, help="gold jsonl files (default: every tests/gold*.jsonl)")
    p_run.add_argument("--json-out", default=None, help="write full results as JSON to this path")
    p_run.add_argument("--verbose", action="store_true", help="don't suppress eval_gold's miss lists")
    p_run.set_defaults(func=cmd_run)

    p_cmp = sub.add_parser("compare", help="compare metrics-json runs (seed variance etc)")
    p_cmp.add_argument("--runs", nargs="+", required=True, help="metrics JSON files (eval_gold/eval_matrix/kaggle matrix.json entries)")
    p_cmp.add_argument("--group-key", default="config", help="dotted field name to group runs by (default: config)")
    p_cmp.set_defaults(func=cmd_compare)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
