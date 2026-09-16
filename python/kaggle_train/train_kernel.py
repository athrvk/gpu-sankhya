"""Self-contained Kaggle "script" kernel: clone gpu-sankhya, generate data
ONCE, then train/export/eval a MATRIX of (arch, channels, seed) configs,
picking a winner and staging everything `kaggle kernels output` downloads
into /kaggle/working/output/.

This file is uploaded as-is to Kaggle (code_file in kernel-metadata.json).
It has no dependency on the rest of this repo at import time -- it clones
its own copy of the repo at runtime and does all real work via subprocess
calls into `python -m sankhya.*`, so it works whether Kaggle checks out
this file alone or the whole `python/kaggle_train/` folder.

Config is read from environment variables, with the defaults below. `run.py
push` rewrites the literals inside the "CONFIG" marker block (only) when a
`--set KEY=VALUE` override is given, so this file stays the single
code_file Kaggle kernels allow while still being overridable from the CLI
wrapper without an out-of-band config file.

MATRIX (new): a comma-separated list of `arch:channels:seed` entries, e.g.
`v1:32:0,v2:32:0,v2:32:1`. For each entry the kernel runs
`sankhya.train --arch A --channels C --seed S --device auto --out
models_<A>_<C>_s<S>/`, exports, and evaluates on gold (float32 + int8 JSON
weights, each with --json-out). CHANNELS/LAYERS/DILATION still exist for
the legacy (non-MATRIX) single-run path, but a matrix entry's own channels
value always wins for that entry's run. Output:
  - `output/matrix.json` -- every entry's val + gold metrics
  - `output/matrix.md`   -- a markdown table of the same, printed at the end
  - `output/models/` + `output/metrics.json` -- the WINNING run's model dir
    and metrics, in the same shape the legacy single-run path wrote them in
    (plus a `matrix_winner` field), so `run.py pull` keeps working unchanged.
Winner selection: the winning **config** (arch:channels) is the one with
the highest mean int8 combined gold value_acc across its seeds; within that
config, the winning **seed** is the one with the best val value_acc (from
`ckpt["val_metrics"]`).
"""
from __future__ import annotations

import json as _json
import os
import shutil
import subprocess
import sys
import time

# === CONFIG START (do not hand-edit markers; run.py rewrites values here) ===
REPO_URL = os.environ.get("REPO_URL", "https://github.com/athrvk/gpu-sankhya")
GIT_REF = os.environ.get("GIT_REF", "master")
N_TRAIN = int(os.environ.get("N_TRAIN", "200000"))
N_VAL = int(os.environ.get("N_VAL", "6000"))
LANGS = os.environ.get("LANGS", "hi_latn,hi_deva")
MIX = os.environ.get("MIX", "0.55,0.45")
CROSS = os.environ.get("CROSS", "0.10")
EPOCHS = int(os.environ.get("EPOCHS", "20"))
CHANNELS = int(os.environ.get("CHANNELS", "32"))
LAYERS = int(os.environ.get("LAYERS", "4"))
DILATION = int(os.environ.get("DILATION", "2"))
TRAIN_SEED = int(os.environ.get("TRAIN_SEED", "5"))
VAL_SEED = int(os.environ.get("VAL_SEED", "6"))
MATRIX = os.environ.get("MATRIX", "v1:32:0")
# === CONFIG END ===

SMOKE = os.environ.get("SMOKE") == "1"
if SMOKE:
    # Local dry-run mode: tiny data, one epoch, clone from a local path
    # instead of GitHub. REPO_URL is reinterpreted as a filesystem path
    # when SMOKE=1 and it doesn't look like a URL.
    N_TRAIN = int(os.environ.get("N_TRAIN", "3000"))
    N_VAL = int(os.environ.get("N_VAL", "500"))
    EPOCHS = int(os.environ.get("EPOCHS", "1"))

WORKDIR = os.environ.get("KAGGLE_WORKDIR", "/kaggle/working")
REPO_DIR = os.path.join(WORKDIR, "gpu-sankhya")
OUTPUT_DIR = os.path.join(WORKDIR, "output")


def banner(msg: str) -> None:
    line = "=" * min(78, max(20, len(msg) + 4))
    print(f"\n{line}\n= {msg}\n{line}\n", flush=True)


def run(cmd, cwd=None, check=True):
    print(f"$ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=cwd)
    dt = time.time() - t0
    print(f"  (exit={proc.returncode}, {dt:.1f}s)", flush=True)
    if check and proc.returncode != 0:
        raise SystemExit(f"command failed ({proc.returncode}): {' '.join(cmd)}")
    return proc.returncode


def step_clone():
    banner(f"1/6 clone repo (ref={GIT_REF}, source={REPO_URL}, smoke={SMOKE})")
    if os.path.isdir(REPO_DIR):
        shutil.rmtree(REPO_DIR)
    if SMOKE and os.path.isdir(REPO_URL):
        # Local smoke test: clone from a filesystem path so we don't need
        # network access or a pushed commit to test against. A plain `git
        # clone` of a local path only copies committed history -- but a
        # sibling implementer may have uncommitted changes on this exact
        # branch (e.g. a concurrently-edited train.py) that we still need
        # to exercise. So: git-clone for speed/history, then rsync the
        # WORKING TREE (including untracked-but-not-ignored files) on top,
        # overwriting the committed copies with whatever is on disk.
        run(["git", "clone", "--no-hardlinks", REPO_URL, REPO_DIR])
        run(["git", "-C", REPO_DIR, "checkout", GIT_REF], check=False)
        if shutil.which("rsync"):
            run([
                "rsync", "-a", "--exclude=.git",
                f"{REPO_URL.rstrip('/')}/", f"{REPO_DIR}/",
            ])
        else:
            # rsync unavailable: fall back to a plain recursive copy over
            # the git-cloned tree (slower, no delete-extraneous semantics,
            # but still picks up uncommitted edits to existing files).
            for root, dirs, files in os.walk(REPO_URL):
                if ".git" in dirs:
                    dirs.remove(".git")
                rel = os.path.relpath(root, REPO_URL)
                dst_root = REPO_DIR if rel == "." else os.path.join(REPO_DIR, rel)
                os.makedirs(dst_root, exist_ok=True)
                for fn in files:
                    shutil.copy2(os.path.join(root, fn), os.path.join(dst_root, fn))
    else:
        run(["git", "clone", REPO_URL, REPO_DIR])
        run(["git", "-C", REPO_DIR, "checkout", GIT_REF])
    subprocess.run(["git", "-C", REPO_DIR, "rev-parse", "HEAD"])


def step_install():
    banner("2/6 pip install (onnx, onnxruntime -- torch is preinstalled on Kaggle)")
    run([sys.executable, "-m", "pip", "install", "-q", "onnx", "onnxruntime"])


def py(args, cwd):
    run([sys.executable, "-m"] + args, cwd=cwd)


def step_generate(pydir):
    banner(f"3/6 generate data (n_train={N_TRAIN} n_val={N_VAL} langs={LANGS} mix={MIX} cross={CROSS})")
    os.makedirs(os.path.join(pydir, "data"), exist_ok=True)
    os.makedirs(os.path.join(pydir, "models"), exist_ok=True)
    gen_train = [
        "sankhya.generator", "--n", str(N_TRAIN), "--seed", str(TRAIN_SEED),
        "--out", "data/train.jsonl", "--lang", LANGS,
    ]
    gen_val = [
        "sankhya.generator", "--n", str(N_VAL), "--seed", str(VAL_SEED),
        "--out", "data/val.jsonl", "--lang", LANGS,
    ]
    if "," in LANGS:
        gen_train += ["--mix", MIX, "--cross", str(CROSS)]
        gen_val += ["--mix", MIX, "--cross", str(CROSS)]
    py(gen_train, cwd=pydir)
    py(gen_val, cwd=pydir)


def parse_matrix(spec):
    """Parse MATRIX="v1:32:0,v2:32:0,..." into a list of dicts."""
    entries = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 3:
            raise SystemExit(f"bad MATRIX entry {item!r}; expected arch:channels:seed")
        arch, channels, seed = parts
        entries.append({"arch": arch, "channels": int(channels), "seed": int(seed)})
    if not entries:
        raise SystemExit(f"MATRIX produced no entries: {spec!r}")
    return entries


def entry_dir_name(entry):
    return f"models_{entry['arch']}_{entry['channels']}_s{entry['seed']}"


def step_train_one(pydir, entry, out_dir):
    banner(f"train {entry['arch']}:{entry['channels']}:{entry['seed']} "
           f"(epochs={EPOCHS} out={out_dir})")
    os.makedirs(os.path.join(pydir, out_dir), exist_ok=True)
    t0 = time.time()
    py([
        "sankhya.train",
        "--train", "data/train.jsonl", "--val", "data/val.jsonl",
        "--epochs", str(EPOCHS), "--batch", "128", "--lr", "3e-3",
        "--arch", entry["arch"], "--channels", str(entry["channels"]),
        "--seed", str(entry["seed"]),
        "--lang", LANGS, "--out", out_dir + "/", "--device", "auto",
    ], cwd=pydir)
    print(f"  train wall time: {time.time() - t0:.1f}s", flush=True)


def step_export_one(pydir, out_dir):
    banner(f"export ({out_dir})")
    t0 = time.time()
    py([
        "sankhya.export", "--ckpt", f"{out_dir}/sankhya.pt",
        "--val", "data/val.jsonl", "--out", out_dir + "/",
    ], cwd=pydir)
    print(f"  export wall time: {time.time() - t0:.1f}s", flush=True)


def step_eval_one(pydir, out_dir):
    """Eval float32 + int8 JSON weights for one trained run; returns a
    {label: metrics_dict} map (metrics_dict is eval_gold's --json-out shape)."""
    banner(f"eval on gold ({out_dir})")
    metrics = {}
    for label, ckpt_flag in [
        ("torch", ["--ckpt", f"{out_dir}/sankhya.pt"]),
        ("json_float32", ["--weights-json", f"{out_dir}/sankhya.weights.json"]),
        ("json_int8", ["--weights-json", f"{out_dir}/sankhya.weights.int8.json", "--int8"]),
    ]:
        json_out = f"{out_dir}/gold_metrics_{label}.json"
        t0 = time.time()
        py([
            "sankhya.eval_gold", "--quiet",
            "--gold", "tests/gold.jsonl", "tests/gold_deva.jsonl",
        ] + ckpt_flag + ["--json-out", json_out], cwd=pydir)
        print(f"  eval[{label}] wall time: {time.time() - t0:.1f}s", flush=True)
        path = os.path.join(pydir, json_out)
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                metrics[label] = _json.load(f)
    return metrics


def load_val_metrics(pydir, out_dir):
    import torch
    ckpt = torch.load(os.path.join(pydir, out_dir, "sankhya.pt"), map_location="cpu")
    return ckpt.get("val_metrics", {})


def select_winner(results):
    """results: list of {"arch","channels","seed","out_dir","val_metrics","gold"}.
    Winning config (arch:channels) = highest mean int8 combined gold
    value_acc across its seeds; within it, best seed by val value_acc."""
    by_config = {}
    order = []
    for r in results:
        key = (r["arch"], r["channels"])
        if key not in by_config:
            by_config[key] = []
            order.append(key)
        by_config[key].append(r)

    def int8_value_acc(r):
        return r["gold"].get("json_int8", {}).get("combined", {}).get("value_acc", 0.0)

    best_key, best_mean = None, None
    for key in order:
        vals = [int8_value_acc(r) for r in by_config[key]]
        m = sum(vals) / len(vals)
        if best_mean is None or m > best_mean:
            best_mean, best_key = m, key

    winners = by_config[best_key]
    best_r, best_val = None, None
    for r in winners:
        v = r["val_metrics"].get("value_acc", 0.0)
        if best_val is None or v > best_val:
            best_val, best_r = v, r
    return best_r, f"{best_key[0]}:{best_key[1]}"


def render_matrix_md(results, winner_label):
    header = ["arch", "channels", "seed", "val_value_acc", "gold_torch_value_acc",
              "gold_json_float32_value_acc", "gold_int8_value_acc", "gold_int8_f1", "winner"]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for r in results:
        config_label = f"{r['arch']}:{r['channels']}"
        is_winner = (r is results_winner_entry.get("entry"))
        row = [
            r["arch"], str(r["channels"]), str(r["seed"]),
            f"{r['val_metrics'].get('value_acc', 0.0):.4f}",
            f"{r['gold'].get('torch', {}).get('combined', {}).get('value_acc', 0.0):.4f}",
            f"{r['gold'].get('json_float32', {}).get('combined', {}).get('value_acc', 0.0):.4f}",
            f"{r['gold'].get('json_int8', {}).get('combined', {}).get('value_acc', 0.0):.4f}",
            f"{r['gold'].get('json_int8', {}).get('combined', {}).get('f1', 0.0):.4f}",
            "<-- winner" if is_winner else "",
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines.append(f"\nwinning config: **{winner_label}**")
    return "\n".join(lines)


# module-level scratch used by render_matrix_md to mark the winning row
results_winner_entry = {}


def step_matrix(pydir):
    banner(f"4-6/6 matrix: {MATRIX}")
    entries = parse_matrix(MATRIX)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = []
    for entry in entries:
        out_dir = entry_dir_name(entry)
        t_entry0 = time.time()
        step_train_one(pydir, entry, out_dir)
        step_export_one(pydir, out_dir)
        gold_metrics = step_eval_one(pydir, out_dir)
        val_metrics = load_val_metrics(pydir, out_dir)
        dt = time.time() - t_entry0
        print(f"[matrix] entry {entry['arch']}:{entry['channels']}:{entry['seed']} "
              f"total wall time: {dt:.1f}s", flush=True)
        results.append({
            "arch": entry["arch"], "channels": entry["channels"], "seed": entry["seed"],
            "out_dir": out_dir, "val_metrics": val_metrics, "gold": gold_metrics,
            "wall_time_s": dt,
        })

    winner, winner_label = select_winner(results)
    results_winner_entry["entry"] = winner

    matrix_json_path = os.path.join(OUTPUT_DIR, "matrix.json")
    with open(matrix_json_path, "w", encoding="utf-8") as f:
        _json.dump({"matrix": MATRIX, "entries": results, "winner": winner_label}, f, indent=2)
    print(f"wrote {matrix_json_path}")

    matrix_md = render_matrix_md(results, winner_label)
    matrix_md_path = os.path.join(OUTPUT_DIR, "matrix.md")
    with open(matrix_md_path, "w", encoding="utf-8") as f:
        f.write(matrix_md + "\n")
    print(f"wrote {matrix_md_path}")

    # stage the winning run as output/models + output/metrics.json, same
    # shape the legacy single-run path used, so `run.py pull` still works.
    models_src = os.path.join(pydir, winner["out_dir"])
    models_dst = os.path.join(OUTPUT_DIR, "models")
    if os.path.isdir(models_dst):
        shutil.rmtree(models_dst)
    shutil.copytree(models_src, models_dst)

    metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
    config_used = {
        "repo_url": REPO_URL, "git_ref": GIT_REF, "n_train": N_TRAIN, "n_val": N_VAL,
        "langs": LANGS, "mix": MIX, "cross": CROSS, "epochs": EPOCHS,
        "arch": winner["arch"], "channels": winner["channels"], "seed": winner["seed"],
        "matrix": MATRIX, "smoke": SMOKE,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        _json.dump({
            "config": config_used, "gold": winner["gold"], "matrix_winner": winner_label,
        }, f, indent=2)
    print(f"wrote {metrics_path}")
    print(f"output dir contents: {os.listdir(OUTPUT_DIR)}")

    print("\n" + matrix_md)


def main():
    t0 = time.time()
    step_clone()
    step_install()
    pydir = os.path.join(REPO_DIR, "python")
    step_generate(pydir)
    step_matrix(pydir)
    banner(f"done in {time.time() - t0:.1f}s total")


if __name__ == "__main__":
    main()
