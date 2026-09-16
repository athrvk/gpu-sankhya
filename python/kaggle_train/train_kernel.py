"""Self-contained Kaggle "script" kernel: clone gpu-sankhya, generate data,
train, export, and evaluate on gold, then stage everything `kaggle kernels
output` downloads into /kaggle/working/output/.

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
"""
from __future__ import annotations

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
        # local smoke test: clone from a filesystem path so we don't need
        # network access or a pushed commit to test against.
        run(["git", "clone", "--no-hardlinks", REPO_URL, REPO_DIR])
        run(["git", "-C", REPO_DIR, "checkout", GIT_REF], check=False)
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


def step_train(pydir):
    banner(f"4/6 train (epochs={EPOCHS} channels={CHANNELS} layers={LAYERS} dilation={DILATION})")
    py([
        "sankhya.train",
        "--train", "data/train.jsonl", "--val", "data/val.jsonl",
        "--epochs", str(EPOCHS), "--batch", "128", "--lr", "3e-3",
        "--channels", str(CHANNELS), "--layers", str(LAYERS), "--dilation", str(DILATION),
        "--lang", LANGS, "--out", "models/", "--device", "auto",
    ], cwd=pydir)


def step_export(pydir):
    banner("5/6 export (onnx + json weights)")
    py([
        "sankhya.export", "--ckpt", "models/sankhya.pt",
        "--val", "data/val.jsonl", "--out", "models/",
    ], cwd=pydir)


def step_eval_and_collect(pydir):
    banner("6/6 eval on gold + collect output/")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    metrics = {}
    for label, ckpt_flag in [
        ("torch", ["--ckpt", "models/sankhya.pt"]),
        ("json_float32", ["--weights-json", "models/sankhya.weights.json"]),
        ("json_int8", ["--weights-json", "models/sankhya.weights.int8.json", "--int8"]),
    ]:
        json_out = f"models/gold_metrics_{label}.json"
        py([
            "sankhya.eval_gold",
            "--gold", "tests/gold.jsonl", "tests/gold_deva.jsonl",
        ] + ckpt_flag + ["--json-out", json_out], cwd=pydir)
        path = os.path.join(pydir, json_out)
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                metrics[label] = __import__("json").load(f)

    models_src = os.path.join(pydir, "models")
    models_dst = os.path.join(OUTPUT_DIR, "models")
    if os.path.isdir(models_dst):
        shutil.rmtree(models_dst)
    shutil.copytree(models_src, models_dst)

    metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
    import json as _json
    config_used = {
        "repo_url": REPO_URL, "git_ref": GIT_REF, "n_train": N_TRAIN, "n_val": N_VAL,
        "langs": LANGS, "mix": MIX, "cross": CROSS, "epochs": EPOCHS,
        "channels": CHANNELS, "layers": LAYERS, "dilation": DILATION,
        "smoke": SMOKE,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        _json.dump({"config": config_used, "gold": metrics}, f, indent=2)
    print(f"wrote {metrics_path}")
    print(f"output dir contents: {os.listdir(OUTPUT_DIR)}")


def main():
    t0 = time.time()
    step_clone()
    step_install()
    pydir = os.path.join(REPO_DIR, "python")
    step_generate(pydir)
    step_train(pydir)
    step_export(pydir)
    step_eval_and_collect(pydir)
    banner(f"done in {time.time() - t0:.1f}s total")


if __name__ == "__main__":
    main()
