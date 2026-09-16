"""CLI wrapper around the Kaggle CLI for the gpu-sankhya training kernel.

Run from `python/`:

    python -m kaggle_train.run push
    python -m kaggle_train.run status [--timeout SECONDS]
    python -m kaggle_train.run pull
    python -m kaggle_train.run all [--timeout SECONDS]

Auth: this wraps the `kaggle` CLI (2.2.4+), which accepts, in order:
  1. `KAGGLE_API_TOKEN` env var (token from https://www.kaggle.com/settings/api)
  2. a token file at `~/.kaggle/access_token`
  3. `kaggle auth login` (OAuth, cached locally)
It also still honours the legacy `~/.kaggle/kaggle.json`
(`KAGGLE_USERNAME`/`KAGGLE_KEY` env vars, or `KAGGLE_CONFIG_DIR` to point
at a different config directory) for older accounts/tooling. See
`python/README.md` for details -- this script does not invent or require
any other env var name.

Overrides: pass `--set KEY=VALUE` (repeatable) to change a training config
default (REPO_URL, GIT_REF, N_TRAIN, N_VAL, LANGS, MIX, CROSS, EPOCHS,
CHANNELS, LAYERS, DILATION, TRAIN_SEED, VAL_SEED, MATRIX) before push.
MATRIX is a comma-separated `arch:channels:seed` list (default "v1:32:0")
that runs a whole matrix of training configs in one kernel invocation --
e.g. `--set MATRIX=v1:32:0,v2:32:0,v2:32:1` -- and stages the winning
run's models/metrics the same way a single-config run would (plus
`output/matrix.json` / `output/matrix.md` with every entry's numbers).
CHANNELS/LAYERS/DILATION are still accepted but only matter if MATRIX is
left at a single legacy-shaped entry. Kaggle
"script" kernels allow exactly one code_file, so there's nowhere to ship a
second config file alongside it -- `push` rewrites the literal defaults
inside the `# === CONFIG START ===` / `# === CONFIG END ===` block of
train_kernel.py in place before uploading, then restores the original file
content on disk afterwards (the uploaded copy keeps the overrides; your
working tree does not).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

KERNEL_SLUG = "athrvk/gpu-sankhya-train"
HERE = Path(__file__).resolve().parent
KERNEL_SCRIPT = HERE / "train_kernel.py"
PY_ROOT = HERE.parent  # python/
REPO_ROOT = PY_ROOT.parent

CONFIG_KEYS = {
    "REPO_URL": str, "GIT_REF": str, "N_TRAIN": int, "N_VAL": int,
    "LANGS": str, "MIX": str, "CROSS": float, "EPOCHS": int,
    "CHANNELS": int, "LAYERS": int, "DILATION": int,
    "TRAIN_SEED": int, "VAL_SEED": int, "MATRIX": str,
}

AUTH_ENV_HINT = (
    "No Kaggle credentials found. The `kaggle` CLI (2.2.4+) accepts one of:\n"
    "  1. KAGGLE_API_TOKEN=<token>            (from https://www.kaggle.com/settings/api)\n"
    "  2. a token file at ~/.kaggle/access_token\n"
    "  3. `kaggle auth login` (OAuth, cached under ~/.kaggle/)\n"
    "  4. legacy: ~/.kaggle/kaggle.json, or KAGGLE_USERNAME + KAGGLE_KEY env vars\n"
    "     (KAGGLE_CONFIG_DIR overrides where kaggle.json is read from)\n"
    "Set one of these and re-run."
)


def check_auth():
    """Fail fast with a clear message if no credential is configured."""
    import os

    if os.environ.get("KAGGLE_API_TOKEN"):
        return
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return
    config_dir = Path(os.environ.get("KAGGLE_CONFIG_DIR") or (Path.home() / ".kaggle"))
    if (config_dir / "access_token").is_file():
        return
    if (config_dir / "kaggle.json").is_file():
        return
    print(AUTH_ENV_HINT, file=sys.stderr)
    raise SystemExit(2)


def require_kaggle_cli():
    if shutil.which("kaggle") is None:
        raise SystemExit("`kaggle` CLI not found on PATH. `pip install kaggle` first.")


def apply_overrides(overrides: dict) -> str:
    """Rewrite the CONFIG block in train_kernel.py with `overrides`, return the original text."""
    original = KERNEL_SCRIPT.read_text(encoding="utf-8")
    if not overrides:
        return original
    text = original
    for key, value in overrides.items():
        py_type = CONFIG_KEYS[key]
        if py_type in (int,):
            new_default = f'int(os.environ.get("{key}", "{value}"))'
        elif py_type in (float,):
            new_default = f'os.environ.get("{key}", "{value}")' if key == "CROSS" else f'float(os.environ.get("{key}", "{value}"))'
        else:
            new_default = f'os.environ.get("{key}", "{value}")'
        pattern = re.compile(rf'^{key} = .*$', re.MULTILINE)
        if not pattern.search(text):
            raise SystemExit(f"could not find CONFIG line for {key} in {KERNEL_SCRIPT}")
        text = pattern.sub(f"{key} = {new_default}", text, count=1)
    KERNEL_SCRIPT.write_text(text, encoding="utf-8")
    return original


def parse_set_args(pairs):
    overrides = {}
    for item in pairs or []:
        if "=" not in item:
            raise SystemExit(f"--set expects KEY=VALUE, got: {item!r}")
        k, v = item.split("=", 1)
        k = k.strip()
        if k not in CONFIG_KEYS:
            raise SystemExit(f"unknown config key {k!r}; valid keys: {sorted(CONFIG_KEYS)}")
        overrides[k] = v.strip()
    return overrides


def cmd_push(args):
    require_kaggle_cli()
    check_auth()
    overrides = parse_set_args(args.set)
    original = apply_overrides(overrides)
    try:
        cmd = ["kaggle", "kernels", "push", "-p", str(HERE)]
        print(f"$ {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    finally:
        if overrides:
            KERNEL_SCRIPT.write_text(original, encoding="utf-8")
            print("(restored train_kernel.py defaults on disk; overrides only applied to the pushed copy)")


def cmd_status(args):
    require_kaggle_cli()
    check_auth()
    deadline = time.time() + args.timeout if args.timeout else None
    terminal = {"complete", "error", "cancelled", "cancelacknowledged"}
    while True:
        proc = subprocess.run(
            ["kaggle", "kernels", "status", KERNEL_SLUG],
            capture_output=True, text=True,
        )
        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        print(out or err)
        low = out.lower()
        if any(t in low for t in terminal):
            if "error" in low or "cancel" in low:
                raise SystemExit(1)
            return
        if deadline and time.time() >= deadline:
            raise SystemExit(f"timed out after {args.timeout}s waiting for kernel to finish")
        time.sleep(30)


def cmd_pull(args):
    require_kaggle_cli()
    check_auth()
    out_dir = PY_ROOT / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "kernels", "output", KERNEL_SLUG, "-p", str(out_dir)],
        check=True,
    )
    int8_src = out_dir / "models" / "sankhya.weights.int8.json"
    if not int8_src.is_file():
        # kaggle kernels output may flatten the tree; also look one level up
        alt = out_dir / "sankhya.weights.int8.json"
        int8_src = alt if alt.is_file() else int8_src
    if not int8_src.is_file():
        raise SystemExit(f"could not find sankhya.weights.int8.json under {out_dir} after pull")

    default_weights = REPO_ROOT / "src" / "data" / "default-weights.json"
    default_weights.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(int8_src, default_weights)
    print(f"copied {int8_src} -> {default_weights}")

    fixtures_cmd = [
        sys.executable, "-m", "sankhya.make_fixtures",
        "--weights", str(default_weights),
        "--gold", "tests/gold.jsonl", "tests/gold_deva.jsonl",
        "--out-parity", "../test/fixtures/parity.jsonl",
        "--out-decoded", "../test/fixtures/decoded.jsonl",
    ]
    print(f"$ {' '.join(fixtures_cmd)}")
    subprocess.run(fixtures_cmd, cwd=str(PY_ROOT), check=True)
    print("fixtures regenerated. Run `npm test` from the repo root before committing.")


def cmd_all(args):
    cmd_push(args)
    cmd_status(args)
    cmd_pull(args)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m kaggle_train.run")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_push = sub.add_parser("push", help="upload the kernel to Kaggle")
    p_push.add_argument("--set", action="append", help="KEY=VALUE override, repeatable")
    p_push.set_defaults(func=cmd_push)

    p_status = sub.add_parser("status", help="poll kernel run status every 30s")
    p_status.add_argument("--timeout", type=int, default=3600, help="seconds to wait before giving up (0 = no timeout)")
    p_status.set_defaults(func=cmd_status)

    p_pull = sub.add_parser("pull", help="download kernel output, update default-weights.json + fixtures")
    p_pull.set_defaults(func=cmd_pull)

    p_all = sub.add_parser("all", help="push, then status, then pull")
    p_all.add_argument("--set", action="append", help="KEY=VALUE override, repeatable")
    p_all.add_argument("--timeout", type=int, default=3600, help="seconds to wait before giving up (0 = no timeout)")
    p_all.set_defaults(func=cmd_all)

    args = ap.parse_args(argv)
    if getattr(args, "timeout", None) == 0:
        args.timeout = None
    args.func(args)


if __name__ == "__main__":
    main()
