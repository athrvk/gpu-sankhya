"""Publish gpu-sankhya artifacts to Hugging Face Hub.

Run from `python/`:

    python -m sankhya.hf_push model|space|dataset|all [--dry-run]

Only `huggingface_hub` + stdlib (json/pathlib) is imported here -- no
torch, no numpy -- so this can run in a lightweight CI job. Auth token
comes from the `HF_TOKEN` env var (a write-scoped HF token). `--dry-run`
writes the generated model/space/dataset cards (and the list of files that
would be uploaded) to `python/hf_out/` without making any network calls or
requiring a token.

Repos (created with exist_ok=True on first push):
  - model:   athrvk/gpu-sankhya            (models/default/*)
  - space:   athrvk/gpu-sankhya-demo       (static Space, built site/)
  - dataset: athrvk/gpu-sankhya-gold       (hand-written gold sets)
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY_ROOT = HERE.parent
REPO_ROOT = PY_ROOT.parent
MODELS_DIR = REPO_ROOT / "models" / "default"
SITE_DIR = REPO_ROOT / "site"
GOLD_DIR = PY_ROOT / "tests"
OUT_DIR = PY_ROOT / "hf_out"

MODEL_REPO = "athrvk/gpu-sankhya"
SPACE_REPO = "athrvk/gpu-sankhya-demo"
DATASET_REPO = "athrvk/gpu-sankhya-gold"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _npm_version() -> str:
    pkg = _read_json(REPO_ROOT / "package.json")
    return pkg["version"]


def _count_params(weights_json: dict) -> int:
    """Sum parameter counts from shapes in a float weights JSON."""
    total = 0

    def shape_prod(t):
        # float JSON tensors/biases are {"shape", "data"} dicts; the int8
        # file stores biases as plain lists -- handle both.
        if isinstance(t, list):
            return len(t)
        n = 1
        for d in t["shape"]:
            n *= d
        return n

    total += shape_prod(weights_json["embed"])
    for layer in weights_json["conv"]:
        total += shape_prod(layer["w"])
        total += shape_prod(layer["b"])
    total += shape_prod(weights_json["bio"]["w"]) + shape_prod(weights_json["bio"]["b"])
    total += shape_prod(weights_json["cls"]["w"]) + shape_prod(weights_json["cls"]["b"])
    return total


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def _fmt_acc(x: float) -> str:
    return f"{x:.4f}"


# ---------------------------------------------------------------------------
# Model card
# ---------------------------------------------------------------------------

def build_model_card() -> str:
    weights = _read_json(MODELS_DIR / "sankhya.weights.json")
    metrics = _read_json(MODELS_DIR / "gold_metrics_json_int8.json")
    version = _npm_version()

    n_layers = len(weights["conv"])
    n_params = _count_params(weights)
    embed_dim = weights["embed_dim"]
    channels = weights["channels"]
    vocab = len(weights["charset"])
    n_classes = len(weights["classes"])

    layer_lines = []
    for i, layer in enumerate(weights["conv"]):
        layer_lines.append(
            f"  {i + 1}. kernel {layer['k']}, dilation {layer['dilation']}, "
            f"residual {'yes' if layer['residual'] else 'no'}"
        )

    combined = metrics["combined"]
    negatives = combined["negatives"]
    categories = combined["categories"]

    per_file_rows = []
    for fname, m in metrics["per_file"].items():
        short = Path(fname).name
        per_file_rows.append(
            f"| {short} | {m['examples']} | {m['spans']} | {_fmt_acc(m['value_acc'])} "
            f"| {_fmt_acc(m['precision'])} | {_fmt_acc(m['recall'])} | {_fmt_acc(m['f1'])} |"
        )
    per_file_rows.append(
        f"| **combined** | {combined['examples']} | {combined['spans']} "
        f"| **{_fmt_acc(combined['value_acc'])}** | {_fmt_acc(combined['precision'])} "
        f"| {_fmt_acc(combined['recall'])} | {_fmt_acc(combined['f1'])} |"
    )

    cat_rows = []
    for cat, c in categories.items():
        cat_rows.append(f"| {cat} | {c['spans']} | {_fmt_acc(c['value_acc'])} |")

    files_table = "\n".join(
        f"| `{p.name}` | {p.stat().st_size:,} bytes |"
        for p in sorted(MODELS_DIR.iterdir())
        if p.is_file()
    )

    card = f"""---
license: mit
language:
  - hi
tags:
  - token-classification
  - onnx
  - char-cnn
  - hinglish
  - devanagari
  - indian-numbering
  - webgpu
library_name: custom
pipeline_tag: token-classification
datasets:
  - athrvk/gpu-sankhya-gold
---

# gpu-sankhya

A small char-level CNN that extracts Indian informal number/currency
shorthand -- Hinglish (romanised Hindi), Devanagari Hindi, Devanagari
Marathi, and Indian-English amount phrases like `sava lakh`, `dedh crore`, `डेढ़ लाख`,
`सवा करोड़`, `2.5L`, `20k`, `2-3 lakh` -- from free text, and turns each
match into a clean numeric value via a deterministic arithmetic core (the
model never predicts the value directly).

- npm package (runtime, ships this model quantized inline):
  https://www.npmjs.com/package/gpu-sankhya
- source / training code:
  https://github.com/athrvk/gpu-sankhya
- live demo: https://huggingface.co/spaces/{SPACE_REPO}
- gold evaluation sets: https://huggingface.co/datasets/{DATASET_REPO}

This model card describes weights version `v{version}` (arch `v2`).

## Architecture

A dilated/residual char-CNN over per-character embeddings:

- embedding dim: {embed_dim}
- conv channels: {channels}
- {n_layers} conv layers:
{chr(10).join(layer_lines)}
- vocab: {vocab} characters (union of every shipped language pack)
- output classes: {n_classes} (BIO span tag + semantic token class)
- parameters: {n_params:,}

A deterministic arithmetic core (not part of this model) then evaluates
the decoded token sequence into a value: prefix semantics (sava = x1.25,
dedh = x1.5, paune = subtract 1/4 from the next cardinal, ...), additive
combination of descending units, multiplicative combination of ascending
units, and range handling.

## Files

| file | size |
| --- | ---: |
{files_table}

- `sankhya.pt` -- torch checkpoint (vocab, classes, arch, channels, state_dict)
- `sankhya.onnx` -- ONNX graph, for interop/inspection
- `sankhya.weights.json` -- float32 weights, human-readable JSON
- `sankhya.weights.int8.json` -- int8-quantized weights (what the npm
  package and this card's accuracy numbers use)
- `charset.json`, `classes.json` -- standalone vocab/class tables
- `gold_metrics_*.json` -- per-file + combined gold evaluation (torch,
  float32 JSON, int8 JSON)
- `matrix.md`, `kaggle_metrics.json` -- training/sweep provenance from the
  Kaggle GPU run that produced this checkpoint

## Accuracy

Evaluated with the int8-quantized weights (what ships in the npm package)
against the hand-written gold sets (`gold_<lang>.jsonl` -- see the
[dataset card]({DATASET_REPO})):

| gold set | examples | spans | value_acc | precision | recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(per_file_rows)}

Negatives (zero-gold-span examples): {negatives['examples']}, false
positives: {negatives['false_positives']} ({_fmt_pct(negatives['fp_rate'])}).

Per-category value accuracy (combined):

| category | spans | value_acc |
| --- | ---: | ---: |
{chr(10).join(cat_rows)}

## Usage

### JavaScript (recommended -- ships this model quantized, no download)

```js
import {{ parse }} from "gpu-sankhya";

parse("sava lakh");
// [{{ span: "sava lakh", value: 125000, unit: "lakh", ... }}]
```

### Python (numpy reference forward pass)

```python
import numpy as np
from sankhya import np_infer, decode, core, charset
from sankhya.train import build_char_to_id

weights_json = json.load(open("sankhya.weights.int8.json"))
weights = np_infer.load_weights_int8_json(weights_json)
char_to_id = build_char_to_id(weights_json["charset"])
unk = char_to_id.get("<unk>", 1)

text = charset.normalize_text("sava lakh")
ids = np_infer.pad_ids([char_to_id.get(c, unk) for c in text])
bio_logits, cls_logits = np_infer.forward(weights, np.array(ids))
n = len(text)
bio_pred = bio_logits[:n].argmax(-1).tolist()
cls_pred = cls_logits[:n].argmax(-1).tolist()
bio_probs = np_infer.softmax(bio_logits[:n]).tolist()

spans = decode.decode_spans(text, bio_pred, cls_pred, bio_probs=bio_probs)
for span in spans:
    result = core.evaluate(span["tokens"])
    print(text[span["start"]:span["end"]], result.value)
```

See `python/README.md` in the source repo for the full training/export
pipeline and `sankhya.eval_gold` for a ready-made evaluation CLI.

## Training

Trained 20 epochs on 200,000 synthetic examples generated from the
`hi_latn` (romanised Hindi), `hi_deva` (Devanagari Hindi), and `mr_deva`
(Devanagari Marathi) grammar packs, mixed 0.40/0.33/0.27 with a 10%
cross-pack share, plus out-of-vocab
"unk noise" augmentation (emoji, CJK, Cyrillic, other symbols inserted as
O-labelled context) so the `<unk>` embedding actually gets gradient
signal. Batch size 128, lr 3e-3. Architecture and channel count (`v2`,
48 channels) were chosen by a config/seed sweep, picking the config with
the highest mean int8 combined gold value_acc across >= 3 seeds (seed
spread is +/-1-2 points), then the seed by val accuracy within a 0.005 tie
band, then higher int8 combined gold F1, then lower negatives
false-positive rate. Full recipe, sweep evidence, and reproduction
commands: `python/README.md` in the source repo.

## Limitations

- JavaScript string indices count UTF-16 code units, so astral characters
  (e.g. some emoji) occupy 2 code units -- span offsets from the JS
  runtime account for this, but consumers indexing raw strings themselves
  should be aware of it.
- The model occasionally produces spurious spans on unfamiliar words near
  number-ish context (a measured trade-off from the out-of-vocab noise
  training -- see Accuracy above).
- Long multi-term/mixed-numeral constructs and multi-number range phrases
  ("तीस पैंतीस हज़ार", "three n half lakh") are the weakest category
  (`long`/`range` value_acc above).
- Scoped to Indian languages: currently Hinglish, Devanagari Hindi, and
  Devanagari Marathi only; other Indian languages are planned (see the
  source repo's roadmap).
"""
    return card


# ---------------------------------------------------------------------------
# Space card
# ---------------------------------------------------------------------------

def build_space_card() -> str:
    return f"""---
title: gpu-sankhya demo
emoji: \U0001f522
colorFrom: indigo
colorTo: pink
sdk: static
pinned: false
license: mit
---

Interactive demo of [gpu-sankhya](https://huggingface.co/{MODEL_REPO}) -- parses Hinglish, Devanagari Hindi, and Devanagari Marathi number/currency shorthand in the browser (CPU + WebGPU backends). Source: https://github.com/athrvk/gpu-sankhya
"""


# ---------------------------------------------------------------------------
# Dataset card
# ---------------------------------------------------------------------------

def _count_jsonl_lines(path: Path) -> int:
    with path.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


# One blurb per pack for the dataset card; a pack with no entry here still
# gets listed, just with a generic description.
_GOLD_BLURBS = {
    "hi_latn": 'romanised Hindi / Hinglish\n  ("bhai sava lakh mein ho jayega kya")',
    "hi_deva": 'Devanagari Hindi\n  ("सवा लाख मिलेगा")',
    "mr_deva": 'Devanagari Marathi\n  ("दीड लाख मिळतील")',
}


def gold_sets():
    """Every `tests/gold*.jsonl` file, as (local_path, repo_name, lang_id).

    Derived from what is on disk plus the pack registry, so adding a
    language (and its gold file) needs no edit here: `gold.jsonl` is
    hi_latn for historical reasons, `gold_<suffix>.jsonl` resolves through
    eval_gold._get_pack (which knows the suffix aliases).
    """
    from . import eval_gold

    out = []
    for path in sorted(GOLD_DIR.glob("gold*.jsonl")):
        suffix = path.stem[len("gold_"):] if path.stem.startswith("gold_") else ""
        pack = eval_gold._get_pack(suffix)
        lang_id = pack.id if pack is not None else (suffix or "hi_latn")
        out.append((path, f"gold_{lang_id}.jsonl", lang_id))
    # stable, human-friendly order: hi_latn, hi_deva, then the rest
    order = {"hi_latn": 0, "hi_deva": 1}
    out.sort(key=lambda row: (order.get(row[2], 2), row[2]))
    return out


def build_dataset_card() -> str:
    sets = gold_sets()
    lines = []
    for path, repo_name, lang_id in sets:
        blurb = _GOLD_BLURBS.get(lang_id, f"`{lang_id}` gold examples")
        lines.append(f"- `{repo_name}` ({_count_jsonl_lines(path)} examples) -- {blurb}")
    gold_list = "\n".join(lines)
    lang_ids = ", ".join(f"`{lang_id}`" for _, _, lang_id in sets)
    eval_cmd = " ".join(repo_name for _, repo_name, _ in sets)
    # ISO-639-1 codes for the front matter, derived from the pack ids
    iso = []
    for _, _, lang_id in sets:
        code = lang_id.split("_")[0]
        if code not in iso:
            iso.append(code)
    language_block = "\n".join(f"  - {code}" for code in iso)

    return f"""---
license: mit
language:
{language_block}
task_categories:
  - token-classification
size_categories:
  - n<1K
---

# gpu-sankhya gold evaluation sets

Hand-written gold sets used to evaluate the
[gpu-sankhya](https://huggingface.co/{MODEL_REPO}) model, written
independently of the synthetic data generator so they measure real-world
quality rather than in-distribution accuracy.

{gold_list}

## Format

One JSON object per line:

```json
{{"text": "bhai sava lakh mein ho jayega kya", "lang": "hi_latn", "spans": [{{"start": 5, "end": 14, "value": 125000}}]}}
```

- `text` -- the input sentence.
- `lang` -- one of {lang_ids}.
- `spans` -- zero or more amount spans (an empty list is a "negative"
  example, used to measure false-positive rate). Each span has:
  - `start`, `end` -- code-point offsets into the *normalized* text
    (Unicode NFC, Devanagari digits mapped to ASCII, lowercased -- see
    `sankhya.charset.normalize_text`)
  - `value` -- the resolved numeric value
  - `range` (optional) -- `[low, high]` for range phrases ("2-3 lakh")
  - `unit` (optional) -- `sau`/`hazaar`/`lakh`/`crore`/...
  - `currency` (optional) -- `INR` when a currency marker is adjacent to
    the span

## Usage

These files are consumed by `python -m sankhya.eval_gold --gold
{eval_cmd} --weights-json <weights>.json
[--int8]` in the source repo, which decodes each text with the model,
matches predicted spans to gold spans, and reports precision/recall/F1,
value accuracy, a per-category breakdown, and the negatives false-positive
rate. See `python/README.md` in
https://github.com/athrvk/gpu-sankhya for the full evaluation pipeline.
"""


# ---------------------------------------------------------------------------
# Push helpers
# ---------------------------------------------------------------------------

def _get_api(dry_run: bool):
    if dry_run:
        return None
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN env var is required (a write-scoped Hugging Face token)")
    return HfApi(token=token)


def push_model(dry_run: bool) -> None:
    card = build_model_card()
    version = _npm_version()

    if dry_run:
        out = OUT_DIR / "model"
        out.mkdir(parents=True, exist_ok=True)
        (out / "README.md").write_text(card, encoding="utf-8")
        files = sorted(p.name for p in MODELS_DIR.iterdir() if p.is_file())
        (out / "files.json").write_text(json.dumps(files, indent=2), encoding="utf-8")
        print(f"[dry-run] model card + file list written to {out}")
        print(f"[dry-run] would tag revision v{version} on {MODEL_REPO}")
        return

    api = _get_api(dry_run)
    api.create_repo(MODEL_REPO, repo_type="model", exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        readme = Path(tmp) / "README.md"
        readme.write_text(card, encoding="utf-8")
        api.upload_file(
            path_or_fileobj=str(readme),
            path_in_repo="README.md",
            repo_id=MODEL_REPO,
            repo_type="model",
        )

    api.upload_folder(
        folder_path=str(MODELS_DIR),
        repo_id=MODEL_REPO,
        repo_type="model",
    )

    tag = f"v{version}"
    try:
        api.create_tag(MODEL_REPO, tag=tag, repo_type="model", exist_ok=True)
    except TypeError:
        # older huggingface_hub without exist_ok on create_tag
        try:
            api.create_tag(MODEL_REPO, tag=tag, repo_type="model")
        except Exception as e:  # noqa: BLE001
            print(f"warning: could not create tag {tag}: {e}")
    print(f"pushed model to {MODEL_REPO}, tagged {tag}")


def push_space(dry_run: bool) -> None:
    card = build_space_card()

    if dry_run:
        out = OUT_DIR / "space"
        out.mkdir(parents=True, exist_ok=True)
        (out / "README.md").write_text(card, encoding="utf-8")
        files = []
        if SITE_DIR.is_dir():
            files = sorted(str(p.relative_to(SITE_DIR)) for p in SITE_DIR.rglob("*") if p.is_file())
        (out / "files.json").write_text(json.dumps(files, indent=2), encoding="utf-8")
        print(f"[dry-run] space card + file list written to {out}")
        return

    if not SITE_DIR.is_dir():
        raise SystemExit(f"{SITE_DIR} does not exist -- run `npm run build && npm run site` first")

    api = _get_api(dry_run)
    api.create_repo(SPACE_REPO, repo_type="space", space_sdk="static", exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        readme = Path(tmp) / "README.md"
        readme.write_text(card, encoding="utf-8")
        api.upload_file(
            path_or_fileobj=str(readme),
            path_in_repo="README.md",
            repo_id=SPACE_REPO,
            repo_type="space",
        )

    api.upload_folder(
        folder_path=str(SITE_DIR),
        repo_id=SPACE_REPO,
        repo_type="space",
    )
    print(f"pushed static site to {SPACE_REPO}")


def push_dataset(dry_run: bool) -> None:
    card = build_dataset_card()
    sets = gold_sets()

    if dry_run:
        out = OUT_DIR / "dataset"
        out.mkdir(parents=True, exist_ok=True)
        (out / "README.md").write_text(card, encoding="utf-8")
        files = [repo_name for _, repo_name, _ in sets]
        (out / "files.json").write_text(json.dumps(files, indent=2), encoding="utf-8")
        print(f"[dry-run] dataset card + file list written to {out}")
        return

    api = _get_api(dry_run)
    api.create_repo(DATASET_REPO, repo_type="dataset", exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        readme = Path(tmp) / "README.md"
        readme.write_text(card, encoding="utf-8")
        api.upload_file(
            path_or_fileobj=str(readme),
            path_in_repo="README.md",
            repo_id=DATASET_REPO,
            repo_type="dataset",
        )

    for path, repo_name, _lang_id in sets:
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=repo_name,
            repo_id=DATASET_REPO,
            repo_type="dataset",
        )
    print(f"pushed gold sets to {DATASET_REPO}")


TARGETS = {
    "model": push_model,
    "space": push_space,
    "dataset": push_dataset,
}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m sankhya.hf_push")
    ap.add_argument("target", choices=["model", "space", "dataset", "all"])
    ap.add_argument("--dry-run", action="store_true", help="write cards + file lists to python/hf_out/, no network")
    args = ap.parse_args(argv)

    targets = list(TARGETS) if args.target == "all" else [args.target]
    for t in targets:
        TARGETS[t](args.dry_run)


if __name__ == "__main__":
    main()
