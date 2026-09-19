"""Tests for sankhya.hf_push -- dry-run only, no network access.

Run with `python -m pytest tests/test_hf_push.py -q` from `python/`.
"""
import os
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sankhya import hf_push  # noqa: E402


def test_dry_run_all(tmp_path, monkeypatch):
    out_dir = tmp_path / "hf_out"
    monkeypatch.setattr(hf_push, "OUT_DIR", out_dir)

    hf_push.main(["all", "--dry-run"])

    assert (out_dir / "model" / "README.md").is_file()
    assert (out_dir / "space" / "README.md").is_file()
    assert (out_dir / "dataset" / "README.md").is_file()


def test_model_card_content():
    card = hf_push.build_model_card()

    weights = json.loads((hf_push.MODELS_DIR / "sankhya.weights.json").read_text())
    n_layers = len(weights["conv"])
    assert f"{n_layers} conv layers" in card

    metrics = json.loads((hf_push.MODELS_DIR / "gold_metrics_json_int8.json").read_text())
    combined_acc = hf_push._fmt_acc(metrics["combined"]["value_acc"])
    assert combined_acc in card

    # front matter keys
    front = card.split("---")[1]
    for key in (
        "license: mit",
        "pipeline_tag: token-classification",
        "library_name: custom",
    ):
        assert key in front
    assert "- hi" in front
    for tag in (
        "token-classification",
        "onnx",
        "char-cnn",
        "hinglish",
        "devanagari",
        "indian-numbering",
        "webgpu",
    ):
        assert tag in front
    assert "athrvk/gpu-sankhya-gold" in front


def test_space_card_front_matter():
    card = hf_push.build_space_card()
    assert "title: gpu-sankhya demo" in card
    assert "sdk: static" in card
    assert "pinned: false" in card
    assert "license: mit" in card
    assert "athrvk/gpu-sankhya" in card


def test_dataset_card_counts():
    card = hf_push.build_dataset_card()

    n_latn = hf_push._count_jsonl_lines(hf_push.GOLD_DIR / "gold.jsonl")
    n_deva = hf_push._count_jsonl_lines(hf_push.GOLD_DIR / "gold_deva.jsonl")

    assert f"gold_hi_latn.jsonl` ({n_latn} examples)" in card
    assert f"gold_hi_deva.jsonl` ({n_deva} examples)" in card

    front = card.split("---")[1]
    assert "license: mit" in front
    assert "- hi" in front
    assert "- token-classification" in front
    assert "n<1K" in front


def test_get_api_requires_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(SystemExit):
        hf_push._get_api(dry_run=False)


def test_hf_push_does_not_need_numpy_or_torch():
    """The huggingface publish jobs install only requirements-hf.txt; the
    dataset card must build without numpy/torch (regression: 0.5.0/0.6.0
    dataset jobs failed with ModuleNotFoundError)."""
    import subprocess, sys
    code = (
        "import sys\n"
        "class Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in ('numpy', 'torch', 'onnx', 'onnxruntime'):\n"
        "            raise ImportError('blocked: ' + name)\n"
        "sys.meta_path.insert(0, Block())\n"
        "from sankhya import hf_push\n"
        "sets = hf_push.gold_sets()\n"
        "assert len(sets) >= 4, sets\n"
        "hf_push.build_dataset_card()\n"
        "print('ok', len(sets))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().startswith("ok")
