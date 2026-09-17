"""Tests for the CRF head: nll, viterbi_np, and end-to-end train/export/np_infer."""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import torch

from sankhya.crf import CRF, viterbi_np
from sankhya import np_infer


def test_nll_single_position_matches_softmax_ce():
    """With start/end/trans all zero, a length-1 sequence's NLL should equal
    -log softmax(emissions)[gold_tag] exactly (the forward algorithm over a
    single position collapses to a plain softmax)."""
    torch.manual_seed(0)
    crf = CRF()
    emissions = torch.randn(1, 1, 3)
    for gold in range(3):
        tags = torch.tensor([[gold]])
        mask = torch.ones(1, 1)
        nll = crf.nll(emissions, tags, mask)
        expected = -torch.log_softmax(emissions[0, 0], dim=-1)[gold]
        assert torch.allclose(nll, expected, atol=1e-5), (nll.item(), expected.item())


def test_viterbi_zero_params_matches_argmax():
    """With zero trans/start/end, Viterbi degenerates to per-position argmax."""
    rng = np.random.RandomState(0)
    emis = rng.randn(50, 3).astype(np.float32)
    trans = np.zeros((3, 3), dtype=np.float32)
    start = np.zeros(3, dtype=np.float32)
    end = np.zeros(3, dtype=np.float32)
    path = viterbi_np(emis, trans, start, end)
    assert path == emis.argmax(-1).tolist()


def test_viterbi_first_max_tiebreak():
    """Construct an exact tie at t=1 between i=0 and i=1 reaching j -- the
    contract says first-max wins (i.e. i=0), so back[1][j] must be 0."""
    emis = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ], dtype=np.float32)
    trans = np.zeros((3, 3), dtype=np.float32)
    start = np.zeros(3, dtype=np.float32)
    end = np.zeros(3, dtype=np.float32)
    # All deltas at t=0 are equal (start+emis all zero) -> all ties.
    # At t=1, for each j, delta[0][i]+trans[i][j] is equal for every i (all 0).
    # First-max tie-break means back[1][j] == 0 for every j, so the decoded
    # path's first label should be 0 (the lowest index among the ties).
    path = viterbi_np(emis, trans, start, end)
    assert path[0] == 0


def test_viterbi_respects_forbidden_transition():
    """trans[O][I] hugely negative should ban O->I transitions even when
    per-position argmax would produce one somewhere in a long random
    sequence."""
    rng = np.random.RandomState(1)
    L = 200
    emis = rng.randn(L, 3).astype(np.float32)
    # bias things so argmax likely produces O->I somewhere: alternate a
    # strong preference for O then I.
    for t in range(0, L, 2):
        emis[t] += np.array([5.0, 0.0, 0.0], dtype=np.float32)
    for t in range(1, L, 2):
        emis[t] += np.array([0.0, 0.0, 5.0], dtype=np.float32)
    argmax_path = emis.argmax(-1).tolist()
    has_o_to_i = any(argmax_path[t] == 0 and argmax_path[t + 1] == 2 for t in range(L - 1))
    assert has_o_to_i, "test setup should produce an O->I transition under plain argmax"

    trans = np.zeros((3, 3), dtype=np.float32)
    trans[0, 2] = -1e4  # O -> I forbidden
    start = np.zeros(3, dtype=np.float32)
    end = np.zeros(3, dtype=np.float32)
    path = viterbi_np(emis, trans, start, end)
    for t in range(L - 1):
        assert not (path[t] == 0 and path[t + 1] == 2), f"O->I at {t}"


def test_train_export_np_infer_with_crf(tmp_path):
    """Train 2 epochs with --crf on a small generated set, export, and check
    the JSON weights carry a crf block that np_infer loads and uses for
    Viterbi decoding; also check the model learned to discourage O->I."""
    py_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "models"
    data_dir.mkdir()
    out_dir.mkdir()

    subprocess.run(
        [sys.executable, "-m", "sankhya.generator", "--n", "500", "--seed", "0",
         "--out", str(data_dir / "train.jsonl"), "--lang", "hi_latn"],
        cwd=py_root, check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "sankhya.generator", "--n", "100", "--seed", "1",
         "--out", str(data_dir / "val.jsonl"), "--lang", "hi_latn"],
        cwd=py_root, check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "sankhya.train",
         "--train", str(data_dir / "train.jsonl"), "--val", str(data_dir / "val.jsonl"),
         "--epochs", "2", "--out", str(out_dir) + "/", "--crf", "--lang", "hi_latn"],
        cwd=py_root, check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "sankhya.export",
         "--ckpt", str(out_dir / "sankhya.pt"), "--val", str(data_dir / "val.jsonl"),
         "--out", str(out_dir) + "/"],
        cwd=py_root, check=True,
    )

    with open(out_dir / "sankhya.weights.json", "r", encoding="utf-8") as f:
        weights_json = json.load(f)
    assert "crf" in weights_json
    weights = np_infer.load_weights_json(weights_json)
    assert weights["crf"] is not None

    import torch as _torch
    ckpt = _torch.load(out_dir / "sankhya.pt", map_location="cpu")
    assert ckpt.get("use_crf") is True
    trans = ckpt["crf"]["trans"]
    # trans[0][2] (O->I) should have been learned to be less attractive than
    # trans[1][2] (B->I), since B->I is the common real transition and O->I
    # essentially never occurs in valid BIO sequences.
    assert trans[0][2] < trans[1][2], (trans[0][2], trans[1][2])

    # run viterbi on a padded example sliced back to L; check output length
    # and no O->I transitions.
    from sankhya.train import build_char_to_id, load_jsonl
    from sankhya.charset import normalize_text
    vocab = weights_json["charset"]
    char_to_id = build_char_to_id(vocab)
    unk = char_to_id.get("<unk>", 1)
    val_ex = load_jsonl(str(data_dir / "val.jsonl"))
    ex = val_ex[0]
    text = normalize_text(ex["text"])[:128]
    L = len(text)
    ids = np_infer.pad_ids(np.array([char_to_id.get(c, unk) for c in text], dtype=np.int64))
    bio_logits, _ = np_infer.forward(weights, ids)
    bio_logits = bio_logits[:L]
    path = np_infer.bio_path(bio_logits, weights)
    assert len(path) == L
    for t in range(L - 1):
        assert not (path[t] == 0 and path[t + 1] == 2)


def test_legacy_weights_bio_path_is_argmax():
    """A legacy weights JSON (no crf block) must decode via plain argmax."""
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(here))
    default_weights_path = os.path.join(repo_root, "src", "data", "default-weights.json")
    if not os.path.isfile(default_weights_path):
        import pytest
        pytest.skip("no default-weights.json present")
    with open(default_weights_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    weights = np_infer.load_weights_int8_json(obj)
    assert weights.get("crf") is None
    rng = np.random.RandomState(0)
    bio_logits = rng.randn(30, 3).astype(np.float32)
    path = np_infer.bio_path(bio_logits, weights)
    assert path == bio_logits.argmax(-1).tolist()
