"""Tests for the generalised conv-stack model (arch presets, np_infer parity)."""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from sankhya import classes as C
from sankhya import export, np_infer
from sankhya.model import ARCHS, SankhyaCNN

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "src", "data")


def test_legacy_kwargs_build_v1_arch():
    m3 = SankhyaCNN(vocab_size=10, n_cls=5, dilation=2, layers=3)
    expect3 = [dict(l) for l in ARCHS["v1"][:3]]
    expect3[2]["dilation"] = 2
    assert m3.arch == expect3

    m4 = SankhyaCNN(vocab_size=10, n_cls=5, dilation=2, layers=4)
    expect4 = [dict(l) for l in ARCHS["v1"][:4]]
    expect4[2]["dilation"] = 2
    assert m4.arch == expect4

    # dilation=2 (today's default python call-site value), layers=4 matches ARCHS["v1"] exactly
    m_default = SankhyaCNN(vocab_size=10, n_cls=5, dilation=2, layers=4)
    assert m_default.arch == ARCHS["v1"]


def test_legacy_model_state_dict_has_conv1_conv4_keys():
    m = SankhyaCNN(vocab_size=10, n_cls=5, dilation=2, layers=4)
    sd = m.state_dict()
    for name in ["conv1", "conv2", "conv3", "conv4"]:
        assert f"{name}.weight" in sd
        assert f"{name}.bias" in sd


def test_v2_forward_shapes():
    n_cls = len(C.CLASSES)
    m = SankhyaCNN(vocab_size=30, n_cls=n_cls, arch=ARCHS["v2"])
    x = torch.randint(0, 30, (2, 40), dtype=torch.int64)
    bio, cls = m(x)
    assert bio.shape == (2, 40, 3)
    assert cls.shape == (2, 40, n_cls)


def test_np_infer_matches_torch_v2():
    n_cls = len(C.CLASSES)
    vocab_size = 30
    torch.manual_seed(0)
    m = SankhyaCNN(vocab_size=vocab_size, n_cls=n_cls, arch=ARCHS["v2"], channels=8, embed_dim=6)
    m.eval()

    vocab = [f"c{i}" for i in range(vocab_size)]
    classes = C.CLASSES
    weights_json = export.build_weights_json(m, vocab, classes)
    weights_int8_json = export.build_weights_int8_json(m, vocab, classes)

    w_float = np_infer.load_weights_json(weights_json)
    w_int8 = np_infer.load_weights_int8_json(weights_int8_json)

    rng = np.random.RandomState(0)
    raw_ids = rng.randint(0, vocab_size, size=20).astype(np.int64)
    padded = np_infer.pad_ids(raw_ids)

    with torch.no_grad():
        t_bio, t_cls = m(torch.from_numpy(padded[None, :]))
    t_bio, t_cls = t_bio[0, :20].numpy(), t_cls[0, :20].numpy()

    n_bio, n_cls_ = np_infer.forward(w_float, padded)
    n_bio, n_cls_ = n_bio[:20], n_cls_[:20]

    assert np.abs(t_bio - n_bio).max() < 1e-4
    assert np.abs(t_cls - n_cls_).max() < 1e-4

    # int8 should be close but not necessarily within 1e-4; sanity check shape/finite
    i_bio, i_cls = np_infer.forward(w_int8, padded)
    assert i_bio[:20].shape == n_bio.shape
    assert np.isfinite(i_bio).all() and np.isfinite(i_cls).all()


def test_load_default_weights_v1_int8():
    path = os.path.join(FIXTURES_DIR, "default-weights.json")
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    weights = np_infer.load_weights_int8_json(obj)
    conv = weights["conv"]
    assert len(conv) == 4
    assert [layer["k"] for layer in conv] == [3, 5, 3, 3]
    assert [layer["dilation"] for layer in conv] == [1, 1, 2, 4]
    assert all(layer["residual"] is False for layer in conv)
