"""Reference numpy forward pass for SankhyaCNN, mirroring model.py exactly.

Kept simple and explicit (loops over kernel taps + matmul) so a JS port can
mirror it 1:1. Weights come from the JSON export (float or int8-dequantized).
"""
from __future__ import annotations

import base64

import numpy as np


def _conv1d(x, w, b, padding, dilation=1):
    """x: (C_in, L). w: (C_out, C_in, K). b: (C_out,). Returns (C_out, L)."""
    c_out, c_in, k = w.shape
    L = x.shape[1]
    xp = np.zeros((c_in, L + 2 * padding), dtype=np.float32)
    xp[:, padding:padding + L] = x
    out = np.zeros((c_out, L), dtype=np.float32)
    for t in range(k):
        offset = t * dilation
        # slice of xp aligned for this tap, length L
        seg = xp[:, offset:offset + L]  # (C_in, L)
        out += w[:, :, t] @ seg  # (C_out, C_in) @ (C_in, L) -> (C_out, L)
    out += b[:, None]
    return out


def _relu(x):
    return np.maximum(x, 0.0)


def forward(weights: dict, char_ids: np.ndarray, dilation: int = 2):
    """char_ids: (L,) int array of char ids (single example, no batch/padding needed
    -- caller may pass a full length row). Returns (bio_logits (L,3), cls_logits (L,n_cls))."""
    embed = weights["embed"]  # (V, 16)
    x = embed[char_ids].T.astype(np.float32)  # (16, L)

    x = _relu(_conv1d(x, weights["conv1"]["w"], weights["conv1"]["b"], padding=1, dilation=1))
    x = _relu(_conv1d(x, weights["conv2"]["w"], weights["conv2"]["b"], padding=2, dilation=1))
    pad3 = dilation * (3 - 1) // 2
    x = _relu(_conv1d(x, weights["conv3"]["w"], weights["conv3"]["b"], padding=pad3, dilation=dilation))

    # x: (32, L) -> (L, 32)
    xt = x.T
    bio_logits = xt @ weights["bio"]["w"].T + weights["bio"]["b"]
    cls_logits = xt @ weights["cls"]["w"].T + weights["cls"]["b"]
    return bio_logits, cls_logits


def load_weights_json(obj: dict) -> dict:
    """Load a float weights JSON (as written by export.py) into numpy arrays."""
    def arr(t):
        return np.array(t["data"], dtype=np.float32).reshape(t["shape"])

    out = {"embed": arr(obj["embed"])}
    for name in ("conv1", "conv2", "conv3", "bio", "cls"):
        out[name] = {"w": arr(obj[name]["w"]), "b": arr(obj[name]["b"])}
    return out


def load_weights_int8_json(obj: dict) -> dict:
    """Load an int8-quantized weights JSON and dequantize to float32 numpy arrays."""
    def dequant(t):
        raw = base64.b64decode(t["data_b64"])
        arr = np.frombuffer(raw, dtype=np.int8).astype(np.float32).reshape(t["shape"])
        return arr * t["scale"]

    def bias(t):
        return np.array(t, dtype=np.float32)

    out = {"embed": dequant(obj["embed"])}
    for name in ("conv1", "conv2", "conv3", "bio", "cls"):
        out[name] = {"w": dequant(obj[name]["w"]), "b": bias(obj[name]["b"])}
    return out
