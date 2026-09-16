"""Reference numpy forward pass for SankhyaCNN, mirroring model.py exactly.

Kept simple and explicit (loops over kernel taps + matmul) so a JS port can
mirror it 1:1. Weights come from the JSON export (float or int8-dequantized).

IMPORTANT -- padding parity with training (read this before porting to JS):
training always right-pads every example to MAX_LEN=128 with char id 0 (the
"<pad>" embedding, which has non-zero learned features) before running the
conv stack, so every real character in training saw at least a long run of
pad-id context to its right. At inference, running the forward pass on a
SHORT, TIGHTLY-CROPPED input (e.g. just "2.5L", 4 chars, no padding) puts the
last few real characters at the literal edge of the array, where conv1d's
zero-padding (numeric 0.0, NOT the id-0 embedding) kicks in instead -- a
distribution mismatch that measurably corrupts predictions on short/bare
inputs (verified: "2.5L" tags the "L" as UNIT_LAKH when padded, but as
UNIT_HAZAAR/UNIT_BILLION/UNIT_CRORE when run tight/unpadded).

The fix used everywhere in this repo (Python AND the JS port MUST mirror
this exactly): before calling forward(), right-pad the char-id array with at
least PAD_TAIL=16 copies of pad id 0 via pad_ids() -- 16 is comfortably past
the 4-layer network's receptive field (+/-9 chars) so the padded tail fully
reproduces the training-time context. Run forward() on the padded array,
then slice bio_logits/cls_logits back down to the ORIGINAL (unpadded) text
length before argmax/decode -- the padded tail's outputs are discarded, only
used to give the real tokens correct right-context.
"""
from __future__ import annotations

import base64

import numpy as np

PAD_TAIL = 16


def pad_ids(ids, tail: int = PAD_TAIL):
    """Right-pad a 1-D sequence of char ids with `tail` copies of pad id 0.
    Accepts a list or a 1-D numpy array; returns the same type. Always pad
    single-example inference inputs with this before calling forward() --
    see the module docstring for why this must match training-time padding.
    """
    if isinstance(ids, np.ndarray):
        return np.concatenate([ids, np.zeros(tail, dtype=ids.dtype)])
    return list(ids) + [0] * tail


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


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax, used to turn bio_logits into bio_probs for
    decode_spans' confidence gate (the JS runtime applies this in production,
    so every Python decode_spans call site must too -- see decode.py)."""
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def forward(weights: dict, char_ids: np.ndarray, dilation: int = 2, layers: int = 3):
    """char_ids: (L,) int array of char ids (single example, no batch/padding needed
    -- caller may pass a full length row). Returns (bio_logits (L,3), cls_logits (L,n_cls))."""
    embed = weights["embed"]  # (V, 16)
    x = embed[char_ids].T.astype(np.float32)  # (16, L)

    x = _relu(_conv1d(x, weights["conv1"]["w"], weights["conv1"]["b"], padding=1, dilation=1))
    x = _relu(_conv1d(x, weights["conv2"]["w"], weights["conv2"]["b"], padding=2, dilation=1))
    pad3 = dilation * (3 - 1) // 2
    x = _relu(_conv1d(x, weights["conv3"]["w"], weights["conv3"]["b"], padding=pad3, dilation=dilation))
    if layers == 4:
        dilation4 = 4
        pad4 = dilation4 * (3 - 1) // 2
        x = _relu(_conv1d(x, weights["conv4"]["w"], weights["conv4"]["b"], padding=pad4, dilation=dilation4))

    # x: (C, L) -> (L, C)
    xt = x.T
    bio_logits = xt @ weights["bio"]["w"].T + weights["bio"]["b"]
    cls_logits = xt @ weights["cls"]["w"].T + weights["cls"]["b"]
    return bio_logits, cls_logits


def load_weights_json(obj: dict) -> dict:
    """Load a float weights JSON (as written by export.py) into numpy arrays."""
    def arr(t):
        return np.array(t["data"], dtype=np.float32).reshape(t["shape"])

    out = {"embed": arr(obj["embed"])}
    names = ["conv1", "conv2", "conv3"] + (["conv4"] if obj.get("layers", 3) == 4 else []) + ["bio", "cls"]
    for name in names:
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
    names = ["conv1", "conv2", "conv3"] + (["conv4"] if obj.get("layers", 3) == 4 else []) + ["bio", "cls"]
    for name in names:
        out[name] = {"w": dequant(obj[name]["w"]), "b": bias(obj[name]["b"])}
    return out
