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
least PAD_TAIL=24 copies of pad id 0 via pad_ids() -- 24 is comfortably past
the widest arch's receptive field (v2's is +/-17 chars) so the padded tail
fully reproduces the training-time context. Run forward() on the padded
array, then slice bio_logits/cls_logits back down to the ORIGINAL (unpadded)
text length before argmax/decode -- the padded tail's outputs are discarded,
only used to give the real tokens correct right-context.
"""
from __future__ import annotations

import base64

import numpy as np

PAD_TAIL = 24


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


def forward(weights: dict, char_ids: np.ndarray, dilation=None, layers=None):
    """char_ids: (L,) int array of char ids (single example, no batch/padding needed
    -- caller may pass a full length row). Returns (bio_logits (L,3), cls_logits (L,n_cls)).

    `dilation`/`layers` are accepted-but-ignored (kept for one release so old
    call sites don't break) -- the arch is now fully described by
    weights["conv"], an ordered list of {k, dilation, residual, w, b}.
    """
    embed = weights["embed"]  # (V, E)
    x = embed[char_ids].T.astype(np.float32)  # (E, L)

    for layer in weights["conv"]:
        k = layer["k"]
        d = layer["dilation"]
        pad = d * (k - 1) // 2
        y = _relu(_conv1d(x, layer["w"], layer["b"], padding=pad, dilation=d))
        if layer["residual"]:
            y = y + x
        x = y

    # x: (C, L) -> (L, C)
    xt = x.T
    bio_logits = xt @ weights["bio"]["w"].T + weights["bio"]["b"]
    cls_logits = xt @ weights["cls"]["w"].T + weights["cls"]["b"]
    return bio_logits, cls_logits


def _v1_conv_specs(obj):
    """v1 file -> ordered list of (json_key, k, dilation, residual)."""
    layers = obj.get("layers", 3)
    dilation = obj.get("dilation", 2)
    specs = [("conv1", 3, 1, False), ("conv2", 5, 1, False), ("conv3", 3, dilation, False)]
    if layers == 4:
        specs.append(("conv4", 3, 4, False))
    return specs


def load_weights_json(obj: dict) -> dict:
    """Load a float weights JSON (as written by export.py) into numpy arrays.
    Accepts both v1 (conv1..conv4 top-level keys) and v2 (weights["conv"]
    list) files and normalises to {"embed", "conv": [...], "bio", "cls"}."""
    def arr(t):
        return np.array(t["data"], dtype=np.float32).reshape(t["shape"])

    out = {"embed": arr(obj["embed"])}
    if obj.get("version", 1) >= 2 and "conv" in obj:
        conv = []
        for layer in obj["conv"]:
            conv.append({
                "k": layer["k"], "dilation": layer["dilation"], "residual": layer["residual"],
                "w": arr(layer["w"]), "b": arr(layer["b"]),
            })
        out["conv"] = conv
    else:
        conv = []
        for key, k, d, residual in _v1_conv_specs(obj):
            conv.append({"k": k, "dilation": d, "residual": residual, "w": arr(obj[key]["w"]), "b": arr(obj[key]["b"])})
        out["conv"] = conv
    out["bio"] = {"w": arr(obj["bio"]["w"]), "b": arr(obj["bio"]["b"])}
    out["cls"] = {"w": arr(obj["cls"]["w"]), "b": arr(obj["cls"]["b"])}
    return out


def load_weights_int8_json(obj: dict) -> dict:
    """Load an int8-quantized weights JSON and dequantize to float32 numpy
    arrays. Accepts both v1 and v2 files, same normalisation as
    load_weights_json."""
    def dequant(t):
        raw = base64.b64decode(t["data_b64"])
        arr = np.frombuffer(raw, dtype=np.int8).astype(np.float32).reshape(t["shape"])
        return arr * t["scale"]

    def bias(t):
        return np.array(t, dtype=np.float32)

    out = {"embed": dequant(obj["embed"])}
    if obj.get("version", 1) >= 2 and "conv" in obj:
        conv = []
        for layer in obj["conv"]:
            conv.append({
                "k": layer["k"], "dilation": layer["dilation"], "residual": layer["residual"],
                "w": dequant(layer["w"]), "b": bias(layer["b"]),
            })
        out["conv"] = conv
    else:
        conv = []
        for key, k, d, residual in _v1_conv_specs(obj):
            conv.append({"k": k, "dilation": d, "residual": residual, "w": dequant(obj[key]["w"]), "b": bias(obj[key]["b"])})
        out["conv"] = conv
    out["bio"] = {"w": dequant(obj["bio"]["w"]), "b": bias(obj["bio"]["b"])}
    out["cls"] = {"w": dequant(obj["cls"]["w"]), "b": bias(obj["cls"]["b"])}
    return out
