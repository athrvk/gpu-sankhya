"""Export best checkpoint to ONNX + compact JSON weights (float + int8)."""
from __future__ import annotations

import argparse
import base64
import json
import os

import numpy as np
import onnx
import onnxruntime as ort
import torch

from . import classes as C
from .model import SankhyaCNN
from .train import load_jsonl, tensorize, build_char_to_id, MAX_LEN
from . import np_infer
from .decode import decode_spans
from . import core


def sig(x, n=5):
    """Round to n significant digits."""
    x = float(x)
    if x == 0:
        return 0.0
    from math import log10, floor
    d = n - 1 - floor(log10(abs(x)))
    return round(x, d)


def round_arr(a, n=5):
    flat = a.reshape(-1)
    return [sig(v, n) for v in flat.tolist()]


def export_onnx(model, vocab_size, out_path):
    model.eval()
    dummy = torch.randint(0, vocab_size, (1, 10), dtype=torch.int64)
    torch.onnx.export(
        model,
        (dummy,),
        out_path,
        input_names=["chars"],
        output_names=["bio", "cls"],
        dynamic_axes={
            "chars": {0: "batch", 1: "length"},
            "bio": {0: "batch", 1: "length"},
            "cls": {0: "batch", 1: "length"},
        },
        opset_version=17,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(out_path))


def verify_onnx(model, onnx_path, chars_np, n=50):
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    max_diff = 0.0
    with torch.no_grad():
        for i in range(min(n, chars_np.shape[0])):
            row = chars_np[i:i + 1]
            t_bio, t_cls = model(torch.from_numpy(row))
            o_bio, o_cls = sess.run(["bio", "cls"], {"chars": row})
            max_diff = max(max_diff, np.abs(t_bio.numpy() - o_bio).max(), np.abs(t_cls.numpy() - o_cls).max())
    return max_diff


def build_weights_json(model, vocab, classes):
    sd = model.state_dict()
    def t(x):
        return x.detach().cpu().numpy()

    embed = t(sd["embed.weight"])
    out = {
        "version": 1,
        "charset": vocab,
        "classes": classes,
        "embed": {"shape": list(embed.shape), "data": round_arr(embed)},
    }
    for name, key in [("conv1", "conv1"), ("conv2", "conv2"), ("conv3", "conv3"),
                       ("bio", "bio_head"), ("cls", "cls_head")]:
        w = t(sd[f"{key}.weight"])
        b = t(sd[f"{key}.bias"])
        out[name] = {
            "w": {"shape": list(w.shape), "data": round_arr(w)},
            "b": {"shape": list(b.shape), "data": round_arr(b)},
        }
    return out


def quantize_tensor_int8(arr: np.ndarray):
    amax = np.abs(arr).max()
    scale = float(amax / 127.0) if amax > 0 else 1.0
    q = np.clip(np.round(arr / scale), -127, 127).astype(np.int8)
    return q, scale


def build_weights_int8_json(model, vocab, classes):
    sd = model.state_dict()
    def t(x):
        return x.detach().cpu().numpy()

    def qtensor(arr):
        q, scale = quantize_tensor_int8(arr)
        return {"shape": list(arr.shape), "scale": scale, "data_b64": base64.b64encode(q.tobytes()).decode("ascii")}

    embed = t(sd["embed.weight"])
    out = {
        "version": 1,
        "quantized": "int8_symmetric_per_tensor",
        "charset": vocab,
        "classes": classes,
        "embed": qtensor(embed),
    }
    for name, key in [("conv1", "conv1"), ("conv2", "conv2"), ("conv3", "conv3"),
                       ("bio", "bio_head"), ("cls", "cls_head")]:
        w = t(sd[f"{key}.weight"])
        b = t(sd[f"{key}.bias"])
        out[name] = {"w": qtensor(w), "b": [sig(v) for v in b.tolist()]}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/sankhya.pt")
    ap.add_argument("--val", default="data/val.jsonl")
    ap.add_argument("--out", default="models/")
    args = ap.parse_args(argv)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    vocab = ckpt["vocab"]
    classes = ckpt["classes"]
    dilation = ckpt.get("dilation", 1)
    model = SankhyaCNN(vocab_size=len(vocab), n_cls=len(classes), dilation=dilation)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    val_ex = load_jsonl(args.val)
    char_to_id = build_char_to_id(vocab)
    va_chars, va_bio, va_cls, va_mask = tensorize(val_ex, char_to_id)
    chars_np = va_chars.numpy()

    onnx_path = os.path.join(args.out, "sankhya.onnx")
    export_onnx(model, len(vocab), onnx_path)
    max_diff = verify_onnx(model, onnx_path, chars_np, n=50)
    print(f"onnx max abs diff vs torch (50 val examples): {max_diff:.6g}")
    assert max_diff < 1e-4, f"onnx parity failed: {max_diff}"

    weights = build_weights_json(model, vocab, classes)
    with open(os.path.join(args.out, "sankhya.weights.json"), "w", encoding="utf-8") as f:
        json.dump(weights, f)
    weights_int8 = build_weights_int8_json(model, vocab, classes)
    with open(os.path.join(args.out, "sankhya.weights.int8.json"), "w", encoding="utf-8") as f:
        json.dump(weights_int8, f)
    print("wrote sankhya.weights.json and sankhya.weights.int8.json")

    # Verify np_infer float matches torch, and int8 value-accuracy loss < 0.5%.
    w_float = np_infer.load_weights_json(weights)
    w_int8 = np_infer.load_weights_int8_json(weights_int8)

    def value_acc_for(weights_np):
        correct = total = 0
        for i, ex in enumerate(val_ex):
            text = ex["text"]
            L = min(len(text), MAX_LEN)
            row = chars_np[i, :L]
            bio_logits, cls_logits = np_infer.forward(weights_np, row, dilation=dilation)
            bio_pred = bio_logits.argmax(-1).tolist()
            cls_pred = cls_logits.argmax(-1).tolist()
            decoded = decode_spans(text, bio_pred, cls_pred)
            pred_by_span = {(d["start"], d["end"]): d for d in decoded}
            for sp in ex["spans"]:
                total += 1
                key = (sp["start"], sp["end"])
                if key in pred_by_span:
                    toks = [(classes[cid], sub) for cid, sub in pred_by_span[key]["tokens"]]
                    res = core.evaluate(toks)
                    if res.value == sp["value"]:
                        correct += 1
        return correct / total if total else 0.0

    acc_float = value_acc_for(w_float)
    acc_int8 = value_acc_for(w_int8)
    print(f"np_infer float value_acc={acc_float:.4f}  int8 value_acc={acc_int8:.4f}  drop={acc_float-acc_int8:.4f}")
    if acc_float - acc_int8 >= 0.005:
        print("WARNING: int8 quantization loses >=0.5% value accuracy")

    # cross-check np float forward vs torch forward on a couple examples
    with torch.no_grad():
        for i in range(3):
            L = min(len(val_ex[i]["text"]), MAX_LEN)
            row = chars_np[i, :L]
            t_bio, t_cls = model(torch.from_numpy(chars_np[i:i+1, :L]))
            n_bio, n_cls = np_infer.forward(w_float, row, dilation=dilation)
            d = max(np.abs(t_bio[0].numpy() - n_bio).max(), np.abs(t_cls[0].numpy() - n_cls).max())
            print(f"np_infer vs torch example {i} max diff: {d:.6g}")


if __name__ == "__main__":
    main()
