"""CNN model for gpu-sankhya char-level BIO + class tagging.

The conv stack is a generic ordered list of layers ("arch"), each
`{k, dilation, residual}` (`k` always odd; `padding = dilation*(k-1)//2`
keeps the sequence length unchanged). See CONTRACT.md section 1 for the
named presets and the JSON weights format they map to.
"""
from __future__ import annotations

import argparse

import torch
import torch.nn as nn

# Named arch presets shared with np_infer.py / export.py / train.py / JS.
# "v1" is the shipped model's exact numerics (no residuals); legacy
# `dilation`/`layers` kwargs below build this same list.
ARCHS = {
    "v1": [
        {"k": 3, "dilation": 1, "residual": False},
        {"k": 5, "dilation": 1, "residual": False},
        {"k": 3, "dilation": 2, "residual": False},
        {"k": 3, "dilation": 4, "residual": False},
    ],
    "v2": [
        {"k": 5, "dilation": 1, "residual": False},
        {"k": 3, "dilation": 1, "residual": True},
        {"k": 3, "dilation": 2, "residual": True},
        {"k": 3, "dilation": 4, "residual": True},
        {"k": 3, "dilation": 8, "residual": True},
    ],
}


def _legacy_arch(dilation: int, layers: int):
    """Build the v1-shaped arch list from the old (dilation, layers) kwargs.
    layers=3 -> first three of ARCHS["v1"] with the third's dilation
    overridden to `dilation`; layers=4 -> all four, third dilation still
    overridden, fourth stays d4. Matches CONTRACT.md section 1 exactly."""
    assert layers in (3, 4), "layers must be 3 or 4"
    base = [dict(layer) for layer in ARCHS["v1"][:layers]]
    base[2]["dilation"] = dilation
    return base


class SankhyaCNN(nn.Module):
    def __init__(self, vocab_size: int, n_cls: int, arch=None, channels: int = 32,
                 embed_dim: int = 16, dilation: int = 1, layers: int = 3):
        """`arch` (list of {k, dilation, residual} dicts) takes priority when
        given. Otherwise the legacy `dilation`/`layers` kwargs reconstruct
        the exact v1 arch, for old checkpoints/call sites."""
        super().__init__()
        self.arch = [dict(layer) for layer in arch] if arch is not None else _legacy_arch(dilation, layers)
        # kept for backward compat with code that still reads model.layers
        self.layers = len(self.arch)
        self.embed_dim = embed_dim
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Register conv modules under conv1, conv2, ... (not a ModuleList
        # index) so state_dict keys stay "conv1.weight" etc -- this is what
        # lets the shipped v1 checkpoint/weights JSON keep loading unchanged.
        self.convs = nn.ModuleList()
        c_in = embed_dim
        for i, layer in enumerate(self.arch):
            k = layer["k"]
            d = layer["dilation"]
            pad = d * (k - 1) // 2
            conv = nn.Conv1d(c_in, channels, kernel_size=k, padding=pad, dilation=d)
            setattr(self, f"conv{i+1}", conv)
            self.convs.append(conv)
            c_in = channels

        self.act = nn.ReLU()
        self.bio_head = nn.Linear(channels, 3)
        self.cls_head = nn.Linear(channels, n_cls)

    def forward(self, chars: torch.Tensor):
        # chars: (B, L) int64
        x = self.embed(chars)          # (B, L, embed_dim)
        x = x.transpose(1, 2)          # (B, embed_dim, L)
        for layer, conv in zip(self.arch, self.convs):
            y = self.act(conv(x))
            if layer["residual"]:
                y = y + x
            x = y
        x = x.transpose(1, 2)          # (B, L, C)
        bio = self.bio_head(x)
        cls = self.cls_head(x)
        return bio, cls


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    from . import classes as C
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--arch", default=None, choices=list(ARCHS.keys()))
    ap.add_argument("--layers", type=int, default=3, choices=[3, 4])
    ap.add_argument("--vocab-size", type=int, default=70)
    args = ap.parse_args()
    arch = ARCHS[args.arch] if args.arch else None
    m = SankhyaCNN(vocab_size=args.vocab_size, n_cls=len(C.CLASSES), arch=arch,
                    channels=args.channels, layers=args.layers)
    n = count_params(m)
    print(f"arch={args.arch or 'legacy'} channels={args.channels} layers={len(m.arch)} param count: {n}")
