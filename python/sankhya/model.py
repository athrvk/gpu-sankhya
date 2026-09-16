"""CNN model for gpu-sankhya char-level BIO + class tagging."""
from __future__ import annotations

import argparse

import torch
import torch.nn as nn


class SankhyaCNN(nn.Module):
    def __init__(self, vocab_size: int, n_cls: int, dilation: int = 1, channels: int = 32, layers: int = 3):
        super().__init__()
        assert layers in (3, 4), "layers must be 3 or 4"
        self.embed = nn.Embedding(vocab_size, 16, padding_idx=0)
        self.conv1 = nn.Conv1d(16, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=5, padding=2)
        pad3 = dilation * (3 - 1) // 2
        self.conv3 = nn.Conv1d(channels, channels, kernel_size=3, padding=pad3, dilation=dilation)
        self.layers = layers
        if layers == 4:
            dilation4 = 4
            pad4 = dilation4 * (3 - 1) // 2
            self.conv4 = nn.Conv1d(channels, channels, kernel_size=3, padding=pad4, dilation=dilation4)
        self.act = nn.ReLU()
        self.bio_head = nn.Linear(channels, 3)
        self.cls_head = nn.Linear(channels, n_cls)

    def forward(self, chars: torch.Tensor):
        # chars: (B, L) int64
        x = self.embed(chars)          # (B, L, 16)
        x = x.transpose(1, 2)          # (B, 16, L)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        if self.layers == 4:
            x = self.act(self.conv4(x))
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
    ap.add_argument("--layers", type=int, default=3, choices=[3, 4])
    ap.add_argument("--vocab-size", type=int, default=70)
    args = ap.parse_args()
    m = SankhyaCNN(vocab_size=args.vocab_size, n_cls=len(C.CLASSES), channels=args.channels, layers=args.layers)
    n = count_params(m)
    print(f"channels={args.channels} layers={args.layers} param count: {n}")
