"""CNN model for gpu-sankhya char-level BIO + class tagging."""
from __future__ import annotations

import torch
import torch.nn as nn


class SankhyaCNN(nn.Module):
    def __init__(self, vocab_size: int, n_cls: int, dilation: int = 1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 16, padding_idx=0)
        self.conv1 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=5, padding=2)
        pad3 = dilation * (3 - 1) // 2
        self.conv3 = nn.Conv1d(32, 32, kernel_size=3, padding=pad3, dilation=dilation)
        self.act = nn.ReLU()
        self.bio_head = nn.Linear(32, 3)
        self.cls_head = nn.Linear(32, n_cls)

    def forward(self, chars: torch.Tensor):
        # chars: (B, L) int64
        x = self.embed(chars)          # (B, L, 16)
        x = x.transpose(1, 2)          # (B, 16, L)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = x.transpose(1, 2)          # (B, L, 32)
        bio = self.bio_head(x)
        cls = self.cls_head(x)
        return bio, cls


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    from . import classes as C
    m = SankhyaCNN(vocab_size=70, n_cls=len(C.CLASSES))
    n = count_params(m)
    print(f"param count: {n}")
