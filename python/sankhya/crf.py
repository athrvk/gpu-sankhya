"""Linear-chain CRF over the 3-label BIO head (O=0, B=1, I=2).

See CRF_CONTRACT.md for the exact semantics this must match (the JS port
mirrors the numpy `viterbi_np` loop exactly, tap for tap).

Path score(y) = start[y0] + sum_t emis[t][y_t] + sum_{t>0} trans[y_{t-1}][y_t]
              + end[y_{L-1}]

Training NLL (per example) = logsumexp over all paths of that score minus
the gold path's score, computed only over the real (unpadded) positions.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

N_LABELS = 3


class CRF(nn.Module):
    """trans[i][j]: score of transitioning label i -> label j.
    start[j]: score of the first label being j. end[j]: score of the last
    label being j. All zero-initialized (contract: "No hard constraints
    baked in -- the model learns them")."""

    def __init__(self, n_labels: int = N_LABELS):
        super().__init__()
        self.n_labels = n_labels
        self.trans = nn.Parameter(torch.zeros(n_labels, n_labels))
        self.start = nn.Parameter(torch.zeros(n_labels))
        self.end = nn.Parameter(torch.zeros(n_labels))

    def nll(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """emissions: (B, L, 3) float. tags: (B, L) int64 gold labels.
        mask: (B, L) float/bool, 1 for real (unpadded) positions.
        Returns a scalar: mean over the batch of (logZ - score(gold)),
        computed using only the real positions of each example (forward
        algorithm in log space via torch.logsumexp)."""
        B, L, K = emissions.shape
        mask = mask.bool()
        device = emissions.device

        # ---- gold path score, real positions only ----
        lengths = mask.sum(dim=1).long()  # (B,)
        gold_score = torch.zeros(B, device=device)
        for b in range(B):
            Lb = int(lengths[b].item())
            if Lb == 0:
                continue
            y = tags[b, :Lb]
            e = emissions[b, :Lb]
            score = self.start[y[0]] + e[0, y[0]]
            for t in range(1, Lb):
                score = score + self.trans[y[t - 1], y[t]] + e[t, y[t]]
            score = score + self.end[y[Lb - 1]]
            gold_score[b] = score

        # ---- log partition (forward algorithm), real positions only ----
        logZ = torch.zeros(B, device=device)
        for b in range(B):
            Lb = int(lengths[b].item())
            if Lb == 0:
                continue
            e = emissions[b, :Lb]
            alpha = self.start + e[0]  # (K,)
            for t in range(1, Lb):
                # alpha_next[j] = logsumexp_i(alpha[i] + trans[i][j]) + e[t][j]
                scores = alpha.unsqueeze(1) + self.trans  # (K_i, K_j)
                alpha = torch.logsumexp(scores, dim=0) + e[t]
            logZ[b] = torch.logsumexp(alpha + self.end, dim=0)

        return (logZ - gold_score).mean()

    def viterbi(self, emissions: torch.Tensor) -> list:
        """Torch convenience wrapper for a single example's emissions
        (L, 3): delegates to the numpy reference implementation so training
        eval and export/np_infer share one algorithm."""
        emis_np = emissions.detach().cpu().numpy().astype(np.float32)
        trans_np = self.trans.detach().cpu().numpy().astype(np.float32)
        start_np = self.start.detach().cpu().numpy().astype(np.float32)
        end_np = self.end.detach().cpu().numpy().astype(np.float32)
        return viterbi_np(emis_np, trans_np, start_np, end_np)


def viterbi_np(emis: np.ndarray, trans: np.ndarray, start: np.ndarray, end: np.ndarray) -> list:
    """Reference Viterbi decode, per CRF_CONTRACT.md, float32 throughout,
    first-max tie-break (iterate i=0,1,2 and replace only on strictly
    greater). emis: (L, 3) float32. trans/start/end: (3,3)/(3,)/(3,) float32.
    L == 0 -> [] (empty path)."""
    emis = np.asarray(emis, dtype=np.float32)
    trans = np.asarray(trans, dtype=np.float32)
    start = np.asarray(start, dtype=np.float32)
    end = np.asarray(end, dtype=np.float32)

    L, K = emis.shape
    if L == 0:
        return []

    delta = np.zeros((L, K), dtype=np.float32)
    back = np.zeros((L, K), dtype=np.int64)

    for j in range(K):
        delta[0, j] = start[j] + emis[0, j]

    for t in range(1, L):
        for j in range(K):
            best_i = 0
            best_val = delta[t - 1, 0] + trans[0, j]
            for i in range(1, K):
                val = delta[t - 1, i] + trans[i, j]
                if val > best_val:
                    best_val = val
                    best_i = i
            delta[t, j] = best_val + emis[t, j]
            back[t, j] = best_i

    best_last = 0
    best_val = delta[L - 1, 0] + end[0]
    for j in range(1, K):
        val = delta[L - 1, j] + end[j]
        if val > best_val:
            best_val = val
            best_last = j

    path = [0] * L
    path[L - 1] = best_last
    for t in range(L - 1, 0, -1):
        path[t - 1] = int(back[t, path[t]])
    return path
