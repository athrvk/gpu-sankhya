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

    def nll(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor, return_logz: bool = False):
        """emissions: (B, L, 3) float. tags: (B, L) int64 gold labels.
        mask: (B, L) float/bool, 1 for real (unpadded) positions.
        Returns a scalar: mean over the examples with length > 0 of
        (logZ - score(gold)), computed using only the real positions of
        each example (forward algorithm in log space via
        torch.logsumexp), vectorised over the batch with a single Python
        loop over time positions."""
        B, L, K = emissions.shape
        mask = mask.bool()
        device = emissions.device

        lengths = mask.sum(dim=1).long()  # (B,)
        has_len = lengths > 0

        # ---- gold path score, real positions only ----
        # emissions at gold tags: (B, L)
        e_gold = emissions.gather(2, tags.unsqueeze(-1)).squeeze(-1)
        emis_score = (e_gold * mask.float()).sum(dim=1)

        # transitions: trans[tags[:, t-1], tags[:, t]] counted when position
        # t (the destination) is real.
        trans_score = torch.zeros(B, device=device)
        if L > 1:
            trans_vals = self.trans[tags[:, :-1], tags[:, 1:]]  # (B, L-1)
            trans_score = (trans_vals * mask[:, 1:].float()).sum(dim=1)

        start_vals = self.start[tags[:, 0]] * mask[:, 0].float()

        # end score at the last real position of each example.
        clamped_last = (lengths - 1).clamp(min=0)
        last_tags = tags.gather(1, clamped_last.unsqueeze(1)).squeeze(1)
        end_vals = self.end[last_tags] * has_len.float()

        gold_score = emis_score + trans_score + start_vals + end_vals

        # ---- log partition (forward algorithm), real positions only ----
        # Loop only out to the batch's longest real sequence -- steps beyond
        # that are masked out for every row (a no-op on alpha), so running
        # them is wasted launch-bound work, especially on GPU where each
        # step is several small kernel launches.
        Lmax = int(lengths.max().item()) if B > 0 else 0
        trans_b = self.trans.unsqueeze(0)  # (1, K_i, K_j), precomputed once
        emis_t = emissions.transpose(0, 1).contiguous()  # (L, B, K), contiguous per-t slices
        alpha = self.start.unsqueeze(0) + emis_t[0]  # (B, K)
        for t in range(1, Lmax):
            # new[b, j] = logsumexp_i(alpha[b, i] + trans[i, j]) + e[b, t, j]
            alpha = torch.where(
                mask[:, t].unsqueeze(1),
                torch.logsumexp(alpha.unsqueeze(2) + trans_b, dim=1) + emis_t[t],
                alpha,
            )
        logZ = torch.logsumexp(alpha + self.end.unsqueeze(0), dim=1)
        logZ = logZ * has_len.float()

        # zero-length examples contribute 0 to both logZ and gold_score, so
        # this matches _nll_reference's plain batch mean exactly.
        per_example = logZ - gold_score
        nll = per_example.mean()
        if return_logz:
            return nll, logZ
        return nll

    def _nll_reference(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Reference (slow, per-example Python loop) implementation of
        `nll`, kept for testing the vectorised version against."""
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

    @torch.no_grad()
    def viterbi_batch(self, emissions: torch.Tensor, mask: torch.Tensor) -> list:
        """Batched torch Viterbi decode, vectorised over the batch with a
        single loop over time. emissions: (B, L, K). mask: (B, L)
        float/bool, 1 for real (unpadded) positions. Returns a list of B
        decoded paths (plain Python lists of ints, one per example, each
        of length = that example's real length), identical to calling
        `viterbi_np` on each example's sliced-to-length emissions with
        this module's trans/start/end (same first-max tie-break: iterate
        i=0,1,2 and replace only on strictly-greater comparisons, done
        explicitly here rather than relying on torch.max's tie behaviour).
        """
        B, L, K = emissions.shape
        device = emissions.device
        mask = mask.bool()
        lengths = mask.sum(dim=1).long()  # (B,)
        trans = self.trans
        start = self.start
        end = self.end

        if L == 0:
            return [[] for _ in range(B)]

        # As in nll: run the recursion only out to the batch's longest real
        # sequence -- every row's `clamped_last` is <= Lmax-1, so steps
        # beyond that never feed the final argmax or the backtrack below.
        Lmax = int(lengths.max().item()) if B > 0 else 0
        Lmax = max(Lmax, 1)  # keep delta[:, 0] valid even for an all-empty batch

        delta = torch.empty(B, Lmax, K, device=device, dtype=emissions.dtype)
        back = torch.zeros(B, Lmax, K, dtype=torch.long, device=device)
        delta[:, 0] = start.unsqueeze(0) + emissions[:, 0]

        for t in range(1, Lmax):
            prev = delta[:, t - 1]  # (B, K)
            best_val = prev[:, 0:1] + trans[0].unsqueeze(0)  # (B, K), i=0 baseline
            best_i = torch.zeros(B, K, dtype=torch.long, device=device)
            for i in range(1, K):
                val = prev[:, i:i + 1] + trans[i].unsqueeze(0)  # (B, K)
                greater = val > best_val
                best_val = torch.where(greater, val, best_val)
                best_i = torch.where(greater, torch.full_like(best_i, i), best_i)
            delta[:, t] = best_val + emissions[:, t]
            back[:, t] = best_i

        clamped_last = (lengths - 1).clamp(min=0)  # (B,)
        last_delta = delta.gather(1, clamped_last.view(B, 1, 1).expand(B, 1, K)).squeeze(1)  # (B, K)
        best_val = last_delta[:, 0] + end[0]
        best_j = torch.zeros(B, dtype=torch.long, device=device)
        for j in range(1, K):
            val = last_delta[:, j] + end[j]
            greater = val > best_val
            best_val = torch.where(greater, val, best_val)
            best_j = torch.where(greater, torch.full_like(best_j, j), best_j)

        paths = torch.zeros(B, Lmax, dtype=torch.long, device=device)
        paths.scatter_(1, clamped_last.view(B, 1), best_j.view(B, 1))
        cur = best_j
        for t in range(Lmax - 1, 0, -1):
            back_t = back[:, t]  # (B, K)
            prev_label = back_t.gather(1, cur.view(B, 1)).squeeze(1)
            active = t <= clamped_last
            cur = torch.where(active, prev_label, cur)
            paths[:, t - 1] = torch.where(active, cur, paths[:, t - 1])

        out = []
        lengths_list = lengths.tolist()
        paths_list = paths.tolist()
        for b in range(B):
            Lb = lengths_list[b]
            out.append(paths_list[b][:Lb] if Lb > 0 else [])
        return out


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
