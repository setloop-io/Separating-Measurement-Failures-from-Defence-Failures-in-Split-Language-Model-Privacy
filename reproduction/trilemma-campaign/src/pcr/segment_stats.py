"""Segment statistics for the P2c selection stage (docs/PREREGISTRATION_P2C.md).

The five candidate statistics, per contiguous segment [start, start+span) of the
8-block backbone, EXACTLY as preregistered:

  1. residual_update_magnitude: mean over blocks in the segment of
     ||h_out - h_in||_2 / ||h_in||_2 (per token, averaged over tokens/batches);
  2. block_influence: mean over blocks of 1 - cos(h_in, h_out) (per token,
     averaged);
  3. representation_similarity: linear CKA between segment input (input of the
     first block) and segment output (output of the last block), column-centered;
  4. layer_drop_sensitivity: increase in probe-batch loss when all blocks in
     the segment are bypassed by identity (single forward pass, no retraining);
  5. intrinsic_dimension: participation ratio (sum lambda)^2 / sum lambda^2 of
     the per-token covariance eigenvalues of segment-output activations.

Measurement stream: 64 fixed batches from the held-out region drawn by
BatchSampler with label "measure" and seed 929 (domain-separated, following
src/pcr/data.py); all tokens of those batches enter the statistics. Gram
matrices for CKA / participation ratio are accumulated online (per-batch
scatter Gram of the n_layers+1 boundary activations), so no full activation
tensor is retained.

This module is measurement-only and trains nothing; the backbone is trained
upstream by scripts/measure_segment_stats.py via the standard train_model.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn.functional as F

from .data import BatchSampler
from .model import Backbone

MEASURE_SEED = 929
MEASURE_LABEL = "measure"
MEASURE_BATCHES = 64

STATISTIC_NAMES = (
    "residual_update_magnitude",
    "block_influence",
    "representation_similarity",
    "layer_drop_sensitivity",
    "intrinsic_dimension",
)


def make_measure_batches(heldout_region: torch.Tensor, batch_size: int, seq_len: int,
                         n_batches: int = MEASURE_BATCHES,
                         seed: int = MEASURE_SEED) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """The fixed measurement batches (dedicated stream: label "measure", seed 929)."""
    sampler = BatchSampler(heldout_region, batch_size, seq_len, seed, label=MEASURE_LABEL)
    return sampler.batches(n_batches)


def all_segments(n_layers: int) -> list[tuple[int, int]]:
    """Every contiguous segment as (start, span), span 1..n_layers, all starts."""
    return [(start, span)
            for span in range(1, n_layers + 1)
            for start in range(0, n_layers - span + 1)]


class BoundaryAccumulator:
    """Online statistics over the n_layers+1 boundary activations a_0..a_n.

    a_0 = block 0 input (embedding output); a_i = block i-1 output = block i
    input; a_n = final block output. Per batch accumulates the scatter Gram
    G = A^T A over the stacked boundary activations A (T tokens, (n+1)*d),
    boundary column sums, token count, and the per-block residual-update
    ratio / cosine sums. Memory is O((n+1)^2 d^2), independent of token count.

    MPS does not support float64, so accumulation runs in float64 on CPU and
    float32 on MPS; all downstream reductions (centering, CKA, eigenvalues)
    are finalized in float64 on CPU.
    """

    def __init__(self, n_layers: int, d_model: int, device: torch.device):
        self.n_layers = n_layers
        self.d_model = d_model
        self.device = device
        self.dtype = torch.float64 if device.type == "cpu" else torch.float32
        n_bound = n_layers + 1
        self.gram = torch.zeros(n_bound * d_model, n_bound * d_model,
                                dtype=self.dtype, device=device)
        self.sums = torch.zeros(n_bound, d_model, dtype=self.dtype, device=device)
        self.count = 0
        self.ratio_sum = torch.zeros(n_layers, dtype=self.dtype, device=device)
        self.cos_sum = torch.zeros(n_layers, dtype=self.dtype, device=device)
        self.eps = 1e-12

    @torch.no_grad()
    def update(self, boundaries: list[torch.Tensor]) -> None:
        """boundaries: [a_0 .. a_n] each (batch, seq, d) on any device."""
        stacked = torch.stack(boundaries).to(self.device, self.dtype)  # (n+1, B, T, d)
        n_bound, bsz, seq, dim = stacked.shape
        tokens = stacked.reshape(n_bound, bsz * seq, dim)
        flat = tokens.permute(1, 0, 2).reshape(bsz * seq, n_bound * dim)
        self.gram += flat.T @ flat
        self.sums += tokens.sum(dim=1)
        self.count += bsz * seq
        for i in range(self.n_layers):
            h_in = tokens[i]
            h_out = tokens[i + 1]
            diff = h_out - h_in
            self.ratio_sum[i] += (diff.norm(dim=-1) / (h_in.norm(dim=-1) + self.eps)).sum()
            cos = F.cosine_similarity(h_in, h_out, dim=-1, eps=1e-8)
            self.cos_sum[i] += (1.0 - cos).sum()

    def _gram_block(self, p: int, q: int) -> torch.Tensor:
        d = self.d_model
        return self.gram[p * d:(p + 1) * d, q * d:(q + 1) * d]

    def centered_grams(self, p: int, q: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Centered G_pp, G_qq, G_pq over all accumulated tokens (float64, CPU).

        Note the order: CPU first, then float64 — MPS cannot host float64
        tensors, so `.to(torch.float64)` while still on device would fail.
        """
        n = self.count
        sp = self.sums[p].cpu().to(torch.float64)
        sq = self.sums[q].cpu().to(torch.float64)
        gpp = self._gram_block(p, p).cpu().to(torch.float64) - torch.outer(sp, sp) / n
        gqq = self._gram_block(q, q).cpu().to(torch.float64) - torch.outer(sq, sq) / n
        gpq = self._gram_block(p, q).cpu().to(torch.float64) - torch.outer(sp, sq) / n
        return gpp, gqq, gpq

    def block_means(self) -> tuple[list[float], list[float]]:
        """Per-block mean residual-update ratio and mean (1 - cos)."""
        ratio = (self.ratio_sum / self.count).cpu().tolist()
        infl = (self.cos_sum / self.count).cpu().tolist()
        return ratio, infl


def linear_cka(gpp: torch.Tensor, gqq: torch.Tensor, gpq: torch.Tensor) -> float:
    """Linear CKA = ||G_pq||_F^2 / (||G_pp||_F * ||G_qq||_F) on centered Grams."""
    num = float((gpq * gpq).sum())
    den = float(torch.sqrt((gpp * gpp).sum() * (gqq * gqq).sum()))
    return num / den if den > 0 else float("nan")


def participation_ratio(gqq: torch.Tensor) -> float:
    """(sum lambda)^2 / sum lambda^2 of the per-token output covariance.

    Computed from the centered scatter Gram; the 1/(N-1) covariance
    normalization cancels in the ratio, so the Gram eigenvalues suffice.
    """
    eig = torch.linalg.eigvalsh(gqq.cpu()).clamp_min(0.0)
    s1 = float(eig.sum())
    s2 = float((eig * eig).sum())
    return (s1 * s1) / s2 if s2 > 0 else float("nan")


@torch.no_grad()
def capture_boundaries(model: Backbone, batches: list, device: torch.device) -> BoundaryAccumulator:
    """One no-grad forward per batch, hooking every block's input/output."""
    n_layers = model.cfg.n_layers
    acc = BoundaryAccumulator(n_layers, model.cfg.d_model, device)
    captured: list[torch.Tensor] = []

    handles = []

    def make_hook(idx: int):
        def hook(_module, inputs, output):
            captured[idx] = inputs[0].detach()
            captured[idx + 1] = output.detach()
        return hook

    for i, block in enumerate(model.blocks):
        handles.append(block.register_forward_hook(make_hook(i)))

    was_training = model.training
    model.eval()
    try:
        for x, _y in batches:
            captured = [None] * (n_layers + 1)  # type: ignore[list-item]
            model(x.to(device))
            acc.update(captured)
    finally:
        for handle in handles:
            handle.remove()
        if was_training:
            model.train()
    return acc


@contextmanager
def identity_bypass(model: Backbone, start: int, span: int):
    """Temporarily replace blocks[start:start+span] forward by identity."""
    originals = []
    for i in range(start, start + span):
        block = model.blocks[i]
        originals.append((block, block.forward))
        block.forward = lambda x: x  # type: ignore[method-assign]
    try:
        yield
    finally:
        for block, forward in originals:
            block.forward = forward  # type: ignore[method-assign]


@torch.no_grad()
def probe_loss(model: Backbone, batches: list, device: torch.device) -> float:
    """Mean cross-entropy over the measurement batches (no grad)."""
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    try:
        for x, y in batches:
            logits = model(x.to(device))
            total += F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), y.reshape(-1).to(device)
            ).item()
            count += 1
    finally:
        if was_training:
            model.train()
    return total / max(1, count)


@torch.no_grad()
def measure_all_segments(model: Backbone, batches: list, device: torch.device) -> list[dict]:
    """All five statistics for every contiguous segment of the backbone."""
    n_layers = model.cfg.n_layers
    acc = capture_boundaries(model, batches, device)
    ratio_by_block, influence_by_block = acc.block_means()
    intact = probe_loss(model, batches, device)

    pr_by_output: dict[int, float] = {}
    for end in range(1, n_layers + 1):
        _g, gee, _x = acc.centered_grams(end, end)
        pr_by_output[end] = participation_ratio(gee)

    segments = []
    for start, span in all_segments(n_layers):
        end = start + span
        blocks = range(start, end)
        g_ss, g_ee, g_se = acc.centered_grams(start, end)
        with identity_bypass(model, start, span):
            bypassed = probe_loss(model, batches, device)
        segments.append({
            "start": start,
            "span": span,
            "segment": f"{span}@{start}",
            "residual_update_magnitude": sum(ratio_by_block[b] for b in blocks) / span,
            "block_influence": sum(influence_by_block[b] for b in blocks) / span,
            "representation_similarity": linear_cka(g_ss, g_ee, g_se),
            "layer_drop_sensitivity": bypassed - intact,
            "intrinsic_dimension": pr_by_output[end],
        })
    return segments
