"""Privacy instrument: token-inversion probe with matched null and dose curve.

Attack family: per-position token inversion. A probe MLP (width -> 512 -> 256,
ReLU, 256-way softmax) predicts the input byte at each sequence position from
the quantized latent vector at that position.

  * Training data: (latent, token) pairs captured by a calibration-cohort run
    (CalibrationCapture in train.py) on seeds disjoint from the pilot.
  * Matched null: the same probe trained on latents whose token labels were
    shuffled (alignment broken); its accuracy estimates the chance baseline.
  * Metric: per-position top-1 accuracy; advantage = accuracy - null accuracy.
  * Positive-control dose curve: the same stored raw latents re-quantized at
    q_bits in {4, 8, 16, 32 (=no quant)}. The scorer is qualified only if
    advantage is monotone non-decreasing in dose and exceeds the predeclared
    detection floor (0.05) at 32 bits.

No estimator output here is a mutual-information bound or a privacy
certificate; it is one attack family's measured advantage over a matched null.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .split import AffineQuantizer
from .utils import derive_seed

DEFAULT_DOSES = (4, 8, 16, 32)
DETECTION_FLOOR = 0.05
NULL_TOLERANCE = 0.02
MONOTONE_SLACK = 0.01


class InversionProbe(nn.Module):
    def __init__(self, width: int, hidden1: int = 512, hidden2: int = 256, n_classes: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def flatten_events(events: list[dict], seq_len: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(latents (N,W) fp32, tokens (N,) int64, positions (N,) int64)."""
    latents, tokens, positions = [], [], []
    pos = np.arange(seq_len, dtype=np.int64)
    for event in events:
        lat = event["latents"].numpy()
        tok = event["tokens"].numpy().astype(np.int64)
        bsz, seq, width = lat.shape
        latents.append(lat.reshape(bsz * seq, width))
        tokens.append(tok.reshape(bsz * seq))
        positions.append(np.tile(pos, bsz))
    return np.concatenate(latents), np.concatenate(tokens), np.concatenate(positions)


def quantize_events_at_dose(events: list[dict], q_bits: int) -> list[dict]:
    """Re-quantize stored raw latents per capture event (per-tensor affine),
    matching the boundary defense's per-step per-tensor policy."""
    quantizer = AffineQuantizer(q_bits)
    out = []
    for event in events:
        lat = event["latents"]
        out.append(
            {
                "step": event["step"],
                "latents": quantizer.quantize_dequantize(lat),
                "tokens": event["tokens"],
            }
        )
    return out


def _train_test_split(n: int, seed: int, frac: float = 0.8):
    """One split per (seed, n), shared across doses so dose comparisons are paired."""
    gen = np.random.default_rng(derive_seed(seed, "probe-split", n))
    perm = gen.permutation(n)
    cut = int(n * frac)
    return perm[:cut], perm[cut:]


def train_probe(
    latents: np.ndarray,
    labels: np.ndarray,
    width: int,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: torch.device,
) -> InversionProbe:
    torch_gen = torch.Generator().manual_seed(derive_seed(seed, "probe-init", latents.shape[0]))
    probe = InversionProbe(width).to(device)
    with torch.no_grad():
        for mod in probe.modules():
            if isinstance(mod, nn.Linear):
                nn.init.normal_(mod.weight, 0.0, 0.02, generator=torch_gen)
                nn.init.zeros_(mod.bias)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    x_all = torch.from_numpy(latents).to(device)
    y_all = torch.from_numpy(labels).to(device)
    n = len(labels)
    for _ in range(epochs):
        order = torch.randperm(n, generator=torch_gen)
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size].to(device)
            loss = F.cross_entropy(probe(x_all[idx]), y_all[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return probe


def per_position_accuracy(
    probe: InversionProbe,
    latents: np.ndarray,
    labels: np.ndarray,
    positions: np.ndarray,
    seq_len: int,
    device: torch.device,
) -> np.ndarray:
    probe.eval()
    with torch.no_grad():
        preds = probe(torch.from_numpy(latents).to(device)).argmax(dim=-1).cpu().numpy()
    correct = preds == labels
    acc = np.full(seq_len, np.nan)
    for pos in range(seq_len):
        mask = positions == pos
        if mask.any():
            acc[pos] = correct[mask].mean()
    return acc


def evaluate_advantage(
    events: list[dict],
    seq_len: int,
    seed: int,
    dose: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> dict:
    """Train the probe and its matched shuffled-label null; return accuracies.

    The null absorbs class-prior exploitation: on a non-uniform byte corpus a
    label-free probe still scores the majority-class rate, so chance is NOT
    1/256. advantage = accuracy - null_accuracy is the honest measure.
    """
    latents, tokens, positions = flatten_events(events, seq_len)
    width = latents.shape[1]
    tr, te = _train_test_split(len(tokens), seed)

    probe = train_probe(latents[tr], tokens[tr], width, epochs, batch_size, lr, seed, device)
    acc_pos = per_position_accuracy(probe, latents[te], tokens[te], positions[te], seq_len, device)

    shuffle_gen = np.random.default_rng(derive_seed(seed, "null-shuffle", dose))
    null_probe = train_probe(
        latents[tr], shuffle_gen.permutation(tokens[tr]), width, epochs, batch_size, lr, seed, device
    )
    null_pos = per_position_accuracy(
        null_probe, latents[te], tokens[te], positions[te], seq_len, device
    )

    accuracy = float(np.nanmean(acc_pos))
    null_accuracy = float(np.nanmean(null_pos))
    return {
        "q_bits": dose,
        "accuracy": accuracy,
        "null_accuracy": null_accuracy,
        "advantage": accuracy - null_accuracy,
        "per_position_accuracy": acc_pos.tolist(),
        "per_position_null_accuracy": null_pos.tolist(),
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
    }


def break_alignment(events: list[dict], seed: int) -> list[dict]:
    """Return events with token labels globally shuffled w.r.t. latents.

    The advantage estimator run on broken-alignment events must report
    advantage ~= 0; this is the matched-null qualification of the scorer.
    """
    flat_tokens = np.concatenate([e["tokens"].numpy().reshape(-1) for e in events])
    gen = np.random.default_rng(derive_seed(seed, "alignment-break"))
    shuffled = gen.permutation(flat_tokens)
    out, offset = [], 0
    for event in events:
        n = event["tokens"].numel()
        out.append(
            {
                "step": event["step"],
                "latents": event["latents"],
                "tokens": torch.from_numpy(shuffled[offset : offset + n].copy()).reshape(
                    event["tokens"].shape
                ),
            }
        )
        offset += n
    return out


def dose_curve(
    raw_events: list[dict],
    seq_len: int,
    seed: int,
    doses=DEFAULT_DOSES,
    epochs: int = 20,
    batch_size: int = 4096,
    lr: float = 1e-3,
    device: torch.device | None = None,
) -> dict:
    device = device or torch.device("cpu")
    results = [
        evaluate_advantage(
            quantize_events_at_dose(raw_events, dose), seq_len, seed, dose, epochs, batch_size, lr, device
        )
        for dose in doses
    ]
    advantages = [r["advantage"] for r in results]
    null_check = evaluate_advantage(
        quantize_events_at_dose(break_alignment(raw_events, seed), max(doses)),
        seq_len, seed, max(doses), epochs, batch_size, lr, device,
    )
    return {
        "attack_family": "token_inversion_probe",
        "doses": list(doses),
        "results": results,
        "advantages": advantages,
        "monotone_non_decreasing": all(
            b >= a - MONOTONE_SLACK for a, b in zip(advantages, advantages[1:])
        ),
        "advantage_at_max_dose": advantages[-1],
        "detection_floor": DETECTION_FLOOR,
        "passes_floor": advantages[-1] > DETECTION_FLOOR,
        "null_pipeline_advantage": null_check["advantage"],
        "null_within_tolerance": abs(null_check["advantage"]) < NULL_TOLERANCE,
    }


def bits_per_token(total_bytes: int, total_tokens: int) -> float:
    """Measured operational rate from transcript counters."""
    return total_bytes * 8 / total_tokens
