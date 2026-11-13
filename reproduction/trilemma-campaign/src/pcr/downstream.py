"""Downstream byte-property probes for P3 (descriptive, not gated).

Multinomial logistic probes (softmax regression: Linear(d_model, 2), Adam
lr 1e-2, 200 full-batch epochs, deterministic init from a probe-keyed
generator) from FINAL-LAYER hidden states (input to the head, post-ln_f) of
A2C2 final checkpoints, predicting per-position byte properties of the input
token at that position:

  is_uppercase_letter, is_vowel, is_digit, is_whitespace

Hidden states are computed on dedicated streams (labels "downstream-train" /
"downstream-eval", seed per run), so probe data never coincides with the
training/eval/probe streams of the study runs. Measurement only; nothing here
trains a backbone.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import BatchSampler
from .utils import derive_seed

TRAIN_LABEL = "downstream-train"
EVAL_LABEL = "downstream-eval"
PROBE_EPOCHS = 200
PROBE_LR = 1e-2


def is_uppercase_letter(b: int) -> bool:
    return 65 <= b <= 90


def is_vowel(b: int) -> bool:
    return b in b"aeiouAEIOU"


def is_digit(b: int) -> bool:
    return 48 <= b <= 57


def is_whitespace(b: int) -> bool:
    return b in b" \t\n\r"


PROPERTIES = {
    "is_uppercase_letter": is_uppercase_letter,
    "is_vowel": is_vowel,
    "is_digit": is_digit,
    "is_whitespace": is_whitespace,
}


def make_downstream_batches(region: torch.Tensor, batch_size: int, seq_len: int,
                            seed: int, label: str, n_batches: int):
    """Dedicated downstream stream (domain-separated label, seed per run)."""
    return BatchSampler(region, batch_size, seq_len, seed, label=label).batches(n_batches)


@torch.no_grad()
def hidden_states(model: nn.Module, batches: list, device: torch.device
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """(tokens (N,) int64, hidden (N, d_model) float32) — the head's input."""
    sink: dict = {}
    handle = model.head.register_forward_hook(lambda _m, inputs, _o: sink.update(
        {"hidden": inputs[0].detach()}))
    was_training = model.training
    model.eval()
    tokens, hidden = [], []
    try:
        for x, _y in batches:
            model(x.to(device))
            hidden.append(sink["hidden"].to("cpu", torch.float32))
            tokens.append(x.detach().to("cpu"))
    finally:
        handle.remove()
        if was_training:
            model.train()
    flat_tokens = torch.cat(tokens).reshape(-1)
    flat_hidden = torch.cat([h.reshape(-1, h.shape[-1]) for h in hidden])
    return flat_tokens, flat_hidden


def property_labels(tokens: torch.Tensor, property_name: str) -> torch.Tensor:
    pred = PROPERTIES[property_name]
    return torch.tensor([pred(int(b)) for b in tokens], dtype=torch.long)


def train_property_probe(hidden: torch.Tensor, labels: torch.Tensor,
                         seed: int, epochs: int = PROBE_EPOCHS, lr: float = PROBE_LR,
                         device: torch.device | None = None) -> nn.Linear:
    """Deterministic softmax regression: init and optimizer keyed by `seed`."""
    device = device or torch.device("cpu")
    gen = torch.Generator().manual_seed(derive_seed("downstream-probe", seed))
    probe = nn.Linear(hidden.shape[1], 2).to(device)
    with torch.no_grad():
        nn.init.normal_(probe.weight, 0.0, 0.02, generator=gen)
        nn.init.zeros_(probe.bias)
    x = hidden.to(device)
    y = labels.to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    for _ in range(epochs):
        loss = F.cross_entropy(probe(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return probe


@torch.no_grad()
def probe_accuracy(probe: nn.Linear, hidden: torch.Tensor, labels: torch.Tensor,
                   device: torch.device | None = None) -> float:
    device = device or torch.device("cpu")
    probe.eval()
    preds = probe(hidden.to(device)).argmax(dim=-1).cpu()
    return float((preds == labels).double().mean())
