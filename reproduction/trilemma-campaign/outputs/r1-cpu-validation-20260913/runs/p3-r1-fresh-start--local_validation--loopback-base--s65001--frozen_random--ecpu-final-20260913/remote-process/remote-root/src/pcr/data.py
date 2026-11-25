"""Byte corpus loading and deterministic batch sampling.

Split rule: first 90% of bytes is the train region, last 10% is held out.
Every sampler is a pure function of (seed, label, draw index): batch k of a run
with a given seed is identical in every run with that seed, regardless of model
architecture or mode. This is what makes trained/frozen/removed/all-local runs
of one seed exactly paired on the data stream.

The probe batch is drawn from a dedicated generator keyed by probe seed 0 and
is identical across ALL runs and seeds.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .utils import derive_seed

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = REPO_ROOT / "data" / "raw" / "tinyshakespeare.txt"


def load_corpus(path: str | Path = DEFAULT_CORPUS) -> torch.Tensor:
    data = Path(path).read_bytes()
    return torch.tensor(list(data), dtype=torch.uint8)


def split_corpus(data: torch.Tensor, train_fraction: float = 0.9) -> tuple[torch.Tensor, torch.Tensor]:
    cut = int(len(data) * train_fraction)
    return data[:cut], data[cut:]


class BatchSampler:
    """Deterministic batcher: draw `step` depends only on (seed, label, step)."""

    def __init__(self, region: torch.Tensor, batch_size: int, seq_len: int, seed: int, label: str):
        self.region = region
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.seed = seed
        self.label = label
        self.max_start = len(region) - seq_len - 1
        assert self.max_start >= 0, "region too small for seq_len"

    def batch(self, step: int) -> tuple[torch.Tensor, torch.Tensor]:
        gen = torch.Generator().manual_seed(derive_seed(self.seed, self.label, step))
        starts = torch.randint(0, self.max_start + 1, (self.batch_size,), generator=gen)
        offsets = torch.arange(self.seq_len + 1)
        windows = self.region[starts[:, None] + offsets[None, :]].to(torch.long)
        return windows[:, :-1], windows[:, 1:]

    def batches(self, count: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return [self.batch(k) for k in range(count)]


def make_train_sampler(
    train_region: torch.Tensor, batch_size: int, seq_len: int, seed: int
) -> BatchSampler:
    return BatchSampler(train_region, batch_size, seq_len, seed, label="train")


def make_eval_batches(
    heldout_region: torch.Tensor, n_batches: int, batch_size: int, seq_len: int, seed: int
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Fixed held-out eval set: identical across all runs sharing `seed`."""
    sampler = BatchSampler(heldout_region, batch_size, seq_len, seed, label="eval")
    return sampler.batches(n_batches)


def make_probe_batch(
    heldout_region: torch.Tensor, batch_size: int, seq_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """ONE fixed probe batch, identical across every run (probe generator seed 0)."""
    sampler = BatchSampler(heldout_region, batch_size, seq_len, seed=0, label="probe")
    return sampler.batch(0)


def make_probe_batches(
    heldout_region: torch.Tensor, batch_size: int, seq_len: int, n_batches: int
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """The first n_batches of the fixed probe stream (batch 0 == make_probe_batch)."""
    sampler = BatchSampler(heldout_region, batch_size, seq_len, seed=0, label="probe")
    return sampler.batches(n_batches)


# --------------------------------------------------------------------------- #
# canary injection (P4a; additive — runs without a `canary` block are unaffected)
# --------------------------------------------------------------------------- #
def draw_canary_strings(count: int, length: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(members, non_members): count random `length`-byte strings each, drawn
    once from the dedicated stream (generator derive_seed("canary", seed);
    members first, then the held-out non-member controls)."""
    gen = torch.Generator().manual_seed(derive_seed("canary", seed))
    members = torch.randint(0, 256, (count, length), generator=gen, dtype=torch.uint8)
    non_members = torch.randint(0, 256, (count, length), generator=gen, dtype=torch.uint8)
    return members, non_members


class CanaryBatchSampler(BatchSampler):
    """BatchSampler with deterministic canary injection (P4a).

    Schedule (documented, frozen by implementation): the injection grid is
    steps x batch rows; `count * repeats` distinct flat slot indices are drawn
    without replacement from a generator keyed by derive_seed("canary-schedule",
    seed); in ascending slot order, slot j injects canary (j % count), so each
    canary appears exactly `repeats` times. Injection overwrites the first
    `length` tokens of the slot's window with the canary string (x and y kept
    consistent: y gets the shifted canary and the boundary token is preserved).
    """

    def __init__(self, region: torch.Tensor, batch_size: int, seq_len: int, seed: int,
                 label: str, canary: dict):
        super().__init__(region, batch_size, seq_len, seed, label)
        members, _ = draw_canary_strings(canary["count"], canary["length"], canary["seed"])
        self.members = members.long()
        self.length = int(canary["length"])
        self.slots = self._build_slots(canary, seed)

    def _build_slots(self, canary: dict, seed: int) -> dict[int, dict[int, int]]:
        n_injections = canary["count"] * canary["repeats"]
        grid = canary["steps"] * self.batch_size
        gen = torch.Generator().manual_seed(derive_seed("canary-schedule", seed))
        flat = torch.randperm(grid, generator=gen)[:n_injections].sort().values.tolist()
        slots: dict[int, dict[int, int]] = {}
        for j, index in enumerate(flat):
            step, row = divmod(index, self.batch_size)
            slots.setdefault(step, {})[row] = j % canary["count"]
        return slots

    def batch(self, step: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = super().batch(step)
        rows = self.slots.get(step)
        if not rows:
            return x, y
        x, y = x.clone(), y.clone()
        for row, canary_idx in rows.items():
            canary = self.members[canary_idx]
            x[row, : self.length] = canary
            y[row, : self.length - 1] = canary[1:]
            y[row, self.length - 1] = x[row, self.length]
        return x, y
