"""Remote-update modes.

  all_local      — plain backbone, no split (constructed without a RemotePath).
  trained        — remote surrogate params require grad and enter the optimizer.
  frozen_random  — remote surrogate params frozen at paired init (verified).
  removed_remote — remote is a fixed identity; zero remote trainable params.
  oracle         — split topology, width == d_model, no quantization.

Only the surrogate's parameter updating differs between trained and frozen;
encoder/decoder and backbone always train in split modes.
"""

from __future__ import annotations

import torch.nn as nn

from .split import SplitBackbone

MODES = ("all_local", "trained", "frozen_random", "removed_remote", "oracle")

# Modes that have a remote path crossing the boundary (and thus a transcript).
SPLIT_MODES = ("trained", "frozen_random", "removed_remote", "oracle")


def has_split(mode: str) -> bool:
    return mode in SPLIT_MODES


def apply_mode(model: nn.Module, mode: str) -> None:
    """Set requires_grad on the remote module per mode; verify frozen/removed."""
    if mode == "all_local":
        return
    assert isinstance(model, SplitBackbone)
    remote_params = model.remote_parameters()
    if mode == "trained" or mode == "oracle":
        for p in remote_params:
            p.requires_grad_(True)
        assert remote_params, f"mode {mode} must have remote trainable params"
    elif mode == "frozen_random":
        assert remote_params, "frozen_random must have remote params to freeze"
        for p in remote_params:
            p.requires_grad_(False)
        assert all(not p.requires_grad for p in remote_params)
    elif mode == "removed_remote":
        trainable = [p for p in remote_params if p.requires_grad]
        assert not trainable, "removed_remote must have zero remote trainable params"
    else:
        raise ValueError(f"unknown mode {mode}")


def remote_parameters(model: nn.Module) -> list[nn.Parameter]:
    if isinstance(model, SplitBackbone):
        return model.remote_parameters()
    return []


def trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]
