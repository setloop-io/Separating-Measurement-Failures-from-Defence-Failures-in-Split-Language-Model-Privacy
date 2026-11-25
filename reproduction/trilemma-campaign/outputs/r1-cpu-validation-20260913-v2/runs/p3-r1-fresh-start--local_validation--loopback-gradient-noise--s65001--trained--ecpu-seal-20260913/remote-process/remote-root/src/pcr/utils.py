"""Deterministic seeding and hashing helpers.

All randomness in a run is keyed by explicit identifiers via `derive_seed`, so
paired runs (trained vs frozen-random) reproduce bit-identical streams when the
identifiers match. SHA-256 based derivation avoids Python's salted hash().
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

import torch


def derive_seed(*parts: Any) -> int:
    """Map arbitrary key parts to a deterministic 63-bit seed."""
    key = "::".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % (2**63)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def sha256_json(obj: Any) -> str:
    return sha256_bytes(canonical_json(obj).encode("utf-8"))


def parameters_checksum(params: Iterable[torch.nn.Parameter]) -> str:
    """Order-stable SHA-256 over parameter values (detached, CPU, fp32 bytes)."""
    h = hashlib.sha256()
    for p in params:
        arr = p.detach().to("cpu", torch.float32).contiguous()
        h.update(str(tuple(arr.shape)).encode("utf-8"))
        h.update(arr.numpy().tobytes())
    return h.hexdigest()


def state_dict_checksum(state: dict) -> str:
    h = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key]
        arr = tensor.detach().to("cpu", torch.float32).contiguous()
        h.update(key.encode("utf-8"))
        h.update(str(tuple(arr.shape)).encode("utf-8"))
        h.update(arr.numpy().tobytes())
    return h.hexdigest()


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    return torch.device(requested)
