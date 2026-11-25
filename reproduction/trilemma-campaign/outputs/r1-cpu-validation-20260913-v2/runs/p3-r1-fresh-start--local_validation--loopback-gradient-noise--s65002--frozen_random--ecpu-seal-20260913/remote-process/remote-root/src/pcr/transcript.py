"""Declared remote-visible transcript capture.

Channels (all remote-visible; gradient names follow the remote's perspective,
matching the protocol's "grad of loss wrt sent latents, as seen by remote"):

  forward_values      z  = quantized latent, trusted -> remote
  returned_values     y  = remote module output, remote -> trusted
  outbound_gradients  dL/dz, remote -> trusted
  inbound_gradients   dL/dy, trusted -> remote
  remote_state        SHA-256 of remote parameters at each eval point

Byte counters accumulate the measured serialized size of every boundary
crossing during training steps (quantized packed size + 8-byte affine metadata
per tensor). Probe tensor sinks (which re-run forward/backward purely for
measurement) are excluded from the counters. Timing is logged per step for
systems accounting only; it is not part of any privacy claim.

Capture is active only for split modes; all_local runs have no transcript.
"""

from __future__ import annotations

import json
from pathlib import Path

from jsonschema import validate

from .split import AffineQuantizer

REPO_ROOT = Path(__file__).resolve().parents[2]
TRANSCRIPT_SCHEMA_PATH = REPO_ROOT / "experiments" / "schemas" / "transcript.schema.json"
TRANSCRIPT_SCHEMA_VERSION = "1.0.0"

VALUE_CHANNELS = ("forward_values", "returned_values")
GRAD_CHANNELS = ("outbound_gradients", "inbound_gradients")
COUNTED_CHANNELS = VALUE_CHANNELS + GRAD_CHANNELS


class TranscriptCapture:
    """Hooks on the boundary tensors; counts measured serialized bytes."""

    def __init__(self, quantizer: AffineQuantizer):
        self.quantizer = quantizer
        self.total_bytes = {ch: 0 for ch in COUNTED_CHANNELS}
        self.step_bytes: list[dict] = []  # per-step retained history
        self._current_step = {ch: 0 for ch in COUNTED_CHANNELS}
        self._counting = True
        self._sink: dict | None = None

    # ---- training-step accounting --------------------------------------
    def begin_step(self) -> None:
        self._current_step = {ch: 0 for ch in COUNTED_CHANNELS}

    def end_step(self) -> None:
        self.step_bytes.append(dict(self._current_step))
        for ch, n in self._current_step.items():
            self.total_bytes[ch] += n

    # ---- probe tensor sink (measurement only, not counted) --------------
    def begin_sink(self, sink: dict) -> None:
        self._sink = sink
        self._counting = False

    def end_sink(self) -> None:
        self._sink = None
        self._counting = True

    # ---- boundary hooks --------------------------------------------------
    def on_forward_values(self, z) -> None:
        if self._sink is not None:
            self._sink["forward_values"] = z.detach().to("cpu")
        if self._counting and torch_grad_enabled():
            self._current_step["forward_values"] += self.quantizer.serialized_nbytes(z)
        if z.requires_grad:
            z.register_hook(self._on_outbound_gradients)

    def on_returned_values(self, y) -> None:
        if self._sink is not None:
            self._sink["returned_values"] = y.detach().to("cpu")
        if self._counting and torch_grad_enabled():
            self._current_step["returned_values"] += self.quantizer.serialized_nbytes(y)
        if y.requires_grad:
            y.register_hook(self._on_inbound_gradients)

    def _on_outbound_gradients(self, grad) -> None:
        if self._sink is not None:
            self._sink["outbound_gradients"] = grad.detach().to("cpu")
        if self._counting:
            self._current_step["outbound_gradients"] += self.quantizer.serialized_nbytes(grad)

    def _on_inbound_gradients(self, grad) -> None:
        if self._sink is not None:
            self._sink["inbound_gradients"] = grad.detach().to("cpu")
        if self._counting:
            self._current_step["inbound_gradients"] += self.quantizer.serialized_nbytes(grad)


def torch_grad_enabled() -> bool:
    import torch

    return torch.is_grad_enabled()


def build_inventory(
    counters: dict,
    remote_state_checksums: list,
    has_remote_state: bool,
    notes: str = "",
) -> dict:
    """Assemble the declared-transcript inventory for a split-mode run."""
    return {
        "schema_version": TRANSCRIPT_SCHEMA_VERSION,
        "system_boundary": (
            "trusted encoder output <-> remote surrogate; uniform per-tensor affine "
            "quantized latent boundary; trusted decoder return path"
        ),
        "channels": {
            "forward_values": True,
            "returned_values": True,
            "outbound_gradients": True,
            "inbound_gradients": True,
            "remote_state": has_remote_state,
            "timing": True,
        },
        "history": {"per_step": True, "accumulated": True},
        "omissions": [],
        "accumulated_bytes": {ch: int(counters.get(ch, 0)) for ch in COUNTED_CHANNELS},
        "remote_state_sha256_per_eval": list(remote_state_checksums),
        "notes": notes
        or "Timing logged for systems accounting only; not part of any privacy claim.",
    }


def validate_inventory(inventory: dict, schema_path: Path = TRANSCRIPT_SCHEMA_PATH) -> None:
    schema = json.loads(Path(schema_path).read_text())
    validate(instance=inventory, schema=schema)
