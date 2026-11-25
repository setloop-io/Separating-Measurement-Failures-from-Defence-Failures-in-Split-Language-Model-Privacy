"""Fail-closed CPU-TEE to CUDA-surrogate boundary for CC engineering runs.

The trusted backbone, boundary encoder/decoder, quantizer and defense stay on
CPU.  Only the remote surrogate is placed on CUDA.  The bridge performs four
explicit, synchronous boundary copies per training step:

``forward_values``
    Per-token affine q8 codes plus the float32 minimum/scale metadata move from
    trusted CPU memory to the remote CUDA device.
``returned_values``
    The remote surrogate output moves from CUDA back to trusted CPU memory.
``inbound_gradients``
    ``dL/dy`` moves from the trusted CPU decoder to the remote CUDA surrogate.
``outbound_gradients``
    ``dL/dz`` moves from the remote CUDA surrogate to the trusted CPU encoder.

The custom autograd functions below make those backward transfers explicit;
they are not observation-only copies.  Traces contain timing and byte metadata
only, never tensor values.  A same-device CPU loopback exists solely so the
transport and autograd contract can be unit-tested on machines without CUDA.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from copy import deepcopy
import os
import threading
import time
from typing import Any, Iterable
import weakref

import torch
from torch import nn

from .model import DefendedRemotePath


TRACE_SCHEMA = "p4-cc-boundary-step-trace-0.1"
CONTROL_TRANSFER_SCHEMA = "p4-cc-control-transfer-trace-0.1"
CHANNELS = (
    "forward_values",
    "returned_values",
    "outbound_gradients",
    "inbound_gradients",
)
CONTROL_TRANSFER_PURPOSES = (
    "model-setup",
    "correctness-setup",
    "training-repetition-setup",
    "checkpoint-restore",
    "model-state-snapshot",
    "optimizer-state-snapshot",
    "checkpoint-snapshot",
)
_DIRECTIONS = {
    "forward_values": "trusted-to-remote",
    "returned_values": "remote-to-trusted",
    "outbound_gradients": "remote-to-trusted",
    "inbound_gradients": "trusted-to-remote",
}


def _tensor_nbytes(value: torch.Tensor) -> int:
    return value.numel() * value.element_size()


def _dtype_name(value: torch.Tensor) -> str:
    return str(value.dtype).removeprefix("torch.")


def _module_tensors(module: nn.Module) -> Iterable[tuple[str, torch.Tensor]]:
    yield from module.named_parameters(recurse=True)
    yield from module.named_buffers(recurse=True)


def _tensor_descriptor(name: str, value: torch.Tensor) -> dict[str, Any]:
    return {
        "name": name,
        "shape": list(value.shape),
        "dtype": _dtype_name(value),
        "bytes": _tensor_nbytes(value),
    }


class ControlTransferLedger:
    """Metadata-only ledger for non-step CPU/CUDA state transfers.

    Boundary activations and gradients belong exclusively to :data:`CHANNELS`.
    This ledger accounts for setup and state/checkpoint materialisation without
    creating a fifth boundary channel or persisting tensor values.
    """

    def __init__(self, trusted_device: torch.device, remote_device: torch.device,
                 test_only_cpu_loopback: bool):
        self.trusted_device = trusted_device
        self.remote_device = remote_device
        self.test_only_cpu_loopback = test_only_cpu_loopback
        self._lock = threading.RLock()
        self._events: list[dict[str, Any]] = []

    def record(self, *, purpose: str, direction: str,
               tensors: Iterable[dict[str, Any]], elapsed_ns: int,
               request_id: str | None = None,
               step_id: int | str | None = None) -> dict[str, Any]:
        if purpose not in CONTROL_TRANSFER_PURPOSES:
            raise ValueError(f"unknown control-transfer purpose: {purpose}")
        if direction not in {"trusted-to-remote", "remote-to-trusted"}:
            raise ValueError(f"unknown control-transfer direction: {direction}")
        if request_id is not None and (not isinstance(request_id, str) or not request_id):
            raise ValueError("request_id must be null or a nonempty string")
        if (step_id is not None
                and (isinstance(step_id, bool) or not isinstance(step_id, (int, str)))):
            raise ValueError("step_id must be null, an integer, or a nonempty string")
        if isinstance(step_id, str) and not step_id:
            raise ValueError("step_id must be null, an integer, or a nonempty string")
        rows = [deepcopy(row) for row in tensors]
        logical_bytes = sum(int(row["bytes"]) for row in rows)
        physical_transfer = self.trusted_device != self.remote_device
        with self._lock:
            event = {
                "schema_version": CONTROL_TRANSFER_SCHEMA,
                "sequence": len(self._events),
                "purpose": purpose,
                "direction": direction,
                "source_device": str(
                    self.trusted_device if direction == "trusted-to-remote"
                    else self.remote_device
                ),
                "target_device": str(
                    self.remote_device if direction == "trusted-to-remote"
                    else self.trusted_device
                ),
                "tensor_count": len(rows),
                "logical_bytes": logical_bytes,
                "transferred_bytes": logical_bytes if physical_transfer else 0,
                "elapsed_ns": int(elapsed_ns),
                "synchronized": True,
                "physical_transfer": physical_transfer,
                "test_only_cpu_loopback": self.test_only_cpu_loopback,
                "request_id": request_id,
                "step_id": step_id,
                "tensors": rows,
                "raw_tensors_persisted": False,
            }
            self._events.append(event)
            return deepcopy(event)

    @property
    def events(self) -> tuple[dict[str, Any], ...]:
        with self._lock:
            return tuple(deepcopy(self._events))

    def events_since(self, sequence: int) -> tuple[dict[str, Any], ...]:
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
            raise ValueError("control-transfer sequence must be a nonnegative integer")
        with self._lock:
            if sequence > len(self._events):
                raise ValueError("control-transfer sequence is beyond the ledger")
            return tuple(deepcopy(self._events[sequence:]))

    def summary(self) -> dict[str, Any]:
        with self._lock:
            totals: dict[str, dict[str, int]] = {}
            for event in self._events:
                bucket = totals.setdefault(
                    event["purpose"],
                    {"events": 0, "tensors": 0, "logical_bytes": 0,
                     "transferred_bytes": 0, "elapsed_ns": 0},
                )
                bucket["events"] += 1
                bucket["tensors"] += event["tensor_count"]
                bucket["logical_bytes"] += event["logical_bytes"]
                bucket["transferred_bytes"] += event["transferred_bytes"]
                bucket["elapsed_ns"] += event["elapsed_ns"]
            return {
                "schema_version": CONTROL_TRANSFER_SCHEMA,
                "event_count": len(self._events),
                "totals_by_purpose": deepcopy(totals),
                "raw_tensors_persisted": False,
            }


class BoundaryStepTrace:
    """One-at-a-time step recorder used by :class:`CCTeeGpuRemotePath`.

    Call ``begin_step`` immediately before the model forward, then call
    ``end_step`` only after backward has completed.  The latter fails closed if
    a required channel did not execute.  For inference-only calls, declare
    ``expect_backward=False`` explicitly.
    """

    def __init__(self, trusted_device: torch.device, remote_device: torch.device,
                 test_only_cpu_loopback: bool):
        self.trusted_device = trusted_device
        self.remote_device = remote_device
        self.test_only_cpu_loopback = test_only_cpu_loopback
        self._lock = threading.RLock()
        self._next_token = 0
        self._active: dict[str, Any] | None = None
        self.completed: list[dict[str, Any]] = []

    def begin_step(self, step_id: int | str, request_id: str,
                   *, expect_backward: bool = True) -> int:
        if isinstance(step_id, bool) or not isinstance(step_id, (int, str)):
            raise ValueError("step_id must be an integer or nonempty string")
        if isinstance(step_id, str) and not step_id:
            raise ValueError("step_id must be an integer or nonempty string")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be a nonempty string")
        if type(expect_backward) is not bool:
            raise ValueError("expect_backward must be boolean")
        with self._lock:
            if self._active is not None:
                raise RuntimeError("a boundary trace step is already active")
            token = self._next_token
            self._next_token += 1
            self._active = {
                "token": token,
                "step_id": step_id,
                "request_id": request_id,
                "expect_backward": expect_backward,
                "events": {},
                "started_ns": time.perf_counter_ns(),
            }
            return token

    def active_token(self) -> int:
        with self._lock:
            if self._active is None:
                raise RuntimeError("begin_step must be called before the boundary forward")
            return int(self._active["token"])

    def record(self, token: int, event: dict[str, Any]) -> None:
        channel = event.get("channel")
        if channel not in CHANNELS:
            raise ValueError(f"unknown boundary channel: {channel}")
        if event.get("direction") != _DIRECTIONS[channel]:
            raise ValueError(f"direction mismatch for {channel}")
        with self._lock:
            if self._active is None or self._active["token"] != token:
                raise RuntimeError("boundary event belongs to no active step")
            if channel in self._active["events"]:
                raise RuntimeError(f"duplicate boundary event for {channel}")
            event = dict(event)
            event["sequence"] = len(self._active["events"])
            self._active["events"][channel] = event

    def end_step(self) -> dict[str, Any]:
        with self._lock:
            if self._active is None:
                raise RuntimeError("no boundary trace step is active")
            expected = set(CHANNELS if self._active["expect_backward"] else CHANNELS[:2])
            actual = set(self._active["events"])
            if actual != expected:
                missing = sorted(expected - actual)
                unexpected = sorted(actual - expected)
                raise RuntimeError(
                    f"incomplete boundary trace; missing={missing}, unexpected={unexpected}"
                )
            events = sorted(self._active["events"].values(), key=lambda item: item["sequence"])
            channel_bytes = {channel: self._active["events"][channel]["wire_bytes"]
                             for channel in CHANNELS if channel in self._active["events"]}
            summary = {
                "schema_version": TRACE_SCHEMA,
                "step_id": self._active["step_id"],
                "request_id": self._active["request_id"],
                "trusted_device": str(self.trusted_device),
                "remote_device": str(self.remote_device),
                "transport": ("test-only-cpu-loopback" if self.test_only_cpu_loopback
                              else "synchronous-cuda-copy"),
                "test_only_cpu_loopback": self.test_only_cpu_loopback,
                "events": events,
                "channel_bytes": channel_bytes,
                "total_wire_bytes": sum(channel_bytes.values()),
                "elapsed_ns": time.perf_counter_ns() - self._active["started_ns"],
                "raw_tensors_persisted": False,
            }
            self.completed.append(deepcopy(summary))
            self._active = None
            return summary

    def abort_step(self) -> None:
        """Clear an incomplete step without manufacturing a completed trace."""
        with self._lock:
            self._active = None


class _Q8ActivationTransfer(torch.autograd.Function):
    """Send q8 codes/metadata to the remote device with an STE backward."""

    @staticmethod
    def forward(ctx, source: torch.Tensor, codes: torch.Tensor, minimum: torch.Tensor,
                scale: torch.Tensor, bridge: "CCTeeGpuRemotePath", token: int):
        bridge._synchronize()
        started = time.perf_counter_ns()
        codes_remote = codes.to(bridge.remote_device, non_blocking=False, copy=True)
        minimum_remote = minimum.to(bridge.remote_device, non_blocking=False, copy=True)
        scale_remote = scale.to(bridge.remote_device, non_blocking=False, copy=True)
        value_remote = codes_remote.to(dtype=source.dtype) * scale_remote + minimum_remote
        bridge._synchronize()
        elapsed_ns = time.perf_counter_ns() - started
        payload_bytes = _tensor_nbytes(codes)
        metadata_bytes = _tensor_nbytes(minimum) + _tensor_nbytes(scale)
        bridge._record_transfer(
            token=token,
            channel="forward_values",
            source=source,
            source_device=bridge.trusted_device,
            target_device=bridge.remote_device,
            payload_bytes=payload_bytes,
            metadata_bytes=metadata_bytes,
            elapsed_ns=elapsed_ns,
            encoding="uint8-affine-per-token",
            tensor_count=3,
            metadata={
                "minimum": {"shape": list(minimum.shape), "dtype": _dtype_name(minimum),
                            "bytes": _tensor_nbytes(minimum)},
                "scale": {"shape": list(scale.shape), "dtype": _dtype_name(scale),
                          "bytes": _tensor_nbytes(scale)},
            },
        )
        ctx.bridge = bridge
        ctx.token = token
        return value_remote

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        bridge = ctx.bridge
        grad_output = grad_output.contiguous()
        bridge._synchronize()
        started = time.perf_counter_ns()
        grad_source = grad_output.to(bridge.trusted_device, non_blocking=False, copy=True)
        bridge._synchronize()
        elapsed_ns = time.perf_counter_ns() - started
        bridge._record_transfer(
            token=ctx.token,
            channel="outbound_gradients",
            source=grad_output,
            source_device=bridge.remote_device,
            target_device=bridge.trusted_device,
            payload_bytes=_tensor_nbytes(grad_output),
            metadata_bytes=0,
            elapsed_ns=elapsed_ns,
            encoding="raw-fp32",
            tensor_count=1,
        )
        return grad_source, None, None, None, None, None


class _ReturnedValueTransfer(torch.autograd.Function):
    """Return the remote result; its backward sends ``dL/dy`` to CUDA."""

    @staticmethod
    def forward(ctx, source: torch.Tensor, bridge: "CCTeeGpuRemotePath", token: int):
        source = source.contiguous()
        bridge._synchronize()
        started = time.perf_counter_ns()
        trusted = source.to(bridge.trusted_device, non_blocking=False, copy=True)
        bridge._synchronize()
        elapsed_ns = time.perf_counter_ns() - started
        bridge._record_transfer(
            token=token,
            channel="returned_values",
            source=source,
            source_device=bridge.remote_device,
            target_device=bridge.trusted_device,
            payload_bytes=_tensor_nbytes(source),
            metadata_bytes=0,
            elapsed_ns=elapsed_ns,
            encoding="raw-fp32",
            tensor_count=1,
        )
        ctx.bridge = bridge
        ctx.token = token
        return trusted

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        bridge = ctx.bridge
        grad_output = grad_output.contiguous()
        bridge._synchronize()
        started = time.perf_counter_ns()
        grad_remote = grad_output.to(bridge.remote_device, non_blocking=False, copy=True)
        bridge._synchronize()
        elapsed_ns = time.perf_counter_ns() - started
        bridge._record_transfer(
            token=ctx.token,
            channel="inbound_gradients",
            source=grad_output,
            source_device=bridge.trusted_device,
            target_device=bridge.remote_device,
            payload_bytes=_tensor_nbytes(grad_output),
            metadata_bytes=0,
            elapsed_ns=elapsed_ns,
            encoding="raw-fp32",
            tensor_count=1,
        )
        return grad_remote, None, None


class CCTeeGpuRemotePath(DefendedRemotePath):
    """State-dict-compatible replacement for an existing defended remote path."""

    def __init__(self, original: DefendedRemotePath, remote_device: torch.device,
                 *, test_only_cpu_loopback: bool):
        nn.Module.__init__(self)
        self.cfg = original.cfg
        for key in ("encoder_ln", "encoder", "remote", "decoder", "quantizer", "defense"):
            setattr(self, key, getattr(original, key))
        if self.cfg.q_bits != 8:
            raise ValueError("the confidential-compute bridge is frozen to q8")
        if self.cfg.grad_defense is not None:
            raise ValueError("gradient defense is outside the frozen CC bridge protocol")
        self.trusted_device = torch.device("cpu")
        self.remote_device = remote_device
        self.test_only_cpu_loopback = test_only_cpu_loopback
        self.trace = BoundaryStepTrace(
            self.trusted_device, self.remote_device, self.test_only_cpu_loopback
        )
        self.control_transfers = ControlTransferLedger(
            self.trusted_device, self.remote_device, self.test_only_cpu_loopback
        )
        self._owner_ref: weakref.ReferenceType[nn.Module] | None = None

    def _set_owner(self, owner: nn.Module) -> None:
        self._owner_ref = weakref.ref(owner)

    def begin_step(self, step_id: int | str, request_id: str,
                   *, expect_backward: bool = True) -> int:
        return self.trace.begin_step(step_id, request_id, expect_backward=expect_backward)

    def end_step(self) -> dict[str, Any]:
        return self.trace.end_step()

    def abort_step(self) -> None:
        self.trace.abort_step()

    @property
    def completed_traces(self) -> tuple[dict[str, Any], ...]:
        return tuple(deepcopy(self.trace.completed))

    def _synchronize(self) -> None:
        if self.remote_device.type == "cuda":
            torch.cuda.synchronize(self.remote_device)

    def _move_remote_to_device(self, purpose: str) -> dict[str, Any]:
        if purpose not in CONTROL_TRANSFER_PURPOSES:
            raise ValueError(f"unknown control-transfer purpose: {purpose}")
        tensors = list(_module_tensors(self.remote))
        if not tensors:
            raise RuntimeError("remote surrogate has no state to transfer")
        wrong = [(name, str(value.device)) for name, value in tensors
                 if value.device != self.trusted_device]
        if wrong:
            raise RuntimeError(
                "remote surrogate setup must originate entirely on trusted CPU; "
                f"unexpected_devices={wrong}"
            )
        rows = [_tensor_descriptor(name, value) for name, value in tensors]
        self._synchronize()
        started = time.perf_counter_ns()
        self.remote.to(self.remote_device)
        self._synchronize()
        elapsed_ns = time.perf_counter_ns() - started
        wrong = [(name, str(value.device)) for name, value in _module_tensors(self.remote)
                 if value.device != self.remote_device]
        if wrong:
            raise RuntimeError(
                "remote surrogate setup did not reach the selected device; "
                f"unexpected_devices={wrong}"
            )
        return self.control_transfers.record(
            purpose=purpose,
            direction="trusted-to-remote",
            tensors=rows,
            elapsed_ns=elapsed_ns,
        )

    def copy_payload_to_trusted(
            self, payload: Any, *, purpose: str,
            request_id: str | None = None,
            step_id: int | str | None = None) -> tuple[Any, dict[str, Any]]:
        """Materialise a model/optimizer/checkpoint payload in trusted CPU memory.

        Every CUDA tensor is copied explicitly with ``copy=True`` and included
        in one synchronized metadata-only control-transfer event.  CPU tensors
        are cloned so the returned snapshot cannot mutate live state, but those
        same-device clones are not counted as boundary traffic.
        """
        if purpose not in CONTROL_TRANSFER_PURPOSES:
            raise ValueError(f"unknown control-transfer purpose: {purpose}")
        rows: list[dict[str, Any]] = []

        def copy_value(value: Any, name: str) -> Any:
            if isinstance(value, torch.Tensor):
                if value.device == self.remote_device and self.remote_device.type == "cuda":
                    rows.append(_tensor_descriptor(name, value))
                    return value.detach().to(
                        self.trusted_device, non_blocking=False, copy=True
                    )
                if value.device == self.trusted_device:
                    return value.detach().clone()
                raise RuntimeError(
                    f"snapshot tensor {name!r} is on undeclared device {value.device}"
                )
            if isinstance(value, OrderedDict):
                copied = OrderedDict(
                    (key, copy_value(item, f"{name}.{key}" if name else str(key)))
                    for key, item in value.items()
                )
                if hasattr(value, "_metadata"):
                    copied._metadata = deepcopy(value._metadata)
                return copied
            if isinstance(value, Mapping):
                return {
                    key: copy_value(item, f"{name}.{key}" if name else str(key))
                    for key, item in value.items()
                }
            if isinstance(value, list):
                return [copy_value(item, f"{name}[{index}]")
                        for index, item in enumerate(value)]
            if isinstance(value, tuple):
                return tuple(copy_value(item, f"{name}[{index}]")
                             for index, item in enumerate(value))
            return deepcopy(value)

        self._synchronize()
        started = time.perf_counter_ns()
        snapshot = copy_value(payload, "payload")
        self._synchronize()
        elapsed_ns = time.perf_counter_ns() - started
        event = self.control_transfers.record(
            purpose=purpose,
            direction="remote-to-trusted",
            tensors=rows,
            elapsed_ns=elapsed_ns,
            request_id=request_id,
            step_id=step_id,
        )
        return snapshot, event

    @property
    def control_transfer_events(self) -> tuple[dict[str, Any], ...]:
        return self.control_transfers.events

    def control_transfer_events_since(self, sequence: int) -> tuple[dict[str, Any], ...]:
        return self.control_transfers.events_since(sequence)

    def control_transfer_summary(self) -> dict[str, Any]:
        return self.control_transfers.summary()

    def _record_transfer(self, *, token: int, channel: str, source: torch.Tensor,
                         source_device: torch.device, target_device: torch.device,
                         payload_bytes: int, metadata_bytes: int, elapsed_ns: int,
                         encoding: str, tensor_count: int,
                         metadata: dict[str, Any] | None = None) -> None:
        if source.device != source_device:
            raise RuntimeError(
                f"{channel} originated on {source.device}, expected {source_device}"
            )
        if encoding == "raw-fp32" and source.dtype != torch.float32:
            raise RuntimeError(f"{channel} left the frozen fp32 transport contract")
        event = {
            "channel": channel,
            "direction": _DIRECTIONS[channel],
            "source_device": str(source_device),
            "target_device": str(target_device),
            "shape": list(source.shape),
            "dtype": _dtype_name(source),
            "encoding": encoding,
            "tensor_count": tensor_count,
            "payload_bytes": int(payload_bytes),
            "metadata_bytes": int(metadata_bytes),
            "wire_bytes": int(payload_bytes + metadata_bytes),
            "elapsed_ns": int(elapsed_ns),
            "synchronized": True,
            "test_only_cpu_loopback": self.test_only_cpu_loopback,
        }
        if metadata is not None:
            event["metadata"] = metadata
        self.trace.record(token, event)

    def placement_report(self) -> dict[str, Any]:
        owner = self._owner_ref() if self._owner_ref is not None else None
        if owner is None:
            raise RuntimeError("CC bridge is not installed on a live model")
        remote_ids = {id(value) for _, value in _module_tensors(self.remote)}
        if not remote_ids:
            raise RuntimeError("remote surrogate has no parameters or buffers to place")
        remote, trusted = [], []
        for name, value in list(owner.named_parameters()) + list(owner.named_buffers()):
            target = remote if id(value) in remote_ids else trusted
            target.append({"name": name, "device": str(value.device),
                           "dtype": _dtype_name(value),
                           "floating_point": value.is_floating_point()})
        invalid_remote = [row for row in remote if torch.device(row["device"]) != self.remote_device]
        invalid_trusted = [row for row in trusted
                           if torch.device(row["device"]) != self.trusted_device]
        invalid_precision = [row for row in remote + trusted
                             if row["floating_point"] and row["dtype"] != "float32"]
        if invalid_remote or invalid_trusted or invalid_precision:
            raise RuntimeError(
                "CC placement drift: only the remote surrogate may reside on the remote device; "
                "all floating tensors must remain fp32; "
                f"invalid_remote={invalid_remote}, invalid_trusted={invalid_trusted}, "
                f"invalid_precision={invalid_precision}"
            )
        return {
            "trusted_device": str(self.trusted_device),
            "remote_device": str(self.remote_device),
            "test_only_cpu_loopback": self.test_only_cpu_loopback,
            "trusted_tensor_count": len(trusted),
            "remote_tensor_count": len(remote),
            "remote_tensor_names": [row["name"] for row in remote],
        }

    @staticmethod
    def _q8_codes(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if value.device.type != "cpu" or value.dtype != torch.float32:
            raise RuntimeError("trusted q8 boundary input must be CPU float32")
        if value.ndim != 3 or not torch.isfinite(value).all():
            raise ValueError("q8 boundary input must be a finite [batch, sequence, width] tensor")
        detached = value.detach()
        minimum = detached.amin(dim=-1, keepdim=True).contiguous()
        maximum = detached.amax(dim=-1, keepdim=True)
        scale = ((maximum - minimum) / 255.0).clamp_min(1e-12).contiguous()
        codes = ((detached - minimum) / scale).round().clamp(0, 255).to(torch.uint8)
        return codes.contiguous(), minimum, scale

    def forward(self, x: torch.Tensor, capture=None) -> torch.Tensor:
        self.placement_report()
        token = self.trace.active_token()
        if x.device != self.trusted_device:
            raise RuntimeError("trusted residual entered the CC bridge off CPU")
        pre = self.encoder(self.encoder_ln(x))
        defended = self.defense(pre)
        codes, minimum, scale = self._q8_codes(defended)
        decoded = codes.to(dtype=defended.dtype) * scale + minimum
        trusted_values = defended + (decoded - defended).detach()
        if capture is not None:
            # Observe the already-available trusted copies.  Calling the legacy
            # capture on CUDA tensors would create four additional, unaccounted
            # diagnostic transfers across the very boundary being measured.
            capture.on_forward_values(trusted_values)
            metadata = {
                "minimum": minimum.detach().clone(),
                "scale": scale.detach().clone(),
            }
            if hasattr(capture, "on_quantizer_metadata"):
                capture.on_quantizer_metadata(metadata)
            elif hasattr(capture, "values"):
                capture.values["quantizer_metadata"] = metadata
        remote_values = _Q8ActivationTransfer.apply(
            trusted_values, codes, minimum, scale, self, token
        )
        remote_result = self.remote(remote_values)
        trusted_result = _ReturnedValueTransfer.apply(remote_result, self, token)
        if capture is not None:
            capture.on_returned_values(trusted_result)
        return x + self.decoder(trusted_result)


def _resolve_remote_device(remote_device: str | torch.device,
                           test_only_cpu_loopback: bool) -> torch.device:
    device = torch.device(remote_device)
    if test_only_cpu_loopback:
        if device.type != "cpu":
            raise ValueError("test-only loopback requires remote_device='cpu'")
        if "PYTEST_CURRENT_TEST" not in os.environ:
            raise PermissionError("CPU loopback is permitted only from a pytest unit test")
        return torch.device("cpu")
    if device.type != "cuda":
        raise PermissionError(
            "production CC bridge requires a distinct CUDA remote device; "
            "same-device and non-CUDA execution are refused"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA remote device requested but CUDA is unavailable")
    index = torch.cuda.current_device() if device.index is None else device.index
    if index < 0 or index >= torch.cuda.device_count():
        raise ValueError(f"CUDA device index {index} is unavailable")
    return torch.device("cuda", index)


def install_cc_bridge(model: nn.Module, remote_device: str | torch.device = "cuda:0",
                      *, test_only_cpu_loopback: bool = False,
                      setup_purpose: str = "model-setup") -> CCTeeGpuRemotePath:
    """Install the mixed-device bridge without changing state-dict key names.

    The complete model is first placed on trusted CPU, then only the surrogate
    subtree is moved to ``remote_device``.  Production mode accepts CUDA only.
    ``test_only_cpu_loopback=True`` is guarded by pytest and must never be used
    as execution evidence.
    """
    resolved = _resolve_remote_device(remote_device, test_only_cpu_loopback)
    if not hasattr(model, "remote_path") or not hasattr(model, "split"):
        raise TypeError("CC bridge requires a split model")
    original = model.remote_path
    if not isinstance(original, DefendedRemotePath):
        raise TypeError("attach the Paper 4 defense before installing the CC bridge")
    before_model_keys = tuple(model.state_dict().keys())
    before_path_keys = tuple(original.state_dict().keys())
    model.to(torch.device("cpu"))
    replacement = CCTeeGpuRemotePath(
        original, resolved, test_only_cpu_loopback=test_only_cpu_loopback
    )
    model.blocks[model.split.start] = replacement
    replacement._move_remote_to_device(setup_purpose)
    replacement._set_owner(model)
    if tuple(replacement.state_dict().keys()) != before_path_keys:
        raise RuntimeError("CC bridge changed remote-path state_dict keys")
    if tuple(model.state_dict().keys()) != before_model_keys:
        raise RuntimeError("CC bridge changed model state_dict keys")
    replacement.placement_report()
    return replacement
