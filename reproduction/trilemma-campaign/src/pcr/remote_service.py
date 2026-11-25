"""Corpus-free remote surrogate service for R1 process-boundary execution."""

from __future__ import annotations

import json
import math
import os
import socket
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from .model import init_module_
from .split import RemovedRemote, Surrogate
from .train import OptimConfig, build_optimizer
from .utils import derive_seed, sha256_file, sha256_json, state_dict_checksum
from .wire import TensorPacket, WireAuditLog, WireError, WireFrame, build_header, encode_tensor, recv_frame, send_frame

MAX_CACHE_ENTRIES = 8
MAX_CACHE_BYTES = 64 << 20


class RemoteProtocolError(RuntimeError):
    """The caller violated the remote execution state machine."""


@dataclass(frozen=True)
class RemoteConfig:
    mode: str
    seed: int
    width: int
    depth: int
    init_std: float
    optimizer: dict[str, Any]
    grad_defense: dict[str, float] | None
    device: str = "cpu"
    resume_checkpoint: str | None = None
    resume_checkpoint_sha256: str | None = None

    @classmethod
    def from_body(cls, body: Mapping[str, Any]) -> "RemoteConfig":
        value = cls(mode=str(body.get("mode")), seed=int(body.get("seed", -1)),
                    width=int(body.get("width", 0)), depth=int(body.get("depth", 0)),
                    init_std=float(body.get("init_std", float("nan"))),
                    optimizer=dict(body.get("optimizer", {})),
                    grad_defense=None if body.get("grad_defense") is None else dict(body["grad_defense"]),
                    device=str(body.get("device", "cpu")),
                    resume_checkpoint=None if body.get("resume_checkpoint") is None else str(body["resume_checkpoint"]),
                    resume_checkpoint_sha256=(None if body.get("resume_checkpoint_sha256") is None
                                              else str(body["resume_checkpoint_sha256"])))
        value.validate()
        return value

    def validate(self) -> None:
        if self.mode not in {"trained", "frozen_random", "removed_remote"}:
            raise RemoteProtocolError(f"unsupported remote mode: {self.mode}")
        if self.seed < 0 or self.width <= 0 or self.width % 32 or self.depth <= 0:
            raise RemoteProtocolError("invalid remote geometry or seed")
        if not math.isfinite(self.init_std) or self.init_std <= 0:
            raise RemoteProtocolError("invalid remote initialization scale")
        if self.device not in {"cpu", "cuda"}:
            raise RemoteProtocolError("remote device must be cpu or cuda")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RemoteProtocolError("remote CUDA device is unavailable")
        if (self.resume_checkpoint is not None and
                (Path(self.resume_checkpoint).name != self.resume_checkpoint or
                 self.resume_checkpoint in {".", ".."})):
            raise RemoteProtocolError("remote resume checkpoint identifier is invalid")
        if (self.resume_checkpoint is None) != (self.resume_checkpoint_sha256 is None):
            raise RemoteProtocolError("remote resume checkpoint identity is incomplete")
        if (self.resume_checkpoint_sha256 is not None and
                (len(self.resume_checkpoint_sha256) != 64 or
                 any(value not in "0123456789abcdef" for value in self.resume_checkpoint_sha256))):
            raise RemoteProtocolError("remote resume checkpoint digest is invalid")
        required = {"lr", "min_lr", "betas", "weight_decay", "warmup_steps", "steps"}
        missing = required - self.optimizer.keys()
        if missing:
            raise RemoteProtocolError(f"missing optimizer fields: {sorted(missing)}")
        lr = self.optimizer["lr"]
        min_lr = self.optimizer["min_lr"]
        weight_decay = self.optimizer["weight_decay"]
        betas = self.optimizer["betas"]
        for name, value in (("lr", lr), ("min_lr", min_lr), ("weight_decay", weight_decay)):
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise RemoteProtocolError(f"invalid optimizer {name}")
        if lr <= 0 or min_lr > lr:
            raise RemoteProtocolError("optimizer learning-rate range is invalid")
        if (not isinstance(betas, (list, tuple)) or len(betas) != 2 or
                any(not isinstance(value, (int, float)) or not math.isfinite(value) or
                    not 0 <= value < 1 for value in betas)):
            raise RemoteProtocolError("optimizer betas are invalid")
        for name in ("warmup_steps", "steps"):
            value = self.optimizer[name]
            if type(value) is not int or value < 0:
                raise RemoteProtocolError(f"invalid optimizer {name}")
        if self.optimizer["steps"] <= 0 or self.optimizer["warmup_steps"] > self.optimizer["steps"]:
            raise RemoteProtocolError("optimizer step schedule is invalid")
        if self.grad_defense is not None:
            clip = self.grad_defense.get("clip_norm")
            noise = self.grad_defense.get("noise_std")
            if (not isinstance(clip, (int, float)) or not math.isfinite(clip) or clip <= 0 or
                    not isinstance(noise, (int, float)) or not math.isfinite(noise) or noise < 0):
                raise RemoteProtocolError("invalid gradient defense")


@dataclass
class ActiveStep:
    request_id: str
    step: int
    measurement: bool
    latent: torch.Tensor
    returned: torch.Tensor
    remote_grad_norm_sq: float | None = None
    capture_kind: str = "panel"


@dataclass(frozen=True)
class CachedResponse:
    fingerprint: str
    header: dict[str, Any]
    tensor: TensorPacket | None


class RemoteSession:
    def __init__(self, session_id: str, config: RemoteConfig, output_dir: Path,
                 resume_root: Path) -> None:
        self.session_id = session_id
        self.config = config
        self.output_dir = output_dir
        self.resume_root = resume_root.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.device = torch.device(config.device)
        self.module = self._build_module().to(self.device)
        self.optimizer = self._build_optimizer()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self.outbound_rng = torch.Generator().manual_seed(
            derive_seed("r1-gradient-outbound-v1", config.seed)
        )
        self.active: ActiveStep | None = None
        self.last_sequence = -1
        self.cache: OrderedDict[str, CachedResponse] = OrderedDict()
        self.cache_bytes = 0
        self.measurement_captures: list[dict[str, Any]] = []
        self.training_captures: list[dict[str, Any]] = []
        self.completed_steps = 0
        self.last_transaction_id: str | None = None
        if config.resume_checkpoint is not None:
            self._load_checkpoint(self.resume_root / config.resume_checkpoint)

    def _build_module(self) -> torch.nn.Module:
        if self.config.mode == "removed_remote":
            return RemovedRemote(self.config.depth)
        module = Surrogate(self.config.width, self.config.depth)
        stream = f"remote:{self.config.width}:{self.config.depth}"
        generator = torch.Generator().manual_seed(derive_seed("r1-init-v1", self.config.seed, stream))
        init_module_(module, generator, 1, self.config.init_std)
        if self.config.mode == "frozen_random":
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        return module

    def _build_optimizer(self) -> torch.optim.AdamW | None:
        if self.config.mode != "trained":
            return None
        values = self.config.optimizer
        cfg = OptimConfig(lr=float(values["lr"]), min_lr=float(values["min_lr"]),
                          betas=tuple(values["betas"]), weight_decay=float(values["weight_decay"]),
                          warmup_steps=int(values["warmup_steps"]),
                          grad_clip=float(values.get("grad_clip", 1.0)), steps=int(values["steps"]))
        return build_optimizer(self.module, cfg)

    def _load_checkpoint(self, path: Path) -> None:
        resolved = path.resolve()
        if (path.is_symlink() or not resolved.is_relative_to(self.resume_root) or
                any(parent.is_symlink() for parent in path.parents
                    if parent.is_relative_to(self.resume_root)) or not resolved.is_file()):
            raise RemoteProtocolError("remote resume checkpoint is unavailable")
        path = resolved
        if sha256_file(str(path)) != self.config.resume_checkpoint_sha256:
            raise RemoteProtocolError("remote resume checkpoint digest differs")
        value = torch.load(path, map_location=self.device, weights_only=True)
        if not isinstance(value, dict) or value.get("schema_version") != "r1-remote-checkpoint-1.0":
            raise RemoteProtocolError("remote resume checkpoint schema is invalid")
        prior = value.get("config", {})
        for field in ("mode", "seed", "width", "depth", "init_std", "grad_defense", "device"):
            if prior.get(field) != getattr(self.config, field):
                raise RemoteProtocolError(f"remote resume checkpoint differs in {field}")
        self.module.load_state_dict(value["model"])
        if self.optimizer is not None:
            if not isinstance(value.get("optimizer"), dict):
                raise RemoteProtocolError("trained remote checkpoint has no optimizer")
            self.optimizer.load_state_dict(value["optimizer"])
        elif value.get("optimizer") is not None:
            raise RemoteProtocolError("non-trained remote checkpoint contains an optimizer")
        self.outbound_rng.set_state(value["outbound_rng"])
        self.completed_steps = int(value["completed_steps"])
        self.last_transaction_id = value.get("last_transaction_id")

    def _fingerprint(self, frame: WireFrame) -> str:
        header = {key: value for key, value in frame.header.items() if key != "auth"}
        return sha256_json(header)

    def _cached(self, frame: WireFrame) -> CachedResponse | None:
        cached = self.cache.get(frame.header["request_id"])
        if cached is None:
            return None
        if cached.fingerprint != self._fingerprint(frame):
            raise RemoteProtocolError("a request id was reused with different content")
        return cached

    def _remember(self, frame: WireFrame, header: dict[str, Any],
                  tensor: TensorPacket | None) -> CachedResponse:
        response = CachedResponse(self._fingerprint(frame), header, tensor)
        response_bytes = len(tensor.payload) if tensor is not None else 0
        if response_bytes > MAX_CACHE_BYTES:
            raise RemoteProtocolError("response exceeds the idempotency cache byte limit")
        self.cache[frame.header["request_id"]] = response
        self.cache_bytes += response_bytes
        while len(self.cache) > MAX_CACHE_ENTRIES or self.cache_bytes > MAX_CACHE_BYTES:
            _, removed = self.cache.popitem(last=False)
            self.cache_bytes -= len(removed.tensor.payload) if removed.tensor is not None else 0
        return response

    def _sequence(self, frame: WireFrame) -> None:
        sequence = frame.header["sequence"]
        if sequence != self.last_sequence + 1:
            raise RemoteProtocolError(f"expected sequence {self.last_sequence + 1}, got {sequence}")
        self.last_sequence = sequence

    def _response(self, frame: WireFrame, body: Mapping[str, Any],
                  tensor: torch.Tensor | None = None) -> CachedResponse:
        header = build_header(frame.header["message_type"], self.session_id,
                              frame.header["request_id"], frame.header["step"],
                              frame.header["sequence"], body)
        packet = None if tensor is None else encode_tensor(tensor, 32)
        return self._remember(frame, header, packet)

    def handle(self, frame: WireFrame) -> CachedResponse:
        if frame.header["session_id"] != self.session_id:
            raise RemoteProtocolError("session id mismatch")
        cached = self._cached(frame)
        if cached is not None:
            return cached
        self._sequence(frame)
        message_type = frame.header["message_type"]
        if message_type == "FORWARD":
            return self._forward(frame)
        if message_type == "BACKWARD":
            return self._backward(frame)
        if message_type == "STEP":
            return self._step(frame)
        if message_type == "CHECKPOINT":
            return self._checkpoint(frame)
        if message_type == "CLOSE":
            if self.active is not None:
                raise RemoteProtocolError("cannot close a session with an active step")
            if frame.header["step"] != self.completed_steps:
                raise RemoteProtocolError("CLOSE step differs from the committed session step")
            return self._response(frame, {"status": "closed", "completed_steps": self.completed_steps})
        raise RemoteProtocolError(f"unexpected message after HELLO: {message_type}")

    def _forward(self, frame: WireFrame) -> CachedResponse:
        if self.active is not None or frame.tensor is None:
            raise RemoteProtocolError("FORWARD requires one tensor and no active step")
        body = frame.header["body"]
        training = bool(body.get("training"))
        measurement = bool(body.get("measurement"))
        capture_kind = str(body.get("capture_kind", "panel"))
        if capture_kind not in ("panel", "training_frame") or (capture_kind != "panel" and not measurement):
            raise RemoteProtocolError("unsupported measurement capture kind")
        if training and measurement:
            raise RemoteProtocolError("FORWARD cannot be training and measurement simultaneously")
        if frame.header["step"] != self.completed_steps:
            raise RemoteProtocolError("FORWARD step differs from the committed session step")
        latent = frame.tensor.detach().to(self.device).requires_grad_(training or measurement)
        with torch.set_grad_enabled(training or measurement):
            returned = self.module(latent)
        if training or measurement:
            self.active = ActiveStep(frame.header["request_id"], frame.header["step"],
                                     measurement, latent, returned, capture_kind=capture_kind)
        return self._response(frame, {"training": training, "measurement": measurement,
                                      "remote_state_sha256": state_dict_checksum(self.module.state_dict())},
                              returned.detach())

    def _defend_outbound(self, gradient: torch.Tensor) -> torch.Tensor:
        defense = self.config.grad_defense
        if defense is None:
            return gradient
        norm = gradient.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        value = gradient * (float(defense["clip_norm"]) / norm).clamp(max=1.0)
        if float(defense["noise_std"]) == 0:
            return value
        noise = torch.randn(value.shape, generator=self.outbound_rng, dtype=torch.float32)
        return value + noise.to(value.device, value.dtype) * float(defense["noise_std"])

    def _backward(self, frame: WireFrame) -> CachedResponse:
        active = self.active
        expected_request = None if active is None else active.request_id.replace(":forward", ":backward")
        if (active is None or frame.tensor is None or frame.header["step"] != active.step or
                frame.header["request_id"] != expected_request or
                bool(frame.header["body"].get("measurement")) != active.measurement):
            raise RemoteProtocolError("BACKWARD does not match the active forward step")
        if self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)
        rng_state = self.outbound_rng.get_state() if active.measurement else None
        inbound = frame.tensor.to(self.device, dtype=active.returned.dtype)
        active.returned.backward(inbound)
        gradients = [parameter.grad for parameter in self.module.parameters() if parameter.grad is not None]
        active.remote_grad_norm_sq = float(sum(torch.sum(value.float() ** 2) for value in gradients))
        raw_gradient = active.latent.grad.detach()
        gradient = self._defend_outbound(raw_gradient)
        if active.measurement:
            event = {
                "step": active.step,
                "forward_values": active.latent.detach().cpu(),
                "returned_values": active.returned.detach().cpu(),
                "inbound_gradients": frame.tensor.detach().cpu(),
                "outbound_gradients_pre_defense": raw_gradient.cpu(),
                "outbound_gradients": gradient.detach().cpu(),
            }
            if active.capture_kind == "panel":
                self.measurement_captures.append(event)
            else:
                self.training_captures.append(event)
            self.outbound_rng.set_state(rng_state)
            if self.optimizer is not None:
                self.optimizer.zero_grad(set_to_none=True)
            self.active = None
        return self._response(frame, {"remote_grad_norm_sq": active.remote_grad_norm_sq,
                                      "measurement": active.measurement}, gradient)

    def _journal(self, event: Mapping[str, Any]) -> None:
        path = self.output_dir / "step-journal.jsonl"
        with path.open("a") as handle:
            handle.write(json.dumps(dict(event), sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _step(self, frame: WireFrame) -> CachedResponse:
        active = self.active
        body = frame.header["body"]
        if active is None or active.remote_grad_norm_sq is None or frame.header["step"] != active.step:
            raise RemoteProtocolError("STEP requires a completed backward pass")
        scale = float(body.get("gradient_scale", float("nan")))
        lr = float(body.get("lr", float("nan")))
        transaction_id = body.get("transaction_id")
        if (not math.isfinite(scale) or not 0 < scale <= 1 or not math.isfinite(lr) or lr <= 0 or
                not isinstance(transaction_id, str) or len(transaction_id) != 64 or
                any(value not in "0123456789abcdef" for value in transaction_id)):
            raise RemoteProtocolError("invalid optimizer step values")
        before = state_dict_checksum(self.module.state_dict())
        self._journal({"status": "prepared", "step": active.step,
                       "transaction_id": transaction_id, "remote_state_before": before})
        if self.optimizer is not None:
            for parameter in self.module.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(scale)
            for group in self.optimizer.param_groups:
                group["lr"] = lr
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
        self.completed_steps += 1
        self.last_transaction_id = transaction_id
        after = state_dict_checksum(self.module.state_dict())
        self._journal({"status": "committed", "step": active.step,
                       "transaction_id": transaction_id, "remote_state_before": before,
                       "remote_state_after": after, "completed_steps": self.completed_steps})
        self.active = None
        return self._response(frame, {"status": "stepped", "completed_steps": self.completed_steps,
                                      "transaction_id": transaction_id,
                                      "remote_state_sha256": after})

    def _checkpoint(self, frame: WireFrame) -> CachedResponse:
        if self.active is not None:
            raise RemoteProtocolError("cannot checkpoint an active step")
        if frame.header["step"] != self.completed_steps:
            raise RemoteProtocolError("checkpoint step differs from the committed session step")
        path = self.output_dir / f"remote-checkpoint-step-{self.completed_steps}.pt"
        if path.exists():
            raise RemoteProtocolError("checkpoint path already exists")
        torch.save({"schema_version": "r1-remote-checkpoint-1.0", "config": self.config.__dict__,
                    "completed_steps": self.completed_steps,
                    "last_transaction_id": self.last_transaction_id,
                    "model": self.module.state_dict(),
                    "optimizer": None if self.optimizer is None else self.optimizer.state_dict(),
                    "outbound_rng": self.outbound_rng.get_state()}, path)
        captures = self.output_dir / f"remote-captures-step-{self.completed_steps}.pt"
        torch.save(self.measurement_captures, captures)
        body = {"path": path.name, "checkpoint_id": path.name,
                "bytes": path.stat().st_size,
                "checkpoint_sha256": sha256_file(str(path)),
                "captures_path": captures.name,
                "captures_bytes": captures.stat().st_size,
                "completed_steps": self.completed_steps,
                "last_transaction_id": self.last_transaction_id,
                "device": str(self.device),
                "cuda_peak_allocated_bytes": (torch.cuda.max_memory_allocated(self.device)
                                              if self.device.type == "cuda" else None),
                "cuda_peak_reserved_bytes": (torch.cuda.max_memory_reserved(self.device)
                                             if self.device.type == "cuda" else None),
                "remote_state_sha256": state_dict_checksum(self.module.state_dict()),
                "model_state_sha256": state_dict_checksum(self.module.state_dict())}
        if self.training_captures:
            training_captures = self.output_dir / f"remote-training-captures-step-{self.completed_steps}.pt"
            if training_captures.exists():
                raise RemoteProtocolError("remote training captures path already exists")
            torch.save(self.training_captures, training_captures)
            body["training_captures_path"] = training_captures.name
            body["training_captures_bytes"] = training_captures.stat().st_size
            body["training_captures_sha256"] = sha256_file(str(training_captures))
        return self._response(frame, body)


def _hello(frame: WireFrame, output_dir: Path,
           resume_root: Path) -> tuple[RemoteSession, CachedResponse]:
    if (frame.header["message_type"] != "HELLO" or frame.tensor is not None or
            frame.header["sequence"] != 0 or frame.header["step"] != 0):
        raise RemoteProtocolError("first message must be HELLO at sequence and step zero")
    config = RemoteConfig.from_body(frame.header["body"])
    session = RemoteSession(frame.header["session_id"], config, output_dir, resume_root)
    session.last_sequence = frame.header["sequence"]
    header = build_header("HELLO", session.session_id, frame.header["request_id"], 0,
                          frame.header["sequence"], {"status": "ready",
                          "remote_state_sha256": state_dict_checksum(session.module.state_dict()),
                          "remote_parameter_count": sum(p.numel() for p in session.module.parameters()),
                          "completed_steps": session.completed_steps,
                          "last_transaction_id": session.last_transaction_id})
    response = session._remember(frame, header, None)
    return session, response


def serve_connection(connection: socket.socket, secret: bytes, output_dir: Path,
                     resume_root: Path) -> None:
    audit = WireAuditLog()
    session: RemoteSession | None = None
    try:
        first = recv_frame(connection, secret, audit.append)
        session, response = _hello(first, output_dir, resume_root)
        send_frame(connection, response.header, secret, response.tensor, audit.append)
        while True:
            frame = recv_frame(connection, secret, audit.append)
            response = session.handle(frame)
            send_frame(connection, response.header, secret, response.tensor, audit.append)
            if frame.header["message_type"] == "CLOSE":
                break
    except BaseException as exc:
        if session is not None:
            try:
                header = build_header("ERROR", session.session_id, "remote-error",
                                      session.completed_steps, session.last_sequence + 1,
                                      {"error_type": type(exc).__name__, "detail": str(exc)})
                send_frame(connection, header, secret, recorder=audit.append)
            except BaseException:
                pass
        raise
    finally:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "remote-wire-audit.json").write_text(
            json.dumps(audit.to_dict(), indent=2, sort_keys=True) + "\n"
        )


def serve_once(host: str, port: int, secret: bytes, output_dir: Path,
               ready_path: Path | None = None, timeout: float = 30.0,
               resume_root: Path | None = None) -> None:
    if host not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError("remote service binds loopback only; use an SSH tunnel for remote hosts")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("remote service timeout must be positive")
    effective_resume_root = (resume_root or output_dir.parent / "resume").resolve()
    effective_resume_root.mkdir(parents=True, exist_ok=True)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen(1)
        if ready_path is not None:
            # The client polls for this file's existence, so publish it only
            # after the full payload is durable: a buffered write into the
            # final name can be observed as an empty file under load.
            temporary = ready_path.with_name(ready_path.name + ".partial")
            with temporary.open("x") as handle:
                handle.write(json.dumps({"host": host, "port": server.getsockname()[1]}) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, ready_path)
        connection, _ = server.accept()
        with connection:
            connection.settimeout(timeout)
            serve_connection(connection, secret, output_dir, effective_resume_root)
