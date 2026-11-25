"""Trusted-client side of the R1 process-boundary split model."""

from __future__ import annotations

import json
import math
import socket
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from .modes import trainable_parameters
from .split import AffineQuantizer, SplitBackbone
from .train import OptimConfig, build_optimizer
from .utils import derive_seed, sha256_json, state_dict_checksum
from .wire import TensorPacket, WireAuditLog, WireFrame, build_header, decode_tensor, encode_tensor, recv_frame, send_frame


class RemoteExecutionError(RuntimeError):
    """The remote service rejected or could not complete a request."""


@dataclass(frozen=True)
class RequestContext:
    step: int
    request_id: str
    training: bool
    measurement: bool = False
    capture_kind: str = "panel"


@dataclass(frozen=True)
class RemoteReply:
    body: dict[str, Any]
    tensor: torch.Tensor | None
    wire_bytes: int


class RemoteClient:
    def __init__(self, connection: socket.socket, secret: bytes, session_id: str,
                 q_bits: int) -> None:
        self.connection = connection
        self.secret = secret
        self.session_id = session_id
        self.q_bits = q_bits
        self.sequence = 0
        self.audit = WireAuditLog()
        self._last_request: tuple[dict[str, Any], TensorPacket | None] | None = None

    @classmethod
    def connect(cls, host: str, port: int, secret: bytes, session_id: str,
                q_bits: int, timeout: float = 30.0) -> "RemoteClient":
        connection = socket.create_connection((host, port), timeout=timeout)
        connection.settimeout(timeout)
        return cls(connection, secret, session_id, q_bits)

    def close_socket(self) -> None:
        self.connection.close()

    def _exchange(self, message_type: str, request_id: str, step: int,
                  body: Mapping[str, Any] | None = None,
                  tensor: TensorPacket | None = None) -> RemoteReply:
        header = build_header(message_type, self.session_id, request_id, step,
                              self.sequence, body)
        self._last_request = (header, tensor)
        sent = send_frame(self.connection, header, self.secret, tensor, self.audit.append)
        frame = recv_frame(self.connection, self.secret, self.audit.append)
        self._validate_response(frame, message_type, request_id, self.sequence)
        self.sequence += 1
        return RemoteReply(dict(frame.header["body"]), frame.tensor, sent + frame.wire_bytes)

    def _validate_response(self, frame: WireFrame, message_type: str,
                           request_id: str, sequence: int) -> None:
        if frame.header["message_type"] == "ERROR":
            body = frame.header["body"]
            raise RemoteExecutionError(f"{body.get('error_type')}: {body.get('detail')}")
        expected = (message_type, self.session_id, request_id, sequence)
        observed = (frame.header["message_type"], frame.header["session_id"],
                    frame.header["request_id"], frame.header["sequence"])
        if observed != expected:
            raise RemoteExecutionError(f"response identity mismatch: {observed!r} != {expected!r}")

    def replay_last(self) -> RemoteReply:
        if self._last_request is None:
            raise RemoteExecutionError("no request is available for replay")
        header, tensor = self._last_request
        sent = send_frame(self.connection, header, self.secret, tensor, self.audit.append)
        frame = recv_frame(self.connection, self.secret, self.audit.append)
        self._validate_response(frame, header["message_type"], header["request_id"],
                                header["sequence"])
        return RemoteReply(dict(frame.header["body"]), frame.tensor, sent + frame.wire_bytes)

    def hello(self, config: Mapping[str, Any], expected_state_sha256: str) -> dict[str, Any]:
        reply = self._exchange("HELLO", "hello", 0, config)
        if reply.body.get("remote_state_sha256") != expected_state_sha256:
            raise RemoteExecutionError("remote initialization does not match the trusted expectation")
        return reply.body

    def forward(self, latent: torch.Tensor, context: RequestContext) -> RemoteReply:
        packet = encode_tensor(latent, self.q_bits)
        body = {"training": context.training,
                "measurement": context.measurement}
        if context.capture_kind != "panel":
            body["capture_kind"] = context.capture_kind
        return self._exchange("FORWARD", context.request_id, context.step, body, packet)

    def backward(self, gradient: torch.Tensor, context: RequestContext) -> RemoteReply:
        request_id = context.request_id.replace(":forward", ":backward")
        return self._exchange("BACKWARD", request_id, context.step,
                              {"measurement": context.measurement}, encode_tensor(gradient, 32))

    def step(self, step: int, gradient_scale: float, lr: float,
             transaction_id: str) -> dict[str, Any]:
        reply = self._exchange("STEP", f"train:{step}:step", step,
                               {"gradient_scale": gradient_scale, "lr": lr,
                                "transaction_id": transaction_id})
        return reply.body

    def checkpoint(self, step: int) -> dict[str, Any]:
        return self._exchange("CHECKPOINT", f"checkpoint:{step}", step).body

    def close(self, step: int) -> dict[str, Any]:
        try:
            return self._exchange("CLOSE", f"close:{step}", step).body
        finally:
            self.close_socket()

    def write_audit(self, path: Path) -> None:
        path.write_text(json.dumps(self.audit.to_dict(), indent=2, sort_keys=True) + "\n")


class _RemoteAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, latent: torch.Tensor, proxy: "RemoteProxy") -> torch.Tensor:
        context = proxy.require_context()
        reply = proxy.client.forward(latent, context)
        if reply.tensor is None:
            raise RemoteExecutionError("FORWARD response contained no tensor")
        ctx.proxy = proxy
        ctx.context = context
        proxy.record_forward(context, latent, reply.tensor)
        return reply.tensor.to(device=latent.device, dtype=latent.dtype)

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        value = ctx.proxy.backward_for(ctx.context, gradient)
        return value, None


class RemoteProxy(torch.nn.Module):
    def __init__(self, client: RemoteClient, seed: int,
                 grad_defense: Mapping[str, float] | None) -> None:
        super().__init__()
        self.client = client
        self.grad_defense = None if grad_defense is None else dict(grad_defense)
        self.inbound_rng = torch.Generator().manual_seed(
            derive_seed("r1-gradient-inbound-v1", seed)
        )
        self.context: RequestContext | None = None
        self.remote_grad_norm_sq = 0.0
        self.last_capture: dict[str, Any] | None = None

    def set_context(self, context: RequestContext) -> None:
        if self.context is not None:
            raise RemoteExecutionError("remote request context is already active")
        self.context = context

    def require_context(self) -> RequestContext:
        if self.context is None:
            raise RemoteExecutionError("remote request context was not set")
        return self.context

    def record_forward(self, context: RequestContext, latent: torch.Tensor,
                       returned: torch.Tensor) -> None:
        if context.measurement:
            packet = encode_tensor(latent, self.client.q_bits)
            self.last_capture = {
                "step": context.step,
                "forward_values_pre_transport": latent.detach().cpu(),
                "forward_values": decode_tensor(packet.metadata, packet.payload),
                "forward_encoding": packet.metadata,
                "forward_payload_bytes": len(packet.payload),
                "returned_values": returned.detach().cpu(),
            }
        if not context.training and not context.measurement:
            self.context = None

    def _defend_inbound(self, gradient: torch.Tensor) -> torch.Tensor:
        if self.grad_defense is None:
            return gradient
        norm = gradient.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        value = gradient * (float(self.grad_defense["clip_norm"]) / norm).clamp(max=1.0)
        noise_std = float(self.grad_defense["noise_std"])
        if noise_std == 0:
            return value
        noise = torch.randn(value.shape, generator=self.inbound_rng, dtype=torch.float32)
        return value + noise.to(value.device, value.dtype) * noise_std

    def backward_for(self, context: RequestContext, gradient: torch.Tensor) -> torch.Tensor:
        if self.context != context:
            raise RemoteExecutionError("backward context differs from forward context")
        rng_state = self.inbound_rng.get_state() if context.measurement else None
        defended = self._defend_inbound(gradient.detach())
        reply = self.client.backward(defended, context)
        if reply.tensor is None:
            raise RemoteExecutionError("BACKWARD response contained no tensor")
        self.remote_grad_norm_sq = float(reply.body["remote_grad_norm_sq"])
        if context.measurement:
            self.inbound_rng.set_state(rng_state)
            assert self.last_capture is not None
            self.last_capture["inbound_gradients_pre_defense"] = gradient.detach().cpu()
            self.last_capture["inbound_gradients"] = defended.detach().cpu()
            self.last_capture["outbound_gradients"] = reply.tensor.detach().cpu()
        self.context = None
        return reply.tensor.to(device=gradient.device, dtype=gradient.dtype)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return _RemoteAutograd.apply(latent, self)


@dataclass
class DistributedModel:
    model: SplitBackbone
    proxy: RemoteProxy
    client: RemoteClient
    trusted_optimizer: torch.optim.AdamW
    optim_config: OptimConfig
    remote_state_sha256: str
    completed_steps: int = 0
    last_transaction_id: str | None = None
    poisoned: bool = False

    def set_context(self, step: int, request_id: str, training: bool,
                    measurement: bool = False, capture_kind: str = "panel") -> None:
        if self.poisoned:
            raise RemoteExecutionError("distributed model is fail-stopped after an incomplete step")
        if step != self.completed_steps:
            raise RemoteExecutionError("request step differs from the committed distributed step")
        self.proxy.set_context(RequestContext(step, request_id, training, measurement, capture_kind))

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.model(tokens.to(self.device))

    def zero_grad(self) -> None:
        self.trusted_optimizer.zero_grad(set_to_none=True)

    def save_checkpoint(self, path: Path, completed_steps: int,
                        remote: Mapping[str, Any]) -> None:
        if self.poisoned or completed_steps != self.completed_steps:
            raise RemoteExecutionError("cannot checkpoint an incomplete distributed step")
        if (remote.get("completed_steps") != completed_steps or
                remote.get("remote_state_sha256") != self.remote_state_sha256 or
                remote.get("last_transaction_id") != self.last_transaction_id):
            raise RemoteExecutionError("remote checkpoint does not match the client commit state")
        rng = {"cpu": torch.random.get_rng_state(),
               "inbound_defense": self.proxy.inbound_rng.get_state()}
        if torch.cuda.is_available():
            rng["cuda"] = torch.cuda.get_rng_state_all()
        torch.save({"schema_version": "r1-training-checkpoint-1.0",
                    "completed_steps": completed_steps,
                    "last_transaction_id": self.last_transaction_id,
                    "remote_state_sha256": self.remote_state_sha256,
                    "remote_checkpoint_id": remote.get("checkpoint_id"),
                    "remote_checkpoint_sha256": remote.get("checkpoint_sha256"),
                    "model": self.model.state_dict(),
                    "optimizer": self.trusted_optimizer.state_dict(),
                    "rng": rng}, path)

    def load_checkpoint(self, path: Path) -> int:
        value = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(value, dict) or value.get("schema_version") != "r1-training-checkpoint-1.0":
            raise RemoteExecutionError("client checkpoint schema is invalid")
        if value.get("remote_state_sha256") != self.remote_state_sha256:
            raise RemoteExecutionError("client and remote checkpoints do not describe the same state")
        self.model.load_state_dict(value["model"])
        self.trusted_optimizer.load_state_dict(value["optimizer"])
        torch.random.set_rng_state(value["rng"]["cpu"])
        self.proxy.inbound_rng.set_state(value["rng"]["inbound_defense"])
        self.completed_steps = int(value["completed_steps"])
        self.last_transaction_id = value.get("last_transaction_id")
        return self.completed_steps

    def step(self, step: int) -> float:
        if self.poisoned or step != self.completed_steps:
            raise RemoteExecutionError("distributed step order is invalid")
        parameters = trainable_parameters(self.model)
        local_sq = sum(float(torch.sum(parameter.grad.detach().float() ** 2))
                       for parameter in parameters if parameter.grad is not None)
        total_norm = math.sqrt(local_sq + self.proxy.remote_grad_norm_sq)
        scale = min(1.0, self.optim_config.grad_clip / max(total_norm, 1e-12))
        lr = self.optim_config.lr_at(step)
        transaction_id = sha256_json({"session_id": self.client.session_id, "step": step,
                                      "local_grad_norm_sq": local_sq,
                                      "remote_grad_norm_sq": self.proxy.remote_grad_norm_sq,
                                      "gradient_scale": scale, "lr": lr})
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.mul_(scale)
        for group in self.trusted_optimizer.param_groups:
            group["lr"] = lr
        try:
            remote = self.client.step(step, scale, lr, transaction_id)
            if (remote.get("status") != "stepped" or
                    remote.get("transaction_id") != transaction_id or
                    remote.get("completed_steps") != step + 1):
                raise RemoteExecutionError("remote step acknowledgement is inconsistent")
            self.trusted_optimizer.step()
        except BaseException:
            self.poisoned = True
            raise
        finally:
            self.zero_grad()
        self.completed_steps = step + 1
        self.last_transaction_id = transaction_id
        self.remote_state_sha256 = str(remote["remote_state_sha256"])
        return total_norm


def attach_remote_client(model: SplitBackbone, client: RemoteClient, seed: int,
                         grad_defense: Mapping[str, float] | None,
                         optim_config: OptimConfig,
                         remote_checkpoint: str | None = None,
                         remote_checkpoint_sha256: str | None = None,
                         expected_remote_state_sha256: str | None = None,
                         remote_device: str = "cpu") -> DistributedModel:
    remote_state = (expected_remote_state_sha256 or
                    state_dict_checksum(model.remote_path.remote.state_dict()))
    config = {"mode": "removed_remote" if not list(model.remote_path.remote.parameters()) else
              ("trained" if any(p.requires_grad for p in model.remote_path.remote.parameters()) else "frozen_random"),
              "seed": seed, "width": model.split.width, "depth": model.split.remote_depth,
              "init_std": model.cfg.init_std, "grad_defense": None if grad_defense is None else dict(grad_defense),
              "device": remote_device,
              "optimizer": {"lr": optim_config.lr, "min_lr": optim_config.min_lr,
                            "betas": list(optim_config.betas), "weight_decay": optim_config.weight_decay,
                            "warmup_steps": optim_config.warmup_steps,
                            "grad_clip": optim_config.grad_clip, "steps": optim_config.steps},
              "resume_checkpoint": remote_checkpoint,
              "resume_checkpoint_sha256": remote_checkpoint_sha256}
    hello = client.hello(config, remote_state)
    proxy = RemoteProxy(client, seed, grad_defense)
    model.remote_path.remote = proxy
    model.remote_path.quantizer = AffineQuantizer(32)
    model.remote_path.cfg = replace(model.remote_path.cfg, grad_defense=None)
    optimizer = build_optimizer(model, optim_config)
    return DistributedModel(model, proxy, client, optimizer, optim_config,
                            remote_state_sha256=str(hello["remote_state_sha256"]),
                            completed_steps=int(hello.get("completed_steps", 0)),
                            last_transaction_id=hello.get("last_transaction_id"))


def train_step(distributed: DistributedModel, tokens: torch.Tensor,
               targets: torch.Tensor, step: int) -> dict[str, float]:
    distributed.set_context(step, f"train:{step}:forward", training=True)
    logits = distributed.forward(tokens)
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                           targets.to(distributed.device).reshape(-1))
    loss.backward()
    norm = distributed.step(step)
    return {"loss": float(loss.detach()), "gradient_norm": norm}


def evaluate(distributed: DistributedModel,
             batches: list[tuple[torch.Tensor, torch.Tensor]], step: int) -> float:
    total = 0.0
    with torch.no_grad():
        for index, (tokens, targets) in enumerate(batches):
            distributed.set_context(step, f"eval:{step}:{index}:forward", training=False)
            logits = distributed.forward(tokens)
            total += float(F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.to(distributed.device).reshape(-1),
            ))
    return total / len(batches)


def capture_measurement(distributed: DistributedModel, tokens: torch.Tensor,
                        targets: torch.Tensor, step: int, label: str,
                        capture_kind: str = "panel") -> dict[str, Any]:
    distributed.zero_grad()
    before = state_dict_checksum(distributed.model.state_dict())
    distributed.set_context(step, f"measure:{step}:{label}:forward", training=False,
                            measurement=True, capture_kind=capture_kind)
    logits = distributed.forward(tokens)
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                           targets.to(distributed.device).reshape(-1))
    loss.backward()
    distributed.zero_grad()
    after = state_dict_checksum(distributed.model.state_dict())
    if before != after or distributed.proxy.last_capture is None:
        raise RemoteExecutionError("measurement changed model state or produced no capture")
    capture = dict(distributed.proxy.last_capture)
    capture.update({"label": label, "tokens": tokens.detach().cpu(),
                    "targets": targets.detach().cpu(), "loss": float(loss.detach())})
    return capture
