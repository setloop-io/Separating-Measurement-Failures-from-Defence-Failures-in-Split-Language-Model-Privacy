"""Plaintext loopback placement diagnostics. This module implements neither TEE nor FHE."""
from dataclasses import dataclass, field
import hashlib
import json
import struct

import numpy as np
import torch
from torch import nn

POLICIES = ("local_reference", "delegate_all_phases", "local_prefill_delegate_decode")
LINEARS = ("qkv", "proj", "mlp_fc", "mlp_proj")


def pack_tensor(tensor):
    if tensor.dtype != torch.float32 or tensor.device.type != "cpu" or not torch.isfinite(tensor).all():
        raise ValueError("Diagnostic transport accepts only finite CPU float32")
    header = json.dumps({"shape": list(tensor.shape), "dtype": "float32-le"}, sort_keys=True).encode()
    values = tensor.detach().contiguous().numpy().astype("<f4", copy=False).tobytes()
    return struct.pack("!I", len(header)) + header + values


def unpack_tensor(packet):
    if len(packet) < 4:
        raise ValueError("Truncated tensor packet")
    size = struct.unpack("!I", packet[:4])[0]
    if size > len(packet) - 4:
        raise ValueError("Truncated tensor header")
    try:
        header = json.loads(packet[4:4 + size])
        shape = header["shape"]
        if (header["dtype"] != "float32-le" or not isinstance(shape, list) or not shape
                or any(type(d) is not int or d <= 0 for d in shape)):
            raise ValueError("Invalid tensor format")
        count = 1
        for d in shape:
            count *= d
        payload = packet[4 + size:]
        if len(payload) != count * 4:
            raise ValueError("Tensor shape and payload disagree")
        out = torch.from_numpy(np.frombuffer(payload, dtype="<f4").copy().reshape(shape))
        if not torch.isfinite(out).all():
            raise ValueError("Nonfinite tensor payload")
        return out
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid tensor header") from exc


@dataclass
class PlaintextLoopback:
    """Public fp32 weights; real serialization; no sockets, GPU, encryption, or enclave."""
    weights: dict = field(default_factory=dict)
    setup_bytes: int = 0

    def provision(self, name, linear):
        if name in self.weights:
            raise ValueError("Operator provisioned twice")
        w = pack_tensor(linear.weight)
        b = None if linear.bias is None else pack_tensor(linear.bias)
        self.setup_bytes += len(w) + (0 if b is None else len(b))
        self.weights[name] = (unpack_tensor(w), None if b is None else unpack_tensor(b))

    def evaluate(self, name, request):
        if name not in self.weights:
            raise ValueError("Unknown delegated operator")
        x = unpack_tensor(request)
        weight, bias = self.weights[name]
        if x.shape[-1] != weight.shape[1]:
            raise ValueError("Projection input width mismatch")
        return pack_tensor(nn.functional.linear(x, weight, bias))


@dataclass
class Router:
    policy: str
    backend: PlaintextLoopback = field(default_factory=PlaintextLoopback)
    phase: str = "prefill"
    step: int = 0
    events: list = field(default_factory=list)

    def __post_init__(self):
        if self.policy not in POLICIES:
            raise ValueError("Unknown placement policy")

    def delegated(self):
        if self.phase not in ("prefill", "decode"):
            raise ValueError("Unknown serving phase")
        return self.policy == "delegate_all_phases" or (
            self.policy == "local_prefill_delegate_decode" and self.phase == "decode")

    def linear(self, name, module, x):
        if torch.is_grad_enabled():
            raise RuntimeError("Inference-only diagnostic; no backward transport exists")
        delegated = self.delegated()
        record = {"operator": name, "phase": self.phase, "step": self.step,
                  "delegated": delegated, "input_shape": list(x.shape)}
        if delegated:
            if name not in self.backend.weights:
                self.backend.provision(name, module)
            request = pack_tensor(x)
            response = self.backend.evaluate(name, request)
            y = unpack_tensor(response)
            if y.shape != (*x.shape[:-1], module.out_features):
                raise ValueError("Delegated output shape mismatch")
            record.update(request_bytes=len(request), response_bytes=len(response),
                          request_sha256=hashlib.sha256(request).hexdigest(),
                          response_sha256=hashlib.sha256(response).hexdigest())
        else:
            y = module(x)
            record.update(request_bytes=0, response_bytes=0)
        self.events.append(record)
        return y


class RoutedLinear(nn.Module):
    def __init__(self, linear, router, name):
        super().__init__()
        self.linear, self.router, self.name = linear, router, name

    def forward(self, x):
        return self.router.linear(self.name, self.linear, x)


def install_router(model, router):
    for i, block in enumerate(model.blocks):
        for key in LINEARS:
            linear = getattr(block, key)
            if not isinstance(linear, nn.Linear):
                raise ValueError("Expected an unwrapped reference backbone")
            setattr(block, key, RoutedLinear(linear, router, f"blocks.{i}.{key}"))
    return model


def generate(model, prompt, new_tokens, router=None):
    """Full-prefix recomputation; explicitly no KV cache or Bifrost serving replication."""
    if new_tokens < 1 or prompt.shape[1] + new_tokens - 1 > model.cfg.max_seq_len:
        raise ValueError("Invalid generation budget")
    sequence, outputs = prompt.clone(), []
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for step in range(new_tokens):
                if router is not None:
                    router.phase = "prefill" if step == 0 else "decode"
                    router.step = step
                logits = model(sequence)[:, -1, :].cpu()
                outputs.append(logits)
                sequence = torch.cat([sequence, logits.argmax(-1, keepdim=True)], dim=1)
    finally:
        model.train(was_training)
    return sequence[:, prompt.shape[1]:], torch.stack(outputs, dim=1)
