#!/usr/bin/env python3
"""Fail-closed D-only training server for latent-native v5."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import struct
import sys
import time
import traceback
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import websockets

from privacy_runtime.latent_native import (assert_koula_latent_only,
                                           build_koula_latent_middle)


def pack_frame(header, payload=b""):
    encoded = json.dumps(header, separators=(",", ":")).encode()
    return struct.pack(">I", len(encoded)) + encoded + payload


def unpack_frame(message):
    size = struct.unpack(">I", message[:4])[0]
    return json.loads(message[4:4 + size]), message[4 + size:]


def tensor_from(payload, shape, device):
    expected = 4
    for dimension in shape:
        expected *= dimension
    if len(payload) != expected:
        raise ValueError("payload length does not match declared shape")
    return torch.frombuffer(bytearray(payload), dtype=torch.float32).reshape(
        shape).clone().to(device)


def tensor_bytes(value):
    return value.detach().float().contiguous().cpu().numpy().tobytes()


def validate_latent_shape(shape, latent_dim):
    """Validate headers before payload allocation or tensor construction."""
    if (not isinstance(shape, list) or len(shape) != 3
            or any(not isinstance(value, int) or isinstance(value, bool)
                   or value <= 0 for value in shape)
            or shape[-1] != latent_dim):
        raise ValueError("non-D latent frame rejected before allocation")
    return tuple(shape)


class LatentServer:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.sessions = {}
        self.max_seen_width = 0

    def new_session_model(self, seed):
        """Build an independently seeded cloud model for one connection."""
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = build_koula_latent_middle(
                self.args.latent_dim, self.args.cloud_layers,
                self.args.cloud_heads, self.args.cloud_kind,
                getattr(self.args, "cloud_experts", 1))
        model = model.to(self.device).train()
        assert_koula_latent_only(model, self.args.latent_dim,
                                 self.args.forbidden_hidden_dim)
        return model

    def capture(self, session_id, mb_id, phase, tensor):
        if not self.args.capture_dir:
            return
        root = Path(self.args.capture_dir)
        root.mkdir(parents=True, exist_ok=True)
        index = len(list(root.glob("wire_*.pt")))
        stem = root / f"wire_{index:06d}"
        torch.save(tensor.detach().float().cpu(), stem.with_suffix(".pt"))
        stem.with_suffix(".json").write_text(json.dumps({
            "session_id": session_id, "mb_id": mb_id, "phase": phase,
            "step": mb_id, "epoch": mb_id,
        }, sort_keys=True) + "\n")

    async def handler(self, websocket):
        session_id = str(uuid.uuid4())
        session = None
        try:
            hello = json.loads(await websocket.recv())
            if (hello.get("op") != "hello"
                    or hello.get("protocol") != "latent-native-v5"):
                raise ValueError("latent-native-v5 hello required")
            if hello.get("latent_dim") != self.args.latent_dim:
                raise ValueError("requested latent width does not match server")
            if hello.get("wire_dtype") != "fp32":
                raise ValueError("only explicit fp32 latent wire is supported")
            if hello.get("cloud_kind", "transformer") != self.args.cloud_kind:
                raise ValueError("requested cloud kind does not match server")
            if int(hello.get("cloud_experts", 1)) != self.args.cloud_experts:
                raise ValueError("requested expert count does not match server")
            if int(hello.get("cloud_layers", self.args.cloud_layers)
                   ) != self.args.cloud_layers:
                raise ValueError("requested layer count does not match server")
            cloud_seed = hello.get("cloud_seed")
            if (not isinstance(cloud_seed, int)
                    or isinstance(cloud_seed, bool) or cloud_seed < 0):
                raise ValueError("non-negative integer cloud_seed required")
            model = self.new_session_model(cloud_seed)
            session = {
                "pending": {},
                "model": model,
                "optimizer": torch.optim.AdamW(
                    model.parameters(), lr=float(hello.get("lr", 3e-4))),
                "active_delta": float(hello.get("active_delta", 0.0)),
                "tampered": 0,
            }
            self.sessions[session_id] = session
            shapes = {name: list(value.shape)
                      for name, value in model.state_dict().items()}
            digest = hashlib.sha256(json.dumps(
                shapes, sort_keys=True).encode()).hexdigest()
            await websocket.send(json.dumps({
                "op": "hello_ack", "session_id": session_id,
                "latent_dim": self.args.latent_dim,
                "cloud_kind": self.args.cloud_kind,
                "cloud_experts": self.args.cloud_experts,
                "cloud_layers": self.args.cloud_layers,
                "cloud_seed": cloud_seed,
                "latent_only_audit": True,
                "max_parameter_dimension": max(
                    max(value.shape or (1,))
                    for value in model.state_dict().values()),
                "state_shape_digest": digest,
            }))
            async for message in websocket:
                if isinstance(message, str):
                    control = json.loads(message)
                    if control.get("op") == "optimizer_step":
                        session["optimizer"].step()
                        session["optimizer"].zero_grad(set_to_none=True)
                        await websocket.send(json.dumps({
                            "op": "step_ack", "pending": len(session["pending"]),
                            "tampered": session["tampered"]}))
                    elif control.get("op") == "close":
                        break
                    else:
                        raise ValueError("unknown latent control operation")
                    continue
                header, payload = unpack_frame(message)
                shape = header.get("shape")
                validate_latent_shape(shape, self.args.latent_dim)
                self.max_seen_width = max(self.max_seen_width, shape[-1])
                value = tensor_from(payload, shape, self.device)
                mb_id = int(header["mb_id"])
                if header.get("op") == "forward":
                    self.capture(session_id, mb_id, "fwd", value)
                    training = bool(header.get("training"))
                    leaf = value.detach().requires_grad_(training)
                    started = time.perf_counter()
                    output = session["model"](leaf)
                    tampered = False
                    if session["active_delta"]:
                        direction = torch.sign(output.detach()).clamp(min=-1, max=1)
                        output = output + session["active_delta"] * direction
                        session["tampered"] += 1
                        tampered = True
                    if self.args.malicious_delta:
                        # True active adversary: perturbs returns and does NOT
                        # report it (unlike active_delta, which is declared in
                        # the reply metadata).  For the Byzantine-verify test.
                        direction = torch.sign(output.detach()).clamp(min=-1, max=1)
                        output = output + self.args.malicious_delta * direction
                    if training:
                        session["pending"][mb_id] = (leaf, output)
                    self.capture(session_id, mb_id, "return", output)
                    await websocket.send(pack_frame({
                        "op": "forward_result", "mb_id": mb_id,
                        "shape": list(output.shape), "tampered": tampered,
                        "cloud_ms": 1000 * (time.perf_counter() - started),
                    }, tensor_bytes(output)))
                elif header.get("op") == "backward":
                    if mb_id not in session["pending"]:
                        raise ValueError("backward has no pending forward")
                    leaf, output = session["pending"].pop(mb_id)
                    self.capture(session_id, mb_id, "bwd", value)
                    torch.autograd.backward(output, value)
                    await websocket.send(pack_frame({
                        "op": "backward_result", "mb_id": mb_id,
                        "shape": list(leaf.grad.shape),
                    }, tensor_bytes(leaf.grad)))
                else:
                    raise ValueError("unknown latent frame operation")
        except Exception as exc:
            traceback.print_exc()
            try:
                await websocket.send(pack_frame({"op": "error",
                                                 "error": str(exc)}))
            except Exception:
                pass
        finally:
            self.sessions.pop(session_id, None)
            print(json.dumps({"session": session_id, "event": "closed",
                              "max_seen_width": self.max_seen_width}))


async def serve(args):
    server = LatentServer(args)
    tls_context = None
    if args.tls_cert or args.tls_key:
        import ssl
        if not (args.tls_cert and args.tls_key):
            raise ValueError("--tls-cert and --tls-key are required together")
        tls_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        tls_context.minimum_version = ssl.TLSVersion.TLSv1_3
        tls_context.load_cert_chain(args.tls_cert, args.tls_key)
    async with websockets.serve(server.handler, "0.0.0.0", args.port,
                                max_size=64 * 1024 * 1024, ssl=tls_context,
                                ping_interval=None):
        print(json.dumps({"status": "ready", "port": args.port,
                          "tls": tls_context is not None,
                          "latent_dim": args.latent_dim,
                          "forbidden_hidden_dim": args.forbidden_hidden_dim}))
        await asyncio.Future()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--forbidden-hidden-dim", type=int, default=1024)
    parser.add_argument("--cloud-layers", type=int, default=2)
    parser.add_argument("--cloud-heads", type=int, default=4)
    parser.add_argument("--cloud-kind",
                        choices=("transformer", "equivariant", "monomial",
                                 "monomial_moe", "monomial_moe_radial",
                                 "invariant_mlp"),
                        default="transformer")
    parser.add_argument("--cloud-experts", type=int, default=1,
                        help="expert count for the monomial_moe cloud")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available()
                        else "cpu")
    parser.add_argument("--port", type=int, default=5013)
    parser.add_argument("--tls-cert", help="PEM certificate for wss serving")
    parser.add_argument("--tls-key", help="PEM private key for wss serving")
    parser.add_argument("--capture-dir")
    parser.add_argument("--malicious-delta", type=float, default=0.0,
                        help="true active adversary: perturb returns without "
                             "reporting it (Byzantine-verify test only)")
    args = parser.parse_args()
    if args.device == "cuda":
        # Force CUDA context init before serving: the first connection's TLS
        # handshake timed out against a cold event loop three times (12B,
        # 35B s43, E3-d64 cells) before this warmup existed.
        torch.zeros(1, device="cuda")
        torch.cuda.synchronize()
    asyncio.run(serve(args))


if __name__ == "__main__":
    main()
