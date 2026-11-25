"""Authenticated framed tensor transport for R1 split execution."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
import socket
import struct
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping

import numpy as np
import torch

MAGIC = b"PCR1"
PREFIX = struct.Struct("!4sI")
QUANT_META = struct.Struct("<fi")
MAX_HEADER_BYTES = 1 << 20
MAX_PAYLOAD_BYTES = 512 << 20
MESSAGE_TYPES = frozenset({"HELLO", "FORWARD", "BACKWARD", "STEP", "CHECKPOINT", "CLOSE", "ERROR"})


class WireError(RuntimeError):
    """A frame or protocol value is invalid."""


@dataclass(frozen=True)
class TensorPacket:
    metadata: dict[str, Any]
    payload: bytes


@dataclass(frozen=True)
class WireFrame:
    header: dict[str, Any]
    tensor: torch.Tensor | None
    wire_bytes: int


@dataclass(frozen=True)
class WireRecord:
    direction: str
    message_type: str
    session_id: str
    request_id: str
    step: int
    sequence: int
    header_bytes: int
    payload_bytes: int
    wire_bytes: int
    payload_sha256: str


class WireAuditLog:
    def __init__(self) -> None:
        self.records: list[WireRecord] = []

    def append(self, record: WireRecord) -> None:
        self.records.append(record)

    @property
    def total_bytes(self) -> int:
        return sum(record.wire_bytes for record in self.records)

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": "r1-wire-audit-1.0", "total_bytes": self.total_bytes,
                "records": [asdict(record) for record in self.records]}


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _shape_numel(shape: list[int]) -> int:
    if not shape or any(type(value) is not int or value <= 0 for value in shape):
        raise WireError(f"invalid tensor shape: {shape!r}")
    return math.prod(shape)


def _pack_codes(codes: np.ndarray, q_bits: int) -> bytes:
    values = codes.astype(np.uint8, copy=False).reshape(-1)
    if q_bits == 8:
        return values.tobytes()
    if q_bits == 16:
        return codes.astype("<u2", copy=False).reshape(-1).tobytes()
    group = 8 // q_bits
    padded = np.zeros(math.ceil(len(values) / group) * group, dtype=np.uint8)
    padded[: len(values)] = values
    matrix = padded.reshape(-1, group)
    packed = np.zeros(len(matrix), dtype=np.uint8)
    for index in range(group):
        packed |= matrix[:, index] << (index * q_bits)
    return packed.tobytes()


def _unpack_codes(payload: bytes, q_bits: int, count: int) -> np.ndarray:
    if q_bits == 8:
        values = np.frombuffer(payload, dtype=np.uint8, count=count)
    elif q_bits == 16:
        values = np.frombuffer(payload, dtype="<u2", count=count)
    else:
        packed = np.frombuffer(payload, dtype=np.uint8)
        group = 8 // q_bits
        mask = (1 << q_bits) - 1
        values = np.empty(len(packed) * group, dtype=np.uint8)
        for index in range(group):
            values[index::group] = (packed >> (index * q_bits)) & mask
        values = values[:count]
    if len(values) != count:
        raise WireError("quantized payload length does not match tensor shape")
    return values.copy()


def encode_tensor(tensor: torch.Tensor, q_bits: int | None = None) -> TensorPacket:
    value = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if not torch.isfinite(value).all():
        raise WireError("cannot encode a tensor containing non-finite values")
    shape = list(value.shape)
    numel = _shape_numel(shape)
    if q_bits is None or q_bits == 32:
        payload = value.numpy().astype("<f4", copy=False).tobytes()
        metadata = {"encoding": "raw-f32-le", "dtype": "float32", "shape": shape,
                    "numel": numel, "q_bits": 32}
        return TensorPacket(metadata, payload)
    if q_bits not in (2, 4, 8, 16):
        raise WireError(f"unsupported q_bits: {q_bits}")
    minimum, maximum = value.amin(), value.amax()
    qmax = float((1 << q_bits) - 1)
    scale = (maximum - minimum) / qmax
    constant_scale = torch.where(minimum.abs() > 0, minimum.abs() / qmax,
                                 torch.ones_like(minimum))
    scale = torch.where(scale > 0, scale, constant_scale)
    constant_zero = torch.where(minimum < 0, torch.full_like(minimum, qmax),
                                torch.zeros_like(minimum))
    zero_point = torch.where(maximum == minimum, constant_zero,
                             torch.round(-minimum / scale).clamp(0, qmax))
    codes = torch.round(value / scale + zero_point).clamp(0, qmax).to(torch.int64).numpy()
    payload = QUANT_META.pack(float(scale), int(zero_point)) + _pack_codes(codes, q_bits)
    metadata = {"encoding": "affine-quantized-le", "dtype": "float32", "shape": shape,
                "numel": numel, "q_bits": q_bits, "metadata_bytes": QUANT_META.size}
    return TensorPacket(metadata, payload)


def decode_tensor(metadata: Mapping[str, Any], payload: bytes) -> torch.Tensor:
    shape = metadata.get("shape")
    if not isinstance(shape, list):
        raise WireError("tensor metadata has no shape")
    numel = _shape_numel(shape)
    if metadata.get("numel") != numel or metadata.get("dtype") != "float32":
        raise WireError("tensor metadata is inconsistent")
    encoding = metadata.get("encoding")
    if encoding == "raw-f32-le":
        if (set(metadata) != {"encoding", "dtype", "shape", "numel", "q_bits"} or
                metadata.get("q_bits") != 32):
            raise WireError("raw tensor metadata is inconsistent")
        expected = numel * 4
        if len(payload) != expected:
            raise WireError(f"raw tensor payload has {len(payload)} bytes, expected {expected}")
        array = np.frombuffer(payload, dtype="<f4", count=numel).copy()
    elif encoding == "affine-quantized-le":
        q_bits = metadata.get("q_bits")
        if (set(metadata) != {"encoding", "dtype", "shape", "numel", "q_bits", "metadata_bytes"} or
                q_bits not in (2, 4, 8, 16) or
                metadata.get("metadata_bytes") != QUANT_META.size or
                len(payload) < QUANT_META.size):
            raise WireError("invalid quantized tensor metadata")
        scale, zero_point = QUANT_META.unpack(payload[: QUANT_META.size])
        expected = math.ceil(numel * q_bits / 8) + QUANT_META.size
        qmax = (1 << q_bits) - 1
        if (len(payload) != expected or not math.isfinite(scale) or scale <= 0 or
                not 0 <= zero_point <= qmax):
            raise WireError("invalid quantized tensor payload")
        codes = _unpack_codes(payload[QUANT_META.size :], q_bits, numel)
        array = (codes.astype(np.float32) - zero_point) * np.float32(scale)
    else:
        raise WireError(f"unsupported tensor encoding: {encoding!r}")
    tensor = torch.from_numpy(array.reshape(shape))
    if not torch.isfinite(tensor).all():
        raise WireError("decoded tensor contains non-finite values")
    return tensor


def _validate_identity(value: Any, field: str) -> str:
    if (not isinstance(value, str) or not value or len(value) > 256 or
            re.fullmatch(r"[A-Za-z0-9_.:-]+", value) is None):
        raise WireError(f"invalid {field}")
    return value


def _validated_header(header: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(header)
    expected = {"schema_version", "message_type", "session_id", "request_id", "step",
                "sequence", "body", "tensor", "payload_bytes", "payload_sha256", "auth"}
    if set(value) != expected:
        raise WireError("wire header fields differ from the protocol")
    if value.get("schema_version") != "r1-wire-1.0":
        raise WireError("unsupported wire schema")
    if value.get("message_type") not in MESSAGE_TYPES:
        raise WireError("unsupported message type")
    for field in ("session_id", "request_id"):
        _validate_identity(value.get(field), field)
    for field in ("step", "sequence", "payload_bytes"):
        if type(value.get(field)) is not int or value[field] < 0:
            raise WireError(f"invalid {field}")
    if value["payload_bytes"] > MAX_PAYLOAD_BYTES:
        raise WireError("payload exceeds protocol limit")
    digest = value.get("payload_sha256")
    if not isinstance(digest, str) or re.fullmatch(r"[a-f0-9]{64}", digest) is None:
        raise WireError("invalid payload digest")
    auth = value.get("auth")
    if not isinstance(auth, str) or re.fullmatch(r"[a-f0-9]{64}", auth) is None:
        raise WireError("invalid frame authentication tag")
    body = value.get("body", {})
    if not isinstance(body, dict):
        raise WireError("frame body must be an object")
    tensor = value.get("tensor")
    if tensor is not None and not isinstance(tensor, dict):
        raise WireError("tensor metadata must be an object")
    if (tensor is None) != (value["payload_bytes"] == 0):
        raise WireError("tensor metadata and payload presence differ")
    return value


def _auth_tag(header: Mapping[str, Any], payload: bytes, secret: bytes) -> str:
    unsigned = dict(header)
    unsigned.pop("auth", None)
    return hmac.new(secret, _canonical(unsigned) + payload, hashlib.sha256).hexdigest()


def build_header(message_type: str, session_id: str, request_id: str, step: int,
                 sequence: int, body: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {"schema_version": "r1-wire-1.0", "message_type": message_type,
            "session_id": session_id, "request_id": request_id, "step": step,
            "sequence": sequence, "body": dict(body or {})}


def _record(header: Mapping[str, Any], direction: str, header_bytes: int,
            payload_bytes: int) -> WireRecord:
    return WireRecord(direction=direction, message_type=str(header["message_type"]),
                      session_id=str(header["session_id"]), request_id=str(header["request_id"]),
                      step=int(header["step"]), sequence=int(header["sequence"]),
                      header_bytes=header_bytes, payload_bytes=payload_bytes,
                      wire_bytes=PREFIX.size + header_bytes + payload_bytes,
                      payload_sha256=str(header["payload_sha256"]))


def send_frame(sock: socket.socket, header: Mapping[str, Any], secret: bytes,
               tensor: TensorPacket | None = None,
               recorder: Callable[[WireRecord], None] | None = None) -> int:
    if len(secret) < 32:
        raise WireError("wire secret must contain at least 32 bytes")
    payload = b"" if tensor is None else tensor.payload
    value = dict(header)
    value["tensor"] = None if tensor is None else tensor.metadata
    value["payload_bytes"] = len(payload)
    value["payload_sha256"] = _sha256(payload)
    value["auth"] = _auth_tag(value, payload, secret)
    value = _validated_header(value)
    encoded = _canonical(value)
    if len(encoded) > MAX_HEADER_BYTES:
        raise WireError("frame header exceeds protocol limit")
    sock.sendall(PREFIX.pack(MAGIC, len(encoded)) + encoded + payload)
    record = _record(value, "sent", len(encoded), len(payload))
    if recorder is not None:
        recorder(record)
    return record.wire_bytes


def _recv_exact(sock: socket.socket, count: int) -> bytes:
    chunks: list[bytes] = []
    remaining = count
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise WireError("connection closed during frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_frame(sock: socket.socket, secret: bytes,
               recorder: Callable[[WireRecord], None] | None = None) -> WireFrame:
    if len(secret) < 32:
        raise WireError("wire secret must contain at least 32 bytes")
    magic, header_size = PREFIX.unpack(_recv_exact(sock, PREFIX.size))
    if magic != MAGIC or not 0 < header_size <= MAX_HEADER_BYTES:
        raise WireError("invalid frame prefix")
    encoded = _recv_exact(sock, header_size)
    try:
        header = _validated_header(json.loads(encoded))
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise WireError(f"invalid frame header: {exc}") from exc
    payload = _recv_exact(sock, header["payload_bytes"])
    if not hmac.compare_digest(header.get("auth", ""), _auth_tag(header, payload, secret)):
        raise WireError("frame authentication failed")
    if not hmac.compare_digest(header["payload_sha256"], _sha256(payload)):
        raise WireError("frame payload digest mismatch")
    tensor = None if header["tensor"] is None else decode_tensor(header["tensor"], payload)
    record = _record(header, "received", header_size, len(payload))
    if recorder is not None:
        recorder(record)
    return WireFrame(header=header, tensor=tensor, wire_bytes=record.wire_bytes)
