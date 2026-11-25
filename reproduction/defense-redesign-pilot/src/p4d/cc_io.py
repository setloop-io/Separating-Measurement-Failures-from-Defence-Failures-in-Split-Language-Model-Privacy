"""Race-resistant input snapshots and durable exclusive evidence writes."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any
import uuid


class SecureIOError(ValueError):
    """A file is unsafe, changed, malformed, or would be overwritten."""


@dataclass(frozen=True)
class FileSnapshot:
    path: Path
    data: bytes
    sha256: str
    device: int
    inode: int
    size: int
    mtime_ns: int


def read_snapshot(path: str | Path, *, max_bytes: int = 256 * 1024 * 1024,
                  expected_sha256: str | None = None) -> FileSnapshot:
    source = Path(path).resolve()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        raise SecureIOError(f"cannot open immutable input: {source}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not 1 <= before.st_size <= max_bytes:
            raise SecureIOError("immutable input must be a bounded nonempty regular file")
        chunks = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise SecureIOError("immutable input ended before its declared size")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise SecureIOError("immutable input grew while it was read")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    if identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise SecureIOError("immutable input changed while it was read")
    data = b"".join(chunks)
    digest = hashlib.sha256(data).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise SecureIOError("immutable input digest differs")
    return FileSnapshot(
        source, data, digest, before.st_dev, before.st_ino,
        before.st_size, before.st_mtime_ns)


def read_json_snapshot(path: str | Path, *, max_bytes: int = 32 * 1024 * 1024,
                       expected_sha256: str | None = None) -> tuple[dict[str, Any], FileSnapshot]:
    snapshot = read_snapshot(path, max_bytes=max_bytes, expected_sha256=expected_sha256)
    try:
        value = json.loads(snapshot.data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SecureIOError("immutable JSON input is malformed") from exc
    if not isinstance(value, dict):
        raise SecureIOError("immutable JSON input must be an object")
    return value, snapshot


def require_unchanged(snapshot: FileSnapshot) -> None:
    current = read_snapshot(
        snapshot.path, max_bytes=max(snapshot.size, 1),
        expected_sha256=snapshot.sha256)
    if (current.device, current.inode, current.size, current.mtime_ns) != (
            snapshot.device, snapshot.inode, snapshot.size, snapshot.mtime_ns):
        raise SecureIOError("immutable input path was replaced after validation")


def write_bytes_exclusive(path: str | Path, payload: bytes, *,
                          mode: int = 0o600) -> Path:
    """Write+fsync a new file and atomically link it without replacement."""
    target = Path(path)
    if not isinstance(payload, bytes) or not payload:
        raise SecureIOError("evidence payload must be nonempty bytes")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.{uuid.uuid4().hex}.pending"
    descriptor = None
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written < 1:
                raise OSError("short evidence write")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.link(temporary, target)
        directory = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        temporary.unlink()
        return target
    except FileExistsError as exc:
        raise SecureIOError(f"refusing to overwrite evidence: {target}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def write_json_exclusive(path: str | Path, value: Any) -> Path:
    payload = json.dumps(
        value, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    return write_bytes_exclusive(path, payload)
