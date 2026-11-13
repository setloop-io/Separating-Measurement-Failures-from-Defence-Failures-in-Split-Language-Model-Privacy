"""Exact runtime-environment and protected-store admission checks.

These checks do not prove a provider TEE on their own. They make the local
state byte-addressable so an independently authenticated admission decision can
bind it, and prevent a lockfile or mode-0700 temporary directory from being
silently treated as the executing environment or protected store.
"""
from __future__ import annotations

import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import stat
import sys
import fcntl
from typing import Any, Mapping

import torch


ENVIRONMENT_SCHEMA = "p4.cc_runtime_environment.v1"
STORE_SCHEMA = "p4.cc_trusted_store.v1"
SHA256 = re.compile(r"^[a-f0-9]{64}$")
ALLOWED_STORE_PROTECTION = {
    "dm-crypt",
    "provider-confidential-ephemeral",
    "tee-sealed-volume",
}


class RuntimeAttestationError(ValueError):
    """Runtime state differs from the independently bound contract."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    state = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            state.update(chunk)
    return state.hexdigest()


def _regular_file(path: Path, label: str) -> os.stat_result:
    try:
        info = path.lstat()
    except OSError as exc:
        raise RuntimeAttestationError(f"{label} is unavailable") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise RuntimeAttestationError(f"{label} must be a regular non-symlink file")
    return info


def validate_challenge_nonce(path: str | Path) -> str:
    """Require one raw 256-bit nontrivial nonce and return its digest."""
    source = Path(path)
    info = _regular_file(source, "challenge nonce")
    if info.st_size != 32:
        raise RuntimeAttestationError("challenge nonce must contain exactly 32 raw bytes")
    value = source.read_bytes()
    if len(set(value)) < 8 or value == bytes(32):
        raise RuntimeAttestationError("challenge nonce is trivial or lacks diversity")
    return _sha256_bytes(value)


def consume_challenge_nonce(*, nonce_path: str | Path, ledger_path: str | Path,
                            admission_id: str, session_id: str) -> str:
    """Atomically reject nonce replay within the protected campaign ledger."""
    nonce_sha = validate_challenge_nonce(nonce_path)
    if not admission_id or not session_id:
        raise RuntimeAttestationError("nonce ledger requires admission/session identities")
    ledger = Path(ledger_path)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) \
        | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(ledger, flags, 0o600)
    except OSError as exc:
        raise RuntimeAttestationError("nonce ledger cannot be opened safely") from exc
    try:
        info = os.fstat(descriptor)
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                or stat.S_IMODE(info.st_mode) != 0o600):
            raise RuntimeAttestationError(
                "nonce ledger must be one owner-only regular file")
        with os.fdopen(descriptor, "r+", encoding="utf-8", closefd=True) as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            rows = []
            previous = "0" * 64
            stream.seek(0)
            for number, line in enumerate(stream, 1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeAttestationError(
                        f"nonce ledger line {number} is invalid") from exc
                required = {
                    "sequence", "nonce_sha256", "admission_id", "session_id",
                    "previous_sha256", "entry_sha256",
                }
                if not isinstance(row, dict) or set(row) != required:
                    raise RuntimeAttestationError("nonce ledger row is malformed")
                if (isinstance(row["sequence"], bool)
                        or row["sequence"] != len(rows)
                        or not isinstance(row["nonce_sha256"], str)
                        or not SHA256.fullmatch(row["nonce_sha256"])
                        or not isinstance(row["admission_id"], str)
                        or not row["admission_id"]
                        or not isinstance(row["session_id"], str)
                        or not row["session_id"]
                        or row["previous_sha256"] != previous):
                    raise RuntimeAttestationError("nonce ledger chain is malformed")
                body = {key: value for key, value in row.items()
                        if key != "entry_sha256"}
                expected_entry = _sha256_bytes(_canonical_bytes(body))
                if row["entry_sha256"] != expected_entry:
                    raise RuntimeAttestationError("nonce ledger chain digest is invalid")
                rows.append(row)
                previous = row["entry_sha256"]
            if any(row.get("nonce_sha256") == nonce_sha for row in rows):
                raise RuntimeAttestationError("challenge nonce was already consumed")
            body = {
                "sequence": len(rows),
                "nonce_sha256": nonce_sha,
                "admission_id": admission_id,
                "session_id": session_id,
                "previous_sha256": previous,
            }
            body["entry_sha256"] = _sha256_bytes(_canonical_bytes(body))
            stream.seek(0, os.SEEK_END)
            stream.write(json.dumps(body, sort_keys=True, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        directory_fd = os.open(
            ledger.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        # fdopen owns the descriptor after construction; otherwise close it here.
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
    return nonce_sha


def _distribution_inventory() -> list[dict[str, str]]:
    rows: set[tuple[str, str]] = set()
    for distribution in metadata.distributions():
        name = distribution.metadata.get("Name") or distribution.name
        if not name or not distribution.version:
            raise RuntimeAttestationError("installed distribution lacks name/version")
        rows.add((str(name).lower().replace("_", "-"), str(distribution.version)))
    return [{"name": name, "version": version} for name, version in sorted(rows)]


def capture_runtime_environment(*, uv_lock_path: str | Path,
                                image_reference: str,
                                requested_cuda_device: str) -> dict[str, Any]:
    """Capture deterministic interpreter/package/Torch/CUDA identities."""
    lock = Path(uv_lock_path).resolve()
    executable = Path(sys.executable).resolve()
    _regular_file(lock, "environment lock")
    _regular_file(executable, "Python executable")
    if (not isinstance(image_reference, str) or len(image_reference.strip()) < 8
            or image_reference != image_reference.strip()
            or "latest" in image_reference.lower()):
        raise RuntimeAttestationError("runtime image reference must be exact and immutable")
    device = torch.device(requested_cuda_device)
    if device.type != "cuda" or device.index is None:
        raise RuntimeAttestationError("runtime contract requires an explicit cuda:N device")
    if not torch.cuda.is_available() or device.index >= torch.cuda.device_count():
        raise RuntimeAttestationError("requested CUDA device is unavailable")
    properties = torch.cuda.get_device_properties(device)
    packages = _distribution_inventory()
    body = {
        "schema": ENVIRONMENT_SCHEMA,
        "image_reference": image_reference,
        "environment_lock_sha256": _sha256_file(lock),
        "python": {
            "executable": str(executable),
            "executable_sha256": _sha256_file(executable),
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "abi": getattr(sys.implementation, "cache_tag", None),
        },
        "packages": packages,
        "packages_sha256": _sha256_bytes(_canonical_bytes(packages)),
        "torch": {
            "version": str(torch.__version__),
            "cuda_build": str(torch.version.cuda),
            "cudnn_version": torch.backends.cudnn.version(),
        },
        "cuda": {
            "device": str(device),
            "name": properties.name,
            "total_memory": int(properties.total_memory),
            "compute_capability": [int(properties.major), int(properties.minor)],
        },
    }
    body["environment_sha256"] = _sha256_bytes(_canonical_bytes(body))
    return body


def validate_runtime_environment(expected: Mapping[str, Any], *,
                                 uv_lock_path: str | Path,
                                 image_reference: str,
                                 requested_cuda_device: str) -> Mapping[str, Any]:
    required = {
        "schema", "image_reference", "environment_lock_sha256", "python",
        "packages", "packages_sha256", "torch", "cuda", "environment_sha256",
    }
    if (not isinstance(expected, Mapping) or set(expected) != required
            or expected.get("schema") != ENVIRONMENT_SCHEMA):
        raise RuntimeAttestationError("runtime environment manifest fields differ")
    body = {key: value for key, value in expected.items() if key != "environment_sha256"}
    if expected.get("environment_sha256") != _sha256_bytes(_canonical_bytes(body)):
        raise RuntimeAttestationError("runtime environment manifest digest mismatch")
    observed = capture_runtime_environment(
        uv_lock_path=uv_lock_path, image_reference=image_reference,
        requested_cuda_device=requested_cuda_device)
    if dict(expected) != observed:
        raise RuntimeAttestationError("executing environment differs from frozen manifest")
    return expected


def current_boot_id(path: str | Path = "/proc/sys/kernel/random/boot_id") -> str:
    source = Path(path)
    _regular_file(source, "boot identity")
    value = source.read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"[A-Fa-f0-9-]{16,64}", value):
        raise RuntimeAttestationError("boot identity is malformed")
    return value.lower()


def mount_identity(root: str | Path,
                   mountinfo_path: str | Path = "/proc/self/mountinfo") -> dict[str, Any]:
    """Return the exact Linux mount that contains root."""
    root = Path(root).resolve()
    source = Path(mountinfo_path)
    _regular_file(source, "mountinfo")
    candidates = []
    for line in source.read_text(encoding="utf-8").splitlines():
        left, separator, right = line.partition(" - ")
        if not separator:
            raise RuntimeAttestationError("mountinfo row is malformed")
        fields = left.split()
        trailing = right.split()
        if len(fields) < 6 or len(trailing) < 3:
            raise RuntimeAttestationError("mountinfo row is incomplete")
        mount_point = Path(fields[4].replace("\\040", " ")).resolve()
        try:
            root.relative_to(mount_point)
        except ValueError:
            continue
        candidates.append((len(mount_point.parts), {
            "mount_point": str(mount_point),
            "major_minor": fields[2],
            "mount_options": sorted(fields[5].split(",")),
            "filesystem_type": trailing[0],
            "mount_source": trailing[1],
            "super_options": sorted(trailing[2].split(",")),
        }))
    if not candidates:
        raise RuntimeAttestationError("trusted-store mount is absent from mountinfo")
    return max(candidates, key=lambda item: item[0])[1]


def capture_trusted_store(*, root: str | Path, instance_id: str,
                          protection_type: str,
                          protection_evidence_path: str | Path,
                          boot_id_path: str | Path =
                          "/proc/sys/kernel/random/boot_id",
                          mountinfo_path: str | Path =
                          "/proc/self/mountinfo") -> dict[str, Any]:
    """Capture exact protected-store state for independent admission.

    Hashing protection evidence does not establish its truth. The independent
    verifier must authenticate those bytes before it signs the admission.
    """
    store = Path(root).resolve()
    try:
        info = store.lstat()
    except OSError as exc:
        raise RuntimeAttestationError("trusted-store root is unavailable") from exc
    if (stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode)
            or stat.S_IMODE(info.st_mode) != 0o700 or info.st_uid != os.getuid()):
        raise RuntimeAttestationError(
            "trusted-store root must be an owned mode-0700 directory")
    if (not isinstance(instance_id, str) or not instance_id.strip()
            or instance_id != instance_id.strip() or len(instance_id) > 256):
        raise RuntimeAttestationError("trusted-store instance identity is invalid")
    if protection_type not in ALLOWED_STORE_PROTECTION:
        raise RuntimeAttestationError("trusted-store protection type is not admitted")
    evidence = Path(protection_evidence_path).resolve()
    evidence_info = _regular_file(evidence, "trusted-store protection evidence")
    if evidence_info.st_size < 1:
        raise RuntimeAttestationError("trusted-store protection evidence is empty")
    mount = mount_identity(store, mountinfo_path)
    body = {
        "schema": STORE_SCHEMA,
        "root": str(store),
        "owner_uid": os.getuid(),
        "mode": "0700",
        "device_id": info.st_dev,
        "instance_id": instance_id,
        "boot_id": current_boot_id(boot_id_path),
        "protection_type": protection_type,
        "protection_evidence_sha256": _sha256_file(evidence),
        "mount": mount,
        "mount_sha256": _sha256_bytes(_canonical_bytes(mount)),
    }
    body["manifest_sha256"] = _sha256_bytes(_canonical_bytes(body))
    return body


def validate_trusted_store(manifest: Mapping[str, Any], *, root: str | Path,
                           instance_id: str, boot_id_path: str | Path =
                           "/proc/sys/kernel/random/boot_id",
                           mountinfo_path: str | Path = "/proc/self/mountinfo") \
        -> Mapping[str, Any]:
    required = {
        "schema", "root", "owner_uid", "mode", "device_id", "instance_id",
        "boot_id", "protection_type", "protection_evidence_sha256",
        "mount", "mount_sha256", "manifest_sha256",
    }
    if (not isinstance(manifest, Mapping) or set(manifest) != required
            or manifest.get("schema") != STORE_SCHEMA):
        raise RuntimeAttestationError("trusted-store manifest fields differ")
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if manifest.get("manifest_sha256") != _sha256_bytes(_canonical_bytes(body)):
        raise RuntimeAttestationError("trusted-store manifest digest mismatch")
    root = Path(root).resolve()
    try:
        info = root.lstat()
    except OSError as exc:
        raise RuntimeAttestationError("trusted-store root is unavailable") from exc
    if (stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode)
            or stat.S_IMODE(info.st_mode) != 0o700):
        raise RuntimeAttestationError("trusted-store root must be a mode-0700 directory")
    if info.st_uid != os.getuid() or manifest.get("owner_uid") != os.getuid():
        raise RuntimeAttestationError("trusted-store owner differs from the workload user")
    if (manifest.get("root") != str(root) or manifest.get("mode") != "0700"
            or manifest.get("device_id") != info.st_dev):
        raise RuntimeAttestationError("trusted-store path/mode/device binding differs")
    if manifest.get("instance_id") != instance_id:
        raise RuntimeAttestationError("trusted-store instance binding differs")
    if manifest.get("boot_id") != current_boot_id(boot_id_path):
        raise RuntimeAttestationError("trusted-store boot binding differs")
    if manifest.get("protection_type") not in ALLOWED_STORE_PROTECTION:
        raise RuntimeAttestationError("trusted-store protection type is not admitted")
    if not SHA256.fullmatch(str(manifest.get("protection_evidence_sha256"))):
        raise RuntimeAttestationError("trusted-store protection evidence is not bound")
    observed_mount = mount_identity(root, mountinfo_path)
    if (manifest.get("mount") != observed_mount
            or manifest.get("mount_sha256") != _sha256_bytes(
                _canonical_bytes(observed_mount))):
        raise RuntimeAttestationError("trusted-store mount identity differs")
    return manifest
