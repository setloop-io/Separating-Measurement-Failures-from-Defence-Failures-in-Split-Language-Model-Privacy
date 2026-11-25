"""Fail-closed Ed25519 key ceremony for confidential-compute engineering.

Only public keys and their content-addressed manifest belong in version control.
Private keys are created in a separately supplied directory, with mode 0600,
and are never copied into the public manifest or deployment bundle.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
from typing import Any, Mapping


CEREMONY_SCHEMA = "p4.cc_key_ceremony.v1"
ROLE_NAMES = (
    "campaign",
    "lifecycle-controller",
    "admission-verifier",
    "external-collector",
    "cc-on-journal",
    "cc-off-journal",
    "cc-on-heartbeat",
    "cc-off-heartbeat",
)
IDENTIFIER = re.compile(r"^[A-Za-z0-9_.:-]{3,128}$")
SHA256 = re.compile(r"^[a-f0-9]{64}$")


class KeyCeremonyError(ValueError):
    """Key ceremony inputs or artifacts are unsafe or inconsistent."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_exclusive(path: Path, value: Mapping[str, Any], mode: int) -> None:
    payload = json.dumps(
        value, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        mode,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def _run(command: list[str], label: str) -> None:
    try:
        result = subprocess.run(
            command, capture_output=True, check=False, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise KeyCeremonyError(f"{label} failed") from exc
    if result.returncode != 0:
        raise KeyCeremonyError(f"{label} failed")


def _require_regular(path: Path, label: str, *, private: bool) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise KeyCeremonyError(f"{label} is missing") from exc
    if (not stat.S_ISREG(metadata.st_mode) or path.is_symlink()
            or metadata.st_size < 1):
        raise KeyCeremonyError(f"{label} must be a nonempty regular file")
    permissions = stat.S_IMODE(metadata.st_mode)
    if private and permissions != 0o600:
        raise KeyCeremonyError(f"{label} must have mode 0600")
    if not private and permissions & 0o022:
        raise KeyCeremonyError(f"{label} must not be group/world writable")


def _require_ed25519_public(path: Path) -> None:
    _require_regular(path, "public key", private=False)
    try:
        result = subprocess.run(
            ["openssl", "pkey", "-pubin", "-in", str(path),
             "-text_pub", "-noout"],
            capture_output=True, check=False, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise KeyCeremonyError("cannot inspect public key") from exc
    first = result.stdout.splitlines()[0] if result.stdout else b""
    if result.returncode != 0 or first.strip() != b"ED25519 Public-Key:":
        raise KeyCeremonyError("public key is not Ed25519")


def validate_key_ceremony(manifest_path: str | Path) -> dict[str, Any]:
    """Validate every public key and reject private material in the manifest."""
    source = Path(manifest_path).resolve()
    _require_regular(source, "key ceremony manifest", private=False)
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise KeyCeremonyError("key ceremony manifest is invalid JSON") from exc
    expected = {
        "schema", "ceremony_id", "created_at", "algorithm", "roles",
        "private_material_committed", "claim_scope",
    }
    if (not isinstance(document, Mapping) or set(document) != expected
            or document.get("schema") != CEREMONY_SCHEMA
            or document.get("algorithm") != "Ed25519"
            or document.get("private_material_committed") is not False
            or document.get("claim_scope") != (
                "Public trust roots for one prospective engineering campaign; "
                "not execution authority or a security claim.")):
        raise KeyCeremonyError("key ceremony manifest fields differ")
    ceremony_id = document.get("ceremony_id")
    if not isinstance(ceremony_id, str) or IDENTIFIER.fullmatch(ceremony_id) is None:
        raise KeyCeremonyError("key ceremony ID is invalid")
    created_at = document.get("created_at")
    try:
        parsed = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise KeyCeremonyError("key ceremony time is invalid") from exc
    if not created_at.endswith("Z") or parsed.tzinfo is None:
        raise KeyCeremonyError("key ceremony time must be UTC")
    roles = document.get("roles")
    if not isinstance(roles, Mapping) or set(roles) != set(ROLE_NAMES):
        raise KeyCeremonyError("key ceremony role inventory differs")
    hashes: list[str] = []
    resolved: dict[str, str] = {}
    for role in ROLE_NAMES:
        binding = roles[role]
        if not isinstance(binding, Mapping) or set(binding) != {
                "public_key_path", "public_key_sha256", "public_key_bytes"}:
            raise KeyCeremonyError(f"{role} public-key binding fields differ")
        relative = binding.get("public_key_path")
        if (not isinstance(relative, str) or not relative
                or Path(relative).is_absolute() or ".." in Path(relative).parts):
            raise KeyCeremonyError(f"{role} public-key path is unsafe")
        path = (source.parent / relative).resolve()
        try:
            path.relative_to(source.parent)
        except ValueError as exc:
            raise KeyCeremonyError(f"{role} public-key path escapes ceremony") from exc
        _require_ed25519_public(path)
        digest = binding.get("public_key_sha256")
        if (not isinstance(digest, str) or SHA256.fullmatch(digest) is None
                or digest != _sha256(path)
                or binding.get("public_key_bytes") != path.stat().st_size):
            raise KeyCeremonyError(f"{role} public-key bytes differ")
        hashes.append(digest)
        resolved[role] = str(path)
    if len(set(hashes)) != len(ROLE_NAMES):
        raise KeyCeremonyError("every key ceremony role must use a distinct key")
    return {
        **dict(document),
        "manifest_sha256": _sha256(source),
        "resolved_public_keys": resolved,
    }


def generate_key_ceremony(
    *, public_dir: str | Path, private_dir: str | Path, ceremony_id: str,
    now: datetime | None = None,
) -> Path:
    """Create eight distinct Ed25519 keypairs without overwriting any bytes."""
    if not isinstance(ceremony_id, str) or IDENTIFIER.fullmatch(ceremony_id) is None:
        raise KeyCeremonyError("key ceremony ID is invalid")
    public_root = Path(public_dir).resolve()
    private_root = Path(private_dir).resolve()
    if public_root == private_root:
        raise KeyCeremonyError("public and private key directories must differ")
    if public_root.exists() or private_root.exists():
        raise KeyCeremonyError("key ceremony directories must not already exist")
    public_root.mkdir(parents=True, mode=0o755)
    private_root.mkdir(parents=True, mode=0o700)
    os.chmod(private_root, 0o700)
    roles: dict[str, dict[str, Any]] = {}
    private_inventory: dict[str, str] = {}
    try:
        for role in ROLE_NAMES:
            private_key = private_root / f"{role}.private.pem"
            public_key = public_root / f"{role}.public.pem"
            _run(
                ["openssl", "genpkey", "-algorithm", "ED25519", "-out",
                 str(private_key)],
                f"{role} private-key generation",
            )
            os.chmod(private_key, 0o600)
            _run(
                ["openssl", "pkey", "-in", str(private_key), "-pubout",
                 "-out", str(public_key)],
                f"{role} public-key derivation",
            )
            os.chmod(public_key, 0o644)
            _require_regular(private_key, f"{role} private key", private=True)
            _require_ed25519_public(public_key)
            roles[role] = {
                "public_key_path": public_key.name,
                "public_key_sha256": _sha256(public_key),
                "public_key_bytes": public_key.stat().st_size,
            }
            private_inventory[role] = private_key.name
        if len({item["public_key_sha256"] for item in roles.values()}) != len(ROLE_NAMES):
            raise KeyCeremonyError("generated key roles are not distinct")
        reference = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        created_at = reference.isoformat(timespec="microseconds").replace("+00:00", "Z")
        manifest = {
            "schema": CEREMONY_SCHEMA,
            "ceremony_id": ceremony_id,
            "created_at": created_at,
            "algorithm": "Ed25519",
            "roles": roles,
            "private_material_committed": False,
            "claim_scope": (
                "Public trust roots for one prospective engineering campaign; "
                "not execution authority or a security claim."),
        }
        manifest_path = public_root / "manifest.json"
        _write_json_exclusive(manifest_path, manifest, 0o644)
        private_manifest = {
            "schema": "p4.cc_private_key_inventory.v1",
            "ceremony_id": ceremony_id,
            "public_manifest_sha256": _sha256(manifest_path),
            "private_keys": private_inventory,
            "warning": "PRIVATE MATERIAL: do not commit, copy to guests, or place in argv.",
        }
        _write_json_exclusive(private_root / "inventory.json", private_manifest, 0o600)
        validate_key_ceremony(manifest_path)
        return manifest_path
    except BaseException:
        # Partial directories deliberately remain visible for incident review;
        # a retry must use a fresh ceremony ID/path rather than overwrite bytes.
        raise
