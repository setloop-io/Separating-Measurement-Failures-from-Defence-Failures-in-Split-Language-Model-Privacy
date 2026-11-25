"""Signed, provider-neutral pre-active-security lifecycle bootstrap.

``L0`` is an external-control-plane contract made after CC-on admission and
before active security.  It binds every byte needed to launch active security
and to tear the resource down, but deliberately contains no active-security
manifest (``M``).  A later signed ``L1`` lifecycle launch envelope supplies
``M``.  The state helpers below make that ordering explicit: L0 success permits
active security, while measurement remains forbidden until the same controller
binds a valid L1.  Every failure state requires teardown.

This module has no provider SDK, networking, provisioning, or command-execution
path.  Its only subprocesses are local OpenSSL signature/key checks.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import tempfile
from typing import Any, Mapping

from .cc_active_security import (
    ActiveSecurityError,
    require_active_security_manifest,
    validate_active_runtime_binding,
    validate_deployment_manifest,
)
from .cc_admission import validate_admission_record, validate_arrival_record
from .cc_campaign import CampaignSealError, validate_pair_declaration
from .cc_io import (
    FileSnapshot,
    SecureIOError,
    read_json_snapshot,
    read_snapshot,
    require_unchanged,
    write_json_exclusive,
)
from .cc_lifecycle import (
    CONTROL_SPECS,
    LEDGER_SCHEMA,
    LifecycleError,
    validate_control_artifact,
    verify_lifecycle_launch_envelope,
)
from .hardware_preorder import HardwarePreorderError, ROOT, validate_packet


L0_ENVELOPE_SCHEMA = "p4.cc_bootstrap_lifecycle_envelope.v1"
L0_PAYLOAD_SCHEMA = "p4.cc_bootstrap_lifecycle_payload.v1"
L0_SIGNATURE_DOMAIN = "p4.cc_bootstrap_lifecycle.l0.v1"
STATE_ENVELOPE_SCHEMA = "p4.cc_bootstrap_lifecycle_state_envelope.v1"
STATE_PAYLOAD_SCHEMA = "p4.cc_bootstrap_lifecycle_state_payload.v1"
STATE_SIGNATURE_DOMAIN = "p4.cc_bootstrap_lifecycle.state.v1"
CONTROLLER_LOCATION = "external-control-plane"
ARMS = ("cc-on", "cc-off")
CONTROL_BINDING_KEYS = {
    "ttl": ("ttl_enforcement_path", "ttl_enforcement_sha256"),
    "deadman": ("deadman_switch_path", "deadman_switch_sha256"),
    "teardown": ("teardown_runbook_path", "teardown_runbook_sha256"),
    "budget": ("budget_enforcement_path", "budget_enforcement_sha256"),
}
RESOURCE_KEYS = {
    "provider", "candidate_id", "sku", "region", "zone", "instance_id",
    "boot_id_sha256", "image_reference", "protection_mode", "pair_id",
    "cuda_device_index", "deployment_manifest_sha256",
    "pair_manifest_sha256", "hardware_packet_sha256",
}
SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_.:-]{3,128}$")
FAILURE_CODE_RE = re.compile(r"^[a-z0-9][a-z0-9_.:-]{2,127}$")
SHELL_PROGRAMS = {
    "sh", "bash", "dash", "zsh", "fish", "csh", "tcsh", "cmd",
    "cmd.exe", "powershell", "powershell.exe", "pwsh", "pwsh.exe",
}


class BootstrapLifecycleError(ValueError):
    """L0, its transition state, or an exact byte binding is invalid."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise BootstrapLifecycleError(f"{label} must be a lowercase SHA-256")
    return value


def _require_identifier(value: Any, label: str) -> str:
    if (not isinstance(value, str) or IDENTIFIER_RE.fullmatch(value) is None
            or SHA256_RE.fullmatch(value) is not None):
        raise BootstrapLifecycleError(f"{label} is not a valid identifier")
    return value


def _artifact(snapshot: FileSnapshot) -> dict[str, Any]:
    return {
        "path": str(snapshot.path),
        "sha256": snapshot.sha256,
        "bytes": snapshot.size,
    }


def _snapshot_artifact(path: str | Path, label: str, *,
                       max_bytes: int = 32 * 1024 * 1024) -> FileSnapshot:
    try:
        return read_snapshot(path, max_bytes=max_bytes)
    except (OSError, SecureIOError) as exc:
        raise BootstrapLifecycleError(f"cannot snapshot {label}") from exc


def _json_artifact(path: str | Path, label: str) -> tuple[dict[str, Any], FileSnapshot]:
    try:
        return read_json_snapshot(path)
    except (OSError, SecureIOError) as exc:
        raise BootstrapLifecycleError(f"cannot read {label}") from exc


def _binding_matches(binding: Any, snapshot: FileSnapshot, label: str) -> None:
    if (not isinstance(binding, Mapping)
            or set(binding) != {"path", "sha256", "bytes"}
            or binding.get("path") != str(snapshot.path)
            or binding.get("sha256") != snapshot.sha256
            or type(binding.get("bytes")) is not int
            or binding.get("bytes") != snapshot.size):
        raise BootstrapLifecycleError(f"{label} byte binding differs")


def _packet_path(packet_value: Any, label: str) -> Path:
    if not isinstance(packet_value, str) or not packet_value.strip():
        raise BootstrapLifecycleError(f"hardware packet {label} path is missing")
    path = Path(packet_value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def _bounded_argv(path: str | Path, label: str) -> tuple[dict[str, Any], FileSnapshot]:
    snapshot = _snapshot_artifact(path, label, max_bytes=1024 * 1024)
    try:
        value = json.loads(snapshot.data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BootstrapLifecycleError(f"{label} is not valid UTF-8 JSON") from exc
    if (not isinstance(value, list) or not 1 <= len(value) <= 256
            or any(not isinstance(item, str) or not item or len(item) > 4096
                   or "\x00" in item for item in value)):
        raise BootstrapLifecycleError(
            f"{label} must be a nonempty bounded string array")
    if Path(value[0]).name.casefold() in SHELL_PROGRAMS:
        raise BootstrapLifecycleError(f"{label} cannot invoke a shell")
    return {"artifact": _artifact(snapshot), "argv": value}, snapshot


def _require_argv_adapter(argv: list[str], adapter_path: Path) -> None:
    matches = []
    for item in argv:
        candidate = Path(item)
        if not candidate.is_absolute():
            candidate = ROOT / candidate
        try:
            matches.append(candidate.resolve() == adapter_path)
        except OSError:
            matches.append(False)
    if not any(matches):
        raise BootstrapLifecycleError(
            "active-security remote-launch argv does not name the frozen adapter")


def _require_public_key(path: str | Path, label: str) -> FileSnapshot:
    snapshot = _snapshot_artifact(path, label, max_bytes=1024 * 1024)
    try:
        result = subprocess.run(
            ["openssl", "pkey", "-pubin", "-in", str(snapshot.path),
             "-text_pub", "-noout"],
            capture_output=True, check=False, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise BootstrapLifecycleError(f"cannot inspect {label}") from exc
    first = result.stdout.splitlines()[0] if result.stdout else b""
    if result.returncode != 0 or first.strip() != b"ED25519 Public-Key:":
        raise BootstrapLifecycleError(f"{label} must be Ed25519")
    require_unchanged(snapshot)
    return snapshot


def _require_keypair(private_key_path: str | Path,
                     public_key_path: str | Path) -> FileSnapshot:
    private = _snapshot_artifact(
        private_key_path, "lifecycle controller private key", max_bytes=1024 * 1024)
    public = _require_public_key(
        public_key_path, "lifecycle controller public key")
    if stat.S_IMODE(private.path.stat().st_mode) & 0o077:
        raise BootstrapLifecycleError(
            "lifecycle controller private key must be owner-only")
    with tempfile.TemporaryDirectory(prefix="p4-l0-key-") as temporary:
        derived = Path(temporary) / "public.pem"
        try:
            result = subprocess.run(
                ["openssl", "pkey", "-in", str(private.path), "-pubout",
                 "-out", str(derived)],
                capture_output=True, check=False, timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise BootstrapLifecycleError("cannot inspect controller keypair") from exc
        if (result.returncode != 0 or not derived.is_file()
                or _snapshot_artifact(derived, "derived controller public key").sha256
                != public.sha256):
            raise BootstrapLifecycleError(
                "lifecycle controller keys do not form an Ed25519 keypair")
    require_unchanged(private)
    return public


def _signature_message(domain: str, payload: Mapping[str, Any]) -> bytes:
    return _canonical_bytes({"domain": domain, "payload": payload})


def _sign(payload: Mapping[str, Any], *, schema: str, domain: str,
          private_key_path: str | Path,
          public_key_path: str | Path) -> dict[str, Any]:
    public = _require_keypair(private_key_path, public_key_path)
    message = _signature_message(domain, payload)
    with tempfile.TemporaryDirectory(prefix="p4-l0-sign-") as temporary:
        message_path = Path(temporary) / "message.bin"
        signature_path = Path(temporary) / "signature.bin"
        message_path.write_bytes(message)
        try:
            result = subprocess.run(
                ["openssl", "pkeyutl", "-sign", "-inkey",
                 str(private_key_path), "-rawin", "-in", str(message_path),
                 "-out", str(signature_path)],
                capture_output=True, check=False, timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise BootstrapLifecycleError("controller signing failed") from exc
        if result.returncode != 0 or not signature_path.is_file():
            raise BootstrapLifecycleError("controller signing failed")
        signature = signature_path.read_bytes()
    if len(signature) != 64:
        raise BootstrapLifecycleError("controller signature is not Ed25519")
    return {
        "schema": schema,
        "payload": dict(payload),
        "payload_sha256": _sha256_bytes(_canonical_bytes(payload)),
        "signature_domain": domain,
        "public_key_sha256": public.sha256,
        "signature_algorithm": "Ed25519",
        "signature_base64": base64.b64encode(signature).decode("ascii"),
    }


def _verify_envelope(path: str | Path, public_key_path: str | Path, *,
                     schema: str, domain: str) -> tuple[dict[str, Any], FileSnapshot]:
    envelope, snapshot = _json_artifact(path, "signed lifecycle artifact")
    public = _require_public_key(public_key_path, "lifecycle controller public key")
    expected = {
        "schema", "payload", "payload_sha256", "signature_domain",
        "public_key_sha256", "signature_algorithm", "signature_base64",
    }
    payload = envelope.get("payload")
    if (set(envelope) != expected or envelope.get("schema") != schema
            or envelope.get("signature_domain") != domain
            or envelope.get("public_key_sha256") != public.sha256
            or envelope.get("signature_algorithm") != "Ed25519"
            or not isinstance(payload, dict)
            or envelope.get("payload_sha256")
            != _sha256_bytes(_canonical_bytes(payload))):
        raise BootstrapLifecycleError("signed lifecycle envelope fields differ")
    try:
        signature = base64.b64decode(
            envelope.get("signature_base64", ""), validate=True)
    except (TypeError, ValueError, binascii.Error) as exc:
        raise BootstrapLifecycleError(
            "lifecycle signature is not canonical base64") from exc
    if len(signature) != 64:
        raise BootstrapLifecycleError("lifecycle Ed25519 signature must be 64 bytes")
    message = _signature_message(domain, payload)
    with tempfile.TemporaryDirectory(prefix="p4-l0-verify-") as temporary:
        message_path = Path(temporary) / "message.bin"
        signature_path = Path(temporary) / "signature.bin"
        message_path.write_bytes(message)
        signature_path.write_bytes(signature)
        try:
            result = subprocess.run(
                ["openssl", "pkeyutl", "-verify", "-pubin", "-inkey",
                 str(public.path), "-rawin", "-in", str(message_path),
                 "-sigfile", str(signature_path)],
                capture_output=True, check=False, timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise BootstrapLifecycleError("lifecycle signature check failed") from exc
    if result.returncode != 0:
        raise BootstrapLifecycleError("lifecycle signature check failed")
    require_unchanged(snapshot)
    return dict(payload), snapshot


def _resource_identity(admission: Mapping[str, Any]) -> dict[str, Any]:
    authentication = admission.get("authentication")
    resource = authentication.get("deployment_binding") \
        if isinstance(authentication, Mapping) else None
    if not isinstance(resource, Mapping) or set(resource) != RESOURCE_KEYS:
        raise BootstrapLifecycleError("CC-on resource identity fields differ")
    if resource.get("protection_mode") != "cc-on":
        raise BootstrapLifecycleError("bootstrap resource is not CC-on")
    for key in ("boot_id_sha256", "deployment_manifest_sha256",
                "pair_manifest_sha256", "hardware_packet_sha256"):
        _require_sha256(resource.get(key), f"resource {key}")
    if type(resource.get("cuda_device_index")) is not int \
            or resource["cuda_device_index"] < 0:
        raise BootstrapLifecycleError("resource CUDA index is invalid")
    return dict(resource)


def _validated_payload(*, pair_declaration_path: str | Path,
                       pair_auth_path: str | Path,
                       pair_campaign_public_key_path: str | Path,
                       hardware_packet_path: str | Path,
                       deployment_manifest_path: str | Path,
                       runtime_binding_path: str | Path,
                       admission_path: str | Path,
                       admission_evidence_root: str | Path,
                       arrival_path: str | Path,
                       active_security_remote_launch_argv_path: str | Path,
                       teardown_argv_path: str | Path,
                       absence_verifier_argv_path: str | Path,
                       remote_launch_adapter_path: str | Path,
                       controller_public_key_path: str | Path,
                       campaign_ledger_path: str | Path,
                       control_paths: Mapping[str, str | Path]) -> tuple[
                           dict[str, Any], tuple[FileSnapshot, ...]]:
    try:
        pair = validate_pair_declaration(
            pair_declaration_path, pair_auth_path,
            pair_campaign_public_key_path)
    except (CampaignSealError, OSError, SecureIOError, ValueError) as exc:
        raise BootstrapLifecycleError("signed pair declaration is invalid") from exc
    pair_document, pair_snapshot = _json_artifact(
        pair_declaration_path, "pair declaration")
    pair_auth_snapshot = _snapshot_artifact(
        pair_auth_path, "pair declaration authentication")
    pair_key_snapshot = _require_public_key(
        pair_campaign_public_key_path, "pair campaign public key")
    if pair != pair_document:
        raise BootstrapLifecycleError("validated pair declaration bytes differ")

    packet, packet_snapshot = _json_artifact(
        hardware_packet_path, "hardware packet")
    deployment, deployment_snapshot = _json_artifact(
        deployment_manifest_path, "deployment manifest")
    runtime, runtime_snapshot = _json_artifact(
        runtime_binding_path, "active runtime binding")
    admission, admission_snapshot = _json_artifact(
        admission_path, "CC-on admission")
    arrival, arrival_snapshot = _json_artifact(arrival_path, "CC-on arrival")
    try:
        validate_packet(packet)
        if packet.get("status") != "frozen":
            raise BootstrapLifecycleError("hardware packet must be frozen before L0")
        validate_deployment_manifest(
            deployment, deployment_snapshot.sha256, deployment.get("profile"),
            deployment.get("adapter", {}).get("command_argv", []))
        validate_active_runtime_binding(runtime, deployment_snapshot.sha256)
        validate_admission_record(
            admission, root=Path(admission_evidence_root).resolve(),
            verify_artifacts=True)
        validate_arrival_record(arrival)
    except (HardwarePreorderError, ActiveSecurityError, PermissionError,
            SecureIOError, ValueError) as exc:
        raise BootstrapLifecycleError(
            "hardware/deployment/runtime/admission chain is invalid") from exc
    if admission.get("admitted") is not True \
            or admission.get("hardware_attested") is not True:
        raise BootstrapLifecycleError("CC-on admission is not passing")
    authentication = admission.get("authentication")
    verifier = authentication.get("verifier") \
        if isinstance(authentication, Mapping) else None
    if (not isinstance(verifier, Mapping)
            or verifier.get("adapter_kind") != "vendor_attestation_adapter"):
        raise BootstrapLifecycleError(
            "L0 requires a production vendor-attestation admission")

    evidence_root = Path(admission_evidence_root).resolve()
    declared_arrival = (evidence_root / str(admission.get("arrival_record"))).resolve()
    try:
        declared_arrival.relative_to(evidence_root)
    except ValueError as exc:
        raise BootstrapLifecycleError("admission arrival path escapes its root") from exc
    if (declared_arrival != arrival_snapshot.path
            or admission.get("arrival_record_sha256") != arrival_snapshot.sha256):
        raise BootstrapLifecycleError("admission does not bind the supplied arrival bytes")

    pair_inputs = pair.get("immutable_inputs")
    pair_keys = pair.get("public_keys")
    if not isinstance(pair_inputs, Mapping) or not isinstance(pair_keys, Mapping):
        raise BootstrapLifecycleError("pair byte inventories are missing")
    _binding_matches(pair_inputs.get("hardware_packet"), packet_snapshot,
                     "pair hardware packet")
    _binding_matches(pair_inputs.get("deployment_manifest"), deployment_snapshot,
                     "pair deployment manifest")
    _binding_matches(pair_keys.get("campaign"), pair_key_snapshot,
                     "pair campaign public key")

    controller_key = _require_public_key(
        controller_public_key_path, "lifecycle controller public key")
    _binding_matches(pair_keys.get("lifecycle_controller"), controller_key,
                     "pair lifecycle controller public key")
    bindings = packet.get("bindings")
    if not isinstance(bindings, Mapping):
        raise BootstrapLifecycleError("hardware packet bindings are missing")
    for path_key, digest_key, snapshot, label in (
        ("lifecycle_controller_public_key_path",
         "lifecycle_controller_public_key_sha256", controller_key,
         "lifecycle controller public key"),
    ):
        if (_packet_path(bindings.get(path_key), label) != snapshot.path
                or bindings.get(digest_key) != snapshot.sha256):
            raise BootstrapLifecycleError(f"packet {label} binding differs")

    adapter_snapshot = _snapshot_artifact(
        remote_launch_adapter_path, "remote-launch adapter")
    if (_packet_path(bindings.get("remote_launch_adapter_path"),
                     "remote-launch adapter") != adapter_snapshot.path
            or bindings.get("remote_launch_adapter_sha256")
            != adapter_snapshot.sha256):
        raise BootstrapLifecycleError(
            "packet remote-launch adapter binding differs")

    command_paths = {
        "active_security_remote_launch": active_security_remote_launch_argv_path,
        "teardown": teardown_argv_path,
        "absence_verifier": absence_verifier_argv_path,
    }
    commands: dict[str, dict[str, Any]] = {}
    command_snapshots: list[FileSnapshot] = []
    for name, path in command_paths.items():
        command, snapshot = _bounded_argv(path, name.replace("_", " "))
        commands[name] = command
        command_snapshots.append(snapshot)
    if len({snapshot.sha256 for snapshot in command_snapshots}) != 3:
        raise BootstrapLifecycleError("bootstrap command artifacts must be distinct")
    _require_argv_adapter(
        commands["active_security_remote_launch"]["argv"],
        adapter_snapshot.path)

    if set(control_paths) != set(CONTROL_BINDING_KEYS):
        raise BootstrapLifecycleError(
            "L0 requires exact ttl/deadman/teardown/budget controls")
    controls: dict[str, dict[str, Any]] = {}
    control_snapshots: list[FileSnapshot] = []
    for name in sorted(CONTROL_BINDING_KEYS):
        path_key, digest_key = CONTROL_BINDING_KEYS[name]
        snapshot = _snapshot_artifact(control_paths[name], f"{name} control")
        try:
            validated = validate_control_artifact(snapshot.path, name)
        except (LifecycleError, ValueError) as exc:
            raise BootstrapLifecycleError(f"{name} control is invalid") from exc
        if (validated.get("sha256") != snapshot.sha256
                or validated.get("bytes") != snapshot.size
                or _packet_path(bindings.get(path_key), f"{name} control")
                != snapshot.path
                or bindings.get(digest_key) != snapshot.sha256):
            raise BootstrapLifecycleError(
                f"packet {name} control binding differs")
        controls[name] = _artifact(snapshot)
        control_snapshots.append(snapshot)

    resource = _resource_identity(admission)
    arrival_deployment = arrival.get("deployment")
    if not isinstance(arrival_deployment, Mapping) \
            or dict(arrival_deployment) != resource:
        raise BootstrapLifecycleError(
            "signed admission resource differs from arrival resource")
    if (runtime.get("admission_id") != admission.get("admission_id")
            or runtime.get("session_id") != admission.get("session_id")
            or runtime.get("admission_sha256") != admission_snapshot.sha256
            or runtime.get("arrival_record_sha256") != arrival_snapshot.sha256
            or runtime.get("instance_id") != resource["instance_id"]
            or runtime.get("hardware_packet_sha256") != packet_snapshot.sha256
            or runtime.get("deployment_manifest_sha256")
            != deployment_snapshot.sha256):
        raise BootstrapLifecycleError(
            "runtime binding differs from admission/arrival/static inputs")
    if (resource["deployment_manifest_sha256"] != deployment_snapshot.sha256
            or resource["hardware_packet_sha256"] != packet_snapshot.sha256
            or resource["pair_manifest_sha256"] != pair_snapshot.sha256
            or resource["pair_id"] != pair.get("pair_id")):
        raise BootstrapLifecycleError(
            "CC-on resource differs from the signed pair/static inputs")
    if (authentication.get("public_key", {}).get("sha256")
            != bindings.get("admission_verifier_public_key_sha256")
            or verifier.get("verifier_id")
            != bindings.get("admission_verifier_id")
            or verifier.get("adapter_measurement_sha256")
            != bindings.get("admission_verifier_adapter_sha256")):
        raise BootstrapLifecycleError(
            "admission verifier differs from the hardware trust anchor")

    ledger = Path(campaign_ledger_path)
    if not ledger.is_absolute():
        ledger = (ROOT / ledger)
    ledger = ledger.resolve(strict=False)
    if ledger == Path(ledger.anchor):
        raise BootstrapLifecycleError("campaign ledger path is too broad")
    try:
        if ledger.is_symlink() or (ledger.exists() and not ledger.is_file()):
            raise BootstrapLifecycleError(
                "campaign ledger must be absent or a regular non-symlink file")
    except OSError as exc:
        raise BootstrapLifecycleError("campaign ledger path is unavailable") from exc

    payload = {
        "schema": L0_PAYLOAD_SCHEMA,
        "scientific_evidence": False,
        "controller_location": CONTROLLER_LOCATION,
        "phase": "pre-active-security",
        "active_security_manifest_sha256": None,
        "campaign": {
            "campaign_id": _require_identifier(
                pair.get("campaign_id"), "campaign_id"),
            "pair_id": _require_identifier(pair.get("pair_id"), "pair_id"),
            "ledger_path": str(ledger),
            "ledger_schema": LEDGER_SCHEMA,
        },
        "pair": {
            "declaration": _artifact(pair_snapshot),
            "authentication": _artifact(pair_auth_snapshot),
            "campaign_public_key": _artifact(pair_key_snapshot),
        },
        "artifacts": {
            "hardware_packet": _artifact(packet_snapshot),
            "deployment_manifest": _artifact(deployment_snapshot),
            "runtime_binding": _artifact(runtime_snapshot),
            "cc_on_admission": _artifact(admission_snapshot),
            "cc_on_arrival": _artifact(arrival_snapshot),
            "remote_launch_adapter": _artifact(adapter_snapshot),
            "controller_public_key": _artifact(controller_key),
        },
        "cc_on": {
            "admission_id": admission["admission_id"],
            "session_id": admission["session_id"],
            "resource_identity": resource,
            "resource_identity_sha256": _sha256_bytes(
                _canonical_bytes(resource)),
        },
        "commands": commands,
        "controls": controls,
    }
    snapshots = (
        pair_snapshot, pair_auth_snapshot, pair_key_snapshot, packet_snapshot,
        deployment_snapshot, runtime_snapshot, admission_snapshot,
        arrival_snapshot, adapter_snapshot, controller_key,
        *command_snapshots, *control_snapshots,
    )
    for snapshot in snapshots:
        require_unchanged(snapshot)
    return payload, snapshots


def create_bootstrap_contract(
    output_path: str | Path, *, pair_declaration_path: str | Path,
    pair_auth_path: str | Path, pair_campaign_public_key_path: str | Path,
    hardware_packet_path: str | Path, deployment_manifest_path: str | Path,
    runtime_binding_path: str | Path, admission_path: str | Path,
    admission_evidence_root: str | Path, arrival_path: str | Path,
    active_security_remote_launch_argv_path: str | Path,
    teardown_argv_path: str | Path, absence_verifier_argv_path: str | Path,
    remote_launch_adapter_path: str | Path,
    controller_private_key_path: str | Path,
    controller_public_key_path: str | Path,
    campaign_ledger_path: str | Path,
    control_paths: Mapping[str, str | Path],
) -> Path:
    """Validate the entire pre-active chain and exclusively write signed L0."""
    payload, snapshots = _validated_payload(
        pair_declaration_path=pair_declaration_path,
        pair_auth_path=pair_auth_path,
        pair_campaign_public_key_path=pair_campaign_public_key_path,
        hardware_packet_path=hardware_packet_path,
        deployment_manifest_path=deployment_manifest_path,
        runtime_binding_path=runtime_binding_path,
        admission_path=admission_path,
        admission_evidence_root=admission_evidence_root,
        arrival_path=arrival_path,
        active_security_remote_launch_argv_path=(
            active_security_remote_launch_argv_path),
        teardown_argv_path=teardown_argv_path,
        absence_verifier_argv_path=absence_verifier_argv_path,
        remote_launch_adapter_path=remote_launch_adapter_path,
        controller_public_key_path=controller_public_key_path,
        campaign_ledger_path=campaign_ledger_path,
        control_paths=control_paths,
    )
    envelope = _sign(
        payload, schema=L0_ENVELOPE_SCHEMA, domain=L0_SIGNATURE_DOMAIN,
        private_key_path=controller_private_key_path,
        public_key_path=controller_public_key_path)
    try:
        path = write_json_exclusive(output_path, envelope)
        os.chmod(path, 0o600)
    except (OSError, SecureIOError) as exc:
        raise BootstrapLifecycleError("cannot exclusively write L0") from exc
    for snapshot in snapshots:
        require_unchanged(snapshot)
    return path


def verify_bootstrap_envelope(
        path: str | Path, controller_public_key_path: str | Path) -> dict[str, Any]:
    """Verify L0's exact schema and Ed25519 signature (not external bytes)."""
    payload, _ = _verify_envelope(
        path, controller_public_key_path, schema=L0_ENVELOPE_SCHEMA,
        domain=L0_SIGNATURE_DOMAIN)
    expected = {
        "schema", "scientific_evidence", "controller_location", "phase",
        "active_security_manifest_sha256", "campaign", "pair", "artifacts",
        "cc_on", "commands", "controls",
    }
    if (set(payload) != expected or payload.get("schema") != L0_PAYLOAD_SCHEMA
            or payload.get("scientific_evidence") is not False
            or payload.get("controller_location") != CONTROLLER_LOCATION
            or payload.get("phase") != "pre-active-security"
            or payload.get("active_security_manifest_sha256") is not None):
        raise BootstrapLifecycleError("L0 payload fields differ")
    if set(payload.get("commands", {})) != {
        "active_security_remote_launch", "teardown", "absence_verifier",
    } or set(payload.get("controls", {})) != set(CONTROL_BINDING_KEYS):
        raise BootstrapLifecycleError("L0 command/control inventory differs")
    campaign = payload.get("campaign")
    if (not isinstance(campaign, Mapping)
            or set(campaign) != {
                "campaign_id", "pair_id", "ledger_path", "ledger_schema"}
            or campaign.get("ledger_schema") != LEDGER_SCHEMA):
        raise BootstrapLifecycleError("L0 campaign binding differs")
    _require_identifier(campaign["campaign_id"], "campaign_id")
    _require_identifier(campaign["pair_id"], "pair_id")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "hardware_packet", "deployment_manifest", "runtime_binding",
        "cc_on_admission", "cc_on_arrival", "remote_launch_adapter",
        "controller_public_key",
    }:
        raise BootstrapLifecycleError("L0 artifact inventory differs")
    pair = payload.get("pair")
    if not isinstance(pair, Mapping) or set(pair) != {
        "declaration", "authentication", "campaign_public_key",
    }:
        raise BootstrapLifecycleError("L0 pair artifact inventory differs")
    for group in (payload.get("pair"), artifacts, payload.get("controls")):
        if not isinstance(group, Mapping):
            raise BootstrapLifecycleError("L0 artifact binding group is invalid")
        for label, binding in group.items():
            if (not isinstance(binding, Mapping)
                    or set(binding) != {"path", "sha256", "bytes"}
                    or not isinstance(binding.get("path"), str)
                    or not binding["path"]
                    or not SHA256_RE.fullmatch(str(binding.get("sha256")))
                    or type(binding.get("bytes")) is not int
                    or binding["bytes"] < 1):
                raise BootstrapLifecycleError(
                    f"L0 {label} artifact binding is invalid")
    commands = payload["commands"]
    for label, command in commands.items():
        if (not isinstance(command, Mapping)
                or set(command) != {"artifact", "argv"}
                or not isinstance(command.get("argv"), list)
                or not 1 <= len(command["argv"]) <= 256
                or any(not isinstance(item, str) or not item
                       or len(item) > 4096 or "\x00" in item
                       for item in command["argv"])
                or Path(command["argv"][0]).name.casefold() in SHELL_PROGRAMS):
            raise BootstrapLifecycleError(f"L0 {label} command binding is invalid")
        binding = command["artifact"]
        if (not isinstance(binding, Mapping)
                or set(binding) != {"path", "sha256", "bytes"}
                or not SHA256_RE.fullmatch(str(binding.get("sha256")))):
            raise BootstrapLifecycleError(
                f"L0 {label} command artifact is invalid")
    cc_on = payload.get("cc_on")
    if (not isinstance(cc_on, Mapping)
            or set(cc_on) != {
                "admission_id", "session_id", "resource_identity",
                "resource_identity_sha256"}
            or not isinstance(cc_on.get("resource_identity"), Mapping)
            or set(cc_on["resource_identity"]) != RESOURCE_KEYS
            or cc_on.get("resource_identity_sha256")
            != _sha256_bytes(_canonical_bytes(cc_on["resource_identity"]))):
        raise BootstrapLifecycleError("L0 CC-on identity binding differs")
    return payload


def validate_bootstrap_contract(
    path: str | Path, *, controller_public_key_path: str | Path,
    pair_declaration_path: str | Path, pair_auth_path: str | Path,
    pair_campaign_public_key_path: str | Path,
    hardware_packet_path: str | Path, deployment_manifest_path: str | Path,
    runtime_binding_path: str | Path, admission_path: str | Path,
    admission_evidence_root: str | Path, arrival_path: str | Path,
    active_security_remote_launch_argv_path: str | Path,
    teardown_argv_path: str | Path, absence_verifier_argv_path: str | Path,
    remote_launch_adapter_path: str | Path,
    campaign_ledger_path: str | Path,
    control_paths: Mapping[str, str | Path],
) -> dict[str, Any]:
    """Rebuild L0 from exact supplied bytes and reject any difference."""
    payload = verify_bootstrap_envelope(path, controller_public_key_path)
    expected, snapshots = _validated_payload(
        pair_declaration_path=pair_declaration_path,
        pair_auth_path=pair_auth_path,
        pair_campaign_public_key_path=pair_campaign_public_key_path,
        hardware_packet_path=hardware_packet_path,
        deployment_manifest_path=deployment_manifest_path,
        runtime_binding_path=runtime_binding_path,
        admission_path=admission_path,
        admission_evidence_root=admission_evidence_root,
        arrival_path=arrival_path,
        active_security_remote_launch_argv_path=(
            active_security_remote_launch_argv_path),
        teardown_argv_path=teardown_argv_path,
        absence_verifier_argv_path=absence_verifier_argv_path,
        remote_launch_adapter_path=remote_launch_adapter_path,
        controller_public_key_path=controller_public_key_path,
        campaign_ledger_path=campaign_ledger_path,
        control_paths=control_paths,
    )
    if payload != expected:
        raise BootstrapLifecycleError("L0 differs from the exact validated inputs")
    for snapshot in snapshots:
        require_unchanged(snapshot)
    return payload


def _state_payload(*, l0_sha256: str | None, campaign_id: str,
                   pair_id: str, sequence: int, phase: str,
                   previous_state_sha256: str | None,
                   l1: Mapping[str, Any] | None,
                   active_security_manifest_sha256: str | None,
                   failure_code: str | None,
                   controller_public_key_sha256: str) -> dict[str, Any]:
    flags = {
        "l0-active-security-authorized": (True, False, False),
        "active-security-complete": (False, False, True),
        "l1-measurement-ready": (False, True, False),
        "teardown-required": (False, False, True),
    }
    if phase not in flags:
        raise BootstrapLifecycleError("unknown bootstrap lifecycle state")
    active, measurement, teardown = flags[phase]
    return {
        "schema": STATE_PAYLOAD_SCHEMA,
        "campaign_id": _require_identifier(campaign_id, "campaign_id"),
        "pair_id": _require_identifier(pair_id, "pair_id"),
        "sequence": sequence,
        "phase": phase,
        "previous_state_sha256": previous_state_sha256,
        "l0_sha256": l0_sha256,
        "l1": dict(l1) if l1 is not None else None,
        "active_security_manifest_sha256": active_security_manifest_sha256,
        "active_security_permitted": active,
        "measurement_permitted": measurement,
        "teardown_required": teardown,
        "failure_code": failure_code,
        "controller_public_key_sha256": controller_public_key_sha256,
    }


def _write_state(path: str | Path, payload: Mapping[str, Any], *,
                 controller_private_key_path: str | Path,
                 controller_public_key_path: str | Path) -> Path:
    envelope = _sign(
        payload, schema=STATE_ENVELOPE_SCHEMA, domain=STATE_SIGNATURE_DOMAIN,
        private_key_path=controller_private_key_path,
        public_key_path=controller_public_key_path)
    try:
        output = write_json_exclusive(path, envelope)
        os.chmod(output, 0o600)
        return output
    except (OSError, SecureIOError) as exc:
        raise BootstrapLifecycleError(
            "cannot exclusively write bootstrap state") from exc


def record_l0_success(
    output_path: str | Path, *, l0_path: str | Path,
    controller_private_key_path: str | Path,
    controller_public_key_path: str | Path,
) -> Path:
    """Record that signed L0 passed; active security, but not measurement, may run."""
    l0 = verify_bootstrap_envelope(l0_path, controller_public_key_path)
    snapshot = _snapshot_artifact(l0_path, "L0")
    campaign = l0["campaign"]
    public = _require_public_key(
        controller_public_key_path, "lifecycle controller public key")
    payload = _state_payload(
        l0_sha256=snapshot.sha256, campaign_id=campaign["campaign_id"],
        pair_id=campaign["pair_id"], sequence=0,
        phase="l0-active-security-authorized", previous_state_sha256=None,
        l1=None, active_security_manifest_sha256=None, failure_code=None,
        controller_public_key_sha256=public.sha256)
    require_unchanged(snapshot)
    return _write_state(
        output_path, payload,
        controller_private_key_path=controller_private_key_path,
        controller_public_key_path=controller_public_key_path)


def record_l0_failure(
    output_path: str | Path, *, campaign_id: str, pair_id: str,
    failure_code: str, controller_private_key_path: str | Path,
    controller_public_key_path: str | Path,
    attempted_l0_path: str | Path | None = None,
) -> Path:
    """Record fail-closed L0 failure; this state always mandates teardown."""
    if not isinstance(failure_code, str) \
            or FAILURE_CODE_RE.fullmatch(failure_code) is None:
        raise BootstrapLifecycleError("failure code is invalid")
    l0_sha256 = None
    if attempted_l0_path is not None:
        l0_sha256 = _snapshot_artifact(
            attempted_l0_path, "attempted L0").sha256
    public = _require_public_key(
        controller_public_key_path, "lifecycle controller public key")
    payload = _state_payload(
        l0_sha256=l0_sha256, campaign_id=campaign_id, pair_id=pair_id,
        sequence=0, phase="teardown-required",
        previous_state_sha256=None, l1=None,
        active_security_manifest_sha256=None, failure_code=failure_code,
        controller_public_key_sha256=public.sha256)
    return _write_state(
        output_path, payload,
        controller_private_key_path=controller_private_key_path,
        controller_public_key_path=controller_public_key_path)


def verify_bootstrap_state(
        path: str | Path, controller_public_key_path: str | Path) -> dict[str, Any]:
    """Verify one exact signed lifecycle state and its fail-closed invariants."""
    payload, _ = _verify_envelope(
        path, controller_public_key_path, schema=STATE_ENVELOPE_SCHEMA,
        domain=STATE_SIGNATURE_DOMAIN)
    expected = {
        "schema", "campaign_id", "pair_id", "sequence", "phase",
        "previous_state_sha256", "l0_sha256", "l1",
        "active_security_manifest_sha256", "active_security_permitted",
        "measurement_permitted", "teardown_required", "failure_code",
        "controller_public_key_sha256",
    }
    if set(payload) != expected or payload.get("schema") != STATE_PAYLOAD_SCHEMA:
        raise BootstrapLifecycleError("bootstrap state fields differ")
    _require_identifier(payload.get("campaign_id"), "campaign_id")
    _require_identifier(payload.get("pair_id"), "pair_id")
    if type(payload.get("sequence")) is not int or payload["sequence"] < 0:
        raise BootstrapLifecycleError("bootstrap state sequence is invalid")
    public = _require_public_key(
        controller_public_key_path, "lifecycle controller public key")
    if payload.get("controller_public_key_sha256") != public.sha256:
        raise BootstrapLifecycleError("bootstrap state controller differs")
    phase = payload.get("phase")
    expected_flags = {
        "l0-active-security-authorized": (True, False, False),
        "active-security-complete": (False, False, True),
        "l1-measurement-ready": (False, True, False),
        "teardown-required": (False, False, True),
    }.get(phase)
    actual_flags = (
        payload.get("active_security_permitted"),
        payload.get("measurement_permitted"),
        payload.get("teardown_required"),
    )
    if expected_flags is None or actual_flags != expected_flags:
        raise BootstrapLifecycleError("bootstrap state permissions are invalid")
    if phase == "l0-active-security-authorized":
        _require_sha256(payload.get("l0_sha256"), "state L0")
        if (payload["sequence"] != 0 or payload.get("previous_state_sha256") is not None
                or payload.get("l1") is not None
                or payload.get("active_security_manifest_sha256") is not None
                or payload.get("failure_code") is not None):
            raise BootstrapLifecycleError("L0-success state is inconsistent")
    elif phase == "active-security-complete":
        _require_sha256(payload.get("l0_sha256"), "state L0")
        _require_sha256(payload.get("previous_state_sha256"), "previous state")
        _require_sha256(payload.get("active_security_manifest_sha256"),
                        "active-security manifest")
        if (payload["sequence"] != 1 or payload.get("l1") is not None
                or payload.get("failure_code") is not None):
            raise BootstrapLifecycleError(
                "active-security-complete state is inconsistent")
    elif phase == "l1-measurement-ready":
        _require_sha256(payload.get("l0_sha256"), "state L0")
        _require_sha256(payload.get("previous_state_sha256"), "previous state")
        _require_sha256(payload.get("active_security_manifest_sha256"),
                        "active-security manifest")
        l1 = payload.get("l1")
        if (payload["sequence"] != 2 or not isinstance(l1, Mapping)
                or set(l1) != {"path", "sha256", "bytes"}
                or not SHA256_RE.fullmatch(str(l1.get("sha256")))
                or payload.get("failure_code") is not None):
            raise BootstrapLifecycleError("measurement-ready state is inconsistent")
    else:
        if (not isinstance(payload.get("failure_code"), str)
                or FAILURE_CODE_RE.fullmatch(payload["failure_code"]) is None
                or payload.get("l1") is not None):
            raise BootstrapLifecycleError("teardown-required state is inconsistent")
        if payload["sequence"] == 0:
            if (payload.get("previous_state_sha256") is not None
                    or payload.get("active_security_manifest_sha256") is not None
                    or (payload.get("l0_sha256") is not None
                        and not SHA256_RE.fullmatch(payload["l0_sha256"]))):
                raise BootstrapLifecycleError(
                    "initial teardown-required state is inconsistent")
        elif payload["sequence"] == 1:
            _require_sha256(payload.get("previous_state_sha256"),
                            "previous state")
            _require_sha256(payload.get("l0_sha256"), "state L0")
            if payload.get("active_security_manifest_sha256") is not None:
                raise BootstrapLifecycleError(
                    "pre-active teardown state cannot bind active security")
        elif payload["sequence"] == 2:
            _require_sha256(payload.get("previous_state_sha256"),
                            "previous state")
            _require_sha256(payload.get("l0_sha256"), "state L0")
            _require_sha256(payload.get("active_security_manifest_sha256"),
                            "active-security manifest")
        else:
            raise BootstrapLifecycleError(
                "teardown-required state sequence is invalid")
    return payload


def record_active_security_success(
    output_path: str | Path, *, state_path: str | Path, l0_path: str | Path,
    active_security_manifest_path: str | Path,
    controller_private_key_path: str | Path,
    controller_public_key_path: str | Path,
) -> Path:
    """Record exact passing M bytes without referring to the later L1.

    This intermediate state is the acyclic transition that L1 must bind.  It
    remains fail-closed: measurement is forbidden and loss of controller
    continuity requires teardown.
    """
    state = verify_bootstrap_state(state_path, controller_public_key_path)
    if state.get("phase") != "l0-active-security-authorized":
        raise BootstrapLifecycleError(
            "active-security success requires an L0-authorized state")
    state_snapshot = _snapshot_artifact(state_path, "bootstrap state")
    l0_snapshot = _snapshot_artifact(l0_path, "L0")
    l0 = verify_bootstrap_envelope(l0_path, controller_public_key_path)
    if (state.get("l0_sha256") != l0_snapshot.sha256
            or state.get("campaign_id") != l0["campaign"]["campaign_id"]
            or state.get("pair_id") != l0["campaign"]["pair_id"]):
        raise BootstrapLifecycleError("active-security L0/state binding differs")
    active_snapshot = _snapshot_artifact(
        active_security_manifest_path, "active-security manifest")
    try:
        require_active_security_manifest(
            active_snapshot.path, active_snapshot.sha256)
    except (ActiveSecurityError, OSError, ValueError) as exc:
        raise BootstrapLifecycleError(
            "active-security manifest is not a passing exact artifact") from exc
    public = _require_public_key(
        controller_public_key_path, "lifecycle controller public key")
    payload = _state_payload(
        l0_sha256=l0_snapshot.sha256,
        campaign_id=state["campaign_id"], pair_id=state["pair_id"],
        sequence=1, phase="active-security-complete",
        previous_state_sha256=state_snapshot.sha256, l1=None,
        active_security_manifest_sha256=active_snapshot.sha256,
        failure_code=None, controller_public_key_sha256=public.sha256)
    for snapshot in (state_snapshot, l0_snapshot, active_snapshot):
        require_unchanged(snapshot)
    return _write_state(
        output_path, payload,
        controller_private_key_path=controller_private_key_path,
        controller_public_key_path=controller_public_key_path)


def transition_after_active_security(
    output_path: str | Path, *, state_path: str | Path,
    l0_path: str | Path, controller_private_key_path: str | Path,
    controller_public_key_path: str | Path,
    l1_path: str | Path | None = None, failure_code: str | None = None,
) -> Path:
    """Bind valid L1 for measurement, or record a teardown-required failure."""
    if (l1_path is None) == (failure_code is None):
        raise BootstrapLifecycleError(
            "transition requires exactly one L1 success or failure code")
    state = verify_bootstrap_state(state_path, controller_public_key_path)
    allowed_phases = {"l0-active-security-authorized", "active-security-complete"}
    if state.get("phase") not in allowed_phases:
        raise BootstrapLifecycleError(
            "only an active lifecycle transition state may advance")
    if l1_path is not None and state.get("phase") != "active-security-complete":
        raise BootstrapLifecycleError(
            "L1 success requires a signed active-security-complete state")
    state_snapshot = _snapshot_artifact(state_path, "bootstrap state")
    l0_snapshot = _snapshot_artifact(l0_path, "L0")
    l0 = verify_bootstrap_envelope(l0_path, controller_public_key_path)
    if (state["l0_sha256"] != l0_snapshot.sha256
            or state["campaign_id"] != l0["campaign"]["campaign_id"]
            or state["pair_id"] != l0["campaign"]["pair_id"]):
        raise BootstrapLifecycleError("transition L0/state binding differs")
    public = _require_public_key(
        controller_public_key_path, "lifecycle controller public key")
    if failure_code is not None:
        if (not isinstance(failure_code, str)
                or FAILURE_CODE_RE.fullmatch(failure_code) is None):
            raise BootstrapLifecycleError("failure code is invalid")
        payload = _state_payload(
            l0_sha256=l0_snapshot.sha256,
            campaign_id=state["campaign_id"], pair_id=state["pair_id"],
            sequence=state["sequence"] + 1, phase="teardown-required",
            previous_state_sha256=state_snapshot.sha256, l1=None,
            active_security_manifest_sha256=(
                state.get("active_security_manifest_sha256")),
            failure_code=failure_code,
            controller_public_key_sha256=public.sha256)
    else:
        l1_snapshot = _snapshot_artifact(l1_path, "L1")
        try:
            l1 = verify_lifecycle_launch_envelope(
                l1_path, controller_public_key_path,
                expected_public_key_sha256=public.sha256)
        except (LifecycleError, ValueError) as exc:
            raise BootstrapLifecycleError("post-active L1 is invalid") from exc
        hashes = {
            "hardware_packet_sha256": l0["artifacts"]["hardware_packet"]["sha256"],
            "deployment_manifest_sha256": l0["artifacts"]["deployment_manifest"]["sha256"],
            "runtime_binding_sha256": l0["artifacts"]["runtime_binding"]["sha256"],
            "active_security_manifest_sha256": state[
                "active_security_manifest_sha256"],
            "pair_declaration_sha256": l0["pair"]["declaration"]["sha256"],
            "campaign_root_sha256": l0_snapshot.sha256,
            "active_transition_sha256": state_snapshot.sha256,
        }
        if (l1.get("campaign_id") != state["campaign_id"]
                or l1.get("campaign_ledger_path")
                != l0["campaign"]["ledger_path"]
                or any(l1.get(key) != value for key, value in hashes.items())
                or l1.get("controller_location") != CONTROLLER_LOCATION
                or l1.get("predecessor") != {
                    "kind": "active-transition",
                    "sha256": state_snapshot.sha256,
                }):
            raise BootstrapLifecycleError("L1 differs from L0")
        active_sha = state["active_security_manifest_sha256"]
        arm = l1.get("arm")
        if not isinstance(arm, Mapping) or arm.get("protection_mode") != "cc-on":
            raise BootstrapLifecycleError("L1 is not the CC-on launch")
        l1_resource = arm.get("resource_identity")
        l0_resource = l0["cc_on"]["resource_identity"]
        resource_subset = {
            key: l0_resource[key]
            for key in ("provider", "candidate_id", "sku", "region", "zone",
                        "instance_id")
        }
        if l1_resource != resource_subset:
            raise BootstrapLifecycleError("L1 resource identity differs from L0")
        expected_arm = {
            "arrival_record_sha256": l0["artifacts"]["cc_on_arrival"]["sha256"],
            "authorization_kind": "cc-on-admission",
            "authorization_sha256": l0["artifacts"]["cc_on_admission"]["sha256"],
            "pair_id": l0["campaign"]["pair_id"],
            "pair_manifest_sha256": l0["pair"]["declaration"]["sha256"],
            "controller_remote_launch_argv_sha256": l0["commands"][
                "active_security_remote_launch"]["artifact"]["sha256"],
            "teardown_argv_sha256": l0["commands"]["teardown"][
                "artifact"]["sha256"],
            "absence_verifier_argv_sha256": l0["commands"][
                "absence_verifier"]["artifact"]["sha256"],
        }
        if any(arm.get(key) != value for key, value in expected_arm.items()):
            raise BootstrapLifecycleError("L1 CC-on command/authorization differs")
        payload = _state_payload(
            l0_sha256=l0_snapshot.sha256,
            campaign_id=state["campaign_id"], pair_id=state["pair_id"],
            sequence=2, phase="l1-measurement-ready",
            previous_state_sha256=state_snapshot.sha256,
            l1=_artifact(l1_snapshot),
            active_security_manifest_sha256=active_sha, failure_code=None,
            controller_public_key_sha256=public.sha256)
        require_unchanged(l1_snapshot)
    require_unchanged(state_snapshot)
    require_unchanged(l0_snapshot)
    return _write_state(
        output_path, payload,
        controller_private_key_path=controller_private_key_path,
        controller_public_key_path=controller_public_key_path)


def require_measurement_ready(
    state_path: str | Path, *, l0_path: str | Path, l1_path: str | Path,
    active_transition_path: str | Path,
    controller_public_key_path: str | Path,
) -> dict[str, Any]:
    """Fail unless signed state binds the exact L0 and L1 bytes for measurement."""
    state = verify_bootstrap_state(state_path, controller_public_key_path)
    if state.get("phase") != "l1-measurement-ready":
        raise BootstrapLifecycleError(
            "measurement requires a signed L1-bound ready state")
    l0_snapshot = _snapshot_artifact(l0_path, "L0")
    l1_snapshot = _snapshot_artifact(l1_path, "L1")
    transition_snapshot = _snapshot_artifact(
        active_transition_path, "active-security transition")
    if state.get("l0_sha256") != l0_snapshot.sha256:
        raise BootstrapLifecycleError("measurement L0 bytes differ")
    if state.get("previous_state_sha256") != transition_snapshot.sha256:
        raise BootstrapLifecycleError("measurement active transition bytes differ")
    _binding_matches(state.get("l1"), l1_snapshot, "measurement L1")
    l0 = verify_bootstrap_envelope(l0_path, controller_public_key_path)
    transition = verify_bootstrap_state(
        active_transition_path, controller_public_key_path)
    if (transition.get("phase") != "active-security-complete"
            or transition.get("l0_sha256") != l0_snapshot.sha256
            or transition.get("campaign_id") != state.get("campaign_id")
            or transition.get("pair_id") != state.get("pair_id")
            or transition.get("active_security_manifest_sha256")
            != state.get("active_security_manifest_sha256")):
        raise BootstrapLifecycleError(
            "measurement active transition binding differs")
    public = _require_public_key(
        controller_public_key_path, "lifecycle controller public key")
    try:
        l1 = verify_lifecycle_launch_envelope(
            l1_path, controller_public_key_path,
            expected_public_key_sha256=public.sha256)
    except (LifecycleError, ValueError) as exc:
        raise BootstrapLifecycleError("measurement L1 is invalid") from exc
    if (l1.get("campaign_id") != l0["campaign"]["campaign_id"]
            or l1.get("active_security_manifest_sha256")
            != state.get("active_security_manifest_sha256")
            or l1.get("campaign_root_sha256") != l0_snapshot.sha256
            or l1.get("active_transition_sha256") != transition_snapshot.sha256
            or l1.get("predecessor") != {
                "kind": "active-transition",
                "sha256": transition_snapshot.sha256,
            }):
        raise BootstrapLifecycleError("measurement L1 state binding differs")
    require_unchanged(l0_snapshot)
    require_unchanged(l1_snapshot)
    require_unchanged(transition_snapshot)
    return state
