"""Fail-closed inventory and admission records for confidential-compute hosts.

The arrival record is deliberately an inventory, never an attestation verdict.  A
separate, Ed25519-authenticated admission record may gate an engineering
workload only after every declared check has passed and its evidence is
content-addressed.  The execution gate also needs an external verifier trust
anchor, strict freshness policy, and a single-use raw challenge.  This module
checks record structure, bindings, timestamps, paths, bytes, authentication and
replay; vendor quote verification remains the responsibility of the named
adapter that emits the bound evidence.
"""
from __future__ import annotations

import base64
import binascii
from copy import deepcopy
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Mapping

import jsonschema

from .cc_runtime_attestation import (
    consume_challenge_nonce,
    validate_challenge_nonce,
)
from .cc_io import read_snapshot


ROOT = Path(__file__).resolve().parents[2]
ARRIVAL_SCHEMA_VERSION = "p4-cc-arrival-inventory-0.4"
ADMISSION_SCHEMA_VERSION = "p4-cc-admission-0.3"
ARRIVAL_SCHEMA_PATH = ROOT / "experiments/schemas/cc_arrival_record.schema.json"
ADMISSION_SCHEMA_PATH = ROOT / "experiments/schemas/cc_admission_record.schema.json"
ADMISSION_SIGNATURE_DOMAIN = "p4.cc_admission.signed_decision.v1"
AUTHENTICATION_SCHEMA_VERSION = "p4.cc_admission_authentication.v1"
ADAPTER_KINDS = ("vendor_attestation_adapter", "synthetic_fixture")
CHECK_NAMES = (
    "gpu_cc_ready",
    "protected_memory_nonzero",
    "cpu_tee_verified",
    "gpu_attestation_verified",
    "cpu_attestation_verified",
    "measured_workload_identity",
    "revocation_freshness",
    "advisory_mapping",
    "environment_lock",
    "test_suite",
)
REQUIRED_EVIDENCE_ROLES = {
    "gpu_cc_ready": {"verifier_output"},
    "protected_memory_nonzero": {"verifier_output"},
    "cpu_tee_verified": {"verifier_output"},
    "gpu_attestation_verified": {"raw_attestation_report", "verifier_output"},
    "cpu_attestation_verified": {"raw_attestation_report", "verifier_output"},
    "measured_workload_identity": {
        "workload_measurement", "journal_public_key", "deployment_manifest",
        "hardware_packet", "collector_manifest", "collector_executable",
        "collector_public_key", "trusted_store_manifest",
        "trusted_store_protection", "verifier_output",
    },
    "revocation_freshness": {"revocation_snapshot", "verifier_output"},
    "advisory_mapping": {"advisory_mapping"},
    "environment_lock": {"environment_lock", "runtime_environment"},
    "test_suite": {"test_output"},
}
CUDA_IDENTITY_FIELDS = (
    "cuda_index", "nvidia_index", "uuid", "pci_bus_id", "name", "driver",
)
DEPLOYMENT_BINDING_FIELDS = (
    "provider", "candidate_id", "sku", "region", "zone", "instance_id",
    "boot_id_sha256", "image_reference", "protection_mode", "pair_id",
    "cuda_device_index", "deployment_manifest_sha256", "pair_manifest_sha256",
    "hardware_packet_sha256",
)


@dataclass(frozen=True)
class AdmissionFreshnessPolicy:
    """Strict wall-clock bounds evaluated immediately before execution."""

    max_arrival_age_seconds: int = 1800
    max_check_age_seconds: int = 900
    max_admission_age_seconds: int = 300
    max_future_skew_seconds: int = 30

    def __post_init__(self) -> None:
        for field_name in (
            "max_arrival_age_seconds", "max_check_age_seconds",
            "max_admission_age_seconds", "max_future_skew_seconds",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a nonnegative integer")
        if min(
            self.max_arrival_age_seconds,
            self.max_check_age_seconds,
            self.max_admission_age_seconds,
        ) == 0:
            raise ValueError("freshness age bounds must be positive")


@dataclass(frozen=True)
class AdmissionTrustPolicy:
    """External trust anchor; never inferred from the self-described envelope."""

    verifier_id: str
    adapter_kind: str
    adapter_measurement_sha256: str
    public_key_sha256: str
    expected_deployment_binding: Mapping[str, Any]
    allow_synthetic: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.verifier_id, str) or not re.fullmatch(
                r"[A-Za-z0-9_.:-]+", self.verifier_id):
            raise ValueError("trusted verifier_id is invalid")
        if self.adapter_kind not in ADAPTER_KINDS:
            raise ValueError("trusted verifier adapter kind is invalid")
        if self.adapter_kind == "synthetic_fixture" and not self.allow_synthetic:
            raise ValueError("synthetic verifier trust requires explicit test-only opt-in")
        for label, value in (
            ("adapter measurement", self.adapter_measurement_sha256),
            ("public key", self.public_key_sha256),
        ):
            if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value):
                raise ValueError(f"trusted verifier {label} digest is invalid")
        if (not isinstance(self.expected_deployment_binding, Mapping)
                or set(self.expected_deployment_binding) != set(DEPLOYMENT_BINDING_FIELDS)):
            raise ValueError("trusted deployment binding fields differ from the contract")


def _load_schema(path: Path) -> dict:
    schema = json.loads(path.read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    return schema


def _schema_validate(instance: dict, schema_path: Path) -> None:
    validator = jsonschema.Draft202012Validator(
        _load_schema(schema_path), format_checker=jsonschema.FormatChecker())
    validator.validate(instance)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _regular_nonsymlink(path: Path, label: str) -> os.stat_result:
    try:
        info = path.lstat()
    except OSError as exc:
        raise ValueError(f"{label} is unavailable") from exc
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    return info


def _require_ed25519_public_key(path: Path) -> None:
    _regular_nonsymlink(path, "admission verifier public key")
    try:
        result = subprocess.run(
            ["openssl", "pkey", "-pubin", "-in", str(path), "-text_pub", "-noout"],
            check=True, capture_output=True, text=True, timeout=15,
        )
    except (FileNotFoundError, subprocess.CalledProcessError,
            subprocess.TimeoutExpired) as exc:
        raise PermissionError("admission verifier public key is unreadable") from exc
    normalized = (result.stdout + result.stderr).casefold()
    if "ed25519" not in normalized:
        raise PermissionError("admission verifier public key must be Ed25519")


def _verify_ed25519(payload: bytes, signature: bytes, public_key_bytes: bytes) -> None:
    if len(signature) != 64:
        raise PermissionError("admission signature must be a 64-byte Ed25519 signature")
    with tempfile.TemporaryDirectory(prefix="p4-admission-verify-") as temporary:
        payload_path = Path(temporary) / "payload.json"
        signature_path = Path(temporary) / "signature.bin"
        public_key = Path(temporary) / "public.pem"
        payload_path.write_bytes(payload)
        signature_path.write_bytes(signature)
        public_key.write_bytes(public_key_bytes)
        _require_ed25519_public_key(public_key)
        try:
            subprocess.run(
                [
                    "openssl", "pkeyutl", "-verify", "-pubin", "-inkey",
                    str(public_key), "-rawin", "-in", str(payload_path),
                    "-sigfile", str(signature_path),
                ],
                check=True, capture_output=True, timeout=15,
            )
        except (FileNotFoundError, subprocess.CalledProcessError,
                subprocess.TimeoutExpired) as exc:
            raise PermissionError("admission signature verification failed") from exc


def _sign_ed25519(payload: bytes, private_key: Path) -> bytes:
    _regular_nonsymlink(private_key, "admission verifier private key")
    with tempfile.TemporaryDirectory(prefix="p4-admission-sign-") as temporary:
        payload_path = Path(temporary) / "payload.json"
        signature_path = Path(temporary) / "signature.bin"
        payload_path.write_bytes(payload)
        try:
            subprocess.run(
                [
                    "openssl", "pkeyutl", "-sign", "-inkey", str(private_key),
                    "-rawin", "-in", str(payload_path), "-out", str(signature_path),
                ],
                check=True, capture_output=True, timeout=15,
            )
        except (FileNotFoundError, subprocess.CalledProcessError,
                subprocess.TimeoutExpired) as exc:
            raise PermissionError("admission signing failed") from exc
        signature = signature_path.read_bytes()
    if len(signature) != 64:
        raise PermissionError("admission signing key must be Ed25519")
    return signature


def _safe_relative(root: Path, value: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("Evidence path must be a nonempty relative string")
    path = Path(value)
    if path.is_absolute():
        raise ValueError("Evidence path must be relative to the admission root")
    resolved_root = root.resolve()
    resolved = (resolved_root / path).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("Evidence path escapes the admission root") from exc
    return resolved


def _as_utc_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Timestamp must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def observe_boot_id_sha256(path: str | Path = "/proc/sys/kernel/random/boot_id") -> str:
    """Hash the canonical Linux boot UUID without exposing it in the record."""
    source = Path(path)
    _regular_nonsymlink(source, "kernel boot identity")
    try:
        boot_id = source.read_text(encoding="ascii").strip().lower()
    except (OSError, UnicodeError) as exc:
        raise RuntimeError("kernel boot identity is unreadable") from exc
    if not re.fullmatch(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            boot_id):
        raise RuntimeError("kernel boot identity has an invalid format")
    return hashlib.sha256(boot_id.encode("ascii")).hexdigest()


def validate_arrival_record(record: dict) -> dict:
    """Validate a structured inventory without upgrading it to attestation."""
    _schema_validate(record, ARRIVAL_SCHEMA_PATH)
    gpu = record["gpu"]
    commands = (
        gpu["identity_query"], gpu["cuda_runtime_query"], gpu["cc_readiness"],
        gpu["protected_memory"],
    )
    if gpu["nvidia_smi_present"]:
        if not all(command["attempted"] for command in commands):
            raise ValueError("Present NVIDIA tooling requires all GPU inventory commands")
        identity_succeeded = all(
            gpu[key]["exit_code"] == 0
            for key in ("identity_query", "cuda_runtime_query")
        )
        has_identity = (
            isinstance(gpu["cuda_index"], int)
            and not isinstance(gpu["cuda_index"], bool)
            and isinstance(gpu["nvidia_index"], int)
            and not isinstance(gpu["nvidia_index"], bool)
            and all(isinstance(gpu[key], str) and gpu[key].strip()
                    for key in ("uuid", "pci_bus_id", "name", "driver"))
        )
        if identity_succeeded != has_identity:
            raise ValueError("Exact selected GPU identity must reflect both identity queries")
        if has_identity and gpu["cuda_index"] != record["deployment"]["cuda_device_index"]:
            raise ValueError("Selected CUDA index differs from the deployment binding")
    else:
        if any(gpu[key] is not None for key in CUDA_IDENTITY_FIELDS):
            raise ValueError("Absent NVIDIA tooling cannot report a selected GPU identity")
        if any(command["attempted"] for command in commands):
            raise ValueError("Absent NVIDIA tooling cannot claim command attempts")
    deployment = record["deployment"]
    if deployment["boot_id_sha256"] == "0" * 64:
        raise ValueError("Arrival boot identity cannot be the zero digest")
    if deployment["pair_manifest_sha256"] == deployment["deployment_manifest_sha256"]:
        raise ValueError("Pair and arm deployment manifests must be distinct artifacts")
    if deployment["provider"] == "azure":
        if deployment["candidate_id"] != "azure-ncc-h100-v5" \
                or deployment["sku"] != "Standard_NCC40ads_H100_v5":
            raise ValueError("Azure arrival identity differs from the selected candidate")
        if deployment["zone"] not in {"1", "2", "3", "regional"}:
            raise ValueError("Azure arrival zone is invalid")
    elif deployment["provider"] == "gcp":
        if deployment["candidate_id"] != "gcp-a3-h100-tdx" \
                or deployment["sku"] != "a3-highgpu-1g":
            raise ValueError("GCP arrival identity differs from the fallback candidate")
        if not re.fullmatch(r"[a-z]+-[a-z0-9]+[0-9]-[a-z]", deployment["zone"]):
            raise ValueError("GCP arrival zone is invalid")
    if "latest" in deployment["image_reference"].lower():
        raise ValueError("arrival image reference must be immutable")
    return record


def selected_cuda_device_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact logical/physical device identity bound at arrival."""
    validate_arrival_record(dict(record))
    gpu = record["gpu"]
    if any(gpu[key] is None for key in CUDA_IDENTITY_FIELDS):
        raise PermissionError("Arrival inventory has no complete selected CUDA identity")
    return {key: gpu[key] for key in CUDA_IDENTITY_FIELDS}


def observe_cuda_device_identity(device: str) -> dict[str, Any]:
    """Resolve one explicit CUDA ordinal through the runtime and NVIDIA tooling."""
    import torch

    resolved = torch.device(device)
    if resolved.type != "cuda" or resolved.index is None:
        raise ValueError("selected device must be an explicit cuda:N ordinal")
    index = resolved.index
    if not torch.cuda.is_available() or index < 0 or index >= torch.cuda.device_count():
        raise RuntimeError(f"selected CUDA device cuda:{index} is unavailable")
    try:
        nvidia_index = int(torch.cuda._get_nvml_device_index(index))
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError("cannot resolve CUDA ordinal to an NVIDIA device") from exc
    try:
        result = subprocess.run(
            [
                "nvidia-smi", f"--id={nvidia_index}",
                "--query-gpu=index,uuid,pci.bus_id,name,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=False, capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"cannot query selected NVIDIA device: {exc}") from exc
    if result.returncode != 0:
        raise RuntimeError(
            "selected NVIDIA device query failed: "
            + (result.stderr.strip() or result.stdout.strip() or "no output")
        )
    rows = list(csv.reader(io.StringIO(result.stdout)))
    if len(rows) != 1 or len(rows[0]) != 5:
        raise RuntimeError("selected NVIDIA device query returned an unexpected shape")
    fields = [field.strip() for field in rows[0]]
    try:
        queried_nvidia_index = int(fields[0])
    except ValueError as exc:
        raise RuntimeError("selected NVIDIA device query returned an invalid index") from exc
    if queried_nvidia_index != nvidia_index:
        raise RuntimeError("CUDA-to-NVIDIA device mapping changed during observation")
    runtime_name = torch.cuda.get_device_name(resolved)
    if runtime_name.strip() != fields[3]:
        raise RuntimeError("CUDA runtime and NVIDIA tooling disagree on the GPU name")
    observed = {
        "cuda_index": index,
        "nvidia_index": nvidia_index,
        "uuid": fields[1],
        "pci_bus_id": fields[2],
        "name": fields[3],
        "driver": fields[4],
    }
    if not re.fullmatch(
            r"GPU-[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-"
            r"[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}", observed["uuid"]):
        raise RuntimeError("selected NVIDIA device query returned an invalid UUID")
    if not re.fullmatch(
            r"(?:[0-9A-Fa-f]{4}|[0-9A-Fa-f]{8}):[0-9A-Fa-f]{2}:"
            r"[0-9A-Fa-f]{2}\.[0-7]", observed["pci_bus_id"]):
        raise RuntimeError("selected NVIDIA device query returned an invalid PCI bus ID")
    return observed


def validate_cuda_device_binding(record: Mapping[str, Any], device: str,
                                 observed: Mapping[str, Any]) -> dict[str, Any]:
    """Compare an observed CUDA/NVML identity with the immutable arrival record."""
    resolved_device = __import__("torch").device(device)
    if resolved_device.type != "cuda" or resolved_device.index is None:
        raise ValueError("selected device must be an explicit cuda:N ordinal")
    expected = selected_cuda_device_identity(record)
    if resolved_device.index != expected["cuda_index"]:
        raise PermissionError(
            f"requested cuda:{resolved_device.index} differs from attested "
            f"cuda:{expected['cuda_index']}"
        )
    if set(observed) != set(CUDA_IDENTITY_FIELDS):
        raise ValueError("Observed CUDA identity fields differ from the frozen contract")
    for key in CUDA_IDENTITY_FIELDS:
        actual = observed[key]
        wanted = expected[key]
        if key in {"uuid", "pci_bus_id"} and isinstance(actual, str) \
                and isinstance(wanted, str):
            matches = actual.casefold() == wanted.casefold()
        else:
            matches = actual == wanted
        if not matches:
            raise PermissionError(f"selected CUDA device {key} differs from arrival")
    return dict(expected)


def require_arrival_cuda_device(record: Mapping[str, Any], device: str) -> dict[str, Any]:
    """Fail unless the executing CUDA device is the GPU inventoried and attested."""
    # Check the logical ordinal before invoking any external query.  In
    # particular, cuda:1 can never consume evidence collected for cuda:0.
    expected = selected_cuda_device_identity(record)
    import torch
    resolved = torch.device(device)
    if resolved.type != "cuda" or resolved.index is None:
        raise ValueError("selected device must be an explicit cuda:N ordinal")
    if resolved.index != expected["cuda_index"]:
        raise PermissionError(
            f"requested cuda:{resolved.index} differs from attested "
            f"cuda:{expected['cuda_index']}"
        )
    observed = observe_cuda_device_identity(str(resolved))
    return validate_cuda_device_binding(record, str(resolved), observed)


def load_arrival_record(path) -> dict:
    record = json.loads(Path(path).read_text())
    return validate_arrival_record(record)


def admission_evidence_record(path, role, media_type, root=ROOT) -> dict:
    """Create a content-addressed evidence reference rooted at ``root``."""
    root = Path(root).resolve()
    path = Path(path).resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError("Admission evidence is outside the declared root") from exc
    if not path.is_file() or path.stat().st_size < 1:
        raise ValueError("Admission evidence must be a nonempty regular file")
    record = {
        "path": str(relative),
        "sha256": _file_sha256(path),
        "bytes": path.stat().st_size,
        "role": role,
        "media_type": media_type,
    }
    # Validate the evidence in context so role/media-type typos fail immediately.
    sample_check = {
        "status": "failed",
        "detail": "Evidence-shape validation.",
        "verifier": "p4d.cc_admission",
        "verified_at": _now(),
        "evidence": [record],
    }
    sample = _synthetic_admission_for_schema(sample_check)
    _schema_validate(sample, ADMISSION_SCHEMA_PATH)
    return record


def admission_check(status, detail, verifier, evidence=(), verified_at=None) -> dict:
    """Build one check object; admission is still decided only at bundle level."""
    check = {
        "status": status,
        "detail": detail,
        "verifier": verifier,
        "verified_at": verified_at or _now(),
        "evidence": deepcopy(list(evidence)),
    }
    sample = _synthetic_admission_for_schema(check)
    _schema_validate(sample, ADMISSION_SCHEMA_PATH)
    return check


def _decision_without_authentication(record: Mapping[str, Any]) -> dict[str, Any]:
    decision = deepcopy(dict(record))
    decision.pop("authentication", None)
    return decision


def _evidence_index(decision: Mapping[str, Any]) -> list[dict[str, Any]]:
    checks = decision.get("checks")
    if not isinstance(checks, Mapping):
        raise ValueError("admission checks are unavailable for authentication")
    return [
        {"check": name, "evidence": deepcopy(checks[name].get("evidence"))}
        for name in CHECK_NAMES
    ]


def _deployment_binding(arrival: Mapping[str, Any],
                        workload_binding: Mapping[str, Any]) -> dict[str, Any]:
    deployment = arrival.get("deployment")
    if not isinstance(deployment, Mapping):
        raise ValueError("arrival deployment binding is unavailable")
    binding = {
        key: deployment[key]
        for key in DEPLOYMENT_BINDING_FIELDS
        if key != "hardware_packet_sha256"
    }
    binding["hardware_packet_sha256"] = workload_binding.get("hardware_packet_sha256")
    if set(binding) != set(DEPLOYMENT_BINDING_FIELDS):
        raise ValueError("admission deployment binding is incomplete")
    return binding


def _authentication_metadata(*, decision: Mapping[str, Any], arrival: Mapping[str, Any],
                             verifier_id: str, adapter_kind: str,
                             adapter_measurement_sha256: str,
                             verifier_public_key_path: str | Path,
                             root: str | Path) -> dict[str, Any]:
    root_path = Path(root).resolve()
    public_key = Path(verifier_public_key_path).resolve()
    try:
        relative_public_key = public_key.relative_to(root_path)
    except ValueError as exc:
        raise ValueError("admission verifier public key is outside the evidence root") from exc
    public_snapshot = read_snapshot(public_key, max_bytes=64 * 1024)
    _require_ed25519_public_key(public_snapshot.path)
    if (not isinstance(verifier_id, str)
            or not re.fullmatch(r"[A-Za-z0-9_.:-]+", verifier_id)):
        raise ValueError("admission verifier identity is invalid")
    if adapter_kind not in ADAPTER_KINDS:
        raise ValueError("admission verifier adapter kind is invalid")
    if (not isinstance(adapter_measurement_sha256, str)
            or not re.fullmatch(r"[a-f0-9]{64}", adapter_measurement_sha256)):
        raise ValueError("admission verifier adapter measurement is invalid")
    workload_binding = decision.get("workload_binding")
    if not isinstance(workload_binding, Mapping):
        raise ValueError("admission workload binding is unavailable")
    return {
        "schema": AUTHENTICATION_SCHEMA_VERSION,
        "verifier": {
            "verifier_id": verifier_id,
            "adapter_kind": adapter_kind,
            "adapter_measurement_sha256": adapter_measurement_sha256,
        },
        "public_key": {
            "path": str(relative_public_key),
            "sha256": public_snapshot.sha256,
            "bytes": public_snapshot.size,
        },
        "deployment_binding": _deployment_binding(arrival, workload_binding),
        "decision_sha256": _canonical_sha256(decision),
        "workload_binding_sha256": _canonical_sha256(workload_binding),
        "evidence_index_sha256": _canonical_sha256(_evidence_index(decision)),
        "signature_algorithm": "Ed25519",
    }


def admission_signature_payload(record: Mapping[str, Any]) -> bytes:
    """Return the canonical domain-separated bytes covered by the signature."""
    if not isinstance(record, Mapping):
        raise ValueError("admission record must be an object")
    authentication = record.get("authentication")
    if not isinstance(authentication, Mapping):
        raise ValueError("admission authentication envelope is missing")
    metadata = deepcopy(dict(authentication))
    metadata.pop("signature_base64", None)
    return _canonical_bytes({
        "domain": ADMISSION_SIGNATURE_DOMAIN,
        "decision": _decision_without_authentication(record),
        "authentication": metadata,
    })


def sign_admission_record(record: Mapping[str, Any], *, verifier_id: str,
                          adapter_kind: str, adapter_measurement_sha256: str,
                          verifier_public_key_path: str | Path,
                          verifier_private_key_path: str | Path,
                          root: str | Path = ROOT) -> dict[str, Any]:
    """Package a decision with an Ed25519 signature; this does not verify quotes.

    The caller is the named adapter and remains responsible for vendor/platform
    attestation verification.  ``synthetic_fixture`` is accepted for tests but
    the execution gate rejects it unless its trust policy opts in explicitly.
    """
    decision = _decision_without_authentication(record)
    root_path = Path(root).resolve()
    arrival_path = _safe_relative(root_path, decision["arrival_record"])
    if (not arrival_path.is_file()
            or _file_sha256(arrival_path) != decision["arrival_record_sha256"]):
        raise ValueError("Arrival record is missing or its digest changed")
    arrival = load_arrival_record(arrival_path)
    authentication = _authentication_metadata(
        decision=decision,
        arrival=arrival,
        verifier_id=verifier_id,
        adapter_kind=adapter_kind,
        adapter_measurement_sha256=adapter_measurement_sha256,
        verifier_public_key_path=verifier_public_key_path,
        root=root_path,
    )
    signed = deepcopy(decision)
    signed["authentication"] = authentication
    signature = _sign_ed25519(
        admission_signature_payload(signed), Path(verifier_private_key_path).resolve())
    signed["authentication"]["signature_base64"] = base64.b64encode(signature).decode("ascii")
    return validate_admission_record(signed, root=root_path, verify_artifacts=True)


def _verify_admission_authentication(record: Mapping[str, Any], root: Path) -> None:
    authentication = record["authentication"]
    decision = _decision_without_authentication(record)
    if authentication["decision_sha256"] != _canonical_sha256(decision):
        raise PermissionError("admission decision digest does not match its signed envelope")
    if authentication["workload_binding_sha256"] != _canonical_sha256(
            decision["workload_binding"]):
        raise PermissionError("admission workload-binding digest does not match its envelope")
    if authentication["evidence_index_sha256"] != _canonical_sha256(
            _evidence_index(decision)):
        raise PermissionError("admission evidence-index digest does not match its envelope")
    public_key = _safe_relative(root, authentication["public_key"]["path"])
    public_snapshot = read_snapshot(
        public_key, max_bytes=64 * 1024,
        expected_sha256=authentication["public_key"]["sha256"])
    if public_snapshot.size != authentication["public_key"]["bytes"]:
        raise PermissionError("admission verifier public key bytes changed")
    try:
        signature = base64.b64decode(
            authentication["signature_base64"], validate=True)
    except (ValueError, binascii.Error) as exc:
        raise PermissionError("admission signature is not canonical base64") from exc
    _verify_ed25519(
        admission_signature_payload(record), signature, public_snapshot.data)


def _synthetic_admission_for_schema(check: dict) -> dict:
    """Embed a check for schema-only helper validation; never written as evidence."""
    checks = {name: deepcopy(check) for name in CHECK_NAMES}
    decision = {
        "schema_version": ADMISSION_SCHEMA_VERSION,
        "record_type": "admission_decision",
        "admission_id": "schema-shape-check",
        "session_id": "schema-shape-check",
        "generated_at": _now(),
        "arrival_record": "schema-shape-check.json",
        "arrival_record_sha256": "0" * 64,
        "workload_binding": {
            "source_tree_sha256": "0" * 64,
            "config_sha256": "0" * 64,
            "environment_lock_sha256": "0" * 64,
            "runtime_environment_sha256": "0" * 64,
            "trusted_store_manifest_sha256": "0" * 64,
            "trusted_store_protection_evidence_sha256": "0" * 64,
            "workload_measurement_sha256": "0" * 64,
            "challenge_nonce_sha256": "0" * 64,
            "journal_public_key_sha256": "0" * 64,
            "deployment_manifest_sha256": "0" * 64,
            "hardware_packet_sha256": "0" * 64,
            "collector_manifest_sha256": "0" * 64,
            "collector_executable_sha256": "0" * 64,
            "collector_public_key_sha256": "0" * 64,
        },
        "checks": checks,
        "hardware_attested": False,
        "admitted": False,
        "blockers": list(CHECK_NAMES),
        "claim_scope": ("Admission permits engineering workload execution only; it is not a "
                        "privacy or security certificate."),
    }
    decision["authentication"] = {
        "schema": AUTHENTICATION_SCHEMA_VERSION,
        "verifier": {
            "verifier_id": "schema-shape-verifier",
            "adapter_kind": "synthetic_fixture",
            "adapter_measurement_sha256": "1" * 64,
        },
        "public_key": {
            "path": "schema-shape-verifier-public.pem",
            "sha256": "2" * 64,
            "bytes": 113,
        },
        "deployment_binding": {
            "provider": "azure",
            "candidate_id": "azure-ncc-h100-v5",
            "sku": "Standard_NCC40ads_H100_v5",
            "region": "uksouth",
            "zone": "1",
            "instance_id": "schema-shape-instance",
            "boot_id_sha256": "3" * 64,
            "image_reference": "example.invalid/image@sha256:" + "4" * 64,
            "protection_mode": "cc-on",
            "pair_id": "schema-shape-pair",
            "cuda_device_index": 0,
            "deployment_manifest_sha256": "5" * 64,
            "pair_manifest_sha256": "6" * 64,
            "hardware_packet_sha256": "7" * 64,
        },
        "decision_sha256": "8" * 64,
        "workload_binding_sha256": "9" * 64,
        "evidence_index_sha256": "a" * 64,
        "signature_algorithm": "Ed25519",
        "signature_base64": base64.b64encode(bytes(64)).decode("ascii"),
    }
    return decision


def build_admission_record(arrival_path, checks, admission_id, session_id,
                           workload_binding, root=ROOT, generated_at=None, *,
                           verifier_id, adapter_kind,
                           adapter_measurement_sha256, verifier_public_key_path,
                           verifier_private_key_path) -> dict:
    """Build and fully validate a decision from independently produced checks."""
    root = Path(root).resolve()
    arrival_path = Path(arrival_path).resolve()
    try:
        relative_arrival = arrival_path.relative_to(root)
    except ValueError as exc:
        raise ValueError("Arrival record is outside the declared root") from exc
    load_arrival_record(arrival_path)
    if set(checks) != set(CHECK_NAMES):
        raise ValueError("Admission requires the exact declared check inventory")
    ordered_checks = {name: deepcopy(checks[name]) for name in CHECK_NAMES}
    blockers = [name for name in CHECK_NAMES
                if ordered_checks[name].get("status") != "passed"]
    admitted = not blockers
    record = {
        "schema_version": ADMISSION_SCHEMA_VERSION,
        "record_type": "admission_decision",
        "admission_id": admission_id,
        "session_id": session_id,
        "generated_at": generated_at or _now(),
        "arrival_record": str(relative_arrival),
        "arrival_record_sha256": _file_sha256(arrival_path),
        "workload_binding": deepcopy(workload_binding),
        "checks": ordered_checks,
        "hardware_attested": admitted,
        "admitted": admitted,
        "blockers": blockers,
        "claim_scope": ("Admission permits engineering workload execution only; it is not a "
                        "privacy or security certificate."),
    }
    return sign_admission_record(
        record,
        verifier_id=verifier_id,
        adapter_kind=adapter_kind,
        adapter_measurement_sha256=adapter_measurement_sha256,
        verifier_public_key_path=verifier_public_key_path,
        verifier_private_key_path=verifier_private_key_path,
        root=root,
    )


def validate_admission_record(record, root=ROOT, verify_artifacts=True) -> dict:
    """Validate decision semantics, all bindings, and optionally the evidence bytes."""
    _schema_validate(record, ADMISSION_SCHEMA_PATH)
    for key in ("admission_id", "session_id"):
        if re.fullmatch(r"[a-f0-9]{64}", record[key]):
            raise ValueError(f"{key} cannot be a fabricated digest-shaped identifier")
    root = Path(root).resolve()
    _verify_admission_authentication(record, root)
    arrival_path = _safe_relative(root, record["arrival_record"])
    arrival = None
    if verify_artifacts:
        if (not arrival_path.is_file()
                or _file_sha256(arrival_path) != record["arrival_record_sha256"]):
            raise ValueError("Arrival record is missing or its digest changed")
        arrival = load_arrival_record(arrival_path)

    nonpassed = [name for name in CHECK_NAMES
                 if record["checks"][name]["status"] != "passed"]
    should_admit = not nonpassed
    if record["blockers"] != nonpassed:
        raise ValueError("Admission blockers must exactly match non-passing checks")
    if record["admitted"] != should_admit or record["hardware_attested"] != should_admit:
        raise ValueError("Admission and hardware-attested flags must equal all-check success")

    generated_at = _as_utc_datetime(record["generated_at"])
    if arrival is not None:
        arrival_at = _as_utc_datetime(arrival["recorded_at"])
        if generated_at < arrival_at:
            raise ValueError("Admission decision predates its arrival inventory")
    else:
        arrival_at = None

    all_evidence = []
    for name in CHECK_NAMES:
        check = record["checks"][name]
        verified_at = _as_utc_datetime(check["verified_at"])
        if verified_at > generated_at:
            raise ValueError(f"Check {name} was verified after the admission decision")
        if arrival_at is not None and verified_at < arrival_at:
            raise ValueError(f"Check {name} predates the arrival inventory")
        roles = {evidence["role"] for evidence in check["evidence"]}
        if check["status"] == "passed":
            missing_roles = REQUIRED_EVIDENCE_ROLES[name] - roles
            if missing_roles:
                raise ValueError(f"Passed check {name} lacks evidence roles: {sorted(missing_roles)}")
        all_evidence.extend(check["evidence"])

    if verify_artifacts:
        for evidence in all_evidence:
            path = _safe_relative(root, evidence["path"])
            if (not path.is_file() or path.stat().st_size != evidence["bytes"]
                    or _file_sha256(path) != evidence["sha256"]):
                raise ValueError(f"Admission evidence is missing or changed: {evidence['path']}")

    if arrival is not None:
        expected_deployment = _deployment_binding(arrival, record["workload_binding"])
        if record["authentication"]["deployment_binding"] != expected_deployment:
            raise PermissionError(
                "signed admission deployment/hardware binding differs from arrival")

    if should_admit:
        if arrival is None:
            raise ValueError("An admitted decision requires verified arrival and evidence bytes")
        gpu = arrival["gpu"]
        if (not gpu["nvidia_smi_present"]
                or any(command["exit_code"] != 0
                       for command in (
                           gpu["identity_query"], gpu["cuda_runtime_query"],
                           gpu["cc_readiness"], gpu["protected_memory"],
                       ))
                or any(gpu[key] is None for key in CUDA_IDENTITY_FIELDS)):
            raise ValueError("An admitted decision contradicts the GPU arrival inventory")
        lock_sha = arrival["environment"]["uv_lock_sha256"]
        if lock_sha is None or lock_sha != record["workload_binding"]["environment_lock_sha256"]:
            raise ValueError("Admission environment lock does not match arrival inventory")
        environment_evidence = [item for item in record["checks"]["environment_lock"]["evidence"]
                                if item["role"] == "environment_lock"]
        if not any(item["sha256"] == lock_sha for item in environment_evidence):
            raise ValueError("Environment-lock evidence does not match the bound lock")
        measurement_sha = record["workload_binding"]["workload_measurement_sha256"]
        measurement_evidence = [
            item for item in record["checks"]["measured_workload_identity"]["evidence"]
            if item["role"] == "workload_measurement"]
        if not any(item["sha256"] == measurement_sha for item in measurement_evidence):
            raise ValueError("Workload-measurement evidence does not match its binding")
        identity_roles = {
            "journal_public_key": "journal_public_key_sha256",
            "deployment_manifest": "deployment_manifest_sha256",
            "hardware_packet": "hardware_packet_sha256",
            "collector_manifest": "collector_manifest_sha256",
            "collector_executable": "collector_executable_sha256",
            "collector_public_key": "collector_public_key_sha256",
            "trusted_store_manifest": "trusted_store_manifest_sha256",
            "trusted_store_protection": (
                "trusted_store_protection_evidence_sha256"),
        }
        if (record["workload_binding"]["collector_public_key_sha256"]
                == record["workload_binding"]["journal_public_key_sha256"]):
            raise ValueError("collector and guest journal signing keys must be distinct")
        for role, binding_key in identity_roles.items():
            evidence_rows = [
                item for item in record["checks"]["measured_workload_identity"]["evidence"]
                if item["role"] == role
            ]
            if not any(item["sha256"] == record["workload_binding"][binding_key]
                       for item in evidence_rows):
                raise ValueError(f"{role} evidence does not match its admission binding")
        runtime_rows = [
            item for item in record["checks"]["environment_lock"]["evidence"]
            if item["role"] == "runtime_environment"
        ]
        if not any(item["sha256"] == record["workload_binding"][
                "runtime_environment_sha256"] for item in runtime_rows):
            raise ValueError(
                "runtime-environment evidence does not match its admission binding")
    return record


def write_admission_record(path, record, root=ROOT):
    """Write a decision only after complete byte-level validation."""
    validate_admission_record(record, root=root, verify_artifacts=True)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n")
    loaded = json.loads(path.read_text())
    validate_admission_record(loaded, root=root, verify_artifacts=True)
    return path


def load_admission_record(path, root=ROOT) -> dict:
    record = json.loads(Path(path).read_text())
    return validate_admission_record(record, root=root, verify_artifacts=True)


def validate_admission_freshness(record: Mapping[str, Any], *, root=ROOT,
                                 policy: AdmissionFreshnessPolicy | None = None,
                                 now: datetime | None = None) -> None:
    """Reject stale/future inventory, checks, or decisions under a strict policy."""
    policy = policy or AdmissionFreshnessPolicy()
    reference = now or datetime.now(timezone.utc)
    if reference.tzinfo is None:
        raise ValueError("admission freshness reference time must be timezone-aware")
    reference = reference.astimezone(timezone.utc)
    future_limit = reference + timedelta(seconds=policy.max_future_skew_seconds)

    arrival_path = _safe_relative(Path(root).resolve(), record["arrival_record"])
    arrival = load_arrival_record(arrival_path)
    timestamps: list[tuple[str, datetime, int]] = [
        ("arrival inventory", _as_utc_datetime(arrival["recorded_at"]),
         policy.max_arrival_age_seconds),
        ("admission decision", _as_utc_datetime(record["generated_at"]),
         policy.max_admission_age_seconds),
    ]
    timestamps.extend(
        (f"admission check {name}",
         _as_utc_datetime(record["checks"][name]["verified_at"]),
         policy.max_check_age_seconds)
        for name in CHECK_NAMES
    )
    for label, timestamp, max_age in timestamps:
        if timestamp > future_limit:
            raise PermissionError(f"{label} is beyond the permitted future clock skew")
        if reference - timestamp > timedelta(seconds=max_age):
            raise PermissionError(f"{label} is stale")


def _require_trusted_authentication(record: Mapping[str, Any],
                                    policy: AdmissionTrustPolicy) -> None:
    authentication = record["authentication"]
    verifier = authentication["verifier"]
    expected_verifier = {
        "verifier_id": policy.verifier_id,
        "adapter_kind": policy.adapter_kind,
        "adapter_measurement_sha256": policy.adapter_measurement_sha256,
    }
    if verifier != expected_verifier:
        raise PermissionError("admission verifier identity differs from the trust anchor")
    if authentication["public_key"]["sha256"] != policy.public_key_sha256:
        raise PermissionError("admission verifier public key differs from the trust anchor")
    if authentication["deployment_binding"] != dict(policy.expected_deployment_binding):
        raise PermissionError("admission deployment binding differs from the trust policy")


def verify_cc_admission_reference(path, root=ROOT, expected_session_id=None,
                                  expected_workload_binding=None, *,
                                  trust_policy=None, freshness_policy=None,
                                  now=None, challenge_nonce_path=None) -> dict:
    """Verify a signed admission as a reference without consuming its nonce.

    This is reserved for the unattested CC-off comparator that is paired to an
    already admitted CC-on deployment. It cannot authorise a CC-on workload.
    """
    record = load_admission_record(path, root=root)
    if not record["admitted"]:
        raise PermissionError(f"Confidential-compute admission blocked: {record['blockers']}")
    if expected_session_id is None:
        raise PermissionError("Confidential-compute admission requires an expected session ID")
    if record["session_id"] != expected_session_id:
        raise PermissionError("Confidential-compute admission session binding mismatch")
    if expected_workload_binding is None:
        raise PermissionError("Confidential-compute admission requires an expected workload binding")
    if record["workload_binding"] != expected_workload_binding:
        raise PermissionError("Confidential-compute admission workload binding mismatch")
    if not isinstance(trust_policy, AdmissionTrustPolicy):
        raise PermissionError("Confidential-compute admission requires an external trust policy")
    _require_trusted_authentication(record, trust_policy)
    if freshness_policy is not None and not isinstance(
            freshness_policy, AdmissionFreshnessPolicy):
        raise TypeError("freshness_policy must be an AdmissionFreshnessPolicy")
    validate_admission_freshness(
        record, root=root, policy=freshness_policy, now=now)
    if challenge_nonce_path is None:
        raise PermissionError("Confidential-compute admission requires its challenge nonce")
    nonce_sha256 = validate_challenge_nonce(challenge_nonce_path)
    if nonce_sha256 != record["workload_binding"]["challenge_nonce_sha256"]:
        raise PermissionError("admission challenge nonce differs from its workload binding")
    return record


def require_cc_admission(path, root=ROOT, expected_session_id=None,
                         expected_workload_binding=None, *, trust_policy=None,
                         freshness_policy=None, now=None,
                         challenge_nonce_path=None, nonce_ledger_path=None) -> dict:
    """Require a fresh, trusted, replay-safe decision before the first CC-on step."""
    record = verify_cc_admission_reference(
        path, root=root, expected_session_id=expected_session_id,
        expected_workload_binding=expected_workload_binding,
        trust_policy=trust_policy, freshness_policy=freshness_policy, now=now,
        challenge_nonce_path=challenge_nonce_path,
    )
    if nonce_ledger_path is None:
        raise PermissionError(
            "Confidential-compute admission requires a nonce replay ledger path")
    nonce_sha256 = record["workload_binding"]["challenge_nonce_sha256"]
    consumed_sha256 = consume_challenge_nonce(
        nonce_path=challenge_nonce_path,
        ledger_path=nonce_ledger_path,
        admission_id=record["admission_id"],
        session_id=record["session_id"],
    )
    if consumed_sha256 != nonce_sha256:  # defensive: helper contract must remain exact.
        raise RuntimeError("nonce replay ledger recorded a different challenge")
    return record
