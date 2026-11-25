"""Digest-only cross-paper secure-compute evidence envelope.

The envelope is an engineering interoperability record.  It binds, but never
copies, a Paper 4 signed exact-three two-view workload aggregate, its
confidential-compute admission decision and lifecycle receipt, and an
LLM-Attacker secure-compute audit artifact.  It also makes the source identity
and phase vocabulary translations explicit.

This module is intentionally standard-library only.  Source repositories remain
responsible for validating their complete native schemas; the checks here cover
the fields needed to join those records without upgrading them into scientific
evidence or a privacy claim.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timezone
import base64
import binascii
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any
import uuid


ENVELOPE_SCHEMA = "cross-paper.secure-compute-evidence.v1"
PROTOCOL_ID = "cross-paper-secure-compute-d0q8-1.0"
CLAIM_STATUS = "UNDETERMINED_ENGINEERING_ONLY"

PAPER4_WORKLOAD_ROLE = "paper4_workload_bundle"
PAPER4_REPETITION_SET_ROLE = "paper4_sealed_repetition_set"
PAPER4_REPETITION_AUTH_ROLE = "paper4_execution_manifest_authentication"
PAPER4_ADMISSION_ROLE = "paper4_admission_bundle"
PAPER4_LIFECYCLE_ROLE = "paper4_lifecycle_receipt"
ATTACK_AUDIT_ROLE = "llm_attacker_audit"
SOURCE_ROLES = (
    PAPER4_REPETITION_SET_ROLE,
    PAPER4_REPETITION_AUTH_ROLE,
    PAPER4_ADMISSION_ROLE,
    PAPER4_LIFECYCLE_ROLE,
    ATTACK_AUDIT_ROLE,
)
ROLE_REPOSITORY = {
    PAPER4_REPETITION_SET_ROLE: "paper4",
    PAPER4_REPETITION_AUTH_ROLE: "paper4",
    PAPER4_ADMISSION_ROLE: "paper4",
    PAPER4_LIFECYCLE_ROLE: "paper4",
    ATTACK_AUDIT_ROLE: "llm_attacker",
}
ROLE_SCHEMA_FIELD = {
    PAPER4_REPETITION_SET_ROLE: "schema_version",
    PAPER4_REPETITION_AUTH_ROLE: "schema_version",
    PAPER4_ADMISSION_ROLE: "schema_version",
    PAPER4_LIFECYCLE_ROLE: "schema",
    ATTACK_AUDIT_ROLE: "schema",
}
ROLE_SCHEMA_PATTERN = {
    # This is deliberately exact.  The older p4-secure-compute-bundle-* files
    # are local plaintext recorder output and must never be accepted as
    # attested cross-paper evidence.
    PAPER4_REPETITION_SET_ROLE: re.compile(
        r"^p4-cc-sealed-repetition-set-0\.1$"
    ),
    PAPER4_REPETITION_AUTH_ROLE: re.compile(
        r"^p4-cc-execution-manifest-auth-0\.1$"
    ),
    PAPER4_ADMISSION_ROLE: re.compile(r"^p4-cc-admission-0\.3$"),
    PAPER4_LIFECYCLE_ROLE: re.compile(r"^p4\.cc_lifecycle_receipt\.v2$"),
    ATTACK_AUDIT_ROLE: re.compile(
        r"^dtraining\.attacker\.secure_compute_audit\.v[0-9]+$"),
}

REQUIRED_VIEWS = ("external_attacker", "trusted_diagnostic")
CANONICAL_PHASES = (
    "ingress",
    "tokenisation",
    "forward",
    "return",
    "backward",
    "update",
    "optimizer",
    "cpu-gpu-staging",
    "ipc",
    "vram",
    "checkpointing",
    "logging",
    "crash-restart",
    "egress",
)
IDENTITY_FIELD_MAP = {
    PAPER4_WORKLOAD_ROLE: {
        "session_id": "session_id",
        "event_identity": "not_embedded_digest_only",
        "journal_identity": "sha256_and_event_count",
    },
    ATTACK_AUDIT_ROLE: {
        "request_id": "request_id",
        "session_id": "session_id",
        "step_id": "step_id",
    },
}

_PHASE_TRANSLATIONS = {
    PAPER4_WORKLOAD_ROLE: {
        phase: ([phase], "exact", "The attested workload and canonical phase are identical.")
        for phase in CANONICAL_PHASES
    },
    ATTACK_AUDIT_ROLE: {
        "forward": (["forward"], "exact", "The source and canonical phase are identical."),
        "backward": (["backward"], "exact", "The source and canonical phase are identical."),
        "update": (["update"], "exact", "The source and canonical phase are identical."),
        "cpu-gpu-staging": (["cpu-gpu-staging"], "exact", "The source and canonical phase are identical."),
        "ipc": (["ipc"], "exact", "The source and canonical phase are identical."),
        "vram": (["vram"], "exact", "The source and canonical phase are identical."),
        "logs": (["logging"], "renamed", "The attacker uses the plural phase name logs."),
        "checkpoints": (["checkpointing"], "renamed", "The attacker uses the plural phase name checkpoints."),
        "dumps": ([], "source_only", "Memory dumps have no exact canonical phase and remain explicitly unmatched."),
        "environment": ([], "source_only", "Environment leakage has no exact canonical phase and remains explicitly unmatched."),
        "telemetry": ([], "source_only", "Telemetry has no exact canonical phase and remains explicitly unmatched."),
    },
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.:-]+$")

ADMISSION_CHECKS = (
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
ADMISSION_BINDING_KEYS = {
    "source_tree_sha256",
    "config_sha256",
    "environment_lock_sha256",
    "runtime_environment_sha256",
    "trusted_store_manifest_sha256",
    "trusted_store_protection_evidence_sha256",
    "workload_measurement_sha256",
    "challenge_nonce_sha256",
    "journal_public_key_sha256",
    "deployment_manifest_sha256",
    "hardware_packet_sha256",
    "collector_manifest_sha256",
    "collector_executable_sha256",
    "collector_public_key_sha256",
}
DEPLOYMENT_BINDING_KEYS = {
    "provider",
    "candidate_id",
    "sku",
    "region",
    "zone",
    "instance_id",
    "boot_id_sha256",
    "image_reference",
    "protection_mode",
    "pair_id",
    "cuda_device_index",
    "deployment_manifest_sha256",
    "pair_manifest_sha256",
    "hardware_packet_sha256",
}
REQUIRED_PAPER4_ARTIFACT_COUNTS = {
    "correctness_comparison": 1,
    "execution_summary": 1,
    "step_metrics": 1,
    "checkpoint": 12,
    "source_snapshot": 1,
    "active_security_gate": 1,
}
SHARED_BINDING_KEYS = {
    "paper4_source_tree_sha256",
    "paper4_config_sha256",
    "paper4_admission_sha256",
    "paper4_session_id",
    "paper4_journal_public_key_sha256",
    "paper4_environment_lock_sha256",
    "paper4_runtime_environment_sha256",
    "paper4_trusted_store_manifest_sha256",
    "paper4_trusted_store_protection_evidence_sha256",
    "paper4_workload_measurement_sha256",
    "paper4_challenge_nonce_sha256",
    "paper4_hardware_packet_sha256",
    "paper4_collector_manifest_sha256",
    "paper4_collector_executable_sha256",
    "paper4_collector_public_key_sha256",
    "paper4_admission_verifier_id",
    "paper4_admission_adapter_measurement_sha256",
    "paper4_admission_public_key_sha256",
    "paper4_admission_deployment_binding_sha256",
    "paper4_admission_signature_sha256",
    "paper4_lifecycle_receipt_sha256",
    "paper4_event_pair_chain_sha256",
    "llm_attack_plan_sha256",
    "deployment_manifest_sha256",
}


class CrossPaperEvidenceError(ValueError):
    """The envelope or one of its bound sources failed closed."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise CrossPaperEvidenceError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream, object_pairs_hook=_reject_duplicate_keys)
    except (OSError, json.JSONDecodeError) as exc:
        raise CrossPaperEvidenceError(f"{path}: invalid or missing JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise CrossPaperEvidenceError(f"{path}: expected a JSON object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise CrossPaperEvidenceError(f"{path}: cannot hash source: {exc}") from exc
    return digest.hexdigest()


def _canonical_digest(value: Mapping[str, Any]) -> str:
    body = {key: item for key, item in value.items() if key != "envelope_sha256"}
    try:
        encoded = json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CrossPaperEvidenceError(f"envelope is not canonical JSON: {exc}") from exc
    return hashlib.sha256(encoded).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CrossPaperEvidenceError(f"value is not canonical JSON: {exc}") from exc


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _decode_ed25519_signature(value: Any, label: str) -> bytes:
    if not isinstance(value, str):
        raise CrossPaperEvidenceError(f"{label}: signature must be base64 text")
    try:
        signature = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise CrossPaperEvidenceError(f"{label}: signature is not canonical base64") from exc
    if len(signature) != 64:
        raise CrossPaperEvidenceError(f"{label}: expected a 64-byte Ed25519 signature")
    return signature


def _attacker_plan_digest(plan: Mapping[str, Any]) -> str:
    """Reproduce LLM-Attacker's v1 plan digest without importing that repo."""
    body = {key: deepcopy(item) for key, item in plan.items() if key != "sha256"}
    cases = body.get("cases")
    if not isinstance(cases, list):
        raise CrossPaperEvidenceError("attacker plan cases are missing")
    normalized_cases = []
    for index, case in enumerate(cases):
        if not isinstance(case, Mapping):
            raise CrossPaperEvidenceError(f"attacker plan case {index} is malformed")
        normalized_cases.append({**case, "plan_sha256": None})
    body["cases"] = normalized_cases
    try:
        encoded = json.dumps(
            body, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CrossPaperEvidenceError(f"attacker plan is not canonical JSON: {exc}") from exc
    return hashlib.sha256(encoded).hexdigest()


def _expect_keys(value: Any, keys: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        got = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise CrossPaperEvidenceError(f"{label}: keys {got!r} != {sorted(keys)!r}")
    return value


def _expect_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise CrossPaperEvidenceError(f"{label}: expected lowercase SHA-256")
    return value


def _expect_identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise CrossPaperEvidenceError(f"{label}: invalid identifier")
    return value


def _safe_source_path(root: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise CrossPaperEvidenceError(f"{label}: path must be a nonempty POSIX-relative string")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts or "." in candidate.parts:
        raise CrossPaperEvidenceError(f"{label}: unsafe repository-relative path")
    root = root.resolve()
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise CrossPaperEvidenceError(f"{label}: source path escapes repository root") from exc
    return resolved


def _safe_bound_path(root: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise CrossPaperEvidenceError(f"{label}: path must be nonempty POSIX-relative text")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts or "." in candidate.parts:
        raise CrossPaperEvidenceError(f"{label}: unsafe relative path")
    root = root.resolve()
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise CrossPaperEvidenceError(f"{label}: path escapes execution root") from exc
    if not resolved.is_file() or resolved.is_symlink():
        raise CrossPaperEvidenceError(f"{label}: referenced file is missing or a symlink")
    return resolved


def _safe_receipt_path(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise CrossPaperEvidenceError(f"{label}: path is missing")
    raw = Path(value)
    resolved = (raw if raw.is_absolute() else root / raw).resolve()
    root = root.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise CrossPaperEvidenceError(f"{label}: path escapes Paper 4 root") from exc
    if not resolved.is_file() or resolved.is_symlink():
        raise CrossPaperEvidenceError(f"{label}: referenced file is missing or a symlink")
    return resolved


def _require_file_binding(
    path: Path, *, expected_sha256: Any, expected_bytes: Any, label: str
) -> None:
    digest = _expect_sha256(expected_sha256, label)
    if type(expected_bytes) is not int or expected_bytes < 1:
        raise CrossPaperEvidenceError(f"{label}: invalid byte count")
    if path.stat().st_size != expected_bytes or _sha256_file(path) != digest:
        raise CrossPaperEvidenceError(f"{label}: referenced bytes changed")


def _verify_ed25519(
    public_key_path: Path, payload: bytes, signature: bytes, label: str
) -> None:
    with tempfile.TemporaryDirectory(prefix="pcr-cross-paper-signature-") as temporary:
        temporary_root = Path(temporary)
        payload_path = temporary_root / "payload.bin"
        signature_path = temporary_root / "signature.bin"
        payload_path.write_bytes(payload)
        signature_path.write_bytes(signature)
        try:
            algorithm = subprocess.run(
                [
                    "openssl", "pkey", "-pubin", "-in", str(public_key_path),
                    "-text_pub", "-noout",
                ],
                capture_output=True,
                check=False,
                timeout=15,
            )
            result = subprocess.run(
                [
                    "openssl", "pkeyutl", "-verify", "-pubin", "-inkey",
                    str(public_key_path), "-rawin", "-in", str(payload_path),
                    "-sigfile", str(signature_path),
                ],
                capture_output=True,
                check=False,
                timeout=15,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise CrossPaperEvidenceError(f"{label}: signature verifier unavailable") from exc
    first = algorithm.stdout.splitlines()[0] if algorithm.stdout else b""
    if algorithm.returncode != 0 or first.strip() != b"ED25519 Public-Key:":
        raise CrossPaperEvidenceError(f"{label}: public key is not Ed25519")
    if result.returncode != 0:
        raise CrossPaperEvidenceError(f"{label}: Ed25519 signature verification failed")


def _admission_evidence_by_role(
    admission: Mapping[str, Any], paper4_root: Path
) -> dict[str, tuple[Mapping[str, Any], Path]]:
    records: dict[str, tuple[Mapping[str, Any], Path]] = {}
    for name in ADMISSION_CHECKS:
        for index, item in enumerate(admission["checks"][name]["evidence"]):
            if not isinstance(item, Mapping) or set(item) != {
                "path", "sha256", "bytes", "role", "media_type"
            }:
                raise CrossPaperEvidenceError(
                    f"Paper 4 admission evidence fields differ: {name}[{index}]"
                )
            path = _safe_source_path(
                paper4_root, item["path"], f"Paper 4 admission evidence {item.get('role')}"
            )
            _require_file_binding(
                path,
                expected_sha256=item["sha256"],
                expected_bytes=item["bytes"],
                label=f"Paper 4 admission evidence {item.get('role')}",
            )
            role = item["role"]
            if role in records and role in {
                "journal_public_key", "collector_manifest",
                "collector_executable", "collector_public_key",
            } and records[role][0] != item:
                raise CrossPaperEvidenceError(
                    f"Paper 4 admission has ambiguous evidence role {role}"
                )
            records.setdefault(role, (item, path))
    return records


def make_source_reference(
    role: str,
    path: str,
    repository_revision: str,
    repository_roots: Mapping[str, str | Path],
) -> dict[str, Any]:
    """Create one byte/schema/revision-bound source reference."""
    if role not in SOURCE_ROLES:
        raise CrossPaperEvidenceError(f"unknown source role {role!r}")
    repository = ROLE_REPOSITORY[role]
    if repository not in repository_roots:
        raise CrossPaperEvidenceError(f"missing repository root for {repository}")
    if not isinstance(repository_revision, str) or not _REVISION.fullmatch(
        repository_revision
    ):
        raise CrossPaperEvidenceError("repository_revision must be a 40- or 64-hex revision")
    source_path = _safe_source_path(
        Path(repository_roots[repository]), path, f"source {role}"
    )
    document = _load_json(source_path)
    schema_field = ROLE_SCHEMA_FIELD[role]
    source_schema = document.get(schema_field)
    if not isinstance(source_schema, str) or not ROLE_SCHEMA_PATTERN[role].fullmatch(
        source_schema
    ):
        raise CrossPaperEvidenceError(
            f"source {role}: unsupported {schema_field} {source_schema!r}"
        )
    return {
        "source_id": role.replace("_", "-"),
        "role": role,
        "repository": repository,
        "repository_revision": repository_revision,
        "path": path,
        "sha256": _sha256_file(source_path),
        "schema_field": schema_field,
        "source_schema": source_schema,
    }


def _validate_source_references(
    references: Any,
    repository_roots: Mapping[str, str | Path],
    *,
    verify_sources: bool,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, dict[str, Any]]]:
    if not isinstance(references, list) or len(references) != len(SOURCE_ROLES):
        raise CrossPaperEvidenceError(
            f"sources must contain exactly the {len(SOURCE_ROLES)} required roles"
        )
    by_role: dict[str, Mapping[str, Any]] = {}
    documents: dict[str, dict[str, Any]] = {}
    source_ids: set[str] = set()
    paths: set[tuple[str, str]] = set()
    digests: set[str] = set()
    keys = {
        "source_id",
        "role",
        "repository",
        "repository_revision",
        "path",
        "sha256",
        "schema_field",
        "source_schema",
    }
    for index, reference in enumerate(references):
        reference = _expect_keys(reference, keys, f"sources[{index}]")
        role = reference["role"]
        if role not in SOURCE_ROLES:
            raise CrossPaperEvidenceError(f"sources[{index}]: unknown role {role!r}")
        if role in by_role:
            raise CrossPaperEvidenceError(f"duplicate source role {role}")
        source_id = _expect_identifier(reference["source_id"], f"sources[{index}].source_id")
        if source_id in source_ids:
            raise CrossPaperEvidenceError(f"duplicate source_id {source_id}")
        source_ids.add(source_id)
        repository = reference["repository"]
        if repository != ROLE_REPOSITORY[role]:
            raise CrossPaperEvidenceError(f"source {role}: repository mismatch")
        revision = reference["repository_revision"]
        if not isinstance(revision, str) or not _REVISION.fullmatch(revision):
            raise CrossPaperEvidenceError(f"source {role}: invalid repository revision")
        if reference["schema_field"] != ROLE_SCHEMA_FIELD[role]:
            raise CrossPaperEvidenceError(f"source {role}: schema field mismatch")
        schema = reference["source_schema"]
        if not isinstance(schema, str) or not ROLE_SCHEMA_PATTERN[role].fullmatch(schema):
            raise CrossPaperEvidenceError(f"source {role}: unsupported source schema")
        digest = _expect_sha256(reference["sha256"], f"source {role}.sha256")
        path_key = (repository, reference["path"])
        if path_key in paths:
            raise CrossPaperEvidenceError("duplicate source path")
        if digest in digests:
            raise CrossPaperEvidenceError("duplicate source digest")
        paths.add(path_key)
        digests.add(digest)
        if verify_sources:
            if repository not in repository_roots:
                raise CrossPaperEvidenceError(f"missing repository root for {repository}")
            path = _safe_source_path(
                Path(repository_roots[repository]), reference["path"], f"source {role}"
            )
            document = _load_json(path)
            if _sha256_file(path) != digest:
                raise CrossPaperEvidenceError(f"source {role}: SHA-256 mismatch")
            if document.get(reference["schema_field"]) != schema:
                raise CrossPaperEvidenceError(f"source {role}: exact schema/version mismatch")
            documents[role] = document
        by_role[role] = reference
    if set(by_role) != set(SOURCE_ROLES):
        raise CrossPaperEvidenceError("required source role missing")
    paper4_revisions = {
        by_role[role]["repository_revision"]
        for role in (
            PAPER4_REPETITION_SET_ROLE,
            PAPER4_REPETITION_AUTH_ROLE,
            PAPER4_ADMISSION_ROLE,
            PAPER4_LIFECYCLE_ROLE,
        )
    }
    if len(paper4_revisions) != 1:
        raise CrossPaperEvidenceError("Paper 4 sources use different repository revisions")
    return by_role, documents


def _admission_authentication_facts(
    admission: Mapping[str, Any], admission_bindings: Mapping[str, Any],
    paper4_root: Path,
) -> tuple[dict[str, Any], dict[str, Path]]:
    checks = admission.get("checks")
    if not isinstance(checks, Mapping) or set(checks) != set(ADMISSION_CHECKS):
        raise CrossPaperEvidenceError("Paper 4 admission check inventory differs")
    evidence_index = []
    for name in ADMISSION_CHECKS:
        check = checks[name]
        if not isinstance(check, Mapping) or check.get("status") != "passed":
            raise CrossPaperEvidenceError(f"Paper 4 admission check did not pass: {name}")
        evidence = check.get("evidence")
        if not isinstance(evidence, list):
            raise CrossPaperEvidenceError(f"Paper 4 admission check evidence differs: {name}")
        evidence_index.append({"check": name, "evidence": deepcopy(evidence)})
    evidence_by_role = _admission_evidence_by_role(admission, paper4_root)

    authentication = _expect_keys(
        admission.get("authentication"),
        {
            "schema",
            "verifier",
            "public_key",
            "deployment_binding",
            "decision_sha256",
            "workload_binding_sha256",
            "evidence_index_sha256",
            "signature_algorithm",
            "signature_base64",
        },
        "Paper 4 admission authentication",
    )
    if authentication["schema"] != "p4.cc_admission_authentication.v1":
        raise CrossPaperEvidenceError("Paper 4 admission authentication schema differs")
    if authentication["signature_algorithm"] != "Ed25519":
        raise CrossPaperEvidenceError("Paper 4 admission signature algorithm differs")

    verifier = _expect_keys(
        authentication["verifier"],
        {"verifier_id", "adapter_kind", "adapter_measurement_sha256"},
        "Paper 4 admission verifier",
    )
    verifier_id = _expect_identifier(verifier["verifier_id"], "Paper 4 verifier_id")
    if _SHA256.fullmatch(verifier_id):
        raise CrossPaperEvidenceError("Paper 4 verifier_id cannot be digest-shaped")
    if verifier["adapter_kind"] != "vendor_attestation_adapter":
        raise CrossPaperEvidenceError(
            "Paper 4 cross-paper evidence requires a vendor attestation adapter"
        )
    adapter_measurement = _expect_sha256(
        verifier["adapter_measurement_sha256"], "Paper 4 verifier measurement"
    )

    public_key = _expect_keys(
        authentication["public_key"],
        {"path", "sha256", "bytes"},
        "Paper 4 admission public key",
    )
    if not isinstance(public_key["path"], str) or not public_key["path"]:
        raise CrossPaperEvidenceError("Paper 4 admission public-key path differs")
    public_key_sha = _expect_sha256(
        public_key["sha256"], "Paper 4 admission public key"
    )
    if type(public_key["bytes"]) is not int or public_key["bytes"] < 1:
        raise CrossPaperEvidenceError("Paper 4 admission public-key size differs")
    admission_public_key_path = _safe_source_path(
        paper4_root, public_key["path"], "Paper 4 admission public key"
    )
    _require_file_binding(
        admission_public_key_path,
        expected_sha256=public_key_sha,
        expected_bytes=public_key["bytes"],
        label="Paper 4 admission public key",
    )

    deployment = _expect_keys(
        authentication["deployment_binding"],
        DEPLOYMENT_BINDING_KEYS,
        "Paper 4 signed deployment binding",
    )
    for key in (
        "boot_id_sha256",
        "deployment_manifest_sha256",
        "pair_manifest_sha256",
        "hardware_packet_sha256",
    ):
        _expect_sha256(deployment[key], f"Paper 4 deployment binding {key}")
    for key in ("candidate_id", "pair_id"):
        _expect_identifier(deployment[key], f"Paper 4 deployment binding {key}")
    if deployment["provider"] not in {"azure", "gcp"}:
        raise CrossPaperEvidenceError("Paper 4 deployment provider differs")
    if deployment["protection_mode"] != "cc-on":
        raise CrossPaperEvidenceError("Paper 4 signed deployment is not CC-on")
    if type(deployment["cuda_device_index"]) is not int or deployment[
        "cuda_device_index"
    ] < 0:
        raise CrossPaperEvidenceError("Paper 4 CUDA device binding differs")
    if deployment["deployment_manifest_sha256"] != admission_bindings[
        "deployment_manifest_sha256"
    ] or deployment["hardware_packet_sha256"] != admission_bindings[
        "hardware_packet_sha256"
    ]:
        raise CrossPaperEvidenceError(
            "Paper 4 signed deployment binding differs from workload admission"
        )

    decision = deepcopy(dict(admission))
    decision.pop("authentication", None)
    decision_sha = _expect_sha256(
        authentication["decision_sha256"], "Paper 4 signed decision"
    )
    workload_binding_sha = _expect_sha256(
        authentication["workload_binding_sha256"], "Paper 4 signed workload binding"
    )
    evidence_index_sha = _expect_sha256(
        authentication["evidence_index_sha256"], "Paper 4 signed evidence index"
    )
    if decision_sha != _canonical_sha256(decision):
        raise CrossPaperEvidenceError("Paper 4 signed decision digest differs")
    if workload_binding_sha != _canonical_sha256(admission_bindings):
        raise CrossPaperEvidenceError("Paper 4 signed workload-binding digest differs")
    if evidence_index_sha != _canonical_sha256(evidence_index):
        raise CrossPaperEvidenceError("Paper 4 signed evidence-index digest differs")
    signature = _decode_ed25519_signature(
        authentication["signature_base64"], "Paper 4 admission"
    )
    metadata = deepcopy(dict(authentication))
    metadata.pop("signature_base64")
    signature_payload = _canonical_json_bytes({
        "domain": "p4.cc_admission.signed_decision.v1",
        "decision": decision,
        "authentication": metadata,
    })
    _verify_ed25519(
        admission_public_key_path, signature_payload, signature, "Paper 4 admission"
    )
    required_roles = {
        "journal_public_key",
        "collector_manifest",
        "collector_executable",
        "collector_public_key",
    }
    if not required_roles.issubset(evidence_by_role):
        raise CrossPaperEvidenceError(
            "Paper 4 admission lacks journal/collector byte evidence"
        )
    paths = {role: evidence_by_role[role][1] for role in required_roles}
    facts = {
        "schema": authentication["schema"],
        "verifier_id": verifier_id,
        "adapter_kind": verifier["adapter_kind"],
        "adapter_measurement_sha256": adapter_measurement,
        "public_key_sha256": public_key_sha,
        "decision_sha256": decision_sha,
        "workload_binding_sha256": workload_binding_sha,
        "evidence_index_sha256": evidence_index_sha,
        "deployment_binding_sha256": _canonical_sha256(deployment),
        "signature_algorithm": "Ed25519",
        "signature_sha256": hashlib.sha256(signature).hexdigest(),
    }
    return facts, paths


def _paper4_lifecycle_facts(
    lifecycle: Mapping[str, Any],
    lifecycle_source_sha256: str,
    admission_bindings: Mapping[str, Any],
    artifacts_by_role: Mapping[str, list[Mapping[str, Any]]],
    paper4_root: Path,
) -> dict[str, Any]:
    lifecycle = _expect_keys(
        lifecycle,
        {
            "schema",
            "run_id",
            "arm_id",
            "packet_sha256",
            "deployment_manifest_sha256",
            "lifecycle_launch_sha256",
            "campaign_root_sha256",
            "pair_declaration_sha256",
            "runtime_binding_sha256",
            "active_security_manifest_sha256",
            "resource_identity_sha256",
            "launch_id",
            "launch_issued_at",
            "launch_expires_at",
            "launch_context",
            "lifecycle_record",
            "campaign_ledger",
            "workload_outputs",
            "resource_absence",
            "exit_code",
            "completed_at",
            "receipt_sha256",
        },
        "Paper 4 lifecycle receipt",
    )
    if lifecycle["schema"] != "p4.cc_lifecycle_receipt.v2":
        raise CrossPaperEvidenceError("Paper 4 lifecycle receipt schema differs")
    body = {key: deepcopy(value) for key, value in lifecycle.items() if key != "receipt_sha256"}
    receipt_sha = _expect_sha256(
        lifecycle["receipt_sha256"], "Paper 4 lifecycle receipt digest"
    )
    if receipt_sha != _canonical_sha256(body):
        raise CrossPaperEvidenceError("Paper 4 lifecycle receipt digest differs")
    run_id = _expect_identifier(lifecycle["run_id"], "Paper 4 lifecycle run_id")
    try:
        uuid.UUID(run_id)
    except ValueError as exc:
        raise CrossPaperEvidenceError("Paper 4 lifecycle run_id is not a UUID") from exc
    if lifecycle["arm_id"] != "cc-on" or lifecycle["exit_code"] != 0:
        raise CrossPaperEvidenceError("Paper 4 lifecycle did not complete the CC-on arm")
    packet_sha = _expect_sha256(lifecycle["packet_sha256"], "Paper 4 lifecycle packet")
    deployment_sha = _expect_sha256(
        lifecycle["deployment_manifest_sha256"], "Paper 4 lifecycle deployment"
    )
    resource_sha = _expect_sha256(
        lifecycle["resource_identity_sha256"], "Paper 4 lifecycle resource identity"
    )
    lifecycle_launch_sha = _expect_sha256(
        lifecycle["lifecycle_launch_sha256"], "Paper 4 lifecycle launch"
    )
    campaign_root_sha = _expect_sha256(
        lifecycle["campaign_root_sha256"], "Paper 4 lifecycle campaign root"
    )
    pair_declaration_sha = _expect_sha256(
        lifecycle["pair_declaration_sha256"], "Paper 4 pair declaration"
    )
    runtime_binding_sha = _expect_sha256(
        lifecycle["runtime_binding_sha256"], "Paper 4 runtime binding"
    )
    active_security_sha = _expect_sha256(
        lifecycle["active_security_manifest_sha256"],
        "Paper 4 active-security manifest",
    )
    if artifacts_by_role.get("active_security_gate", [{}])[0].get("sha256") \
            != active_security_sha:
        raise CrossPaperEvidenceError(
            "Paper 4 lifecycle/active-security artifact binding mismatch")
    try:
        uuid.UUID(lifecycle["launch_id"])
        launch_issued = datetime.fromisoformat(
            lifecycle["launch_issued_at"].replace("Z", "+00:00")
        ).astimezone(timezone.utc)
        launch_expires = datetime.fromisoformat(
            lifecycle["launch_expires_at"].replace("Z", "+00:00")
        ).astimezone(timezone.utc)
    except (TypeError, ValueError, AttributeError) as exc:
        raise CrossPaperEvidenceError(
            "Paper 4 lifecycle launch window differs") from exc
    if (not lifecycle["launch_issued_at"].endswith("Z")
            or not lifecycle["launch_expires_at"].endswith("Z")
            or launch_expires <= launch_issued
            or (launch_expires - launch_issued).total_seconds() > 8 * 3600):
        raise CrossPaperEvidenceError("Paper 4 lifecycle launch window differs")
    if packet_sha != admission_bindings["hardware_packet_sha256"] or deployment_sha != \
            admission_bindings["deployment_manifest_sha256"]:
        raise CrossPaperEvidenceError("Paper 4 lifecycle/admission binding mismatch")

    launch = _expect_keys(
        lifecycle["launch_context"], {"path", "sha256"}, "Paper 4 launch context"
    )
    launch_sha = _expect_sha256(launch["sha256"], "Paper 4 launch context")
    launch_path = _safe_receipt_path(
        paper4_root, launch["path"], "Paper 4 launch context"
    )
    if _sha256_file(launch_path) != launch_sha:
        raise CrossPaperEvidenceError("Paper 4 launch-context bytes differ")
    record = _expect_keys(
        lifecycle["lifecycle_record"],
        {"path", "sha256", "chain_head_sha256"},
        "Paper 4 lifecycle record",
    )
    record_sha = _expect_sha256(record["sha256"], "Paper 4 lifecycle record")
    chain_sha = _expect_sha256(
        record["chain_head_sha256"], "Paper 4 lifecycle chain head"
    )
    record_path = _safe_receipt_path(
        paper4_root, record["path"], "Paper 4 lifecycle record"
    )
    lifecycle_rows = _read_jsonl(record_path, "Paper 4 lifecycle record")
    lifecycle_start = lifecycle_rows[0].get("details")
    if (not isinstance(lifecycle_start, Mapping)
            or lifecycle_start.get("lifecycle_launch_sha256")
            != lifecycle_launch_sha
            or lifecycle_start.get("campaign_root_sha256") != campaign_root_sha
            or lifecycle_start.get("pair_declaration_sha256")
            != pair_declaration_sha
            or lifecycle_start.get("runtime_binding_sha256")
            != runtime_binding_sha
            or lifecycle_start.get("active_security_manifest_sha256")
            != active_security_sha
            or lifecycle_start.get("launch_id") != lifecycle["launch_id"]):
        raise CrossPaperEvidenceError(
            "Paper 4 lifecycle start/receipt binding differs")
    previous = "0" * 64
    events = []
    for sequence, event in enumerate(lifecycle_rows):
        if (event.get("schema") != "p4.cc_lifecycle_event.v1"
                or event.get("sequence") != sequence
                or event.get("run_id") != run_id
                or event.get("previous_sha256") != previous):
            raise CrossPaperEvidenceError("Paper 4 lifecycle event identity differs")
        body = {key: value for key, value in event.items() if key != "event_sha256"}
        if event.get("event_sha256") != _canonical_sha256(body):
            raise CrossPaperEvidenceError("Paper 4 lifecycle event chain differs")
        previous = event["event_sha256"]
        events.append(event.get("event"))
    required_events = (
        "lifecycle_started", "teardown_started", "teardown_finished",
        "absence_verification_started", "absence_verification_finished",
        "lifecycle_finished",
    )
    if any(events.count(name) != 1 for name in required_events) \
            or events[0] != "lifecycle_started" or events[-1] != "lifecycle_finished":
        raise CrossPaperEvidenceError("Paper 4 lifecycle record is incomplete")
    if (_sha256_file(record_path) != record_sha or previous != chain_sha
            or lifecycle_rows[-1].get("details", {}).get("exit_code") != 0):
        raise CrossPaperEvidenceError("Paper 4 lifecycle record receipt binding differs")
    ledger = _expect_keys(
        lifecycle["campaign_ledger"],
        {"path", "file_sha256_at_completion", "state_sha256", "entry_sha256"},
        "Paper 4 campaign ledger",
    )
    for key in ("file_sha256_at_completion", "state_sha256", "entry_sha256"):
        _expect_sha256(ledger[key], f"Paper 4 campaign ledger {key}")
    ledger_path = _safe_receipt_path(
        paper4_root, ledger["path"], "Paper 4 campaign ledger"
    )
    ledger_document = _load_json(ledger_path)
    ledger_body = {
        key: value for key, value in ledger_document.items() if key != "ledger_sha256"
    }
    if ledger_document.get("ledger_sha256") != _canonical_sha256(ledger_body):
        raise CrossPaperEvidenceError("Paper 4 campaign ledger digest differs")
    if (ledger_document.get("campaign_root_sha256") != campaign_root_sha
            or ledger_document.get("pair_declaration_sha256")
            != pair_declaration_sha):
        raise CrossPaperEvidenceError("Paper 4 campaign ledger root binding differs")
    previous_entry = "0" * 64
    matched_entry = False
    entries = ledger_document.get("entries")
    if not isinstance(entries, list):
        raise CrossPaperEvidenceError("Paper 4 campaign ledger entries differ")
    for sequence, entry in enumerate(entries):
        entry_body = {key: value for key, value in entry.items() if key != "entry_sha256"}
        entry_sha = _canonical_sha256(entry_body)
        if (entry.get("sequence") != sequence
                or entry.get("previous_entry_sha256") != previous_entry
                or entry.get("entry_sha256") != entry_sha):
            raise CrossPaperEvidenceError("Paper 4 campaign ledger chain differs")
        previous_entry = entry_sha
        if entry_sha == ledger["entry_sha256"]:
            receipt_absence = lifecycle.get("resource_absence")
            matched_entry = (
                entry.get("run_id") == run_id
                and entry.get("lifecycle_record_sha256") == record_sha
                and entry.get("arm_id") == "cc-on"
                and entry.get("exit_code") == 0
                and entry.get("lifecycle_launch_sha256")
                == lifecycle_launch_sha
                and isinstance(receipt_absence, Mapping)
                and entry.get("absence_artifact_sha256")
                == receipt_absence.get("sha256")
            )
    if not matched_entry:
        raise CrossPaperEvidenceError("Paper 4 lifecycle ledger entry is missing")

    outputs = _expect_keys(
        lifecycle["workload_outputs"],
        {"status", "artifacts"},
        "Paper 4 lifecycle workload outputs",
    )
    if outputs["status"] != "captured":
        raise CrossPaperEvidenceError("Paper 4 lifecycle did not capture workload outputs")
    output_artifacts = _expect_keys(
        outputs["artifacts"],
        {"execution_summary.json", "artifact_manifest.json"},
        "Paper 4 lifecycle workload artifact bindings",
    )
    output_digests: dict[str, str] = {}
    for name, record_value in output_artifacts.items():
        binding = _expect_keys(
            record_value, {"path", "sha256", "bytes"}, f"Paper 4 lifecycle {name}"
        )
        output_digests[name] = _expect_sha256(
            binding["sha256"], f"Paper 4 lifecycle {name}"
        )
        if type(binding["bytes"]) is not int or binding["bytes"] < 1:
            raise CrossPaperEvidenceError(f"Paper 4 lifecycle {name} size differs")
        artifact_path = _safe_receipt_path(
            paper4_root, binding["path"], f"Paper 4 lifecycle {name}"
        )
        _require_file_binding(
            artifact_path,
            expected_sha256=binding["sha256"],
            expected_bytes=binding["bytes"],
            label=f"Paper 4 lifecycle {name}",
        )
    if output_digests["execution_summary.json"] != artifacts_by_role[
        "execution_summary"
    ][0]["sha256"]:
        raise CrossPaperEvidenceError(
            "Paper 4 lifecycle/workload execution-summary binding mismatch"
        )

    absence = _expect_keys(
        lifecycle["resource_absence"],
        {"path", "sha256", "bytes", "provider_response_sha256"},
        "Paper 4 lifecycle resource absence",
    )
    absence_sha = _expect_sha256(absence["sha256"], "Paper 4 resource absence")
    provider_response_sha = _expect_sha256(
        absence["provider_response_sha256"], "Paper 4 absence provider response"
    )
    if type(absence["bytes"]) is not int or absence["bytes"] < 1:
        raise CrossPaperEvidenceError("Paper 4 resource-absence size differs")
    absence_path = _safe_receipt_path(
        paper4_root, absence["path"], "Paper 4 resource absence"
    )
    _require_file_binding(
        absence_path,
        expected_sha256=absence_sha,
        expected_bytes=absence["bytes"],
        label="Paper 4 resource absence",
    )
    absence_document = _load_json(absence_path)
    response = absence_document.get("provider_response")
    if (absence_document.get("schema")
            != "p4.cc_resource_absence_verification.v2"
            or absence_document.get("packet_sha256") != packet_sha
            or absence_document.get("deployment_manifest_sha256") != deployment_sha
            or absence_document.get("lifecycle_launch_sha256")
            != lifecycle_launch_sha
            or absence_document.get("launch_id") != lifecycle["launch_id"]
            or absence_document.get("resource_identity_sha256") != resource_sha
            or absence_document.get("absent") is not True
            or absence_document.get("observed_state") != "absent"
            or not isinstance(response, Mapping)
            or response.get("sha256") != provider_response_sha):
        raise CrossPaperEvidenceError("Paper 4 resource-absence semantics differ")
    response_path = _safe_receipt_path(
        paper4_root, response.get("path"), "Paper 4 raw provider response")
    _require_file_binding(
        response_path, expected_sha256=provider_response_sha,
        expected_bytes=response.get("bytes"),
        label="Paper 4 raw provider response",
    )
    try:
        checked_at = datetime.fromisoformat(
            str(absence_document["checked_at"]).replace("Z", "+00:00")
        ).astimezone(timezone.utc)
        completed_at = datetime.fromisoformat(
            str(lifecycle["completed_at"]).replace("Z", "+00:00")
        ).astimezone(timezone.utc)
    except (KeyError, ValueError) as exc:
        raise CrossPaperEvidenceError(
            "Paper 4 lifecycle/absence timestamps differ"
        ) from exc
    age_seconds = (completed_at - checked_at).total_seconds()
    if age_seconds < 0 or age_seconds > 900:
        raise CrossPaperEvidenceError(
            "Paper 4 resource-absence evidence is future-dated or stale at receipt"
        )

    return {
        "source_sha256": lifecycle_source_sha256,
        "receipt_sha256": receipt_sha,
        "run_id": run_id,
        "arm_id": "cc-on",
        "exit_code": 0,
        "resource_identity_sha256": resource_sha,
        "lifecycle_launch_sha256": lifecycle_launch_sha,
        "campaign_root_sha256": campaign_root_sha,
        "pair_declaration_sha256": pair_declaration_sha,
        "runtime_binding_sha256": runtime_binding_sha,
        "active_security_manifest_sha256": active_security_sha,
        "launch_context_sha256": launch_sha,
        "lifecycle_record_sha256": record_sha,
        "lifecycle_chain_head_sha256": chain_sha,
        "campaign_ledger_state_sha256": ledger["state_sha256"],
        "campaign_ledger_entry_sha256": ledger["entry_sha256"],
        "execution_summary_sha256": output_digests["execution_summary.json"],
        "artifact_manifest_sha256": output_digests["artifact_manifest.json"],
        "resource_absence_sha256": absence_sha,
        "provider_response_sha256": provider_response_sha,
    }


def _verify_signed_json_record(
    path: Path,
    *,
    record_schema: str,
    expected_payload: Mapping[str, Any],
    public_key_path: Path,
    label: str,
) -> dict[str, Any]:
    record = _expect_keys(
        _load_json(path),
        {"schema_version", "payload", "payload_sha256", "signature_base64"},
        label,
    )
    if record["schema_version"] != record_schema or record["payload"] != expected_payload:
        raise CrossPaperEvidenceError(f"{label}: signed payload binding differs")
    payload_bytes = _canonical_json_bytes(expected_payload)
    if record["payload_sha256"] != hashlib.sha256(payload_bytes).hexdigest():
        raise CrossPaperEvidenceError(f"{label}: payload digest differs")
    signature = _decode_ed25519_signature(record["signature_base64"], label)
    _verify_ed25519(public_key_path, payload_bytes, signature, label)
    return record


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise CrossPaperEvidenceError(f"{label}: cannot read JSONL") from exc
    if not lines or any(not line for line in lines):
        raise CrossPaperEvidenceError(f"{label}: empty or blank JSONL row")
    rows = []
    for number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
        except (json.JSONDecodeError, CrossPaperEvidenceError) as exc:
            raise CrossPaperEvidenceError(f"{label}: invalid row {number}") from exc
        if not isinstance(row, dict):
            raise CrossPaperEvidenceError(f"{label}: row {number} is not an object")
        rows.append(row)
    return rows


def _verify_active_security_join(
    *,
    manifest_path: Path,
    attacker_artifact: Mapping[str, Any],
    attacker_source_sha256: str,
    attacker_revision: str,
    admission: Mapping[str, Any],
    admission_source_sha256: str,
    lifecycle: Mapping[str, Any],
) -> None:
    manifest = _load_json(manifest_path)
    if (manifest.get("schema") != "p4-cc-active-security-manifest-0.1"
            or manifest.get("scientific_evidence") is not False):
        raise CrossPaperEvidenceError("Paper 4 active-security manifest differs")
    inputs = manifest.get("inputs")
    outcome = manifest.get("outcome")
    if not isinstance(inputs, Mapping) or not isinstance(outcome, Mapping) \
            or outcome.get("status") != "PASS":
        raise CrossPaperEvidenceError("Paper 4 active-security gate is not PASS")
    for key in (
        "invalid_secret_releases", "external_canary_exposures",
        "accepted_mutation_cells", "fail_cells", "unassessed_cells",
    ):
        if outcome.get(key) != 0:
            raise CrossPaperEvidenceError(
                f"Paper 4 active-security counter is nonzero: {key}"
            )
    plan = attacker_artifact.get("plan")
    if (not isinstance(plan, Mapping)
            or inputs.get("attacker_repository_revision") != attacker_revision
            or inputs.get("plan_sha256") != plan.get("sha256")
            or inputs.get("deployment_manifest_sha256")
            != admission["workload_binding"]["deployment_manifest_sha256"]):
        raise CrossPaperEvidenceError("Paper 4 active-security plan/source binding differs")
    runtime = inputs.get("runtime_binding")
    expected_runtime = {
        "schema": "p4-cc-active-runtime-binding-0.1",
        "deployment_manifest_sha256": admission["workload_binding"][
            "deployment_manifest_sha256"
        ],
        "admission_id": admission["admission_id"],
        "admission_sha256": admission_source_sha256,
        "session_id": admission["session_id"],
        "challenge_nonce_sha256": admission["workload_binding"][
            "challenge_nonce_sha256"
        ],
        "arrival_record_sha256": admission["arrival_record_sha256"],
        "instance_id": admission["authentication"]["deployment_binding"]["instance_id"],
        "protection_mode": "cc-on",
        "hardware_packet_sha256": admission["workload_binding"][
            "hardware_packet_sha256"
        ],
        "workload_config_sha256": admission["workload_binding"]["config_sha256"],
        "source_tree_sha256": admission["workload_binding"]["source_tree_sha256"],
    }
    if (runtime != expected_runtime
            or inputs.get("runtime_binding_sha256")
            != _canonical_sha256(expected_runtime)
            or inputs.get("runtime_binding_file_sha256")
            != lifecycle.get("runtime_binding_sha256")
            or _sha256_file(manifest_path)
            != lifecycle.get("active_security_manifest_sha256")):
        raise CrossPaperEvidenceError("Paper 4 active-security runtime binding differs")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise CrossPaperEvidenceError("Paper 4 active-security artifacts are missing")
    by_role = {}
    for index, record in enumerate(artifacts):
        record = _expect_keys(
            record, {"path", "sha256", "bytes", "role"},
            f"Paper 4 active-security artifact {index}",
        )
        path = _safe_bound_path(
            manifest_path.parent, record["path"],
            f"Paper 4 active-security artifact {index}",
        )
        _require_file_binding(
            path, expected_sha256=record["sha256"], expected_bytes=record["bytes"],
            label=f"Paper 4 active-security artifact {index}",
        )
        if record["role"] in by_role:
            raise CrossPaperEvidenceError("Paper 4 active-security artifact role duplicated")
        by_role[record["role"]] = (record, path)
    if "attacker_artifact" not in by_role or "paper4_active_security_result" not in by_role:
        raise CrossPaperEvidenceError("Paper 4 active-security join artifacts are missing")
    attack_record, attack_path = by_role["attacker_artifact"]
    if (attack_record["sha256"] != attacker_source_sha256
            or _load_json(attack_path) != attacker_artifact):
        raise CrossPaperEvidenceError(
            "Paper 4 active-security attacker artifact differs from cross-paper source"
        )
    result_record, result_path = by_role["paper4_active_security_result"]
    result = _load_json(result_path)
    bindings = result.get("bindings")
    if (result.get("schema") != "p4-cc-active-security-result-0.1"
            or result.get("status") != "PASS"
            or not isinstance(bindings, Mapping)
            or bindings.get("attacker_artifact_sha256") != attack_record["sha256"]
            or bindings.get("frozen_plan_sha256") != plan["sha256"]
            or bindings.get("deployment_manifest_sha256")
            != inputs["deployment_manifest_sha256"]
            or result_record["sha256"] != _sha256_file(result_path)):
        raise CrossPaperEvidenceError("Paper 4 active-security result binding differs")


def _verify_repetition_runtime_bytes(
    *,
    bundle: Mapping[str, Any],
    repetition: int,
    output_root: Path,
    execution_summary: Mapping[str, Any],
    admission_paths: Mapping[str, Path],
) -> None:
    repetitions = execution_summary.get("repetitions")
    if not isinstance(repetitions, list) or len(repetitions) != 3:
        raise CrossPaperEvidenceError("Paper 4 execution summary lacks three repetitions")
    row = repetitions[repetition]
    if row.get("repetition") != repetition or row.get("collector_journal_id") != \
            f"repetition-{repetition:02d}":
        raise CrossPaperEvidenceError("Paper 4 execution repetition identity differs")
    transfer = row.get("control_transfer_summary")
    if (not isinstance(transfer, Mapping)
            or transfer.get("schema_version") != "p4-cc-control-transfer-trace-0.1"
            or type(transfer.get("event_count")) is not int
            or transfer["event_count"] < 1
            or transfer.get("raw_tensors_persisted") is not False
            or not isinstance(transfer.get("totals_by_purpose"), Mapping)
            or not {"checkpoint-restore", "checkpoint-snapshot"}.issubset(
                transfer["totals_by_purpose"]
            )):
        raise CrossPaperEvidenceError(
            "Paper 4 control-transfer summary is incomplete"
        )
    paths = {
        key: _safe_bound_path(output_root, row.get(key), f"Paper 4 repetition {repetition} {key}")
        for key in (
            "trusted_journal", "trusted_journal_auth", "external_journal",
            "external_journal_auth", "metrics",
        )
    }
    journals = bundle["journals"]
    trusted_rows = _read_jsonl(paths["trusted_journal"], "Paper 4 trusted journal")
    external_rows = _read_jsonl(paths["external_journal"], "Paper 4 external journal")
    if len(trusted_rows) != journals["trusted_diagnostic"]["events"] \
            or len(external_rows) != journals["external_attacker"]["events"]:
        raise CrossPaperEvidenceError("Paper 4 journal event count differs from bundle")
    if _sha256_file(paths["trusted_journal"]) != journals["trusted_diagnostic"]["sha256"] \
            or _sha256_file(paths["external_journal"]) != journals["external_attacker"]["sha256"]:
        raise CrossPaperEvidenceError("Paper 4 journal bytes differ from bundle")
    pair_keys = ("correlation_id", "sequence", "session_id", "request_id", "step_id", "phase", "operation")
    for sequence, (trusted, external) in enumerate(zip(trusted_rows, external_rows)):
        if trusted.get("sequence") != sequence or external.get("sequence") != sequence \
                or any(trusted.get(key) != external.get(key) for key in pair_keys):
            raise CrossPaperEvidenceError("Paper 4 paired journal identity differs")

    admission = bundle["admission"]
    trusted_payload = {
        "schema_version": "p4-cc-journal-auth-payload-0.1",
        "session_id": bundle["session_id"],
        "journal_sha256": journals["trusted_diagnostic"]["sha256"],
        "chain_head_sha256": trusted_rows[-1]["event_sha256"],
        "event_count": len(trusted_rows),
        "admission_sha256": admission["admission_sha256"],
        "workload_config_sha256": admission["workload_config_sha256"],
        "public_key_sha256": admission["journal_public_key_sha256"],
        "signature_algorithm": "Ed25519",
    }
    trusted_record = _verify_signed_json_record(
        paths["trusted_journal_auth"],
        record_schema="p4-cc-journal-auth-0.1",
        expected_payload=trusted_payload,
        public_key_path=admission_paths["journal_public_key"],
        label="Paper 4 trusted journal authentication",
    )
    if _sha256_file(paths["trusted_journal_auth"]) != journals[
        "trusted_authentication"
    ]["sha256"] or trusted_record["payload_sha256"] != journals[
        "trusted_authentication"
    ]["payload_sha256"]:
        raise CrossPaperEvidenceError("Paper 4 trusted authentication bytes differ")

    external_payload = {
        "schema_version": "p4-cc-external-journal-auth-payload-0.1",
        "session_id": bundle["session_id"],
        "journal_id": f"repetition-{repetition:02d}",
        "external_journal_sha256": journals["external_attacker"]["sha256"],
        "event_count": len(external_rows),
        "admission_sha256": admission["admission_sha256"],
        "workload_config_sha256": admission["workload_config_sha256"],
        "collector_manifest_sha256": admission["collector_manifest_sha256"],
        "collector_executable_sha256": admission["collector_executable_sha256"],
        "collector_public_key_sha256": admission["collector_public_key_sha256"],
        "signature_algorithm": "Ed25519",
    }
    external_record = _verify_signed_json_record(
        paths["external_journal_auth"],
        record_schema="p4-cc-external-journal-auth-0.1",
        expected_payload=external_payload,
        public_key_path=admission_paths["collector_public_key"],
        label="Paper 4 external journal authentication",
    )
    if _sha256_file(paths["external_journal_auth"]) != journals[
        "external_authentication"
    ]["sha256"] or external_record["payload_sha256"] != journals[
        "external_authentication"
    ]["payload_sha256"]:
        raise CrossPaperEvidenceError("Paper 4 external authentication bytes differ")


def _paper4_source_facts(
    workload: Mapping[str, Any],
    admission: Mapping[str, Any],
    lifecycle: Mapping[str, Any],
    admission_source_sha256: str,
    lifecycle_source_sha256: str,
    paper4_root: Path,
) -> tuple[list[dict[str, Any]], set[str], dict[str, Any]]:
    if workload.get("schema_version") != "p4-cc-workload-bundle-0.1":
        raise CrossPaperEvidenceError(
            "Paper 4 source is not the attested p4-cc-workload-bundle-0.1 format"
        )
    if workload.get("protocol_id") != PROTOCOL_ID:
        raise CrossPaperEvidenceError("Paper 4 workload protocol mismatch")
    if workload.get("cohort") != "engineering" or workload.get(
        "scientific_evidence"
    ) is not False:
        raise CrossPaperEvidenceError("Paper 4 workload must remain engineering-only")
    if workload.get("privacy_outcome") != "undetermined":
        raise CrossPaperEvidenceError("Paper 4 workload made a privacy outcome claim")
    if workload.get("protection_mode") != "cc-on":
        raise CrossPaperEvidenceError(
            "attested cross-paper evidence requires the CC-on workload; "
            "CC-off/local recorder output is comparator evidence only"
        )
    session_id = _expect_identifier(workload.get("session_id"), "Paper 4 session_id")
    if workload.get("hardware_attested") is not True:
        raise CrossPaperEvidenceError("Paper 4 CC-on workload lacks admitted hardware")

    bundle_admission = workload.get("admission")
    bundle_admission_keys = {
        "admission_id",
        "admission_sha256",
        "session_id",
        "hardware_attested",
        "workload_config_sha256",
        "journal_public_key_sha256",
        "collector_manifest_sha256",
        "collector_executable_sha256",
        "collector_public_key_sha256",
    }
    bundle_admission = _expect_keys(
        bundle_admission, bundle_admission_keys, "Paper 4 bundle admission"
    )
    _expect_identifier(bundle_admission["admission_id"], "Paper 4 admission_id")
    for key in (
        "admission_sha256",
        "workload_config_sha256",
        "journal_public_key_sha256",
        "collector_manifest_sha256",
        "collector_executable_sha256",
        "collector_public_key_sha256",
    ):
        _expect_sha256(bundle_admission[key], f"Paper 4 bundle admission {key}")
    if bundle_admission["admission_sha256"] != admission_source_sha256:
        raise CrossPaperEvidenceError(
            "Paper 4 workload does not bind the supplied admission source bytes"
        )
    if bundle_admission["session_id"] != session_id:
        raise CrossPaperEvidenceError("Paper 4 bundle admission/session mismatch")
    if bundle_admission["hardware_attested"] is not True:
        raise CrossPaperEvidenceError("Paper 4 CC-on bundle admission is not attested")

    if admission.get("schema_version") != "p4-cc-admission-0.3" or admission.get(
        "record_type"
    ) != "admission_decision":
        raise CrossPaperEvidenceError("Paper 4 admission source is not an admission decision")
    if admission.get("admission_id") != bundle_admission["admission_id"]:
        raise CrossPaperEvidenceError("Paper 4 admission identity mismatch")
    if admission.get("session_id") != session_id:
        raise CrossPaperEvidenceError("Paper 4 admission/workload session mismatch")
    if admission.get("admitted") is not True or admission.get(
        "hardware_attested"
    ) is not True:
        raise CrossPaperEvidenceError("Paper 4 CC-on workload lacks admitted hardware")

    admission_bindings = admission.get("workload_binding")
    admission_bindings = _expect_keys(
        admission_bindings,
        ADMISSION_BINDING_KEYS,
        "Paper 4 admission workload_binding",
    )
    for key in sorted(ADMISSION_BINDING_KEYS):
        _expect_sha256(admission_bindings.get(key), f"Paper 4 admission {key}")
    if (
        bundle_admission["workload_config_sha256"]
        != admission_bindings["config_sha256"]
    ):
        raise CrossPaperEvidenceError("Paper 4 workload/admission config_sha256 mismatch")
    if (
        bundle_admission["journal_public_key_sha256"]
        != admission_bindings["journal_public_key_sha256"]
    ):
        raise CrossPaperEvidenceError("Paper 4 workload/admission journal key mismatch")
    for key in (
        "collector_manifest_sha256",
        "collector_executable_sha256",
        "collector_public_key_sha256",
    ):
        if bundle_admission[key] != admission_bindings[key]:
            raise CrossPaperEvidenceError(
                f"Paper 4 workload/admission {key} mismatch"
            )
    if admission_bindings["journal_public_key_sha256"] == admission_bindings[
        "collector_public_key_sha256"
    ]:
        raise CrossPaperEvidenceError(
            "Paper 4 trusted and external collector keys are not independent"
        )

    admission_authentication, _ = _admission_authentication_facts(
        admission, admission_bindings, paper4_root
    )

    journals = workload.get("journals")
    journals = _expect_keys(
        journals,
        {
            "trusted_diagnostic",
            "external_attacker",
            "trusted_authentication",
            "external_authentication",
            "collector",
        },
        "Paper 4 journals",
    )
    journal_index: list[dict[str, Any]] = []
    paired_counts: list[int] = []
    for view in REQUIRED_VIEWS:
        record = _expect_keys(journals[view], {"sha256", "events"}, f"Paper 4 {view} journal")
        digest = _expect_sha256(record["sha256"], f"Paper 4 {view} journal")
        if type(record["events"]) is not int or record["events"] < 1:
            raise CrossPaperEvidenceError(f"Paper 4 {view} journal has no events")
        paired_counts.append(record["events"])
        journal_index.append(
            {"view": view, "sha256": digest, "events": record["events"]}
        )
    if len(set(paired_counts)) != 1:
        raise CrossPaperEvidenceError("Paper 4 paired journal event counts differ")
    authentication = _expect_keys(
        journals["trusted_authentication"],
        {"sha256", "payload_sha256", "chain_head_sha256", "signature_algorithm"},
        "Paper 4 trusted journal authentication",
    )
    if authentication["signature_algorithm"] != "Ed25519":
        raise CrossPaperEvidenceError("Paper 4 trusted journal signature algorithm differs")
    for key in ("sha256", "payload_sha256", "chain_head_sha256"):
        _expect_sha256(authentication[key], f"Paper 4 journal authentication {key}")
    journal_index.append(
        {
            "view": "trusted_authentication",
            "sha256": authentication["sha256"],
            "payload_sha256": authentication["payload_sha256"],
            "chain_head_sha256": authentication["chain_head_sha256"],
            "signature_algorithm": "Ed25519",
            "events": None,
        }
    )
    external_authentication = _expect_keys(
        journals["external_authentication"],
        {"sha256", "payload_sha256", "journal_id", "signature_algorithm"},
        "Paper 4 external journal authentication",
    )
    if external_authentication["signature_algorithm"] != "Ed25519":
        raise CrossPaperEvidenceError("Paper 4 external signature algorithm differs")
    for key in ("sha256", "payload_sha256"):
        _expect_sha256(
            external_authentication[key], f"Paper 4 external authentication {key}"
        )
    journal_id = _expect_identifier(
        external_authentication["journal_id"], "Paper 4 collector journal_id"
    )
    journal_index.append(
        {
            "view": "external_authentication",
            "sha256": external_authentication["sha256"],
            "payload_sha256": external_authentication["payload_sha256"],
            "journal_id": journal_id,
            "signature_algorithm": "Ed25519",
            "events": None,
        }
    )
    collector = _expect_keys(
        journals["collector"],
        {
            "collector_id",
            "manifest_sha256",
            "executable_sha256",
            "public_key_sha256",
        },
        "Paper 4 collector",
    )
    collector_id = _expect_identifier(collector["collector_id"], "Paper 4 collector_id")
    if _SHA256.fullmatch(collector_id):
        raise CrossPaperEvidenceError("Paper 4 collector_id cannot be digest-shaped")
    collector_binding_map = {
        "manifest_sha256": "collector_manifest_sha256",
        "executable_sha256": "collector_executable_sha256",
        "public_key_sha256": "collector_public_key_sha256",
    }
    for journal_key, admission_key in collector_binding_map.items():
        value = _expect_sha256(collector[journal_key], f"Paper 4 collector {journal_key}")
        if value != admission_bindings[admission_key]:
            raise CrossPaperEvidenceError(
                f"Paper 4 collector/admission {admission_key} mismatch"
            )

    phase_coverage = workload.get("phase_coverage")
    if not isinstance(phase_coverage, Mapping) or set(phase_coverage) != set(
        CANONICAL_PHASES
    ):
        raise CrossPaperEvidenceError("Paper 4 phase coverage inventory differs")
    phase_index: list[dict[str, Any]] = []
    for phase in CANONICAL_PHASES:
        row = _expect_keys(
            phase_coverage[phase],
            {"status", "event_count", "reason"},
            f"Paper 4 phase {phase}",
        )
        if row["status"] not in {"observed", "unavailable"}:
            raise CrossPaperEvidenceError(f"Paper 4 phase {phase} has invalid status")
        if type(row["event_count"]) is not int or row["event_count"] < 1:
            raise CrossPaperEvidenceError(f"Paper 4 phase {phase} has no paired events")
        if not isinstance(row["reason"], str) or not row["reason"].strip():
            raise CrossPaperEvidenceError(f"Paper 4 phase {phase} lacks a reason")
        phase_index.append(
            {
                "phase": phase,
                "status": row["status"],
                "event_count": row["event_count"],
            }
        )
    unassessed = [
        phase
        for phase in CANONICAL_PHASES
        if phase_coverage[phase]["status"] == "unavailable"
    ]
    if workload.get("unassessed_phases") != unassessed:
        raise CrossPaperEvidenceError("Paper 4 unassessed phase inventory differs")
    expected_collection = "unassessed" if unassessed else "collected-not-a-verdict"
    if workload.get("security_collection_status") != expected_collection:
        raise CrossPaperEvidenceError("Paper 4 collection status contradicts phase coverage")
    event_pair_chain_sha256 = _expect_sha256(
        workload.get("event_pair_chain_sha256"), "Paper 4 event-pair chain"
    )

    artifacts = workload.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise CrossPaperEvidenceError("Paper 4 workload lacks its artifact inventory")
    artifact_fields = {
        "role",
        "path",
        "sha256",
        "bytes",
        "media_type",
        "sensitivity",
    }
    artifact_index: list[dict[str, Any]] = []
    artifacts_by_role: dict[str, list[Mapping[str, Any]]] = {}
    artifact_keys: set[tuple[str, str]] = set()
    roles: set[str] = set()
    for index, artifact in enumerate(artifacts):
        artifact = _expect_keys(artifact, artifact_fields, f"Paper 4 artifact {index}")
        role = artifact["role"]
        path = artifact["path"]
        if not isinstance(role, str) or not role or not isinstance(path, str) or not path:
            raise CrossPaperEvidenceError(f"Paper 4 artifact {index} lacks role/path")
        if (role, path) in artifact_keys:
            raise CrossPaperEvidenceError("duplicate Paper 4 artifact role/path")
        artifact_keys.add((role, path))
        roles.add(role)
        artifacts_by_role.setdefault(role, []).append(artifact)
        artifact_sha = _expect_sha256(
            artifact["sha256"], f"Paper 4 artifact {index}"
        )
        if type(artifact["bytes"]) is not int or artifact["bytes"] < 0:
            raise CrossPaperEvidenceError(f"Paper 4 artifact {index} has invalid size")
        if artifact["sensitivity"] not in {"public-metadata", "trusted-sensitive"}:
            raise CrossPaperEvidenceError(f"Paper 4 artifact {index} sensitivity differs")
        artifact_index.append(
            {
                "role": role,
                "sha256": artifact_sha,
                "bytes": artifact["bytes"],
                "sensitivity": artifact["sensitivity"],
            }
        )
    for role, expected_count in REQUIRED_PAPER4_ARTIFACT_COUNTS.items():
        if len(artifacts_by_role.get(role, [])) != expected_count:
            raise CrossPaperEvidenceError(
                f"Paper 4 artifact count differs for {role}: expected {expected_count}"
            )

    lifecycle_index = _paper4_lifecycle_facts(
        lifecycle,
        lifecycle_source_sha256,
        admission_bindings,
        artifacts_by_role,
        paper4_root,
    )

    facts = {
        "paper4_source_tree_sha256": admission_bindings["source_tree_sha256"],
        "paper4_config_sha256": admission_bindings["config_sha256"],
        "paper4_admission_sha256": admission_source_sha256,
        "paper4_session_id": session_id,
        "paper4_journal_public_key_sha256": admission_bindings[
            "journal_public_key_sha256"
        ],
        "paper4_environment_lock_sha256": admission_bindings[
            "environment_lock_sha256"
        ],
        "paper4_runtime_environment_sha256": admission_bindings[
            "runtime_environment_sha256"
        ],
        "paper4_trusted_store_manifest_sha256": admission_bindings[
            "trusted_store_manifest_sha256"
        ],
        "paper4_trusted_store_protection_evidence_sha256": admission_bindings[
            "trusted_store_protection_evidence_sha256"
        ],
        "paper4_workload_measurement_sha256": admission_bindings[
            "workload_measurement_sha256"
        ],
        "paper4_challenge_nonce_sha256": admission_bindings[
            "challenge_nonce_sha256"
        ],
        "paper4_hardware_packet_sha256": admission_bindings[
            "hardware_packet_sha256"
        ],
        "paper4_collector_manifest_sha256": admission_bindings[
            "collector_manifest_sha256"
        ],
        "paper4_collector_executable_sha256": admission_bindings[
            "collector_executable_sha256"
        ],
        "paper4_collector_public_key_sha256": admission_bindings[
            "collector_public_key_sha256"
        ],
        "paper4_admission_verifier_id": admission_authentication["verifier_id"],
        "paper4_admission_adapter_measurement_sha256": admission_authentication[
            "adapter_measurement_sha256"
        ],
        "paper4_admission_public_key_sha256": admission_authentication[
            "public_key_sha256"
        ],
        "paper4_admission_deployment_binding_sha256": admission_authentication[
            "deployment_binding_sha256"
        ],
        "paper4_admission_signature_sha256": admission_authentication[
            "signature_sha256"
        ],
        "paper4_lifecycle_receipt_sha256": lifecycle_source_sha256,
        "paper4_event_pair_chain_sha256": event_pair_chain_sha256,
        "paper4_deployment_manifest_sha256": admission_bindings[
            "deployment_manifest_sha256"
        ],
        "paper4_admitted": True,
        "hardware_attested": True,
        "paper4_collection_status": expected_collection,
        "paper4_unassessed_phases": unassessed,
        "paper4_evidence_index": {
            "session_id": session_id,
            "admission_id": bundle_admission["admission_id"],
            "protection_mode": "cc-on",
            "journals": journal_index,
            "collector": {
                "collector_id": collector_id,
                "manifest_sha256": collector["manifest_sha256"],
                "executable_sha256": collector["executable_sha256"],
                "public_key_sha256": collector["public_key_sha256"],
            },
            "admission_authentication": admission_authentication,
            "lifecycle": lifecycle_index,
            "artifacts": artifact_index,
            "phases": phase_index,
        },
    }
    # Request/step rows deliberately stay in the signed journals.  Their bytes
    # are not copied into the cross-paper envelope.
    return [], set(CANONICAL_PHASES), facts


def _paper4_aggregate_facts(
    documents: Mapping[str, Mapping[str, Any]],
    references: Mapping[str, Mapping[str, Any]],
    repository_roots: Mapping[str, str | Path],
) -> tuple[list[dict[str, Any]], set[str], dict[str, Any]]:
    paper4_root = Path(repository_roots["paper4"]).resolve()
    manifest = documents[PAPER4_REPETITION_SET_ROLE]
    manifest_fields = {
        "schema_version", "protocol_id", "session_id", "pair_id",
        "protection_mode", "scientific_evidence", "admission_sha256",
        "workload_config_sha256", "hardware_packet_sha256",
        "artifact_manifest_sha256", "correctness_comparison_sha256",
        "source_snapshot_sha256", "active_security_manifest_sha256",
        "repetition_count", "repetitions", "claim_scope",
    }
    manifest = _expect_keys(manifest, manifest_fields, "Paper 4 repetition set")
    if (manifest["schema_version"] != "p4-cc-sealed-repetition-set-0.1"
            or manifest["protocol_id"] != PROTOCOL_ID
            or manifest["protection_mode"] != "cc-on"
            or manifest["scientific_evidence"] is not False
            or manifest["repetition_count"] != 3):
        raise CrossPaperEvidenceError("Paper 4 repetition-set identity differs")
    for key in (
        "admission_sha256", "workload_config_sha256", "hardware_packet_sha256",
        "artifact_manifest_sha256", "correctness_comparison_sha256",
        "source_snapshot_sha256", "active_security_manifest_sha256",
    ):
        _expect_sha256(manifest[key], f"Paper 4 repetition set {key}")
    _expect_identifier(manifest["session_id"], "Paper 4 repetition-set session")
    _expect_identifier(manifest["pair_id"], "Paper 4 repetition-set pair")

    manifest_path = _safe_source_path(
        paper4_root,
        references[PAPER4_REPETITION_SET_ROLE]["path"],
        "Paper 4 repetition set",
    )
    output_root = manifest_path.parent.parent.resolve()
    try:
        output_root.relative_to(paper4_root)
    except ValueError as exc:
        raise CrossPaperEvidenceError("Paper 4 execution root escapes repository") from exc

    admission = documents[PAPER4_ADMISSION_ROLE]
    admission_bindings = admission.get("workload_binding")
    if not isinstance(admission_bindings, Mapping):
        raise CrossPaperEvidenceError("Paper 4 admission workload binding is missing")
    _, admission_paths = _admission_authentication_facts(
        admission, admission_bindings, paper4_root
    )
    admission_source_sha = references[PAPER4_ADMISSION_ROLE]["sha256"]
    if (manifest["admission_sha256"] != admission_source_sha
            or manifest["workload_config_sha256"] != admission_bindings["config_sha256"]
            or manifest["hardware_packet_sha256"] != admission_bindings[
                "hardware_packet_sha256"]):
        raise CrossPaperEvidenceError("Paper 4 repetition-set/admission binding mismatch")

    auth_path = _safe_source_path(
        paper4_root,
        references[PAPER4_REPETITION_AUTH_ROLE]["path"],
        "Paper 4 repetition-set authentication",
    )
    execution_payload = {
        "schema_version": "p4-cc-execution-manifest-auth-payload-0.1",
        "session_id": manifest["session_id"],
        "pair_id": manifest["pair_id"],
        "protection_mode": "cc-on",
        "execution_manifest_sha256": references[PAPER4_REPETITION_SET_ROLE]["sha256"],
        "admission_sha256": admission_source_sha,
        "workload_config_sha256": admission_bindings["config_sha256"],
        "hardware_packet_sha256": admission_bindings["hardware_packet_sha256"],
        "public_key_sha256": admission_bindings["journal_public_key_sha256"],
        "signature_algorithm": "Ed25519",
    }
    _verify_signed_json_record(
        auth_path,
        record_schema="p4-cc-execution-manifest-auth-0.1",
        expected_payload=execution_payload,
        public_key_path=admission_paths["journal_public_key"],
        label="Paper 4 repetition-set authentication",
    )

    conventional = {
        "artifact_manifest_sha256": output_root / "artifact_manifest.json",
        "correctness_comparison_sha256": output_root / "correctness_comparison.json",
        "source_snapshot_sha256": output_root / "source_snapshot.json",
    }
    for key, path in conventional.items():
        if not path.is_file() or path.is_symlink() or _sha256_file(path) != manifest[key]:
            raise CrossPaperEvidenceError(f"Paper 4 {key} bytes differ")
    active_root = output_root / "active-security"
    active_candidates = [
        path for path in active_root.iterdir()
        if path.is_file() and not path.is_symlink()
        and _sha256_file(path) == manifest["active_security_manifest_sha256"]
    ] if active_root.is_dir() else []
    if len(active_candidates) != 1:
        raise CrossPaperEvidenceError("Paper 4 active-security manifest bytes differ")
    _verify_active_security_join(
        manifest_path=active_candidates[0],
        attacker_artifact=documents[ATTACK_AUDIT_ROLE],
        attacker_source_sha256=references[ATTACK_AUDIT_ROLE]["sha256"],
        attacker_revision=references[ATTACK_AUDIT_ROLE]["repository_revision"],
        admission=admission,
        admission_source_sha256=admission_source_sha,
        lifecycle=documents[PAPER4_LIFECYCLE_ROLE],
    )

    artifact_manifest = _expect_keys(
        _load_json(conventional["artifact_manifest_sha256"]),
        {"schema_version", "artifacts"},
        "Paper 4 artifact manifest",
    )
    if artifact_manifest["schema_version"] != "p4-cc-seal-artifact-manifest-0.1" \
            or not isinstance(artifact_manifest["artifacts"], list):
        raise CrossPaperEvidenceError("Paper 4 artifact manifest schema differs")
    for index, artifact in enumerate(artifact_manifest["artifacts"]):
        artifact = _expect_keys(
            artifact,
            {"role", "path", "sha256", "bytes", "media_type", "sensitivity"},
            f"Paper 4 artifact manifest row {index}",
        )
        path = _safe_bound_path(
            output_root, artifact["path"], f"Paper 4 artifact {index}"
        )
        _require_file_binding(
            path,
            expected_sha256=artifact["sha256"],
            expected_bytes=artifact["bytes"],
            label=f"Paper 4 artifact {index}",
        )
    execution_summary = _load_json(output_root / "execution_summary.json")
    correctness = _load_json(output_root / "correctness_comparison.json")
    if (correctness.get("schema_version") != "p4-cc-paired-correctness-0.1"
            or correctness.get("status") != "PASS"
            or correctness.get("raw_logits_persisted") is not False):
        raise CrossPaperEvidenceError(
            "Paper 4 correctness artifact is not an exact bound PASS"
        )
    if (execution_summary.get("schema_version")
            != "p4-cc-engineering-execution-0.2"
            or execution_summary.get("correctness_status") != "PASS"
            or execution_summary.get("sealed") is not True):
        raise CrossPaperEvidenceError("Paper 4 execution summary is incomplete")
    source_snapshot = _load_json(output_root / "source_snapshot.json")
    revisions = source_snapshot.get("repository_revisions")
    if not isinstance(revisions, Mapping):
        raise CrossPaperEvidenceError("Paper 4 source snapshot lacks repository revisions")
    expected_revisions = {
        "paper4": references[PAPER4_REPETITION_SET_ROLE]["repository_revision"],
        "llm-attacker": references[ATTACK_AUDIT_ROLE]["repository_revision"],
    }
    for repository, revision in expected_revisions.items():
        if revisions.get(repository) != revision:
            raise CrossPaperEvidenceError(
                f"Paper 4 source snapshot revision differs for {repository}"
            )

    rows = manifest["repetitions"]
    if not isinstance(rows, list) or len(rows) != 3:
        raise CrossPaperEvidenceError("Paper 4 repetition set must contain exactly three rows")
    repetition_facts = []
    common_facts = None
    for index, row in enumerate(rows):
        row = _expect_keys(
            row,
            {"repetition", "journal_id", "path", "sha256", "bytes", "event_pair_chain_sha256"},
            f"Paper 4 repetition row {index}",
        )
        if (row["repetition"] != index
                or row["journal_id"] != f"repetition-{index:02d}"
                or row["path"] != f"sealed/repetition-{index:02d}.bundle.json"):
            raise CrossPaperEvidenceError("Paper 4 repetition row identity differs")
        bundle_path = _safe_bound_path(output_root, row["path"], f"Paper 4 bundle {index}")
        _require_file_binding(
            bundle_path,
            expected_sha256=row["sha256"],
            expected_bytes=row["bytes"],
            label=f"Paper 4 bundle {index}",
        )
        bundle = _load_json(bundle_path)
        _, phases, facts = _paper4_source_facts(
            bundle,
            admission,
            documents[PAPER4_LIFECYCLE_ROLE],
            admission_source_sha,
            references[PAPER4_LIFECYCLE_ROLE]["sha256"],
            paper4_root,
        )
        lifecycle_binding = facts["paper4_evidence_index"]["lifecycle"]
        if (lifecycle_binding["artifact_manifest_sha256"]
                != manifest["artifact_manifest_sha256"]
                or lifecycle_binding["execution_summary_sha256"]
                != _sha256_file(output_root / "execution_summary.json")):
            raise CrossPaperEvidenceError(
                "Paper 4 lifecycle/repetition-set output binding mismatch"
            )
        if facts["paper4_event_pair_chain_sha256"] != row["event_pair_chain_sha256"]:
            raise CrossPaperEvidenceError("Paper 4 repetition chain binding differs")
        if facts["paper4_evidence_index"]["artifacts"] != [
            {
                "role": item["role"],
                "sha256": item["sha256"],
                "bytes": item["bytes"],
                "sensitivity": item["sensitivity"],
            }
            for item in artifact_manifest["artifacts"]
        ]:
            raise CrossPaperEvidenceError("Paper 4 bundle/artifact manifest differs")
        _verify_repetition_runtime_bytes(
            bundle=bundle,
            repetition=index,
            output_root=output_root,
            execution_summary=execution_summary,
            admission_paths=admission_paths,
        )
        repetition_facts.append({
            "repetition": index,
            "bundle_sha256": row["sha256"],
            "event_pair_chain_sha256": row["event_pair_chain_sha256"],
            "journals": facts["paper4_evidence_index"]["journals"],
        })
        comparison = {
            key: value for key, value in facts.items()
            if key not in {"paper4_event_pair_chain_sha256", "paper4_evidence_index"}
        }
        if common_facts is None:
            common_facts = comparison
            base_facts = facts
        elif comparison != common_facts:
            raise CrossPaperEvidenceError("Paper 4 repetition bindings differ")
    assert common_facts is not None
    base_facts["paper4_event_pair_chain_sha256"] = _canonical_sha256(
        [row["event_pair_chain_sha256"] for row in rows]
    )
    base_facts["paper4_evidence_index"]["repetitions"] = repetition_facts
    base_facts["paper4_evidence_index"].pop("journals", None)
    return [], phases, base_facts


def _attacker_source_facts(
    artifact: Mapping[str, Any], repository_revision: str
) -> tuple[list[dict[str, Any]], set[str], dict[str, Any]]:
    config = artifact.get("config")
    plan = artifact.get("plan")
    results = artifact.get("results")
    gate = artifact.get("gate")
    provenance = artifact.get("provenance")
    if not all(isinstance(value, Mapping) for value in (config, plan, gate, provenance)):
        raise CrossPaperEvidenceError("attacker artifact lacks config/plan/gate/provenance")
    if not isinstance(results, list):
        raise CrossPaperEvidenceError("attacker artifact results must be a list")
    if config.get("toy") is not False or config.get("evidence_bundle_verified") is not True:
        raise CrossPaperEvidenceError("attacker source must be a non-toy, evidence-verified run")
    if provenance.get("dtraining_commit") != repository_revision:
        raise CrossPaperEvidenceError("attacker artifact/repository revision mismatch")
    if plan.get("schema") != "llm_attacker.secure_compute_plan.v1":
        raise CrossPaperEvidenceError("attacker plan schema mismatch")
    if gate.get("schema") != "llm_attacker.secure_compute_gate.v1":
        raise CrossPaperEvidenceError("attacker gate schema mismatch")
    plan_sha = _expect_sha256(plan.get("sha256"), "attacker plan.sha256")
    if _attacker_plan_digest(plan) != plan_sha:
        raise CrossPaperEvidenceError("attacker plan digest does not bind its contents")
    deployment_sha = _expect_sha256(
        plan.get("deployment_manifest_sha256"), "attacker deployment manifest"
    )
    if config.get("plan_sha256") != plan_sha:
        raise CrossPaperEvidenceError("attacker config/plan digest mismatch")
    if config.get("deployment_manifest_sha256") != deployment_sha:
        raise CrossPaperEvidenceError("attacker config/deployment digest mismatch")

    cases = plan.get("cases")
    cells = gate.get("cells")
    if not isinstance(cases, list) or not cases or not isinstance(cells, list):
        raise CrossPaperEvidenceError("attacker plan/gate cells are missing")

    def unique_by(items: list[Any], field: str, label: str) -> dict[str, Mapping[str, Any]]:
        indexed: dict[str, Mapping[str, Any]] = {}
        for index, item in enumerate(items):
            if not isinstance(item, Mapping) or not isinstance(item.get(field), str):
                raise CrossPaperEvidenceError(f"{label}[{index}] lacks {field}")
            key = item[field]
            if key in indexed:
                raise CrossPaperEvidenceError(f"duplicate attacker {label} {key}")
            indexed[key] = item
        return indexed

    case_by_id = unique_by(cases, "case_id", "case")
    result_by_id = unique_by(results, "case_id", "result")
    cell_by_id = unique_by(cells, "case_id", "gate cell")
    if set(case_by_id) != set(result_by_id) or set(case_by_id) != set(cell_by_id):
        raise CrossPaperEvidenceError("attacker cases/results/gate cells are missing or extra")
    for key in ("missing_case_ids", "extra_case_ids", "duplicate_case_ids"):
        if not isinstance(gate.get(key), list) or gate[key]:
            raise CrossPaperEvidenceError(
                "attacker gate reports missing, extra, or duplicate cases"
            )

    identities: list[dict[str, Any]] = []
    native_identities: set[tuple[str, str, int]] = set()
    phases: set[str] = set()
    for case_id, case in case_by_id.items():
        result = result_by_id[case_id]
        cell = cell_by_id[case_id]
        if case.get("schema") != "llm_attacker.secure_compute_case.v1":
            raise CrossPaperEvidenceError(f"attacker case schema mismatch: {case_id}")
        if result.get("schema") != "llm_attacker.secure_compute_result.v1":
            raise CrossPaperEvidenceError(f"attacker result schema mismatch: {case_id}")
        if case.get("plan_sha256") != plan_sha:
            raise CrossPaperEvidenceError(f"attacker case plan binding mismatch: {case_id}")
        payload = case.get("payload")
        if not isinstance(payload, Mapping) or payload.get(
            "deployment_manifest_sha256"
        ) != deployment_sha:
            raise CrossPaperEvidenceError(
                f"attacker case deployment binding mismatch: {case_id}"
            )
        identity = (case.get("request_id"), case.get("session_id"), case.get("step_id"))
        if (
            not isinstance(identity[0], str)
            or not identity[0]
            or not isinstance(identity[1], str)
            or not identity[1]
            or type(identity[2]) is not int
            or identity[2] < 0
        ):
            raise CrossPaperEvidenceError(f"attacker case {case_id} has invalid identity")
        if identity != (
            result.get("request_id"),
            result.get("session_id"),
            result.get("step_id"),
        ):
            raise CrossPaperEvidenceError(f"attacker case/result identity mismatch: {case_id}")
        if identity in native_identities:
            raise CrossPaperEvidenceError("duplicate attacker request/session/step identity")
        native_identities.add(identity)
        if result.get("plan_sha256") != plan_sha or result.get(
            "deployment_manifest_sha256"
        ) != deployment_sha:
            raise CrossPaperEvidenceError(f"attacker result binding mismatch: {case_id}")
        _expect_sha256(
            result.get("evidence_manifest_sha256"),
            f"attacker result evidence manifest: {case_id}",
        )
        if any(
            cell.get(key) != case.get(key)
            for key in ("category", "arm", "view")
        ):
            raise CrossPaperEvidenceError(f"attacker gate coordinate mismatch: {case_id}")
        if cell.get("verdict") not in ("PASS", "FAIL", "UNASSESSED"):
            raise CrossPaperEvidenceError(f"attacker gate verdict invalid: {case_id}")
        if result.get("status") not in {"observed", "unavailable", "error"}:
            raise CrossPaperEvidenceError(f"attacker result status invalid: {case_id}")
        if result.get("status") == "unavailable" and cell.get("verdict") not in {
            "FAIL",
            "UNASSESSED",
        }:
            raise CrossPaperEvidenceError(
                f"unavailable attacker result was credited as PASS: {case_id}"
            )
        if cell.get("verdict") == "UNASSESSED" and (
            result.get("status") != "unavailable" or case.get("required") is not False
        ):
            raise CrossPaperEvidenceError(
                f"UNASSESSED attacker cell is not an optional unavailable source: {case_id}"
            )
        if case.get("category") == "boundary-canary":
            phase = case.get("arm")
            if not isinstance(phase, str) or not phase:
                raise CrossPaperEvidenceError("attacker boundary case has no phase")
            phases.add(phase)
        identities.append(
            {
                "source_role": ATTACK_AUDIT_ROLE,
                "source_event_id": case_id,
                "source_request_id": identity[0],
                "source_session_id": identity[1],
                "source_step_field": "step_id",
                "source_step": identity[2],
                "canonical_request_id": identity[0],
                "canonical_session_id": identity[1],
                "canonical_step_id": identity[2],
            }
        )
    views = {case.get("view") for case in cases}
    if not set(REQUIRED_VIEWS).issubset(views):
        raise CrossPaperEvidenceError("attacker plan lacks both required views")
    counts = {
        verdict: sum(cell.get("verdict") == verdict for cell in cells)
        for verdict in ("PASS", "FAIL", "UNASSESSED")
    }
    if gate.get("counts") != counts:
        raise CrossPaperEvidenceError("attacker gate counts do not match cells")
    overall = "FAIL" if counts["FAIL"] else (
        "UNASSESSED" if counts["UNASSESSED"] else "PASS"
    )
    if gate.get("overall") != overall:
        raise CrossPaperEvidenceError("attacker overall gate does not match cells")
    facts = {
        "llm_attack_plan_sha256": plan_sha,
        "deployment_manifest_sha256": deployment_sha,
        "llm_attacker_gate": overall,
        "unassessed_case_ids": sorted(
            cell["case_id"] for cell in cells if cell["verdict"] == "UNASSESSED"
        ),
    }
    return identities, phases, facts


def _source_facts(
    documents: Mapping[str, Mapping[str, Any]],
    references: Mapping[str, Mapping[str, Any]],
    repository_roots: Mapping[str, str | Path],
) -> tuple[list[dict[str, Any]], dict[str, set[str]], dict[str, Any]]:
    paper4_identities, paper4_phases, paper4 = _paper4_aggregate_facts(
        documents, references, repository_roots
    )
    attacker_identities, attacker_phases, attacker = _attacker_source_facts(
        documents[ATTACK_AUDIT_ROLE],
        references[ATTACK_AUDIT_ROLE]["repository_revision"],
    )
    if (
        paper4["paper4_deployment_manifest_sha256"]
        != attacker["deployment_manifest_sha256"]
    ):
        raise CrossPaperEvidenceError(
            "Paper 4 admission/LLM-Attacker deployment manifest mismatch"
        )
    facts = {**paper4, **attacker}
    facts["overall"] = (
        "FAIL"
        if attacker["llm_attacker_gate"] == "FAIL"
        else "UNASSESSED"
        if attacker["llm_attacker_gate"] == "UNASSESSED"
        or not paper4["paper4_admitted"]
        or bool(paper4["paper4_unassessed_phases"])
        else "PASS"
    )
    return (
        paper4_identities + attacker_identities,
        {
            PAPER4_WORKLOAD_ROLE: paper4_phases,
            ATTACK_AUDIT_ROLE: attacker_phases,
        },
        facts,
    )


def _phase_crosswalks(source_phases: Mapping[str, set[str]]) -> list[dict[str, Any]]:
    rows = []
    for role in (PAPER4_WORKLOAD_ROLE, ATTACK_AUDIT_ROLE):
        for phase in sorted(source_phases[role]):
            canonical, relation, rationale = _PHASE_TRANSLATIONS[role].get(
                phase,
                (
                    [],
                    "source_only",
                    "This source phase has no reviewed canonical translation.",
                ),
            )
            rows.append(
                {
                    "source_role": role,
                    "source_phase": phase,
                    "canonical_phases": list(canonical),
                    "relation": relation,
                    "rationale": rationale,
                }
            )
    return rows


def _assessment(facts: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "paper4_admitted": facts["paper4_admitted"],
        "hardware_attested": facts["hardware_attested"],
        "paper4_collection_status": facts["paper4_collection_status"],
        "paper4_unassessed_phases": list(facts["paper4_unassessed_phases"]),
        "llm_attacker_gate": facts["llm_attacker_gate"],
        "unassessed_case_ids": list(facts["unassessed_case_ids"]),
        "overall": facts["overall"],
    }


def build_envelope(
    envelope_id: str,
    sources: Sequence[Mapping[str, Any]],
    repository_roots: Mapping[str, str | Path],
) -> dict[str, Any]:
    """Build and revalidate a canonical envelope from frozen source files."""
    _expect_identifier(envelope_id, "envelope_id")
    references, documents = _validate_source_references(
        list(sources), repository_roots, verify_sources=True
    )
    identities, source_phases, facts = _source_facts(
        documents, references, repository_roots
    )
    envelope: dict[str, Any] = {
        "schema": ENVELOPE_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "envelope_id": envelope_id,
        "stage": "engineering",
        "scientific_evidence": False,
        "claim_status": CLAIM_STATUS,
        "sources": [dict(source) for source in sources],
        "shared_bindings": {
            key: facts[key] for key in sorted(SHARED_BINDING_KEYS)
        },
        "paper4_evidence_index": deepcopy(facts["paper4_evidence_index"]),
        "required_views": list(REQUIRED_VIEWS),
        "identity_field_map": deepcopy(IDENTITY_FIELD_MAP),
        "identity_mappings": identities,
        "phase_crosswalks": _phase_crosswalks(source_phases),
        "assessment": _assessment(facts),
    }
    envelope["envelope_sha256"] = _canonical_digest(envelope)
    validate_envelope(envelope, repository_roots)
    return envelope


def validate_envelope(
    envelope: Mapping[str, Any],
    repository_roots: Mapping[str, str | Path],
    *,
    verify_sources: bool = True,
) -> Mapping[str, Any]:
    """Validate envelope structure, source bytes, joins, and derived fields."""
    top_keys = {
        "schema",
        "protocol_id",
        "envelope_id",
        "stage",
        "scientific_evidence",
        "claim_status",
        "sources",
        "shared_bindings",
        "paper4_evidence_index",
        "required_views",
        "identity_field_map",
        "identity_mappings",
        "phase_crosswalks",
        "assessment",
        "envelope_sha256",
    }
    _expect_keys(envelope, top_keys, "envelope")
    if envelope["schema"] != ENVELOPE_SCHEMA or envelope["protocol_id"] != PROTOCOL_ID:
        raise CrossPaperEvidenceError("envelope schema/protocol mismatch")
    _expect_identifier(envelope["envelope_id"], "envelope_id")
    if (
        envelope["stage"] != "engineering"
        or envelope["scientific_evidence"] is not False
        or envelope["claim_status"] != CLAIM_STATUS
    ):
        raise CrossPaperEvidenceError("envelope must remain engineering-only and undetermined")
    if envelope["required_views"] != list(REQUIRED_VIEWS):
        raise CrossPaperEvidenceError("both ordered canonical views are required")
    if envelope["identity_field_map"] != IDENTITY_FIELD_MAP:
        raise CrossPaperEvidenceError("identity field map changed")
    expected_digest = _canonical_digest(envelope)
    if envelope["envelope_sha256"] != expected_digest:
        raise CrossPaperEvidenceError("envelope canonical digest mismatch")
    references, documents = _validate_source_references(
        envelope["sources"], repository_roots, verify_sources=verify_sources
    )
    if not verify_sources:
        return envelope
    identities, source_phases, facts = _source_facts(
        documents, references, repository_roots
    )

    bindings = _expect_keys(
        envelope["shared_bindings"], SHARED_BINDING_KEYS, "shared_bindings"
    )
    expected_bindings = {key: facts[key] for key in SHARED_BINDING_KEYS}
    if bindings != expected_bindings:
        raise CrossPaperEvidenceError("shared digest bindings do not match source records")
    if envelope["paper4_evidence_index"] != facts["paper4_evidence_index"]:
        raise CrossPaperEvidenceError(
            "Paper 4 digest-only evidence index does not match the workload bundle"
        )

    mappings = envelope["identity_mappings"]
    if not isinstance(mappings, list) or mappings != identities:
        raise CrossPaperEvidenceError("identity mapping inventory is missing, duplicate, or misbound")
    mapping_keys = {
        "source_role",
        "source_event_id",
        "source_request_id",
        "source_session_id",
        "source_step_field",
        "source_step",
        "canonical_request_id",
        "canonical_session_id",
        "canonical_step_id",
    }
    seen_source: set[tuple[str, str]] = set()
    for index, mapping in enumerate(mappings):
        mapping = _expect_keys(mapping, mapping_keys, f"identity_mappings[{index}]")
        source_key = (mapping["source_role"], mapping["source_event_id"])
        if source_key in seen_source:
            raise CrossPaperEvidenceError("duplicate identity mapping")
        seen_source.add(source_key)
        if mapping["canonical_request_id"] != mapping["source_request_id"] or mapping[
            "canonical_session_id"
        ] != mapping["source_session_id"] or mapping["canonical_step_id"] != mapping[
            "source_step"
        ]:
            raise CrossPaperEvidenceError("canonical identity silently renames a source identity")

    crosswalks = envelope["phase_crosswalks"]
    expected_crosswalks = _phase_crosswalks(source_phases)
    if not isinstance(crosswalks, list) or crosswalks != expected_crosswalks:
        raise CrossPaperEvidenceError("phase crosswalk is missing, duplicate, or misbound")
    for row in crosswalks:
        canonical = row["canonical_phases"]
        if any(phase not in CANONICAL_PHASES for phase in canonical):
            raise CrossPaperEvidenceError("phase crosswalk names an unknown canonical phase")
        if (row["relation"] == "source_only") != (canonical == []):
            raise CrossPaperEvidenceError("source-only phase relation is inconsistent")

    if envelope["assessment"] != _assessment(facts):
        raise CrossPaperEvidenceError("assessment does not preserve source gate state")
    return envelope


def load_envelope(
    path: str | Path,
    repository_roots: Mapping[str, str | Path],
    *,
    verify_sources: bool = True,
) -> dict[str, Any]:
    """Load an envelope with duplicate-key detection and validate it."""
    value = _load_json(Path(path))
    validate_envelope(value, repository_roots, verify_sources=verify_sources)
    return value


def write_envelope(
    path: str | Path,
    envelope: Mapping[str, Any],
    repository_roots: Mapping[str, str | Path],
) -> Path:
    """Write only after validation, then validate the serialized bytes again."""
    validate_envelope(envelope, repository_roots)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(envelope, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    load_envelope(destination, repository_roots)
    return destination
