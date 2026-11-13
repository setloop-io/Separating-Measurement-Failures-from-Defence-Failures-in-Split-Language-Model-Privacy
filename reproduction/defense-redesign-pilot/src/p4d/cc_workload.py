"""Attested d0-q8 engineering workload and two-view evidence sealing.

Execution is deliberately split in two.  The trusted workload writes only
content digests and metadata to an append-only journal.  A separately collected
external-attacker journal must later be paired event-for-event before an
attested workload bundle can be sealed.  Missing external observations remain
``unavailable``/undetermined; they never become automatic passes.
"""

from __future__ import annotations

from copy import deepcopy
import base64
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import tempfile
import time
from typing import Any, Callable, Iterable, Mapping

import torch
import torch.nn.functional as F
from pcr.model import BackboneConfig
from pcr.modes import remote_parameters, trainable_parameters
from pcr.split import SplitConfig
from pcr.train import OptimConfig, build_initialized_model, build_optimizer
from pcr.utils import parameters_checksum

from .cc_admission import (
    AdmissionTrustPolicy,
    load_admission_record,
    load_arrival_record,
    require_arrival_cuda_device,
    require_cc_admission,
    verify_cc_admission_reference,
)
from .cc_active_security import require_active_security_manifest
from .cc_bridge import CCTeeGpuRemotePath, install_cc_bridge
from .cc_deployment import (validate_bundle_manifest,
                            validate_runtime_source_manifest)
from .cc_external_collector import (
    ExternalCollectorClient,
    canonical_sha256 as collector_canonical_sha256,
    load_collector_manifest,
    trigger_from_trusted_event,
    verify_external_journal_auth,
)
from .cc_io import (
    read_json_snapshot,
    read_snapshot,
    require_unchanged,
    write_bytes_exclusive,
    write_json_exclusive,
)
from .cc_lifecycle import emit_lifecycle_heartbeat, require_supervisor_launch_context
from .cc_runtime_attestation import (
    validate_challenge_nonce,
    validate_runtime_environment,
    validate_trusted_store,
)
from .corpus import contiguous_region, load_document_index, load_pinned_corpus
from .experiment import ROOT, digest
from .model import BoundaryCapture, attach_defense
from .hardware_preorder import load_packet as load_hardware_packet, preorder_blockers
from .warmup_scenarios import trusted_checksum


CONFIG_SCHEMA = "p4-cc-engineering-workload-config-0.1"
TRUSTED_EVENT_SCHEMA = "p4-cc-trusted-event-0.1"
EXTERNAL_EVENT_SCHEMA = "p4-cc-external-event-0.2"
BUNDLE_SCHEMA = "p4-cc-workload-bundle-0.1"
PROTOCOL_ID = "cross-paper-secure-compute-d0q8-1.0"
PHASES = (
    "ingress", "tokenisation", "forward", "return", "backward", "update",
    "optimizer", "cpu-gpu-staging", "ipc", "vram", "checkpointing",
    "logging", "crash-restart", "egress",
)
CHANNELS = (
    "forward_values", "returned_values", "outbound_gradients", "inbound_gradients",
)
IDENTIFIER = re.compile(r"^[A-Za-z0-9_.:-]+$")
SHA256 = re.compile(r"^[a-f0-9]{64}$")
GIT_REVISION = re.compile(r"^(?:[a-f0-9]{40}|[a-f0-9]{64})$")
ZERO_SHA256 = "0" * 64
CLAIM_SCOPE = (
    "Attested engineering workload record for one declared deployment and two-view "
    "collection contract; not scientific evidence, a privacy certificate, or a "
    "claim about unassessed host/physical channels."
)

WORKLOAD_PATH_BINDINGS = (
    "checkpoint_manifest_path", "checkpoint_artifact_path", "resolved_config_path",
    "metrics_path", "panels_manifest_path", "panels_artifact_path", "uv_lock_path",
    "source_tree_manifest_path", "attack_plan_path",
    "evidence_envelope_schema_path",
)

WORKLOAD_DIGEST_BINDINGS = tuple(
    key.replace("_path", "_sha256") for key in WORKLOAD_PATH_BINDINGS
)

EXTERNAL_OBSERVATION_KEYS = {
    "collector_id", "collector_measurement_sha256", "metadata_fields", "wire_bytes",
    "packet_count", "elapsed_ns", "status_code", "observation_sha256", "limitation",
}
EXTERNAL_METADATA_FIELDS = {
    "length", "timing", "packet_count", "direction", "phase", "status_code",
    "resource_counter_names", "address_bucket", "access_count",
}

REQUIRED_SEALED_ARTIFACT_ROLES = {
    "correctness_comparison", "execution_summary", "step_metrics", "checkpoint",
    "source_snapshot", "active_security_gate",
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _tensor_identity(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().cpu().contiguous()
    raw = value.numpy().tobytes()
    header = _canonical_bytes({"dtype": str(value.dtype), "shape": list(value.shape)})
    return {
        "sha256": _sha256_bytes(header + b"\0" + raw),
        "bytes": len(raw),
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "finite": bool(torch.isfinite(value).all()) if value.is_floating_point() else True,
    }


def _state_identity(value: Any) -> dict[str, Any]:
    """Stable digest for nested tensor state without persisting its raw values."""
    digest_state = hashlib.sha256()
    byte_count = 0

    def walk(item: Any, path: str) -> None:
        nonlocal byte_count
        digest_state.update(path.encode("utf-8") + b"\0")
        if isinstance(item, torch.Tensor):
            identity = _tensor_identity(item)
            digest_state.update(_canonical_bytes(identity))
            byte_count += identity["bytes"]
        elif isinstance(item, Mapping):
            for key in sorted(item, key=lambda entry: str(entry)):
                walk(item[key], f"{path}/{key}")
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                walk(child, f"{path}/{index}")
        elif item is None or isinstance(item, (str, int, float, bool)):
            encoded = _canonical_bytes(item)
            digest_state.update(encoded)
            byte_count += len(encoded)
        else:
            raise TypeError(f"Unsupported state value at {path}: {type(item).__name__}")

    walk(value, "root")
    return {"sha256": digest_state.hexdigest(), "bytes": byte_count}


def _model_state_identity(model: torch.nn.Module) -> dict[str, Any]:
    return _state_identity({name: value.detach() for name, value in model.state_dict().items()})


def _optimizer_state_identity(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    return _state_identity(optimizer.state_dict())


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"{path}:{number}: blank journal row")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number}: invalid JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{number}: journal row must be an object")
        rows.append(row)
    if not rows:
        raise ValueError(f"{path}: empty journal")
    return rows


def _write_json(path: str | Path, value: Any) -> Path:
    return write_json_exclusive(path, value)


def validate_workload_config(config: Mapping[str, Any], execute: bool = False) -> Mapping[str, Any]:
    if not isinstance(config, Mapping) or config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError(f"expected {CONFIG_SCHEMA}")
    if config.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("cross-paper protocol identity mismatch")
    if config.get("cohort") != "engineering" or config.get("scientific_evidence") is not False:
        raise ValueError("CC workload must remain engineering-only")
    if config.get("secure_compute_profile") != "tee":
        raise ValueError("the initial H100 confidential-compute workload must use TEE")
    if config.get("status") not in {"draft", "frozen"}:
        raise ValueError("workload status must be draft or frozen")
    if execute and config.get("status") != "frozen":
        raise PermissionError("draft CC workload cannot execute")
    expected_anchor = {
        "id": "p4-d0-q8", "corpus": "gutenberg_p4", "start": 2, "span": 4,
        "width": 256, "remote_depth": 1, "q_bits": 8, "sequence_length": 128,
        "batch_size": 32,
    }
    if config.get("anchor") != expected_anchor:
        raise ValueError("workload anchor differs from d0-q8")
    execution = config.get("execution")
    if not isinstance(execution, Mapping):
        raise ValueError("execution block is required")
    frozen = {
        "correctness_windows": 512,
        "correctness_batch_size": 32,
        "warmup_steps": 1,
        "measured_steps": 3,
        "repetitions": 3,
        "engineering_seed": 410901,
        "maximum_wall_minutes": 240,
        "protection_modes": ["cc-off", "cc-on"],
        "required_phases": list(PHASES),
        "persist_raw_tensors": False,
        "external_journal_required": True,
        "automatic_scientific_launch": False,
    }
    if any(execution.get(key) != value for key, value in frozen.items()):
        raise ValueError("execution contract differs from the minimum engineering matrix")
    tolerances = config.get("tolerances")
    if tolerances != {
        "max_abs_logit_delta": 1e-4,
        "absolute_loss_delta": 1e-5,
        "token_agreement": 1.0,
        "invalid_secret_releases": 0,
        "accepted_mutations": 0,
        "external_canary_exposures": 0,
    }:
        raise ValueError("correctness or zero-tolerance security gate changed")
    bindings = config.get("bindings")
    required_paths = set(WORKLOAD_PATH_BINDINGS)
    required_hashes = set(WORKLOAD_DIGEST_BINDINGS)
    required_keys = required_paths | required_hashes | {"repository_revision"}
    if not isinstance(bindings, Mapping) or set(bindings) != required_keys:
        raise ValueError("workload bindings differ from the exact path/digest inventory")
    for key in required_paths:
        if not isinstance(bindings[key], str) or not bindings[key]:
            raise ValueError(f"bindings.{key} is required")
    for key in required_hashes:
        value = bindings[key]
        if value is not None and (not isinstance(value, str) or not SHA256.fullmatch(value)):
            raise ValueError(f"bindings.{key} must be null or lowercase SHA-256")
    revision = bindings["repository_revision"]
    if revision is not None and (not isinstance(revision, str)
                                 or not GIT_REVISION.fullmatch(revision)):
        raise ValueError("bindings.repository_revision must be null or a full Git object ID")
    if config.get("claim_scope") != CLAIM_SCOPE:
        raise ValueError("claim scope differs")
    return config


def workload_blockers(config: Mapping[str, Any], root: str | Path = ROOT) -> list[str]:
    validate_workload_config(config)
    root = Path(root).resolve()
    blockers = []
    if config["status"] != "frozen":
        blockers.append("workload_not_frozen")
    bindings = config["bindings"]
    for key in (*WORKLOAD_DIGEST_BINDINGS, "repository_revision"):
        if bindings[key] is None:
            blockers.append(f"missing_{key}")
    for key, value in bindings.items():
        if not key.endswith("_path"):
            continue
        path = (root / value).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            blockers.append(f"unsafe_{key}")
            continue
        digest_key = key.replace("_path", "_sha256")
        if not path.is_file():
            blockers.append(f"missing_{key.removesuffix('_path')}")
        elif bindings.get(digest_key) and digest(path) != bindings[digest_key]:
            blockers.append(f"changed_{key.removesuffix('_path')}")
    revision = bindings.get("repository_revision")
    if revision is not None:
        try:
            check = subprocess.run(
                ["git", "cat-file", "-e", f"{revision}^{{commit}}"], cwd=root,
                check=False, capture_output=True, text=True, timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            blockers.append("repository_revision_unverifiable")
        else:
            if check.returncode != 0:
                blockers.append("repository_revision_unverifiable")
    return blockers


class TrustedJournal:
    """Append-only, fsync-per-event hash chain; raw values are never written.

    The chain detects mutation once its final head is signed and bound to the
    admitted session.  A chain without that authentication is diagnostic only
    and cannot be sealed as attested evidence.
    """

    def __init__(self, path: str | Path, session_id: str,
                 external_collector: ExternalCollectorClient | None = None):
        if not isinstance(session_id, str) or not IDENTIFIER.fullmatch(session_id):
            raise ValueError("invalid session_id")
        self.path = Path(path)
        if self.path.exists():
            raise FileExistsError(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id
        self.sequence = 0
        self.chain_head = ZERO_SHA256
        self.external_collector = external_collector

    def _append(self, request_id: str, step_id: int, phase: str, operation: str,
                identity: Mapping[str, Any], limitation: str | None = None,
                exercise_status: str = "exercised") -> dict[str, Any]:
        if not isinstance(request_id, str) or not IDENTIFIER.fullmatch(request_id):
            raise ValueError("invalid request_id")
        if type(step_id) is not int or step_id < 0 or phase not in PHASES:
            raise ValueError("invalid step or phase")
        if not isinstance(operation, str) or not operation:
            raise ValueError("operation is required")
        if not isinstance(identity.get("sha256"), str) or not SHA256.fullmatch(identity["sha256"]):
            raise ValueError("event identity requires SHA-256")
        if exercise_status not in {"exercised", "simulated", "unavailable"}:
            raise ValueError("invalid trusted exercise status")
        if exercise_status != "exercised" and not limitation:
            raise ValueError("non-exercised trusted events require a limitation")
        event = {
            "schema_version": TRUSTED_EVENT_SCHEMA,
            "event_id": f"trusted-{self.sequence:08d}",
            "correlation_id": f"event-{self.sequence:08d}",
            "sequence": self.sequence,
            "view": "trusted_diagnostic",
            "session_id": self.session_id,
            "request_id": request_id,
            "step_id": step_id,
            "phase": phase,
            "operation": operation,
            "identity": dict(identity),
            "exercise_status": exercise_status,
            "previous_event_sha256": self.chain_head,
        }
        if limitation:
            event["limitation"] = limitation
        event["event_sha256"] = _canonical_sha256(event)
        encoded = json.dumps(event, sort_keys=True, separators=(",", ":"), allow_nan=False)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(encoded + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        if self.external_collector is not None:
            # Synchronous acknowledgement makes this a live correlation path,
            # rather than a template that can be populated after execution.
            self.external_collector.observe(event)
        self.sequence += 1
        self.chain_head = event["event_sha256"]
        return event

    def tensor(self, request_id: str, step_id: int, phase: str, operation: str,
               tensor: torch.Tensor, limitation: str | None = None) -> dict[str, Any]:
        identity = _tensor_identity(tensor)
        if not identity["finite"]:
            raise ValueError("nonfinite tensor cannot enter trusted journal")
        return self._append(request_id, step_id, phase, operation, identity, limitation)

    def state(self, request_id: str, step_id: int, phase: str, operation: str,
              value: Any, limitation: str | None = None,
              exercise_status: str = "exercised") -> dict[str, Any]:
        return self._append(request_id, step_id, phase, operation,
                            _state_identity(value), limitation, exercise_status)

    def artifact(self, request_id: str, step_id: int, phase: str, operation: str,
                 path: str | Path, limitation: str | None = None,
                 exercise_status: str = "exercised") -> dict[str, Any]:
        path = Path(path)
        return self._append(request_id, step_id, phase, operation,
                            {"sha256": digest(path), "bytes": path.stat().st_size}, limitation,
                            exercise_status)


def _read_trusted_journal(path: str | Path) -> list[dict[str, Any]]:
    rows = _read_jsonl(path)
    previous = ZERO_SHA256
    common = {
        "schema_version", "event_id", "correlation_id", "sequence", "view",
        "session_id", "request_id", "step_id", "phase", "operation", "identity",
        "exercise_status", "previous_event_sha256", "event_sha256",
    }
    for sequence, row in enumerate(rows):
        expected_keys = common | ({"limitation"} if row.get("exercise_status") != "exercised"
                                  or "limitation" in row else set())
        if set(row) != expected_keys:
            raise ValueError("trusted journal event fields differ")
        if (row.get("schema_version") != TRUSTED_EVENT_SCHEMA
                or row.get("view") != "trusted_diagnostic"
                or row.get("sequence") != sequence
                or row.get("event_id") != f"trusted-{sequence:08d}"
                or row.get("correlation_id") != f"event-{sequence:08d}"):
            raise ValueError("trusted journal identity or sequence differs")
        if (row.get("phase") not in PHASES
                or row.get("exercise_status") not in {"exercised", "simulated", "unavailable"}):
            raise ValueError("trusted journal phase/status differs")
        if row["exercise_status"] != "exercised" and not row.get("limitation"):
            raise ValueError("non-exercised trusted event lacks a limitation")
        if row.get("previous_event_sha256") != previous:
            raise ValueError("trusted journal hash chain is broken")
        event_sha = row.get("event_sha256")
        unsigned = {key: value for key, value in row.items() if key != "event_sha256"}
        if not isinstance(event_sha, str) or event_sha != _canonical_sha256(unsigned):
            raise ValueError("trusted journal event digest mismatch")
        identity = row.get("identity")
        if (not isinstance(identity, Mapping)
                or not SHA256.fullmatch(str(identity.get("sha256")))):
            raise ValueError("trusted journal content identity differs")
        previous = event_sha
    return rows


def _journal_auth_payload(session_id: str, journal_path: str | Path,
                          admission_sha256: str, workload_config_sha256: str,
                          public_key_sha256: str) -> dict[str, Any]:
    rows = _read_trusted_journal(journal_path)
    for value, label in ((admission_sha256, "admission"),
                         (workload_config_sha256, "workload config"),
                         (public_key_sha256, "journal public key")):
        if not isinstance(value, str) or not SHA256.fullmatch(value):
            raise ValueError(f"invalid {label} SHA-256")
    if any(row["session_id"] != session_id for row in rows):
        raise ValueError("trusted journal/session mismatch")
    return {
        "schema_version": "p4-cc-journal-auth-payload-0.1",
        "session_id": session_id,
        "journal_sha256": digest(journal_path),
        "chain_head_sha256": rows[-1]["event_sha256"],
        "event_count": len(rows),
        "admission_sha256": admission_sha256,
        "workload_config_sha256": workload_config_sha256,
        "public_key_sha256": public_key_sha256,
        "signature_algorithm": "Ed25519",
    }


def _require_ed25519_journal_public_key(public_key_path: str | Path) -> None:
    try:
        result = subprocess.run(
            ["openssl", "pkey", "-pubin", "-in", str(public_key_path),
             "-text_pub", "-noout"],
            capture_output=True, check=False, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"cannot inspect journal public key algorithm: {exc}") from exc
    first = result.stdout.splitlines()[0] if result.stdout else b""
    if result.returncode != 0 or first.strip() != b"ED25519 Public-Key:":
        raise PermissionError("journal public key must be Ed25519")


def authenticate_trusted_journal(journal_path: str | Path, output_path: str | Path,
                                 session_id: str, admission_sha256: str,
                                 workload_config_sha256: str,
                                 private_key_path: str | Path,
                                 public_key_path: str | Path) -> Path:
    """Sign the final hash-chain head using the admission-bound guest key."""
    private_key_path = Path(private_key_path)
    public_key_path = Path(public_key_path)
    _require_journal_keypair(private_key_path, public_key_path)
    if not private_key_path.is_file() or not public_key_path.is_file():
        raise ValueError("journal signing keypair is missing")
    if stat.S_IMODE(private_key_path.stat().st_mode) & 0o077:
        raise PermissionError("journal private key must not be group/world accessible")
    public_sha = digest(public_key_path)
    payload = _journal_auth_payload(
        session_id, journal_path, admission_sha256, workload_config_sha256, public_sha,
    )
    encoded = _canonical_bytes(payload)
    with tempfile.TemporaryDirectory(prefix="p4-cc-sign-") as temporary:
        payload_path = Path(temporary) / "payload.bin"
        signature_path = Path(temporary) / "signature.bin"
        payload_path.write_bytes(encoded)
        try:
            proc = subprocess.run(
                ["openssl", "pkeyutl", "-sign", "-inkey", str(private_key_path),
                 "-rawin", "-in", str(payload_path), "-out", str(signature_path)],
                capture_output=True, check=False, timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"journal signing failed: {exc}") from exc
        if proc.returncode != 0 or not signature_path.is_file() \
                or signature_path.stat().st_size < 1:
            raise RuntimeError("journal signing failed closed")
        signature = signature_path.read_bytes()
    record = {
        "schema_version": "p4-cc-journal-auth-0.1",
        "payload": payload,
        "payload_sha256": _sha256_bytes(encoded),
        "signature_base64": base64.b64encode(signature).decode("ascii"),
    }
    path = _write_json(output_path, record)
    os.chmod(path, 0o600)
    verify_trusted_journal_auth(path, journal_path, public_key_path,
                                expected_admission_sha256=admission_sha256,
                                expected_workload_config_sha256=workload_config_sha256)
    return path


def verify_trusted_journal_auth(auth_path: str | Path, journal_path: str | Path,
                                public_key_path: str | Path, *,
                                expected_admission_sha256: str,
                                expected_workload_config_sha256: str) -> dict[str, Any]:
    """Verify an Ed25519 journal signature and all session/content bindings."""
    _require_ed25519_journal_public_key(public_key_path)
    record = json.loads(Path(auth_path).read_text())
    if (not isinstance(record, Mapping)
            or set(record) != {"schema_version", "payload", "payload_sha256",
                               "signature_base64"}
            or record.get("schema_version") != "p4-cc-journal-auth-0.1"):
        raise ValueError("journal authentication record fields differ")
    payload = record.get("payload")
    if not isinstance(payload, Mapping):
        raise ValueError("journal authentication payload is missing")
    expected = _journal_auth_payload(
        str(payload.get("session_id")), journal_path, expected_admission_sha256,
        expected_workload_config_sha256, digest(public_key_path),
    )
    if dict(payload) != expected:
        raise ValueError("journal authentication payload binding mismatch")
    encoded = _canonical_bytes(expected)
    if record.get("payload_sha256") != _sha256_bytes(encoded):
        raise ValueError("journal authentication payload digest mismatch")
    try:
        signature = base64.b64decode(record.get("signature_base64", ""), validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("journal signature is not canonical base64") from exc
    with tempfile.TemporaryDirectory(prefix="p4-cc-verify-") as temporary:
        payload_path = Path(temporary) / "payload.bin"
        signature_path = Path(temporary) / "signature.bin"
        payload_path.write_bytes(encoded)
        signature_path.write_bytes(signature)
        try:
            proc = subprocess.run(
                ["openssl", "pkeyutl", "-verify", "-pubin", "-inkey",
                 str(public_key_path), "-rawin", "-in", str(payload_path),
                 "-sigfile", str(signature_path)],
                capture_output=True, check=False, timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"journal signature verification failed: {exc}") from exc
    if proc.returncode != 0:
        raise ValueError("journal signature verification failed")
    return dict(record)


def authenticate_execution_manifest(
        manifest_path: str | Path, output_path: str | Path, *, session_id: str,
        pair_id: str, protection_mode: str, admission_sha256: str,
        workload_config_sha256: str, hardware_packet_sha256: str,
        private_key_path: str | Path, public_key_path: str | Path) -> Path:
    """Sign the exact aggregate of all repetition bundles with the guest key."""
    _require_journal_keypair(Path(private_key_path), Path(public_key_path))
    if protection_mode not in {"cc-on", "cc-off"}:
        raise ValueError("execution-manifest protection mode differs")
    for value, label in ((session_id, "session"), (pair_id, "pair")):
        if (not isinstance(value, str) or not IDENTIFIER.fullmatch(value)
                or SHA256.fullmatch(value)):
            raise ValueError(f"execution-manifest {label} identity is invalid")
    payload = {
        "schema_version": "p4-cc-execution-manifest-auth-payload-0.1",
        "session_id": session_id,
        "pair_id": pair_id,
        "protection_mode": protection_mode,
        "execution_manifest_sha256": digest(manifest_path),
        "admission_sha256": admission_sha256,
        "workload_config_sha256": workload_config_sha256,
        "hardware_packet_sha256": hardware_packet_sha256,
        "public_key_sha256": digest(public_key_path),
        "signature_algorithm": "Ed25519",
    }
    for key, value in payload.items():
        if key.endswith("_sha256") and not SHA256.fullmatch(str(value)):
            raise ValueError(f"execution-manifest {key} is invalid")
    encoded = _canonical_bytes(payload)
    with tempfile.TemporaryDirectory(prefix="p4-cc-execution-sign-") as temporary:
        payload_path = Path(temporary) / "payload.bin"
        signature_path = Path(temporary) / "signature.bin"
        payload_path.write_bytes(encoded)
        result = subprocess.run(
            ["openssl", "pkeyutl", "-sign", "-inkey", str(private_key_path),
             "-rawin", "-in", str(payload_path), "-out", str(signature_path)],
            capture_output=True, check=False, timeout=30,
        )
        if result.returncode != 0 or not signature_path.is_file():
            raise RuntimeError("execution-manifest signing failed closed")
        signature = signature_path.read_bytes()
    record = {
        "schema_version": "p4-cc-execution-manifest-auth-0.1",
        "payload": payload,
        "payload_sha256": _sha256_bytes(encoded),
        "signature_base64": base64.b64encode(signature).decode("ascii"),
    }
    path = _write_json(output_path, record)
    os.chmod(path, 0o600)
    verify_execution_manifest_auth(
        path, manifest_path, public_key_path, expected_session_id=session_id,
        expected_pair_id=pair_id, expected_protection_mode=protection_mode,
        expected_admission_sha256=admission_sha256,
        expected_workload_config_sha256=workload_config_sha256,
        expected_hardware_packet_sha256=hardware_packet_sha256,
    )
    return path


def verify_execution_manifest_auth(
        auth_path: str | Path, manifest_path: str | Path,
        public_key_path: str | Path, *, expected_session_id: str,
        expected_pair_id: str, expected_protection_mode: str,
        expected_admission_sha256: str, expected_workload_config_sha256: str,
        expected_hardware_packet_sha256: str) -> dict[str, Any]:
    _require_ed25519_journal_public_key(public_key_path)
    record = _load_json_object(Path(auth_path), "execution manifest authentication")
    if (set(record) != {"schema_version", "payload", "payload_sha256",
                        "signature_base64"}
            or record["schema_version"] != "p4-cc-execution-manifest-auth-0.1"):
        raise ValueError("execution-manifest authentication fields differ")
    expected = {
        "schema_version": "p4-cc-execution-manifest-auth-payload-0.1",
        "session_id": expected_session_id,
        "pair_id": expected_pair_id,
        "protection_mode": expected_protection_mode,
        "execution_manifest_sha256": digest(manifest_path),
        "admission_sha256": expected_admission_sha256,
        "workload_config_sha256": expected_workload_config_sha256,
        "hardware_packet_sha256": expected_hardware_packet_sha256,
        "public_key_sha256": digest(public_key_path),
        "signature_algorithm": "Ed25519",
    }
    if record.get("payload") != expected:
        raise ValueError("execution-manifest authentication binding mismatch")
    encoded = _canonical_bytes(expected)
    if record.get("payload_sha256") != _sha256_bytes(encoded):
        raise ValueError("execution-manifest authentication digest mismatch")
    try:
        signature = base64.b64decode(record.get("signature_base64", ""), validate=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("execution-manifest signature is not canonical base64") from exc
    with tempfile.TemporaryDirectory(prefix="p4-cc-execution-verify-") as temporary:
        payload_path = Path(temporary) / "payload.bin"
        signature_path = Path(temporary) / "signature.bin"
        payload_path.write_bytes(encoded)
        signature_path.write_bytes(signature)
        result = subprocess.run(
            ["openssl", "pkeyutl", "-verify", "-pubin", "-inkey",
             str(public_key_path), "-rawin", "-in", str(payload_path),
             "-sigfile", str(signature_path)],
            capture_output=True, check=False, timeout=30,
        )
    if result.returncode != 0:
        raise ValueError("execution-manifest signature verification failed")
    return record


def validate_sealed_repetition_set(
        manifest_path: str | Path, auth_path: str | Path,
        public_key_path: str | Path, *, expected_session_id: str,
        expected_pair_id: str, expected_protection_mode: str,
        expected_admission_sha256: str, expected_workload_config_sha256: str,
        expected_hardware_packet_sha256: str) -> dict[str, Any]:
    """Verify the signed exact-three bundle aggregate and every bundle byte."""
    manifest_path = Path(manifest_path).resolve()
    root = manifest_path.parent.parent
    manifest = _load_json_object(manifest_path, "sealed repetition set")
    fields = {
        "schema_version", "protocol_id", "session_id", "pair_id",
        "protection_mode", "scientific_evidence", "admission_sha256",
        "workload_config_sha256", "hardware_packet_sha256",
        "artifact_manifest_sha256", "correctness_comparison_sha256",
        "source_snapshot_sha256", "active_security_manifest_sha256",
        "repetition_count", "repetitions", "claim_scope",
    }
    if (set(manifest) != fields
            or manifest["schema_version"] != "p4-cc-sealed-repetition-set-0.1"
            or manifest["protocol_id"] != PROTOCOL_ID
            or manifest["session_id"] != expected_session_id
            or manifest["pair_id"] != expected_pair_id
            or manifest["protection_mode"] != expected_protection_mode
            or manifest["scientific_evidence"] is not False
            or manifest["admission_sha256"] != expected_admission_sha256
            or manifest["workload_config_sha256"] != expected_workload_config_sha256
            or manifest["hardware_packet_sha256"] != expected_hardware_packet_sha256
            or manifest["repetition_count"] != 3
            or manifest["claim_scope"] != CLAIM_SCOPE):
        raise ValueError("sealed repetition-set identity or binding differs")
    conventional = {
        "artifact_manifest_sha256": root / "artifact_manifest.json",
        "correctness_comparison_sha256": root / "correctness_comparison.json",
        "source_snapshot_sha256": root / "source_snapshot.json",
        "active_security_manifest_sha256": root / "active-security",
    }
    for key, path in conventional.items():
        if key == "active_security_manifest_sha256":
            candidates = [candidate for candidate in path.iterdir() if candidate.is_file()]
            candidates = [candidate for candidate in candidates
                          if digest(candidate) == manifest[key]]
            if len(candidates) != 1:
                raise ValueError("sealed repetition-set active manifest bytes differ")
        elif not path.is_file() or digest(path) != manifest[key]:
            raise ValueError(f"sealed repetition-set {key} bytes differ")
    rows = manifest.get("repetitions")
    row_fields = {"repetition", "journal_id", "path", "sha256", "bytes",
                  "event_pair_chain_sha256"}
    if (not isinstance(rows, list) or len(rows) != 3
            or [row.get("repetition") for row in rows] != [0, 1, 2]):
        raise ValueError("sealed repetition-set must contain exactly repetitions 0,1,2")
    for index, row in enumerate(rows):
        if (not isinstance(row, Mapping) or set(row) != row_fields
                or row["journal_id"] != f"repetition-{index:02d}"
                or row["path"] != f"sealed/repetition-{index:02d}.bundle.json"
                or type(row["bytes"]) is not int or row["bytes"] < 1
                or not SHA256.fullmatch(str(row["sha256"]))
                or not SHA256.fullmatch(str(row["event_pair_chain_sha256"]))):
            raise ValueError("sealed repetition row fields differ")
        bundle_path = root / row["path"]
        if (not bundle_path.is_file() or bundle_path.stat().st_size != row["bytes"]
                or digest(bundle_path) != row["sha256"]):
            raise ValueError("sealed repetition bundle bytes differ")
        bundle = _load_json_object(bundle_path, "sealed repetition bundle")
        validate_workload_bundle(bundle)
        if (bundle["session_id"] != expected_session_id
                or bundle["protection_mode"] != expected_protection_mode
                or bundle["admission"]["admission_sha256"]
                != expected_admission_sha256
                or bundle["admission"]["workload_config_sha256"]
                != expected_workload_config_sha256
                or bundle["event_pair_chain_sha256"]
                != row["event_pair_chain_sha256"]):
            raise ValueError("sealed repetition bundle/aggregate binding differs")
    verify_execution_manifest_auth(
        auth_path, manifest_path, public_key_path,
        expected_session_id=expected_session_id, expected_pair_id=expected_pair_id,
        expected_protection_mode=expected_protection_mode,
        expected_admission_sha256=expected_admission_sha256,
        expected_workload_config_sha256=expected_workload_config_sha256,
        expected_hardware_packet_sha256=expected_hardware_packet_sha256,
    )
    return manifest


def _prepare_trusted_output(output: str | Path, trusted_store_root: str | Path | None,
                            *, required: bool) -> Path:
    output = Path(output).resolve()
    if trusted_store_root is None:
        if required:
            raise PermissionError("production workload requires an explicit trusted-store root")
        return output
    root = Path(trusted_store_root).resolve()
    if not root.is_dir():
        raise PermissionError("trusted-store root must already exist")
    try:
        output.relative_to(root)
    except ValueError as exc:
        raise PermissionError("workload output escapes the trusted-store root") from exc
    if stat.S_IMODE(root.stat().st_mode) != 0o700:
        raise PermissionError("trusted-store root must have mode 0700")
    return output


def run_training_probe(model: torch.nn.Module, batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
                       optimizer_config: OptimConfig, output: str | Path, session_id: str,
                       device: str | torch.device = "cpu", start_step: int = 0,
                       warmup_steps: int = 0, *,
                       trusted_store_root: str | Path | None = None,
                       require_hardware_bridge: bool = False,
                       test_only_cpu_bridge: bool = False,
                       deadline_monotonic: float | None = None,
                       journal_auth_context: Mapping[str, Any] | None = None,
                       external_collector: ExternalCollectorClient | None = None,
                       collector_journal_id: str | None = None,
                       heartbeat_callback: Callable[[str, int | None], None] | None = None,
                       ) -> dict[str, Any]:
    """Exercise complete training paths; intended for admitted wrapper or local tests."""
    output = _prepare_trusted_output(
        output, trusted_store_root, required=require_hardware_bridge,
    )
    output.mkdir(parents=True, exist_ok=False)
    os.chmod(output, 0o700)
    device = torch.device(device)
    if isinstance(getattr(model, "remote_path", None), CCTeeGpuRemotePath):
        bridge = model.remote_path
        if bridge.remote_device != device:
            raise ValueError("installed CC bridge remote device differs from requested device")
        if require_hardware_bridge and bridge.test_only_cpu_loopback:
            raise PermissionError("test-only bridge cannot satisfy a hardware workload")
    else:
        bridge = install_cc_bridge(
            model, device, test_only_cpu_loopback=test_only_cpu_bridge,
            setup_purpose="checkpoint-restore",
        )
    if require_hardware_bridge and bridge.remote_device.type != "cuda":
        raise PermissionError("hardware workload requires a CUDA remote surrogate")
    model.train()
    optimizer = build_optimizer(model, optimizer_config)
    if external_collector is not None:
        if not collector_journal_id:
            raise ValueError("live external collector requires an explicit journal ID")
        external_collector.begin_journal(
            collector_journal_id, output / "external_journal.jsonl",
            output / "external_journal.auth.json",
        )
    elif require_hardware_bridge:
        raise PermissionError("production workload requires a ready live external collector")
    journal = TrustedJournal(
        output / "trusted_journal.jsonl", session_id, external_collector,
    )
    rows = []
    batches = list(batches)
    if not batches:
        raise ValueError("training probe requires at least one batch")

    control_transfer_cursor = 0

    def journal_control_transfers(request_id: str, step_id: int) -> None:
        nonlocal control_transfer_cursor
        events = bridge.control_transfer_events_since(control_transfer_cursor)
        for event in events:
            journal.state(
                request_id, step_id, "cpu-gpu-staging",
                f"control_transfer_{event['sequence']:06d}_{event['purpose']}",
                event,
            )
        control_transfer_cursor += len(events)

    for offset, (x_cpu, y_cpu) in enumerate(batches):
        if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
            raise TimeoutError("engineering workload reached its wall-time deadline")
        step_id = start_step + offset
        if heartbeat_callback is not None:
            heartbeat_callback("training-step-start", step_id)
        request_id = f"{session_id}-step-{step_id:06d}"
        journal_control_transfers(request_id, step_id)
        journal.tensor(request_id, step_id, "ingress", "input_tokens", x_cpu)
        journal.tensor(request_id, step_id, "tokenisation", "byte_token_ids", x_cpu)
        before_model_state, _ = bridge.copy_payload_to_trusted(
            model.state_dict(), purpose="model-state-snapshot",
            request_id=request_id, step_id=step_id)
        before_optimizer_state, _ = bridge.copy_payload_to_trusted(
            optimizer.state_dict(), purpose="optimizer-state-snapshot",
            request_id=request_id, step_id=step_id)
        before_remote_state, _ = bridge.copy_payload_to_trusted(
            bridge.remote.state_dict(), purpose="model-state-snapshot",
            request_id=request_id, step_id=step_id)
        journal_control_transfers(request_id, step_id)
        before_model = _state_identity(before_model_state)
        before_remote = _state_identity(before_remote_state)["sha256"]
        journal.state(request_id, step_id, "update", "model_state_before",
                      before_model_state)
        journal.state(request_id, step_id, "optimizer", "optimizer_state_before",
                      before_optimizer_state)
        try:
            for group in optimizer.param_groups:
                group["lr"] = optimizer_config.lr_at(offset)
            optimizer.zero_grad(set_to_none=True)
            x, y = x_cpu.to("cpu"), y_cpu.to("cpu")
            journal.state(request_id, step_id, "cpu-gpu-staging", "input_stage",
                          {"trusted_device": "cpu", "remote_device": str(device),
                           "tokens": _tensor_identity(x_cpu),
                           "placement": bridge.placement_report()})
            capture = BoundaryCapture()
            bridge.begin_step(step_id, request_id, expect_backward=True)
            step_started = time.perf_counter_ns()
            forward_started = time.perf_counter_ns()
            logits = model(x, capture=capture)
            forward_ns = time.perf_counter_ns() - forward_started
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite training loss")
            backward_started = time.perf_counter_ns()
            loss.backward()
            backward_ns = time.perf_counter_ns() - backward_started
            bridge_trace = bridge.end_step()
            missing = [name for name in CHANNELS if name not in capture.values]
            if missing:
                raise RuntimeError(f"missing boundary capture channels: {missing}")
            journal.tensor(request_id, step_id, "forward", "forward_values",
                           capture.values["forward_values"])
            journal.tensor(request_id, step_id, "return", "returned_values",
                           capture.values["returned_values"])
            journal.tensor(request_id, step_id, "backward", "outbound_gradients",
                           capture.values["outbound_gradients"])
            journal.tensor(request_id, step_id, "backward", "inbound_gradients",
                           capture.values["inbound_gradients"])
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters(model),
                                                        optimizer_config.grad_clip)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            update_started = time.perf_counter_ns()
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            update_ns = time.perf_counter_ns() - update_started
            step_compute_ns = time.perf_counter_ns() - step_started
            for transfer in bridge_trace["events"]:
                journal.state(
                    request_id, step_id, "cpu-gpu-staging",
                    f"boundary_transfer_{transfer['channel']}", transfer,
                )
            journal.state(
                request_id, step_id, "ipc", "protected_device_bridge_trace",
                bridge_trace,
                "Same-process CPU/GPU device transport; no inter-process channel is claimed.",
                exercise_status="simulated",
            )
            after_model_state, _ = bridge.copy_payload_to_trusted(
                model.state_dict(), purpose="model-state-snapshot",
                request_id=request_id, step_id=step_id)
            after_optimizer_state, _ = bridge.copy_payload_to_trusted(
                optimizer.state_dict(), purpose="optimizer-state-snapshot",
                request_id=request_id, step_id=step_id)
            after_remote_state, _ = bridge.copy_payload_to_trusted(
                bridge.remote.state_dict(), purpose="model-state-snapshot",
                request_id=request_id, step_id=step_id)
            journal_control_transfers(request_id, step_id)
            after_model = _state_identity(after_model_state)
            after_remote = _state_identity(after_remote_state)["sha256"]
            after_optimizer = _state_identity(after_optimizer_state)
            journal.state(request_id, step_id, "update", "model_state_after",
                          after_model_state)
            journal.state(request_id, step_id, "optimizer", "optimizer_state_after",
                          after_optimizer_state)
            vram = {
                "device": str(device),
                "allocated_bytes": torch.cuda.memory_allocated(device) if device.type == "cuda" else 0,
                "reserved_bytes": torch.cuda.memory_reserved(device) if device.type == "cuda" else 0,
            }
            journal.state(request_id, step_id, "vram", "device_memory_inventory", vram)
            journal.state(request_id, step_id, "logging", "step_log",
                          {"loss": float(loss.detach()), "grad_norm": float(grad_norm),
                           "forward_ns": forward_ns, "backward_ns": backward_ns,
                           "update_ns": update_ns})

            checkpoint = output / f"checkpoint-step-{step_id:06d}.pt"
            pending = checkpoint.with_suffix(".pt.pending")
            checkpoint_payload, _ = bridge.copy_payload_to_trusted(
                {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "step_id": step_id},
                purpose="checkpoint-snapshot", request_id=request_id,
                step_id=step_id)
            journal_control_transfers(request_id, step_id)
            try:
                torch.save(checkpoint_payload, pending)
                os.chmod(pending, 0o600)
                os.replace(pending, checkpoint)
            finally:
                if pending.exists():
                    pending.unlink()
            journal.artifact(request_id, step_id, "checkpointing", "protected_checkpoint",
                             checkpoint,
                             "Checkpoint contains sensitive state and must remain in the trusted store.")
            restored = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if _state_identity(restored["model"])["sha256"] != after_model["sha256"]:
                raise RuntimeError("checkpoint recovery model digest mismatch")
            if _state_identity(restored["optimizer"])["sha256"] != after_optimizer["sha256"]:
                raise RuntimeError("checkpoint recovery optimizer digest mismatch")
            journal.state(
                request_id, step_id, "crash-restart", "checkpoint_roundtrip_recovery_probe",
                {"checkpoint_sha256": digest(checkpoint), "restored_step": restored["step_id"]},
                "Serialization/reload recovery only; a host crash is not exercised by this event.",
                exercise_status="simulated",
            )
            row = {
                "step_id": step_id,
                "measurement_role": "warmup" if offset < warmup_steps else "measured",
                "request_id": request_id,
                "loss": float(loss.detach()),
                "grad_norm": float(grad_norm),
                "forward_ns": forward_ns,
                "backward_ns": backward_ns,
                "update_ns": update_ns,
                "complete_step_ns": step_compute_ns,
                "bridge_total_wire_bytes": bridge_trace["total_wire_bytes"],
                "bridge_elapsed_ns": bridge_trace["elapsed_ns"],
                "bridge_channel_bytes": bridge_trace["channel_bytes"],
                "model_before_sha256": before_model["sha256"],
                "model_after_sha256": after_model["sha256"],
                "remote_before_sha256": before_remote,
                "remote_after_sha256": after_remote,
                "optimizer_after_sha256": after_optimizer["sha256"],
                "checkpoint_sha256": digest(checkpoint),
            }
            rows.append(row)
            journal.state(request_id, step_id, "egress", "step_result", row)
            if heartbeat_callback is not None:
                heartbeat_callback("training-step-complete", step_id)
            if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
                raise TimeoutError("engineering workload reached its wall-time deadline")
        except Exception as exc:
            bridge.abort_step()
            journal.state(request_id, step_id, "crash-restart", "failed_step",
                          {"exception_type": type(exc).__name__},
                          "Failure retained; no success or recovery is inferred.")
            if external_collector is not None:
                external_collector.abort_journal()
            raise

    metrics_path = _write_json(output / "step_metrics.json", {
        "schema_version": "p4-cc-step-metrics-0.1",
        "session_id": session_id,
        "scientific_evidence": False,
        "rows": rows,
    })
    os.chmod(metrics_path, 0o600)
    os.chmod(journal.path, 0o600)
    trusted = _read_trusted_journal(journal.path)
    template_path = None
    external_path = None
    external_auth_path = None
    if external_collector is None:
        template_path = output / "external_journal.template.jsonl"
        with template_path.open("w", encoding="utf-8") as stream:
            for event in trusted:
                template = {
                    "schema_version": EXTERNAL_EVENT_SCHEMA,
                    "correlation_id": event["correlation_id"],
                    "sequence": event["sequence"],
                    "view": "external_attacker",
                    "session_id": event["session_id"],
                    "request_id": event["request_id"],
                    "step_id": event["step_id"],
                    "phase": event["phase"],
                    "operation": event["operation"],
                    "collector_id": "pending-live-collector",
                    "collector_measurement_sha256": ZERO_SHA256,
                    "collector_sequence": event["sequence"],
                    "trigger_sha256": ZERO_SHA256,
                    "status": "pending",
                    "observation": None,
                    "reason": "Diagnostic template only; production cannot seal it.",
                }
                stream.write(json.dumps(template, sort_keys=True, separators=(",", ":")) + "\n")
        os.chmod(template_path, 0o600)
    else:
        external_path, external_auth_path = external_collector.finish_journal()
    auth_path = None
    if journal_auth_context is not None:
        required_auth = {"admission_sha256", "workload_config_sha256",
                         "private_key_path", "public_key_path"}
        if set(journal_auth_context) != required_auth:
            raise ValueError("journal authentication context fields differ")
        auth_path = authenticate_trusted_journal(
            journal.path, output / "trusted_journal.auth.json", session_id,
            journal_auth_context["admission_sha256"],
            journal_auth_context["workload_config_sha256"],
            journal_auth_context["private_key_path"],
            journal_auth_context["public_key_path"],
        )
    elif require_hardware_bridge:
        raise PermissionError("production workload requires journal authentication")
    return {
        "trusted_journal": journal.path,
        "external_template": template_path,
        "external_journal": external_path,
        "external_journal_auth": external_auth_path,
        "metrics": metrics_path,
        "rows": rows,
        "event_count": len(trusted),
        "chain_head_sha256": journal.chain_head,
        "journal_auth": auth_path,
        "control_transfer_summary": bridge.control_transfer_summary(),
    }


def run_correctness_summary(model: torch.nn.Module, tokens: torch.Tensor,
                            next_tokens: torch.Tensor, batch_size: int,
                            device: str | torch.device = "cpu") -> dict[str, Any]:
    """Retain equality-sufficient summaries without persisting raw logits."""
    if len(tokens) != len(next_tokens) or not 1 <= batch_size <= len(tokens):
        raise ValueError("invalid correctness panel")
    device = torch.device(device)
    model = model.to(device)
    was_training = model.training
    model.eval()
    logit_digests, argmax_digests, losses = [], [], []
    try:
        with torch.no_grad():
            for start in range(0, len(tokens), batch_size):
                x = tokens[start:start + batch_size].to(device)
                y = next_tokens[start:start + batch_size].to(device)
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
                if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                    raise RuntimeError("nonfinite correctness output")
                logit_digests.append(_tensor_identity(logits)["sha256"])
                argmax_digests.append(_tensor_identity(logits.argmax(-1))["sha256"])
                losses.append(float(loss))
    finally:
        model.train(was_training)
    return {
        "schema_version": "p4-cc-correctness-summary-0.1",
        "windows": len(tokens),
        "batch_size": batch_size,
        "batches": len(logit_digests),
        "logit_batch_sha256": logit_digests,
        "argmax_batch_sha256": argmax_digests,
        "mean_loss": sum(losses) / len(losses),
        "raw_logits_persisted": False,
    }


def compare_correctness_summaries(reference: Mapping[str, Any], candidate: Mapping[str, Any],
                                  tolerances: Mapping[str, Any]) -> dict[str, Any]:
    """Exact digest equality proves zero delta; mismatches fail closed as undetermined."""
    identity_keys = ("schema_version", "windows", "batch_size", "batches")
    if any(reference.get(key) != candidate.get(key) for key in identity_keys):
        raise ValueError("correctness panel identities differ")
    logits_equal = reference.get("logit_batch_sha256") == candidate.get("logit_batch_sha256")
    argmax_equal = reference.get("argmax_batch_sha256") == candidate.get("argmax_batch_sha256")
    loss_delta = abs(float(candidate["mean_loss"]) - float(reference["mean_loss"]))
    if logits_equal:
        max_delta = 0.0
        agreement = 1.0
        status = "PASS" if (loss_delta <= tolerances["absolute_loss_delta"]
                            and agreement >= tolerances["token_agreement"]) else "FAIL"
        reason = "Exact logit digests match; maximum logit delta is zero."
    else:
        max_delta = None
        agreement = 1.0 if argmax_equal else None
        status = "UNDETERMINED"
        reason = ("Logit digests differ and raw logits are intentionally not persisted; "
                  "a trusted paired replay is required to quantify the delta.")
    return {
        "schema_version": "p4-cc-correctness-comparison-0.1",
        "status": status,
        "max_abs_logit_delta": max_delta,
        "absolute_loss_delta": loss_delta,
        "token_agreement": agreement,
        "logits_exactly_equal": logits_equal,
        "argmax_exactly_equal": argmax_equal,
        "reason": reason,
    }


CORRECTNESS_BINDING_KEYS = {
    "pair_id", "session_id", "config_sha256", "checkpoint_sha256", "panels_sha256",
    "hardware_identity_sha256", "pair_manifest_sha256",
    "deployment_manifest_sha256", "protection_mode",
}


def run_paired_correctness(reference_model: torch.nn.Module,
                           candidate_model: torch.nn.Module,
                           tokens: torch.Tensor, next_tokens: torch.Tensor,
                           batch_size: int, tolerances: Mapping[str, Any],
                           binding: Mapping[str, Any]) -> dict[str, Any]:
    """Measure exact deltas in trusted memory and retain only digest summaries."""
    if set(binding) != CORRECTNESS_BINDING_KEYS:
        raise ValueError("correctness binding fields differ")
    if binding["protection_mode"] not in {"cc-on", "cc-off"}:
        raise ValueError("correctness protection mode differs")
    for key in ("config_sha256", "checkpoint_sha256", "panels_sha256",
                "hardware_identity_sha256", "pair_manifest_sha256",
                "deployment_manifest_sha256"):
        if not SHA256.fullmatch(str(binding[key])):
            raise ValueError(f"correctness binding {key} is invalid")
    for key in ("pair_id", "session_id"):
        if not isinstance(binding[key], str) or not IDENTIFIER.fullmatch(binding[key]):
            raise ValueError(f"correctness binding {key} is invalid")
    if len(tokens) != len(next_tokens) or not 1 <= batch_size <= len(tokens):
        raise ValueError("invalid paired correctness panel")
    if (not torch.is_tensor(tokens) or not torch.is_tensor(next_tokens)
            or tokens.device.type != "cpu" or next_tokens.device.type != "cpu"):
        raise ValueError("paired correctness inputs must remain in trusted CPU memory")
    reference_model = reference_model.to("cpu")
    reference_training = reference_model.training
    candidate_training = candidate_model.training
    reference_model.eval()
    candidate_model.eval()
    max_delta = 0.0
    agreements = 0
    token_count = 0
    reference_losses: list[float] = []
    candidate_losses: list[float] = []
    reference_digests: list[str] = []
    candidate_digests: list[str] = []
    bridge_trace_digests: list[str] = []
    bridge_wire_bytes = 0
    try:
        with torch.no_grad():
            for start in range(0, len(tokens), batch_size):
                x = tokens[start:start + batch_size]
                y = next_tokens[start:start + batch_size]
                reference = reference_model(x)
                bridge = getattr(candidate_model, "remote_path", None)
                if not isinstance(bridge, CCTeeGpuRemotePath):
                    raise RuntimeError("paired correctness candidate lacks the CC device bridge")
                request_id = f"correctness-{start // batch_size:06d}"
                bridge.begin_step(request_id, request_id, expect_backward=False)
                try:
                    candidate = candidate_model(x)
                    trace = bridge.end_step()
                    bridge_trace_digests.append(_canonical_sha256(trace))
                    bridge_wire_bytes += trace["total_wire_bytes"]
                except Exception:
                    bridge.abort_step()
                    raise
                if candidate.device.type != "cpu":
                    raise RuntimeError("bridged candidate output left trusted CPU memory")
                if reference.shape != candidate.shape:
                    raise RuntimeError("paired correctness output shape mismatch")
                if not torch.isfinite(reference).all() or not torch.isfinite(candidate).all():
                    raise RuntimeError("nonfinite paired correctness output")
                reference_loss = F.cross_entropy(
                    reference.reshape(-1, reference.shape[-1]), y.reshape(-1))
                candidate_loss = F.cross_entropy(
                    candidate.reshape(-1, candidate.shape[-1]), y.reshape(-1))
                max_delta = max(max_delta, float((candidate - reference).abs().max()))
                reference_tokens = reference.argmax(-1)
                candidate_tokens = candidate.argmax(-1)
                agreements += int((reference_tokens == candidate_tokens).sum())
                token_count += reference_tokens.numel()
                reference_losses.append(float(reference_loss))
                candidate_losses.append(float(candidate_loss))
                reference_digests.append(_tensor_identity(reference)["sha256"])
                candidate_digests.append(_tensor_identity(candidate)["sha256"])
    finally:
        reference_model.train(reference_training)
        candidate_model.train(candidate_training)
    reference_mean = sum(reference_losses) / len(reference_losses)
    candidate_mean = sum(candidate_losses) / len(candidate_losses)
    loss_delta = abs(candidate_mean - reference_mean)
    agreement = agreements / token_count
    passed = (max_delta <= tolerances["max_abs_logit_delta"]
              and loss_delta <= tolerances["absolute_loss_delta"]
              and agreement >= tolerances["token_agreement"])
    input_identity = _state_identity({"tokens": tokens, "next_tokens": next_tokens})
    return {
        "schema_version": "p4-cc-paired-correctness-0.1",
        "status": "PASS" if passed else "FAIL",
        "binding": dict(binding),
        "input_sha256": input_identity["sha256"],
        "windows": len(tokens),
        "batch_size": batch_size,
        "batches": len(reference_digests),
        "reference_logit_batch_sha256": reference_digests,
        "candidate_logit_batch_sha256": candidate_digests,
        "bridge_trace_sha256": bridge_trace_digests,
        "bridge_total_wire_bytes": bridge_wire_bytes,
        "control_transfer_summary": bridge.control_transfer_summary(),
        "max_abs_logit_delta": max_delta,
        "absolute_loss_delta": loss_delta,
        "token_agreement": agreement,
        "reference_mean_loss": reference_mean,
        "candidate_mean_loss": candidate_mean,
        "raw_logits_persisted": False,
        "reason": "Trusted paired replay measured exact deltas without exporting logits.",
    }


def _validate_external_event(event: Mapping[str, Any]) -> None:
    required = {
        "schema_version", "correlation_id", "sequence", "view", "session_id",
        "request_id", "step_id", "phase", "operation", "collector_id",
        "collector_measurement_sha256", "collector_sequence", "trigger_sha256",
        "status", "observation", "reason",
    }
    if set(event) != required or event.get("schema_version") != EXTERNAL_EVENT_SCHEMA:
        raise ValueError("external event schema/fields differ")
    if event.get("view") != "external_attacker" or event.get("phase") not in PHASES:
        raise ValueError("external event view or phase differs")
    if (not isinstance(event.get("collector_id"), str)
            or not IDENTIFIER.fullmatch(event["collector_id"])
            or SHA256.fullmatch(event["collector_id"])):
        raise ValueError("external collector identity is invalid or digest-shaped")
    if (not SHA256.fullmatch(str(event.get("collector_measurement_sha256")))
            or not SHA256.fullmatch(str(event.get("trigger_sha256")))
            or type(event.get("collector_sequence")) is not int
            or event["collector_sequence"] < 0):
        raise ValueError("external collector measurement/sequence binding differs")
    if event.get("status") not in {"observed", "unavailable"}:
        raise ValueError("external status must be observed or unavailable; pending cannot seal")
    if event["status"] == "observed":
        observation = event.get("observation")
        if not isinstance(observation, Mapping) or not observation:
            raise ValueError("observed external event requires metadata")
        if not set(observation).issubset(EXTERNAL_OBSERVATION_KEYS):
            raise ValueError("external observation contains an unapproved field")
        if not isinstance(observation.get("collector_id"), str) \
                or observation["collector_id"] != event["collector_id"]:
            raise ValueError("external observation requires a collector identity")
        if observation.get("collector_measurement_sha256") != \
                event["collector_measurement_sha256"]:
            raise ValueError("external observation requires a collector measurement")
        fields = observation.get("metadata_fields")
        if (not isinstance(fields, list) or not fields
                or any(field not in EXTERNAL_METADATA_FIELDS for field in fields)
                or len(fields) != len(set(fields))):
            raise ValueError("external metadata field inventory differs")
        for key in ("wire_bytes", "packet_count", "elapsed_ns"):
            if key in observation and (type(observation[key]) is not int
                                       or observation[key] < 0):
                raise ValueError(f"external observation {key} must be nonnegative integer")
        for key in ("status_code", "limitation"):
            if key in observation and (not isinstance(observation[key], str)
                                       or not observation[key].strip()):
                raise ValueError(f"external observation {key} must be nonempty text")
        if "observation_sha256" in observation \
                and not SHA256.fullmatch(str(observation["observation_sha256"])):
            raise ValueError("external observation digest is invalid")
        # No nested object can smuggle plaintext/tensors past the field allowlist.
        if any(isinstance(value, (Mapping, tuple, bytes, bytearray))
               for value in observation.values()):
            raise ValueError("nested or binary external observation values are forbidden")
        if any(not isinstance(value, (str, int, list))
               for value in observation.values()):
            raise ValueError("external observation values must be approved JSON scalars")
        if any(isinstance(value, list) and key != "metadata_fields"
               for key, value in observation.items()):
            raise ValueError("unapproved external observation list")
    elif event.get("observation") is not None:
        raise ValueError("unavailable external event cannot invent an observation")
    if not isinstance(event.get("reason"), str) or not event["reason"].strip():
        raise ValueError("external event requires a reason")


def _validate_sealed_artifacts(artifacts: Iterable[Mapping[str, Any]],
                               artifact_root: str | Path,
                               admission_binding: Mapping[str, Any],
                               protection_mode: str) -> list[dict[str, Any]]:
    root = Path(artifact_root).resolve()
    records = deepcopy(list(artifacts))
    fields = {"role", "path", "sha256", "bytes", "media_type", "sensitivity"}
    if not records:
        raise ValueError("sealed workload requires an artifact inventory")
    roles = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping) or set(record) != fields:
            raise ValueError(f"sealed artifact {index} fields differ")
        role = record["role"]
        if not isinstance(role, str) or not role:
            raise ValueError("sealed artifact role is required")
        if record["sensitivity"] not in {"public-metadata", "trusted-sensitive"}:
            raise ValueError("sealed artifact sensitivity differs")
        if not isinstance(record["media_type"], str) or not record["media_type"]:
            raise ValueError("sealed artifact media type is required")
        relative = Path(str(record["path"]))
        if relative.is_absolute() or ".." in relative.parts or "." in relative.parts:
            raise ValueError("sealed artifact path must be safe and relative")
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError("sealed artifact escapes its root") from exc
        if (not path.is_file() or path.stat().st_size != record["bytes"]
                or digest(path) != record["sha256"]):
            raise ValueError(f"sealed artifact is missing or changed: {record['path']}")
        roles.append(role)
    missing = REQUIRED_SEALED_ARTIFACT_ROLES - set(roles)
    if missing:
        raise ValueError(f"sealed artifact roles missing: {sorted(missing)}")
    for role in REQUIRED_SEALED_ARTIFACT_ROLES - {"checkpoint"}:
        if roles.count(role) != 1:
            raise ValueError(f"sealed artifact role must occur exactly once: {role}")
    by_role = {
        role: [root / item["path"] for item in records if item["role"] == role]
        for role in set(roles)
    }
    summary = _load_json_object(by_role["execution_summary"][0], "execution summary")
    if (summary.get("schema_version") != "p4-cc-engineering-execution-0.2"
            or summary.get("protocol_id") != PROTOCOL_ID
            or summary.get("scientific_evidence") is not False
            or summary.get("privacy_outcome") != "undetermined"
            or summary.get("session_id") != admission_binding["session_id"]
            or summary.get("evidence_id") != admission_binding["admission_id"]
            or summary.get("evidence_sha256") != admission_binding["admission_sha256"]
            or summary.get("protection_mode") != protection_mode
            or summary.get("hardware_attested") != admission_binding["hardware_attested"]
            or summary.get("correctness_status") != "PASS"
            or summary.get("sealed") is not True):
        raise ValueError("execution summary does not prove a completed bound runner output")
    repetitions = summary.get("repetitions")
    if (not isinstance(repetitions, list) or len(repetitions) != 3
            or [item.get("repetition") for item in repetitions] != [0, 1, 2]):
        raise ValueError("execution summary must inventory exactly three repetitions")
    repetition_fields = {
        "repetition", "trusted_journal", "trusted_journal_auth",
        "external_journal", "external_journal_auth", "collector_journal_id",
        "metrics", "events", "control_transfer_summary",
    }
    for index, repetition in enumerate(repetitions):
        if (not isinstance(repetition, Mapping) or set(repetition) != repetition_fields
                or repetition["collector_journal_id"] != f"repetition-{index:02d}"
                or type(repetition["events"]) is not int or repetition["events"] < 1):
            raise ValueError("execution repetition inventory differs")
        transfer = repetition["control_transfer_summary"]
        if (not isinstance(transfer, Mapping)
                or transfer.get("schema_version")
                != "p4-cc-control-transfer-trace-0.1"
                or type(transfer.get("event_count")) is not int
                or transfer["event_count"] < 1
                or transfer.get("raw_tensors_persisted") is not False
                or not isinstance(transfer.get("totals_by_purpose"), Mapping)
                or "checkpoint-restore" not in transfer["totals_by_purpose"]
                or "checkpoint-snapshot" not in transfer["totals_by_purpose"]):
            raise ValueError("execution control-transfer summary differs")
        for key in ("trusted_journal", "trusted_journal_auth", "external_journal",
                    "external_journal_auth", "metrics"):
            relative = Path(str(repetition[key]))
            if relative.is_absolute() or ".." in relative.parts or "." in relative.parts:
                raise ValueError("execution repetition path is unsafe")
            if not (root / relative).is_file():
                raise ValueError("execution repetition evidence is missing")
    correctness = _load_json_object(
        by_role["correctness_comparison"][0], "correctness comparison")
    binding = correctness.get("binding")
    if (correctness.get("schema_version") != "p4-cc-paired-correctness-0.1"
            or correctness.get("status") != "PASS"
            or not isinstance(binding, Mapping)
            or binding.get("session_id") != admission_binding["session_id"]
            or binding.get("config_sha256") != admission_binding["workload_config_sha256"]
            or binding.get("protection_mode") != protection_mode
            or binding.get("pair_id") != summary.get("pair_id")
            or correctness.get("max_abs_logit_delta", math.inf) > 1e-4
            or correctness.get("absolute_loss_delta", math.inf) > 1e-5
            or correctness.get("token_agreement") != 1.0
            or correctness.get("raw_logits_persisted") is not False):
        raise ValueError("paired correctness artifact is not a bound PASS")
    snapshot = _load_json_object(by_role["source_snapshot"][0], "source snapshot")
    if (snapshot.get("schema_version") != "p4-cc-source-snapshot-0.1"
            or snapshot.get("scientific_evidence") is not False
            or snapshot.get("workload_config_sha256") != admission_binding[
                "workload_config_sha256"]
            or snapshot.get("hardware_packet_sha256") is None
            or not SHA256.fullmatch(str(snapshot.get("source_tree_sha256")))):
        raise ValueError("source snapshot binding differs")
    active_path = by_role["active_security_gate"][0]
    active = require_active_security_manifest(active_path, digest(active_path))
    if (snapshot.get("active_security_manifest_sha256") != digest(active_path)
            or active.get("outcome", {}).get("status") != "PASS"):
        raise ValueError("active-security artifact/source binding differs")
    metrics = _load_json_object(by_role["step_metrics"][0], "aggregate step metrics")
    metric_rows = metrics.get("rows")
    if (metrics.get("schema_version") != "p4-cc-aggregate-step-metrics-0.1"
            or metrics.get("session_id") != admission_binding["session_id"]
            or metrics.get("scientific_evidence") is not False
            or not isinstance(metric_rows, list) or len(metric_rows) != 12
            or {row.get("repetition") for row in metric_rows} != {0, 1, 2}
            or any(sum(row.get("repetition") == repetition for row in metric_rows) != 4
                   for repetition in range(3))):
        raise ValueError("aggregate metrics do not contain the exact three-by-four matrix")
    if len(by_role["checkpoint"]) != 12:
        raise ValueError("sealed artifact inventory requires twelve step checkpoints")
    return records


def seal_journals(trusted_path: str | Path, external_path: str | Path,
                  trusted_auth_path: str | Path, public_key_path: str | Path,
                  external_auth_path: str | Path,
                  collector_manifest_path: str | Path,
                  collector_public_key_path: str | Path,
                  collector_journal_id: str,
                  admission_binding: Mapping[str, Any], protection_mode: str,
                  artifacts: Iterable[Mapping[str, Any]],
                  artifact_root: str | Path, *,
                  production_collector: bool = True) -> dict[str, Any]:
    """Pair independent views and build an engineering-only attested bundle."""
    if protection_mode not in {"cc-on", "cc-off"}:
        raise ValueError("invalid protection mode")
    if set(admission_binding) != {"admission_id", "admission_sha256", "session_id",
                                 "hardware_attested", "workload_config_sha256",
                                 "journal_public_key_sha256",
                                 "collector_manifest_sha256",
                                 "collector_executable_sha256",
                                 "collector_public_key_sha256"}:
        raise ValueError("admission binding fields differ")
    for key in ("admission_sha256", "workload_config_sha256",
                "journal_public_key_sha256", "collector_manifest_sha256",
                "collector_executable_sha256", "collector_public_key_sha256"):
        if not SHA256.fullmatch(str(admission_binding[key])):
            raise ValueError(f"invalid {key}")
    if admission_binding["journal_public_key_sha256"] == \
            admission_binding["collector_public_key_sha256"]:
        raise ValueError("trusted journal and external collector keys must be distinct")
    if protection_mode == "cc-on" and admission_binding["hardware_attested"] is not True:
        raise ValueError("cc-on bundle requires admitted hardware")
    if protection_mode == "cc-off" and admission_binding["hardware_attested"] is not False:
        raise ValueError("cc-off comparator must not claim hardware attestation")

    if digest(public_key_path) != admission_binding["journal_public_key_sha256"]:
        raise ValueError("journal public key/admission binding mismatch")
    authentication = verify_trusted_journal_auth(
        trusted_auth_path, trusted_path, public_key_path,
        expected_admission_sha256=admission_binding["admission_sha256"],
        expected_workload_config_sha256=admission_binding["workload_config_sha256"],
    )
    trusted = _read_trusted_journal(trusted_path)
    external_authentication = verify_external_journal_auth(
        external_auth_path, external_path, collector_manifest_path,
        collector_public_key_path,
        expected_session_id=admission_binding["session_id"],
        expected_journal_id=collector_journal_id,
        expected_admission_sha256=admission_binding["admission_sha256"],
        expected_workload_config_sha256=admission_binding["workload_config_sha256"],
        expected_collector_manifest_sha256=admission_binding[
            "collector_manifest_sha256"],
        expected_collector_executable_sha256=admission_binding[
            "collector_executable_sha256"],
        expected_collector_public_key_sha256=admission_binding[
            "collector_public_key_sha256"],
        event_validator=_validate_external_event,
        production=production_collector,
    )
    collector_manifest = load_collector_manifest(
        collector_manifest_path, production=production_collector,
        expected_public_key_path=collector_public_key_path,
    )
    external = _read_jsonl(external_path)
    for event in external:
        _validate_external_event(event)
    if len(trusted) != len(external):
        raise ValueError("trusted and external event counts differ")
    paired_keys = ("correlation_id", "sequence", "session_id", "request_id",
                   "step_id", "phase", "operation")
    for expected_sequence, (left, right) in enumerate(zip(trusted, external)):
        if left.get("sequence") != expected_sequence or right.get("sequence") != expected_sequence:
            raise ValueError("journal sequence is not contiguous")
        if any(left.get(key) != right.get(key) for key in paired_keys):
            raise ValueError("paired view identity mismatch")
        expected_trigger = collector_canonical_sha256(trigger_from_trusted_event(left))
        if (right.get("trigger_sha256") != expected_trigger
                or right.get("collector_sequence") != expected_sequence
                or right.get("collector_id") != collector_manifest["collector_id"]
                or right.get("collector_measurement_sha256") != collector_manifest[
                    "executable_sha256"]):
            raise ValueError("external live-correlation or collector binding mismatch")
    if any(event["session_id"] != admission_binding["session_id"] for event in trusted):
        raise ValueError("workload/admission session mismatch")

    coverage = {}
    for phase in PHASES:
        external_rows = [event for event in external if event["phase"] == phase]
        trusted_rows = [event for event in trusted if event["phase"] == phase]
        if not external_rows or not trusted_rows:
            coverage[phase] = {"status": "missing", "event_count": 0,
                               "reason": "No paired event was produced."}
        elif (any(event["status"] == "unavailable" for event in external_rows)
              or any(event["exercise_status"] != "exercised" for event in trusted_rows)):
            reasons = []
            if any(event["status"] == "unavailable" for event in external_rows):
                reasons.append("external view unavailable")
            if any(event["exercise_status"] != "exercised" for event in trusted_rows):
                reasons.append("trusted operation not actually exercised")
            coverage[phase] = {"status": "unavailable", "event_count": len(external_rows),
                               "reason": "; ".join(reasons) + "."}
        else:
            coverage[phase] = {"status": "observed", "event_count": len(external_rows),
                               "reason": "Both views were collected for exercised operations."}
    missing = [phase for phase, value in coverage.items() if value["status"] == "missing"]
    if missing:
        raise ValueError(f"attested bundle lacks phase events: {missing}")
    unassessed = [phase for phase, value in coverage.items() if value["status"] == "unavailable"]
    sealed_artifacts = _validate_sealed_artifacts(
        artifacts, artifact_root, admission_binding, protection_mode)
    bundle = {
        "schema_version": BUNDLE_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "session_id": admission_binding["session_id"],
        "cohort": "engineering",
        "scientific_evidence": False,
        "hardware_attested": bool(admission_binding["hardware_attested"]),
        "protection_mode": protection_mode,
        "privacy_outcome": "undetermined",
        "security_collection_status": "unassessed" if unassessed else "collected-not-a-verdict",
        "claim_scope": CLAIM_SCOPE,
        "admission": dict(admission_binding),
        "journals": {
            "trusted_diagnostic": {"sha256": digest(trusted_path), "events": len(trusted)},
            "external_attacker": {"sha256": digest(external_path), "events": len(external)},
            "trusted_authentication": {
                "sha256": digest(trusted_auth_path),
                "payload_sha256": authentication["payload_sha256"],
                "chain_head_sha256": authentication["payload"]["chain_head_sha256"],
                "signature_algorithm": "Ed25519",
            },
            "external_authentication": {
                "sha256": digest(external_auth_path),
                "payload_sha256": external_authentication["payload_sha256"],
                "journal_id": collector_journal_id,
                "signature_algorithm": "Ed25519",
            },
            "collector": {
                "collector_id": collector_manifest["collector_id"],
                "manifest_sha256": collector_manifest["manifest_sha256"],
                "executable_sha256": collector_manifest["executable_sha256"],
                "public_key_sha256": collector_manifest["public_key_sha256"],
            },
        },
        "phase_coverage": coverage,
        "unassessed_phases": unassessed,
        "event_pair_chain_sha256": _canonical_sha256([
            {"trusted": left, "external": right} for left, right in zip(trusted, external)
        ]),
        "artifacts": sealed_artifacts,
    }
    validate_workload_bundle(bundle)
    return bundle


def validate_workload_bundle(bundle: Mapping[str, Any]) -> Mapping[str, Any]:
    required = {
        "schema_version", "protocol_id", "session_id", "cohort",
        "scientific_evidence", "hardware_attested", "protection_mode",
        "privacy_outcome", "security_collection_status", "claim_scope", "admission",
        "journals", "phase_coverage", "unassessed_phases",
        "event_pair_chain_sha256", "artifacts",
    }
    if (not isinstance(bundle, Mapping) or set(bundle) != required
            or bundle.get("schema_version") != BUNDLE_SCHEMA):
        raise ValueError(f"expected {BUNDLE_SCHEMA}")
    if bundle.get("protocol_id") != PROTOCOL_ID or bundle.get("cohort") != "engineering":
        raise ValueError("bundle protocol/cohort mismatch")
    if bundle.get("scientific_evidence") is not False or bundle.get("privacy_outcome") != "undetermined":
        raise ValueError("engineering bundle cannot make a scientific/privacy claim")
    if bundle.get("claim_scope") != CLAIM_SCOPE:
        raise ValueError("bundle claim scope differs")
    if set(bundle.get("phase_coverage", {})) != set(PHASES):
        raise ValueError("bundle phase inventory differs")
    for phase, row in bundle["phase_coverage"].items():
        if (not isinstance(row, Mapping)
                or set(row) != {"status", "event_count", "reason"}
                or row["status"] not in {"observed", "unavailable"}
                or type(row["event_count"]) is not int or row["event_count"] < 1
                or not isinstance(row["reason"], str) or not row["reason"]):
            raise ValueError(f"bundle phase coverage differs: {phase}")
    expected_unassessed = [phase for phase in PHASES
                           if bundle["phase_coverage"][phase]["status"] == "unavailable"]
    if bundle.get("unassessed_phases") != expected_unassessed:
        raise ValueError("unassessed phase inventory differs")
    if bundle.get("protection_mode") == "cc-on" and bundle.get("hardware_attested") is not True:
        raise ValueError("cc-on bundle lacks hardware admission")
    if bundle.get("protection_mode") == "cc-off" and bundle.get("hardware_attested") is not False:
        raise ValueError("cc-off comparator cannot be hardware-attested")
    if not SHA256.fullmatch(str(bundle.get("event_pair_chain_sha256"))):
        raise ValueError("invalid event-pair chain")
    admission = bundle.get("admission")
    if (not isinstance(admission, Mapping)
            or set(admission) != {"admission_id", "admission_sha256", "session_id",
                                  "hardware_attested", "workload_config_sha256",
                                  "journal_public_key_sha256",
                                  "collector_manifest_sha256",
                                  "collector_executable_sha256",
                                  "collector_public_key_sha256"}
            or admission["session_id"] != bundle["session_id"]
            or admission["hardware_attested"] != bundle["hardware_attested"]):
        raise ValueError("bundle admission binding differs")
    for key in ("admission_sha256", "workload_config_sha256",
                "journal_public_key_sha256", "collector_manifest_sha256",
                "collector_executable_sha256", "collector_public_key_sha256"):
        if not SHA256.fullmatch(str(admission[key])):
            raise ValueError("bundle admission digest is invalid")
    journals = bundle.get("journals")
    if (not isinstance(journals, Mapping)
            or set(journals) != {"trusted_diagnostic", "external_attacker",
                                 "trusted_authentication", "external_authentication",
                                 "collector"}):
        raise ValueError("bundle journal inventory differs")
    for view in ("trusted_diagnostic", "external_attacker"):
        if (set(journals[view]) != {"sha256", "events"}
                or not SHA256.fullmatch(str(journals[view]["sha256"]))
                or type(journals[view]["events"]) is not int
                or journals[view]["events"] < 1):
            raise ValueError("bundle journal binding differs")
    authentication = journals["trusted_authentication"]
    if (set(authentication) != {"sha256", "payload_sha256", "chain_head_sha256",
                                "signature_algorithm"}
            or authentication["signature_algorithm"] != "Ed25519"
            or any(not SHA256.fullmatch(str(authentication[key]))
                   for key in ("sha256", "payload_sha256", "chain_head_sha256"))):
        raise ValueError("bundle journal authentication differs")
    external_authentication = journals["external_authentication"]
    if (set(external_authentication) != {"sha256", "payload_sha256", "journal_id",
                                         "signature_algorithm"}
            or external_authentication["signature_algorithm"] != "Ed25519"
            or not isinstance(external_authentication["journal_id"], str)
            or any(not SHA256.fullmatch(str(external_authentication[key]))
                   for key in ("sha256", "payload_sha256"))):
        raise ValueError("bundle external authentication differs")
    collector = journals["collector"]
    if (not isinstance(collector, Mapping)
            or set(collector) != {"collector_id", "manifest_sha256",
                                  "executable_sha256", "public_key_sha256"}
            or not isinstance(collector["collector_id"], str)
            or any(not SHA256.fullmatch(str(collector[key]))
                   for key in ("manifest_sha256", "executable_sha256",
                               "public_key_sha256"))):
        raise ValueError("bundle collector binding differs")
    artifacts = bundle.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("bundle artifact inventory is missing")
    roles = [item.get("role") for item in artifacts if isinstance(item, Mapping)]
    if not REQUIRED_SEALED_ARTIFACT_ROLES.issubset(set(roles)):
        raise ValueError("bundle required artifact roles are missing")
    return bundle


def _load_seal_artifacts(path: str | Path) -> list[dict[str, Any]]:
    value = json.loads(Path(path).read_text())
    if (not isinstance(value, Mapping)
            or set(value) != {"schema_version", "artifacts"}
            or value.get("schema_version") != "p4-cc-seal-artifact-manifest-0.1"
            or not isinstance(value.get("artifacts"), list)):
        raise ValueError("seal artifact manifest fields differ")
    return value["artifacts"]


def _cc_off_session_id(inventory: Mapping[str, Any], inventory_sha256: str) -> str:
    if not SHA256.fullmatch(inventory_sha256):
        raise ValueError("CC-off inventory SHA-256 is invalid")
    pair_id = inventory.get("deployment", {}).get("pair_id")
    if not isinstance(pair_id, str) or not IDENTIFIER.fullmatch(pair_id):
        raise ValueError("CC-off inventory pair ID is invalid")
    return f"cc-off-{pair_id}-{inventory_sha256[:12]}"


def seal_with_evidence(trusted_path: str | Path, external_path: str | Path,
                       trusted_auth_path: str | Path, public_key_path: str | Path,
                       external_auth_path: str | Path,
                       collector_manifest_path: str | Path,
                       collector_public_key_path: str | Path,
                       collector_journal_id: str,
                       evidence_path: str | Path, artifact_manifest_path: str | Path,
                       output_path: str | Path, protection_mode: str,
                       workload_config_path: str | Path,
                       root: str | Path = ROOT, *,
                       paired_cc_on_admission_path: str | Path | None = None) -> Path:
    """Seal CC-on or CC-off only after byte-verifying the relevant evidence."""
    root = Path(root).resolve()
    config_sha = digest(workload_config_path)
    public_sha = digest(public_key_path)
    collector = load_collector_manifest(
        collector_manifest_path, production=True,
        expected_public_key_path=collector_public_key_path,
    )
    if protection_mode == "cc-on":
        expected = None
        admission = require_cc_admission(evidence_path, root=root,
                                         expected_workload_binding=expected)
        if admission["workload_binding"]["config_sha256"] != config_sha:
            raise PermissionError("seal admission/workload config mismatch")
        if admission["workload_binding"]["journal_public_key_sha256"] != public_sha:
            raise PermissionError("seal admission/journal key mismatch")
        binding = {
            "admission_id": admission["admission_id"],
            "admission_sha256": digest(evidence_path),
            "session_id": admission["session_id"],
            "hardware_attested": True,
            "workload_config_sha256": config_sha,
            "journal_public_key_sha256": public_sha,
            "collector_manifest_sha256": admission["workload_binding"][
                "collector_manifest_sha256"],
            "collector_executable_sha256": admission["workload_binding"][
                "collector_executable_sha256"],
            "collector_public_key_sha256": admission["workload_binding"][
                "collector_public_key_sha256"],
        }
    elif protection_mode == "cc-off":
        inventory = load_arrival_record(evidence_path)
        if inventory["deployment"]["protection_mode"] != "cc-off":
            raise PermissionError("seal comparator inventory is not CC-off")
        inventory_sha = digest(evidence_path)
        if paired_cc_on_admission_path is None:
            raise PermissionError("CC-off sealing requires the paired CC-on admission")
        paired = require_cc_admission(paired_cc_on_admission_path, root=root)
        paired_arrival = load_arrival_record(root / paired["arrival_record"])
        if _arrival_pair_identity(paired_arrival) != _arrival_pair_identity(inventory):
            raise PermissionError("CC-off seal inventory differs from paired CC-on admission")
        if paired["workload_binding"]["config_sha256"] != config_sha:
            raise PermissionError("CC-off seal config differs from paired CC-on admission")
        binding = {
            "admission_id": f"cc-off-inventory-{inventory_sha[:16]}",
            "admission_sha256": inventory_sha,
            "session_id": _cc_off_session_id(inventory, inventory_sha),
            "hardware_attested": False,
            "workload_config_sha256": config_sha,
            "journal_public_key_sha256": public_sha,
            "collector_manifest_sha256": paired["workload_binding"][
                "collector_manifest_sha256"],
            "collector_executable_sha256": paired["workload_binding"][
                "collector_executable_sha256"],
            "collector_public_key_sha256": paired["workload_binding"][
                "collector_public_key_sha256"],
        }
    else:
        raise ValueError("protection_mode must be cc-on or cc-off")
    artifacts = _load_seal_artifacts(artifact_manifest_path)
    for key in ("collector_manifest_sha256", "collector_executable_sha256",
                "collector_public_key_sha256"):
        actual = {
            "collector_manifest_sha256": collector["manifest_sha256"],
            "collector_executable_sha256": collector["executable_sha256"],
            "collector_public_key_sha256": collector["public_key_sha256"],
        }[key]
        if binding[key] != actual:
            raise PermissionError(f"seal {key} differs from admitted collector bytes")
    bundle = seal_journals(
        trusted_path, external_path, trusted_auth_path, public_key_path,
        external_auth_path, collector_manifest_path, collector_public_key_path,
        collector_journal_id,
        binding, protection_mode, artifacts, Path(artifact_manifest_path).parent,
    )
    path = _write_json(output_path, bundle)
    os.chmod(path, 0o600)
    return path


def seal_with_admission(trusted_path: str | Path, external_path: str | Path,
                        trusted_auth_path: str | Path, public_key_path: str | Path,
                        external_auth_path: str | Path,
                        collector_manifest_path: str | Path,
                        collector_public_key_path: str | Path,
                        collector_journal_id: str,
                        admission_path: str | Path, artifact_manifest_path: str | Path,
                        output_path: str | Path, workload_config_path: str | Path,
                        root: str | Path = ROOT) -> Path:
    """Compatibility wrapper for the strict CC-on sealing path."""
    return seal_with_evidence(
        trusted_path, external_path, trusted_auth_path, public_key_path,
        external_auth_path, collector_manifest_path, collector_public_key_path,
        collector_journal_id,
        admission_path, artifact_manifest_path, output_path, "cc-on",
        workload_config_path, root,
    )


def make_deployment_model(config: Mapping[str, Any], device: str | torch.device = "cpu"):
    """Restore only the manifest-bound checkpoint inputs needed for deployment.

    The 4+ GB calibration capture is intentionally not a deployment dependency;
    checkpoint, resolved config and metrics remain independently hash-bound by
    the original run manifest and this workload packet.
    """
    validate_workload_config(config)
    bindings = config["bindings"]
    for path_key in ("checkpoint_manifest_path", "checkpoint_artifact_path",
                     "resolved_config_path", "metrics_path"):
        path = ROOT / bindings[path_key]
        digest_key = path_key.replace("_path", "_sha256")
        if not path.is_file() or digest(path) != bindings[digest_key]:
            raise ValueError(f"missing or changed deployment input: {path_key}")
    manifest = json.loads((ROOT / bindings["checkpoint_manifest_path"]).read_text())
    if (manifest.get("status") != "completed" or manifest.get("protocol_id") != "p4-qualification-0.1"
            or manifest.get("cell") != "d0-q8" or manifest.get("mode") != "trained"):
        raise ValueError("checkpoint source manifest differs from d0-q8 qualification")
    checkpoint_entries = [item for item in manifest["artifacts"]
                          if item["path"] == bindings["checkpoint_artifact_path"]]
    if len(checkpoint_entries) != 1 or checkpoint_entries[0]["sha256"] != bindings[
            "checkpoint_artifact_sha256"]:
        raise ValueError("checkpoint is not bound by the source manifest")
    resolved = json.loads((ROOT / bindings["resolved_config_path"]).read_text())
    metrics = json.loads((ROOT / bindings["metrics_path"]).read_text())
    artifact_map = {item.get("path"): item.get("sha256") for item in manifest["artifacts"]}
    for path_key in ("resolved_config_path", "metrics_path"):
        if artifact_map.get(bindings[path_key]) != bindings[path_key.replace("_path", "_sha256")]:
            raise ValueError(f"{path_key} is not bound by the source manifest")
    if (resolved.get("mode") != "trained" or resolved.get("seed") != manifest.get("seed")
            or resolved.get("cell") != manifest.get("inputs", {}).get("cell")):
        raise ValueError("resolved checkpoint identity differs from source manifest")
    cfg = resolved["config"]
    cell = resolved["cell"]
    expected_split = {"start": 2, "span": 4, "width": 256, "remote_depth": 1}
    expected_backbone = {
        "vocab_size": 256, "d_model": 256, "n_heads": 8, "n_layers": 8,
        "d_ff": 1024, "max_seq_len": 128, "init_std": 0.02,
    }
    if (cfg.get("split") != expected_split or cfg.get("backbone") != expected_backbone
            or cell != {"id": "d0-q8", "defense": {"name": "none"}, "q_bits": 8}
            or cfg.get("data", {}).get("sequence_length") != 128
            or cfg.get("data", {}).get("batch_size") != 32
            or cfg.get("data", {}).get("sha256") != manifest.get("corpus_sha256")):
        raise ValueError("resolved deployment is not the declared d0-q8 anchor")
    model = build_initialized_model(
        BackboneConfig(**cfg["backbone"]), "trained",
        SplitConfig(**cfg["split"], q_bits=cell["q_bits"]),
        resolved["seed"], "cc-engineering-checkpoint-restore",
    )
    attach_defense(model, {"name": "none"})
    state = torch.load(ROOT / bindings["checkpoint_artifact_path"], map_location="cpu",
                       weights_only=True)
    model.load_state_dict(state, strict=True)
    if (trusted_checksum(model) != metrics["trusted_final_sha256"]
            or parameters_checksum(remote_parameters(model)) != metrics["remote_final_sha256"]):
        raise ValueError("restored deployment model differs from qualification metrics")
    return model.to(device), cfg, manifest


def load_deployment_panels(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_workload_config(config)
    bindings = config["bindings"]
    for path_key in ("panels_manifest_path", "panels_artifact_path"):
        path = ROOT / bindings[path_key]
        if not path.is_file() or digest(path) != bindings[path_key.replace("_path", "_sha256")]:
            raise ValueError(f"missing or changed deployment input: {path_key}")
    manifest = json.loads((ROOT / bindings["panels_manifest_path"]).read_text())
    if (manifest.get("status") != "completed"
            or manifest.get("protocol_id") != "p4-defense-pilot-1.0"
            or manifest.get("cohort") != "exploratory"
            or manifest.get("cell") != "d0-q8"
            or manifest.get("mode") != "trained"
            or manifest.get("inputs", {}).get("cell")
            != {"id": "d0-q8", "defense": {"name": "none"}, "q_bits": 8}
            or manifest.get("inputs", {}).get("panels_sha256")
            != bindings["panels_artifact_sha256"]):
        raise ValueError("panel manifest binding differs")
    panels = torch.load(ROOT / bindings["panels_artifact_path"], weights_only=True)
    if "evaluation" not in panels or not {"tokens", "next_tokens"}.issubset(panels["evaluation"]):
        raise ValueError("evaluation panel is missing")
    return panels


def require_cc_off_inventory(path: str | Path,
                             expected_identity: Mapping[str, Any]) -> dict[str, Any]:
    """Admit only an explicit GPU-CC-off inventory as the unattested comparator."""
    record = load_arrival_record(path)
    feature = record["gpu"]["protected_memory"]
    if set(expected_identity) != {
        "provider", "candidate_id", "sku", "region", "zone", "image_reference",
        "pair_id", "pair_manifest_sha256", "cuda_device_index", "gpu_name",
        "driver", "cpu_model",
    }:
        raise ValueError("CC-off expected identity fields differ")
    deployment = record["deployment"]
    if deployment["protection_mode"] != "cc-off":
        raise PermissionError("comparator inventory is not explicitly CC-off")
    for key in ("provider", "region", "zone", "image_reference",
                "pair_id", "pair_manifest_sha256", "cuda_device_index"):
        if deployment[key] != expected_identity[key]:
            raise PermissionError(f"CC-off comparator {key} identity mismatch")
    if (record["gpu"]["name"] != expected_identity["gpu_name"]
            or record["gpu"]["driver"] != expected_identity["driver"]):
        raise PermissionError("CC-off comparator GPU/driver identity mismatch")
    if record["cpu"]["model"] != expected_identity["cpu_model"]:
        raise PermissionError("CC-off comparator CPU identity mismatch")
    output = feature["output"]
    exact_off = re.search(r"(?im)^\s*CC\s+status\s*:\s*OFF\s*$", output) is not None
    if (not record["gpu"]["nvidia_smi_present"] or not feature["attempted"]
            or feature["exit_code"] != 0 or not exact_off):
        raise PermissionError("CC-off comparator inventory does not prove GPU CC status OFF")
    return record


def _load_json_object(path: str | Path, label: str) -> dict[str, Any]:
    try:
        value, _ = read_json_snapshot(path)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read {label}: {exc}") from exc
    return value


def _provider_slug(provider: str) -> str:
    values = {"Microsoft Azure": "azure", "Google Cloud": "gcp"}
    try:
        return values[provider]
    except KeyError as exc:
        raise ValueError(f"unsupported hardware provider: {provider!r}") from exc


def _arrival_pair_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    deployment = record["deployment"]
    return {
        "provider": deployment["provider"],
        "candidate_id": deployment["candidate_id"],
        "sku": deployment["sku"],
        "region": deployment["region"],
        "zone": deployment["zone"],
        "image_reference": deployment["image_reference"],
        "pair_id": deployment["pair_id"],
        "pair_manifest_sha256": deployment["pair_manifest_sha256"],
        "cuda_device_index": deployment["cuda_device_index"],
        "gpu_name": record["gpu"]["name"],
        "driver": record["gpu"]["driver"],
        "cpu_model": record["cpu"]["model"],
    }


def _admission_trust_policy(packet: Mapping[str, Any],
                            arrival: Mapping[str, Any],
                            hardware_packet_sha256: str) -> AdmissionTrustPolicy:
    """Build the verifier trust anchor without trusting admission self-description."""
    bindings = packet["bindings"]
    deployment = arrival["deployment"]
    deployment_binding = {
        key: deployment[key]
        for key in (
            "provider", "candidate_id", "sku", "region", "zone", "instance_id",
            "boot_id_sha256", "image_reference", "protection_mode", "pair_id",
            "cuda_device_index", "deployment_manifest_sha256",
            "pair_manifest_sha256",
        )
    }
    deployment_binding["hardware_packet_sha256"] = hardware_packet_sha256
    return AdmissionTrustPolicy(
        verifier_id=bindings["admission_verifier_id"],
        adapter_kind=bindings["admission_verifier_adapter_kind"],
        adapter_measurement_sha256=bindings["admission_verifier_adapter_sha256"],
        public_key_sha256=bindings["admission_verifier_public_key_sha256"],
        expected_deployment_binding=deployment_binding,
    )


def _validate_live_runtime_evidence(
        *, runtime_environment_manifest_path: Path,
        trusted_store_manifest_path: Path,
        trusted_store_protection_evidence_path: Path,
        trusted_store_root: Path, config: Mapping[str, Any],
        arrival: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    """Validate exact runtime/store evidence against this process and host."""
    runtime_manifest, runtime_snapshot = read_json_snapshot(
        runtime_environment_manifest_path)
    store_manifest, store_snapshot = read_json_snapshot(trusted_store_manifest_path)
    protection_snapshot = read_snapshot(trusted_store_protection_evidence_path)
    validate_runtime_environment(
        runtime_manifest,
        uv_lock_path=ROOT / config["bindings"]["uv_lock_path"],
        image_reference=arrival["deployment"]["image_reference"],
        requested_cuda_device=str(device),
    )
    validate_trusted_store(
        store_manifest, root=trusted_store_root,
        instance_id=arrival["deployment"]["instance_id"],
    )
    if store_manifest["protection_evidence_sha256"] != protection_snapshot.sha256:
        raise PermissionError(
            "trusted-store protection evidence differs from the live-store manifest")
    for snapshot in (runtime_snapshot, store_snapshot, protection_snapshot):
        require_unchanged(snapshot)
    return {
        "runtime_environment_sha256": runtime_snapshot.sha256,
        "trusted_store_manifest_sha256": store_snapshot.sha256,
        "trusted_store_protection_evidence_sha256": protection_snapshot.sha256,
    }


def _require_arrival_matches_packet(record: Mapping[str, Any],
                                    packet: Mapping[str, Any],
                                    protection_mode: str,
                                    deployment_manifest_sha256: str) -> None:
    deployment = record["deployment"]
    order = packet["order_parameters"]
    if protection_mode == "cc-on":
        expected = {
            "provider": _provider_slug(order["provider"]),
            "candidate_id": order["candidate_id"],
            "sku": order["sku"],
            "region": order["region"],
            "zone": order["zone"],
            "image_reference": order["immutable_image_reference"],
            "protection_mode": "cc-on",
            "deployment_manifest_sha256": deployment_manifest_sha256,
        }
        gpu_name = order["gpu_model"]
        driver = order["pinned_driver_version"]
        cpu_model = order["cpu_family"]
    elif protection_mode == "cc-off":
        comparator = packet["cc_off_comparator"]
        expected = {
            "provider": _provider_slug(comparator["provider"]),
            "candidate_id": comparator["candidate_id"],
            "sku": comparator["sku_or_mode"],
            "region": comparator["region"],
            "zone": comparator["zone"],
            "image_reference": comparator["image_reference"],
            "protection_mode": "cc-off",
            "deployment_manifest_sha256": deployment_manifest_sha256,
        }
        gpu_name = comparator["gpu_model"]
        driver = comparator["driver_version"]
        cpu_model = comparator["cpu_family"]
    else:
        raise ValueError("protection_mode must be cc-on or cc-off")
    for key, value in expected.items():
        if deployment.get(key) != value:
            raise PermissionError(f"arrival deployment {key} differs from the frozen order")
    if record["gpu"].get("name") != gpu_name:
        raise PermissionError("arrival GPU identity differs from the frozen order")
    if record["gpu"].get("driver") != driver:
        raise PermissionError("arrival driver differs from the frozen order")
    if record["cpu"].get("model") != cpu_model:
        raise PermissionError("arrival CPU identity differs from the frozen order")


def _artifact_entry(path: Path, root: Path, role: str, *,
                    sensitivity: str = "public-metadata") -> dict[str, Any]:
    path = path.resolve()
    root = root.resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError("execution artifact escapes its output root") from exc
    if not path.is_file() or path.is_symlink() or path.stat().st_size < 1:
        raise ValueError(f"execution artifact is missing: {relative}")
    return {
        "role": role,
        "path": str(relative),
        "sha256": digest(path),
        "bytes": path.stat().st_size,
        "media_type": (
            "application/json" if path.suffix == ".json"
            else "application/x-ndjson" if path.suffix == ".jsonl"
            else "application/octet-stream"
        ),
        "sensitivity": sensitivity,
    }


def _copy_active_security_evidence(manifest_path: Path, manifest: Mapping[str, Any],
                                   output: Path, expected_sha256: str) -> Path:
    source_root = manifest_path.parent.resolve()
    destination_root = output / "active-security"
    destination_root.mkdir(mode=0o700)
    snapshots = []
    for record in manifest["artifacts"]:
        relative = Path(record["path"])
        source = (source_root / relative).resolve()
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(destination.parent, 0o700)
        snapshot = read_snapshot(source, expected_sha256=record["sha256"])
        snapshots.append(snapshot)
        write_bytes_exclusive(destination, snapshot.data)
        if (destination.stat().st_size != record["bytes"]
                or digest(destination) != record["sha256"]):
            raise RuntimeError("active-security artifact changed while copying")
    copied_manifest = destination_root / manifest_path.name
    manifest_snapshot = read_snapshot(
        manifest_path, expected_sha256=expected_sha256)
    write_bytes_exclusive(copied_manifest, manifest_snapshot.data)
    for snapshot in snapshots + [manifest_snapshot]:
        require_unchanged(snapshot)
    return copied_manifest


def _require_journal_keypair(private_key_path: Path,
                             public_key_path: Path) -> None:
    if (not private_key_path.is_file() or private_key_path.is_symlink()
            or not public_key_path.is_file() or public_key_path.is_symlink()):
        raise PermissionError("journal signing keypair must be regular files")
    if stat.S_IMODE(private_key_path.stat().st_mode) & 0o077:
        raise PermissionError("journal private key must not be group/world accessible")
    inspections = (
        ["openssl", "pkey", "-in", str(private_key_path), "-text", "-noout"],
        ["openssl", "pkey", "-pubin", "-in", str(public_key_path),
         "-text_pub", "-noout"],
    )
    for command, expected in zip(
            inspections, (b"ED25519 Private-Key:", b"ED25519 Public-Key:")):
        try:
            result = subprocess.run(
                command, capture_output=True, check=False, timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"cannot inspect journal signing key algorithm: {exc}") from exc
        first = result.stdout.splitlines()[0] if result.stdout else b""
        if result.returncode != 0 or first.strip() != expected:
            raise PermissionError("journal signing keypair must use Ed25519")
    commands = (
        ["openssl", "pkey", "-in", str(private_key_path), "-pubout", "-outform", "DER"],
        ["openssl", "pkey", "-pubin", "-in", str(public_key_path), "-outform", "DER"],
    )
    outputs = []
    for command in commands:
        try:
            result = subprocess.run(
                command, capture_output=True, check=False, timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"cannot validate journal signing keypair: {exc}") from exc
        if result.returncode != 0 or not result.stdout:
            raise PermissionError("journal signing keypair is invalid")
        outputs.append(result.stdout)
    if outputs[0] != outputs[1]:
        raise PermissionError("journal private/public keys do not form a pair")


def _validate_execution_bindings(config_path: Path, config: Mapping[str, Any],
                                 hardware_packet_path: Path,
                                 deployment_manifest_path: Path,
                                 active_security_manifest_path: Path,
                                 active_security_manifest_sha256: str,
                                 challenge_nonce_path: Path,
                                 journal_private_key_path: Path,
                                 journal_public_key_path: Path,
                                 collector_manifest_path: Path,
                                 collector_argv: Iterable[str],
                                 collector_public_key_path: Path) -> dict[str, Any]:
    """Validate every immutable precondition before creating an output directory."""
    challenge_nonce_sha256 = validate_challenge_nonce(challenge_nonce_path)
    _require_journal_keypair(journal_private_key_path, journal_public_key_path)
    collector = load_collector_manifest(
        collector_manifest_path, production=True,
        expected_argv=list(collector_argv),
        expected_public_key_path=collector_public_key_path,
    )
    if collector["public_key_sha256"] == digest(journal_public_key_path):
        raise PermissionError("trusted journal and external collector require distinct keys")
    packet = load_hardware_packet(hardware_packet_path)
    blockers = preorder_blockers(packet)
    if blockers:
        raise PermissionError(f"hardware execution packet is not authorised: {blockers}")
    bindings = config["bindings"]
    packet_bindings = packet["bindings"]
    config_sha256 = digest(config_path)
    hardware_packet_sha256 = digest(hardware_packet_path)
    deployment_manifest_sha256 = digest(deployment_manifest_path)
    journal_public_key_sha256 = digest(journal_public_key_path)
    if packet_bindings["workload_config_sha256"] != config_sha256:
        raise PermissionError("hardware packet is not bound to this workload config")
    if packet_bindings["paper4_revision"] != bindings["repository_revision"]:
        raise PermissionError("hardware/workload Paper 4 revisions differ")
    for key in ("checkpoint_manifest_sha256", "checkpoint_artifact_sha256",
                "panels_manifest_sha256", "panels_artifact_sha256"):
        if packet_bindings[key] != bindings[key]:
            raise PermissionError(f"hardware/workload {key} differs")
    if packet_bindings["attack_plan_sha256"] != bindings["attack_plan_sha256"]:
        raise PermissionError("hardware/workload attack-plan bindings differ")
    if packet_bindings["evidence_schema_sha256"] != \
            bindings["evidence_envelope_schema_sha256"]:
        raise PermissionError("hardware/workload evidence-schema bindings differ")

    active = require_active_security_manifest(
        active_security_manifest_path, active_security_manifest_sha256)
    if active["inputs"]["attacker_repository_revision"] != \
            packet_bindings["llm_attacker_revision"]:
        raise PermissionError("active-security attacker revision differs from hardware packet")
    if (active["inputs"]["profile"] != config["secure_compute_profile"]
            or active["inputs"]["profile"] != packet["secure_compute_profile"]):
        raise PermissionError("active-security profile differs from the frozen TEE profile")
    if active["inputs"]["plan_file_sha256"] != bindings["attack_plan_sha256"]:
        raise PermissionError("active-security plan differs from workload config")
    source_manifest_path = ROOT / bindings["source_tree_manifest_path"]
    source_manifest = _load_json_object(source_manifest_path, "source manifest")
    runtime_repositories = {
        "paper3": {
            "root": (ROOT / packet_bindings["paper3_repository_path"]).resolve(),
            "revision": packet_bindings["paper3_revision"],
        },
        "paper4": {
            "root": (ROOT / packet_bindings["paper4_repository_path"]).resolve(),
            "revision": packet_bindings["paper4_revision"],
        },
        "llm-attacker": {
            "root": (ROOT / packet_bindings["llm_attacker_repository_path"]).resolve(),
            "revision": packet_bindings["llm_attacker_revision"],
        },
    }
    validate_runtime_source_manifest(source_manifest, runtime_repositories)
    expected_revisions = {
        "paper3": packet_bindings["paper3_revision"],
        "paper4": packet_bindings["paper4_revision"],
        "llm-attacker": packet_bindings["llm_attacker_revision"],
    }
    if source_manifest["repositories"] != expected_revisions:
        raise PermissionError("source manifest revisions differ from the hardware packet")

    bundle_manifest_path = ROOT / packet_bindings[
        "deployment_bundle_manifest_path"]
    bundle_manifest = _load_json_object(bundle_manifest_path, "deployment bundle manifest")
    archive_path = bundle_manifest_path.parent / bundle_manifest.get("archive", {}).get("path", "")
    validate_bundle_manifest(bundle_manifest, archive_path=archive_path)
    if digest(bundle_manifest_path) != packet_bindings[
            "deployment_bundle_manifest_sha256"]:
        raise PermissionError(
            "deployment bundle manifest differs from the hardware packet")
    if bundle_manifest["source_tree_sha256"] != source_manifest["source_tree_sha256"]:
        raise PermissionError("deployment bundle source tree differs from source manifest")
    if bundle_manifest["source_manifest_sha256"] != _canonical_sha256(source_manifest):
        raise PermissionError("deployment bundle source-manifest binding differs")
    if packet_bindings["deployment_bundle_sha256"] != bundle_manifest["archive"]["sha256"]:
        raise PermissionError("hardware packet deployment archive binding differs")
    if digest(archive_path) != packet_bindings["deployment_bundle_sha256"]:
        raise PermissionError("deployment archive bytes differ from the hardware packet")
    collector_packet_bindings = {
        "collector_manifest_sha256": collector["manifest_sha256"],
        "collector_executable_sha256": collector["executable_sha256"],
        "collector_public_key_sha256": collector["public_key_sha256"],
    }
    if any(packet_bindings.get(key) != value
           for key, value in collector_packet_bindings.items()):
        raise PermissionError("hardware packet differs from exact collector bytes")

    return {
        "packet": packet,
        "packet_sha256": hardware_packet_sha256,
        "config_sha256": config_sha256,
        "deployment_manifest_sha256": deployment_manifest_sha256,
        "challenge_nonce_sha256": challenge_nonce_sha256,
        "journal_public_key_sha256": journal_public_key_sha256,
        "collector": collector,
        "active_security": active,
        "active_security_sha256": active_security_manifest_sha256,
        "source_manifest": source_manifest,
        "source_manifest_file_sha256": digest(source_manifest_path),
        "bundle_manifest": bundle_manifest,
        "bundle_manifest_file_sha256": digest(bundle_manifest_path),
        "archive_path": archive_path,
    }


def run_engineering_workload(
        config_path: str | Path, evidence_path: str | Path,
        output: str | Path, protection_mode: str,
        device: str | torch.device = "cuda", *,
        hardware_packet_path: str | Path,
        deployment_manifest_path: str | Path,
        active_security_manifest_path: str | Path,
        active_security_manifest_sha256: str,
        challenge_nonce_path: str | Path,
        journal_private_key_path: str | Path,
        journal_public_key_path: str | Path,
        collector_manifest_path: str | Path,
        collector_argv: Iterable[str],
        collector_public_key_path: str | Path,
        runtime_environment_manifest_path: str | Path,
        trusted_store_manifest_path: str | Path,
        trusted_store_protection_evidence_path: str | Path,
        trusted_store_root: str | Path,
        admitted_session_id: str,
        admission_nonce_ledger_path: str | Path,
        paired_cc_on_admission_path: str | Path | None = None) -> dict[str, Any]:
    """Execute the engineering canary only after all pre-order controls pass.

    This function never provisions hardware.  It assumes an independently
    created host and fails closed unless the frozen order packet, admission,
    active attacks, source/deployment bytes, journal key and paired inventory all
    identify that exact host and workload.
    """
    config_path = Path(config_path).resolve()
    evidence_path = Path(evidence_path).resolve()
    hardware_packet_path = Path(hardware_packet_path).resolve()
    deployment_manifest_path = Path(deployment_manifest_path).resolve()
    active_security_manifest_path = Path(active_security_manifest_path).resolve()
    challenge_nonce_path = Path(challenge_nonce_path).resolve()
    journal_private_key_path = Path(journal_private_key_path).resolve()
    journal_public_key_path = Path(journal_public_key_path).resolve()
    collector_manifest_path = Path(collector_manifest_path).resolve()
    collector_argv = list(collector_argv)
    collector_public_key_path = Path(collector_public_key_path).resolve()
    runtime_environment_manifest_path = Path(
        runtime_environment_manifest_path).resolve()
    trusted_store_manifest_path = Path(trusted_store_manifest_path).resolve()
    trusted_store_protection_evidence_path = Path(
        trusted_store_protection_evidence_path).resolve()
    trusted_store_root = Path(trusted_store_root).resolve()
    admission_nonce_ledger_path = Path(admission_nonce_ledger_path).resolve()
    output = Path(output).resolve()
    if (not trusted_store_root.is_dir()
            or stat.S_IMODE(trusted_store_root.stat().st_mode) != 0o700):
        raise PermissionError("trusted-store root must exist with mode 0700")
    try:
        journal_private_key_path.relative_to(trusted_store_root)
    except ValueError as exc:
        raise PermissionError("journal private key must remain in the trusted store") from exc
    try:
        admission_nonce_ledger_path.relative_to(trusted_store_root)
    except ValueError as exc:
        raise PermissionError("admission nonce ledger must remain in the trusted store") from exc
    config = _load_json_object(config_path, "workload config")
    validate_workload_config(config, execute=True)
    blockers = workload_blockers(config)
    if blockers:
        raise PermissionError(f"CC workload bindings are incomplete: {blockers}")
    device = torch.device(device)
    if device.type != "cuda" or device.index is None:
        raise PermissionError(
            "hardware engineering workloads require an explicit cuda:N device")
    frozen = _validate_execution_bindings(
        config_path, config, hardware_packet_path, deployment_manifest_path,
        active_security_manifest_path, active_security_manifest_sha256,
        challenge_nonce_path, journal_private_key_path, journal_public_key_path,
        collector_manifest_path, collector_argv, collector_public_key_path,
    )
    launch_context = require_supervisor_launch_context(
        expected_packet_sha256=frozen["packet_sha256"],
        expected_deployment_manifest_sha256=frozen["deployment_manifest_sha256"],
        expected_output_path=output,
    )
    emit_lifecycle_heartbeat(launch_context, "execution-bindings-validated")
    packet = frozen["packet"]
    common_admission_binding = {
        "source_tree_sha256": frozen["source_manifest"]["source_tree_sha256"],
        "config_sha256": frozen["config_sha256"],
        "environment_lock_sha256": config["bindings"]["uv_lock_sha256"],
        "workload_measurement_sha256": frozen["bundle_manifest"]["archive"]["sha256"],
        "journal_public_key_sha256": frozen["journal_public_key_sha256"],
        "hardware_packet_sha256": frozen["packet_sha256"],
        "collector_manifest_sha256": frozen["collector"]["manifest_sha256"],
        "collector_executable_sha256": frozen["collector"]["executable_sha256"],
        "collector_public_key_sha256": frozen["collector"]["public_key_sha256"],
    }

    if protection_mode == "cc-on":
        admission_snapshot = read_snapshot(evidence_path)
        preliminary_admission_sha256 = admission_snapshot.sha256
        preliminary_admission = load_admission_record(evidence_path)
        arrival_path = (ROOT / preliminary_admission["arrival_record"]).resolve()
        arrival = load_arrival_record(arrival_path)
        _require_arrival_matches_packet(
            arrival, packet, protection_mode, frozen["deployment_manifest_sha256"])
        require_arrival_cuda_device(arrival, str(device))
        runtime_evidence = _validate_live_runtime_evidence(
            runtime_environment_manifest_path=runtime_environment_manifest_path,
            trusted_store_manifest_path=trusted_store_manifest_path,
            trusted_store_protection_evidence_path=(
                trusted_store_protection_evidence_path),
            trusted_store_root=trusted_store_root, config=config,
            arrival=arrival, device=device,
        )
        expected_admission_binding = {
            **common_admission_binding,
            **runtime_evidence,
            "challenge_nonce_sha256": frozen["challenge_nonce_sha256"],
            "deployment_manifest_sha256": frozen["deployment_manifest_sha256"],
        }
        admission_trust_policy = _admission_trust_policy(
            packet, arrival, frozen["packet_sha256"])
        evidence = verify_cc_admission_reference(
            evidence_path, expected_session_id=admitted_session_id,
            expected_workload_binding=expected_admission_binding,
            trust_policy=admission_trust_policy,
            challenge_nonce_path=challenge_nonce_path,
        )
        require_unchanged(admission_snapshot)
        if evidence != preliminary_admission:
            raise PermissionError("admission record changed during execution gating")
        session_id = evidence["session_id"]
        evidence_id = evidence["admission_id"]
        evidence_sha256 = preliminary_admission_sha256
        hardware_attested = True
        active_admission = evidence
        active_admission_sha256 = evidence_sha256
        active_arrival = arrival
    elif protection_mode == "cc-off":
        if paired_cc_on_admission_path is None:
            raise PermissionError("CC-off execution requires the paired CC-on admission")
        paired_admission_path = Path(paired_cc_on_admission_path).resolve()
        admission_snapshot = read_snapshot(paired_admission_path)
        preliminary_admission_sha256 = admission_snapshot.sha256
        preliminary_admission = load_admission_record(paired_admission_path)
        paired_arrival = load_arrival_record(
            (ROOT / preliminary_admission["arrival_record"]).resolve())
        _require_arrival_matches_packet(
            paired_arrival, packet, "cc-on",
            preliminary_admission["workload_binding"]["deployment_manifest_sha256"])
        if paired_arrival["deployment"]["protection_mode"] != "cc-on":
            raise PermissionError("paired comparator reference is not a CC-on inventory")
        arrival = require_cc_off_inventory(evidence_path, _arrival_pair_identity(paired_arrival))
        _require_arrival_matches_packet(
            arrival, packet, protection_mode, frozen["deployment_manifest_sha256"])
        require_arrival_cuda_device(arrival, str(device))
        runtime_evidence = _validate_live_runtime_evidence(
            runtime_environment_manifest_path=runtime_environment_manifest_path,
            trusted_store_manifest_path=trusted_store_manifest_path,
            trusted_store_protection_evidence_path=(
                trusted_store_protection_evidence_path),
            trusted_store_root=trusted_store_root, config=config,
            arrival=arrival, device=device,
        )
        arm_specific = {
            key: preliminary_admission["workload_binding"][key]
            for key in (
                "runtime_environment_sha256", "trusted_store_manifest_sha256",
                "trusted_store_protection_evidence_sha256",
                "challenge_nonce_sha256", "deployment_manifest_sha256",
            )
        }
        expected_paired_binding = {**common_admission_binding, **arm_specific}
        paired_admission = verify_cc_admission_reference(
            paired_admission_path, expected_session_id=admitted_session_id,
            expected_workload_binding=expected_paired_binding,
            trust_policy=_admission_trust_policy(
                packet, paired_arrival, frozen["packet_sha256"]),
            challenge_nonce_path=challenge_nonce_path,
        )
        require_unchanged(admission_snapshot)
        if paired_admission != preliminary_admission:
            raise PermissionError("paired CC-on admission changed during execution gating")
        session_id = _cc_off_session_id(arrival, digest(evidence_path))
        evidence_id = "unattested-comparator-inventory"
        evidence_sha256 = digest(evidence_path)
        hardware_attested = False
        active_admission = paired_admission
        active_admission_sha256 = preliminary_admission_sha256
        active_arrival = paired_arrival
    else:
        raise ValueError("protection_mode must be cc-on or cc-off")
    if frozen["active_security"]["inputs"]["deployment_manifest_sha256"] != \
            active_admission["workload_binding"]["deployment_manifest_sha256"]:
        raise PermissionError(
            "active-security deployment differs from the trusted CC-on admission")
    expected_active_runtime = {
        "schema": "p4-cc-active-runtime-binding-0.1",
        "deployment_manifest_sha256": active_admission["workload_binding"][
            "deployment_manifest_sha256"],
        "admission_id": active_admission["admission_id"],
        "admission_sha256": active_admission_sha256,
        "session_id": active_admission["session_id"],
        "challenge_nonce_sha256": active_admission["workload_binding"][
            "challenge_nonce_sha256"],
        "arrival_record_sha256": active_admission["arrival_record_sha256"],
        "instance_id": active_arrival["deployment"]["instance_id"],
        "protection_mode": "cc-on",
        "hardware_packet_sha256": frozen["packet_sha256"],
        "workload_config_sha256": frozen["config_sha256"],
        "source_tree_sha256": frozen["source_manifest"]["source_tree_sha256"],
    }
    if frozen["active_security"]["inputs"].get("runtime_binding") != \
            expected_active_runtime:
        raise PermissionError(
            "active-security gate is stale or replayed against a different admitted runtime")
    if protection_mode == "cc-on":
        consumed_admission = require_cc_admission(
            evidence_path, expected_session_id=admitted_session_id,
            expected_workload_binding=expected_admission_binding,
            trust_policy=admission_trust_policy,
            challenge_nonce_path=challenge_nonce_path,
            nonce_ledger_path=admission_nonce_ledger_path,
        )
        require_unchanged(admission_snapshot)
        if consumed_admission != evidence:
            raise PermissionError("admission changed before nonce consumption")
    emit_lifecycle_heartbeat(launch_context, "admission-validated")
    pair_identity = _arrival_pair_identity(arrival)
    hardware_identity_sha256 = _canonical_sha256({
        **pair_identity,
        "cpu_model": arrival["cpu"]["model"],
    })

    output = _prepare_trusted_output(output, trusted_store_root, required=True)
    output.mkdir(parents=True, exist_ok=False)
    os.chmod(output, 0o700)
    runtime_evidence_root = output / "runtime-evidence"
    runtime_evidence_root.mkdir(mode=0o700)
    copied_runtime_evidence = {}
    for role, source, destination in (
        ("runtime_environment", runtime_environment_manifest_path,
         runtime_evidence_root / "runtime_environment.json"),
        ("trusted_store_manifest", trusted_store_manifest_path,
         runtime_evidence_root / "trusted_store_manifest.json"),
        ("trusted_store_protection", trusted_store_protection_evidence_path,
         runtime_evidence_root / "trusted_store_protection_evidence.bin"),
    ):
        expected_sha256 = {
            "runtime_environment": runtime_evidence["runtime_environment_sha256"],
            "trusted_store_manifest": runtime_evidence[
                "trusted_store_manifest_sha256"],
            "trusted_store_protection": runtime_evidence[
                "trusted_store_protection_evidence_sha256"],
        }[role]
        snapshot = read_snapshot(source, expected_sha256=expected_sha256)
        write_bytes_exclusive(destination, snapshot.data)
        require_unchanged(snapshot)
        copied_runtime_evidence[role] = destination
    started = time.monotonic()
    deadline = started + config["execution"]["maximum_wall_minutes"] * 60
    copied_active_manifest = _copy_active_security_evidence(
        active_security_manifest_path, frozen["active_security"], output,
        frozen["active_security_sha256"])
    collector = ExternalCollectorClient(
        collector_manifest_path, collector_argv, collector_public_key_path,
        session_id=session_id, admission_sha256=evidence_sha256,
        workload_config_sha256=frozen["config_sha256"],
        stderr_path=output / "external_collector.stderr.log",
        production=True, event_validator=_validate_external_event,
    )

    panels = load_deployment_panels(config)
    reference_model, source_cfg, checkpoint_manifest = make_deployment_model(
        config, device="cpu")
    candidate_model, _, _ = make_deployment_model(config, device="cpu")
    install_cc_bridge(candidate_model, device, setup_purpose="checkpoint-restore")
    windows = config["execution"]["correctness_windows"]
    evaluation = panels["evaluation"]
    correctness_binding = {
        "pair_id": arrival["deployment"]["pair_id"],
        "session_id": session_id,
        "config_sha256": frozen["config_sha256"],
        "checkpoint_sha256": config["bindings"]["checkpoint_artifact_sha256"],
        "panels_sha256": config["bindings"]["panels_artifact_sha256"],
        "hardware_identity_sha256": hardware_identity_sha256,
        "pair_manifest_sha256": arrival["deployment"]["pair_manifest_sha256"],
        "deployment_manifest_sha256": frozen["deployment_manifest_sha256"],
        "protection_mode": protection_mode,
    }
    correctness = run_paired_correctness(
        reference_model, candidate_model,
        evaluation["tokens"][:windows].cpu(),
        evaluation["next_tokens"][:windows].cpu(),
        config["execution"]["correctness_batch_size"],
        config["tolerances"], correctness_binding,
    )
    correctness_path = _write_json(output / "correctness_comparison.json", correctness)
    os.chmod(correctness_path, 0o600)
    if correctness["status"] != "PASS":
        collector.terminate()
        raise PermissionError("same-checkpoint CPU/CC bridge correctness comparison failed")
    emit_lifecycle_heartbeat(launch_context, "correctness-complete")

    corpus = load_pinned_corpus(source_cfg["data"])
    index = load_document_index(source_cfg["data"])
    region = contiguous_region(corpus, index["train"])
    from pcr.data import BatchSampler
    sampler = BatchSampler(
        region, config["anchor"]["batch_size"],
        config["anchor"]["sequence_length"],
        config["execution"]["engineering_seed"],
        "cc-engineering-complete-step",
    )
    steps_per_rep = (
        config["execution"]["warmup_steps"]
        + config["execution"]["measured_steps"])
    repetitions = []
    combined_rows: list[dict[str, Any]] = []
    checkpoint_paths: list[Path] = []
    optimizer_cfg = OptimConfig(**source_cfg["optimizer"], steps=steps_per_rep)
    for repetition in range(config["execution"]["repetitions"]):
        if time.monotonic() >= deadline:
            collector.terminate()
            raise TimeoutError("engineering workload exceeded the frozen wall-time cap")
        model, _, _ = make_deployment_model(config, device="cpu")
        emit_lifecycle_heartbeat(
            launch_context, "repetition-start", step_id=repetition,
        )
        install_cc_bridge(model, device, setup_purpose="checkpoint-restore")
        offset = repetition * steps_per_rep
        batches = [sampler.batch(offset + step) for step in range(steps_per_rep)]
        result = run_training_probe(
            model, batches, optimizer_cfg, output / f"repetition-{repetition:02d}",
            session_id, device=device, start_step=offset,
            warmup_steps=config["execution"]["warmup_steps"],
            trusted_store_root=trusted_store_root, require_hardware_bridge=True,
            deadline_monotonic=deadline,
            journal_auth_context={
                "admission_sha256": evidence_sha256,
                "workload_config_sha256": frozen["config_sha256"],
                "private_key_path": journal_private_key_path,
                "public_key_path": journal_public_key_path,
            },
            external_collector=collector,
            collector_journal_id=f"repetition-{repetition:02d}",
            heartbeat_callback=lambda stage, step: emit_lifecycle_heartbeat(
                launch_context, stage, step_id=step,
            ),
        )
        combined_rows.extend({"repetition": repetition, **row} for row in result["rows"])
        checkpoint_paths.extend(sorted((output / f"repetition-{repetition:02d}").glob(
            "checkpoint-step-*.pt")))
        repetitions.append({
            "repetition": repetition,
            "trusted_journal": str(result["trusted_journal"].relative_to(output)),
            "trusted_journal_auth": str(result["journal_auth"].relative_to(output)),
            "external_journal": str(result["external_journal"].relative_to(output)),
            "external_journal_auth": str(
                result["external_journal_auth"].relative_to(output)),
            "collector_journal_id": f"repetition-{repetition:02d}",
            "metrics": str(result["metrics"].relative_to(output)),
            "events": result["event_count"],
            "control_transfer_summary": result["control_transfer_summary"],
        })
        emit_lifecycle_heartbeat(
            launch_context, "repetition-complete", step_id=repetition,
        )
    collector.close()

    aggregate_metrics_path = _write_json(output / "step_metrics.json", {
        "schema_version": "p4-cc-aggregate-step-metrics-0.1",
        "session_id": session_id,
        "scientific_evidence": False,
        "rows": combined_rows,
    })
    os.chmod(aggregate_metrics_path, 0o600)
    source_snapshot_path = _write_json(output / "source_snapshot.json", {
        "schema_version": "p4-cc-source-snapshot-0.1",
        "scientific_evidence": False,
        "repository_revisions": frozen["source_manifest"]["repositories"],
        "source_manifest_file_sha256": frozen["source_manifest_file_sha256"],
        "source_tree_sha256": frozen["source_manifest"]["source_tree_sha256"],
        "deployment_bundle_manifest_sha256": frozen["bundle_manifest_file_sha256"],
        "deployment_archive_sha256": frozen["bundle_manifest"]["archive"]["sha256"],
        "workload_config_sha256": frozen["config_sha256"],
        "hardware_packet_sha256": frozen["packet_sha256"],
        "deployment_manifest_sha256": frozen["deployment_manifest_sha256"],
        "runtime_environment_sha256": runtime_evidence[
            "runtime_environment_sha256"],
        "trusted_store_manifest_sha256": runtime_evidence[
            "trusted_store_manifest_sha256"],
        "trusted_store_protection_evidence_sha256": runtime_evidence[
            "trusted_store_protection_evidence_sha256"],
        "attack_plan_sha256": config["bindings"]["attack_plan_sha256"],
        "active_security_manifest_sha256": frozen["active_security_sha256"],
    })
    os.chmod(source_snapshot_path, 0o600)

    summary = {
        "schema_version": "p4-cc-engineering-execution-0.2",
        "protocol_id": PROTOCOL_ID,
        "protection_mode": protection_mode,
        "hardware_attested": hardware_attested,
        "scientific_evidence": False,
        "privacy_outcome": "undetermined",
        "session_id": session_id,
        "pair_id": arrival["deployment"]["pair_id"],
        "evidence_id": evidence_id,
        "evidence_sha256": evidence_sha256,
        "source_checkpoint_run_id": checkpoint_manifest["run_id"],
        "correctness_comparison": "correctness_comparison.json",
        "correctness_status": correctness["status"],
        "source_snapshot": "source_snapshot.json",
        "active_security_manifest": str(copied_active_manifest.relative_to(output)),
        "active_security_status": frozen["active_security"]["outcome"]["status"],
        "collector_manifest_sha256": frozen["collector"]["manifest_sha256"],
        "collector_executable_sha256": frozen["collector"]["executable_sha256"],
        "collector_public_key_sha256": frozen["collector"]["public_key_sha256"],
        "runtime_evidence": {
            role: str(path.relative_to(output))
            for role, path in sorted(copied_runtime_evidence.items())
        },
        "repetitions": repetitions,
        "artifact_manifest": "artifact_manifest.json",
        "sealed_repetition_manifest": "sealed/repetition-set.json",
        "sealed_repetition_manifest_auth": "sealed/repetition-set.auth.json",
        "sealed": True,
        "next_required_action": "validate the signed three-repetition engineering bundle set",
        "wall_seconds": time.monotonic() - started,
        "claim_scope": CLAIM_SCOPE,
    }
    summary_path = _write_json(output / "execution_summary.json", summary)
    os.chmod(summary_path, 0o600)
    artifacts = [
        _artifact_entry(correctness_path, output, "correctness_comparison"),
        _artifact_entry(summary_path, output, "execution_summary"),
        _artifact_entry(aggregate_metrics_path, output, "step_metrics"),
        _artifact_entry(source_snapshot_path, output, "source_snapshot"),
        _artifact_entry(copied_active_manifest, output, "active_security_gate"),
    ]
    artifacts.extend(
        _artifact_entry(path, output, role, sensitivity="trusted-sensitive")
        for role, path in sorted(copied_runtime_evidence.items())
    )
    artifacts.extend(
        _artifact_entry(path, output, "checkpoint", sensitivity="trusted-sensitive")
        for path in checkpoint_paths
    )
    artifact_manifest_path = _write_json(output / "artifact_manifest.json", {
        "schema_version": "p4-cc-seal-artifact-manifest-0.1",
        "artifacts": artifacts,
    })
    os.chmod(artifact_manifest_path, 0o600)
    if len(repetitions) != 3 or [row["repetition"] for row in repetitions] != [0, 1, 2]:
        raise RuntimeError("runner must seal exactly three ordered repetitions")
    seal_binding = {
        "admission_id": evidence_id,
        "admission_sha256": evidence_sha256,
        "session_id": session_id,
        "hardware_attested": hardware_attested,
        "workload_config_sha256": frozen["config_sha256"],
        "journal_public_key_sha256": frozen["journal_public_key_sha256"],
        "collector_manifest_sha256": frozen["collector"]["manifest_sha256"],
        "collector_executable_sha256": frozen["collector"]["executable_sha256"],
        "collector_public_key_sha256": frozen["collector"]["public_key_sha256"],
    }
    sealed_root = output / "sealed"
    sealed_root.mkdir(mode=0o700)
    sealed_rows = []
    for repetition in repetitions:
        bundle = seal_journals(
            output / repetition["trusted_journal"],
            output / repetition["external_journal"],
            output / repetition["trusted_journal_auth"],
            journal_public_key_path,
            output / repetition["external_journal_auth"],
            collector_manifest_path, collector_public_key_path,
            repetition["collector_journal_id"], seal_binding,
            protection_mode, artifacts, output,
        )
        bundle_path = _write_json(
            sealed_root / f"repetition-{repetition['repetition']:02d}.bundle.json",
            bundle,
        )
        os.chmod(bundle_path, 0o600)
        sealed_rows.append({
            "repetition": repetition["repetition"],
            "journal_id": repetition["collector_journal_id"],
            "path": str(bundle_path.relative_to(output)),
            "sha256": digest(bundle_path),
            "bytes": bundle_path.stat().st_size,
            "event_pair_chain_sha256": bundle["event_pair_chain_sha256"],
        })
    repetition_manifest_path = _write_json(sealed_root / "repetition-set.json", {
        "schema_version": "p4-cc-sealed-repetition-set-0.1",
        "protocol_id": PROTOCOL_ID,
        "session_id": session_id,
        "pair_id": arrival["deployment"]["pair_id"],
        "protection_mode": protection_mode,
        "scientific_evidence": False,
        "admission_sha256": evidence_sha256,
        "workload_config_sha256": frozen["config_sha256"],
        "hardware_packet_sha256": frozen["packet_sha256"],
        "artifact_manifest_sha256": digest(artifact_manifest_path),
        "correctness_comparison_sha256": digest(correctness_path),
        "source_snapshot_sha256": digest(source_snapshot_path),
        "active_security_manifest_sha256": digest(copied_active_manifest),
        "repetition_count": 3,
        "repetitions": sealed_rows,
        "claim_scope": CLAIM_SCOPE,
    })
    os.chmod(repetition_manifest_path, 0o600)
    repetition_auth_path = authenticate_execution_manifest(
        repetition_manifest_path, sealed_root / "repetition-set.auth.json",
        session_id=session_id, pair_id=arrival["deployment"]["pair_id"],
        protection_mode=protection_mode, admission_sha256=evidence_sha256,
        workload_config_sha256=frozen["config_sha256"],
        hardware_packet_sha256=frozen["packet_sha256"],
        private_key_path=journal_private_key_path,
        public_key_path=journal_public_key_path,
    )
    validate_sealed_repetition_set(
        repetition_manifest_path, repetition_auth_path, journal_public_key_path,
        expected_session_id=session_id,
        expected_pair_id=arrival["deployment"]["pair_id"],
        expected_protection_mode=protection_mode,
        expected_admission_sha256=evidence_sha256,
        expected_workload_config_sha256=frozen["config_sha256"],
        expected_hardware_packet_sha256=frozen["packet_sha256"],
    )
    emit_lifecycle_heartbeat(launch_context, "workload-artifacts-complete")
    return summary
