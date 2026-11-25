"""Fail-closed, two-view records for pre-hardware secure-compute diagnostics.

This module records what a hypothetical external infrastructure operator observes
separately from plaintext available only to trusted diagnostics.  Version 0.1 is
deliberately limited to the d0-q8 in-process plaintext loopback: it cannot claim
encryption, attestation, confidential hardware, privacy, or scientific evidence.

No raw tensor or packet payload is written to the JSON bundle.  The external view
contains the digest and metadata of the packet it observes; the trusted view adds
the canonical tensor digest.  Every event is paired by correlation ID, bound to a
session, request and step, and every declared phase must be recorded or explicitly
marked not exercised/not applicable before a bundle can be emitted.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

import jsonschema
import torch


ROOT = Path(__file__).resolve().parents[2]
EVENT_SCHEMA_VERSION = "p4-secure-compute-event-0.1"
BUNDLE_SCHEMA_VERSION = "p4-secure-compute-bundle-0.1"
CLAIM_SCOPE = ("No-hardware two-view recorder diagnostic; no confidentiality, integrity, "
               "attestation, privacy, or defense claim.")
PHASES = ("forward", "backward", "update", "optimizer", "staging", "log",
          "checkpoint", "crash")
DIRECTIONS = ("trusted_to_external", "external_to_trusted", "trusted_internal",
              "external_internal")
ABSENT_STATUSES = ("not_exercised", "not_applicable")
IDENTIFIER = re.compile(r"^[A-Za-z0-9_.:-]+$")
SHA256 = re.compile(r"^[a-f0-9]{64}$")
EVENT_SCHEMA_PATH = ROOT / "experiments/schemas/secure_compute_event.schema.json"
BUNDLE_SCHEMA_PATH = ROOT / "experiments/schemas/secure_compute_bundle.schema.json"


def _canonical_sha256(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    if (tensor.dtype != torch.float32 or tensor.device.type != "cpu"
            or not torch.isfinite(tensor).all()):
        raise ValueError("Secure recorder accepts only finite CPU float32 tensors")
    return _bytes_sha256(tensor.detach().contiguous().numpy().astype("<f4", copy=False).tobytes())


def _load_schema(path: Path) -> dict:
    schema = json.loads(path.read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    return schema


def _safe_artifact_path(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute():
        raise ValueError("Artifact path must be relative to the declared root")
    resolved_root = root.resolve()
    resolved = (resolved_root / path).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("Artifact path escapes the declared root") from exc
    return resolved


def artifact_record(path, artifact_id, view, media_type, root=ROOT) -> dict:
    """Create a hash/size-bound artifact declaration rooted at ``root``."""
    root = Path(root).resolve()
    path = Path(path).resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError("Artifact is outside the declared root") from exc
    if not IDENTIFIER.fullmatch(artifact_id):
        raise ValueError("Invalid artifact ID")
    if view not in ("external_attacker", "trusted_diagnostic", "shared_metadata"):
        raise ValueError("Invalid artifact view")
    if not path.is_file() or path.stat().st_size < 1:
        raise ValueError("Artifact must be a nonempty regular file")
    return {"artifact_id": artifact_id, "path": str(relative),
            "sha256": _bytes_sha256(path.read_bytes()), "bytes": path.stat().st_size,
            "media_type": media_type, "view": view}


def _event_chain(views: dict) -> str:
    return _canonical_sha256({
        "external_attacker": views["external_attacker"]["events"],
        "trusted_diagnostic": views["trusted_diagnostic"]["events"],
    })


class SecureComputeRecorder:
    """Build a strict no-hardware d0-q8 two-view event bundle."""

    def __init__(self, bundle_id, protocol_id, run_id, mode, session_id, bindings):
        for name, value in (("bundle_id", bundle_id), ("session_id", session_id)):
            if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
                raise ValueError(f"Invalid {name}")
        if not isinstance(protocol_id, str) or not protocol_id.startswith("p4-"):
            raise ValueError("Invalid Paper 4 protocol ID")
        if not isinstance(run_id, str) or not run_id.startswith("p4-"):
            raise ValueError("Invalid Paper 4 run ID")
        if mode not in ("trained", "frozen_random", "removed_remote"):
            raise ValueError("Recorder requires a split d0-q8 mode")
        expected_bindings = {"source_tree_sha256", "config_sha256",
                             "checkpoint_manifest_sha256", "panels_manifest_sha256",
                             "panels_sha256"}
        if set(bindings) != expected_bindings or any(
                not isinstance(value, str) or not SHA256.fullmatch(value)
                for value in bindings.values()):
            raise ValueError("Recorder bindings must be the exact five SHA-256 identities")
        self.bundle_id = bundle_id
        self.protocol_id = protocol_id
        self.run_id = run_id
        self.mode = mode
        self.session_id = session_id
        self.bindings = dict(bindings)
        self.external_events = []
        self.trusted_events = []
        self._requests = {}
        self._steps = {}
        self._absent_phases = {}

    def declare_absent_phase(self, phase, reason, status="not_exercised"):
        if phase not in PHASES:
            raise ValueError("Unknown secure-compute phase")
        if status not in ABSENT_STATUSES:
            raise ValueError("Absent phase must be not_exercised or not_applicable")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Absent phase requires a reason")
        if any(phase in event["phases"] for event in self.external_events):
            raise ValueError("Cannot declare a recorded phase absent")
        if phase in self._absent_phases:
            raise ValueError("Phase absence declared twice")
        self._absent_phases[phase] = {"status": status, "event_count": 0,
                                      "reason": reason.strip()}

    def record_pair(self, request_id, step, phases, direction, operation, operator,
                    tensor, packet, artifact_ids=()):
        """Record one correlated external/trusted observation of a boundary payload."""
        if not isinstance(request_id, str) or not IDENTIFIER.fullmatch(request_id):
            raise ValueError("Invalid request ID")
        if type(step) is not int or step < 0:
            raise ValueError("Step must be a nonnegative integer")
        phases = list(phases)
        if not phases or len(phases) != len(set(phases)) or any(p not in PHASES for p in phases):
            raise ValueError("Event phases must be a nonempty unique subset of the phase inventory")
        if any(phase in self._absent_phases for phase in phases):
            raise ValueError("Cannot record an event for a phase declared absent")
        if direction not in DIRECTIONS:
            raise ValueError("Invalid event direction")
        if not isinstance(operation, str) or not operation or not isinstance(operator, str) or not operator:
            raise ValueError("Operation and operator are required")
        if not isinstance(packet, bytes) or not packet:
            raise ValueError("Packet must be nonempty immutable bytes")
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("Trusted diagnostic payload must be a tensor")
        tensor = tensor.detach().cpu().float().contiguous()
        if not torch.isfinite(tensor).all() or not tensor.shape or any(d < 1 for d in tensor.shape):
            raise ValueError("Trusted diagnostic tensor must be finite and nonempty")
        artifact_ids = list(artifact_ids)
        if len(artifact_ids) != len(set(artifact_ids)) or any(
                not isinstance(value, str) or not IDENTIFIER.fullmatch(value)
                for value in artifact_ids):
            raise ValueError("Invalid event artifact references")
        previous = self._requests.get(request_id)
        if previous is not None and previous != step:
            raise ValueError("Request ID cannot be rebound to another step")
        other_request = self._steps.get(step)
        if other_request is not None and other_request != request_id:
            raise ValueError("Step cannot be rebound to another request")
        self._requests[request_id] = step
        self._steps[step] = request_id

        sequence = len(self.external_events)
        correlation_id = f"correlation-{sequence:08d}"
        common = {"schema_version": EVENT_SCHEMA_VERSION,
                  "correlation_id": correlation_id, "sequence": sequence,
                  "session_id": self.session_id, "request_id": request_id, "step": step,
                  "phases": phases, "direction": direction, "operation": operation,
                  "operator": operator, "artifact_ids": artifact_ids}
        packet_sha = _bytes_sha256(packet)
        shape = list(tensor.shape)
        dtype = str(tensor.dtype)
        external = {**common, "event_id": f"external-{sequence:08d}",
                    "view": "external_attacker",
                    "external_observation": {
                        "payload_state": "plaintext_unprotected",
                        "serialized_bytes": len(packet), "packet_sha256": packet_sha,
                        "shape": shape, "dtype": dtype,
                        "metadata_visible": ["session_id", "request_id", "step", "phase",
                                             "direction", "operator", "shape", "dtype",
                                             "serialized_bytes", "packet_bytes"]}}
        trusted = {**common, "event_id": f"trusted-{sequence:08d}",
                   "view": "trusted_diagnostic",
                   "trusted_diagnostic": {
                       "payload_state": "plaintext_trusted_diagnostic",
                       "serialized_bytes": len(packet), "packet_sha256": packet_sha,
                       "tensor_sha256": _tensor_sha256(tensor), "shape": shape,
                       "dtype": dtype, "finite": True}}
        event_schema = _load_schema(EVENT_SCHEMA_PATH)
        jsonschema.validate(external, event_schema)
        jsonschema.validate(trusted, event_schema)
        self.external_events.append(external)
        self.trusted_events.append(trusted)
        return correlation_id

    def build_bundle(self, artifacts, omissions):
        if not self.external_events:
            raise ValueError("Secure-compute bundle requires at least one paired event")
        counts = {phase: sum(phase in event["phases"] for event in self.external_events)
                  for phase in PHASES}
        missing = [phase for phase, count in counts.items()
                   if count == 0 and phase not in self._absent_phases]
        if missing:
            raise ValueError(f"Every unrecorded phase needs an explicit declaration: {missing}")
        coverage = {}
        for phase, count in counts.items():
            if count:
                coverage[phase] = {"status": "recorded", "event_count": count,
                                   "reason": "Paired external and trusted events recorded."}
            else:
                coverage[phase] = deepcopy(self._absent_phases[phase])
        views = {
            "external_attacker": {
                "description": ("Host-visible metadata and plaintext serialized packets in the "
                                "unprotected local loopback; this is not a confidential view."),
                "events": deepcopy(self.external_events),
            },
            "trusted_diagnostic": {
                "description": ("Plaintext diagnostic tensor identities retained only for local "
                                "correctness and positive-control checks."),
                "events": deepcopy(self.trusted_events),
            },
        }
        bundle = {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "bundle_id": self.bundle_id,
            "protocol_id": self.protocol_id,
            "run_id": self.run_id,
            "cohort": "engineering",
            "scientific_evidence": False,
            "hardware_attested": False,
            "privacy_outcome": "undetermined",
            "cell": "d0-q8",
            "mode": self.mode,
            "claim_scope": CLAIM_SCOPE,
            "deployment": {
                "environment": "local_no_hardware_diagnostic",
                "hardware": "none",
                "transport": "in_process_plaintext_serialized_loopback",
                "encryption": "none",
                "attestation": "none",
                "protection_implemented": False,
            },
            "bindings": dict(self.bindings),
            "session": {
                "session_id": self.session_id,
                "requests": [{"request_id": request_id, "step": step}
                             for request_id, step in sorted(self._requests.items(),
                                                            key=lambda item: (item[1], item[0]))],
            },
            "phase_coverage": coverage,
            "views": views,
            "artifacts": deepcopy(list(artifacts)),
            "event_chain_sha256": _event_chain(views),
            "omissions": list(omissions),
        }
        validate_secure_compute_bundle(bundle)
        return bundle


def validate_secure_compute_bundle(bundle, root=ROOT, verify_artifacts=False) -> dict:
    """Validate schema, pair/binding semantics, chain, and optionally artifact bytes."""
    bundle_schema = _load_schema(BUNDLE_SCHEMA_PATH)
    event_schema = _load_schema(EVENT_SCHEMA_PATH)
    jsonschema.validate(bundle, bundle_schema)
    external = bundle["views"]["external_attacker"]["events"]
    trusted = bundle["views"]["trusted_diagnostic"]["events"]
    for event in external + trusted:
        jsonschema.validate(event, event_schema)
    if len(external) != len(trusted):
        raise ValueError("Two-view event counts differ")
    external_by = {event["correlation_id"]: event for event in external}
    trusted_by = {event["correlation_id"]: event for event in trusted}
    if len(external_by) != len(external) or len(trusted_by) != len(trusted):
        raise ValueError("Duplicate correlation ID")
    if set(external_by) != set(trusted_by):
        raise ValueError("Every external event needs exactly one trusted pair")
    expected_sequences = list(range(len(external)))
    if ([event["sequence"] for event in external] != expected_sequences
            or [event["sequence"] for event in trusted] != expected_sequences):
        raise ValueError("View event sequences must be contiguous and aligned")
    common = ("correlation_id", "sequence", "session_id", "request_id", "step", "phases",
              "direction", "operation", "operator", "artifact_ids")
    for correlation_id, left in external_by.items():
        right = trusted_by[correlation_id]
        if any(left[key] != right[key] for key in common):
            raise ValueError("Correlated views disagree on event identity")
        observed = left["external_observation"]
        diagnostic = right["trusted_diagnostic"]
        if any(observed[key] != diagnostic[key]
               for key in ("serialized_bytes", "packet_sha256", "shape", "dtype")):
            raise ValueError("Correlated views disagree on payload metadata")

    session = bundle["session"]
    if any(event["session_id"] != session["session_id"] for event in external + trusted):
        raise ValueError("Event/session binding mismatch")
    declared_requests = {(entry["request_id"], entry["step"]) for entry in session["requests"]}
    if len(declared_requests) != len(session["requests"]):
        raise ValueError("Duplicate request binding")
    event_requests = {(event["request_id"], event["step"]) for event in external}
    if declared_requests != event_requests:
        raise ValueError("Declared requests do not exactly match recorded events")
    if len({request for request, _ in declared_requests}) != len(declared_requests):
        raise ValueError("Request ID reused across steps")
    if len({step for _, step in declared_requests}) != len(declared_requests):
        raise ValueError("Step rebound to multiple requests")

    for phase in PHASES:
        count = sum(phase in event["phases"] for event in external)
        record = bundle["phase_coverage"][phase]
        if record["event_count"] != count:
            raise ValueError("Phase coverage count differs from events")
        if (count > 0) != (record["status"] == "recorded"):
            raise ValueError("Phase coverage status differs from events")
    if bundle["event_chain_sha256"] != _event_chain(bundle["views"]):
        raise ValueError("Event-chain digest mismatch")

    artifact_ids = [artifact["artifact_id"] for artifact in bundle["artifacts"]]
    artifact_paths = [artifact["path"] for artifact in bundle["artifacts"]]
    if len(set(artifact_ids)) != len(artifact_ids) or len(set(artifact_paths)) != len(artifact_paths):
        raise ValueError("Artifact identities and paths must be unique")
    referenced = {artifact_id for event in external + trusted
                  for artifact_id in event["artifact_ids"]}
    if not referenced <= set(artifact_ids):
        raise ValueError("Event references an undeclared artifact")
    if verify_artifacts:
        root = Path(root)
        for artifact in bundle["artifacts"]:
            path = _safe_artifact_path(root, artifact["path"])
            if (not path.is_file() or path.stat().st_size != artifact["bytes"]
                    or _bytes_sha256(path.read_bytes()) != artifact["sha256"]):
                raise ValueError(f"Missing or changed secure-compute artifact: {artifact['path']}")
    return bundle


def write_secure_compute_bundle(path, bundle, root=ROOT):
    """Validate fully before writing; then reload and validate the serialized form."""
    validate_secure_compute_bundle(bundle, root=root, verify_artifacts=True)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle, indent=2, sort_keys=True, allow_nan=False) + "\n")
    loaded = json.loads(path.read_text())
    validate_secure_compute_bundle(loaded, root=root, verify_artifacts=True)
    return path


def load_secure_compute_bundle(path, root=ROOT) -> dict:
    bundle = json.loads(Path(path).read_text())
    return validate_secure_compute_bundle(bundle, root=root, verify_artifacts=True)
