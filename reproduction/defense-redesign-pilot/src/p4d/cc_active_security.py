"""Fail-closed bridge to the LLM-Attacker confidential-compute gate.

The bridge has two jobs and deliberately does neither provisioning nor scientific
execution:

* run one already-frozen ``secure-compute-audit`` plan through an explicitly
  frozen process-adapter argv; and
* independently verify the resulting artifact and evidence bundle before
  emitting a compact Paper 4 admission manifest.

The LLM-Attacker exit status is necessary but is not trusted as sufficient.  This
module rechecks plan/deployment identities, every case/result binding, every
referenced evidence byte, and all zero-tolerance counters.  Any missing,
unavailable, malformed, stale, dirty-tree, timed-out, or non-PASS condition raises
``ActiveSecurityError`` and no passing manifest is written.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
from typing import Sequence

from .cc_admission import (
    load_admission_record,
    load_arrival_record,
    validate_challenge_nonce,
)
from .cc_io import read_json_snapshot, write_json_exclusive


ACTIVE_RESULT_SCHEMA = "p4-cc-active-security-result-0.1"
ACTIVE_MANIFEST_SCHEMA = "p4-cc-active-security-manifest-0.1"
ACTIVE_RUNTIME_SCHEMA = "p4-cc-active-runtime-binding-0.1"
ATTACK_ARTIFACT_SCHEMA = "dtraining.attacker.secure_compute_audit.v1"
PLAN_SCHEMA = "llm_attacker.secure_compute_plan.v1"
CASE_SCHEMA = "llm_attacker.secure_compute_case.v1"
RESULT_SCHEMA = "llm_attacker.secure_compute_result.v1"
GATE_SCHEMA = "llm_attacker.secure_compute_gate.v1"
DEPLOYMENT_SCHEMA = "llm_attacker.secure_compute_deployment.v1"
EVIDENCE_MANIFEST_SCHEMA = "llm_attacker.secure_compute_evidence_manifest.v1"
FRAMEWORK_EVIDENCE_SCHEMA = "llm_attacker.evidence.v2"
RUNTIME_BINDING_KEYS = {
    "schema", "deployment_manifest_sha256", "admission_id",
    "admission_sha256", "session_id", "challenge_nonce_sha256",
    "arrival_record_sha256", "instance_id", "protection_mode",
    "hardware_packet_sha256", "workload_config_sha256", "source_tree_sha256",
}

PROFILES = ("tee", "fhe", "hybrid")
VIEWS = ("external_attacker", "trusted_diagnostic")
ATTESTATION_ARMS = (
    "valid", "wrong-measurement", "modified-quote", "stale-collateral",
    "replayed-nonce", "wrong-machine-relay", "endpoint-key-substitution",
    "channel-binding-mismatch",
)
PROTOCOL_ARMS = (
    "clean-control", "replay", "reorder", "duplicate", "truncate",
    "cross-session-swap", "downgrade", "tamper", "stale-step-result",
    "request-result-substitution",
)
COMPUTATION_ARMS = (
    "correct-result-control", "stale-weights", "substituted-weights",
    "skipped-operator", "scaled-gradient", "zeroed-gradient",
    "replayed-output", "incorrect-result",
)
BOUNDARY_PHASES = (
    "forward", "backward", "update", "cpu-gpu-staging", "ipc", "vram",
    "logs", "checkpoints", "dumps", "environment", "telemetry",
)
HOST_TRACE_ARMS = (
    "packet-shape", "timing", "performance-counters", "access-trace",
    "crash-restart",
)
FHE_ARMS = (
    "malformed-ciphertext", "modified-ciphertext", "replayed-ciphertext",
    "wrong-context", "extra-slot", "cross-tenant-context",
    "production-feedback", "leaky-positive-control",
)

CASE_KEYS = {
    "schema", "case_id", "category", "arm", "profile", "expected",
    "view", "request_id", "session_id", "step_id", "canary_id",
    "required", "payload", "plan_sha256",
}
RESULT_KEYS = {
    "schema", "case_id", "case_sha256", "plan_sha256",
    "deployment_manifest_sha256", "evidence_manifest_sha256", "accepted",
    "secret_released", "canary_observed", "signal_observed", "status",
    "error_code", "request_id", "session_id", "step_id", "evidence",
    "details",
}
EVIDENCE_FIELDS = {
    "attestation-key-release": (
        "quote", "collateral", "firmware", "verifier_policy", "nonce",
        "endpoint_key", "decision",
    ),
    "protocol-integrity": ("transcript", "mutation", "decision"),
    "malicious-computation": ("request", "mutation", "result", "decision"),
    "boundary-canary": ("observation", "canary_manifest"),
    "confidential-host-trace": ("trace", "collection_policy"),
    "fhe-feedback-oracle": ("ciphertext", "context", "feedback", "decision"),
}

SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
GIT_REVISION_RE = re.compile(r"^[a-f0-9]{40}(?:[a-f0-9]{24})?$")
CLAIM_SCOPE = (
    "P0 engineering admission for one exact confidential-compute deployment and "
    "finite active-security plan; not scientific evidence and not a general "
    "security or privacy certificate."
)


class ActiveSecurityError(RuntimeError):
    """A fail-closed active-security precondition or result failure."""


def _canonical_bytes(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ActiveSecurityError(f"Cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _require_sha256(value, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ActiveSecurityError(f"{label} must be 64 lowercase hexadecimal characters")
    return value


def _load_json(path, label: str, expected_sha256: str | None = None) -> dict:
    try:
        value, _snapshot = read_json_snapshot(path, expected_sha256=expected_sha256)
    except (OSError, ValueError) as exc:
        raise ActiveSecurityError(f"Cannot read {label}: {exc}") from exc
    return value


def _safe_bundle_path(root: Path, relative, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ActiveSecurityError(f"{label} path must be a nonempty relative string")
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ActiveSecurityError(f"Unsafe {label} path: {relative!r}")
    root = root.resolve()
    candidate = root / relative_path
    if candidate.is_symlink():
        raise ActiveSecurityError(f"{label} cannot be a symlink")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ActiveSecurityError(f"{label} path escapes its root") from exc
    return resolved


def _expected_contract(category: str, arm: str, view: str) -> str:
    if category == "attestation-key-release" and arm in ATTESTATION_ARMS:
        return "allow-release" if arm == "valid" else "deny-release"
    if category == "protocol-integrity" and arm in PROTOCOL_ARMS:
        return "accept" if arm == "clean-control" else "reject"
    if category == "malicious-computation" and arm in COMPUTATION_ARMS:
        return "accept" if arm == "correct-result-control" else "reject"
    if category == "boundary-canary" and arm in BOUNDARY_PHASES and view in VIEWS:
        return "observe-canary" if view == "trusted_diagnostic" else "no-canary"
    if category == "confidential-host-trace" and arm in HOST_TRACE_ARMS:
        return "observe-or-declare"
    if category == "fhe-feedback-oracle" and arm in FHE_ARMS:
        if arm == "leaky-positive-control":
            return "signal"
        if arm == "production-feedback":
            return "no-signal"
        return "reject"
    raise ActiveSecurityError(f"Unknown active-security coordinate {category!r}/{arm!r}")


def _expected_case_ids(profile: str) -> set[str]:
    identifiers = {
        f"attestation-key-release:{arm}:external_attacker"
        for arm in ATTESTATION_ARMS
    }
    identifiers |= {
        f"protocol-integrity:{arm}:external_attacker" for arm in PROTOCOL_ARMS
    }
    identifiers |= {
        f"malicious-computation:{arm}:external_attacker"
        for arm in COMPUTATION_ARMS
    }
    identifiers |= {
        f"boundary-canary:{phase}:{view}"
        for phase in BOUNDARY_PHASES for view in VIEWS
    }
    identifiers |= {
        f"confidential-host-trace:{arm}:external_attacker"
        for arm in HOST_TRACE_ARMS
    }
    if profile in ("fhe", "hybrid"):
        identifiers |= {
            f"fhe-feedback-oracle:{arm}:external_attacker" for arm in FHE_ARMS
        }
    return identifiers


def _plan_digest(plan: dict) -> str:
    body = {key: value for key, value in plan.items() if key != "sha256"}
    body["cases"] = [
        {**case, "plan_sha256": None} for case in body.get("cases", [])
    ]
    return _canonical_sha256(body)


def _case_digest(case: dict) -> str:
    return _canonical_sha256(case)


def validate_frozen_plan(plan: dict, deployment_sha256: str | None,
                         profile: str | None = None) -> dict:
    """Independently validate the complete finite LLM-Attacker plan."""
    expected_keys = {
        "schema", "profile", "gate_schema", "require_host_traces",
        "deployment_manifest_sha256", "cases", "sha256",
    }
    if set(plan) != expected_keys or plan.get("schema") != PLAN_SCHEMA:
        raise ActiveSecurityError("Frozen plan has the wrong schema or field inventory")
    if plan.get("gate_schema") != GATE_SCHEMA:
        raise ActiveSecurityError("Frozen plan names an unexpected gate schema")
    if plan.get("profile") not in PROFILES:
        raise ActiveSecurityError("Frozen plan has an invalid profile")
    if profile is not None and plan["profile"] != profile:
        raise ActiveSecurityError("Frozen plan profile does not match the requested profile")
    if not isinstance(plan.get("require_host_traces"), bool):
        raise ActiveSecurityError("Frozen plan require_host_traces must be boolean")
    if plan.get("deployment_manifest_sha256") != deployment_sha256:
        raise ActiveSecurityError("Frozen plan deployment binding mismatch")
    cases = plan.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ActiveSecurityError("Frozen plan cases must be a nonempty list")
    plan_sha256 = _require_sha256(plan.get("sha256"), "Frozen plan identity")
    if _plan_digest(plan) != plan_sha256:
        raise ActiveSecurityError("Frozen plan logical digest mismatch")

    seen: set[str] = set()
    requests: set[str] = set()
    for index, case in enumerate(cases):
        if not isinstance(case, dict) or set(case) != CASE_KEYS:
            raise ActiveSecurityError(f"Frozen plan case {index} has the wrong fields")
        if case.get("schema") != CASE_SCHEMA or case.get("profile") != plan["profile"]:
            raise ActiveSecurityError(f"Frozen plan case {index} schema/profile mismatch")
        if case.get("view") not in VIEWS:
            raise ActiveSecurityError(f"Frozen plan case {index} has an invalid view")
        try:
            expected = _expected_contract(case["category"], case["arm"], case["view"])
        except (KeyError, TypeError) as exc:
            raise ActiveSecurityError(f"Frozen plan case {index} is malformed") from exc
        canonical_id = f"{case['category']}:{case['arm']}:{case['view']}"
        if case.get("case_id") != canonical_id or case.get("expected") != expected:
            raise ActiveSecurityError(f"Frozen plan case {index} changes the case contract")
        if case["case_id"] in seen:
            raise ActiveSecurityError(f"Frozen plan duplicates {case['case_id']!r}")
        seen.add(case["case_id"])
        request_id = case.get("request_id")
        if not isinstance(request_id, str) or not request_id or request_id in requests:
            raise ActiveSecurityError("Frozen plan request IDs must be nonempty and unique")
        requests.add(request_id)
        if not isinstance(case.get("session_id"), str) or not case["session_id"]:
            raise ActiveSecurityError("Frozen plan session IDs must be nonempty")
        if case.get("step_id") != index:
            raise ActiveSecurityError("Frozen plan steps are not in canonical order")
        if case.get("plan_sha256") != plan_sha256:
            raise ActiveSecurityError("Frozen plan case identity mismatch")
        if not isinstance(case.get("payload"), dict) or case["payload"].get(
                "deployment_manifest_sha256") != deployment_sha256:
            raise ActiveSecurityError("Frozen plan case deployment binding mismatch")
        should_require = (
            plan["require_host_traces"]
            if case["category"] == "confidential-host-trace" else True
        )
        if case.get("required") is not should_require:
            raise ActiveSecurityError("Frozen plan case required flag weakens the plan")
        if case["category"] == "boundary-canary":
            if case.get("canary_id") != f"canary-{case['arm']}":
                raise ActiveSecurityError("Frozen plan boundary canary binding mismatch")
        elif case.get("canary_id") is not None:
            raise ActiveSecurityError("Only boundary cases may bind canary IDs")

    if seen != _expected_case_ids(plan["profile"]):
        raise ActiveSecurityError("Frozen plan does not contain the exact v1 case inventory")
    return plan


def materialize_frozen_plan(plan_template: dict, deployment_sha256: str,
                            profile: str | None = None) -> dict:
    """Bind a pre-order plan template to one admitted deployment.

    The exact case inventory must be frozen before hardware is ordered, while
    the deployment-manifest digest necessarily includes the later arrival and
    admission evidence.  A template therefore carries ``null`` in only the
    deployment binding.  Materialisation changes that binding and the derived
    logical plan digest; it cannot add, remove, reorder, or weaken a case.

    Already-bound plans remain accepted for backwards-compatible verification,
    but a plan bound to any other deployment is rejected.
    """
    deployment_sha256 = _require_sha256(
        deployment_sha256, "Deployment manifest digest")
    bound = plan_template.get("deployment_manifest_sha256")
    if bound is not None:
        if bound != deployment_sha256:
            raise ActiveSecurityError(
                "Frozen plan is bound to a different deployment manifest")
        return validate_frozen_plan(plan_template, deployment_sha256, profile)

    validate_frozen_plan(plan_template, None, profile)
    plan = deepcopy(plan_template)
    plan["deployment_manifest_sha256"] = deployment_sha256
    for case in plan["cases"]:
        case["payload"]["deployment_manifest_sha256"] = deployment_sha256
        case["plan_sha256"] = None
    plan["sha256"] = _plan_digest(plan)
    for case in plan["cases"]:
        case["plan_sha256"] = plan["sha256"]
    return validate_frozen_plan(plan, deployment_sha256, profile)


def validate_deployment_manifest(document: dict, deployment_sha256: str,
                                 profile: str, driver_argv: Sequence[str]) -> dict:
    """Check the static, pre-admission deployment contract.

    Arrival/admission identities are deliberately absent: both later records
    bind this file's digest, so embedding their digests here would create an
    impossible mutual hash dependency.
    """
    if document.get("schema") != DEPLOYMENT_SCHEMA:
        raise ActiveSecurityError("Deployment manifest schema mismatch")
    if document.get("ready_for_execution") is not True:
        raise ActiveSecurityError("Deployment manifest is not frozen for execution")
    if document.get("profile") != profile:
        raise ActiveSecurityError("Deployment manifest profile mismatch")
    adapter = document.get("adapter")
    if not isinstance(adapter, dict) or adapter.get("command_argv") != list(driver_argv):
        raise ActiveSecurityError("Explicit driver argv differs from the deployment manifest")
    for key in ("provider", "target_factory", "implementation_revision",
                "credential_mode"):
        if not isinstance(adapter.get(key), str) or not adapter[key]:
            raise ActiveSecurityError(f"Deployment adapter.{key} is not frozen")
    if document.get("hardware", {}).get("debug_enabled") is not False:
        raise ActiveSecurityError("Deployment debug mode must be disabled")
    gates = document.get("gates")
    if not isinstance(gates, dict) or gates.get("valid_release_required") is not True:
        raise ActiveSecurityError("Deployment does not require the valid-release control")
    for key in ("invalid_release_tolerance", "external_canary_tolerance",
                "mutation_accept_tolerance"):
        if gates.get(key) != 0:
            raise ActiveSecurityError(f"Deployment gates.{key} must be exactly zero")
    _require_sha256(deployment_sha256, "Deployment manifest digest")
    return document


def validate_active_runtime_binding(document: dict, deployment_sha256: str) -> dict:
    """Validate the post-admission one-way binding to the static deployment."""
    if (not isinstance(document, dict) or set(document) != RUNTIME_BINDING_KEYS
            or document.get("schema") != ACTIVE_RUNTIME_SCHEMA):
        raise ActiveSecurityError("Active runtime binding fields/schema differ")
    _require_sha256(deployment_sha256, "Deployment manifest digest")
    if document.get("deployment_manifest_sha256") != deployment_sha256:
        raise ActiveSecurityError("Active runtime deployment binding differs")
    for key in ("admission_sha256", "challenge_nonce_sha256",
                "arrival_record_sha256", "hardware_packet_sha256",
                "workload_config_sha256", "source_tree_sha256"):
        _require_sha256(document.get(key), f"Active runtime binding {key}")
    for key in ("admission_id", "session_id", "instance_id"):
        if not isinstance(document.get(key), str) or not document[key].strip():
            raise ActiveSecurityError(f"Active runtime binding {key} is missing")
    if document.get("protection_mode") != "cc-on":
        raise ActiveSecurityError("Active runtime binding is not CC-on")
    return document


def build_active_runtime_binding(
        *, admission_path, evidence_root, deployment_manifest_path,
        hardware_packet_path, workload_config_path, source_manifest_path,
        challenge_nonce_path, expected_session_id: str) -> dict:
    """Construct the post-admission binding without modifying earlier bytes.

    This performs structural and content-address checks.  The engineering
    runner later repeats admission verification against its external verifier
    trust policy before any GPU work.
    """
    root = Path(evidence_root).resolve()
    admission_path = Path(admission_path).resolve()
    deployment_manifest_path = Path(deployment_manifest_path).resolve()
    hardware_packet_path = Path(hardware_packet_path).resolve()
    workload_config_path = Path(workload_config_path).resolve()
    source_manifest_path = Path(source_manifest_path).resolve()
    challenge_nonce_path = Path(challenge_nonce_path).resolve()
    admission = load_admission_record(admission_path, root=root)
    if not admission.get("admitted"):
        raise ActiveSecurityError("Active runtime requires an admitted decision")
    if (not isinstance(expected_session_id, str) or not expected_session_id
            or admission.get("session_id") != expected_session_id):
        raise ActiveSecurityError("Active runtime admission session differs")
    deployment_sha256 = file_sha256(deployment_manifest_path)
    deployment = _load_json(
        deployment_manifest_path, "deployment manifest", deployment_sha256)
    validate_deployment_manifest(
        deployment, deployment_sha256, deployment.get("profile"),
        deployment.get("adapter", {}).get("command_argv", []),
    )
    arrival_path = (root / admission["arrival_record"]).resolve()
    try:
        arrival_path.relative_to(root)
    except ValueError as exc:
        raise ActiveSecurityError("Admission arrival path escapes its evidence root") from exc
    arrival = load_arrival_record(arrival_path)
    if file_sha256(arrival_path) != admission["arrival_record_sha256"]:
        raise ActiveSecurityError("Active runtime arrival digest differs from admission")
    if arrival["deployment"]["protection_mode"] != "cc-on":
        raise ActiveSecurityError("Active runtime arrival is not CC-on")
    binding = admission["workload_binding"]
    expected = {
        "deployment_manifest_sha256": deployment_sha256,
        "hardware_packet_sha256": file_sha256(hardware_packet_path),
        "config_sha256": file_sha256(workload_config_path),
        "challenge_nonce_sha256": validate_challenge_nonce(challenge_nonce_path),
    }
    source_manifest = _load_json(source_manifest_path, "source manifest")
    source_tree_sha256 = source_manifest.get("source_tree_sha256")
    _require_sha256(source_tree_sha256, "Source tree digest")
    expected["source_tree_sha256"] = source_tree_sha256
    for key, value in expected.items():
        if binding.get(key) != value:
            raise ActiveSecurityError(f"Active runtime admission {key} differs")
    if arrival["deployment"]["deployment_manifest_sha256"] != deployment_sha256:
        raise ActiveSecurityError("Active runtime arrival deployment differs")
    document = {
        "schema": ACTIVE_RUNTIME_SCHEMA,
        "deployment_manifest_sha256": deployment_sha256,
        "admission_id": admission["admission_id"],
        "admission_sha256": file_sha256(admission_path),
        "session_id": admission["session_id"],
        "challenge_nonce_sha256": expected["challenge_nonce_sha256"],
        "arrival_record_sha256": admission["arrival_record_sha256"],
        "instance_id": arrival["deployment"]["instance_id"],
        "protection_mode": "cc-on",
        "hardware_packet_sha256": expected["hardware_packet_sha256"],
        "workload_config_sha256": expected["config_sha256"],
        "source_tree_sha256": source_tree_sha256,
    }
    return validate_active_runtime_binding(document, deployment_sha256)


def write_active_runtime_binding(path, document: dict) -> Path:
    validate_active_runtime_binding(
        document, document.get("deployment_manifest_sha256"))
    return write_json_exclusive(path, document)


def _git(attacker_repo: Path, arguments: Sequence[str], timeout: int = 10) -> str:
    try:
        completed = subprocess.run(
            ["git", "-c", "safe.directory=*", "-C", str(attacker_repo), *arguments],
            text=True, capture_output=True, check=False, timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ActiveSecurityError(f"Cannot inspect attacker repository: {exc}") from exc
    if completed.returncode != 0:
        raise ActiveSecurityError("Cannot inspect attacker repository")
    return completed.stdout.strip()


def verify_attacker_revision(attacker_repo, expected_revision: str) -> str:
    """Require an exact committed, clean LLM-Attacker source tree."""
    repo = Path(attacker_repo).resolve()
    if not GIT_REVISION_RE.fullmatch(expected_revision):
        raise ActiveSecurityError("Attacker revision must be a full Git object ID")
    actual = _git(repo, ("rev-parse", "HEAD"))
    if actual != expected_revision:
        raise ActiveSecurityError("LLM-Attacker HEAD does not match the frozen revision")
    dirty = _git(repo, ("status", "--porcelain", "--untracked-files=all"))
    if dirty:
        raise ActiveSecurityError("LLM-Attacker worktree is dirty; commit or remove every change")
    return actual


def _result_satisfies(case: dict, result: dict) -> bool:
    expected = case["expected"]
    if expected == "allow-release":
        return result["accepted"] and result["secret_released"]
    if expected == "accept":
        return result["accepted"] and not result["secret_released"]
    if expected in ("deny-release", "reject"):
        return not result["accepted"] and not result["secret_released"]
    if expected == "no-canary":
        return not result["canary_observed"]
    if expected == "observe-canary":
        return result["canary_observed"]
    if expected == "observe-or-declare":
        return True
    if expected == "signal":
        return result["signal_observed"]
    if expected == "no-signal":
        return not result["signal_observed"]
    return False


def _verify_case_evidence(result: dict, case: dict, evidence_root: Path,
                          digest_cache: dict[Path, str]) -> list[dict]:
    relative = result["details"].get("evidence_manifest")
    manifest_path = _safe_bundle_path(evidence_root, relative, "case evidence manifest")
    if not manifest_path.is_file():
        raise ActiveSecurityError("Case evidence manifest is missing")
    manifest_digest = digest_cache.setdefault(manifest_path, file_sha256(manifest_path))
    if manifest_digest != result["evidence_manifest_sha256"]:
        raise ActiveSecurityError("Case evidence manifest digest mismatch")
    manifest = _load_json(manifest_path, "case evidence manifest")
    expected_keys = {
        "schema", "category", "case_id", "case_sha256", "plan_sha256",
        "deployment_manifest_sha256", "evidence", "files",
    }
    if set(manifest) != expected_keys or manifest.get("schema") != EVIDENCE_MANIFEST_SCHEMA:
        raise ActiveSecurityError("Case evidence manifest schema mismatch")
    for key in ("case_id", "case_sha256", "plan_sha256",
                "deployment_manifest_sha256", "evidence"):
        if manifest.get(key) != result.get(key):
            raise ActiveSecurityError(f"Case evidence manifest {key} binding mismatch")
    if manifest.get("category") != case["category"]:
        raise ActiveSecurityError("Case evidence category binding mismatch")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(result["evidence"]):
        raise ActiveSecurityError("Case evidence file inventory mismatch")
    inventory = [{
        "path": str(manifest_path.relative_to(evidence_root.resolve())),
        "sha256": manifest_digest,
        "bytes": manifest_path.stat().st_size,
        "role": "case_evidence_manifest",
    }]
    for field, relative_path in sorted(files.items()):
        evidence_path = _safe_bundle_path(evidence_root, relative_path, f"evidence {field}")
        if not evidence_path.is_file():
            raise ActiveSecurityError(f"Evidence file {field!r} is missing")
        digest = digest_cache.setdefault(evidence_path, file_sha256(evidence_path))
        if digest != result["evidence"][field]:
            raise ActiveSecurityError(f"Evidence file {field!r} digest mismatch")
        inventory.append({
            "path": str(evidence_path.relative_to(evidence_root.resolve())),
            "sha256": digest,
            "bytes": evidence_path.stat().st_size,
            "role": field,
        })
    return inventory


def _validate_results(plan: dict, results, evidence_bundle_dir) -> tuple[dict, list[dict]]:
    if not isinstance(results, list):
        raise ActiveSecurityError("Attacker results must be a list")
    cases = {case["case_id"]: case for case in plan["cases"]}
    rows: dict[str, dict] = {}
    evidence_root = Path(evidence_bundle_dir).resolve()
    if not evidence_root.is_dir():
        raise ActiveSecurityError("A readable evidence bundle directory is mandatory")
    digest_cache: dict[Path, str] = {}
    inventory_by_path: dict[str, dict] = {}

    for index, result in enumerate(results):
        if not isinstance(result, dict) or set(result) != RESULT_KEYS:
            raise ActiveSecurityError(f"Result {index} has the wrong field inventory")
        if result.get("schema") != RESULT_SCHEMA:
            raise ActiveSecurityError(f"Result {index} schema mismatch")
        case_id = result.get("case_id")
        if case_id not in cases or case_id in rows:
            raise ActiveSecurityError("Results contain an unknown or duplicate case")
        case = cases[case_id]
        rows[case_id] = result
        if result.get("case_sha256") != _case_digest(case):
            raise ActiveSecurityError(f"Result {case_id!r} case binding mismatch")
        for key in ("plan_sha256", "deployment_manifest_sha256", "request_id",
                    "session_id", "step_id"):
            expected = (
                case["payload"]["deployment_manifest_sha256"]
                if key == "deployment_manifest_sha256" else case[key]
            )
            if result.get(key) != expected:
                raise ActiveSecurityError(f"Result {case_id!r} {key} binding mismatch")
        if result.get("status") != "observed":
            raise ActiveSecurityError(f"Result {case_id!r} is not an observed result")
        for key in ("accepted", "secret_released", "canary_observed",
                    "signal_observed"):
            if not isinstance(result.get(key), bool):
                raise ActiveSecurityError(f"Result {case_id!r} {key} is not boolean")
        if result.get("error_code") is not None \
                and (not isinstance(result["error_code"], str)
                     or not result["error_code"]):
            raise ActiveSecurityError(f"Result {case_id!r} error code is malformed")
        if not isinstance(result.get("details"), dict):
            raise ActiveSecurityError(f"Result {case_id!r} details are malformed")
        expected_fields = set(EVIDENCE_FIELDS[case["category"]])
        evidence = result.get("evidence")
        if not isinstance(evidence, dict) or set(evidence) != expected_fields:
            raise ActiveSecurityError(f"Result {case_id!r} evidence roles mismatch")
        for digest in evidence.values():
            _require_sha256(digest, f"Result {case_id!r} evidence digest")
        _require_sha256(
            result.get("evidence_manifest_sha256"),
            f"Result {case_id!r} evidence manifest digest",
        )
        if not _result_satisfies(case, result):
            raise ActiveSecurityError(f"Result {case_id!r} violates its frozen expectation")
        for item in _verify_case_evidence(result, case, evidence_root, digest_cache):
            previous = inventory_by_path.get(item["path"])
            if previous is not None and previous != item:
                raise ActiveSecurityError("Evidence path is rebound to another digest or role")
            inventory_by_path[item["path"]] = item

    if set(rows) != set(cases):
        raise ActiveSecurityError("Results do not cover the exact frozen case inventory")
    invalid_releases = sum(
        bool(rows[case["case_id"]]["secret_released"])
        for case in plan["cases"] if case["expected"] == "deny-release"
    )
    external_exposures = sum(
        bool(rows[case["case_id"]]["canary_observed"])
        for case in plan["cases"] if case["expected"] == "no-canary"
    )
    accepted_mutations = sum(
        bool(rows[case["case_id"]]["accepted"])
        for case in plan["cases"] if case["expected"] in ("deny-release", "reject")
    )
    counters = {
        "case_count": len(cases),
        "invalid_secret_releases": invalid_releases,
        "external_canary_exposures": external_exposures,
        "accepted_mutation_cells": accepted_mutations,
        "fail_cells": 0,
        "unassessed_cells": 0,
    }
    if any(counters[key] != 0 for key in (
            "invalid_secret_releases", "external_canary_exposures",
            "accepted_mutation_cells")):
        raise ActiveSecurityError("Independent zero-tolerance result counters failed")
    inventory = sorted(inventory_by_path.values(), key=lambda item: item["path"])
    return counters, inventory


def _validate_gate(gate: dict, plan: dict, counters: dict) -> None:
    expected_keys = {
        "schema", "overall", "counts", "invalid_secret_releases",
        "external_canary_exposures", "missing_case_ids", "extra_case_ids",
        "duplicate_case_ids", "cells",
    }
    if not isinstance(gate, dict) or set(gate) != expected_keys:
        raise ActiveSecurityError("Attacker gate has the wrong field inventory")
    if gate.get("schema") != GATE_SCHEMA or gate.get("overall") != "PASS":
        raise ActiveSecurityError("Attacker gate is not PASS under the frozen schema")
    expected_counts = {"PASS": len(plan["cases"]), "FAIL": 0, "UNASSESSED": 0}
    if gate.get("counts") != expected_counts:
        raise ActiveSecurityError("Attacker gate has FAIL, UNASSESSED, or missing PASS cells")
    for key in ("missing_case_ids", "extra_case_ids", "duplicate_case_ids"):
        if gate.get(key) != []:
            raise ActiveSecurityError(f"Attacker gate {key} is not empty")
    if gate.get("invalid_secret_releases") != counters["invalid_secret_releases"]:
        raise ActiveSecurityError("Attacker gate invalid-release count mismatch")
    if gate.get("external_canary_exposures") != counters["external_canary_exposures"]:
        raise ActiveSecurityError("Attacker gate external-canary count mismatch")
    cells = gate.get("cells")
    if not isinstance(cells, list) or len(cells) != len(plan["cases"]):
        raise ActiveSecurityError("Attacker gate cell inventory mismatch")
    by_id = {case["case_id"]: case for case in plan["cases"]}
    seen = set()
    for cell in cells:
        if not isinstance(cell, dict) or set(cell) != {
                "case_id", "category", "arm", "view", "required", "verdict", "reason"}:
            raise ActiveSecurityError("Attacker gate cell schema mismatch")
        case = by_id.get(cell.get("case_id"))
        if case is None or cell["case_id"] in seen:
            raise ActiveSecurityError("Attacker gate has an unknown or duplicate cell")
        seen.add(cell["case_id"])
        if cell.get("verdict") != "PASS":
            raise ActiveSecurityError("Attacker gate contains a non-PASS cell")
        for key in ("category", "arm", "view", "required"):
            if cell.get(key) != case[key]:
                raise ActiveSecurityError("Attacker gate cell does not bind its plan case")
    if seen != set(by_id):
        raise ActiveSecurityError("Attacker gate omits frozen cases")


def _verify_framework_evidence(artifact_path: Path, revision: str) -> dict:
    """Verify the attack framework's own event hash chain independently."""
    manifest_path = Path(str(artifact_path) + ".evidence.json")
    events_path = Path(str(artifact_path) + ".evidence.jsonl")
    manifest = _load_json(manifest_path, "attacker framework evidence manifest")
    if manifest.get("schema") != FRAMEWORK_EVIDENCE_SCHEMA:
        raise ActiveSecurityError("Attacker framework evidence schema mismatch")
    stored_manifest_sha = manifest.get("manifest_sha256")
    _require_sha256(stored_manifest_sha, "Attacker framework evidence identity")
    if _canonical_sha256({k: v for k, v in manifest.items()
                          if k != "manifest_sha256"}) != stored_manifest_sha:
        raise ActiveSecurityError("Attacker framework evidence manifest hash mismatch")
    artifact = manifest.get("artifact")
    if not isinstance(artifact, dict) or artifact.get("sha256") != file_sha256(artifact_path):
        raise ActiveSecurityError("Attacker framework evidence artifact binding mismatch")
    git_state = manifest.get("git")
    if not isinstance(git_state, dict) or git_state.get("commit") != revision:
        raise ActiveSecurityError("Attacker framework evidence revision mismatch")
    if git_state.get("dirty") is not False:
        raise ActiveSecurityError("Attacker framework evidence reports a dirty source tree")
    if not isinstance(git_state.get("source_sha256"), str) \
            or not SHA256_RE.fullmatch(git_state["source_sha256"]):
        raise ActiveSecurityError("Attacker framework evidence lacks its source-tree digest")
    if not events_path.is_file():
        raise ActiveSecurityError("Attacker framework evidence event stream is missing")
    previous = "0" * 64
    count = 0
    run_id = manifest.get("run_id")
    for line_number, line in enumerate(events_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ActiveSecurityError(
                f"Attacker framework evidence line {line_number} is invalid JSON") from exc
        if not isinstance(event, dict) or event.get("seq") != count:
            raise ActiveSecurityError("Attacker framework evidence sequence mismatch")
        if event.get("run_id") != run_id or event.get("prev_hash") != previous:
            raise ActiveSecurityError("Attacker framework evidence chain binding mismatch")
        recorded_hash = event.get("hash")
        _require_sha256(recorded_hash, "Attacker framework evidence event hash")
        if _canonical_sha256({k: v for k, v in event.items() if k != "hash"}) != recorded_hash:
            raise ActiveSecurityError("Attacker framework evidence event hash mismatch")
        previous = recorded_hash
        count += 1
    if count != manifest.get("event_count") or previous != manifest.get("chain_hash"):
        raise ActiveSecurityError("Attacker framework evidence final-chain mismatch")
    return manifest


def validate_attacker_artifact(artifact_path, frozen_plan: dict,
                               deployment_sha256: str, attacker_revision: str,
                               evidence_bundle_dir) -> dict:
    """Independently grade the complete attacker artifact and evidence bytes."""
    artifact_path = Path(artifact_path).resolve()
    artifact = _load_json(artifact_path, "LLM-Attacker active-security artifact")
    expected_top = {
        "schema", "config", "threat_model", "provenance", "results",
        "summary", "interpretation", "plan", "gate",
    }
    if set(artifact) != expected_top or artifact.get("schema") != ATTACK_ARTIFACT_SCHEMA:
        raise ActiveSecurityError("LLM-Attacker artifact schema/fields mismatch")
    config = artifact.get("config")
    expected_config_keys = {
        "attack", "mode", "profile", "source", "toy", "plan_schema",
        "plan_sha256", "deployment_manifest_sha256",
        "evidence_bundle_verified", "require_host_traces",
        "driver_stderr_lines",
    }
    if not isinstance(config, dict) or set(config) != expected_config_keys:
        raise ActiveSecurityError("LLM-Attacker artifact config fields mismatch")
    expected_config = {
        "attack": "secure_compute_audit",
        "mode": "training",
        "profile": frozen_plan["profile"],
        "source": "process-driver",
        "toy": False,
        "plan_schema": PLAN_SCHEMA,
        "plan_sha256": frozen_plan["sha256"],
        "deployment_manifest_sha256": deployment_sha256,
        "evidence_bundle_verified": True,
        "require_host_traces": frozen_plan["require_host_traces"],
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            raise ActiveSecurityError(f"LLM-Attacker artifact config.{key} mismatch")
    if not isinstance(config.get("driver_stderr_lines"), int) \
            or config["driver_stderr_lines"] < 0:
        raise ActiveSecurityError("LLM-Attacker driver stderr count is invalid")
    provenance = artifact.get("provenance")
    if not isinstance(provenance, dict) \
            or provenance.get("dtraining_commit") != attacker_revision:
        raise ActiveSecurityError("LLM-Attacker artifact revision mismatch")
    if artifact.get("plan") != frozen_plan:
        raise ActiveSecurityError("LLM-Attacker artifact does not contain the frozen plan")
    counters, inventory = _validate_results(
        frozen_plan, artifact.get("results"), evidence_bundle_dir)
    _validate_gate(artifact.get("gate"), frozen_plan, counters)
    summary = artifact.get("summary")
    expected_categories = {case["category"] for case in frozen_plan["cases"]}
    if not isinstance(summary, list) or {
            row.get("category") for row in summary if isinstance(row, dict)
            } != expected_categories:
        raise ActiveSecurityError("LLM-Attacker category summary inventory mismatch")
    for row in summary:
        if row.get("verdict") != "PASS" or row.get("FAIL") != 0 \
                or row.get("UNASSESSED") != 0:
            raise ActiveSecurityError("LLM-Attacker category summary is not wholly PASS")
    framework_evidence = _verify_framework_evidence(artifact_path, attacker_revision)
    return {
        "counters": counters,
        "evidence_inventory": inventory,
        "evidence_inventory_sha256": _canonical_sha256(inventory),
        "framework_evidence_manifest_sha256": file_sha256(
            str(artifact_path) + ".evidence.json"),
        "framework_evidence_events_sha256": file_sha256(
            str(artifact_path) + ".evidence.jsonl"),
        "framework_source_sha256": framework_evidence["git"]["source_sha256"],
    }


def _validate_argv(argv: Sequence[str], label: str) -> list[str]:
    if isinstance(argv, (str, bytes)) or not isinstance(argv, Sequence):
        raise ActiveSecurityError(f"{label} must be an explicit argv array")
    values = list(argv)
    if not values or not all(isinstance(value, str) and value for value in values):
        raise ActiveSecurityError(f"{label} must be a nonempty array of nonempty strings")
    if any("\x00" in value for value in values):
        raise ActiveSecurityError(f"{label} contains a NUL byte")
    shell_names = {
        "sh", "bash", "dash", "zsh", "ksh", "csh", "tcsh", "fish",
        "powershell", "powershell.exe", "pwsh", "pwsh.exe", "cmd", "cmd.exe",
    }
    if any(Path(value).name.lower() in shell_names for value in values):
        raise ActiveSecurityError(f"{label} must not invoke a command shell")
    return values


def _write_json(path: Path, value: dict) -> None:
    try:
        write_json_exclusive(path, value)
    except ValueError as exc:
        raise ActiveSecurityError(f"Cannot write exclusive evidence {path}: {exc}") from exc


def _artifact_record(path: Path, root: Path, role: str) -> dict:
    path = path.resolve()
    root = root.resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ActiveSecurityError("Result artifact is outside the result root") from exc
    if not path.is_file() or path.is_symlink():
        raise ActiveSecurityError(f"Result artifact {relative} is missing or is a symlink")
    return {
        "path": str(relative), "sha256": file_sha256(path),
        "bytes": path.stat().st_size, "role": role,
    }


def _copy_case_evidence(inventory: Sequence[dict], source_root: Path,
                        output_root: Path) -> list[tuple[Path, str]]:
    """Copy every independently graded case-evidence byte into the result bundle."""
    destination_root = output_root / "case-evidence"
    destination_root.mkdir(mode=0o700)
    created: list[tuple[Path, str]] = []
    for index, item in enumerate(inventory):
        source = _safe_bundle_path(source_root, item["path"], "case evidence")
        destination = destination_root / item["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(destination.parent, 0o700)
        if destination.exists():
            raise ActiveSecurityError("Case-evidence destination is not fresh")
        shutil.copyfile(source, destination)
        os.chmod(destination, 0o600)
        if (destination.stat().st_size != item["bytes"]
                or file_sha256(destination) != item["sha256"]):
            raise ActiveSecurityError("Case evidence changed while it was copied")
        created.append((destination, f"case_evidence_{index:06d}"))
    return created


def run_active_security_gate(
    *, attacker_repo, attacker_revision: str, frozen_plan_path,
    frozen_plan_file_sha256: str, deployment_manifest_path,
    deployment_manifest_sha256: str, runtime_binding_path,
    runtime_binding_file_sha256: str, evidence_bundle_dir,
    driver_argv: Sequence[str], output_dir, profile: str,
    timeout_seconds: int = 900, attacker_python: str | None = None,
) -> dict:
    """Execute and independently admit one frozen active-security plan.

    This function never provisions a resource.  ``driver_argv`` is passed as
    data to the frozen LLM-Attacker process adapter and every subprocess is
    launched with ``shell=False`` (the Python default for an argv array).
    """
    if profile not in PROFILES:
        raise ActiveSecurityError(f"profile must be one of {PROFILES}")
    if type(timeout_seconds) is not int or timeout_seconds < 1:
        raise ActiveSecurityError("timeout_seconds must be a positive integer")
    driver_argv = _validate_argv(driver_argv, "Driver argv")
    plan_file_sha = _require_sha256(
        frozen_plan_file_sha256, "Frozen plan file digest")
    deployment_sha = _require_sha256(
        deployment_manifest_sha256, "Deployment manifest digest")
    runtime_file_sha = _require_sha256(
        runtime_binding_file_sha256, "Active runtime binding file digest")
    frozen_plan_path = Path(frozen_plan_path).resolve()
    deployment_manifest_path = Path(deployment_manifest_path).resolve()
    runtime_binding_path = Path(runtime_binding_path).resolve()
    evidence_bundle_dir = Path(evidence_bundle_dir).resolve()
    attacker_repo = Path(attacker_repo).resolve()
    output_dir = Path(output_dir).resolve()

    plan_template = _load_json(
        frozen_plan_path, "frozen plan template", plan_file_sha)
    plan = materialize_frozen_plan(plan_template, deployment_sha, profile)
    deployment = validate_deployment_manifest(
        _load_json(deployment_manifest_path, "deployment manifest", deployment_sha),
        deployment_sha, profile, driver_argv)
    runtime_binding = validate_active_runtime_binding(
        _load_json(runtime_binding_path, "active runtime binding", runtime_file_sha),
        deployment_sha)
    runtime_binding_sha256 = _canonical_sha256(runtime_binding)
    if not evidence_bundle_dir.is_dir():
        raise ActiveSecurityError("Evidence bundle directory is mandatory")
    verify_attacker_revision(attacker_repo, attacker_revision)

    output_dir.mkdir(parents=True, exist_ok=True)
    reserved = {
        "frozen-plan.json", "frozen-deployment-manifest.json",
        "frozen-runtime-binding.json",
        "attacker-generated-plan.json", "attacker-artifact.json",
        "attacker-artifact.json.jsonl", "attacker-artifact.json.evidence.json",
        "attacker-artifact.json.evidence.jsonl", "active-security-result.json",
        "active-security-manifest.json",
        "case-evidence",
    }
    occupied = sorted(name for name in reserved if (output_dir / name).exists())
    if occupied:
        raise ActiveSecurityError(f"Refusing to reuse active-security outputs: {occupied}")

    plan_copy = output_dir / "frozen-plan.json"
    deployment_copy = output_dir / "frozen-deployment-manifest.json"
    runtime_copy = output_dir / "frozen-runtime-binding.json"
    shutil.copyfile(frozen_plan_path, plan_copy)
    shutil.copyfile(deployment_manifest_path, deployment_copy)
    shutil.copyfile(runtime_binding_path, runtime_copy)
    os.chmod(plan_copy, 0o600)
    os.chmod(deployment_copy, 0o600)
    os.chmod(runtime_copy, 0o600)
    if file_sha256(plan_copy) != plan_file_sha \
            or file_sha256(deployment_copy) != deployment_sha \
            or file_sha256(runtime_copy) != runtime_file_sha:
        raise ActiveSecurityError("Frozen input changed while it was being copied")
    generated_plan_path = output_dir / "attacker-generated-plan.json"
    artifact_path = output_dir / "attacker-artifact.json"
    python_executable = attacker_python or sys.executable
    if not isinstance(python_executable, str) or not python_executable:
        raise ActiveSecurityError("attacker_python must be a nonempty executable path")
    _validate_argv([python_executable], "Attacker Python argv")
    command = [
        python_executable, "-m", "attacker", "--mode", "training", "--attack",
        "secure-compute-audit", "--profile", profile, "--driver-command",
        shlex.join(driver_argv), "--driver-timeout", str(timeout_seconds),
        "--deployment-manifest", str(deployment_copy), "--evidence-bundle-dir",
        str(evidence_bundle_dir), "--plan-output", str(generated_plan_path),
        "--output", str(artifact_path), "--evidence",
    ]
    if plan["require_host_traces"]:
        command.append("--require-host-traces")
    try:
        completed = subprocess.run(
            command, cwd=attacker_repo, text=True, capture_output=True,
            check=False, timeout=timeout_seconds + 30,
        )
    except subprocess.TimeoutExpired as exc:
        raise ActiveSecurityError("LLM-Attacker active-security execution timed out") from exc
    except OSError as exc:
        raise ActiveSecurityError(f"Cannot execute LLM-Attacker: {exc}") from exc
    if completed.returncode != 0:
        raise ActiveSecurityError(
            f"LLM-Attacker active-security execution exited {completed.returncode}")
    if not generated_plan_path.is_file() or not artifact_path.is_file():
        raise ActiveSecurityError("LLM-Attacker exited successfully without required artifacts")
    generated_plan = _load_json(generated_plan_path, "generated attacker plan")
    if generated_plan != plan:
        raise ActiveSecurityError(
            "Generated attacker plan differs from the materialised frozen template")

    validation = validate_attacker_artifact(
        artifact_path, plan, deployment_sha, attacker_revision,
        evidence_bundle_dir)
    copied_case_evidence = _copy_case_evidence(
        validation["evidence_inventory"], evidence_bundle_dir, output_dir)
    artifact_digest = file_sha256(artifact_path)
    execution_argv_sha = _canonical_sha256(command)
    driver_argv_sha = _canonical_sha256(driver_argv)
    result = {
        "schema": ACTIVE_RESULT_SCHEMA,
        "result_id": (
            f"p4-cc-active-{plan['sha256'][:12]}-{runtime_binding_sha256[:12]}"),
        "generated_at": _now(),
        "scientific_evidence": False,
        "claim_scope": CLAIM_SCOPE,
        "status": "PASS",
        "profile": profile,
        "bindings": {
            "attacker_repository_revision": attacker_revision,
            "attacker_source_sha256": validation["framework_source_sha256"],
            "frozen_plan_file_sha256": plan_file_sha,
            "frozen_plan_sha256": plan["sha256"],
            "deployment_manifest_sha256": deployment_sha,
            "runtime_binding_file_sha256": runtime_file_sha,
            "driver_argv_sha256": driver_argv_sha,
            "execution_argv_sha256": execution_argv_sha,
            "attacker_artifact_sha256": artifact_digest,
            "attacker_generated_plan_file_sha256": file_sha256(generated_plan_path),
            "attacker_framework_evidence_manifest_sha256": validation[
                "framework_evidence_manifest_sha256"],
            "attacker_framework_evidence_events_sha256": validation[
                "framework_evidence_events_sha256"],
            "evidence_bundle_inventory_sha256": validation[
                "evidence_inventory_sha256"],
            "runtime_binding_sha256": runtime_binding_sha256,
        },
        "gate": {
            "schema": GATE_SCHEMA,
            "overall": "PASS",
            **validation["counters"],
        },
    }
    result_path = output_dir / "active-security-result.json"
    _write_json(result_path, result)

    created = [
        (plan_copy, "frozen_plan"),
        (deployment_copy, "frozen_deployment_manifest"),
        (runtime_copy, "frozen_runtime_binding"),
        (generated_plan_path, "attacker_generated_plan"),
        (artifact_path, "attacker_artifact"),
        (Path(str(artifact_path) + ".jsonl"), "attacker_case_journal"),
        (Path(str(artifact_path) + ".evidence.json"),
         "attacker_framework_evidence_manifest"),
        (Path(str(artifact_path) + ".evidence.jsonl"),
         "attacker_framework_evidence_events"),
        (result_path, "paper4_active_security_result"),
    ] + copied_case_evidence
    for path, _role in created:
        os.chmod(path, 0o600)
    records = [_artifact_record(path, output_dir, role) for path, role in created]
    manifest = {
        "schema": ACTIVE_MANIFEST_SCHEMA,
        "manifest_id": result["result_id"],
        "generated_at": result["generated_at"],
        "scientific_evidence": False,
        "claim_scope": CLAIM_SCOPE,
        "inputs": {
            "attacker_repository_revision": attacker_revision,
            "profile": profile,
            "plan_file_sha256": plan_file_sha,
            "plan_sha256": plan["sha256"],
            "deployment_manifest_sha256": deployment_sha,
            "runtime_binding_file_sha256": runtime_file_sha,
            "driver_argv": driver_argv,
            "driver_argv_sha256": driver_argv_sha,
            "evidence_bundle_inventory_sha256": validation[
                "evidence_inventory_sha256"],
            "evidence_bundle_inventory": validation["evidence_inventory"],
            "runtime_binding": runtime_binding,
            "runtime_binding_sha256": runtime_binding_sha256,
        },
        "execution": {
            "argv": command,
            "argv_sha256": execution_argv_sha,
            "shell": False,
            "timeout_seconds": timeout_seconds,
            "return_code": completed.returncode,
            "stdout_sha256": hashlib.sha256(
                completed.stdout.encode("utf-8")).hexdigest(),
            "stderr_sha256": hashlib.sha256(
                completed.stderr.encode("utf-8")).hexdigest(),
        },
        "outcome": {
            "status": "PASS",
            **validation["counters"],
        },
        "artifacts": records,
    }
    manifest_path = output_dir / "active-security-manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "result_path": str(result_path),
        "result_sha256": file_sha256(result_path),
        "status": "PASS",
    }


def validate_active_security_manifest(manifest: dict, root,
                                      expected_manifest_sha256: str | None = None,
                                      manifest_path=None) -> dict:
    """Validate a Paper 4 active-security manifest for a downstream runner."""
    root = Path(root).resolve()
    expected_keys = {
        "schema", "manifest_id", "generated_at", "scientific_evidence",
        "claim_scope", "inputs", "execution", "outcome", "artifacts",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected_keys \
            or manifest.get("schema") != ACTIVE_MANIFEST_SCHEMA:
        raise ActiveSecurityError("Active-security manifest schema/fields mismatch")
    if manifest.get("scientific_evidence") is not False \
            or manifest.get("claim_scope") != CLAIM_SCOPE:
        raise ActiveSecurityError("Active-security manifest overstates its claim scope")
    if expected_manifest_sha256 is not None:
        _require_sha256(expected_manifest_sha256, "Expected active-security manifest digest")
        if manifest_path is None or file_sha256(manifest_path) != expected_manifest_sha256:
            raise ActiveSecurityError("Active-security manifest digest mismatch")
    outcome = manifest.get("outcome")
    if not isinstance(outcome, dict) or outcome.get("status") != "PASS":
        raise ActiveSecurityError("Active-security manifest is not PASS")
    for key in ("invalid_secret_releases", "external_canary_exposures",
                "accepted_mutation_cells", "fail_cells", "unassessed_cells"):
        if outcome.get(key) != 0:
            raise ActiveSecurityError(f"Active-security outcome {key} is nonzero")
    execution = manifest.get("execution")
    if not isinstance(execution, dict) or execution.get("shell") is not False \
            or execution.get("return_code") != 0:
        raise ActiveSecurityError("Active-security execution did not use the admitted path")
    if execution.get("argv_sha256") != _canonical_sha256(execution.get("argv")):
        raise ActiveSecurityError("Active-security execution argv binding mismatch")
    inputs = manifest.get("inputs")
    expected_inputs = {
        "attacker_repository_revision", "profile", "plan_file_sha256",
        "plan_sha256", "deployment_manifest_sha256",
        "runtime_binding_file_sha256", "driver_argv",
        "driver_argv_sha256", "evidence_bundle_inventory_sha256",
        "evidence_bundle_inventory", "runtime_binding", "runtime_binding_sha256",
    }
    if (not isinstance(inputs, dict) or set(inputs) != expected_inputs
            or inputs.get("driver_argv_sha256") !=
            _canonical_sha256(inputs.get("driver_argv"))):
        raise ActiveSecurityError("Active-security driver argv binding mismatch")
    if inputs.get("profile") not in PROFILES:
        raise ActiveSecurityError("Active-security profile is invalid")
    runtime = inputs.get("runtime_binding")
    validate_active_runtime_binding(
        runtime, inputs.get("deployment_manifest_sha256"))
    _require_sha256(inputs.get("runtime_binding_file_sha256"),
                    "Active runtime binding file digest")
    if inputs.get("runtime_binding_sha256") != _canonical_sha256(runtime):
        raise ActiveSecurityError("Active-security runtime binding digest mismatch")
    for key in ("admission_sha256", "challenge_nonce_sha256",
                "arrival_record_sha256", "hardware_packet_sha256",
                "workload_config_sha256", "source_tree_sha256"):
        _require_sha256(runtime.get(key), f"Active-security runtime binding {key}")
    if runtime.get("protection_mode") != "cc-on":
        raise ActiveSecurityError("Active-security runtime is not bound to CC-on")
    inventory = inputs.get("evidence_bundle_inventory")
    if (not isinstance(inventory, list) or not inventory
            or inputs.get("evidence_bundle_inventory_sha256") !=
            _canonical_sha256(inventory)):
        raise ActiveSecurityError("Active-security evidence inventory binding mismatch")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ActiveSecurityError("Active-security manifest has no artifact inventory")
    roles = set()
    records = {}
    for record in artifacts:
        if not isinstance(record, dict) or set(record) != {"path", "sha256", "bytes", "role"}:
            raise ActiveSecurityError("Active-security artifact record is malformed")
        path = _safe_bundle_path(root, record["path"], "active-security artifact")
        if not path.is_file() or path.stat().st_size != record["bytes"] \
                or file_sha256(path) != record["sha256"]:
            raise ActiveSecurityError("Active-security artifact is missing or changed")
        if record["role"] in roles:
            raise ActiveSecurityError("Active-security artifact role is duplicated")
        roles.add(record["role"])
        records[record["role"]] = (record, path)
    required_roles = {
        "frozen_plan", "frozen_deployment_manifest", "frozen_runtime_binding",
        "attacker_generated_plan",
        "attacker_artifact", "attacker_case_journal",
        "attacker_framework_evidence_manifest", "attacker_framework_evidence_events",
        "paper4_active_security_result",
    }
    evidence_roles = {f"case_evidence_{index:06d}" for index in range(len(inventory))}
    if roles != required_roles | evidence_roles:
        raise ActiveSecurityError("Active-security artifact role inventory mismatch")
    for index, item in enumerate(inventory):
        if (not isinstance(item, dict)
                or set(item) != {"path", "sha256", "bytes", "role"}):
            raise ActiveSecurityError("Active-security evidence inventory item is malformed")
        record, _path = records[f"case_evidence_{index:06d}"]
        if (record["path"] != f"case-evidence/{item['path']}"
                or record["sha256"] != item["sha256"]
                or record["bytes"] != item["bytes"]):
            raise ActiveSecurityError("Copied case-evidence binding mismatch")
    result_record, result_path = records["paper4_active_security_result"]
    result = _load_json(
        result_path, "Paper 4 active-security result", result_record["sha256"])
    if result.get("schema") != ACTIVE_RESULT_SCHEMA or result.get("status") != "PASS":
        raise ActiveSecurityError("Paper 4 active-security result is not PASS")
    if result.get("result_id") != manifest.get("manifest_id"):
        raise ActiveSecurityError("Active-security result/manifest identity mismatch")
    result_gate = result.get("gate")
    if not isinstance(result_gate, dict) or result_gate.get("overall") != "PASS":
        raise ActiveSecurityError("Paper 4 active-security result gate is not PASS")
    for key in ("invalid_secret_releases", "external_canary_exposures",
                "accepted_mutation_cells", "fail_cells", "unassessed_cells"):
        if result_gate.get(key) != 0 or result_gate.get(key) != outcome.get(key):
            raise ActiveSecurityError("Active-security result counters mismatch")
    bindings = result.get("bindings")
    if not isinstance(bindings, dict):
        raise ActiveSecurityError("Active-security result bindings are missing")
    cross_checks = {
        "attacker_repository_revision": inputs.get("attacker_repository_revision"),
        "frozen_plan_file_sha256": inputs.get("plan_file_sha256"),
        "frozen_plan_sha256": inputs.get("plan_sha256"),
        "deployment_manifest_sha256": inputs.get("deployment_manifest_sha256"),
        "runtime_binding_file_sha256": inputs.get(
            "runtime_binding_file_sha256"),
        "driver_argv_sha256": inputs.get("driver_argv_sha256"),
        "evidence_bundle_inventory_sha256": inputs.get(
            "evidence_bundle_inventory_sha256"),
        "runtime_binding_sha256": inputs.get("runtime_binding_sha256"),
        "attacker_artifact_sha256": records["attacker_artifact"][0]["sha256"],
        "attacker_generated_plan_file_sha256": records[
            "attacker_generated_plan"][0]["sha256"],
        "attacker_framework_evidence_manifest_sha256": records[
            "attacker_framework_evidence_manifest"][0]["sha256"],
        "attacker_framework_evidence_events_sha256": records[
            "attacker_framework_evidence_events"][0]["sha256"],
    }
    for key, expected in cross_checks.items():
        if bindings.get(key) != expected:
            raise ActiveSecurityError(f"Active-security binding {key} mismatch")
    if records["frozen_plan"][0]["sha256"] != inputs.get("plan_file_sha256"):
        raise ActiveSecurityError("Manifest frozen-plan artifact binding mismatch")
    if records["frozen_deployment_manifest"][0]["sha256"] != \
            inputs.get("deployment_manifest_sha256"):
        raise ActiveSecurityError("Manifest deployment artifact binding mismatch")
    runtime_record, runtime_path = records["frozen_runtime_binding"]
    if runtime_record["sha256"] != inputs.get("runtime_binding_file_sha256"):
        raise ActiveSecurityError("Manifest runtime-binding artifact mismatch")
    frozen_runtime = _load_json(
        runtime_path, "frozen runtime binding", runtime_record["sha256"])
    if frozen_runtime != runtime:
        raise ActiveSecurityError("Frozen runtime binding content mismatch")
    frozen_plan_template = _load_json(
        records["frozen_plan"][1], "frozen plan template",
        records["frozen_plan"][0]["sha256"])
    materialized_plan = materialize_frozen_plan(
        frozen_plan_template, inputs.get("deployment_manifest_sha256"),
        inputs.get("profile"))
    if materialized_plan.get("sha256") != inputs.get("plan_sha256"):
        raise ActiveSecurityError(
            "Frozen plan template does not materialise to the admitted plan")
    if result_record["sha256"] != file_sha256(result_path):
        raise ActiveSecurityError("Active-security result digest mismatch")
    return manifest


def require_active_security_manifest(path, expected_sha256: str) -> dict:
    """Require a digest-pinned passing active gate before another P0 command."""
    path = Path(path).resolve()
    manifest = _load_json(path, "active-security manifest")
    return validate_active_security_manifest(
        manifest, path.parent, expected_manifest_sha256=expected_sha256,
        manifest_path=path)


def run_gated_workload(*, active_security_manifest,
                       active_security_manifest_sha256: str,
                       workload_argv: Sequence[str], timeout_seconds: int,
                       cwd=None) -> int:
    """Run only Paper 4's existing workload CLI after the active gate passes.

    The wrapper accepts an argv array, never a shell command, and intentionally
    restricts the child to ``run_cc_engineering_workload.py``.  Consequently it
    cannot serve as a cloud-provisioning wrapper.
    """
    require_active_security_manifest(
        active_security_manifest, active_security_manifest_sha256)
    argv = _validate_argv(workload_argv, "Workload argv")
    expected_script = (Path(__file__).resolve().parents[2]
                       / "scripts/run_cc_engineering_workload.py").resolve()
    executable = Path(argv[0]).resolve()
    expected_python = Path(sys.executable).resolve()
    script_index = 2 if len(argv) > 2 and argv[1] == "-u" else 1
    if (executable != expected_python or len(argv) <= script_index
            or Path(argv[script_index]).resolve() != expected_script
            or any(value in {"-c", "-m"} for value in argv[:script_index + 1])):
        raise ActiveSecurityError(
            "Gated command must start with this interpreter and the exact workload script")
    if type(timeout_seconds) is not int or timeout_seconds < 1:
        raise ActiveSecurityError("Workload timeout must be a positive integer")
    try:
        completed = subprocess.run(
            argv, cwd=cwd, text=True, capture_output=False, check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise ActiveSecurityError("Gated Paper 4 workload timed out") from exc
    except OSError as exc:
        raise ActiveSecurityError(f"Cannot execute gated Paper 4 workload: {exc}") from exc
    if completed.returncode != 0:
        raise ActiveSecurityError(
            f"Gated Paper 4 workload exited {completed.returncode}")
    return completed.returncode
