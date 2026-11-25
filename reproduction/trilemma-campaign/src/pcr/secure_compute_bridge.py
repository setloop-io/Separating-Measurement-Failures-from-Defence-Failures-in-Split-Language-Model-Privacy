"""Prospective admission bridge for the shared d0-q8 secure-compute cohort.

This module never executes a workload and never turns engineering evidence into
an R1 result.  It validates the cross-repository pre-order packet and classifies
later LLM-Attacker audit artifacts conservatively.  Historical R1 observations,
checkpoints, scores, or qualification decisions are forbidden inputs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

PACKET_SCHEMA = "p3.secure_compute_preorder.v1"
PROTOCOL_ID = "cross-paper-secure-compute-d0q8-1.0"
COHORT_ID = "cross-paper-secure-compute-d0q8-engineering-1"
ANCHOR_ID = "p4-d0-q8"
ANCHOR_CORPUS = "gutenberg_p4"
AUDIT_ARTIFACT_SCHEMA = "dtraining.attacker.secure_compute_audit.v1"
AUDIT_PLAN_SCHEMA = "llm_attacker.secure_compute_plan.v1"
AUDIT_GATE_SCHEMA = "llm_attacker.secure_compute_gate.v1"
EVIDENCE_BUNDLE_SCHEMA = "cross-paper.secure-compute-evidence.v1"

REQUIRED_REPOSITORIES = ("paper3", "paper4", "llm_attacker")
REQUIRED_VIEWS = ("external_attacker", "trusted_diagnostic")
REQUIRED_PHASES = (
    "ingress", "tokenisation", "forward", "return", "backward", "update",
    "optimizer", "cpu-gpu-staging", "ipc", "vram", "checkpointing",
    "logging", "crash-restart", "egress",
)
REQUIRED_ATTACK_CATEGORIES = (
    "attestation-key-release", "protocol-integrity",
    "malicious-computation", "boundary-canary", "confidential-host-trace",
)


class SecureComputeBridgeError(ValueError):
    """Fail-closed validation error for prospective secure-compute records."""


def _sha256(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


def _git_revision(value: Any) -> bool:
    return (isinstance(value, str) and len(value) in {40, 64}
            and all(c in "0123456789abcdef" for c in value))


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SecureComputeBridgeError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise SecureComputeBridgeError(f"{path}: expected a JSON object")
    return value


def validate_preorder_packet(packet: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate a draft or frozen packet without authorising execution."""
    if not isinstance(packet, Mapping) or packet.get("schema") != PACKET_SCHEMA:
        raise SecureComputeBridgeError(f"expected schema {PACKET_SCHEMA}")
    if packet.get("protocol_id") != PROTOCOL_ID or packet.get("cohort_id") != COHORT_ID:
        raise SecureComputeBridgeError("shared protocol/cohort identity mismatch")
    if packet.get("status") not in {"draft", "frozen"}:
        raise SecureComputeBridgeError("status must be draft or frozen")
    anchor = packet.get("anchor")
    if not isinstance(anchor, Mapping):
        raise SecureComputeBridgeError("anchor must be an object")
    expected_anchor = {
        "id": ANCHOR_ID, "source_repository": "paper4",
        "corpus": ANCHOR_CORPUS, "start": 2, "span": 4,
        "width": 256, "remote_depth": 1, "q_bits": 8,
        "sequence_length": 128, "batch_size": 32, "training_steps": 2000,
    }
    if any(anchor.get(key) != value for key, value in expected_anchor.items()):
        raise SecureComputeBridgeError("anchor differs from the prospective d0-q8 contract")
    if packet.get("legacy_r1_evidence_used") is not False:
        raise SecureComputeBridgeError("legacy R1 evidence must not be used")
    bindings = packet.get("bindings")
    if not isinstance(bindings, Mapping):
        raise SecureComputeBridgeError("bindings must be an object")
    if bindings.get("evidence_bundle_schema") != EVIDENCE_BUNDLE_SCHEMA:
        raise SecureComputeBridgeError("evidence bundle schema mismatch")
    revisions = bindings.get("repository_revisions")
    if not isinstance(revisions, Mapping) or set(revisions) != set(REQUIRED_REPOSITORIES):
        raise SecureComputeBridgeError("all three repository revisions are required")
    for name, revision in revisions.items():
        if revision is not None and not _git_revision(revision):
            raise SecureComputeBridgeError(f"invalid {name} revision digest")
    for key in ("deployment_manifest_sha256", "attack_plan_sha256",
                "workload_config_sha256"):
        value = bindings.get(key)
        if value is not None and not _sha256(value):
            raise SecureComputeBridgeError(f"invalid bindings.{key}")
    observation = packet.get("observation_contract")
    if not isinstance(observation, Mapping):
        raise SecureComputeBridgeError("observation_contract must be an object")
    if set(observation.get("views", [])) != set(REQUIRED_VIEWS):
        raise SecureComputeBridgeError("trusted and external views are both required")
    if set(observation.get("protected_phases", [])) != set(REQUIRED_PHASES):
        raise SecureComputeBridgeError("protected phase inventory is incomplete")
    if observation.get("common_identity_fields") != [
            "request_id", "session_id", "step_id"]:
        raise SecureComputeBridgeError("common event identity fields mismatch")
    if observation.get("trusted_view_is_attacker_visible") is not False:
        raise SecureComputeBridgeError("trusted diagnostics cannot be attacker-visible")
    attacks = packet.get("attack_contract")
    if not isinstance(attacks, Mapping):
        raise SecureComputeBridgeError("attack_contract must be an object")
    if not set(REQUIRED_ATTACK_CATEGORIES).issubset(attacks.get("categories", [])):
        raise SecureComputeBridgeError("required attack categories are incomplete")
    if attacks.get("invalid_secret_release_tolerance") != 0 \
            or attacks.get("mutation_accept_tolerance") != 0 \
            or attacks.get("external_canary_tolerance") != 0:
        raise SecureComputeBridgeError("security tolerances must be exactly zero")
    governance = packet.get("governance")
    if not isinstance(governance, Mapping):
        raise SecureComputeBridgeError("governance must be an object")
    if governance.get("stage") != "engineering" \
            or governance.get("scientific_evidence") is not False:
        raise SecureComputeBridgeError("pre-order packet must remain engineering-only")
    if governance.get("confirmation_authorized") is not False:
        raise SecureComputeBridgeError("confirmation cannot be authorized here")
    claims = packet.get("claim_status")
    if not isinstance(claims, Mapping) or set(claims) != {
            "paper3", "paper4", "llm_attacker"}:
        raise SecureComputeBridgeError("all three scoped claim statuses are required")
    if any(value != "UNDETERMINED_NO_EXECUTION" for value in claims.values()):
        raise SecureComputeBridgeError("pre-execution claims must remain undetermined")
    return packet


def preorder_blockers(packet: Mapping[str, Any]) -> list[str]:
    """Return explicit blockers; an empty list is necessary, not authorization."""
    validate_preorder_packet(packet)
    blockers: list[str] = []
    if packet["status"] != "frozen":
        blockers.append("packet_not_frozen")
    bindings = packet["bindings"]
    for key in ("deployment_manifest_sha256", "attack_plan_sha256",
                "workload_config_sha256"):
        if bindings.get(key) is None:
            blockers.append(f"missing_{key}")
    for name, revision in bindings["repository_revisions"].items():
        if revision is None:
            blockers.append(f"missing_{name}_revision")
    hardware = packet.get("hardware_selection")
    for key in ("provider", "sku", "cpu_tee", "gpu", "cc_mode",
                "attestation_service", "billing_unit", "quote_reference"):
        if not isinstance(hardware, Mapping) or not hardware.get(key):
            blockers.append(f"missing_hardware_{key}")
    budget = packet.get("budget")
    for key in ("currency", "maximum_cost", "maximum_wall_minutes",
                "review_reference"):
        if not isinstance(budget, Mapping) or budget.get(key) in (None, ""):
            blockers.append(f"missing_budget_{key}")
    if not isinstance(budget, Mapping) or budget.get("approved") is not True:
        blockers.append("budget_not_approved")
    authorization = packet.get("authorization")
    if not isinstance(authorization, Mapping) \
            or authorization.get("hardware_order_authorized") is not True:
        blockers.append("hardware_order_not_authorized")
    if not isinstance(authorization, Mapping) or not authorization.get("reference"):
        blockers.append("authorization_reference_missing")
    return blockers


def classify_audit_artifact(artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Classify new engineering evidence without producing a Paper 3 result."""
    if not isinstance(artifact, Mapping) \
            or artifact.get("schema") != AUDIT_ARTIFACT_SCHEMA:
        raise SecureComputeBridgeError("not a secure-compute audit artifact")
    config = artifact.get("config")
    plan = artifact.get("plan")
    gate = artifact.get("gate")
    if not all(isinstance(value, Mapping) for value in (config, plan, gate)):
        raise SecureComputeBridgeError("artifact config, plan and gate are required")
    if plan.get("schema") != AUDIT_PLAN_SCHEMA or gate.get("schema") != AUDIT_GATE_SCHEMA:
        raise SecureComputeBridgeError("audit plan/gate schema mismatch")
    if config.get("plan_sha256") != plan.get("sha256"):
        raise SecureComputeBridgeError("artifact plan binding mismatch")
    if config.get("deployment_manifest_sha256") != plan.get(
            "deployment_manifest_sha256"):
        raise SecureComputeBridgeError("deployment-manifest binding mismatch")
    if config.get("toy") is True:
        return {"engineering_status": "NOT_EVIDENCE_TOY",
                "paper3_claim_status": "UNDETERMINED_NO_ADMISSIBLE_EXECUTION"}
    categories = {cell.get("category") for cell in gate.get("cells", [])
                  if isinstance(cell, Mapping)}
    if not set(REQUIRED_ATTACK_CATEGORIES).issubset(categories):
        raise SecureComputeBridgeError("audit gate lacks required categories")
    overall = gate.get("overall")
    if overall == "FAIL":
        engineering = "REJECTED_ENGINEERING_GATE"
    elif overall == "UNASSESSED":
        engineering = "UNDETERMINED_UNASSESSED_SECURITY"
    elif overall == "PASS":
        engineering = "ADMITTED_ENGINEERING_SECURITY"
    else:
        raise SecureComputeBridgeError("unknown audit gate verdict")
    return {
        "engineering_status": engineering,
        "paper3_claim_status": "UNDETERMINED_PENDING_FRESH_SCIENTIFIC_COHORT",
        "scientific_evidence": False,
        "updates_existing_r1_verdict": False,
    }
