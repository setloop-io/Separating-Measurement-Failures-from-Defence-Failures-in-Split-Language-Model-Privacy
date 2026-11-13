"""Fail-closed comparison of paired CC-off and CC-on correctness evidence."""
from __future__ import annotations

import math
from pathlib import Path
import re
from typing import Any, Mapping

from .cc_io import read_json_snapshot, require_unchanged, write_json_exclusive


PAIRED_CORRECTNESS_SCHEMA = "p4-cc-paired-correctness-0.1"
CROSS_MODE_COMPARISON_SCHEMA = "p4-cc-cross-mode-correctness-0.1"
SHA256 = re.compile(r"^[a-f0-9]{64}$")
IDENTIFIER = re.compile(r"^[A-Za-z0-9_.:-]+$")

_ARTIFACT_FIELDS = {
    "schema_version", "status", "binding", "input_sha256", "windows",
    "batch_size", "batches", "reference_logit_batch_sha256",
    "candidate_logit_batch_sha256", "bridge_trace_sha256",
    "bridge_total_wire_bytes", "max_abs_logit_delta", "absolute_loss_delta",
    "token_agreement", "reference_mean_loss", "candidate_mean_loss",
    "raw_logits_persisted", "reason",
}
_BINDING_FIELDS = {
    "pair_id", "session_id", "config_sha256", "checkpoint_sha256",
    "panels_sha256", "hardware_identity_sha256", "pair_manifest_sha256",
    "deployment_manifest_sha256", "protection_mode",
}
_SHARED_BINDING_FIELDS = (
    "pair_id", "config_sha256", "checkpoint_sha256", "panels_sha256",
    "hardware_identity_sha256", "pair_manifest_sha256",
)
_SHARED_ARTIFACT_FIELDS = (
    "input_sha256", "windows", "batch_size", "batches",
    "reference_logit_batch_sha256",
)


class CorrectnessComparisonError(ValueError):
    """A correctness artifact is malformed, unqualified, or not paired."""


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise CorrectnessComparisonError(f"{label} must be a lowercase SHA-256")
    return value


def _require_identifier(value: Any, label: str) -> str:
    if (not isinstance(value, str) or not 1 <= len(value) <= 256
            or IDENTIFIER.fullmatch(value) is None):
        raise CorrectnessComparisonError(f"{label} is not a valid identifier")
    return value


def _require_positive_integer(value: Any, label: str) -> int:
    if type(value) is not int or value < 1:
        raise CorrectnessComparisonError(f"{label} must be a positive integer")
    return value


def _require_finite_number(value: Any, label: str, *, minimum: float | None = None,
                           maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CorrectnessComparisonError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise CorrectnessComparisonError(f"{label} must be a finite number")
    if minimum is not None and number < minimum:
        raise CorrectnessComparisonError(f"{label} is below its valid range")
    if maximum is not None and number > maximum:
        raise CorrectnessComparisonError(f"{label} is above its valid range")
    return number


def _require_digest_sequence(value: Any, label: str, batches: int) -> list[str]:
    if not isinstance(value, list) or len(value) != batches:
        raise CorrectnessComparisonError(
            f"{label} must contain exactly one digest per batch")
    for index, digest in enumerate(value):
        _require_sha256(digest, f"{label}[{index}]")
    return value


def validate_paired_correctness_artifact(
        artifact: Mapping[str, Any], *, expected_mode: str) -> dict[str, Any]:
    """Validate one digest-only, within-arm paired correctness PASS artifact."""
    if not isinstance(artifact, Mapping) or set(artifact) != _ARTIFACT_FIELDS:
        raise CorrectnessComparisonError("paired correctness artifact fields differ")
    if artifact.get("schema_version") != PAIRED_CORRECTNESS_SCHEMA:
        raise CorrectnessComparisonError("paired correctness schema differs")
    if artifact.get("status") != "PASS":
        raise CorrectnessComparisonError(
            f"{expected_mode} paired correctness is not PASS")
    if artifact.get("raw_logits_persisted") is not False:
        raise CorrectnessComparisonError("paired correctness must not persist raw logits")
    reason = artifact.get("reason")
    if not isinstance(reason, str) or not reason.strip() or reason != reason.strip():
        raise CorrectnessComparisonError("paired correctness reason is invalid")

    binding = artifact.get("binding")
    if not isinstance(binding, Mapping) or set(binding) != _BINDING_FIELDS:
        raise CorrectnessComparisonError("paired correctness binding fields differ")
    if binding.get("protection_mode") != expected_mode:
        raise CorrectnessComparisonError(
            f"paired correctness protection mode is not {expected_mode}")
    _require_identifier(binding.get("pair_id"), "binding.pair_id")
    _require_identifier(binding.get("session_id"), "binding.session_id")
    for field in (
            "config_sha256", "checkpoint_sha256", "panels_sha256",
            "hardware_identity_sha256", "pair_manifest_sha256",
            "deployment_manifest_sha256"):
        _require_sha256(binding.get(field), f"binding.{field}")

    _require_sha256(artifact.get("input_sha256"), "input_sha256")
    windows = _require_positive_integer(artifact.get("windows"), "windows")
    batch_size = _require_positive_integer(artifact.get("batch_size"), "batch_size")
    batches = _require_positive_integer(artifact.get("batches"), "batches")
    if batches != math.ceil(windows / batch_size):
        raise CorrectnessComparisonError("batch count does not cover the declared windows")
    _require_digest_sequence(
        artifact.get("reference_logit_batch_sha256"),
        "reference_logit_batch_sha256", batches)
    _require_digest_sequence(
        artifact.get("candidate_logit_batch_sha256"),
        "candidate_logit_batch_sha256", batches)
    _require_digest_sequence(
        artifact.get("bridge_trace_sha256"), "bridge_trace_sha256", batches)
    if (type(artifact.get("bridge_total_wire_bytes")) is not int
            or artifact["bridge_total_wire_bytes"] < 1):
        raise CorrectnessComparisonError("bridge_total_wire_bytes must be positive")
    _require_finite_number(
        artifact.get("max_abs_logit_delta"), "max_abs_logit_delta", minimum=0.0)
    recorded_loss_delta = _require_finite_number(
        artifact.get("absolute_loss_delta"), "absolute_loss_delta", minimum=0.0)
    _require_finite_number(
        artifact.get("token_agreement"), "token_agreement", minimum=0.0, maximum=1.0)
    reference_loss = _require_finite_number(
        artifact.get("reference_mean_loss"), "reference_mean_loss")
    candidate_loss = _require_finite_number(
        artifact.get("candidate_mean_loss"), "candidate_mean_loss")
    if recorded_loss_delta != abs(candidate_loss - reference_loss):
        raise CorrectnessComparisonError("absolute_loss_delta is internally inconsistent")
    return dict(artifact)


def compare_paired_correctness_modes(
        cc_off: Mapping[str, Any], cc_on: Mapping[str, Any], *,
        cc_off_artifact_sha256: str,
        cc_on_artifact_sha256: str) -> dict[str, Any]:
    """Compare exact candidate-logit digests without inventing a tolerance result.

    The two inputs already contain trusted, within-arm measurements against the
    same plaintext reference.  Matching candidate digests prove a zero cross-mode
    candidate-output delta.  A mismatch contains no magnitude information and is
    therefore ``UNDETERMINED``, never a guessed failure or pass.
    """
    off = validate_paired_correctness_artifact(cc_off, expected_mode="cc-off")
    on = validate_paired_correctness_artifact(cc_on, expected_mode="cc-on")
    off_binding = off["binding"]
    on_binding = on["binding"]
    for field in _SHARED_BINDING_FIELDS:
        if off_binding[field] != on_binding[field]:
            raise CorrectnessComparisonError(
                f"cross-mode shared binding differs: {field}")
    for field in _SHARED_ARTIFACT_FIELDS:
        if off[field] != on[field]:
            raise CorrectnessComparisonError(
                f"cross-mode shared correctness field differs: {field}")

    _require_sha256(cc_off_artifact_sha256, "cc_off_artifact_sha256")
    _require_sha256(cc_on_artifact_sha256, "cc_on_artifact_sha256")

    logits_equal = (
        off["candidate_logit_batch_sha256"]
        == on["candidate_logit_batch_sha256"]
    )
    if logits_equal:
        status = "PASS"
        max_delta = 0.0
        reason = (
            "CC-off and CC-on candidate logit digests match for every bound batch; "
            "the cross-mode maximum candidate-logit delta is zero."
        )
    else:
        status = "UNDETERMINED"
        max_delta = None
        reason = (
            "CC-off and CC-on candidate logit digests differ. Raw logits are not "
            "persisted, so their cross-mode numerical delta cannot be inferred."
        )

    return {
        "schema_version": CROSS_MODE_COMPARISON_SCHEMA,
        "status": status,
        "binding": {
            **{field: off_binding[field] for field in _SHARED_BINDING_FIELDS},
            "cc_off_session_id": off_binding["session_id"],
            "cc_on_session_id": on_binding["session_id"],
            "cc_off_deployment_manifest_sha256": off_binding[
                "deployment_manifest_sha256"],
            "cc_on_deployment_manifest_sha256": on_binding[
                "deployment_manifest_sha256"],
            "protection_modes": {"cc_off": "cc-off", "cc_on": "cc-on"},
        },
        "source_artifact_sha256": {
            "cc_off": cc_off_artifact_sha256,
            "cc_on": cc_on_artifact_sha256,
        },
        "input_sha256": off["input_sha256"],
        "windows": off["windows"],
        "batch_size": off["batch_size"],
        "batches": off["batches"],
        "reference_logit_batch_sha256": off["reference_logit_batch_sha256"],
        "candidate_logit_batch_sha256": {
            "cc_off": off["candidate_logit_batch_sha256"],
            "cc_on": on["candidate_logit_batch_sha256"],
        },
        "logits_exactly_equal": logits_equal,
        "max_abs_candidate_logit_delta": max_delta,
        "raw_logits_persisted": False,
        "reason": reason,
    }


def compare_paired_correctness_files(
        cc_off_path: str | Path, cc_on_path: str | Path,
        output_path: str | Path) -> dict[str, Any]:
    """Snapshot two input artifacts and write one new comparison exclusively."""
    cc_off, off_snapshot = read_json_snapshot(cc_off_path)
    cc_on, on_snapshot = read_json_snapshot(cc_on_path)
    result = compare_paired_correctness_modes(
        cc_off, cc_on,
        cc_off_artifact_sha256=off_snapshot.sha256,
        cc_on_artifact_sha256=on_snapshot.sha256,
    )
    require_unchanged(off_snapshot)
    require_unchanged(on_snapshot)
    write_json_exclusive(output_path, result)
    return result
