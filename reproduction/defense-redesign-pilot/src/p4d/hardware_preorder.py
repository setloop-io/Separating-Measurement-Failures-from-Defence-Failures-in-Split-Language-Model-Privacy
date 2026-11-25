"""Fail-closed validation for the TEE/FHE hardware pre-order packet.

The packet is a procurement input, not an order instruction. This module has no
cloud SDK dependency and cannot create, start, stop, or delete resources.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping
from urllib.parse import urlparse

from .cc_keys import KeyCeremonyError, validate_key_ceremony


SCHEMA = "p4.tee_fhe_hardware_preorder.v1"
PROTOCOL_ID = "cross-paper-secure-compute-d0q8-1.0"
ANCHOR = {
    "id": "p4-d0-q8",
    "corpus": "gutenberg_p4",
    "start": 2,
    "span": 4,
    "width": 256,
    "remote_depth": 1,
    "q_bits": 8,
    "sequence_length": 128,
    "batch_size": 32,
    "training_steps": 2000,
}
ALLOWED_CPU_TEES = {"AMD SEV-SNP", "Intel TDX"}
ALLOWED_PROVISIONING_MODELS = {
    "pay-as-you-go",
    "standard-on-demand",
    "spot",
    "flex-start",
}
ALLOWED_BILLING_GRANULARITIES = {"per-second", "per-minute", "per-hour"}
ALLOWED_COMPARATOR_STRATEGIES = {
    "same-instance-reboot-mode-toggle",
    "matched-separate-instance",
}
REQUIRED_STAGES = (
    "arrival-admission",
    "active-security",
    "same-checkpoint-correctness",
    "paired-cc-on-off",
    "post-mitigation-remeasure",
    "evidence-seal-and-teardown",
)
REVISION_BINDINGS = (
    ("paper3_repository_path", "paper3_revision"),
    ("paper4_repository_path", "paper4_revision"),
    ("llm_attacker_repository_path", "llm_attacker_revision"),
)
ARTIFACT_BINDINGS = (
    ("workload_config_path", "workload_config_sha256"),
    ("attack_plan_path", "attack_plan_sha256"),
    ("evidence_schema_path", "evidence_schema_sha256"),
    ("deployment_bundle_path", "deployment_bundle_sha256"),
    ("deployment_bundle_manifest_path", "deployment_bundle_manifest_sha256"),
    ("collector_manifest_path", "collector_manifest_sha256"),
    ("collector_executable_path", "collector_executable_sha256"),
    ("collector_public_key_path", "collector_public_key_sha256"),
    ("key_ceremony_manifest_path", "key_ceremony_manifest_sha256"),
    ("cc_on_journal_public_key_path", "cc_on_journal_public_key_sha256"),
    ("cc_off_journal_public_key_path", "cc_off_journal_public_key_sha256"),
    ("cc_on_heartbeat_public_key_path", "cc_on_heartbeat_public_key_sha256"),
    ("cc_off_heartbeat_public_key_path", "cc_off_heartbeat_public_key_sha256"),
    ("admission_verifier_adapter_path", "admission_verifier_adapter_sha256"),
    ("admission_verifier_public_key_path",
     "admission_verifier_public_key_sha256"),
    ("remote_launch_adapter_path", "remote_launch_adapter_sha256"),
    ("lifecycle_controller_public_key_path",
     "lifecycle_controller_public_key_sha256"),
    ("campaign_public_key_path", "campaign_public_key_sha256"),
    ("ttl_enforcement_path", "ttl_enforcement_sha256"),
    ("deadman_switch_path", "deadman_switch_sha256"),
    ("teardown_runbook_path", "teardown_runbook_sha256"),
    ("budget_enforcement_path", "budget_enforcement_sha256"),
    ("checkpoint_manifest_path", "checkpoint_manifest_sha256"),
    ("checkpoint_artifact_path", "checkpoint_artifact_sha256"),
    ("panels_manifest_path", "panels_manifest_sha256"),
    ("panels_artifact_path", "panels_artifact_sha256"),
)
KEY_CEREMONY_BINDINGS = {
    "external-collector": (
        "collector_public_key_path", "collector_public_key_sha256"),
    "admission-verifier": (
        "admission_verifier_public_key_path",
        "admission_verifier_public_key_sha256",
    ),
    "lifecycle-controller": (
        "lifecycle_controller_public_key_path",
        "lifecycle_controller_public_key_sha256",
    ),
    "campaign": ("campaign_public_key_path", "campaign_public_key_sha256"),
    "cc-on-journal": (
        "cc_on_journal_public_key_path", "cc_on_journal_public_key_sha256"),
    "cc-off-journal": (
        "cc_off_journal_public_key_path", "cc_off_journal_public_key_sha256"),
    "cc-on-heartbeat": (
        "cc_on_heartbeat_public_key_path", "cc_on_heartbeat_public_key_sha256"),
    "cc-off-heartbeat": (
        "cc_off_heartbeat_public_key_path", "cc_off_heartbeat_public_key_sha256"),
}
ROOT = Path(__file__).resolve().parents[2]


class HardwarePreorderError(ValueError):
    """The packet violates the frozen procurement contract."""


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_git_revision(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) in {40, 64}
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_finite_number(value: object, *, positive: bool = False) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    if not math.isfinite(float(value)):
        return False
    return value > 0 if positive else value >= 0


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def _path_is_concrete(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value.strip())
        and value == value.strip()
        and "\x00" not in value
    )


def _official_https_reference(provider: object, value: object) -> bool:
    if not isinstance(value, str):
        return False
    parsed = urlparse(value)
    allowed_hosts = {
        "Microsoft Azure": {"learn.microsoft.com"},
        "Google Cloud": {"cloud.google.com", "docs.cloud.google.com"},
    }
    return (
        parsed.scheme == "https"
        and parsed.hostname in allowed_hosts.get(provider, set())
        and bool(parsed.path)
    )


def _parse_utc_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z", value
    ):
        return None
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        return None
    return parsed if parsed.tzinfo == timezone.utc else None


def _valid_region(provider: str, value: object) -> bool:
    if not isinstance(value, str):
        return False
    if provider == "Microsoft Azure":
        return re.fullmatch(r"[a-z][a-z0-9]{1,31}", value) is not None
    if provider == "Google Cloud":
        return re.fullmatch(r"[a-z]+-[a-z0-9]+[0-9]", value) is not None
    return False


def _valid_zone(provider: str, region: str, value: object) -> bool:
    if not isinstance(value, str):
        return False
    if provider == "Microsoft Azure":
        return re.fullmatch(r"[1-9][0-9]?", value) is not None
    if provider == "Google Cloud":
        return re.fullmatch(re.escape(region) + r"-[a-z]", value) is not None
    return False


def _valid_immutable_image(provider: str, value: object) -> bool:
    if not isinstance(value, str) or not value or value != value.strip():
        return False
    lowered = value.lower()
    if any(token in lowered for token in ("latest", "current", "stable", "*")):
        return False
    if provider == "Microsoft Azure":
        if value.startswith("/"):
            return (
                re.fullmatch(r"/[A-Za-z0-9._/-]+", value) is not None
                and "/versions/" in lowered
                and lowered.rsplit("/versions/", 1)[1] != ""
            )
        parts = value.split(":")
        return (
            len(parts) == 4
            and all(re.fullmatch(r"[A-Za-z0-9._-]+", part) for part in parts)
            and any(character.isdigit() for character in parts[-1])
        )
    if provider == "Google Cloud":
        return (
            re.fullmatch(
                r"projects/[a-z][a-z0-9-]+/(?:global/)?images/[a-z][a-z0-9-]+", value
            )
            is not None
        )
    return False


def _valid_pinned_driver(value: object, minimum_series: object) -> bool:
    if not isinstance(value, str) or re.fullmatch(r"\d+\.\d+\.\d+", value) is None:
        return False
    if not isinstance(minimum_series, int) or isinstance(minimum_series, bool):
        return False
    return int(value.split(".", 1)[0]) >= minimum_series


def _git_revision_exists(repository: Path, revision: str) -> bool:
    if not repository.is_dir():
        return False
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), "cat-file", "-e", f"{revision}^{{commit}}"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _decimal(value: object) -> Decimal | None:
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return number if number.is_finite() else None


def load_packet(path: str | Path) -> dict[str, Any]:
    try:
        packet = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise HardwarePreorderError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(packet, dict):
        raise HardwarePreorderError("hardware pre-order packet must be a JSON object")
    return packet


def validate_packet(packet: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate packet structure and immutable consistency, without authority."""
    if not isinstance(packet, Mapping) or packet.get("schema") != SCHEMA:
        raise HardwarePreorderError(f"expected schema {SCHEMA}")
    if packet.get("protocol_id") != PROTOCOL_ID:
        raise HardwarePreorderError("cross-paper protocol identity mismatch")
    if packet.get("status") not in {"draft", "frozen"}:
        raise HardwarePreorderError("status must be draft or frozen")
    if packet.get("purpose") != "engineering-only":
        raise HardwarePreorderError("the first rental must remain engineering-only")
    if packet.get("scientific_evidence") is not False:
        raise HardwarePreorderError("the pre-order packet cannot authorise scientific evidence")
    if packet.get("secure_compute_profile") != "tee":
        raise HardwarePreorderError(
            "the initial H100 confidential-compute order must use the TEE profile")
    if packet.get("anchor") != ANCHOR:
        raise HardwarePreorderError("anchor differs from the shared d0-q8 contract")

    candidates = packet.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise HardwarePreorderError("at least one hardware candidate is required")
    by_id: dict[str, Mapping[str, Any]] = {}
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise HardwarePreorderError("candidate records must be objects")
        identifier = candidate.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in by_id:
            raise HardwarePreorderError("candidate IDs must be nonempty and unique")
        if candidate.get("provider") not in {"Microsoft Azure", "Google Cloud"}:
            raise HardwarePreorderError(f"{identifier}: unsupported provider")
        if not isinstance(candidate.get("sku"), str) or not candidate["sku"]:
            raise HardwarePreorderError(f"{identifier}: exact SKU is required")
        models = candidate.get("allowed_provisioning_models")
        if (
            not isinstance(models, list)
            or not models
            or len(models) != len(set(models))
            or not set(models).issubset(ALLOWED_PROVISIONING_MODELS)
        ):
            raise HardwarePreorderError(f"{identifier}: invalid provisioning models")
        interruptible_models = candidate.get("interruptible_provisioning_models")
        if (
            not isinstance(interruptible_models, list)
            or not set(interruptible_models).issubset(set(models))
        ):
            raise HardwarePreorderError(f"{identifier}: invalid interruption semantics")
        status = candidate.get("technical_status")
        if status not in {"eligible", "rejected"}:
            raise HardwarePreorderError(f"{identifier}: invalid technical_status")
        if not _official_https_reference(
            candidate.get("provider"), candidate.get("official_reference")
        ):
            raise HardwarePreorderError(f"{identifier}: official HTTPS reference is required")
        if status == "eligible":
            if candidate.get("cpu_tee") not in ALLOWED_CPU_TEES:
                raise HardwarePreorderError(
                    f"{identifier}: eligible CPU TEE must be TDX or SEV-SNP"
                )
            if candidate.get("gpu_cc_capable") is not True:
                raise HardwarePreorderError(f"{identifier}: NVIDIA GPU CC is required")
            if not isinstance(candidate.get("gpu"), str) or not candidate["gpu"].strip():
                raise HardwarePreorderError(f"{identifier}: exact GPU model is required")
            if candidate.get("gpu_count") != 1:
                raise HardwarePreorderError(f"{identifier}: the minimum campaign uses one GPU")
            memory = candidate.get("gpu_memory_gb")
            if not _is_finite_number(memory, positive=True) or memory < 80:
                raise HardwarePreorderError(f"{identifier}: at least 80 GB GPU memory is required")
            if candidate.get("guest_attestation") is not True:
                raise HardwarePreorderError(f"{identifier}: guest attestation is required")
            if candidate.get("secure_boot_required") is not True:
                raise HardwarePreorderError(f"{identifier}: Secure Boot must be required")
        else:
            reason = candidate.get("rejection_reason")
            if not isinstance(reason, str) or not reason:
                raise HardwarePreorderError(f"{identifier}: rejected candidates need a reason")
        by_id[identifier] = candidate

    selected = packet.get("selected_candidate")
    if selected not in by_id:
        raise HardwarePreorderError("selected_candidate must name a declared candidate")
    selected_record = by_id[selected]
    if selected_record.get("technical_status") != "eligible":
        raise HardwarePreorderError("selected candidate is not technically eligible")

    execution = packet.get("execution_contract")
    if not isinstance(execution, Mapping):
        raise HardwarePreorderError("execution_contract is required")
    if execution.get("stages") != list(REQUIRED_STAGES):
        raise HardwarePreorderError("minimum engineering-stage sequence differs")
    if execution.get("maximum_wall_minutes") != 240:
        raise HardwarePreorderError("first rental is capped at four wall-clock hours")
    if execution.get("hard_ttl_minutes") != 240:
        raise HardwarePreorderError("provider TTL must equal the four-hour hard cap")
    deadman_minutes = execution.get("deadman_timeout_minutes")
    if (
        not isinstance(deadman_minutes, int)
        or isinstance(deadman_minutes, bool)
        or not 1 <= deadman_minutes <= 30
    ):
        raise HardwarePreorderError("deadman timeout must be an integer from 1 to 30 minutes")
    interval = execution.get("budget_check_interval_minutes")
    if (
        not isinstance(interval, int)
        or isinstance(interval, bool)
        or not 1 <= interval <= 15
    ):
        raise HardwarePreorderError("budget check interval must be from 1 to 15 minutes")
    for key in (
        "ttl_enforcement_ready",
        "deadman_switch_ready",
        "teardown_automation_ready",
        "budget_enforcement_ready",
    ):
        if execution.get(key) not in {True, False}:
            raise HardwarePreorderError(f"execution_contract.{key} must be boolean")
    for key in (
        "ttl_enforcement_reference",
        "deadman_switch_reference",
        "teardown_runbook_reference",
        "budget_enforcement_reference",
    ):
        value = execution.get(key)
        if value is not None and not _path_is_concrete(value):
            raise HardwarePreorderError(f"execution_contract.{key} must be null or a path")
    if (
        execution.get("teardown_on_completion") is not True
        or execution.get("teardown_on_failure") is not True
    ):
        raise HardwarePreorderError("automatic teardown is required on completion and failure")
    if execution.get("budget_enforcement_mode") != "hard-stop-and-teardown":
        raise HardwarePreorderError("budget enforcement must hard-stop and tear down")
    if execution.get("automatic_scientific_launch") is not False:
        raise HardwarePreorderError("engineering must not automatically launch science")
    if execution.get("stop_on_failed_admission") is not True:
        raise HardwarePreorderError("failed admission must stop before workload execution")
    if execution.get("stop_on_budget_cap") is not True:
        raise HardwarePreorderError("budget enforcement must fail closed")

    bindings = packet.get("bindings")
    if not isinstance(bindings, Mapping):
        raise HardwarePreorderError("immutable bindings object is required")
    for path_key, revision_key in REVISION_BINDINGS:
        path_value = bindings.get(path_key)
        revision = bindings.get(revision_key)
        if not _path_is_concrete(path_value):
            raise HardwarePreorderError(f"bindings.{path_key} is required")
        if revision is not None and not _is_git_revision(revision):
            raise HardwarePreorderError(
                f"bindings.{revision_key} must be null or a 40/64-character Git object ID"
            )
    bound_paths: set[str] = set()
    for path_key, digest_key in ARTIFACT_BINDINGS:
        path_value = bindings.get(path_key)
        digest = bindings.get(digest_key)
        if digest is not None and not _is_sha256(digest):
            raise HardwarePreorderError(
                f"bindings.{digest_key} must be null or lowercase SHA-256"
            )
        if digest is not None and not _path_is_concrete(path_value):
            raise HardwarePreorderError(f"bindings.{digest_key} has no concrete {path_key}")
        if path_value is not None and not _path_is_concrete(path_value):
            raise HardwarePreorderError(f"bindings.{path_key} must be null or a concrete path")
        if digest is not None:
            normalized = str(_resolve_path(ROOT, path_value))
            if normalized in bound_paths:
                raise HardwarePreorderError("artifact digest bindings must use distinct paths")
            bound_paths.add(normalized)
    verifier_id = bindings.get("admission_verifier_id")
    if verifier_id is not None and (
            not isinstance(verifier_id, str)
            or re.fullmatch(r"[A-Za-z0-9_.:-]+", verifier_id) is None
            or re.fullmatch(r"[a-f0-9]{64}", verifier_id) is not None):
        raise HardwarePreorderError("admission verifier identity is invalid")
    verifier_kind = bindings.get("admission_verifier_adapter_kind")
    if verifier_kind not in {None, "vendor_attestation_adapter"}:
        raise HardwarePreorderError(
            "hardware orders require a vendor attestation adapter")
    ceremony_id = bindings.get("key_ceremony_id")
    if ceremony_id is not None and (
            not isinstance(ceremony_id, str)
            or re.fullmatch(r"[A-Za-z0-9_.:-]{3,128}", ceremony_id) is None):
        raise HardwarePreorderError("key ceremony identity is invalid")

    order = packet.get("order_parameters")
    if not isinstance(order, Mapping):
        raise HardwarePreorderError("order_parameters is required")
    for key, expected in (
        ("candidate_id", selected),
        ("provider", selected_record["provider"]),
        ("sku", selected_record["sku"]),
        ("gpu_model", selected_record["gpu"]),
    ):
        if order.get(key) != expected:
            raise HardwarePreorderError(
                f"order_parameters.{key} does not match selected candidate"
            )
    provisioning_model = order.get("provisioning_model")
    if provisioning_model not in selected_record["allowed_provisioning_models"]:
        raise HardwarePreorderError("order provisioning model is not allowed by the candidate")
    expected_interruptible = (
        provisioning_model in selected_record["interruptible_provisioning_models"]
    )
    if order.get("interruptible") is not expected_interruptible:
        raise HardwarePreorderError("order interruption semantics contradict provisioning model")
    if order.get("secure_boot") is not True or order.get("debug_mode") is not False:
        raise HardwarePreorderError("Secure Boot is required and debug mode is forbidden")
    if order.get("gpu_count") != selected_record["gpu_count"] or order.get("gpu_count") != 1:
        raise HardwarePreorderError("order GPU count must match the one-GPU candidate")
    if order.get("auto_delete_os_disk") is not True:
        raise HardwarePreorderError("automatic OS-disk deletion is required")
    if order.get("public_ip") is not False:
        raise HardwarePreorderError("the packet forbids a public IP")
    provider = order["provider"]
    cpu_family = order.get("cpu_family")
    if cpu_family is not None and (
        not isinstance(cpu_family, str)
        or not cpu_family.strip()
        or cpu_family != cpu_family.strip()
    ):
        raise HardwarePreorderError("order CPU family must be an exact nonempty identity")
    region = order.get("region")
    zone = order.get("zone")
    if region is not None and not _valid_region(provider, region):
        raise HardwarePreorderError("order_parameters.region is not a valid provider region")
    if zone is not None and (region is None or not _valid_zone(provider, region, zone)):
        raise HardwarePreorderError("order_parameters.zone is not valid for the provider region")
    image = order.get("immutable_image_reference")
    if image is not None and not _valid_immutable_image(provider, image):
        raise HardwarePreorderError("immutable image reference is mutable or malformed")
    driver = order.get("pinned_driver_version")
    if driver is not None and not _valid_pinned_driver(
        driver, order.get("minimum_driver_series")
    ):
        raise HardwarePreorderError("pinned driver must be an exact supported three-part version")

    comparator = packet.get("cc_off_comparator")
    if not isinstance(comparator, Mapping):
        raise HardwarePreorderError("cc_off_comparator is required")
    if comparator.get("same_physical_host_guaranteed") is not False:
        raise HardwarePreorderError("cloud packet must not claim same-host pairing")
    required_matches = {"gpu", "cpu_family", "region", "image", "driver", "workload"}
    matched_fields = comparator.get("required_matched_fields")
    if (
        not isinstance(matched_fields, list)
        or len(matched_fields) != len(required_matches)
        or set(matched_fields) != required_matches
    ):
        raise HardwarePreorderError("CC-off comparator matching fields differ")
    strategy = comparator.get("strategy")
    if strategy is not None and strategy not in ALLOWED_COMPARATOR_STRATEGIES:
        raise HardwarePreorderError("unsupported CC-off comparator strategy")
    comparator_candidate = comparator.get("candidate_id")
    if comparator_candidate is not None and (
        not isinstance(comparator_candidate, str)
        or not comparator_candidate.strip()
        or comparator_candidate != comparator_candidate.strip()
    ):
        raise HardwarePreorderError("comparator candidate_id must be an exact identity")
    if (strategy == "same-instance-reboot-mode-toggle"
            and comparator_candidate is not None
            and comparator_candidate != order["candidate_id"]):
        raise HardwarePreorderError("same-instance comparator candidate must match the order")
    comparator_model = comparator.get("provisioning_model")
    if (
        comparator_model is not None
        and comparator_model not in ALLOWED_PROVISIONING_MODELS
    ):
        raise HardwarePreorderError("unsupported comparator provisioning model")
    comparator_identity = comparator.get("sku_or_mode")
    if comparator_identity is not None and (
        not isinstance(comparator_identity, str)
        or not comparator_identity.strip()
        or comparator_identity != comparator_identity.strip()
    ):
        raise HardwarePreorderError("comparator SKU or mode must be an exact identity")
    expected_comparator = {
        "provider": order.get("provider"),
        "gpu_model": order.get("gpu_model"),
        "cpu_family": cpu_family,
        "region": region,
        "zone": zone,
        "image_reference": image,
        "driver_version": driver,
    }
    for key, expected in expected_comparator.items():
        value = comparator.get(key)
        if value is not None and expected is not None and value != expected:
            raise HardwarePreorderError(
                f"comparator {key} does not match the selected order"
            )
    comparator_workload = comparator.get("workload_config_sha256")
    bound_workload = bindings.get("workload_config_sha256")
    if comparator_workload is not None and not _is_sha256(comparator_workload):
        raise HardwarePreorderError("comparator workload binding must be lowercase SHA-256")
    if (
        comparator_workload is not None
        and bound_workload is not None
        and comparator_workload != bound_workload
    ):
        raise HardwarePreorderError("comparator workload differs from immutable workload")
    limitations = comparator.get("limitations")
    if limitations is not None and (
        not isinstance(limitations, list)
        or not limitations
        or any(not isinstance(item, str) or not item.strip() for item in limitations)
    ):
        raise HardwarePreorderError("comparator limitations must be a nonempty text list")

    budget = packet.get("budget")
    if not isinstance(budget, Mapping) or budget.get("currency") != "USD":
        raise HardwarePreorderError("budget must be declared in USD")
    proposed = budget.get("proposed_maximum_cost")
    if not _is_finite_number(proposed, positive=True):
        raise HardwarePreorderError("a positive finite proposed maximum cost is required")
    if budget.get("maximum_wall_minutes") != 240:
        raise HardwarePreorderError("budget and execution wall-time caps differ")

    quote = packet.get("quote")
    if not isinstance(quote, Mapping):
        raise HardwarePreorderError("quote object is required")
    if quote.get("currency") != "USD":
        raise HardwarePreorderError("quote currency must be USD")
    expected_quote = {
        "candidate_id": selected,
        "provider": order.get("provider"),
        "sku": order.get("sku"),
        "provisioning_model": order.get("provisioning_model"),
        "region": region,
    }
    for key, expected in expected_quote.items():
        value = quote.get(key)
        if value is not None and expected is not None and value != expected:
            raise HardwarePreorderError(f"quote {key} does not match the selected order")
    for key in ("hourly_rate", "comparator_hourly_rate"):
        value = quote.get(key)
        if value is not None and not _is_finite_number(value, positive=True):
            raise HardwarePreorderError(f"quote.{key} must be positive and finite")
    granularity = quote.get("billing_granularity")
    if granularity is not None and granularity not in ALLOWED_BILLING_GRANULARITIES:
        raise HardwarePreorderError("unsupported quote billing granularity")
    retrieved_at = quote.get("retrieved_at")
    if retrieved_at is not None and _parse_utc_timestamp(retrieved_at) is None:
        raise HardwarePreorderError(
            "quote.retrieved_at must be an ISO-8601 UTC timestamp ending Z"
        )
    quote_reference = quote.get("reference")
    if quote_reference is not None and (
        not isinstance(quote_reference, str)
        or len(quote_reference.strip()) < 8
        or quote_reference != quote_reference.strip()
    ):
        raise HardwarePreorderError("quote.reference must be an exact provider quote identity")
    maximum_age = quote.get("maximum_age_hours")
    if not _is_finite_number(maximum_age, positive=True) or maximum_age > 168:
        raise HardwarePreorderError(
            "quote maximum age must be positive and no more than 168 hours"
        )
    quote_path = quote.get("artifact_path")
    quote_digest = quote.get("artifact_sha256")
    if quote_digest is not None and not _is_sha256(quote_digest):
        raise HardwarePreorderError(
            "quote.artifact_sha256 must be null or lowercase SHA-256"
        )
    if quote_digest is not None and not _path_is_concrete(quote_path):
        raise HardwarePreorderError("quote artifact digest has no concrete artifact path")
    if quote_path is not None and not _path_is_concrete(quote_path):
        raise HardwarePreorderError("quote artifact path must be null or concrete")
    if quote_digest is not None and str(_resolve_path(ROOT, quote_path)) in bound_paths:
        raise HardwarePreorderError("quote and immutable bindings must use distinct paths")
    projection = quote.get("cost_projection")
    if not isinstance(projection, Mapping):
        raise HardwarePreorderError("quote.cost_projection is required")
    positive_projection_fields = {
        "primary_billable_hours",
        "comparator_billable_hours",
        "contingency_cost",
        "projected_maximum_cost",
    }
    for key in (
        "primary_billable_hours",
        "comparator_billable_hours",
        "disk_cost",
        "egress_cost",
        "private_access_cost",
        "contingency_cost",
        "projected_maximum_cost",
    ):
        value = projection.get(key)
        if value is not None and not _is_finite_number(
            value, positive=key in positive_projection_fields
        ):
            raise HardwarePreorderError(f"quote.cost_projection.{key} is invalid")

    authorization = packet.get("authorization")
    if not isinstance(authorization, Mapping):
        raise HardwarePreorderError("authorization object is required")
    if authorization.get("order_authorized") not in {True, False}:
        raise HardwarePreorderError("order_authorized must be boolean")
    return packet


def preorder_blockers(
    packet: Mapping[str, Any],
    *,
    root: str | Path = ROOT,
    now: datetime | None = None,
) -> list[str]:
    """Return everything still required before a human may place the order."""
    validate_packet(packet)
    root_path = Path(root).resolve()
    current_time = now or datetime.now(timezone.utc)
    if current_time.tzinfo is None or current_time.utcoffset() is None:
        raise HardwarePreorderError("readiness time must be timezone-aware")
    current_time = current_time.astimezone(timezone.utc)
    blockers: list[str] = []
    if packet["status"] != "frozen":
        blockers.append("packet_not_frozen")

    bindings = packet["bindings"]
    for path_key, revision_key in REVISION_BINDINGS:
        revision = bindings.get(revision_key)
        if revision is None:
            blockers.append(f"missing_{revision_key}")
            continue
        repository = _resolve_path(root_path, bindings[path_key])
        if not _git_revision_exists(repository, revision):
            blockers.append(f"unverifiable_{revision_key}")

    for path_key, digest_key in ARTIFACT_BINDINGS:
        digest = bindings.get(digest_key)
        if digest is None:
            blockers.append(f"missing_{digest_key}")
            continue
        path = _resolve_path(root_path, bindings[path_key])
        stem = path_key.removesuffix("_path")
        if not path.is_file():
            blockers.append(f"missing_{stem}")
        elif _file_sha256(path) != digest:
            blockers.append(f"changed_{stem}")
    ceremony_path_value = bindings.get("key_ceremony_manifest_path")
    ceremony_digest = bindings.get("key_ceremony_manifest_sha256")
    if ceremony_path_value and ceremony_digest:
        ceremony_path = _resolve_path(root_path, ceremony_path_value)
        if ceremony_path.is_file() and _file_sha256(ceremony_path) == ceremony_digest:
            try:
                ceremony = validate_key_ceremony(ceremony_path)
            except (KeyCeremonyError, OSError):
                blockers.append("invalid_key_ceremony_manifest")
            else:
                if not bindings.get("key_ceremony_id"):
                    blockers.append("missing_key_ceremony_id")
                elif ceremony["ceremony_id"] != bindings["key_ceremony_id"]:
                    blockers.append("changed_key_ceremony_id")
                for role, (path_key, digest_key) in KEY_CEREMONY_BINDINGS.items():
                    path_value = bindings.get(path_key)
                    if not path_value or not bindings.get(digest_key):
                        continue
                    role_binding = ceremony["roles"][role]
                    if (_resolve_path(root_path, path_value) != Path(
                            ceremony["resolved_public_keys"][role])
                            or bindings[digest_key] != role_binding["public_key_sha256"]):
                        blockers.append(
                            f"unbound_{path_key.removesuffix('_path')}_to_key_ceremony")
    elif not bindings.get("key_ceremony_id"):
        blockers.append("missing_key_ceremony_id")
    if not bindings.get("admission_verifier_id"):
        blockers.append("missing_admission_verifier_id")
    if bindings.get("admission_verifier_adapter_kind") != \
            "vendor_attestation_adapter":
        blockers.append("missing_vendor_admission_verifier_adapter")

    order = packet["order_parameters"]
    for key in (
        "cpu_family",
        "region",
        "zone",
        "immutable_image_reference",
        "pinned_driver_version",
    ):
        if not order.get(key):
            blockers.append(f"missing_order_{key}")
    for key in (
        "account_ready",
        "onboarding_access_confirmed",
        "quota_confirmed",
        "capacity_checked",
        "private_access_ready",
    ):
        if order.get(key) is not True:
            blockers.append(key)
    if not order.get("private_access_method"):
        blockers.append("missing_private_access_method")

    execution = packet["execution_contract"]
    for ready_key, reference_key, binding_path_key in (
        ("ttl_enforcement_ready", "ttl_enforcement_reference", "ttl_enforcement_path"),
        ("deadman_switch_ready", "deadman_switch_reference", "deadman_switch_path"),
        (
            "teardown_automation_ready",
            "teardown_runbook_reference",
            "teardown_runbook_path",
        ),
        (
            "budget_enforcement_ready",
            "budget_enforcement_reference",
            "budget_enforcement_path",
        ),
    ):
        if execution.get(ready_key) is not True:
            blockers.append(ready_key)
        if not isinstance(execution.get(reference_key), str) or not execution[
            reference_key
        ].strip():
            blockers.append(f"missing_{reference_key}")
        elif execution[reference_key] != bindings.get(binding_path_key):
            blockers.append(f"unbound_{reference_key}")

    comparator = packet["cc_off_comparator"]
    if comparator.get("strategy") != "matched-separate-instance":
        blockers.append("unsupported_lifecycle_comparator_strategy")
    for key in (
        "strategy",
        "candidate_id",
        "provider",
        "sku_or_mode",
        "provisioning_model",
        "gpu_model",
        "cpu_family",
        "region",
        "zone",
        "image_reference",
        "driver_version",
        "workload_config_sha256",
    ):
        if not comparator.get(key):
            blockers.append(f"missing_comparator_{key}")
    if not comparator.get("limitations"):
        blockers.append("missing_comparator_limitations")
    if comparator.get("matched_instance_limitation_accepted") is not True:
        blockers.append("comparator_limitation_not_accepted")

    quote = packet["quote"]
    for key in (
        "reference",
        "retrieved_at",
        "hourly_rate",
        "comparator_hourly_rate",
        "billing_granularity",
    ):
        if quote.get(key) in (None, ""):
            blockers.append(f"missing_quote_{key}")
    if quote.get("artifact_sha256") is None:
        blockers.append("missing_quote_artifact_sha256")
    elif quote.get("artifact_path"):
        quote_path = _resolve_path(root_path, quote["artifact_path"])
        if not quote_path.is_file():
            blockers.append("missing_quote_artifact")
        elif _file_sha256(quote_path) != quote["artifact_sha256"]:
            blockers.append("changed_quote_artifact")
    timestamp = _parse_utc_timestamp(quote.get("retrieved_at"))
    if timestamp is not None:
        if timestamp > current_time + timedelta(minutes=5):
            blockers.append("quote_timestamp_in_future")
        elif current_time > timestamp + timedelta(hours=float(quote["maximum_age_hours"])):
            blockers.append("quote_stale")

    projection = quote["cost_projection"]
    projection_keys = (
        "primary_billable_hours",
        "comparator_billable_hours",
        "disk_cost",
        "egress_cost",
        "private_access_cost",
        "contingency_cost",
        "projected_maximum_cost",
    )
    for key in projection_keys:
        if projection.get(key) is None:
            blockers.append(f"missing_quote_projection_{key}")
    if all(projection.get(key) is not None for key in projection_keys) and all(
        quote.get(key) is not None for key in ("hourly_rate", "comparator_hourly_rate")
    ):
        four_hours = Decimal(execution["maximum_wall_minutes"]) / Decimal(60)
        primary_hours = _decimal(projection["primary_billable_hours"])
        comparator_hours = _decimal(projection["comparator_billable_hours"])
        if primary_hours < four_hours or comparator_hours < four_hours:
            blockers.append("quote_projection_does_not_cover_per_resource_ttl")
        computed = (
            _decimal(quote["hourly_rate"]) * primary_hours
            + _decimal(quote["comparator_hourly_rate"]) * comparator_hours
            + _decimal(projection["disk_cost"])
            + _decimal(projection["egress_cost"])
            + _decimal(projection["private_access_cost"])
            + _decimal(projection["contingency_cost"])
        )
        declared = _decimal(projection["projected_maximum_cost"])
        if abs(computed - declared) > Decimal("0.01"):
            blockers.append("quote_projection_mismatch")
        proposed = _decimal(packet["budget"]["proposed_maximum_cost"])
        if declared > proposed:
            blockers.append("projected_cost_exceeds_budget")

    budget = packet["budget"]
    if budget.get("approved") is not True:
        blockers.append("budget_not_approved")
    if not budget.get("approval_reference"):
        blockers.append("budget_approval_reference_missing")

    authorization = packet["authorization"]
    if authorization.get("order_authorized") is not True:
        blockers.append("hardware_order_not_authorized")
    if not authorization.get("reference"):
        blockers.append("authorization_reference_missing")
    return blockers


def readiness_report(
    packet: Mapping[str, Any],
    *,
    root: str | Path = ROOT,
    now: datetime | None = None,
) -> dict[str, Any]:
    blockers = preorder_blockers(packet, root=root, now=now)
    return {
        "schema": "p4.tee_fhe_hardware_readiness.v1",
        "protocol_id": packet["protocol_id"],
        "selected_candidate": packet["selected_candidate"],
        "ready_for_order": not blockers,
        "blockers": blockers,
        "creates_cloud_resources": False,
        "scientific_evidence": False,
    }
