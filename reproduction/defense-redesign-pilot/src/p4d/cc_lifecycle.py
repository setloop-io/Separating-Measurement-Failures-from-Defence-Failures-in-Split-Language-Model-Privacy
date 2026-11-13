"""Provider-neutral lifecycle supervision for confidential-compute workloads.

This module deliberately has no cloud SDK and cannot provision a resource.  It
supervises three explicit, shell-free commands: the engineering workload, an
idempotent teardown adapter, and an independent resource-absence verifier.
All enforcement clocks use :func:`time.monotonic_ns`; wall-clock timestamps are
record metadata only.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import base64
import binascii
import fcntl
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import signal
import stat
import subprocess
import tempfile
import threading
import time
from types import FrameType
from typing import Any, Callable, Mapping, Sequence
import uuid


EVENT_SCHEMA = "p4.cc_lifecycle_event.v1"
VALIDATION_SCHEMA = "p4.cc_lifecycle_control_validation.v1"
CONTROL_SCHEMA = "p4.cc_lifecycle_control.v1"
LAUNCH_CONTEXT_SCHEMA = "p4.cc_supervisor_launch_context.v1"
LEDGER_SCHEMA = "p4.cc_campaign_cost_ledger.v3"
ABSENCE_SCHEMA = "p4.cc_resource_absence_verification.v2"
LEGACY_ABSENCE_SCHEMA = "p4.cc_resource_absence_verification.v1"
RECEIPT_SCHEMA = "p4.cc_lifecycle_receipt.v2"
DEPLOYMENT_LIFECYCLE_SCHEMA = "p4.cc_deployment_lifecycle.v1"  # test-only legacy
LIFECYCLE_LAUNCH_SCHEMA = "p4.cc_lifecycle_launch_envelope.v1"
LIFECYCLE_LAUNCH_PAYLOAD_SCHEMA = "p4.cc_lifecycle_launch_payload.v1"
SIGNED_HEARTBEAT_SCHEMA = "p4.cc_signed_heartbeat_envelope.v1"
SIGNED_HEARTBEAT_PAYLOAD_SCHEMA = "p4.cc_signed_heartbeat_payload.v1"
ZERO_HASH = "0" * 64
LAUNCH_CONTEXT_ENV = "P4_CC_SUPERVISOR_LAUNCH_CONTEXT"
LAUNCH_TOKEN_ENV = "P4_CC_SUPERVISOR_LAUNCH_TOKEN"
SHA256_HEX = frozenset("0123456789abcdef")
SAFE_SIGNALS = tuple(
    value for value in (getattr(signal, "SIGINT", None),
                        getattr(signal, "SIGTERM", None),
                        getattr(signal, "SIGHUP", None))
    if value is not None
)
FIXED_COST_FIELDS = (
    "disk_cost",
    "egress_cost",
    "private_access_cost",
    "contingency_cost",
)


CONTROL_SPECS: dict[str, dict[str, Any]] = {
    "ttl": {
        "schema": CONTROL_SCHEMA,
        "control": "ttl",
        "implementation": "p4d.cc_lifecycle",
        "clock": "time.monotonic_ns",
        "packet_bound": "execution_contract.hard_ttl_minutes",
        "runtime_must_not_exceed_packet": True,
        "on_violation": [
            "terminate-workload-group",
            "kill-after-grace",
            "run-teardown-adapter",
            "run-resource-absence-verifier",
        ],
    },
    "deadman": {
        "schema": CONTROL_SCHEMA,
        "control": "deadman",
        "implementation": "p4d.cc_lifecycle",
        "clock": "time.monotonic_ns",
        "packet_bound": "execution_contract.deadman_timeout_minutes",
        "heartbeat_observation": "ed25519-signed-hash-chained-guest-heartbeat",
        "workload_heartbeat_contract": "launch-nonce-arm-instance-and-guest-argv-bound",
        "supervisor_launch_context_required": True,
        "stale_preexisting_heartbeat_gives_credit": False,
        "on_violation": [
            "terminate-workload-group",
            "kill-after-grace",
            "run-teardown-adapter",
            "run-resource-absence-verifier",
        ],
    },
    "teardown": {
        "schema": CONTROL_SCHEMA,
        "control": "teardown",
        "implementation": "p4d.cc_lifecycle",
        "adapter_contract": "explicit-nonempty-json-string-array",
        "shell": False,
        "required_paths": [
            "success",
            "workload-failure",
            "signal",
            "ttl-timeout",
            "deadman-timeout",
            "budget-stop",
            "workload-spawn-failure",
            "supervisor-error",
        ],
        "teardown_failure_forces_nonzero": True,
        "deployment_binds_resource_and_argv": True,
        "successful_exit_is_not_deletion_evidence": True,
        "post_teardown_absence_artifact_required": True,
        "final_receipt_binds_workload_lifecycle_and_ledger": True,
    },
    "budget": {
        "schema": CONTROL_SCHEMA,
        "control": "budget",
        "implementation": "p4d.cc_lifecycle",
        "clock": "time.monotonic_ns",
        "packet_interval_bound": "execution_contract.budget_check_interval_minutes",
        "rate_source": ["quote.hourly_rate", "quote.comparator_hourly_rate"],
        "fixed_cost_sources": [f"quote.cost_projection.{key}" for key in FIXED_COST_FIELDS],
        "cap_source": "budget.proposed_maximum_cost",
        "formula": "ledger_accrued_usd + first_run_fixed_usd + hourly_rate_usd * elapsed_seconds / 3600",
        "campaign_ledger": "exclusive-flock-plus-fsync-atomic-replace",
        "caller_supplied_prior_cost_allowed": False,
        "paired_arms_share_ledger": ["cc-off", "cc-on"],
        "hourly_rate_must_be_positive": True,
        "predictive_hard_stop": True,
        "on_violation": [
            "terminate-workload-group",
            "kill-after-grace",
            "run-teardown-adapter",
            "run-resource-absence-verifier",
        ],
    },
}


class LifecycleError(ValueError):
    """The lifecycle policy, binding, record, or invocation is invalid."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in SHA256_HEX for character in value)
    )


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    _regular_file(path, label=label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LifecycleError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise LifecycleError(f"{label} must be a JSON object")
    return value


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> Path:
    """Durably create one private JSON object without following a target link."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except OSError as exc:
        raise LifecycleError(f"artifact path must be new and writable: {path.name}") from exc
    try:
        payload = json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"
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
    return path


def _atomic_private_json(path: Path, value: Mapping[str, Any]) -> None:
    """Replace a private JSON state file while its separate lock is held."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        payload = json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _regular_file(path: Path, *, label: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise LifecycleError(f"{label} is not a readable regular file") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise LifecycleError(f"{label} must be a regular file, not a symlink")
    return metadata


def _decimal(value: Any, *, label: str, positive: bool = False) -> Decimal:
    if isinstance(value, bool):
        raise LifecycleError(f"{label} must be numeric")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise LifecycleError(f"{label} must be numeric") from exc
    if not result.is_finite() or result < 0 or (positive and result <= 0):
        qualifier = "positive and finite" if positive else "nonnegative and finite"
        raise LifecycleError(f"{label} must be {qualifier}")
    return result


def _seconds(value: Any, *, label: str) -> Decimal:
    result = _decimal(value, label=label, positive=True)
    if result > Decimal("86400"):
        raise LifecycleError(f"{label} exceeds the one-day implementation bound")
    return result


def _seconds_to_ns(value: Decimal) -> int:
    result = int(value * Decimal(1_000_000_000))
    if result < 1:
        raise LifecycleError("lifecycle interval rounds below one nanosecond")
    return result


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


@dataclass(frozen=True)
class ArgvBinding:
    """A validated command plus the digest of its source JSON file."""

    argv: tuple[str, ...]
    sha256: str
    bytes: int


def load_argv_json(path: str | Path) -> ArgvBinding:
    """Load an explicit command array without ever interpreting a shell string."""
    source = Path(path)
    metadata = _regular_file(source, label="argv JSON")
    if metadata.st_size < 2 or metadata.st_size > 1024 * 1024:
        raise LifecycleError("argv JSON size is outside the supported range")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LifecycleError("argv JSON is not valid UTF-8 JSON") from exc
    if (
        not isinstance(value, list)
        or not 1 <= len(value) <= 256
        or any(
            not isinstance(item, str)
            or not item
            or len(item) > 4096
            or "\x00" in item
            for item in value
        )
    ):
        raise LifecycleError("argv JSON must be a nonempty bounded string array")
    return ArgvBinding(tuple(value), _sha256_file(source), metadata.st_size)


def _require_adapter_command(
    binding: ArgvBinding, *, adapter_path: Path, action: str, cwd: Path,
) -> None:
    resolved_arguments = []
    for value in binding.argv:
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = cwd / candidate
        resolved_arguments.append(candidate.resolve(strict=False))
    if adapter_path not in resolved_arguments or action not in binding.argv:
        raise LifecycleError(
            f"{action} argv must invoke the frozen remote-launch adapter")


@dataclass(frozen=True)
class SupervisorLaunchContext:
    """Validated one-run capability issued by :func:`run_supervised`."""

    run_id: str
    packet_sha256: str
    deployment_manifest_sha256: str
    workload_argv_sha256: str
    heartbeat_path: Path
    output_path: Path
    record_path: Path
    resource_identity_sha256: str
    context_path: Path
    context_sha256: str


def _launch_context_body(
    *, run_id: str, packet_sha256: str, deployment_manifest_sha256: str,
    workload: ArgvBinding, heartbeat_path: Path, output_path: Path,
    record_path: Path, resource_identity_sha256: str, supervisor_pid: int,
    token_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": LAUNCH_CONTEXT_SCHEMA,
        "run_id": run_id,
        "packet_sha256": packet_sha256,
        "deployment_manifest_sha256": deployment_manifest_sha256,
        "workload_argv_sha256": workload.sha256,
        "workload_argv_bytes": workload.bytes,
        "heartbeat_path": str(heartbeat_path),
        "output_path": str(output_path),
        "record_path": str(record_path),
        "resource_identity_sha256": resource_identity_sha256,
        "supervisor_pid": supervisor_pid,
        "issued_monotonic_ns": time.monotonic_ns(),
        "token_sha256": token_sha256,
    }


def _issue_launch_context(
    path: Path, *, run_id: str, packet_sha256: str,
    deployment_manifest_sha256: str, workload: ArgvBinding,
    heartbeat_path: Path, output_path: Path, record_path: Path,
    resource_identity_sha256: str,
) -> tuple[str, str]:
    """Create a private HMAC-authenticated context and return token and digest."""
    token = secrets.token_hex(32)
    body = _launch_context_body(
        run_id=run_id,
        packet_sha256=packet_sha256,
        deployment_manifest_sha256=deployment_manifest_sha256,
        workload=workload,
        heartbeat_path=heartbeat_path,
        output_path=output_path,
        record_path=record_path,
        resource_identity_sha256=resource_identity_sha256,
        supervisor_pid=os.getpid(),
        token_sha256=_sha256_bytes(token.encode("ascii")),
    )
    authentication = hmac.new(
        token.encode("ascii"), _canonical_bytes(body), hashlib.sha256,
    ).hexdigest()
    _write_json_exclusive(path, {**body, "authentication_hmac_sha256": authentication})
    return token, _sha256_file(path)


def require_supervisor_launch_context(
    *, expected_packet_sha256: str,
    expected_deployment_manifest_sha256: str,
    expected_output_path: str | Path,
) -> SupervisorLaunchContext:
    """Reject a production workload not launched by the live supervisor.

    The capability secret is supplied only to the supervisor's child process and
    is not present in argv, the context artifact, or the lifecycle journal.
    This is a process-bound launch control, not a provider attestation claim.
    """
    context_value = os.environ.get(LAUNCH_CONTEXT_ENV)
    token = os.environ.get(LAUNCH_TOKEN_ENV)
    if not context_value or not token:
        raise PermissionError("production workload requires supervisor launch context")
    context_path = Path(context_value).resolve()
    metadata = _regular_file(context_path, label="supervisor launch context")
    if stat.S_IMODE(metadata.st_mode) & 0o077:
        raise PermissionError("supervisor launch context must not be group/world accessible")
    document = _load_json_object(context_path, label="supervisor launch context")
    expected_keys = {
        "schema", "run_id", "packet_sha256", "deployment_manifest_sha256",
        "workload_argv_sha256", "workload_argv_bytes", "heartbeat_path",
        "output_path", "record_path", "resource_identity_sha256",
        "supervisor_pid", "issued_monotonic_ns", "token_sha256",
        "authentication_hmac_sha256",
    }
    if set(document) != expected_keys or document.get("schema") != LAUNCH_CONTEXT_SCHEMA:
        raise PermissionError("supervisor launch context fields differ")
    authentication = document.pop("authentication_hmac_sha256")
    expected_authentication = hmac.new(
        token.encode("ascii"), _canonical_bytes(document), hashlib.sha256,
    ).hexdigest()
    if (
        not hmac.compare_digest(str(authentication), expected_authentication)
        or not hmac.compare_digest(
            str(document.get("token_sha256")),
            _sha256_bytes(token.encode("ascii")),
        )
    ):
        raise PermissionError("supervisor launch context authentication failed")
    for key in (
        "packet_sha256", "deployment_manifest_sha256", "workload_argv_sha256",
        "resource_identity_sha256", "token_sha256",
    ):
        if not _is_sha256(document.get(key)):
            raise PermissionError(f"supervisor launch context {key} is invalid")
    try:
        uuid.UUID(str(document.get("run_id")))
    except (ValueError, AttributeError) as exc:
        raise PermissionError("supervisor launch run ID is invalid") from exc
    if document["packet_sha256"] != expected_packet_sha256:
        raise PermissionError("supervisor launch packet binding differs")
    if document["deployment_manifest_sha256"] != expected_deployment_manifest_sha256:
        raise PermissionError("supervisor launch deployment binding differs")
    output_path = Path(str(document["output_path"])).resolve()
    if output_path != Path(expected_output_path).resolve():
        raise PermissionError("supervisor launch output binding differs")
    supervisor_pid = document.get("supervisor_pid")
    if type(supervisor_pid) is not int or supervisor_pid <= 1:
        raise PermissionError("supervisor launch parent identity is invalid")
    if os.getppid() != supervisor_pid:
        raise PermissionError("supervisor launch context is not owned by this parent")
    try:
        os.kill(supervisor_pid, 0)
    except OSError as exc:
        raise PermissionError("supervisor process is no longer live") from exc
    heartbeat_path = Path(str(document["heartbeat_path"])).resolve()
    record_path = Path(str(document["record_path"])).resolve()
    context = SupervisorLaunchContext(
        run_id=document["run_id"],
        packet_sha256=document["packet_sha256"],
        deployment_manifest_sha256=document["deployment_manifest_sha256"],
        workload_argv_sha256=document["workload_argv_sha256"],
        heartbeat_path=heartbeat_path,
        output_path=output_path,
        record_path=record_path,
        resource_identity_sha256=document["resource_identity_sha256"],
        context_path=context_path,
        context_sha256=_sha256_file(context_path),
    )
    # Consume the capability before the workload spawns collectors or helpers.
    os.environ.pop(LAUNCH_TOKEN_ENV, None)
    os.environ.pop(LAUNCH_CONTEXT_ENV, None)
    return context


def emit_lifecycle_heartbeat(
    context: SupervisorLaunchContext, stage: str, *, step_id: int | None = None,
) -> None:
    """Atomically advance the supervisor-observed liveness marker."""
    if (
        not isinstance(stage, str)
        or not stage
        or len(stage) > 128
        or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for character in stage)
    ):
        raise ValueError("heartbeat stage is invalid")
    if step_id is not None and (type(step_id) is not int or step_id < 0):
        raise ValueError("heartbeat step ID is invalid")
    path = context.heartbeat_path
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.is_symlink():
        raise PermissionError("heartbeat target cannot be a symlink")
    payload = {
        "schema": "p4.cc_workload_heartbeat.v1",
        "run_id": context.run_id,
        "stage": stage,
        "step_id": step_id,
        "monotonic_ns": time.monotonic_ns(),
    }
    _atomic_private_json(path, payload)


def validate_control_artifact(path: str | Path, control: str) -> dict[str, Any]:
    """Validate one byte-addressable static control declaration."""
    if control not in CONTROL_SPECS:
        raise LifecycleError("unknown lifecycle control")
    source = Path(path)
    metadata = _regular_file(source, label=f"{control} control")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LifecycleError(f"{control} control is invalid JSON") from exc
    if value != CONTROL_SPECS[control]:
        raise LifecycleError(f"{control} control differs from the implemented contract")
    return {
        "control": control,
        "sha256": _sha256_file(source),
        "bytes": metadata.st_size,
    }


def validate_control_bundle(
    controls: Mapping[str, str | Path],
    *,
    packet_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return a deterministic offline validation report; contact no provider."""
    if set(controls) != set(CONTROL_SPECS):
        raise LifecycleError("exactly ttl, deadman, teardown, and budget controls are required")
    records = {
        name: validate_control_artifact(controls[name], name)
        for name in sorted(CONTROL_SPECS)
    }
    report: dict[str, Any] = {
        "schema": VALIDATION_SCHEMA,
        "valid": True,
        "creates_cloud_resources": False,
        "controls": records,
    }
    if packet_path is not None:
        source = Path(packet_path)
        metadata = _regular_file(source, label="hardware pre-order packet")
        try:
            packet = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LifecycleError("hardware pre-order packet is invalid JSON") from exc
        bounds = _packet_bounds(packet)
        report["packet"] = {
            "sha256": _sha256_file(source),
            "bytes": metadata.st_size,
            "hard_ttl_seconds": str(bounds["ttl"]),
            "deadman_seconds": str(bounds["deadman"]),
            "budget_check_seconds": str(bounds["budget_interval"]),
            "budget_cap_usd": str(bounds["budget_cap"]),
        }
    report["validation_sha256"] = _sha256_bytes(_canonical_bytes(report))
    return report


def write_validation_report(report: Mapping[str, Any], path: str | Path) -> Path:
    """Create, never overwrite, a content-addressable offline report."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except OSError as exc:
        raise LifecycleError("validation report path already exists or is unavailable") from exc
    try:
        payload = json.dumps(report, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            target.unlink()
        except OSError:
            pass
        raise
    return target


def _packet_bounds(packet: Mapping[str, Any]) -> dict[str, Decimal]:
    if not isinstance(packet, Mapping):
        raise LifecycleError("hardware pre-order packet must be an object")
    execution = packet.get("execution_contract")
    quote = packet.get("quote")
    budget = packet.get("budget")
    if not all(isinstance(value, Mapping) for value in (execution, quote, budget)):
        raise LifecycleError("packet lacks lifecycle execution, quote, or budget objects")
    if quote.get("currency") != "USD" or budget.get("currency") != "USD":
        raise LifecycleError("lifecycle supervisor currently requires a USD packet")
    ttl = _decimal(execution.get("hard_ttl_minutes"), label="packet TTL", positive=True) * 60
    wall = _decimal(
        execution.get("maximum_wall_minutes"), label="packet wall cap", positive=True
    ) * 60
    if ttl > wall:
        raise LifecycleError("packet TTL exceeds its wall-clock cap")
    deadman = _decimal(
        execution.get("deadman_timeout_minutes"),
        label="packet deadman timeout",
        positive=True,
    ) * 60
    budget_interval = _decimal(
        execution.get("budget_check_interval_minutes"),
        label="packet budget interval",
        positive=True,
    ) * 60
    budget_cap = _decimal(
        budget.get("proposed_maximum_cost"), label="packet budget cap", positive=True
    )
    return {
        "ttl": ttl,
        "deadman": deadman,
        "budget_interval": budget_interval,
        "budget_cap": budget_cap,
    }


@dataclass(frozen=True)
class DeploymentLifecycleBinding:
    campaign_id: str
    ledger_path: Path
    deployment_manifest_sha256: str
    lifecycle_launch_sha256: str
    campaign_root_sha256: str
    pair_declaration_sha256: str
    runtime_binding_sha256: str
    active_security_manifest_sha256: str
    launch_id: str
    launch_issued_at: str
    launch_expires_at: str
    arm_id: str
    guest_workload_argv_sha256: str
    launch_nonce_sha256: str
    heartbeat_public_key_path: Path | None
    resource_identity: Mapping[str, str]
    resource_identity_sha256: str


def _require_ed25519_public_key(path: str | Path) -> str:
    source = Path(path)
    _regular_file(source, label="lifecycle controller public key")
    try:
        result = subprocess.run(
            ["openssl", "pkey", "-pubin", "-in", str(source),
             "-text_pub", "-noout"],
            capture_output=True, check=False, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise LifecycleError("cannot inspect lifecycle controller public key") from exc
    first = result.stdout.splitlines()[0] if result.stdout else b""
    if result.returncode != 0 or first.strip() != b"ED25519 Public-Key:":
        raise LifecycleError("lifecycle controller public key must be Ed25519")
    return _sha256_file(source)


def _require_ed25519_keypair(private_key_path: str | Path,
                             public_key_path: str | Path) -> None:
    private = Path(private_key_path)
    public = Path(public_key_path)
    _regular_file(private, label="lifecycle controller private key")
    expected_public_sha = _require_ed25519_public_key(public)
    if stat.S_IMODE(private.stat().st_mode) & 0o077:
        raise LifecycleError(
            "lifecycle controller private key must not be group/world accessible")
    with tempfile.TemporaryDirectory(prefix="p4-cc-lifecycle-key-") as temporary:
        derived = Path(temporary) / "derived-public.pem"
        try:
            result = subprocess.run(
                ["openssl", "pkey", "-in", str(private), "-pubout", "-out", str(derived)],
                capture_output=True, check=False, timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise LifecycleError("cannot validate lifecycle controller keypair") from exc
        if result.returncode != 0 or not derived.is_file() \
                or _sha256_file(derived) != expected_public_sha:
            raise LifecycleError("lifecycle controller keys do not form an Ed25519 pair")


def create_signed_lifecycle_heartbeat(
    path: str | Path, *, launch_id: str, launch_nonce_sha256: str,
    protection_mode: str, instance_id: str,
    guest_workload_argv_sha256: str, sequence: int,
    previous_heartbeat_sha256: str, stage: str, step_id: int | None,
    private_key_path: str | Path, public_key_path: str | Path,
    emitted_at: str | None = None,
) -> Path:
    """Atomically emit one guest-signed, launch-bound heartbeat.

    The private key is a per-arm guest/journal key.  It is never required by
    the external controller; the signed launch envelope binds only its public
    key.  The caller must feed the prior envelope digest into the next event.
    """
    try:
        uuid.UUID(launch_id)
    except (ValueError, AttributeError) as exc:
        raise LifecycleError("heartbeat launch ID is invalid") from exc
    if protection_mode not in {"cc-on", "cc-off"}:
        raise LifecycleError("heartbeat protection mode is invalid")
    if (not isinstance(instance_id, str) or not instance_id
            or type(sequence) is not int or sequence < 0
            or not _is_sha256(launch_nonce_sha256)
            or not _is_sha256(guest_workload_argv_sha256)
            or not _is_sha256(previous_heartbeat_sha256)
            or (sequence == 0 and previous_heartbeat_sha256 != ZERO_HASH)):
        raise LifecycleError("heartbeat launch/sequence binding is invalid")
    if (not isinstance(stage, str) or not stage or len(stage) > 128
            or any(character not in
                   "abcdefghijklmnopqrstuvwxyz0123456789-_" for character in stage)
            or (step_id is not None
                and (type(step_id) is not int or step_id < 0))):
        raise LifecycleError("heartbeat stage is invalid")
    timestamp = emitted_at or _utc_now()
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise LifecycleError("heartbeat timestamp is invalid") from exc
    if not timestamp.endswith("Z") or parsed.tzinfo is None:
        raise LifecycleError("heartbeat timestamp is invalid")
    _require_ed25519_keypair(private_key_path, public_key_path)
    payload = {
        "schema": SIGNED_HEARTBEAT_PAYLOAD_SCHEMA,
        "launch_id": launch_id,
        "launch_nonce_sha256": launch_nonce_sha256,
        "protection_mode": protection_mode,
        "instance_id": instance_id,
        "guest_workload_argv_sha256": guest_workload_argv_sha256,
        "sequence": sequence,
        "previous_heartbeat_sha256": previous_heartbeat_sha256,
        "emitted_at": timestamp,
        "stage": stage,
        "step_id": step_id,
    }
    encoded = _canonical_bytes(payload)
    with tempfile.TemporaryDirectory(prefix="p4-cc-heartbeat-sign-") as temporary:
        message_path = Path(temporary) / "heartbeat.bin"
        signature_path = Path(temporary) / "signature.bin"
        message_path.write_bytes(encoded)
        try:
            result = subprocess.run(
                ["openssl", "pkeyutl", "-sign", "-inkey",
                 str(private_key_path), "-rawin", "-in", str(message_path),
                 "-out", str(signature_path)],
                capture_output=True, check=False, timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise LifecycleError("heartbeat signing failed") from exc
        if result.returncode != 0 or not signature_path.is_file():
            raise LifecycleError("heartbeat signing failed")
        signature = signature_path.read_bytes()
    if len(signature) != 64:
        raise LifecycleError("heartbeat signature is not Ed25519")
    envelope = {
        "schema": SIGNED_HEARTBEAT_SCHEMA,
        "payload": payload,
        "payload_sha256": _sha256_bytes(encoded),
        "public_key_sha256": _sha256_file(Path(public_key_path)),
        "signature_algorithm": "Ed25519",
        "signature_base64": base64.b64encode(signature).decode("ascii"),
    }
    target = Path(path)
    if target.exists() and target.is_symlink():
        raise LifecycleError("heartbeat target cannot be a symlink")
    _atomic_private_json(target, envelope)
    return target


def verify_signed_lifecycle_heartbeat(
    path: str | Path, *, public_key_path: str | Path,
    launch_id: str, launch_nonce_sha256: str, protection_mode: str,
    instance_id: str, guest_workload_argv_sha256: str,
    expected_sequence: int, expected_previous_sha256: str,
    issued_at: str, expires_at: str,
) -> dict[str, Any]:
    """Verify one exact heartbeat and its place in the per-launch chain."""
    source = Path(path).resolve()
    envelope = _load_json_object(source, label="signed lifecycle heartbeat")
    public_sha256 = _require_ed25519_public_key(public_key_path)
    expected_envelope = {
        "schema", "payload", "payload_sha256", "public_key_sha256",
        "signature_algorithm", "signature_base64",
    }
    payload = envelope.get("payload")
    expected_payload = {
        "schema", "launch_id", "launch_nonce_sha256", "protection_mode",
        "instance_id", "guest_workload_argv_sha256", "sequence",
        "previous_heartbeat_sha256", "emitted_at", "stage", "step_id",
    }
    if (set(envelope) != expected_envelope
            or envelope.get("schema") != SIGNED_HEARTBEAT_SCHEMA
            or envelope.get("public_key_sha256") != public_sha256
            or envelope.get("signature_algorithm") != "Ed25519"
            or not isinstance(payload, dict) or set(payload) != expected_payload
            or payload.get("schema") != SIGNED_HEARTBEAT_PAYLOAD_SCHEMA):
        raise LifecycleError("signed heartbeat fields differ")
    exact = {
        "launch_id": launch_id,
        "launch_nonce_sha256": launch_nonce_sha256,
        "protection_mode": protection_mode,
        "instance_id": instance_id,
        "guest_workload_argv_sha256": guest_workload_argv_sha256,
        "sequence": expected_sequence,
        "previous_heartbeat_sha256": expected_previous_sha256,
    }
    if any(payload.get(key) != value for key, value in exact.items()):
        raise LifecycleError("signed heartbeat launch or chain binding differs")
    if (not isinstance(payload.get("stage"), str) or not payload["stage"]
            or len(payload["stage"]) > 128
            or any(character not in
                   "abcdefghijklmnopqrstuvwxyz0123456789-_"
                   for character in payload["stage"])
            or (payload.get("step_id") is not None
                and (type(payload["step_id"]) is not int
                     or payload["step_id"] < 0))):
        raise LifecycleError("signed heartbeat stage is invalid")
    try:
        emitted = datetime.fromisoformat(
            payload["emitted_at"].replace("Z", "+00:00"))
        issued = datetime.fromisoformat(issued_at.replace("Z", "+00:00"))
        expires = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
    except (KeyError, AttributeError, ValueError) as exc:
        raise LifecycleError("signed heartbeat timestamp is invalid") from exc
    if (not payload["emitted_at"].endswith("Z")
            or emitted < issued - timedelta(seconds=30)
            or emitted > expires):
        raise LifecycleError("signed heartbeat is outside the launch window")
    encoded = _canonical_bytes(payload)
    if envelope.get("payload_sha256") != _sha256_bytes(encoded):
        raise LifecycleError("signed heartbeat payload digest differs")
    try:
        signature = base64.b64decode(
            envelope.get("signature_base64", ""), validate=True)
    except (TypeError, ValueError, binascii.Error) as exc:
        raise LifecycleError("heartbeat signature is not canonical base64") from exc
    if len(signature) != 64:
        raise LifecycleError("heartbeat Ed25519 signature must be 64 bytes")
    with tempfile.TemporaryDirectory(prefix="p4-cc-heartbeat-verify-") as temporary:
        message_path = Path(temporary) / "heartbeat.bin"
        signature_path = Path(temporary) / "signature.bin"
        message_path.write_bytes(encoded)
        signature_path.write_bytes(signature)
        try:
            result = subprocess.run(
                ["openssl", "pkeyutl", "-verify", "-pubin", "-inkey",
                 str(public_key_path), "-rawin", "-in", str(message_path),
                 "-sigfile", str(signature_path)],
                capture_output=True, check=False, timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise LifecycleError("heartbeat signature verification failed") from exc
    if result.returncode != 0:
        raise LifecycleError("heartbeat signature verification failed")
    return {
        "sequence": payload["sequence"],
        "stage": payload["stage"],
        "step_id": payload["step_id"],
        "envelope_sha256": _sha256_file(source),
        "emitted_at": payload["emitted_at"],
    }


def create_lifecycle_launch_envelope(
    path: str | Path, *, campaign_id: str, campaign_ledger_path: str | Path,
    packet_sha256: str, deployment_manifest_sha256: str,
    runtime_binding_sha256: str, active_security_manifest_sha256: str,
    pair_declaration_sha256: str, comparator_strategy: str,
    campaign_root_sha256: str, active_transition_sha256: str,
    predecessor_kind: str, predecessor_sha256: str,
    launch_id: str, issued_at: str, expires_at: str,
    arm: Mapping[str, Any],
    private_key_path: str | Path, public_key_path: str | Path,
) -> Path:
    """Create the signed, post-active-security launch envelope.

    This object is deliberately separate from the static deployment contract.
    It is created only after arrival, admission, the runtime binding, and the
    active-security manifest exist, which keeps the complete hash graph
    acyclic.  The signing key belongs to the external control plane; it must
    never be copied into the guest deployment bundle.
    """
    if (not isinstance(campaign_id, str) or not 3 <= len(campaign_id) <= 128
            or any(character not in
                   "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
                   for character in campaign_id)):
        raise LifecycleError("lifecycle campaign ID is invalid")
    for value, label in (
        (packet_sha256, "packet"),
        (deployment_manifest_sha256, "deployment manifest"),
        (runtime_binding_sha256, "runtime binding"),
        (active_security_manifest_sha256, "active-security manifest"),
        (pair_declaration_sha256, "pair declaration"),
        (campaign_root_sha256, "campaign root"),
        (active_transition_sha256, "active transition"),
        (predecessor_sha256, "predecessor"),
    ):
        if not _is_sha256(value):
            raise LifecycleError(f"lifecycle {label} digest is invalid")
    if comparator_strategy != "matched-separate-instance":
        raise LifecycleError(
            "lifecycle v1 requires matched-separate-instance; safe shared-resource "
            "deferred teardown is not implemented")
    try:
        uuid.UUID(launch_id)
    except (ValueError, AttributeError) as exc:
        raise LifecycleError("lifecycle launch ID is invalid") from exc
    try:
        issued = datetime.fromisoformat(issued_at.replace("Z", "+00:00"))
        expires = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise LifecycleError("lifecycle launch validity window is invalid") from exc
    if (not issued_at.endswith("Z") or not expires_at.endswith("Z")
            or issued.tzinfo is None or expires.tzinfo is None or expires <= issued
            or expires - issued > timedelta(hours=4)):
        raise LifecycleError("lifecycle launch validity window is invalid")
    expected_resource_keys = {
        "provider", "candidate_id", "sku", "region", "zone", "instance_id",
    }
    expected_arm_keys = {
        "protection_mode", "arrival_record_sha256", "authorization_kind",
        "authorization_sha256", "pair_id", "pair_manifest_sha256",
        "resource_identity", "controller_remote_launch_argv_sha256",
        "guest_workload_argv_sha256", "teardown_argv_sha256",
        "absence_verifier_argv_sha256", "launch_nonce_sha256",
        "heartbeat_public_key_sha256",
    }
    if not isinstance(arm, Mapping) or set(arm) != expected_arm_keys:
        raise LifecycleError("lifecycle arm binding fields differ")
    arm_id = arm.get("protection_mode")
    if arm_id not in {"cc-off", "cc-on"}:
        raise LifecycleError("lifecycle arm protection mode differs")
    expected_kind = "cc-on-admission" if arm_id == "cc-on" \
        else "paired-cc-on-admission-reference"
    if arm.get("authorization_kind") != expected_kind:
        raise LifecycleError(f"lifecycle {arm_id} authorization kind differs")
    expected_predecessor = (
        "active-transition" if arm_id == "cc-on" else "cc-on-lifecycle-receipt")
    if predecessor_kind != expected_predecessor:
        raise LifecycleError(f"lifecycle {arm_id} predecessor kind differs")
    if arm_id == "cc-on" and predecessor_sha256 != active_transition_sha256:
        raise LifecycleError("CC-on lifecycle predecessor is not the active transition")
    for key in (
        "arrival_record_sha256", "authorization_sha256",
        "pair_manifest_sha256", "controller_remote_launch_argv_sha256",
        "guest_workload_argv_sha256", "teardown_argv_sha256",
        "absence_verifier_argv_sha256", "launch_nonce_sha256",
        "heartbeat_public_key_sha256",
    ):
        if not _is_sha256(arm.get(key)):
            raise LifecycleError(f"lifecycle {arm_id} {key} is invalid")
    pair_id = arm.get("pair_id")
    if not isinstance(pair_id, str) or not pair_id.strip():
        raise LifecycleError(f"lifecycle {arm_id} pair ID is invalid")
    resource = arm.get("resource_identity")
    if (not isinstance(resource, Mapping) or set(resource) != expected_resource_keys
            or any(not isinstance(value, str) or not value.strip()
                   for value in resource.values())):
        raise LifecycleError(f"lifecycle {arm_id} resource identity is invalid")
    normalized_arm = {
        **dict(arm),
        "resource_identity": dict(resource),
        "resource_identity_sha256": _sha256_bytes(_canonical_bytes(resource)),
    }
    if normalized_arm["pair_manifest_sha256"] != pair_declaration_sha256:
        raise LifecycleError("lifecycle arm pair-declaration binding differs")
    _require_ed25519_keypair(private_key_path, public_key_path)
    public_key_sha256 = _sha256_file(Path(public_key_path))
    payload = {
        "schema": LIFECYCLE_LAUNCH_PAYLOAD_SCHEMA,
        "scientific_evidence": False,
        "campaign_id": campaign_id,
        "campaign_ledger_path": str(campaign_ledger_path),
        "launch_id": launch_id,
        "issued_at": issued_at,
        "expires_at": expires_at,
        "controller_location": "external-control-plane",
        "workload_execution_kind": "external-remote-launch-adapter",
        "comparator_strategy": comparator_strategy,
        "pair_declaration_sha256": pair_declaration_sha256,
        "campaign_root_sha256": campaign_root_sha256,
        "active_transition_sha256": active_transition_sha256,
        "predecessor": {
            "kind": predecessor_kind,
            "sha256": predecessor_sha256,
        },
        "hardware_packet_sha256": packet_sha256,
        "deployment_manifest_sha256": deployment_manifest_sha256,
        "runtime_binding_sha256": runtime_binding_sha256,
        "active_security_manifest_sha256": active_security_manifest_sha256,
        "required_stage_order": [
            "arrival-admission", "active-security", "same-checkpoint-correctness",
            "paired-cc-on-off", "post-mitigation-remeasure",
            "evidence-seal-and-teardown",
        ],
        "arm": normalized_arm,
    }
    encoded = _canonical_bytes(payload)
    with tempfile.TemporaryDirectory(prefix="p4-cc-lifecycle-sign-") as temporary:
        payload_path = Path(temporary) / "payload.bin"
        signature_path = Path(temporary) / "signature.bin"
        payload_path.write_bytes(encoded)
        try:
            result = subprocess.run(
                ["openssl", "pkeyutl", "-sign", "-inkey", str(private_key_path),
                 "-rawin", "-in", str(payload_path), "-out", str(signature_path)],
                capture_output=True, check=False, timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise LifecycleError("lifecycle launch signing failed") from exc
        if result.returncode != 0 or not signature_path.is_file():
            raise LifecycleError("lifecycle launch signing failed")
        signature = signature_path.read_bytes()
    envelope = {
        "schema": LIFECYCLE_LAUNCH_SCHEMA,
        "payload": payload,
        "payload_sha256": _sha256_bytes(encoded),
        "public_key_sha256": public_key_sha256,
        "signature_algorithm": "Ed25519",
        "signature_base64": base64.b64encode(signature).decode("ascii"),
    }
    return _write_json_exclusive(Path(path), envelope)


def verify_lifecycle_launch_envelope(
    path: str | Path, public_key_path: str | Path, *,
    expected_public_key_sha256: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Verify the signed launch envelope and return its exact payload."""
    public_sha = _require_ed25519_public_key(public_key_path)
    if (not _is_sha256(expected_public_key_sha256)
            or public_sha != expected_public_key_sha256):
        raise LifecycleError("lifecycle controller public-key binding differs")
    envelope = _load_json_object(Path(path), label="lifecycle launch envelope")
    expected_keys = {
        "schema", "payload", "payload_sha256", "public_key_sha256",
        "signature_algorithm", "signature_base64",
    }
    if (set(envelope) != expected_keys
            or envelope.get("schema") != LIFECYCLE_LAUNCH_SCHEMA
            or envelope.get("public_key_sha256") != public_sha
            or envelope.get("signature_algorithm") != "Ed25519"):
        raise LifecycleError("lifecycle launch envelope fields differ")
    payload = envelope.get("payload")
    expected_payload_keys = {
        "schema", "scientific_evidence", "campaign_id", "campaign_ledger_path",
        "launch_id", "issued_at", "expires_at", "controller_location",
        "workload_execution_kind", "comparator_strategy",
        "pair_declaration_sha256", "campaign_root_sha256",
        "active_transition_sha256", "predecessor",
        "required_stage_order", "arm",
        "hardware_packet_sha256", "deployment_manifest_sha256",
        "runtime_binding_sha256", "active_security_manifest_sha256",
    }
    if (not isinstance(payload, dict) or set(payload) != expected_payload_keys
            or payload.get("schema") != LIFECYCLE_LAUNCH_PAYLOAD_SCHEMA
            or payload.get("scientific_evidence") is not False
            or payload.get("controller_location") != "external-control-plane"
            or payload.get("workload_execution_kind")
            != "external-remote-launch-adapter"
            or payload.get("comparator_strategy") != "matched-separate-instance"):
        raise LifecycleError("lifecycle launch payload fields differ")
    expected_stages = [
        "arrival-admission", "active-security", "same-checkpoint-correctness",
        "paired-cc-on-off", "post-mitigation-remeasure",
        "evidence-seal-and-teardown",
    ]
    if payload.get("required_stage_order") != expected_stages:
        raise LifecycleError("lifecycle launch stage order differs")
    try:
        uuid.UUID(payload.get("launch_id"))
        issued = datetime.fromisoformat(payload["issued_at"].replace("Z", "+00:00"))
        expires = datetime.fromisoformat(payload["expires_at"].replace("Z", "+00:00"))
    except (TypeError, ValueError, AttributeError) as exc:
        raise LifecycleError("lifecycle launch validity fields are invalid") from exc
    if (not payload["issued_at"].endswith("Z") or not payload["expires_at"].endswith("Z")
            or issued.tzinfo is None or expires.tzinfo is None or expires <= issued
            or expires - issued > timedelta(hours=8)):
        raise LifecycleError("lifecycle launch validity fields are invalid")
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if current < issued - timedelta(seconds=30) or current > expires:
        raise LifecycleError("lifecycle launch is not currently valid")
    for key in (
        "pair_declaration_sha256", "hardware_packet_sha256",
        "deployment_manifest_sha256", "runtime_binding_sha256",
        "active_security_manifest_sha256", "campaign_root_sha256",
        "active_transition_sha256",
    ):
        if not _is_sha256(payload.get(key)):
            raise LifecycleError(f"lifecycle launch {key} is invalid")
    arm = payload.get("arm")
    expected_arm_keys = {
        "protection_mode", "arrival_record_sha256", "authorization_kind",
        "authorization_sha256", "pair_id", "pair_manifest_sha256",
        "resource_identity", "resource_identity_sha256",
        "controller_remote_launch_argv_sha256", "guest_workload_argv_sha256",
        "teardown_argv_sha256", "absence_verifier_argv_sha256",
        "launch_nonce_sha256", "heartbeat_public_key_sha256",
    }
    arm_id = arm.get("protection_mode") if isinstance(arm, dict) else None
    if (not isinstance(arm, dict) or set(arm) != expected_arm_keys
            or arm_id not in {"cc-off", "cc-on"}):
        raise LifecycleError("lifecycle launch arm fields differ")
    expected_kind = "cc-on-admission" if arm_id == "cc-on" \
        else "paired-cc-on-admission-reference"
    if arm.get("authorization_kind") != expected_kind:
        raise LifecycleError("lifecycle launch arm authorization differs")
    resource = arm.get("resource_identity")
    expected_resource_keys = {
        "provider", "candidate_id", "sku", "region", "zone", "instance_id",
    }
    if (not isinstance(resource, dict) or set(resource) != expected_resource_keys
            or any(not isinstance(value, str) or not value.strip()
                   for value in resource.values())
            or arm.get("resource_identity_sha256")
            != _sha256_bytes(_canonical_bytes(resource))):
        raise LifecycleError("lifecycle launch arm resource differs")
    if not isinstance(arm.get("pair_id"), str) or not arm["pair_id"].strip():
        raise LifecycleError("lifecycle launch arm pair ID is invalid")
    for key in expected_arm_keys - {
        "protection_mode", "authorization_kind", "pair_id", "resource_identity",
    }:
        if not _is_sha256(arm.get(key)):
            raise LifecycleError(f"lifecycle launch arm {key} is invalid")
    if arm.get("pair_manifest_sha256") != payload["pair_declaration_sha256"]:
        raise LifecycleError("lifecycle launch pair declaration differs")
    predecessor = payload.get("predecessor")
    expected_predecessor_kind = (
        "active-transition" if arm_id == "cc-on" else "cc-on-lifecycle-receipt")
    if (not isinstance(predecessor, dict)
            or set(predecessor) != {"kind", "sha256"}
            or predecessor.get("kind") != expected_predecessor_kind
            or not _is_sha256(predecessor.get("sha256"))
            or (arm_id == "cc-on" and predecessor.get("sha256")
                != payload["active_transition_sha256"])):
        raise LifecycleError("lifecycle launch predecessor differs")
    encoded = _canonical_bytes(payload)
    if envelope.get("payload_sha256") != _sha256_bytes(encoded):
        raise LifecycleError("lifecycle launch payload digest differs")
    try:
        signature = base64.b64decode(envelope.get("signature_base64", ""), validate=True)
    except (TypeError, ValueError) as exc:
        raise LifecycleError("lifecycle launch signature is not canonical base64") from exc
    with tempfile.TemporaryDirectory(prefix="p4-cc-lifecycle-verify-") as temporary:
        payload_path = Path(temporary) / "payload.bin"
        signature_path = Path(temporary) / "signature.bin"
        payload_path.write_bytes(encoded)
        signature_path.write_bytes(signature)
        try:
            result = subprocess.run(
                ["openssl", "pkeyutl", "-verify", "-pubin", "-inkey",
                 str(public_key_path), "-rawin", "-in", str(payload_path),
                 "-sigfile", str(signature_path)],
                capture_output=True, check=False, timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise LifecycleError("lifecycle launch signature verification failed") from exc
    if result.returncode != 0:
        raise LifecycleError("lifecycle launch signature verification failed")
    return dict(payload)


def consume_launch_nonce(
    lifecycle_launch_path: str | Path, *,
    controller_public_key_path: str | Path,
    expected_controller_public_key_sha256: str,
    launch_nonce_path: str | Path,
    guest_workload_argv_path: str | Path,
    heartbeat_public_key_path: str | Path,
    protection_mode: str,
    instance_id: str,
) -> Path:
    """Atomically consume the one-shot launch nonce inside the target guest."""
    payload = verify_lifecycle_launch_envelope(
        lifecycle_launch_path, controller_public_key_path,
        expected_public_key_sha256=expected_controller_public_key_sha256)
    arm = payload.get("arm")
    if (not isinstance(arm, Mapping)
            or arm.get("protection_mode") != protection_mode
            or arm.get("resource_identity", {}).get("instance_id") != instance_id):
        raise LifecycleError("launch nonce arm/resource binding differs")
    nonce_path = Path(launch_nonce_path).resolve()
    nonce_metadata = _regular_file(nonce_path, label="one-shot launch nonce")
    if (nonce_metadata.st_size != 32
            or stat.S_IMODE(nonce_metadata.st_mode) & 0o077
            or _sha256_file(nonce_path) != arm.get("launch_nonce_sha256")):
        raise LifecycleError("one-shot launch nonce bytes differ")
    guest_workload = load_argv_json(guest_workload_argv_path)
    if guest_workload.sha256 != arm.get("guest_workload_argv_sha256"):
        raise LifecycleError("launch nonce guest-workload binding differs")
    heartbeat_key_sha256 = _require_ed25519_public_key(
        heartbeat_public_key_path)
    if heartbeat_key_sha256 != arm.get("heartbeat_public_key_sha256"):
        raise LifecycleError("launch nonce heartbeat-key binding differs")
    launch_path = Path(lifecycle_launch_path).resolve()
    consumption_path = nonce_path.with_name(nonce_path.name + ".consumed.json")
    record = {
        "schema": "p4.cc_launch_nonce_consumption.v1",
        "launch_id": payload["launch_id"],
        "lifecycle_launch_path": str(launch_path),
        "lifecycle_launch_sha256": _sha256_file(launch_path),
        "launch_nonce_sha256": arm["launch_nonce_sha256"],
        "protection_mode": protection_mode,
        "instance_id": instance_id,
        "guest_workload_argv_sha256": guest_workload.sha256,
        "heartbeat_public_key_sha256": heartbeat_key_sha256,
        "consumed_at": _utc_now(),
    }
    return _write_json_exclusive(consumption_path, record)


def validate_deployment_lifecycle_binding(
    *, packet: Mapping[str, Any], packet_sha256: str,
    deployment_manifest_path: str | Path, workload: ArgvBinding,
    teardown: ArgvBinding, absence_verifier: ArgvBinding,
    ledger_path: str | Path, arm_id: str, cwd: str | Path,
    lifecycle_launch_manifest_path: str | Path | None = None,
    runtime_binding_path: str | Path | None = None,
    active_security_manifest_path: str | Path | None = None,
    controller_public_key_path: str | Path | None = None,
    pair_declaration_path: str | Path | None = None,
    pair_auth_path: str | Path | None = None,
    pair_campaign_public_key_path: str | Path | None = None,
    arm_arrival_path: str | Path | None = None,
    guest_workload: ArgvBinding | None = None,
    launch_nonce_path: str | Path | None = None,
    heartbeat_public_key_path: str | Path | None = None,
    remote_launch_adapter_path: str | Path | None = None,
    bootstrap_lifecycle_path: str | Path | None = None,
    active_transition_path: str | Path | None = None,
    measurement_ready_state_path: str | Path | None = None,
    cc_on_receipt_path: str | Path | None = None,
    test_only_allow_legacy_local: bool = False,
) -> DeploymentLifecycleBinding:
    """Bind every executable adapter to one signed, post-active launch.

    The legacy in-deployment lifecycle object remains reachable only through
    the explicit test-only switch so existing synthetic process-control tests
    cannot accidentally become a production launch path.
    """
    if not _is_sha256(packet_sha256):
        raise LifecycleError("packet digest is invalid")
    manifest_path = Path(deployment_manifest_path).resolve()
    deployment = _load_json_object(manifest_path, label="deployment manifest")
    deployment_sha256 = _sha256_file(manifest_path)
    order = packet.get("order_parameters")
    if not isinstance(order, Mapping):
        raise LifecycleError("packet order parameters are missing")
    workdir = Path(cwd).resolve()
    actual_ledger = Path(ledger_path)
    if not actual_ledger.is_absolute():
        actual_ledger = workdir / actual_ledger
    actual_ledger = actual_ledger.resolve(strict=False)

    if test_only_allow_legacy_local:
        runtime = deployment.get("runtime_binding")
        lifecycle = deployment.get("lifecycle")
        if not isinstance(runtime, Mapping) or not isinstance(lifecycle, Mapping):
            raise LifecycleError("legacy test deployment lacks runtime or lifecycle binding")
        expected_lifecycle_keys = {
            "schema", "campaign_id", "campaign_ledger_path", "resource_identity",
            "resource_identity_sha256", "workload_argv_sha256",
            "teardown_argv_sha256", "absence_verifier_argv_sha256",
        }
        if set(lifecycle) != expected_lifecycle_keys \
                or lifecycle.get("schema") != DEPLOYMENT_LIFECYCLE_SCHEMA:
            raise LifecycleError("legacy test deployment lifecycle fields differ")
        if runtime.get("hardware_packet_sha256") != packet_sha256:
            raise LifecycleError("legacy test deployment packet binding differs")
        expected_resource = {
            "provider": order.get("provider"),
            "candidate_id": order.get("candidate_id"),
            "sku": order.get("sku"),
            "region": order.get("region"),
            "zone": order.get("zone"),
            "instance_id": runtime.get("instance_id"),
        }
        resource = lifecycle.get("resource_identity")
        resource_sha256 = _sha256_bytes(_canonical_bytes(resource))
        if (resource != expected_resource
                or lifecycle.get("resource_identity_sha256") != resource_sha256):
            raise LifecycleError("legacy test resource identity differs")
        workload_digests = lifecycle.get("workload_argv_sha256")
        verifier_digests = lifecycle.get("absence_verifier_argv_sha256")
        if (not isinstance(workload_digests, Mapping)
                or set(workload_digests) != {"cc-off", "cc-on"}
                or workload_digests.get(arm_id) != workload.sha256
                or lifecycle.get("teardown_argv_sha256") != teardown.sha256
                or not isinstance(verifier_digests, Mapping)
                or set(verifier_digests) != {"cc-off", "cc-on"}
                or verifier_digests.get(arm_id) != absence_verifier.sha256):
            raise LifecycleError("legacy test lifecycle argv binding differs")
        declared_ledger = Path(str(lifecycle.get("campaign_ledger_path")))
        campaign_id = lifecycle.get("campaign_id")
        lifecycle_launch_sha256 = deployment_sha256
        campaign_root_sha256 = deployment_sha256
        pair_declaration_sha256 = deployment_sha256
        runtime_binding_sha256 = ZERO_HASH
        active_security_manifest_sha256 = ZERO_HASH
        launch_id = "00000000-0000-0000-0000-000000000000"
        launch_issued_at = "1970-01-01T00:00:00Z"
        launch_expires_at = "9998-12-31T23:59:59Z"
        guest_workload_argv_sha256 = ZERO_HASH
        launch_nonce_sha256 = ZERO_HASH
        heartbeat_key_path = None
    else:
        required = {
            "lifecycle launch manifest": lifecycle_launch_manifest_path,
            "runtime binding": runtime_binding_path,
            "active-security manifest": active_security_manifest_path,
            "controller public key": controller_public_key_path,
            "pair declaration": pair_declaration_path,
            "pair authentication": pair_auth_path,
            "pair campaign public key": pair_campaign_public_key_path,
            "arm arrival record": arm_arrival_path,
            "guest workload argv": guest_workload,
            "launch nonce": launch_nonce_path,
            "heartbeat public key": heartbeat_public_key_path,
            "remote-launch adapter": remote_launch_adapter_path,
            "bootstrap lifecycle": bootstrap_lifecycle_path,
            "active-security transition": active_transition_path,
        }
        if arm_id == "cc-on":
            required["measurement-ready state"] = measurement_ready_state_path
        elif arm_id == "cc-off":
            required["CC-on lifecycle receipt"] = cc_on_receipt_path
        missing = sorted(label for label, value in required.items() if value is None)
        if missing:
            raise LifecycleError(
                "production lifecycle inputs are missing: " + ", ".join(missing))
        from .cc_active_security import (
            ActiveSecurityError,
            validate_active_runtime_binding,
            validate_active_security_manifest,
            validate_deployment_manifest,
        )
        try:
            validate_deployment_manifest(
                deployment, deployment_sha256, deployment.get("profile"),
                deployment.get("adapter", {}).get("command_argv", []),
            )
            runtime_path = Path(runtime_binding_path).resolve()
            runtime = _load_json_object(runtime_path, label="active runtime binding")
            runtime_binding_sha256 = _sha256_file(runtime_path)
            validate_active_runtime_binding(runtime, deployment_sha256)
            active_path = Path(active_security_manifest_path).resolve()
            active = _load_json_object(active_path, label="active-security manifest")
            active_security_manifest_sha256 = _sha256_file(active_path)
            validate_active_security_manifest(
                active, active_path.parent,
                expected_manifest_sha256=active_security_manifest_sha256,
                manifest_path=active_path,
            )
        except (ActiveSecurityError, OSError, ValueError) as exc:
            raise LifecycleError("deployment/runtime/active launch chain is invalid") from exc
        active_inputs = active.get("inputs", {})
        if (active_inputs.get("deployment_manifest_sha256") != deployment_sha256
                or active_inputs.get("runtime_binding_file_sha256")
                != runtime_binding_sha256
                or active_inputs.get("runtime_binding") != runtime
                or runtime.get("hardware_packet_sha256") != packet_sha256):
            raise LifecycleError("active launch chain byte bindings differ")
        packet_bindings = packet.get("bindings")
        if not isinstance(packet_bindings, Mapping):
            raise LifecycleError("packet lifecycle controller binding is missing")
        controller_key_sha = packet_bindings.get(
            "lifecycle_controller_public_key_sha256")
        from .cc_campaign import CampaignSealError, validate_pair_declaration
        try:
            pair = validate_pair_declaration(
                pair_declaration_path, pair_auth_path,
                pair_campaign_public_key_path)
        except (CampaignSealError, OSError, ValueError) as exc:
            raise LifecycleError("signed lifecycle pair declaration is invalid") from exc
        pair_path = Path(pair_declaration_path).resolve()
        pair_declaration_sha256 = _sha256_file(pair_path)
        pair_keys = pair.get("public_keys", {})
        if (pair_keys.get("lifecycle_controller", {}).get("sha256")
                != controller_key_sha
                or pair_keys.get("campaign", {}).get("sha256")
                != packet_bindings.get("campaign_public_key_sha256")):
            raise LifecycleError("lifecycle pair public-key trust roots differ")
        from .hardware_preorder import ROOT
        for path_key, digest_key, supplied in (
            ("lifecycle_controller_public_key_path",
             "lifecycle_controller_public_key_sha256",
             controller_public_key_path),
            ("campaign_public_key_path", "campaign_public_key_sha256",
             pair_campaign_public_key_path),
        ):
            declared = Path(str(packet_bindings.get(path_key)))
            if not declared.is_absolute():
                declared = ROOT / declared
            supplied_path = Path(supplied).resolve()
            if (declared.resolve() != supplied_path
                    or packet_bindings.get(digest_key)
                    != _sha256_file(supplied_path)):
                raise LifecycleError("packet lifecycle public-key binding differs")

        adapter_path = Path(remote_launch_adapter_path).resolve()
        _regular_file(adapter_path, label="remote-launch adapter")
        declared_adapter = Path(str(
            packet_bindings.get("remote_launch_adapter_path")))
        if not declared_adapter.is_absolute():
            declared_adapter = ROOT / declared_adapter
        if (declared_adapter.resolve() != adapter_path
                or packet_bindings.get("remote_launch_adapter_sha256")
                != _sha256_file(adapter_path)):
            raise LifecycleError("packet remote-launch adapter binding differs")
        _require_adapter_command(
            workload, adapter_path=adapter_path, action="launch", cwd=workdir)
        _require_adapter_command(
            teardown, adapter_path=adapter_path, action="delete", cwd=workdir)
        _require_adapter_command(
            absence_verifier, adapter_path=adapter_path,
            action="verify-absence", cwd=workdir)

        nonce_path = Path(launch_nonce_path).resolve()
        nonce_metadata = _regular_file(nonce_path, label="one-shot launch nonce")
        if nonce_metadata.st_size != 32 \
                or stat.S_IMODE(nonce_metadata.st_mode) & 0o077:
            raise LifecycleError(
                "one-shot launch nonce must be 32 private raw bytes")
        nonce_sha256 = _sha256_file(nonce_path)
        heartbeat_key_sha256 = _require_ed25519_public_key(
            heartbeat_public_key_path)
        heartbeat_key_path = Path(heartbeat_public_key_path).resolve()
        launch_path = Path(lifecycle_launch_manifest_path).resolve()
        payload = verify_lifecycle_launch_envelope(
            launch_path, controller_public_key_path,
            expected_public_key_sha256=controller_key_sha,
        )
        lifecycle_launch_sha256 = _sha256_file(launch_path)
        campaign_root_sha256 = payload.get("campaign_root_sha256")
        if not _is_sha256(campaign_root_sha256):
            raise LifecycleError("signed lifecycle campaign root is invalid")
        launch_id = payload["launch_id"]
        launch_issued_at = payload["issued_at"]
        launch_expires_at = payload["expires_at"]
        if payload.get("pair_declaration_sha256") != pair_declaration_sha256:
            raise LifecycleError("signed lifecycle pair declaration differs")
        pairing_contract = pair.get("pairing_contract")
        if (not isinstance(pairing_contract, Mapping)
                or pairing_contract.get("strategy") != "matched-separate-instance"
                or payload.get("comparator_strategy")
                != pairing_contract.get("strategy")):
            raise LifecycleError("lifecycle comparator strategy is not safely supported")
        from .cc_admission import load_arrival_record
        try:
            arrival_path = Path(arm_arrival_path).resolve()
            arrival = load_arrival_record(arrival_path)
        except (OSError, ValueError) as exc:
            raise LifecycleError("lifecycle arm arrival is invalid") from exc
        deployment_identity = arrival.get("deployment", {})
        expected_resource = {
            "provider": deployment_identity.get("provider"),
            "candidate_id": deployment_identity.get("candidate_id"),
            "sku": deployment_identity.get("sku"),
            "region": deployment_identity.get("region"),
            "zone": deployment_identity.get("zone"),
            "instance_id": deployment_identity.get("instance_id"),
        }
        expected_contract = pairing_contract.get("arms", {}).get(arm_id)
        if not isinstance(expected_contract, Mapping):
            raise LifecycleError("pair lifecycle arm resource contract is missing")
        gpu = arrival.get("gpu")
        cpu = arrival.get("cpu")
        exact_resource_fields = {
            "provider": deployment_identity.get("provider"),
            "candidate_id": deployment_identity.get("candidate_id"),
            "sku": deployment_identity.get("sku"),
            "region": deployment_identity.get("region"),
            "zone": deployment_identity.get("zone"),
            "image_reference": deployment_identity.get("image_reference"),
            "gpu_name": gpu.get("name") if isinstance(gpu, Mapping) else None,
            "driver": gpu.get("driver") if isinstance(gpu, Mapping) else None,
            "cpu_model": cpu.get("model") if isinstance(cpu, Mapping) else None,
        }
        if any(exact_resource_fields.get(key) != expected_contract.get(key)
               for key in exact_resource_fields):
            raise LifecycleError("lifecycle resource differs from signed pair contract")
        if (arm_id == "cc-on"
                and (expected_resource["instance_id"] != runtime.get("instance_id")
                     or _sha256_file(arrival_path)
                     != runtime.get("arrival_record_sha256"))):
            raise LifecycleError("CC-on lifecycle resource differs from runtime")
        arm = payload.get("arm")
        if not isinstance(arm, Mapping):
            raise LifecycleError("signed lifecycle arm is missing")
        resource = arm.get("resource_identity")
        resource_sha256 = _sha256_bytes(_canonical_bytes(expected_resource))
        expected_bindings = {
            "hardware_packet_sha256": packet_sha256,
            "deployment_manifest_sha256": deployment_sha256,
            "runtime_binding_sha256": runtime_binding_sha256,
            "active_security_manifest_sha256": active_security_manifest_sha256,
        }
        if any(payload.get(key) != value for key, value in expected_bindings.items()):
            raise LifecycleError("signed lifecycle launch binding differs")
        expected_arm = {
            "protection_mode": arm_id,
            "arrival_record_sha256": _sha256_file(arrival_path),
            "authorization_kind": (
                "cc-on-admission" if arm_id == "cc-on"
                else "paired-cc-on-admission-reference"),
            "authorization_sha256": runtime.get("admission_sha256"),
            "pair_id": deployment_identity.get("pair_id"),
            "pair_manifest_sha256": pair_declaration_sha256,
            "resource_identity": expected_resource,
            "resource_identity_sha256": resource_sha256,
            "controller_remote_launch_argv_sha256": workload.sha256,
            "guest_workload_argv_sha256": guest_workload.sha256,
            "teardown_argv_sha256": teardown.sha256,
            "absence_verifier_argv_sha256": absence_verifier.sha256,
            "launch_nonce_sha256": nonce_sha256,
            "heartbeat_public_key_sha256": heartbeat_key_sha256,
        }
        if any(arm.get(key) != value for key, value in expected_arm.items()):
            raise LifecycleError("signed lifecycle per-arm argv binding differs")
        if deployment_identity.get("protection_mode") != arm_id:
            raise LifecycleError("arrival protection mode differs from lifecycle arm")
        if deployment_identity.get("pair_manifest_sha256") != pair_declaration_sha256:
            raise LifecycleError("arrival pair-declaration binding differs")
        guest_workload_argv_sha256 = guest_workload.sha256
        launch_nonce_sha256 = nonce_sha256
        from .cc_bootstrap_lifecycle import (
            BootstrapLifecycleError,
            require_measurement_ready,
            verify_bootstrap_envelope,
            verify_bootstrap_state,
        )
        bootstrap_path = Path(bootstrap_lifecycle_path).resolve()
        transition_path = Path(active_transition_path).resolve()
        bootstrap = verify_bootstrap_envelope(
            bootstrap_path, controller_public_key_path)
        transition = verify_bootstrap_state(
            transition_path, controller_public_key_path)
        if (payload.get("campaign_root_sha256") != _sha256_file(bootstrap_path)
                or payload.get("active_transition_sha256")
                != _sha256_file(transition_path)
                or bootstrap.get("campaign", {}).get("campaign_id")
                != pair.get("campaign_id")
                or bootstrap.get("campaign", {}).get("pair_id")
                != pair.get("pair_id")
                or transition.get("phase") != "active-security-complete"
                or transition.get("l0_sha256") != _sha256_file(bootstrap_path)
                or transition.get("active_security_manifest_sha256")
                != active_security_manifest_sha256):
            raise LifecycleError("lifecycle bootstrap transition chain differs")
        try:
            if arm_id == "cc-on":
                require_measurement_ready(
                    measurement_ready_state_path,
                    l0_path=bootstrap_path,
                    l1_path=launch_path,
                    active_transition_path=transition_path,
                    controller_public_key_path=controller_public_key_path,
                )
            else:
                receipt_path = Path(cc_on_receipt_path).resolve()
                receipt = _load_json_object(
                    receipt_path, label="CC-on lifecycle receipt")
                receipt_validation = validate_lifecycle_receipt(receipt_path)
                predecessor = payload.get("predecessor")
                if (predecessor != {
                        "kind": "cc-on-lifecycle-receipt",
                        "sha256": _sha256_file(receipt_path),
                    }
                        or receipt_validation.get("arm_id") != "cc-on"
                        or receipt_validation.get("exit_code") != 0
                        or receipt.get("campaign_root_sha256")
                        != payload.get("campaign_root_sha256")
                        or receipt.get("pair_declaration_sha256")
                        != pair_declaration_sha256
                        or receipt.get("packet_sha256") != packet_sha256
                        or receipt.get("deployment_manifest_sha256")
                        != deployment_sha256
                        or receipt.get("runtime_binding_sha256")
                        != runtime_binding_sha256
                        or receipt.get("active_security_manifest_sha256")
                        != active_security_manifest_sha256):
                    raise LifecycleError("CC-off predecessor receipt differs")
        except (BootstrapLifecycleError, OSError, ValueError) as exc:
            raise LifecycleError(
                "lifecycle controller authorization chain is invalid") from exc
        declared_ledger = Path(str(payload.get("campaign_ledger_path")))
        campaign_id = payload.get("campaign_id")

    instance_id = resource.get("instance_id") if isinstance(resource, Mapping) else None
    if (not isinstance(instance_id, str) or instance_id not in teardown.argv
            or instance_id not in absence_verifier.argv):
        raise LifecycleError("teardown and absence verifier must name the exact resource")
    if not declared_ledger.is_absolute():
        declared_ledger = workdir / declared_ledger
    declared_ledger = declared_ledger.resolve(strict=False)
    if declared_ledger != actual_ledger:
        raise LifecycleError("campaign ledger path differs from lifecycle launch")
    if (
        not isinstance(campaign_id, str)
        or not 3 <= len(campaign_id) <= 128
        or any(character not in
               "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
               for character in campaign_id)
    ):
        raise LifecycleError("deployment campaign ID is invalid")
    return DeploymentLifecycleBinding(
        campaign_id=campaign_id,
        ledger_path=actual_ledger,
        deployment_manifest_sha256=deployment_sha256,
        lifecycle_launch_sha256=lifecycle_launch_sha256,
        campaign_root_sha256=campaign_root_sha256,
        pair_declaration_sha256=pair_declaration_sha256,
        runtime_binding_sha256=runtime_binding_sha256,
        active_security_manifest_sha256=active_security_manifest_sha256,
        launch_id=launch_id,
        launch_issued_at=launch_issued_at,
        launch_expires_at=launch_expires_at,
        arm_id=arm_id,
        guest_workload_argv_sha256=guest_workload_argv_sha256,
        launch_nonce_sha256=launch_nonce_sha256,
        heartbeat_public_key_path=heartbeat_key_path,
        resource_identity=dict(resource),
        resource_identity_sha256=resource_sha256,
    )


class CampaignCostLedger(AbstractContextManager["CampaignCostLedger"]):
    """Exclusive campaign ledger held across one complete supervised arm.

    A separate stable lock inode protects atomic replacement of the JSON state.
    Leaving an ``active_run`` after host/process loss intentionally blocks reuse;
    it must be reconciled against provider billing rather than caller-reset.
    """

    def __init__(
        self, path: Path, *, campaign_id: str, packet_sha256: str,
        deployment_manifest_sha256: str, campaign_root_sha256: str,
        pair_declaration_sha256: str,
        budget_cap_usd: Decimal,
    ) -> None:
        self.path = path
        self.lock_path = path.with_name(path.name + ".lock")
        self.campaign_id = campaign_id
        self.packet_sha256 = packet_sha256
        self.deployment_manifest_sha256 = deployment_manifest_sha256
        self.campaign_root_sha256 = campaign_root_sha256
        self.pair_declaration_sha256 = pair_declaration_sha256
        self.budget_cap_usd = budget_cap_usd
        self.document: dict[str, Any] | None = None
        self._lock_descriptor: int | None = None

    @staticmethod
    def _body(document: Mapping[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in document.items() if key != "ledger_sha256"}

    def _seal(self, body: Mapping[str, Any]) -> dict[str, Any]:
        return {**body, "ledger_sha256": _sha256_bytes(_canonical_bytes(body))}

    def _validate(self, document: Mapping[str, Any]) -> dict[str, Any]:
        expected_keys = {
            "schema", "campaign_id", "packet_sha256",
            "deployment_manifest_sha256", "campaign_root_sha256",
            "pair_declaration_sha256", "budget_cap_usd",
            "accrued_cost_usd", "active_run", "entries", "ledger_sha256",
        }
        if set(document) != expected_keys or document.get("schema") != LEDGER_SCHEMA:
            raise LifecycleError("campaign ledger fields differ")
        body = self._body(document)
        if document.get("ledger_sha256") != _sha256_bytes(_canonical_bytes(body)):
            raise LifecycleError("campaign ledger digest mismatch")
        if (
            document.get("campaign_id") != self.campaign_id
            or document.get("packet_sha256") != self.packet_sha256
            or document.get("deployment_manifest_sha256")
            != self.deployment_manifest_sha256
            or document.get("campaign_root_sha256") != self.campaign_root_sha256
            or document.get("pair_declaration_sha256")
            != self.pair_declaration_sha256
            or _decimal(document.get("budget_cap_usd"), label="ledger budget cap")
            != self.budget_cap_usd
        ):
            raise LifecycleError("campaign ledger frozen binding differs")
        accrued = _decimal(document.get("accrued_cost_usd"), label="ledger accrued cost")
        entries = document.get("entries")
        if not isinstance(entries, list) or len(entries) > 16:
            raise LifecycleError("campaign ledger entries are invalid")
        if accrued > self.budget_cap_usd:
            raise LifecycleError("campaign ledger already exceeds the budget cap")
        if document.get("active_run") is not None:
            raise LifecycleError("campaign ledger has an unreconciled active run")
        arms = []
        previous_entry = ZERO_HASH
        entry_keys = {
            "sequence", "run_id", "arm_id", "rate_role", "finished_at",
            "total_campaign_cost_usd", "lifecycle_record_sha256",
            "lifecycle_launch_sha256", "absence_artifact_sha256", "exit_code",
            "previous_entry_sha256",
            "entry_sha256",
        }
        for sequence, entry in enumerate(entries):
            if not isinstance(entry, Mapping) or set(entry) != entry_keys:
                raise LifecycleError("campaign ledger entry fields differ")
            if entry.get("sequence") != sequence \
                    or entry.get("previous_entry_sha256") != previous_entry:
                raise LifecycleError("campaign ledger entry chain differs")
            entry_body = {
                key: value for key, value in entry.items() if key != "entry_sha256"
            }
            expected_entry = _sha256_bytes(_canonical_bytes(entry_body))
            if entry.get("entry_sha256") != expected_entry:
                raise LifecycleError("campaign ledger entry digest mismatch")
            previous_entry = expected_entry
            arms.append(entry.get("arm_id"))
        if len(arms) != len(set(arms)):
            raise LifecycleError("campaign ledger arm history is invalid")
        if arms != ["cc-on", "cc-off"][:len(arms)]:
            raise LifecycleError("campaign ledger arm order differs")
        return dict(document)

    def __enter__(self) -> "CampaignCostLedger":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_descriptor = os.open(
            self.lock_path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600,
        )
        try:
            fcntl.flock(self._lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as exc:
            os.close(self._lock_descriptor)
            self._lock_descriptor = None
            raise LifecycleError("campaign ledger is already locked by another run") from exc
        try:
            os.fchmod(self._lock_descriptor, 0o600)
            if self.path.exists():
                metadata = _regular_file(self.path, label="campaign cost ledger")
                if stat.S_IMODE(metadata.st_mode) & 0o077:
                    raise LifecycleError(
                        "campaign ledger must not be group/world accessible"
                    )
                self.document = self._validate(
                    _load_json_object(self.path, label="campaign cost ledger")
                )
            else:
                body = {
                    "schema": LEDGER_SCHEMA,
                    "campaign_id": self.campaign_id,
                    "packet_sha256": self.packet_sha256,
                    "deployment_manifest_sha256": self.deployment_manifest_sha256,
                    "campaign_root_sha256": self.campaign_root_sha256,
                    "pair_declaration_sha256": self.pair_declaration_sha256,
                    "budget_cap_usd": _money(self.budget_cap_usd),
                    "accrued_cost_usd": _money(Decimal(0)),
                    "active_run": None,
                    "entries": [],
                }
                self.document = self._seal(body)
                _atomic_private_json(self.path, self.document)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    @property
    def accrued_cost_usd(self) -> Decimal:
        if self.document is None:
            raise LifecycleError("campaign ledger is not open")
        return _decimal(self.document["accrued_cost_usd"], label="ledger accrued cost")

    @property
    def fixed_cost_already_charged(self) -> bool:
        return bool(self.document and self.document["entries"])

    def begin(
        self, *, run_id: str, arm_id: str, rate_role: str,
        lifecycle_launch_sha256: str,
    ) -> None:
        if self.document is None:
            raise LifecycleError("campaign ledger is not open")
        if arm_id not in {"cc-off", "cc-on"}:
            raise LifecycleError("campaign arm must be cc-off or cc-on")
        expected_role = "comparator" if arm_id == "cc-off" else "primary"
        if rate_role != expected_role:
            raise LifecycleError("campaign arm and quote role differ")
        if not _is_sha256(lifecycle_launch_sha256):
            raise LifecycleError("campaign arm lifecycle-launch digest is invalid")
        if any(entry.get("arm_id") == arm_id for entry in self.document["entries"]):
            raise LifecycleError("campaign arm has already been charged")
        if self.document["entries"]:
            previous = self.document["entries"][-1]
            if (previous.get("exit_code") != 0
                    or not _is_sha256(previous.get("absence_artifact_sha256"))):
                raise LifecycleError(
                    "campaign cannot continue after failed or unverified prior arm")
        expected_next = "cc-on" if not self.document["entries"] else "cc-off"
        if arm_id != expected_next:
            raise LifecycleError("campaign arms must execute sequentially cc-on then cc-off")
        body = self._body(self.document)
        body["active_run"] = {
            "run_id": run_id,
            "arm_id": arm_id,
            "rate_role": rate_role,
            "lifecycle_launch_sha256": lifecycle_launch_sha256,
            "started_at": _utc_now(),
            "prior_cost_usd": _money(self.accrued_cost_usd),
        }
        self.document = self._seal(body)
        _atomic_private_json(self.path, self.document)

    def finish(
        self, *, run_id: str, arm_id: str, rate_role: str,
        total_cost_usd: Decimal, lifecycle_record_sha256: str,
        absence_artifact_sha256: str | None, exit_code: int,
    ) -> dict[str, str]:
        if self.document is None:
            raise LifecycleError("campaign ledger is not open")
        active = self.document.get("active_run")
        if (not isinstance(active, Mapping) or active.get("run_id") != run_id
                or active.get("arm_id") != arm_id
                or active.get("rate_role") != rate_role
                or not _is_sha256(active.get("lifecycle_launch_sha256"))):
            raise LifecycleError("campaign ledger active-run binding differs")
        if total_cost_usd < self.accrued_cost_usd:
            raise LifecycleError("campaign cost cannot regress")
        body = self._body(self.document)
        body["active_run"] = None
        body["accrued_cost_usd"] = _money(total_cost_usd)
        previous_entry = (
            body["entries"][-1]["entry_sha256"] if body["entries"] else ZERO_HASH
        )
        entry_body = {
            "sequence": len(body["entries"]),
            "run_id": run_id,
            "arm_id": arm_id,
            "rate_role": rate_role,
            "finished_at": _utc_now(),
            "total_campaign_cost_usd": _money(total_cost_usd),
            "lifecycle_record_sha256": lifecycle_record_sha256,
            "lifecycle_launch_sha256": active["lifecycle_launch_sha256"],
            "absence_artifact_sha256": absence_artifact_sha256,
            "exit_code": exit_code,
            "previous_entry_sha256": previous_entry,
        }
        entry = {
            **entry_body,
            "entry_sha256": _sha256_bytes(_canonical_bytes(entry_body)),
        }
        body["entries"] = [*body["entries"], entry]
        self.document = self._seal(body)
        _atomic_private_json(self.path, self.document)
        return {
            "file_sha256": _sha256_file(self.path),
            "state_sha256": self.document["ledger_sha256"],
            "entry_sha256": entry["entry_sha256"],
        }

    def __exit__(self, *_exc: Any) -> None:
        if self._lock_descriptor is not None:
            try:
                fcntl.flock(self._lock_descriptor, fcntl.LOCK_UN)
            finally:
                os.close(self._lock_descriptor)
                self._lock_descriptor = None


@dataclass(frozen=True)
class LifecyclePolicy:
    """Fully resolved runtime policy bound to one pre-order packet."""

    packet_sha256: str
    rate_role: str
    ttl_seconds: Decimal
    deadman_seconds: Decimal
    budget_check_seconds: Decimal
    hourly_rate_usd: Decimal
    fixed_cost_usd: Decimal
    prior_cost_usd: Decimal
    budget_cap_usd: Decimal
    termination_grace_seconds: Decimal
    teardown_timeout_seconds: Decimal
    control_sha256: Mapping[str, str]

    def __post_init__(self) -> None:
        if (
            len(self.packet_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.packet_sha256)
        ):
            raise LifecycleError("packet digest is invalid")
        if self.rate_role not in {"primary", "comparator"}:
            raise LifecycleError("rate role must be primary or comparator")
        for label, value in (
            ("TTL", self.ttl_seconds),
            ("deadman", self.deadman_seconds),
            ("budget interval", self.budget_check_seconds),
            ("hourly rate", self.hourly_rate_usd),
            ("budget cap", self.budget_cap_usd),
            ("termination grace", self.termination_grace_seconds),
            ("teardown timeout", self.teardown_timeout_seconds),
        ):
            if not isinstance(value, Decimal) or not value.is_finite() or value <= 0:
                raise LifecycleError(f"{label} must be positive and finite")
        for label, value in (
            ("fixed cost", self.fixed_cost_usd),
            ("prior cost", self.prior_cost_usd),
        ):
            if not isinstance(value, Decimal) or not value.is_finite() or value < 0:
                raise LifecycleError(f"{label} must be nonnegative and finite")
        if set(self.control_sha256) != set(CONTROL_SPECS) or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in self.control_sha256.values()
        ):
            raise LifecycleError("all four lifecycle-control digests are required")


def policy_from_packet(
    packet: Mapping[str, Any],
    *,
    packet_sha256: str,
    rate_role: str,
    control_sha256: Mapping[str, str],
    prior_cost_usd: Any = 0,
    ttl_seconds: Any | None = None,
    deadman_seconds: Any | None = None,
    budget_check_seconds: Any | None = None,
    termination_grace_seconds: Any = 10,
    teardown_timeout_seconds: Any = 120,
) -> LifecyclePolicy:
    """Resolve a policy and reject any runtime bound weaker than its packet."""
    bounds = _packet_bounds(packet)
    requested_ttl = bounds["ttl"] if ttl_seconds is None else _seconds(
        ttl_seconds, label="runtime TTL"
    )
    requested_deadman = bounds["deadman"] if deadman_seconds is None else _seconds(
        deadman_seconds, label="runtime deadman"
    )
    requested_interval = (
        bounds["budget_interval"]
        if budget_check_seconds is None
        else _seconds(budget_check_seconds, label="runtime budget interval")
    )
    if requested_ttl > bounds["ttl"]:
        raise LifecycleError("runtime TTL exceeds the packet TTL")
    if requested_deadman > bounds["deadman"]:
        raise LifecycleError("runtime deadman exceeds the packet timeout")
    if requested_interval > bounds["budget_interval"]:
        raise LifecycleError("runtime budget interval exceeds the packet interval")

    quote = packet["quote"]
    rate_key = "hourly_rate" if rate_role == "primary" else "comparator_hourly_rate"
    if rate_role not in {"primary", "comparator"}:
        raise LifecycleError("rate role must be primary or comparator")
    rate = _decimal(quote.get(rate_key), label=f"quote {rate_key}", positive=True)
    projection = quote.get("cost_projection")
    if not isinstance(projection, Mapping):
        raise LifecycleError("quote cost projection is missing")
    fixed = sum(
        (_decimal(projection.get(key), label=f"quote {key}") for key in FIXED_COST_FIELDS),
        Decimal(0),
    )
    return LifecyclePolicy(
        packet_sha256=packet_sha256,
        rate_role=rate_role,
        ttl_seconds=requested_ttl,
        deadman_seconds=requested_deadman,
        budget_check_seconds=requested_interval,
        hourly_rate_usd=rate,
        fixed_cost_usd=fixed,
        prior_cost_usd=_decimal(prior_cost_usd, label="prior cost"),
        budget_cap_usd=bounds["budget_cap"],
        termination_grace_seconds=_seconds(
            termination_grace_seconds, label="termination grace"
        ),
        teardown_timeout_seconds=_seconds(
            teardown_timeout_seconds, label="teardown timeout"
        ),
        control_sha256=dict(control_sha256),
    )


class LifecycleJournal:
    """Exclusive-create, O_APPEND, fsync-per-event hash-chained JSONL journal."""

    def __init__(self, path: str | Path, run_id: str, started_ns: int):
        self.path = Path(path)
        self.run_id = run_id
        self.started_ns = started_ns
        self.sequence = 0
        self.previous = ZERO_HASH
        self.failed = False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._descriptor = os.open(
                self.path,
                os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        except OSError as exc:
            raise LifecycleError("lifecycle record must be a new writable file") from exc

    def append(self, event: str, details: Mapping[str, Any]) -> str:
        if not isinstance(event, str) or not event or not isinstance(details, Mapping):
            raise LifecycleError("invalid lifecycle event")
        elapsed_ns = max(0, time.monotonic_ns() - self.started_ns)
        body = {
            "schema": EVENT_SCHEMA,
            "run_id": self.run_id,
            "sequence": self.sequence,
            "event": event,
            "recorded_at": _utc_now(),
            "monotonic_elapsed_ns": elapsed_ns,
            "previous_sha256": self.previous,
            "details": dict(details),
        }
        event_hash = _sha256_bytes(_canonical_bytes(body))
        record = dict(body, event_sha256=event_hash)
        payload = _canonical_bytes(record) + b"\n"
        view = memoryview(payload)
        while view:
            written = os.write(self._descriptor, view)
            if written < 1:
                raise OSError("short lifecycle journal write")
            view = view[written:]
        os.fsync(self._descriptor)
        self.previous = event_hash
        self.sequence += 1
        return event_hash

    def close(self) -> None:
        if getattr(self, "_descriptor", None) is not None:
            os.close(self._descriptor)
            self._descriptor = None


def _safe_journal(
    journal: LifecycleJournal, event: str, details: Mapping[str, Any]
) -> bool:
    """Best-effort evidence write; safety actions must not depend on storage."""
    try:
        journal.append(event, details)
    except BaseException:
        journal.failed = True
        return False
    return True


class _SignalLatch(AbstractContextManager["_SignalLatch"]):
    def __init__(self) -> None:
        self.signum: int | None = None
        self._previous: dict[int, Any] = {}
        self._enabled = threading.current_thread() is threading.main_thread()

    def _handler(self, signum: int, _frame: FrameType | None) -> None:
        if self.signum is None:
            self.signum = signum

    def __enter__(self) -> "_SignalLatch":
        if self._enabled:
            for signum in SAFE_SIGNALS:
                self._previous[signum] = signal.getsignal(signum)
                signal.signal(signum, self._handler)
        return self

    def __exit__(self, *_exc: Any) -> None:
        if self._enabled:
            for signum, handler in self._previous.items():
                signal.signal(signum, handler)


def _spawn(
    argv: Sequence[str], cwd: Path, *, environment: Mapping[str, str] | None = None,
) -> subprocess.Popen[bytes]:
    # No argv, environment, stdout, or stderr is captured into lifecycle evidence.
    return subprocess.Popen(
        tuple(argv),
        cwd=cwd,
        shell=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        start_new_session=(os.name == "posix"),
        env=dict(environment) if environment is not None else None,
    )


def _signal_process_group(process: subprocess.Popen[bytes], signum: int) -> None:
    try:
        if os.name == "posix":
            os.killpg(process.pid, signum)
        elif signum == getattr(signal, "SIGKILL", -1):
            process.kill()
        else:
            process.terminate()
    except OSError:
        pass


def _stop_process(
    process: subprocess.Popen[bytes],
    grace_seconds: Decimal,
    journal: LifecycleJournal,
    *,
    label: str,
) -> None:
    _safe_journal(journal, f"{label}_terminate_requested", {})
    _signal_process_group(process, getattr(signal, "SIGTERM", 15))
    try:
        process.wait(timeout=float(grace_seconds))
    except subprocess.TimeoutExpired:
        _safe_journal(journal, f"{label}_kill_requested", {})
        _signal_process_group(process, getattr(signal, "SIGKILL", 9))
        try:
            process.wait(timeout=max(1.0, min(10.0, float(grace_seconds))))
        except subprocess.TimeoutExpired:
            _safe_journal(journal, f"{label}_kill_unconfirmed", {})


def _heartbeat_fingerprint(path: Path) -> tuple[int, int, int, int] | None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise LifecycleError("heartbeat path is not inspectable") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise LifecycleError("heartbeat path must be a regular file, not a symlink")
    return (
        metadata.st_ino,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
        metadata.st_size,
    )


def _estimated_cost(policy: LifecyclePolicy, elapsed_ns: int) -> Decimal:
    seconds = Decimal(elapsed_ns) / Decimal(1_000_000_000)
    return (
        policy.prior_cost_usd
        + policy.fixed_cost_usd
        + policy.hourly_rate_usd * seconds / Decimal(3600)
    )


def _money(value: Decimal) -> str:
    return format(value.quantize(Decimal("0.000001")), "f")


def validate_absence_artifact(
    path: str | Path, *, packet_sha256: str,
    deployment_manifest_sha256: str, resource_identity: Mapping[str, str],
    resource_identity_sha256: str, verifier_argv_sha256: str,
    lifecycle_launch_sha256: str | None = None,
    launch_id: str | None = None,
    launch_nonce_sha256: str | None = None,
    not_before_ns: int | None = None,
    allow_legacy_test: bool = False,
) -> dict[str, Any]:
    """Require structured, resource-bound post-teardown absence evidence."""
    source = Path(path).resolve()
    metadata = _regular_file(source, label="resource absence artifact")
    if not_before_ns is not None and metadata.st_mtime_ns < not_before_ns:
        raise LifecycleError("resource absence artifact predates teardown")
    document = _load_json_object(source, label="resource absence artifact")
    if allow_legacy_test and document.get("schema") == LEGACY_ABSENCE_SCHEMA:
        expected_keys = {
            "schema", "packet_sha256", "deployment_manifest_sha256",
            "resource_identity_sha256", "verifier_argv_sha256", "absent",
            "observed_state", "queried_resource_id", "provider_response_sha256",
            "checked_at",
        }
        if set(document) != expected_keys:
            raise LifecycleError("legacy resource absence artifact fields differ")
        if (document.get("packet_sha256") != packet_sha256
                or document.get("deployment_manifest_sha256")
                != deployment_manifest_sha256
                or document.get("resource_identity_sha256")
                != resource_identity_sha256
                or document.get("verifier_argv_sha256") != verifier_argv_sha256
                or document.get("absent") is not True
                or document.get("observed_state") != "absent"
                or document.get("queried_resource_id")
                != resource_identity.get("instance_id")
                or not _is_sha256(document.get("provider_response_sha256"))):
            raise LifecycleError("legacy resource absence artifact binding differs")
        checked_at = document.get("checked_at")
        try:
            datetime.fromisoformat(checked_at.replace("Z", "+00:00"))
        except (AttributeError, ValueError) as exc:
            raise LifecycleError("absence artifact timestamp is invalid") from exc
        return {
            "path": str(source),
            "sha256": _sha256_file(source),
            "bytes": metadata.st_size,
            "provider_response_sha256": document["provider_response_sha256"],
        }
    expected_keys = {
        "schema", "packet_sha256", "deployment_manifest_sha256",
        "lifecycle_launch_sha256", "launch_id", "launch_nonce_sha256",
        "resource_identity_sha256", "verifier_argv_sha256", "absent",
        "observed_state", "queried_resource_id", "provider_response",
        "query_started_at", "checked_at",
    }
    if set(document) != expected_keys or document.get("schema") != ABSENCE_SCHEMA:
        raise LifecycleError("resource absence artifact fields differ")
    if (
        document.get("packet_sha256") != packet_sha256
        or document.get("deployment_manifest_sha256")
        != deployment_manifest_sha256
        or document.get("resource_identity_sha256") != resource_identity_sha256
        or document.get("verifier_argv_sha256") != verifier_argv_sha256
        or document.get("lifecycle_launch_sha256") != lifecycle_launch_sha256
        or document.get("launch_id") != launch_id
        or document.get("launch_nonce_sha256") != launch_nonce_sha256
    ):
        raise LifecycleError("resource absence artifact binding differs")
    if document.get("absent") is not True or document.get("observed_state") != "absent":
        raise LifecycleError("resource absence was not independently observed")
    if document.get("queried_resource_id") != resource_identity.get("instance_id"):
        raise LifecycleError("absence verifier queried a different resource")
    response = document.get("provider_response")
    if (not isinstance(response, Mapping)
            or set(response) != {"path", "sha256", "bytes"}
            or not _is_sha256(response.get("sha256"))
            or type(response.get("bytes")) is not int
            or response["bytes"] < 1):
        raise LifecycleError("absence artifact lacks raw provider-response evidence")
    response_path = Path(str(response.get("path"))).resolve()
    response_metadata = _regular_file(
        response_path, label="raw provider absence response")
    if (response_metadata.st_size != response["bytes"]
            or _sha256_file(response_path) != response["sha256"]
            or (not_before_ns is not None
                and response_metadata.st_mtime_ns < not_before_ns)):
        raise LifecycleError("raw provider absence response bytes differ")
    query_started_at = document.get("query_started_at")
    checked_at = document.get("checked_at")
    try:
        query_started = datetime.fromisoformat(
            query_started_at.replace("Z", "+00:00"))
        checked = datetime.fromisoformat(checked_at.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise LifecycleError("absence artifact timestamp is invalid") from exc
    if (not query_started_at.endswith("Z") or not checked_at.endswith("Z")
            or checked < query_started
            or checked - query_started > timedelta(minutes=10)
            or datetime.now(timezone.utc) - checked > timedelta(minutes=10)
            or checked - datetime.now(timezone.utc) > timedelta(seconds=30)):
        raise LifecycleError("absence artifact freshness window is invalid")
    return {
        "path": str(source),
        "sha256": _sha256_file(source),
        "bytes": metadata.st_size,
        "provider_response_sha256": response["sha256"],
    }


def _capture_workload_outputs(output_path: Path, workload_exit_code: int | None) -> dict[str, Any]:
    if workload_exit_code != 0:
        return {"status": "unavailable", "artifacts": {}}
    artifacts: dict[str, Any] = {}
    for name in ("execution_summary.json", "artifact_manifest.json"):
        path = output_path / name
        metadata = _regular_file(path, label=f"workload output {name}")
        if metadata.st_size < 2:
            raise LifecycleError(f"workload output {name} is empty")
        _load_json_object(path, label=f"workload output {name}")
        artifacts[name] = {
            "path": str(path.resolve()),
            "sha256": _sha256_file(path),
            "bytes": metadata.st_size,
        }
    return {"status": "captured", "artifacts": artifacts}


def _write_lifecycle_receipt(
    path: Path, *, run_id: str, arm_id: str, packet_sha256: str,
    deployment_manifest_sha256: str, lifecycle_launch_sha256: str,
    campaign_root_sha256: str, pair_declaration_sha256: str,
    runtime_binding_sha256: str, active_security_manifest_sha256: str,
    resource_identity_sha256: str,
    launch_id: str, launch_issued_at: str, launch_expires_at: str,
    launch_context_path: Path, launch_context_sha256: str,
    lifecycle_validation: Mapping[str, Any], ledger_path: Path,
    ledger_checkpoint: Mapping[str, str], workload_outputs: Mapping[str, Any],
    absence_artifact: Mapping[str, Any] | None, exit_code: int,
) -> Path:
    body = {
        "schema": RECEIPT_SCHEMA,
        "run_id": run_id,
        "arm_id": arm_id,
        "packet_sha256": packet_sha256,
        "deployment_manifest_sha256": deployment_manifest_sha256,
        "lifecycle_launch_sha256": lifecycle_launch_sha256,
        "campaign_root_sha256": campaign_root_sha256,
        "pair_declaration_sha256": pair_declaration_sha256,
        "runtime_binding_sha256": runtime_binding_sha256,
        "active_security_manifest_sha256": active_security_manifest_sha256,
        "resource_identity_sha256": resource_identity_sha256,
        "launch_id": launch_id,
        "launch_issued_at": launch_issued_at,
        "launch_expires_at": launch_expires_at,
        "launch_context": {
            "path": str(launch_context_path.resolve()),
            "sha256": launch_context_sha256,
        },
        "lifecycle_record": {
            "path": str(Path(str(lifecycle_validation["path"])).resolve()),
            "sha256": lifecycle_validation["record_sha256"],
            "chain_head_sha256": lifecycle_validation["chain_head_sha256"],
        },
        "campaign_ledger": {
            "path": str(ledger_path.resolve()),
            "file_sha256_at_completion": ledger_checkpoint["file_sha256"],
            "state_sha256": ledger_checkpoint["state_sha256"],
            "entry_sha256": ledger_checkpoint["entry_sha256"],
        },
        "workload_outputs": dict(workload_outputs),
        "resource_absence": dict(absence_artifact) if absence_artifact else None,
        "exit_code": exit_code,
        "completed_at": _utc_now(),
    }
    receipt = {**body, "receipt_sha256": _sha256_bytes(_canonical_bytes(body))}
    return _write_json_exclusive(path, receipt)


def validate_lifecycle_receipt(path: str | Path) -> dict[str, Any]:
    """Validate a receipt and all post-run byte bindings it still references."""
    source = Path(path).resolve()
    document = _load_json_object(source, label="lifecycle receipt")
    expected_keys = {
        "schema", "run_id", "arm_id", "packet_sha256",
        "deployment_manifest_sha256", "lifecycle_launch_sha256",
        "campaign_root_sha256", "pair_declaration_sha256",
        "runtime_binding_sha256", "active_security_manifest_sha256",
        "resource_identity_sha256",
        "launch_id", "launch_issued_at", "launch_expires_at",
        "launch_context", "lifecycle_record", "campaign_ledger",
        "workload_outputs", "resource_absence", "exit_code", "completed_at",
        "receipt_sha256",
    }
    if set(document) != expected_keys or document.get("schema") != RECEIPT_SCHEMA:
        raise LifecycleError("lifecycle receipt fields differ")
    body = {key: value for key, value in document.items() if key != "receipt_sha256"}
    if document.get("receipt_sha256") != _sha256_bytes(_canonical_bytes(body)):
        raise LifecycleError("lifecycle receipt digest mismatch")
    for key in ("packet_sha256", "deployment_manifest_sha256",
                "lifecycle_launch_sha256", "runtime_binding_sha256",
                "active_security_manifest_sha256", "campaign_root_sha256",
                "pair_declaration_sha256", "resource_identity_sha256"):
        if not _is_sha256(document.get(key)):
            raise LifecycleError(f"lifecycle receipt {key} is invalid")
    try:
        uuid.UUID(document.get("launch_id"))
        issued = datetime.fromisoformat(
            document["launch_issued_at"].replace("Z", "+00:00"))
        expires = datetime.fromisoformat(
            document["launch_expires_at"].replace("Z", "+00:00"))
        completed = datetime.fromisoformat(
            document["completed_at"].replace("Z", "+00:00"))
    except (TypeError, ValueError, AttributeError) as exc:
        raise LifecycleError("lifecycle receipt launch window is invalid") from exc
    if completed > expires + timedelta(minutes=5) or completed < issued:
        raise LifecycleError("lifecycle receipt falls outside its launch window")
    lifecycle = document.get("lifecycle_record")
    if not isinstance(lifecycle, Mapping):
        raise LifecycleError("lifecycle receipt record binding is invalid")
    record_path = Path(str(lifecycle.get("path"))).resolve()
    record_validation = validate_lifecycle_record(record_path)
    if (
        lifecycle.get("sha256") != record_validation["record_sha256"]
        or lifecycle.get("chain_head_sha256")
        != record_validation["chain_head_sha256"]
    ):
        raise LifecycleError("lifecycle receipt record bytes differ")
    launch = document.get("launch_context")
    if not isinstance(launch, Mapping) or set(launch) != {"path", "sha256"}:
        raise LifecycleError("lifecycle receipt launch-context binding is invalid")
    launch_artifact = Path(str(launch.get("path"))).resolve()
    _regular_file(launch_artifact, label="receipt launch context")
    if launch.get("sha256") != _sha256_file(launch_artifact):
        raise LifecycleError("lifecycle receipt launch-context bytes differ")
    ledger = document.get("campaign_ledger")
    if not isinstance(ledger, Mapping) or set(ledger) != {
        "path", "file_sha256_at_completion", "state_sha256", "entry_sha256",
    }:
        raise LifecycleError("lifecycle receipt campaign-ledger binding is invalid")
    for key in ("file_sha256_at_completion", "state_sha256", "entry_sha256"):
        if not _is_sha256(ledger.get(key)):
            raise LifecycleError("lifecycle receipt campaign-ledger digest is invalid")
    ledger_artifact = Path(str(ledger.get("path"))).resolve()
    current_ledger = _load_json_object(ledger_artifact, label="receipt campaign ledger")
    ledger_body = {
        key: value for key, value in current_ledger.items() if key != "ledger_sha256"
    }
    if current_ledger.get("ledger_sha256") != _sha256_bytes(
        _canonical_bytes(ledger_body)
    ):
        raise LifecycleError("receipt campaign ledger digest mismatch")
    entries = current_ledger.get("entries")
    if not isinstance(entries, list):
        raise LifecycleError("receipt campaign ledger entries are invalid")
    previous_entry = ZERO_HASH
    matched = False
    for sequence, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise LifecycleError("receipt campaign ledger entry is invalid")
        entry_body = {key: value for key, value in entry.items() if key != "entry_sha256"}
        entry_sha = _sha256_bytes(_canonical_bytes(entry_body))
        if (entry.get("sequence") != sequence
                or entry.get("previous_entry_sha256") != previous_entry
                or entry.get("entry_sha256") != entry_sha):
            raise LifecycleError("receipt campaign ledger entry chain differs")
        previous_entry = entry_sha
        if entry_sha == ledger["entry_sha256"]:
            matched = (
                entry.get("run_id") == document["run_id"]
                and entry.get("lifecycle_record_sha256")
                == lifecycle["sha256"]
            )
    if not matched:
        raise LifecycleError("receipt campaign ledger entry is missing or changed")
    outputs = document.get("workload_outputs")
    if not isinstance(outputs, Mapping) or outputs.get("status") not in {
        "captured", "unavailable",
    }:
        raise LifecycleError("lifecycle receipt workload outputs are invalid")
    artifacts = outputs.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise LifecycleError("lifecycle receipt workload artifact map is invalid")
    for name, binding in artifacts.items():
        if name not in {"execution_summary.json", "artifact_manifest.json"} \
                or not isinstance(binding, Mapping):
            raise LifecycleError("lifecycle receipt workload artifact differs")
        artifact = Path(str(binding.get("path"))).resolve()
        metadata = _regular_file(artifact, label=f"receipt workload artifact {name}")
        if (binding.get("sha256") != _sha256_file(artifact)
                or binding.get("bytes") != metadata.st_size):
            raise LifecycleError("lifecycle receipt workload artifact bytes differ")
    absence = document.get("resource_absence")
    if document.get("exit_code") == 0 and not isinstance(absence, Mapping):
        raise LifecycleError("successful lifecycle receipt lacks absence evidence")
    if isinstance(absence, Mapping):
        artifact = Path(str(absence.get("path"))).resolve()
        metadata = _regular_file(artifact, label="receipt absence artifact")
        if (absence.get("sha256") != _sha256_file(artifact)
                or absence.get("bytes") != metadata.st_size):
            raise LifecycleError("lifecycle receipt absence bytes differ")
    return {
        "schema": "p4.cc_lifecycle_receipt_validation.v1",
        "valid": True,
        "run_id": document["run_id"],
        "arm_id": document["arm_id"],
        "receipt_sha256": _sha256_file(source),
        "record_sha256": record_validation["record_sha256"],
        "chain_head_sha256": record_validation["chain_head_sha256"],
        "exit_code": document["exit_code"],
    }


@dataclass(frozen=True)
class LifecycleResult:
    run_id: str
    reason: str
    workload_exit_code: int | None
    teardown_exit_code: int | None
    elapsed_seconds: Decimal
    estimated_cost_usd: Decimal
    record_path: Path
    receipt_path: Path
    ledger_path: Path
    absence_artifact_path: Path
    exit_code: int


def _result_exit_code(
    reason: str,
    workload_exit_code: int | None,
    teardown_exit_code: int | None,
    absence_verified: bool,
    workload_outputs_captured: bool,
    signum: int | None,
) -> int:
    if teardown_exit_code != 0:
        return 70
    if not absence_verified:
        return 71
    if reason == "success" and not workload_outputs_captured:
        return 72
    if reason == "success":
        return 0
    if reason == "workload_failed" and workload_exit_code is not None:
        return workload_exit_code if 1 <= workload_exit_code <= 125 else 1
    if reason == "ttl_exceeded":
        return 124
    if reason == "heartbeat_deadman_exceeded":
        return 125
    if reason == "budget_exceeded":
        return 126
    if reason == "signal" and signum is not None:
        return min(255, 128 + signum)
    return 1


def _run_supervised_legacy(
    *,
    workload: ArgvBinding,
    teardown: ArgvBinding,
    policy: LifecyclePolicy,
    heartbeat_path: str | Path,
    record_path: str | Path,
    cwd: str | Path,
) -> LifecycleResult:
    """Run one workload and unconditionally invoke its teardown adapter.

    The returned exit code is nonzero on a workload violation/failure and is
    always nonzero when teardown fails, times out, or cannot start.
    """
    raise LifecycleError(
        "legacy lifecycle path is disabled; use the deployment-bound supervisor"
    )
    if not workload.argv or not teardown.argv:
        raise LifecycleError("workload and teardown argv bindings are required")
    workdir = Path(cwd).resolve()
    if not workdir.is_dir():
        raise LifecycleError("lifecycle workdir must be a directory")
    heartbeat = Path(heartbeat_path)
    if not heartbeat.is_absolute():
        heartbeat = workdir / heartbeat
    heartbeat = heartbeat.resolve(strict=False)

    start_ns = time.monotonic_ns()
    run_id = str(uuid.uuid4())
    journal = LifecycleJournal(record_path, run_id, start_ns)
    workload_process: subprocess.Popen[bytes] | None = None
    workload_code: int | None = None
    teardown_code: int | None = None
    signum: int | None = None
    reason = "supervisor_error"
    last_heartbeat_ns = start_ns
    heartbeat_state: tuple[int, int, int, int] | None = None
    teardown_status = "not-started"
    outer_signals = _SignalLatch()
    outer_signals.__enter__()

    try:
        heartbeat_state = _heartbeat_fingerprint(heartbeat)
        journal.append(
            "lifecycle_started",
            {
                "packet_sha256": policy.packet_sha256,
                "control_sha256": dict(sorted(policy.control_sha256.items())),
                "workload_argv_sha256": workload.sha256,
                "workload_argv_bytes": workload.bytes,
                "teardown_argv_sha256": teardown.sha256,
                "teardown_argv_bytes": teardown.bytes,
                "rate_role": policy.rate_role,
                "ttl_seconds": str(policy.ttl_seconds),
                "deadman_seconds": str(policy.deadman_seconds),
                "budget_check_seconds": str(policy.budget_check_seconds),
                "hourly_rate_usd": _money(policy.hourly_rate_usd),
                "fixed_cost_usd": _money(policy.fixed_cost_usd),
                "prior_cost_usd": _money(policy.prior_cost_usd),
                "budget_cap_usd": _money(policy.budget_cap_usd),
                "shell": False,
                "child_output_captured": False,
            },
        )
        base_cost = _estimated_cost(policy, 0)
        cleanup_reserve_seconds = (
            policy.termination_grace_seconds + policy.teardown_timeout_seconds
        )
        cleanup_reserve_cost = (
            policy.hourly_rate_usd * cleanup_reserve_seconds / Decimal(3600)
        )
        if base_cost + cleanup_reserve_cost >= policy.budget_cap_usd:
            reason = "budget_exceeded"
            journal.append(
                "budget_violation",
                {
                    "estimated_cost_usd": _money(base_cost),
                    "cleanup_reserve_cost_usd": _money(cleanup_reserve_cost),
                    "budget_cap_usd": _money(policy.budget_cap_usd),
                    "phase": "pre-spawn",
                },
            )
        else:
            try:
                workload_process = _spawn(workload.argv, workdir)
            except (OSError, ValueError):
                reason = "workload_spawn_failed"
                journal.append("workload_spawn_failed", {})
            else:
                journal.append("workload_started", {"pid": workload_process.pid})
                ttl_ns = _seconds_to_ns(policy.ttl_seconds)
                deadman_ns = _seconds_to_ns(policy.deadman_seconds)
                interval_ns = _seconds_to_ns(policy.budget_check_seconds)
                remaining = policy.budget_cap_usd - base_cost
                raw_budget_deadline_ns = int(
                    remaining * Decimal(3600) * Decimal(1_000_000_000)
                    / policy.hourly_rate_usd
                )
                cleanup_reserve_ns = _seconds_to_ns(cleanup_reserve_seconds)
                budget_deadline_ns = max(0, raw_budget_deadline_ns - cleanup_reserve_ns)
                next_budget_ns = start_ns
                with _SignalLatch() as signals:
                    while True:
                        now_ns = time.monotonic_ns()
                        elapsed_ns = max(0, now_ns - start_ns)
                        code = workload_process.poll()
                        if code is not None:
                            workload_code = code
                            reason = "success" if code == 0 else "workload_failed"
                            journal.append("workload_exited", {"exit_code": code})
                            break
                        observed_signal = (
                            signals.signum
                            if signals.signum is not None
                            else outer_signals.signum
                        )
                        if observed_signal is not None:
                            signum = observed_signal
                            reason = "signal"
                            journal.append("signal_received", {"signal": signum})
                            break
                        try:
                            current_heartbeat = _heartbeat_fingerprint(heartbeat)
                        except LifecycleError:
                            reason = "heartbeat_invalid"
                            journal.append("heartbeat_invalid", {})
                            break
                        if current_heartbeat != heartbeat_state:
                            heartbeat_state = current_heartbeat
                            if current_heartbeat is not None:
                                last_heartbeat_ns = now_ns
                                journal.append("heartbeat_observed", {})
                        if elapsed_ns >= ttl_ns:
                            reason = "ttl_exceeded"
                            journal.append(
                                "ttl_violation",
                                {"ttl_seconds": str(policy.ttl_seconds)},
                            )
                            break
                        if now_ns - last_heartbeat_ns >= deadman_ns:
                            reason = "heartbeat_deadman_exceeded"
                            journal.append(
                                "deadman_violation",
                                {"deadman_seconds": str(policy.deadman_seconds)},
                            )
                            break
                        if elapsed_ns >= budget_deadline_ns:
                            reason = "budget_exceeded"
                            journal.append(
                                "budget_violation",
                                {
                                    "estimated_cost_usd": _money(
                                        _estimated_cost(policy, elapsed_ns)
                                    ),
                                    "budget_cap_usd": _money(policy.budget_cap_usd),
                                    "phase": "predictive-deadline",
                                },
                            )
                            break
                        if now_ns >= next_budget_ns:
                            estimated = _estimated_cost(policy, elapsed_ns)
                            estimated_with_cleanup = estimated + cleanup_reserve_cost
                            journal.append(
                                "budget_checked",
                                {
                                    "estimated_cost_usd": _money(estimated),
                                    "estimated_with_cleanup_usd": _money(
                                        estimated_with_cleanup
                                    ),
                                    "budget_cap_usd": _money(policy.budget_cap_usd),
                                },
                            )
                            if estimated_with_cleanup >= policy.budget_cap_usd:
                                reason = "budget_exceeded"
                                journal.append(
                                    "budget_violation",
                                    {
                                        "estimated_cost_usd": _money(estimated),
                                        "estimated_with_cleanup_usd": _money(
                                            estimated_with_cleanup
                                        ),
                                        "budget_cap_usd": _money(policy.budget_cap_usd),
                                        "phase": "periodic-check",
                                    },
                                )
                                break
                            next_budget_ns = now_ns + interval_ns

                        wake_ns = min(
                            start_ns + ttl_ns,
                            last_heartbeat_ns + deadman_ns,
                            start_ns + budget_deadline_ns,
                            next_budget_ns,
                            now_ns + 50_000_000,
                        )
                        if wake_ns > now_ns:
                            time.sleep((wake_ns - now_ns) / 1_000_000_000)
    except BaseException as exc:  # teardown must run even for an unexpected supervisor error
        reason = "supervisor_error"
        _safe_journal(
            journal, "supervisor_error", {"exception_type": type(exc).__name__}
        )
    finally:
        if workload_process is not None:
            try:
                if workload_process.poll() is None:
                    _stop_process(
                        workload_process,
                        policy.termination_grace_seconds,
                        journal,
                        label="workload",
                    )
            except BaseException as exc:
                _safe_journal(
                    journal,
                    "workload_stop_error",
                    {"exception_type": type(exc).__name__},
                )
            finally:
                if workload_code is None:
                    workload_code = workload_process.poll()

        _safe_journal(journal, "teardown_started", {"shell": False})
        try:
            try:
                teardown_process = _spawn(teardown.argv, workdir)
            except (OSError, ValueError):
                teardown_status = "spawn-failed"
                teardown_code = None
            else:
                try:
                    teardown_code = teardown_process.wait(
                        timeout=float(policy.teardown_timeout_seconds)
                    )
                    teardown_status = "succeeded" if teardown_code == 0 else "failed"
                except subprocess.TimeoutExpired:
                    teardown_status = "timed-out"
                    _stop_process(
                        teardown_process,
                        policy.termination_grace_seconds,
                        journal,
                        label="teardown",
                    )
                    teardown_code = teardown_process.poll()
            _safe_journal(
                journal,
                "teardown_finished",
                {"status": teardown_status, "exit_code": teardown_code},
            )
        except BaseException as exc:
            teardown_status = "supervisor-error"
            teardown_code = None
            _safe_journal(
                journal,
                "teardown_finished",
                {"status": teardown_status, "exception_type": type(exc).__name__},
            )

        if outer_signals.signum is not None and signum is None:
            signum = outer_signals.signum
            if reason == "success":
                reason = "signal"
                _safe_journal(
                    journal,
                    "signal_received",
                    {"signal": signum, "phase": "teardown"},
                )
        end_ns = time.monotonic_ns()
        elapsed_ns = max(0, end_ns - start_ns)
        estimated = _estimated_cost(policy, elapsed_ns)
        if journal.failed and reason == "success":
            reason = "journal_failure"
        exit_code = _result_exit_code(reason, workload_code, teardown_code, signum)
        final_written = _safe_journal(
            journal,
            "lifecycle_finished",
            {
                "reason": reason,
                "workload_exit_code": workload_code,
                "teardown_status": teardown_status,
                "teardown_exit_code": teardown_code,
                "estimated_cost_usd": _money(estimated),
                "exit_code": exit_code,
            },
        )
        if not final_written and exit_code == 0:
            reason = "journal_failure"
            exit_code = 1
        try:
            journal.close()
        finally:
            outer_signals.__exit__(None, None, None)

    return LifecycleResult(
        run_id=run_id,
        reason=reason,
        workload_exit_code=workload_code,
        teardown_exit_code=teardown_code,
        elapsed_seconds=Decimal(max(0, time.monotonic_ns() - start_ns))
        / Decimal(1_000_000_000),
        estimated_cost_usd=estimated,
        record_path=Path(record_path),
        exit_code=exit_code,
    )


def _argv_option_path(argv: Sequence[str], option: str, cwd: Path) -> Path:
    if argv.count(option) != 1:
        raise LifecycleError(f"workload argv must contain exactly one {option}")
    index = argv.index(option)
    if index + 1 >= len(argv):
        raise LifecycleError(f"workload argv {option} has no value")
    value = Path(argv[index + 1])
    return (value if value.is_absolute() else cwd / value).resolve(strict=False)


def run_supervised(
    *, workload: ArgvBinding, teardown: ArgvBinding,
    absence_verifier: ArgvBinding, policy: LifecyclePolicy,
    packet: Mapping[str, Any], deployment_manifest_path: str | Path,
    heartbeat_path: str | Path, record_path: str | Path,
    launch_context_path: str | Path, ledger_path: str | Path,
    absence_artifact_path: str | Path, receipt_path: str | Path,
    workload_output_path: str | Path, arm_id: str, cwd: str | Path,
    lifecycle_launch_manifest_path: str | Path | None = None,
    runtime_binding_path: str | Path | None = None,
    active_security_manifest_path: str | Path | None = None,
    controller_public_key_path: str | Path | None = None,
    pair_declaration_path: str | Path | None = None,
    pair_auth_path: str | Path | None = None,
    pair_campaign_public_key_path: str | Path | None = None,
    arm_arrival_path: str | Path | None = None,
    guest_workload: ArgvBinding | None = None,
    launch_nonce_path: str | Path | None = None,
    heartbeat_public_key_path: str | Path | None = None,
    remote_launch_adapter_path: str | Path | None = None,
    bootstrap_lifecycle_path: str | Path | None = None,
    active_transition_path: str | Path | None = None,
    measurement_ready_state_path: str | Path | None = None,
    cc_on_receipt_path: str | Path | None = None,
    test_only_allow_legacy_local: bool = False,
) -> LifecycleResult:
    """Run one campaign arm from an external, fail-closed control plane.

    Production ``workload`` is the exact remote-launch adapter, not a process
    that executes inside the resource being deleted.  Local direct execution
    is available only to synthetic unit tests through the explicit test-only
    switch, which the production CLI never exposes.
    """
    if not workload.argv or not teardown.argv or not absence_verifier.argv:
        raise LifecycleError("workload, teardown, and absence-verifier argv are required")
    if policy.prior_cost_usd != 0:
        raise LifecycleError("prior cost is ledger-owned and cannot be caller supplied")
    workdir = Path(cwd).resolve()
    if not workdir.is_dir():
        raise LifecycleError("lifecycle workdir must be a directory")

    def resolved(value: str | Path) -> Path:
        candidate = Path(value)
        return (candidate if candidate.is_absolute() else workdir / candidate).resolve(
            strict=False
        )

    heartbeat = resolved(heartbeat_path)
    record = resolved(record_path)
    context_path = resolved(launch_context_path)
    absence_path = resolved(absence_artifact_path)
    receipt = resolved(receipt_path)
    output_path = resolved(workload_output_path)
    if _argv_option_path(workload.argv, "--output", workdir) != output_path:
        raise LifecycleError("workload output differs from its frozen argv")
    if workload.argv.count("--protection-mode") != 1:
        raise LifecycleError("workload argv must freeze one protection mode")
    arm_index = workload.argv.index("--protection-mode") + 1
    if arm_index >= len(workload.argv) or workload.argv[arm_index] != arm_id:
        raise LifecycleError("campaign arm differs from workload argv")
    if (absence_path.exists() or context_path.exists() or receipt.exists()
            or record.exists()):
        raise LifecycleError("per-run lifecycle artifacts must not pre-exist")
    if not test_only_allow_legacy_local and heartbeat.exists():
        raise LifecycleError("production heartbeat path must not pre-exist")

    binding = validate_deployment_lifecycle_binding(
        packet=packet,
        packet_sha256=policy.packet_sha256,
        deployment_manifest_path=deployment_manifest_path,
        workload=workload,
        teardown=teardown,
        absence_verifier=absence_verifier,
        ledger_path=ledger_path,
        arm_id=arm_id,
        cwd=workdir,
        lifecycle_launch_manifest_path=lifecycle_launch_manifest_path,
        runtime_binding_path=runtime_binding_path,
        active_security_manifest_path=active_security_manifest_path,
        controller_public_key_path=controller_public_key_path,
        pair_declaration_path=pair_declaration_path,
        pair_auth_path=pair_auth_path,
        pair_campaign_public_key_path=pair_campaign_public_key_path,
        arm_arrival_path=arm_arrival_path,
        guest_workload=guest_workload,
        launch_nonce_path=launch_nonce_path,
        heartbeat_public_key_path=heartbeat_public_key_path,
        remote_launch_adapter_path=remote_launch_adapter_path,
        bootstrap_lifecycle_path=bootstrap_lifecycle_path,
        active_transition_path=active_transition_path,
        measurement_ready_state_path=measurement_ready_state_path,
        cc_on_receipt_path=cc_on_receipt_path,
        test_only_allow_legacy_local=test_only_allow_legacy_local,
    )
    if str(absence_path) not in absence_verifier.argv:
        raise LifecycleError("absence verifier argv must name its exact artifact path")

    start_ns = time.monotonic_ns()
    run_id = str(uuid.uuid4())
    journal: LifecycleJournal
    workload_process: subprocess.Popen[bytes] | None = None
    workload_code: int | None = None
    teardown_code: int | None = None
    verifier_code: int | None = None
    signum: int | None = None
    reason = "supervisor_error"
    teardown_status = "not-started"
    absence_status = "not-started"
    absence_evidence: dict[str, Any] | None = None
    workload_outputs: dict[str, Any] = {"status": "unavailable", "artifacts": {}}
    last_heartbeat_ns = start_ns
    heartbeat_state: tuple[int, int, int, int] | None = None
    heartbeat_sequence = 0
    previous_heartbeat_sha256 = ZERO_HASH
    context_sha256 = ""
    estimated = Decimal(0)
    exit_code = 1
    ledger_checkpoint: dict[str, str] = {}
    outer_signals = _SignalLatch()

    with CampaignCostLedger(
        binding.ledger_path,
        campaign_id=binding.campaign_id,
        packet_sha256=policy.packet_sha256,
        deployment_manifest_sha256=binding.deployment_manifest_sha256,
        campaign_root_sha256=binding.campaign_root_sha256,
        pair_declaration_sha256=binding.pair_declaration_sha256,
        budget_cap_usd=policy.budget_cap_usd,
    ) as ledger:
        effective_policy = replace(
            policy,
            prior_cost_usd=ledger.accrued_cost_usd,
            fixed_cost_usd=(Decimal(0) if ledger.fixed_cost_already_charged
                            else policy.fixed_cost_usd),
        )
        ledger.begin(
            run_id=run_id, arm_id=arm_id, rate_role=policy.rate_role,
            lifecycle_launch_sha256=binding.lifecycle_launch_sha256,
        )
        journal = LifecycleJournal(record, run_id, start_ns)
        outer_signals.__enter__()
        try:
            token, context_sha256 = _issue_launch_context(
                context_path,
                run_id=run_id,
                packet_sha256=policy.packet_sha256,
                deployment_manifest_sha256=binding.deployment_manifest_sha256,
                workload=workload,
                heartbeat_path=heartbeat,
                output_path=output_path,
                record_path=record,
                resource_identity_sha256=binding.resource_identity_sha256,
            )
            heartbeat_state = _heartbeat_fingerprint(heartbeat)
            journal.append(
                "lifecycle_started",
                {
                    "packet_sha256": effective_policy.packet_sha256,
                    "deployment_manifest_sha256": binding.deployment_manifest_sha256,
                    "lifecycle_launch_sha256": binding.lifecycle_launch_sha256,
                    "runtime_binding_sha256": binding.runtime_binding_sha256,
                    "active_security_manifest_sha256": (
                        binding.active_security_manifest_sha256),
                    "resource_identity_sha256": binding.resource_identity_sha256,
                    "campaign_id": binding.campaign_id,
                    "campaign_root_sha256": binding.campaign_root_sha256,
                    "pair_declaration_sha256": binding.pair_declaration_sha256,
                    "launch_id": binding.launch_id,
                    "launch_issued_at": binding.launch_issued_at,
                    "launch_expires_at": binding.launch_expires_at,
                    "arm_id": arm_id,
                    "control_sha256": dict(sorted(effective_policy.control_sha256.items())),
                    "workload_argv_sha256": workload.sha256,
                    "workload_argv_bytes": workload.bytes,
                    "teardown_argv_sha256": teardown.sha256,
                    "teardown_argv_bytes": teardown.bytes,
                    "absence_verifier_argv_sha256": absence_verifier.sha256,
                    "absence_verifier_argv_bytes": absence_verifier.bytes,
                    "launch_context_sha256": context_sha256,
                    "rate_role": effective_policy.rate_role,
                    "ttl_seconds": str(effective_policy.ttl_seconds),
                    "deadman_seconds": str(effective_policy.deadman_seconds),
                    "budget_check_seconds": str(effective_policy.budget_check_seconds),
                    "hourly_rate_usd": _money(effective_policy.hourly_rate_usd),
                    "fixed_cost_usd": _money(effective_policy.fixed_cost_usd),
                    "prior_cost_usd": _money(effective_policy.prior_cost_usd),
                    "budget_cap_usd": _money(effective_policy.budget_cap_usd),
                    "shell": False,
                    "child_output_captured": False,
                },
            )
            base_cost = _estimated_cost(effective_policy, 0)
            cleanup_reserve_seconds = (
                effective_policy.termination_grace_seconds
                + effective_policy.teardown_timeout_seconds * 2
            )
            cleanup_reserve_cost = (
                effective_policy.hourly_rate_usd
                * cleanup_reserve_seconds / Decimal(3600)
            )
            if base_cost + cleanup_reserve_cost >= effective_policy.budget_cap_usd:
                reason = "budget_exceeded"
                journal.append("budget_violation", {
                    "estimated_cost_usd": _money(base_cost),
                    "cleanup_reserve_cost_usd": _money(cleanup_reserve_cost),
                    "budget_cap_usd": _money(effective_policy.budget_cap_usd),
                    "phase": "pre-spawn",
                })
            else:
                child_environment = dict(os.environ)
                child_environment[LAUNCH_CONTEXT_ENV] = str(context_path)
                child_environment[LAUNCH_TOKEN_ENV] = token
                try:
                    workload_process = _spawn(
                        workload.argv, workdir, environment=child_environment,
                    )
                except (OSError, ValueError):
                    reason = "workload_spawn_failed"
                    journal.append("workload_spawn_failed", {})
                else:
                    journal.append("workload_started", {"pid": workload_process.pid})
                    ttl_ns = _seconds_to_ns(effective_policy.ttl_seconds)
                    deadman_ns = _seconds_to_ns(effective_policy.deadman_seconds)
                    interval_ns = _seconds_to_ns(effective_policy.budget_check_seconds)
                    remaining = effective_policy.budget_cap_usd - base_cost
                    raw_budget_deadline_ns = int(
                        remaining * Decimal(3600) * Decimal(1_000_000_000)
                        / effective_policy.hourly_rate_usd
                    )
                    cleanup_reserve_ns = _seconds_to_ns(cleanup_reserve_seconds)
                    budget_deadline_ns = max(0, raw_budget_deadline_ns - cleanup_reserve_ns)
                    next_budget_ns = start_ns
                    with _SignalLatch() as signals:
                        while True:
                            now_ns = time.monotonic_ns()
                            elapsed_ns = max(0, now_ns - start_ns)
                            code = workload_process.poll()
                            if code is not None:
                                workload_code = code
                                reason = "success" if code == 0 else "workload_failed"
                                journal.append("workload_exited", {"exit_code": code})
                                break
                            observed_signal = signals.signum or outer_signals.signum
                            if observed_signal is not None:
                                signum = observed_signal
                                reason = "signal"
                                journal.append("signal_received", {"signal": signum})
                                break
                            try:
                                current_heartbeat = _heartbeat_fingerprint(heartbeat)
                            except LifecycleError:
                                reason = "heartbeat_invalid"
                                journal.append("heartbeat_invalid", {})
                                break
                            if current_heartbeat != heartbeat_state:
                                heartbeat_state = current_heartbeat
                                if current_heartbeat is not None:
                                    heartbeat_details: dict[str, Any] = {}
                                    if not test_only_allow_legacy_local:
                                        if binding.heartbeat_public_key_path is None:
                                            raise LifecycleError(
                                                "production heartbeat key is missing")
                                        try:
                                            verified_heartbeat = \
                                                verify_signed_lifecycle_heartbeat(
                                                    heartbeat,
                                                    public_key_path=(
                                                        binding.heartbeat_public_key_path),
                                                    launch_id=binding.launch_id,
                                                    launch_nonce_sha256=(
                                                        binding.launch_nonce_sha256),
                                                    protection_mode=binding.arm_id,
                                                    instance_id=str(
                                                        binding.resource_identity[
                                                            "instance_id"]),
                                                    guest_workload_argv_sha256=(
                                                        binding.guest_workload_argv_sha256),
                                                    expected_sequence=(
                                                        heartbeat_sequence),
                                                    expected_previous_sha256=(
                                                        previous_heartbeat_sha256),
                                                    issued_at=binding.launch_issued_at,
                                                    expires_at=binding.launch_expires_at,
                                                )
                                        except LifecycleError:
                                            reason = "heartbeat_invalid"
                                            journal.append("heartbeat_invalid", {})
                                            break
                                        heartbeat_sequence += 1
                                        previous_heartbeat_sha256 = \
                                            verified_heartbeat[
                                                "envelope_sha256"]
                                        heartbeat_details = {
                                            "sequence": verified_heartbeat[
                                                "sequence"],
                                            "stage": verified_heartbeat["stage"],
                                            "step_id": verified_heartbeat["step_id"],
                                            "envelope_sha256": (
                                                previous_heartbeat_sha256),
                                        }
                                    last_heartbeat_ns = now_ns
                                    journal.append(
                                        "heartbeat_observed", heartbeat_details)
                            if elapsed_ns >= ttl_ns:
                                reason = "ttl_exceeded"
                                journal.append("ttl_violation", {
                                    "ttl_seconds": str(effective_policy.ttl_seconds),
                                })
                                break
                            if now_ns - last_heartbeat_ns >= deadman_ns:
                                reason = "heartbeat_deadman_exceeded"
                                journal.append("deadman_violation", {
                                    "deadman_seconds": str(effective_policy.deadman_seconds),
                                })
                                break
                            if elapsed_ns >= budget_deadline_ns:
                                reason = "budget_exceeded"
                                journal.append("budget_violation", {
                                    "estimated_cost_usd": _money(
                                        _estimated_cost(effective_policy, elapsed_ns)),
                                    "budget_cap_usd": _money(
                                        effective_policy.budget_cap_usd),
                                    "phase": "predictive-deadline",
                                })
                                break
                            if now_ns >= next_budget_ns:
                                current_cost = _estimated_cost(effective_policy, elapsed_ns)
                                with_cleanup = current_cost + cleanup_reserve_cost
                                journal.append("budget_checked", {
                                    "estimated_cost_usd": _money(current_cost),
                                    "estimated_with_cleanup_usd": _money(with_cleanup),
                                    "budget_cap_usd": _money(
                                        effective_policy.budget_cap_usd),
                                })
                                if with_cleanup >= effective_policy.budget_cap_usd:
                                    reason = "budget_exceeded"
                                    journal.append("budget_violation", {
                                        "estimated_cost_usd": _money(current_cost),
                                        "estimated_with_cleanup_usd": _money(with_cleanup),
                                        "budget_cap_usd": _money(
                                            effective_policy.budget_cap_usd),
                                        "phase": "periodic-check",
                                    })
                                    break
                                next_budget_ns = now_ns + interval_ns
                            wake_ns = min(
                                start_ns + ttl_ns,
                                last_heartbeat_ns + deadman_ns,
                                start_ns + budget_deadline_ns,
                                next_budget_ns,
                                now_ns + 50_000_000,
                            )
                            if wake_ns > now_ns:
                                time.sleep((wake_ns - now_ns) / 1_000_000_000)
        except BaseException as exc:
            reason = "supervisor_error"
            _safe_journal(journal, "supervisor_error", {
                "exception_type": type(exc).__name__,
            })
        finally:
            if workload_process is not None:
                try:
                    if workload_process.poll() is None:
                        _stop_process(
                            workload_process,
                            effective_policy.termination_grace_seconds,
                            journal,
                            label="workload",
                        )
                except BaseException as exc:
                    _safe_journal(journal, "workload_stop_error", {
                        "exception_type": type(exc).__name__,
                    })
                finally:
                    if workload_code is None:
                        workload_code = workload_process.poll()

            try:
                workload_outputs = _capture_workload_outputs(output_path, workload_code)
            except LifecycleError:
                workload_outputs = {"status": "invalid", "artifacts": {}}
                if reason == "success":
                    reason = "workload_outputs_invalid"
                _safe_journal(journal, "workload_outputs_invalid", {})
            else:
                _safe_journal(journal, "workload_outputs_captured", {
                    "status": workload_outputs["status"],
                    "artifact_sha256": {
                        name: value["sha256"]
                        for name, value in workload_outputs["artifacts"].items()
                    },
                })

            teardown_started_ns = time.time_ns()
            _safe_journal(journal, "teardown_started", {"shell": False})
            try:
                try:
                    teardown_process = _spawn(teardown.argv, workdir)
                except (OSError, ValueError):
                    teardown_status = "spawn-failed"
                else:
                    try:
                        teardown_code = teardown_process.wait(
                            timeout=float(effective_policy.teardown_timeout_seconds))
                        teardown_status = "succeeded" if teardown_code == 0 else "failed"
                    except subprocess.TimeoutExpired:
                        teardown_status = "timed-out"
                        _stop_process(
                            teardown_process,
                            effective_policy.termination_grace_seconds,
                            journal,
                            label="teardown",
                        )
                        teardown_code = teardown_process.poll()
                _safe_journal(journal, "teardown_finished", {
                    "status": teardown_status, "exit_code": teardown_code,
                })
            except BaseException as exc:
                teardown_status = "supervisor-error"
                _safe_journal(journal, "teardown_finished", {
                    "status": teardown_status,
                    "exception_type": type(exc).__name__,
                })

            _safe_journal(journal, "absence_verification_started", {"shell": False})
            try:
                try:
                    verifier_process = _spawn(absence_verifier.argv, workdir)
                except (OSError, ValueError):
                    absence_status = "spawn-failed"
                else:
                    try:
                        verifier_code = verifier_process.wait(
                            timeout=float(effective_policy.teardown_timeout_seconds))
                        absence_status = "candidate" if verifier_code == 0 else "failed"
                    except subprocess.TimeoutExpired:
                        absence_status = "timed-out"
                        _stop_process(
                            verifier_process,
                            effective_policy.termination_grace_seconds,
                            journal,
                            label="absence_verifier",
                        )
                        verifier_code = verifier_process.poll()
                if verifier_code == 0:
                    absence_evidence = validate_absence_artifact(
                        absence_path,
                        packet_sha256=effective_policy.packet_sha256,
                        deployment_manifest_sha256=(
                            binding.deployment_manifest_sha256),
                        resource_identity=binding.resource_identity,
                        resource_identity_sha256=binding.resource_identity_sha256,
                        verifier_argv_sha256=absence_verifier.sha256,
                        lifecycle_launch_sha256=(
                            binding.lifecycle_launch_sha256),
                        launch_id=binding.launch_id,
                        launch_nonce_sha256=binding.launch_nonce_sha256,
                        not_before_ns=teardown_started_ns,
                        allow_legacy_test=test_only_allow_legacy_local,
                    )
                    absence_status = "verified"
            except BaseException:
                absence_status = "invalid"
                absence_evidence = None
            _safe_journal(journal, "absence_verification_finished", {
                "status": absence_status,
                "exit_code": verifier_code,
                "artifact_sha256": (
                    absence_evidence["sha256"] if absence_evidence else None),
            })

            if outer_signals.signum is not None and signum is None:
                signum = outer_signals.signum
                if reason == "success":
                    reason = "signal"
                    _safe_journal(journal, "signal_received", {
                        "signal": signum, "phase": "teardown",
                    })
            if reason == "success" and absence_evidence is None:
                reason = "resource_absence_unverified"
            end_ns = time.monotonic_ns()
            elapsed_ns = max(0, end_ns - start_ns)
            estimated = _estimated_cost(effective_policy, elapsed_ns)
            if journal.failed and reason == "success":
                reason = "journal_failure"
            exit_code = _result_exit_code(
                reason,
                workload_code,
                teardown_code,
                absence_evidence is not None,
                workload_outputs.get("status") == "captured",
                signum,
            )
            _safe_journal(journal, "lifecycle_finished", {
                "reason": reason,
                "workload_exit_code": workload_code,
                "teardown_status": teardown_status,
                "teardown_exit_code": teardown_code,
                "absence_status": absence_status,
                "absence_verifier_exit_code": verifier_code,
                "absence_artifact_sha256": (
                    absence_evidence["sha256"] if absence_evidence else None),
                "estimated_cost_usd": _money(estimated),
                "exit_code": exit_code,
            })
            try:
                journal.close()
            finally:
                outer_signals.__exit__(None, None, None)

        lifecycle_validation = validate_lifecycle_record(record)
        ledger_checkpoint = ledger.finish(
            run_id=run_id,
            arm_id=arm_id,
            rate_role=effective_policy.rate_role,
            total_cost_usd=estimated,
            lifecycle_record_sha256=lifecycle_validation["record_sha256"],
            absence_artifact_sha256=(
                absence_evidence["sha256"] if absence_evidence else None),
            exit_code=exit_code,
        )
        _write_lifecycle_receipt(
            receipt,
            run_id=run_id,
            arm_id=arm_id,
            packet_sha256=effective_policy.packet_sha256,
            deployment_manifest_sha256=binding.deployment_manifest_sha256,
            lifecycle_launch_sha256=binding.lifecycle_launch_sha256,
            campaign_root_sha256=binding.campaign_root_sha256,
            pair_declaration_sha256=binding.pair_declaration_sha256,
            runtime_binding_sha256=binding.runtime_binding_sha256,
            active_security_manifest_sha256=(
                binding.active_security_manifest_sha256),
            resource_identity_sha256=binding.resource_identity_sha256,
            launch_id=binding.launch_id,
            launch_issued_at=binding.launch_issued_at,
            launch_expires_at=binding.launch_expires_at,
            launch_context_path=context_path,
            launch_context_sha256=context_sha256,
            lifecycle_validation=lifecycle_validation,
            ledger_path=binding.ledger_path,
            ledger_checkpoint=ledger_checkpoint,
            workload_outputs=workload_outputs,
            absence_artifact=absence_evidence,
            exit_code=exit_code,
        )

    return LifecycleResult(
        run_id=run_id,
        reason=reason,
        workload_exit_code=workload_code,
        teardown_exit_code=teardown_code,
        elapsed_seconds=(Decimal(max(0, time.monotonic_ns() - start_ns))
                         / Decimal(1_000_000_000)),
        estimated_cost_usd=estimated,
        record_path=record,
        receipt_path=receipt,
        ledger_path=binding.ledger_path,
        absence_artifact_path=absence_path,
        exit_code=exit_code,
    )


def validate_lifecycle_record(path: str | Path) -> dict[str, Any]:
    """Verify the complete lifecycle journal and return a safe summary."""
    source = Path(path)
    metadata = _regular_file(source, label="lifecycle record")
    if metadata.st_size < 1:
        raise LifecycleError("lifecycle record is empty")
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise LifecycleError("lifecycle record is not readable UTF-8") from exc
    if not lines:
        raise LifecycleError("lifecycle record is empty")
    previous = ZERO_HASH
    run_id: str | None = None
    last_elapsed = -1
    events: list[str] = []
    records: list[Mapping[str, Any]] = []
    final: Mapping[str, Any] | None = None
    required_fields = {
        "schema",
        "run_id",
        "sequence",
        "event",
        "recorded_at",
        "monotonic_elapsed_ns",
        "previous_sha256",
        "details",
        "event_sha256",
    }
    for sequence, line in enumerate(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LifecycleError("lifecycle record contains invalid JSON") from exc
        if not isinstance(record, Mapping) or set(record) != required_fields:
            raise LifecycleError("lifecycle event fields differ")
        if record["schema"] != EVENT_SCHEMA or record["sequence"] != sequence:
            raise LifecycleError("lifecycle event schema or sequence differs")
        if not isinstance(record["run_id"], str) or not record["run_id"]:
            raise LifecycleError("lifecycle run ID is invalid")
        try:
            uuid.UUID(record["run_id"])
        except (ValueError, AttributeError) as exc:
            raise LifecycleError("lifecycle run ID is not a UUID") from exc
        if (
            not isinstance(record["recorded_at"], str)
            or not record["recorded_at"].endswith("Z")
        ):
            raise LifecycleError("lifecycle UTC timestamp is invalid")
        try:
            datetime.fromisoformat(record["recorded_at"][:-1] + "+00:00")
        except ValueError as exc:
            raise LifecycleError("lifecycle UTC timestamp is invalid") from exc
        if run_id is None:
            run_id = record["run_id"]
        elif record["run_id"] != run_id:
            raise LifecycleError("lifecycle run ID changed within the record")
        if record["previous_sha256"] != previous:
            raise LifecycleError("lifecycle hash chain is broken")
        elapsed = record["monotonic_elapsed_ns"]
        if type(elapsed) is not int or elapsed < last_elapsed:
            raise LifecycleError("lifecycle monotonic elapsed time regressed")
        last_elapsed = elapsed
        body = {key: value for key, value in record.items() if key != "event_sha256"}
        expected = _sha256_bytes(_canonical_bytes(body))
        if record["event_sha256"] != expected:
            raise LifecycleError("lifecycle event digest mismatch")
        previous = expected
        if not isinstance(record["event"], str) or not record["event"]:
            raise LifecycleError("lifecycle event name is invalid")
        if not isinstance(record["details"], Mapping):
            raise LifecycleError("lifecycle event details are invalid")
        events.append(record["event"])
        records.append(record)
        final = record
    if (
        events[0] != "lifecycle_started"
        or events.count("lifecycle_started") != 1
        or events[-1] != "lifecycle_finished"
        or events.count("lifecycle_finished") != 1
    ):
        raise LifecycleError("lifecycle record is truncated or lacks terminal events")
    for required in (
        "teardown_started", "teardown_finished",
        "absence_verification_started", "absence_verification_finished",
    ):
        if events.count(required) != 1:
            raise LifecycleError(f"lifecycle record requires exactly one {required} event")
    if events.index("teardown_started") >= events.index("teardown_finished"):
        raise LifecycleError("teardown event order differs")
    if not (
        events.index("teardown_finished")
        < events.index("absence_verification_started")
        < events.index("absence_verification_finished")
    ):
        raise LifecycleError("absence verification event order differs")
    start_keys = {
        "packet_sha256",
        "deployment_manifest_sha256",
        "lifecycle_launch_sha256",
        "runtime_binding_sha256",
        "active_security_manifest_sha256",
        "resource_identity_sha256",
        "campaign_id",
        "campaign_root_sha256",
        "pair_declaration_sha256",
        "launch_id",
        "launch_issued_at",
        "launch_expires_at",
        "arm_id",
        "control_sha256",
        "workload_argv_sha256",
        "workload_argv_bytes",
        "teardown_argv_sha256",
        "teardown_argv_bytes",
        "absence_verifier_argv_sha256",
        "absence_verifier_argv_bytes",
        "launch_context_sha256",
        "rate_role",
        "ttl_seconds",
        "deadman_seconds",
        "budget_check_seconds",
        "hourly_rate_usd",
        "fixed_cost_usd",
        "prior_cost_usd",
        "budget_cap_usd",
        "shell",
        "child_output_captured",
    }
    if set(records[0]["details"]) != start_keys:
        raise LifecycleError("lifecycle start bindings differ")
    if records[0]["details"]["shell"] is not False \
            or records[0]["details"]["child_output_captured"] is not False:
        raise LifecycleError("lifecycle command isolation differs")
    for key in (
        "packet_sha256", "deployment_manifest_sha256",
        "lifecycle_launch_sha256", "runtime_binding_sha256",
        "active_security_manifest_sha256",
        "campaign_root_sha256", "pair_declaration_sha256",
        "resource_identity_sha256", "workload_argv_sha256",
        "teardown_argv_sha256", "absence_verifier_argv_sha256",
        "launch_context_sha256",
    ):
        if not _is_sha256(records[0]["details"].get(key)):
            raise LifecycleError(f"lifecycle start {key} is invalid")
    try:
        uuid.UUID(records[0]["details"].get("launch_id"))
        launch_issued = datetime.fromisoformat(
            records[0]["details"]["launch_issued_at"].replace("Z", "+00:00"))
        launch_expires = datetime.fromisoformat(
            records[0]["details"]["launch_expires_at"].replace("Z", "+00:00"))
        recorded = datetime.fromisoformat(
            records[0]["recorded_at"].replace("Z", "+00:00"))
    except (TypeError, ValueError, AttributeError) as exc:
        raise LifecycleError("lifecycle launch timing evidence is invalid") from exc
    if not (launch_issued - timedelta(seconds=30) <= recorded <= launch_expires):
        raise LifecycleError("lifecycle started outside its signed validity window")
    teardown_finished = records[events.index("teardown_finished")]["details"]
    teardown_status = teardown_finished.get("status")
    teardown_code = teardown_finished.get("exit_code")
    if teardown_status == "succeeded":
        if teardown_code != 0:
            raise LifecycleError("successful teardown must have exit code zero")
    elif teardown_status == "failed":
        if type(teardown_code) is not int or teardown_code == 0:
            raise LifecycleError("failed teardown must have a nonzero exit code")
    elif teardown_status in {"spawn-failed", "timed-out", "supervisor-error"}:
        pass
    else:
        raise LifecycleError("teardown status is invalid")
    absence_finished = records[events.index("absence_verification_finished")]["details"]
    if set(absence_finished) != {"status", "exit_code", "artifact_sha256"}:
        raise LifecycleError("absence verification summary fields differ")
    absence_status = absence_finished.get("status")
    verifier_code = absence_finished.get("exit_code")
    absence_sha256 = absence_finished.get("artifact_sha256")
    if absence_status == "verified":
        if verifier_code != 0 or not _is_sha256(absence_sha256):
            raise LifecycleError("verified absence lacks successful bound evidence")
    elif absence_status in {
        "failed", "spawn-failed", "timed-out", "invalid", "candidate",
    }:
        if absence_sha256 is not None:
            raise LifecycleError("unverified absence cannot bind passing evidence")
    else:
        raise LifecycleError("absence verification status is invalid")
    assert final is not None and run_id is not None
    expected_final_keys = {
        "reason", "workload_exit_code", "teardown_status",
        "teardown_exit_code", "absence_status", "absence_verifier_exit_code",
        "absence_artifact_sha256", "estimated_cost_usd", "exit_code",
    }
    if set(final["details"]) != expected_final_keys:
        raise LifecycleError("final lifecycle summary fields differ")
    final_exit = final["details"].get("exit_code")
    if type(final_exit) is not int or not 0 <= final_exit <= 255:
        raise LifecycleError("final lifecycle exit code is invalid")
    if teardown_status != "succeeded" and final_exit == 0:
        raise LifecycleError("failed teardown cannot have a successful lifecycle exit")
    if absence_status != "verified" and final_exit == 0:
        raise LifecycleError("unverified resource absence cannot have a successful exit")
    if (
        final["details"].get("teardown_status") != teardown_status
        or final["details"].get("teardown_exit_code") != teardown_code
        or final["details"].get("absence_status") != absence_status
        or final["details"].get("absence_verifier_exit_code") != verifier_code
        or final["details"].get("absence_artifact_sha256") != absence_sha256
    ):
        raise LifecycleError("final teardown/absence summary differs from its events")
    return {
        "schema": "p4.cc_lifecycle_record_validation.v1",
        "valid": True,
        "run_id": run_id,
        "event_count": len(events),
        "chain_head_sha256": previous,
        "record_sha256": _sha256_file(source),
        "path": str(source.resolve()),
        "reason": final["details"].get("reason"),
        "teardown_status": final["details"].get("teardown_status"),
        "absence_status": final["details"].get("absence_status"),
        "exit_code": final_exit,
        "creates_cloud_resources": False,
    }
