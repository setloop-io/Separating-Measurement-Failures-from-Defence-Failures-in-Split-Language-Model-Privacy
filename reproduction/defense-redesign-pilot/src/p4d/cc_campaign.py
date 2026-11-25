"""Signed, byte-addressed cross-arm seal for one CC engineering campaign.

The pre-run pair declaration gives ``pair_manifest_sha256`` real bytes without
creating a hash cycle: it binds the immutable inputs and public keys that exist
before arrival.  The post-run campaign manifest then binds that declaration to
the CC-on and CC-off repetition sets, lifecycle receipts, and shared ledger.

This module never provisions hardware and never accepts a shell command.
"""
from __future__ import annotations

import base64
from datetime import datetime
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import tempfile
from typing import Any, Mapping

from .cc_admission import load_admission_record, load_arrival_record
from .cc_active_security import (
    require_active_security_manifest,
    validate_active_runtime_binding,
    validate_deployment_manifest,
)
from .cc_external_collector import load_collector_manifest
from .cc_io import (
    FileSnapshot,
    read_json_snapshot,
    read_snapshot,
    require_unchanged,
    write_json_exclusive,
)
from .cc_lifecycle import (
    LEDGER_SCHEMA,
    validate_lifecycle_receipt,
    verify_lifecycle_launch_envelope,
)
from .cc_workload import (
    PROTOCOL_ID,
    _validate_sealed_artifacts,
    seal_journals,
    validate_sealed_repetition_set,
    validate_workload_config,
)
from .hardware_preorder import ROOT, validate_packet


PAIR_SCHEMA = "p4-cc-pair-declaration-0.1"
CAMPAIGN_SCHEMA = "p4-cc-campaign-manifest-0.1"
AUTH_SCHEMA = "p4-cc-campaign-auth-0.1"
AUTH_PAYLOAD_SCHEMA = "p4-cc-campaign-auth-payload-0.1"
SHA256 = re.compile(r"^[a-f0-9]{64}$")
IDENTIFIER = re.compile(r"^[A-Za-z0-9_.:-]{3,128}$")
ZERO_SHA256 = "0" * 64
ARMS = ("cc-on", "cc-off")
CAMPAIGN_CLAIM_SCOPE = (
    "Authenticated cross-arm engineering inventory for one declared CC-off/CC-on "
    "pair; not scientific evidence, a privacy certificate, or proof of an "
    "unobserved provider or physical channel."
)


class CampaignSealError(ValueError):
    """The pair/campaign seal is malformed, incomplete, or byte-inconsistent."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise CampaignSealError(f"{label} must be a lowercase SHA-256")
    return value


def _require_identifier(value: Any, label: str) -> str:
    if (not isinstance(value, str) or IDENTIFIER.fullmatch(value) is None
            or SHA256.fullmatch(value) is not None):
        raise CampaignSealError(
            f"{label} must be a non-digest campaign identifier")
    return value


def _artifact(snapshot: FileSnapshot) -> dict[str, Any]:
    return {
        "path": str(snapshot.path),
        "sha256": snapshot.sha256,
        "bytes": snapshot.size,
    }


def _snapshot_binding(binding: Mapping[str, Any], label: str, *,
                      json_object: bool = False) -> tuple[Any, FileSnapshot]:
    if not isinstance(binding, Mapping) or set(binding) != {
        "path", "sha256", "bytes",
    }:
        raise CampaignSealError(f"{label} artifact binding fields differ")
    path = binding.get("path")
    if not isinstance(path, str) or not path:
        raise CampaignSealError(f"{label} artifact path is invalid")
    _require_sha256(binding.get("sha256"), f"{label} artifact digest")
    if type(binding.get("bytes")) is not int or binding["bytes"] < 1:
        raise CampaignSealError(f"{label} artifact size is invalid")
    if json_object:
        document, snapshot = read_json_snapshot(
            path, expected_sha256=binding["sha256"])
        value: Any = document
    else:
        snapshot = read_snapshot(path, expected_sha256=binding["sha256"])
        value = snapshot.data
    if snapshot.size != binding["bytes"]:
        raise CampaignSealError(f"{label} artifact size differs")
    return value, snapshot


def _require_ed25519_public_key(path: str | Path) -> str:
    snapshot = read_snapshot(path, max_bytes=1024 * 1024)
    try:
        result = subprocess.run(
            ["openssl", "pkey", "-pubin", "-in", str(snapshot.path),
             "-text_pub", "-noout"],
            capture_output=True, check=False, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CampaignSealError("cannot inspect campaign public key") from exc
    first = result.stdout.splitlines()[0] if result.stdout else b""
    if result.returncode != 0 or first.strip() != b"ED25519 Public-Key:":
        raise CampaignSealError("campaign public key must be Ed25519")
    require_unchanged(snapshot)
    return snapshot.sha256


def _require_ed25519_keypair(private_key_path: str | Path,
                             public_key_path: str | Path) -> None:
    private = read_snapshot(private_key_path, max_bytes=1024 * 1024)
    public_sha256 = _require_ed25519_public_key(public_key_path)
    if stat.S_IMODE(private.path.stat().st_mode) & 0o077:
        raise CampaignSealError(
            "campaign private key must not be group/world accessible")
    with tempfile.TemporaryDirectory(prefix="p4-cc-campaign-key-") as temporary:
        derived = Path(temporary) / "derived-public.pem"
        try:
            result = subprocess.run(
                ["openssl", "pkey", "-in", str(private.path), "-pubout",
                 "-out", str(derived)],
                capture_output=True, check=False, timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise CampaignSealError("cannot inspect campaign keypair") from exc
        if (result.returncode != 0 or not derived.is_file()
                or read_snapshot(derived).sha256 != public_sha256):
            raise CampaignSealError(
                "campaign keys do not form an Ed25519 keypair")
    require_unchanged(private)


def _sign_auth(document_path: str | Path, output_path: str | Path, *,
               document_schema: str, campaign_id: str, pair_id: str,
               private_key_path: str | Path,
               public_key_path: str | Path) -> Path:
    _require_ed25519_keypair(private_key_path, public_key_path)
    document = read_snapshot(document_path)
    public_sha256 = _require_ed25519_public_key(public_key_path)
    payload = {
        "schema_version": AUTH_PAYLOAD_SCHEMA,
        "document_schema": document_schema,
        "document_sha256": document.sha256,
        "campaign_id": _require_identifier(campaign_id, "campaign_id"),
        "pair_id": _require_identifier(pair_id, "pair_id"),
        "public_key_sha256": public_sha256,
        "signature_algorithm": "Ed25519",
    }
    encoded = _canonical_bytes(payload)
    with tempfile.TemporaryDirectory(prefix="p4-cc-campaign-sign-") as temporary:
        payload_path = Path(temporary) / "payload.bin"
        signature_path = Path(temporary) / "signature.bin"
        payload_path.write_bytes(encoded)
        try:
            result = subprocess.run(
                ["openssl", "pkeyutl", "-sign", "-inkey",
                 str(private_key_path), "-rawin", "-in", str(payload_path),
                 "-out", str(signature_path)],
                capture_output=True, check=False, timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise CampaignSealError("campaign signing failed") from exc
        if result.returncode != 0 or not signature_path.is_file():
            raise CampaignSealError("campaign signing failed")
        signature = signature_path.read_bytes()
    record = {
        "schema_version": AUTH_SCHEMA,
        "payload": payload,
        "payload_sha256": _sha256_bytes(encoded),
        "signature_base64": base64.b64encode(signature).decode("ascii"),
    }
    path = write_json_exclusive(output_path, record)
    os.chmod(path, 0o600)
    require_unchanged(document)
    return path


def _verify_auth(auth_path: str | Path, document_path: str | Path,
                 public_key_path: str | Path, *, document_schema: str,
                 campaign_id: str, pair_id: str) -> dict[str, Any]:
    public_sha256 = _require_ed25519_public_key(public_key_path)
    document = read_snapshot(document_path)
    auth, auth_snapshot = read_json_snapshot(auth_path)
    expected_payload = {
        "schema_version": AUTH_PAYLOAD_SCHEMA,
        "document_schema": document_schema,
        "document_sha256": document.sha256,
        "campaign_id": campaign_id,
        "pair_id": pair_id,
        "public_key_sha256": public_sha256,
        "signature_algorithm": "Ed25519",
    }
    if (set(auth) != {"schema_version", "payload", "payload_sha256",
                      "signature_base64"}
            or auth.get("schema_version") != AUTH_SCHEMA
            or auth.get("payload") != expected_payload):
        raise CampaignSealError("campaign authentication binding differs")
    encoded = _canonical_bytes(expected_payload)
    if auth.get("payload_sha256") != _sha256_bytes(encoded):
        raise CampaignSealError("campaign authentication payload digest differs")
    try:
        signature = base64.b64decode(
            auth.get("signature_base64", ""), validate=True)
    except (TypeError, ValueError) as exc:
        raise CampaignSealError(
            "campaign signature is not canonical base64") from exc
    if len(signature) != 64:
        raise CampaignSealError("campaign Ed25519 signature must be 64 bytes")
    with tempfile.TemporaryDirectory(prefix="p4-cc-campaign-verify-") as temporary:
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
            raise CampaignSealError(
                "campaign signature verification failed") from exc
    if result.returncode != 0:
        raise CampaignSealError("campaign signature verification failed")
    require_unchanged(document)
    require_unchanged(auth_snapshot)
    return auth


def _validate_static_inputs(inputs: Mapping[str, Any]) -> tuple[
        dict[str, Any], dict[str, FileSnapshot]]:
    expected = {
        "deployment_manifest", "hardware_packet", "workload_config",
        "checkpoint_manifest", "checkpoint_artifact",
    }
    if not isinstance(inputs, Mapping) or set(inputs) != expected:
        raise CampaignSealError("pair immutable input inventory differs")
    documents: dict[str, Any] = {}
    snapshots: dict[str, FileSnapshot] = {}
    for name in expected:
        value, snapshot = _snapshot_binding(
            inputs[name], name, json_object=name != "checkpoint_artifact")
        documents[name] = value
        snapshots[name] = snapshot
    deployment = documents["deployment_manifest"]
    adapter = deployment.get("adapter") if isinstance(deployment, Mapping) else None
    validate_deployment_manifest(
        deployment, snapshots["deployment_manifest"].sha256,
        deployment.get("profile") if isinstance(deployment, Mapping) else None,
        adapter.get("command_argv", []) if isinstance(adapter, Mapping) else [],
    )
    packet = documents["hardware_packet"]
    validate_packet(packet)
    if packet.get("status") != "frozen":
        raise CampaignSealError("campaign hardware packet is not frozen")
    config = documents["workload_config"]
    validate_workload_config(config, execute=True)
    bindings = config["bindings"]
    for name in ("checkpoint_manifest", "checkpoint_artifact"):
        if bindings.get(f"{name}_sha256") != snapshots[name].sha256:
            raise CampaignSealError(
                f"workload config {name} binding differs")
    return documents, snapshots


def _pairing_contract(packet: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the static two-resource policy without inventing instance IDs."""
    order = packet.get("order_parameters")
    comparator = packet.get("cc_off_comparator")
    if not isinstance(order, Mapping) or not isinstance(comparator, Mapping):
        raise CampaignSealError("packet pairing policy is missing")
    strategy = comparator.get("strategy")
    if strategy != "matched-separate-instance":
        raise CampaignSealError(
            "campaign v1 requires matched-separate-instance lifecycle")
    provider_slugs = {"Microsoft Azure": "azure", "Google Cloud": "gcp"}
    arm_contracts = {
        "cc-off": {
            "provider": provider_slugs.get(comparator.get("provider")),
            "provider_control_plane": comparator.get("provider"),
            "candidate_id": comparator.get("candidate_id"),
            "sku": comparator.get("sku_or_mode"),
            "region": comparator.get("region"),
            "zone": comparator.get("zone"),
            "image_reference": comparator.get("image_reference"),
            "gpu_name": comparator.get("gpu_model"),
            "driver": comparator.get("driver_version"),
            "cpu_model": comparator.get("cpu_family"),
        },
        "cc-on": {
            "provider": provider_slugs.get(order.get("provider")),
            "provider_control_plane": order.get("provider"),
            "candidate_id": order.get("candidate_id"),
            "sku": order.get("sku"),
            "region": order.get("region"),
            "zone": order.get("zone"),
            "image_reference": order.get("immutable_image_reference"),
            "gpu_name": order.get("gpu_model"),
            "driver": order.get("pinned_driver_version"),
            "cpu_model": order.get("cpu_family"),
        },
    }
    if any(not isinstance(value, str) or not value.strip()
           for contract in arm_contracts.values()
           for value in contract.values()):
        raise CampaignSealError("packet per-arm resource policy is not frozen")
    matched = comparator.get("required_matched_fields")
    if not isinstance(matched, list) or len(matched) != len(set(matched)):
        raise CampaignSealError("packet comparator match policy is invalid")
    return {
        "strategy": strategy,
        "same_physical_host_guaranteed": comparator.get(
            "same_physical_host_guaranteed"),
        "required_matched_fields": list(matched),
        "arms": arm_contracts,
    }


def _require_packet_public_key_binding(
    packet: Mapping[str, Any], *, name: str, snapshot: FileSnapshot,
) -> None:
    fields = {
        "campaign": ("campaign_public_key_path", "campaign_public_key_sha256"),
        "lifecycle_controller": (
            "lifecycle_controller_public_key_path",
            "lifecycle_controller_public_key_sha256",
        ),
    }
    path_key, digest_key = fields[name]
    bindings = packet.get("bindings")
    if not isinstance(bindings, Mapping):
        raise CampaignSealError("hardware packet key bindings are missing")
    value = bindings.get(path_key)
    if not isinstance(value, str) or not value:
        raise CampaignSealError(f"packet {name} public-key path is missing")
    declared = Path(value)
    if not declared.is_absolute():
        declared = ROOT / declared
    if (declared.resolve() != snapshot.path
            or bindings.get(digest_key) != snapshot.sha256):
        raise CampaignSealError(
            f"packet {name} public-key binding differs")


def create_pair_declaration(
    output_path: str | Path, auth_output_path: str | Path, *,
    campaign_id: str, pair_id: str, deployment_manifest_path: str | Path,
    hardware_packet_path: str | Path, workload_config_path: str | Path,
    checkpoint_manifest_path: str | Path, checkpoint_artifact_path: str | Path,
    cc_off_journal_public_key_path: str | Path,
    cc_on_journal_public_key_path: str | Path,
    lifecycle_controller_public_key_path: str | Path,
    campaign_private_key_path: str | Path,
    campaign_public_key_path: str | Path,
) -> tuple[Path, Path]:
    """Freeze and sign the acyclic declaration used by both arrivals."""
    campaign_id = _require_identifier(campaign_id, "campaign_id")
    pair_id = _require_identifier(pair_id, "pair_id")
    source_paths = {
        "deployment_manifest": deployment_manifest_path,
        "hardware_packet": hardware_packet_path,
        "workload_config": workload_config_path,
        "checkpoint_manifest": checkpoint_manifest_path,
        "checkpoint_artifact": checkpoint_artifact_path,
    }
    inputs = {
        name: _artifact(read_snapshot(path)) for name, path in source_paths.items()
    }
    documents, snapshots = _validate_static_inputs(inputs)
    key_paths = {
        "campaign": campaign_public_key_path,
        "lifecycle_controller": lifecycle_controller_public_key_path,
        "cc-off": cc_off_journal_public_key_path,
        "cc-on": cc_on_journal_public_key_path,
    }
    keys = {
        name: _artifact(read_snapshot(path, max_bytes=1024 * 1024))
        for name, path in key_paths.items()
    }
    key_hashes = {name: _require_ed25519_public_key(path)
                  for name, path in key_paths.items()}
    if len(set(key_hashes.values())) != len(key_hashes):
        raise CampaignSealError(
            "campaign, lifecycle, and per-arm signing keys must be distinct")
    for name in ("campaign", "lifecycle_controller"):
        _require_packet_public_key_binding(
            documents["hardware_packet"], name=name,
            snapshot=read_snapshot(key_paths[name], max_bytes=1024 * 1024),
        )
    document = {
        "schema_version": PAIR_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "campaign_id": campaign_id,
        "pair_id": pair_id,
        "scientific_evidence": False,
        "protection_modes": list(ARMS),
        "execution_order": list(ARMS),
        "repetitions_per_arm": 3,
        "pairing_contract": _pairing_contract(documents["hardware_packet"]),
        "immutable_inputs": inputs,
        "public_keys": keys,
        "claim_scope": CAMPAIGN_CLAIM_SCOPE,
    }
    path = write_json_exclusive(output_path, document)
    os.chmod(path, 0o600)
    auth_path = _sign_auth(
        path, auth_output_path, document_schema=PAIR_SCHEMA,
        campaign_id=campaign_id, pair_id=pair_id,
        private_key_path=campaign_private_key_path,
        public_key_path=campaign_public_key_path,
    )
    validate_pair_declaration(path, auth_path, campaign_public_key_path)
    for snapshot in snapshots.values():
        require_unchanged(snapshot)
    return path, auth_path


def validate_pair_declaration(
    declaration_path: str | Path, auth_path: str | Path,
    campaign_public_key_path: str | Path,
) -> dict[str, Any]:
    declaration, declaration_snapshot = read_json_snapshot(declaration_path)
    expected = {
        "schema_version", "protocol_id", "campaign_id", "pair_id",
        "scientific_evidence", "protection_modes", "repetitions_per_arm",
        "execution_order", "pairing_contract", "immutable_inputs",
        "public_keys", "claim_scope",
    }
    if (set(declaration) != expected
            or declaration.get("schema_version") != PAIR_SCHEMA
            or declaration.get("protocol_id") != PROTOCOL_ID
            or declaration.get("scientific_evidence") is not False
            or declaration.get("protection_modes") != list(ARMS)
            or declaration.get("execution_order") != list(ARMS)
            or declaration.get("repetitions_per_arm") != 3
            or declaration.get("claim_scope") != CAMPAIGN_CLAIM_SCOPE):
        raise CampaignSealError("pair declaration fields differ")
    campaign_id = _require_identifier(
        declaration.get("campaign_id"), "campaign_id")
    pair_id = _require_identifier(declaration.get("pair_id"), "pair_id")
    documents, snapshots = _validate_static_inputs(declaration["immutable_inputs"])
    if declaration.get("pairing_contract") != _pairing_contract(
            documents["hardware_packet"]):
        raise CampaignSealError("pair declaration resource policy differs")
    keys = declaration.get("public_keys")
    if not isinstance(keys, Mapping) or set(keys) != {
        "campaign", "lifecycle_controller", "cc-off", "cc-on",
    }:
        raise CampaignSealError("pair public-key inventory differs")
    key_hashes: dict[str, str] = {}
    key_snapshots: list[FileSnapshot] = []
    for name, binding in keys.items():
        _, snapshot = _snapshot_binding(binding, f"{name} public key")
        key_hashes[name] = _require_ed25519_public_key(snapshot.path)
        key_snapshots.append(snapshot)
    if len(set(key_hashes.values())) != len(key_hashes):
        raise CampaignSealError(
            "campaign, lifecycle, and per-arm signing keys must be distinct")
    for name in ("campaign", "lifecycle_controller"):
        _require_packet_public_key_binding(
            documents["hardware_packet"], name=name,
            snapshot=key_snapshots[list(keys).index(name)],
        )
    if key_hashes["campaign"] != _require_ed25519_public_key(
            campaign_public_key_path):
        raise CampaignSealError("pair campaign public-key binding differs")
    _verify_auth(
        auth_path, declaration_path, campaign_public_key_path,
        document_schema=PAIR_SCHEMA, campaign_id=campaign_id, pair_id=pair_id,
    )
    require_unchanged(declaration_snapshot)
    for snapshot in (*snapshots.values(), *key_snapshots):
        require_unchanged(snapshot)
    return declaration


def _safe_relative(root: Path, value: Any, label: str) -> Path:
    relative = Path(str(value))
    if relative.is_absolute() or ".." in relative.parts or "." in relative.parts:
        raise CampaignSealError(f"{label} path is unsafe")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise CampaignSealError(f"{label} path escapes its arm root") from exc
    return resolved


def _arrival_resource_identity(arrival: Mapping[str, Any]) -> dict[str, Any]:
    deployment = arrival.get("deployment")
    gpu = arrival.get("gpu")
    cpu = arrival.get("cpu")
    if (not isinstance(deployment, Mapping) or not isinstance(gpu, Mapping)
            or not isinstance(cpu, Mapping)):
        raise CampaignSealError("arrival resource identity is missing")
    identity = {
        key: deployment.get(key) for key in (
            "provider", "candidate_id", "sku", "region", "zone",
            "instance_id", "boot_id_sha256", "image_reference",
            "protection_mode", "pair_id", "pair_manifest_sha256",
            "cuda_device_index", "deployment_manifest_sha256",
            "hardware_packet_sha256",
        )
    }
    identity.update({
        "gpu_uuid": gpu.get("uuid"),
        "gpu_pci_bus_id": gpu.get("pci_bus_id"),
        "gpu_name": gpu.get("name"),
        "driver": gpu.get("driver"),
        "cpu_model": cpu.get("model"),
    })
    if any(value is None or value == "" for value in identity.values()):
        raise CampaignSealError("arrival exact resource identity is incomplete")
    return identity


def _resource_record(arrival: Mapping[str, Any],
                     contract: Mapping[str, Any]) -> dict[str, Any]:
    arrival_identity = _arrival_resource_identity(arrival)
    expected = contract.get("arms", {}).get(
        arrival_identity["protection_mode"])
    if not isinstance(expected, Mapping):
        raise CampaignSealError("pair resource contract arm is missing")
    for key in (
        "provider", "candidate_id", "sku", "region", "zone",
        "image_reference", "gpu_name", "driver", "cpu_model",
    ):
        if arrival_identity.get(key) != expected.get(key):
            raise CampaignSealError(
                f"arrival resource differs from pair contract: {key}")
    lifecycle_identity = {
        "provider": arrival_identity["provider"],
        **{key: arrival_identity[key] for key in (
            "candidate_id", "sku", "region", "zone", "instance_id",
        )},
    }
    return {"arrival": arrival_identity, "lifecycle": lifecycle_identity}


def _load_argv_binding(binding: Mapping[str, Any], label: str) -> tuple[
        list[str], FileSnapshot]:
    _, snapshot = _snapshot_binding(binding, label)
    try:
        value = json.loads(snapshot.data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CampaignSealError(f"{label} is not valid JSON") from exc
    if (not isinstance(value, list) or not 1 <= len(value) <= 256
            or any(not isinstance(item, str) or not item or "\x00" in item
                   or len(item) > 4096 for item in value)):
        raise CampaignSealError(
            f"{label} must be a bounded nonempty shell-free argv array")
    return value, snapshot


def _launch_arm_value(value: Any, arm: str, *, strategy: str,
                      label: str) -> Any:
    if isinstance(value, Mapping) and set(value) == set(ARMS):
        return value.get(arm)
    if strategy != "same-instance-reboot-mode-toggle":
        raise CampaignSealError(
            f"matched-separate lifecycle requires per-arm {label}")
    return value


def _validate_arm_tree(
    arm: str, binding: Mapping[str, Any], *, pair_id: str,
    pair_declaration_sha256: str, expected_static: Mapping[str, str],
    expected_active_security_sha256: str,
    expected_runtime_binding_sha256: str,
    expected_lifecycle_launch_sha256: str,
    pairing_contract: Mapping[str, Any], production: bool,
) -> dict[str, Any]:
    expected_fields = {
        "protection_mode", "session_id", "admission_sha256", "evidence_kind",
        "evidence_record", "arrival_record", "resource_identity",
        "repetition_set", "repetition_set_auth", "journal_public_key",
        "collector_manifest", "collector_public_key", "lifecycle_receipt",
        "teardown_argv", "absence_verifier_argv",
    }
    if (not isinstance(binding, Mapping) or set(binding) != expected_fields
            or binding.get("protection_mode") != arm):
        raise CampaignSealError(f"{arm} campaign arm fields differ")
    session_id = _require_identifier(binding.get("session_id"), f"{arm} session_id")
    admission_sha256 = _require_sha256(
        binding.get("admission_sha256"), f"{arm} admission_sha256")
    documents: dict[str, Any] = {}
    snapshots: dict[str, FileSnapshot] = {}
    for name in expected_fields - {"protection_mode", "session_id",
                                  "admission_sha256", "evidence_kind",
                                  "resource_identity", "teardown_argv",
                                  "absence_verifier_argv"}:
        is_json = name not in {"journal_public_key", "collector_public_key"}
        value, snapshot = _snapshot_binding(
            binding[name], f"{arm} {name}", json_object=is_json)
        documents[name] = value
        snapshots[name] = snapshot
    expected_kind = "cc-on-admission" if arm == "cc-on" \
        else "cc-off-arrival-inventory"
    if binding.get("evidence_kind") != expected_kind:
        raise CampaignSealError(f"{arm} evidence kind differs")
    evidence = documents["evidence_record"]
    arrival = load_arrival_record(snapshots["arrival_record"].path)
    resource_identity = _resource_record(arrival, pairing_contract)
    if binding.get("resource_identity") != resource_identity:
        raise CampaignSealError(f"{arm} signed resource identity differs")
    deployment = arrival["deployment"]
    if (deployment.get("protection_mode") != arm
            or deployment.get("pair_id") != pair_id
            or deployment.get("pair_manifest_sha256")
            != pair_declaration_sha256
            or deployment.get("deployment_manifest_sha256")
            != expected_static["deployment_manifest"]
            or deployment.get("hardware_packet_sha256")
            != expected_static["hardware_packet"]):
        raise CampaignSealError(f"{arm} arrival pair/resource binding differs")
    if arm == "cc-on":
        admission = load_admission_record(snapshots["evidence_record"].path)
        if (admission != evidence or not admission.get("admitted")
                or admission.get("session_id") != session_id
                or admission.get("arrival_record_sha256")
                != snapshots["arrival_record"].sha256
                or admission.get("workload_binding", {}).get("config_sha256")
                != expected_static["workload_config"]
                or admission.get("workload_binding", {}).get(
                    "deployment_manifest_sha256")
                != expected_static["deployment_manifest"]
                or admission.get("workload_binding", {}).get(
                    "hardware_packet_sha256")
                != expected_static["hardware_packet"]):
            raise CampaignSealError("cc-on admission/arrival binding differs")
    elif (evidence != arrival
          or snapshots["evidence_record"].sha256
          != snapshots["arrival_record"].sha256):
        raise CampaignSealError(
            "cc-off evidence must be the exact arrival inventory bytes")
    if admission_sha256 != snapshots["evidence_record"].sha256:
        raise CampaignSealError(f"{arm} evidence digest differs")
    _, teardown_snapshot = _load_argv_binding(
        binding["teardown_argv"], f"{arm} teardown argv")
    _, absence_argv_snapshot = _load_argv_binding(
        binding["absence_verifier_argv"], f"{arm} absence verifier argv")
    snapshots["teardown_argv"] = teardown_snapshot
    snapshots["absence_verifier_argv"] = absence_argv_snapshot
    journal_key_sha256 = _require_ed25519_public_key(
        snapshots["journal_public_key"].path)
    collector_key_sha256 = _require_ed25519_public_key(
        snapshots["collector_public_key"].path)
    if journal_key_sha256 == collector_key_sha256:
        raise CampaignSealError(
            f"{arm} journal and collector keys must be distinct")
    collector = load_collector_manifest(
        snapshots["collector_manifest"].path, production=production,
        expected_public_key_path=snapshots["collector_public_key"].path,
    )
    repetition_set = validate_sealed_repetition_set(
        snapshots["repetition_set"].path,
        snapshots["repetition_set_auth"].path,
        snapshots["journal_public_key"].path,
        expected_session_id=session_id, expected_pair_id=pair_id,
        expected_protection_mode=arm,
        expected_admission_sha256=admission_sha256,
        expected_workload_config_sha256=expected_static["workload_config"],
        expected_hardware_packet_sha256=expected_static["hardware_packet"],
    )
    if repetition_set.get("active_security_manifest_sha256") \
            != expected_active_security_sha256:
        raise CampaignSealError(f"{arm} active-security binding differs")
    arm_root = snapshots["repetition_set"].path.parent.parent
    artifact_manifest_path = arm_root / "artifact_manifest.json"
    artifact_manifest, artifact_snapshot = read_json_snapshot(
        artifact_manifest_path,
        expected_sha256=repetition_set["artifact_manifest_sha256"],
    )
    if (set(artifact_manifest) != {"schema_version", "artifacts"}
            or artifact_manifest.get("schema_version")
            != "p4-cc-seal-artifact-manifest-0.1"
            or not isinstance(artifact_manifest.get("artifacts"), list)):
        raise CampaignSealError(f"{arm} artifact manifest fields differ")
    rows = repetition_set["repetitions"]
    first_bundle_path = _safe_relative(arm_root, rows[0]["path"],
                                       f"{arm} first bundle")
    first_bundle, first_bundle_snapshot = read_json_snapshot(first_bundle_path)
    admission_binding = first_bundle.get("admission")
    if not isinstance(admission_binding, Mapping):
        raise CampaignSealError(f"{arm} bundle admission binding is missing")
    if (admission_binding.get("journal_public_key_sha256") != journal_key_sha256
            or admission_binding.get("collector_manifest_sha256")
            != snapshots["collector_manifest"].sha256
            or admission_binding.get("collector_public_key_sha256")
            != collector_key_sha256
            or admission_binding.get("collector_executable_sha256")
            != collector["executable_sha256"]):
        raise CampaignSealError(f"{arm} bundle key/collector binding differs")
    artifacts = _validate_sealed_artifacts(
        artifact_manifest["artifacts"], arm_root, admission_binding, arm)
    by_role = {
        role: [arm_root / item["path"] for item in artifacts
               if item["role"] == role]
        for role in {item["role"] for item in artifacts}
    }
    summary, summary_snapshot = read_json_snapshot(
        by_role["execution_summary"][0])
    correctness, correctness_snapshot = read_json_snapshot(
        by_role["correctness_comparison"][0])
    source, source_snapshot = read_json_snapshot(by_role["source_snapshot"][0])
    correctness_binding = correctness.get("binding")
    if (not isinstance(correctness_binding, Mapping)
            or correctness_binding.get("pair_id") != pair_id
            or correctness_binding.get("pair_manifest_sha256")
            != pair_declaration_sha256
            or correctness_binding.get("config_sha256")
            != expected_static["workload_config"]
            or correctness_binding.get("checkpoint_sha256")
            != expected_static["checkpoint_artifact"]
            or correctness_binding.get("deployment_manifest_sha256")
            != expected_static["deployment_manifest"]):
        raise CampaignSealError(f"{arm} correctness pair/input binding differs")
    for name, expected in (
        ("workload_config_sha256", expected_static["workload_config"]),
        ("hardware_packet_sha256", expected_static["hardware_packet"]),
        ("deployment_manifest_sha256", expected_static["deployment_manifest"]),
        ("active_security_manifest_sha256", expected_active_security_sha256),
    ):
        if source.get(name) != expected:
            raise CampaignSealError(f"{arm} source snapshot {name} differs")
    if (summary.get("pair_id") != pair_id
            or summary.get("session_id") != session_id
            or summary.get("protection_mode") != arm):
        raise CampaignSealError(f"{arm} execution identity differs")
    repetitions = summary["repetitions"]
    for index, row in enumerate(rows):
        bundle_path = _safe_relative(arm_root, row["path"],
                                     f"{arm} repetition bundle")
        bundle, bundle_snapshot = read_json_snapshot(
            bundle_path, expected_sha256=row["sha256"])
        repetition = repetitions[index]
        rebuilt = seal_journals(
            _safe_relative(arm_root, repetition["trusted_journal"],
                           f"{arm} trusted journal"),
            _safe_relative(arm_root, repetition["external_journal"],
                           f"{arm} external journal"),
            _safe_relative(arm_root, repetition["trusted_journal_auth"],
                           f"{arm} trusted journal auth"),
            snapshots["journal_public_key"].path,
            _safe_relative(arm_root, repetition["external_journal_auth"],
                           f"{arm} external journal auth"),
            snapshots["collector_manifest"].path,
            snapshots["collector_public_key"].path,
            repetition["collector_journal_id"], admission_binding, arm,
            artifacts, arm_root, production_collector=production,
        )
        if _canonical_bytes(rebuilt) != _canonical_bytes(bundle):
            raise CampaignSealError(
                f"{arm} repetition {index} bundle is not reproducible")
        require_unchanged(bundle_snapshot)
    receipt_validation = validate_lifecycle_receipt(
        snapshots["lifecycle_receipt"].path)
    receipt = documents["lifecycle_receipt"]
    lifecycle_identity_sha256 = _sha256_bytes(_canonical_bytes(
        resource_identity["lifecycle"]))
    if (receipt_validation.get("arm_id") != arm
            or receipt.get("arm_id") != arm
            or receipt.get("exit_code") != 0
            or receipt.get("packet_sha256")
            != expected_static["hardware_packet"]
            or receipt.get("deployment_manifest_sha256")
            != expected_static["deployment_manifest"]
            or receipt.get("runtime_binding_sha256")
            != expected_runtime_binding_sha256
            or receipt.get("active_security_manifest_sha256")
            != expected_active_security_sha256
            or receipt.get("lifecycle_launch_sha256")
            != expected_lifecycle_launch_sha256
            or receipt.get("resource_identity_sha256")
            != lifecycle_identity_sha256):
        raise CampaignSealError(f"{arm} lifecycle receipt binding differs")
    lifecycle_record_path = Path(str(
        receipt.get("lifecycle_record", {}).get("path"))).resolve()
    lifecycle_snapshot = read_snapshot(lifecycle_record_path)
    try:
        first_event = json.loads(lifecycle_snapshot.data.splitlines()[0])
    except (IndexError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CampaignSealError(
            f"{arm} lifecycle start event is unreadable") from exc
    start = first_event.get("details") if isinstance(first_event, Mapping) else None
    if (not isinstance(start, Mapping)
            or start.get("campaign_id") is None
            or start.get("arm_id") != arm
            or start.get("resource_identity_sha256")
            != lifecycle_identity_sha256
            or start.get("teardown_argv_sha256") != teardown_snapshot.sha256
            or start.get("absence_verifier_argv_sha256")
            != absence_argv_snapshot.sha256
            or start.get("lifecycle_launch_sha256")
            != expected_lifecycle_launch_sha256
            or start.get("runtime_binding_sha256")
            != expected_runtime_binding_sha256
            or start.get("active_security_manifest_sha256")
            != expected_active_security_sha256):
        raise CampaignSealError(f"{arm} lifecycle start contract differs")
    outputs = receipt.get("workload_outputs", {}).get("artifacts", {})
    expected_outputs = {
        "execution_summary.json": summary_snapshot,
        "artifact_manifest.json": artifact_snapshot,
    }
    for name, snapshot in expected_outputs.items():
        record = outputs.get(name)
        if (not isinstance(record, Mapping)
                or Path(str(record.get("path"))).resolve() != snapshot.path
                or record.get("sha256") != snapshot.sha256
                or record.get("bytes") != snapshot.size):
            raise CampaignSealError(
                f"{arm} lifecycle receipt workload output differs: {name}")
    for snapshot in (
        *snapshots.values(), artifact_snapshot, first_bundle_snapshot,
        summary_snapshot, correctness_snapshot, source_snapshot,
        lifecycle_snapshot,
    ):
        require_unchanged(snapshot)
    return {
        "session_id": session_id,
        "admission_sha256": admission_sha256,
        "journal_public_key_sha256": journal_key_sha256,
        "collector_public_key_sha256": collector_key_sha256,
        "correctness": correctness,
        "source": source,
        "summary": summary,
        "receipt": receipt,
        "receipt_validation": receipt_validation,
        "resource_identity": resource_identity,
        "teardown_argv_sha256": teardown_snapshot.sha256,
        "absence_verifier_argv_sha256": absence_argv_snapshot.sha256,
        "lifecycle_start": dict(start),
    }


def _validate_ledger(
    binding: Mapping[str, Any], *, campaign_id: str, packet_sha256: str,
    deployment_sha256: str, campaign_root_sha256: str,
    pair_declaration_sha256: str,
    lifecycle_launch_sha256: Mapping[str, str],
    arm_results: Mapping[str, Mapping[str, Any]],
) -> FileSnapshot:
    ledger, snapshot = _snapshot_binding(
        binding, "campaign ledger", json_object=True)
    expected_fields = {
        "schema", "campaign_id", "packet_sha256",
        "deployment_manifest_sha256", "campaign_root_sha256",
        "pair_declaration_sha256", "budget_cap_usd", "accrued_cost_usd",
        "active_run", "entries", "ledger_sha256",
    }
    body = {key: value for key, value in ledger.items()
            if key != "ledger_sha256"}
    if (set(ledger) != expected_fields or ledger.get("schema") != LEDGER_SCHEMA
            or ledger.get("ledger_sha256") != _sha256_bytes(_canonical_bytes(body))
            or ledger.get("campaign_id") != campaign_id
            or ledger.get("packet_sha256") != packet_sha256
            or ledger.get("deployment_manifest_sha256") != deployment_sha256
            or ledger.get("campaign_root_sha256") != campaign_root_sha256
            or ledger.get("pair_declaration_sha256")
            != pair_declaration_sha256
            or ledger.get("active_run") is not None):
        raise CampaignSealError("shared campaign ledger binding differs")
    try:
        budget = Decimal(str(ledger["budget_cap_usd"]))
        accrued = Decimal(str(ledger["accrued_cost_usd"]))
    except (InvalidOperation, ValueError) as exc:
        raise CampaignSealError("campaign ledger cost is invalid") from exc
    if (not budget.is_finite() or not accrued.is_finite()
            or budget <= 0 or accrued < 0 or accrued > budget):
        raise CampaignSealError("campaign ledger cost is invalid")
    entries = ledger.get("entries")
    if (not isinstance(entries, list) or len(entries) != 2
            or [entry.get("arm_id") for entry in entries] != list(ARMS)):
        raise CampaignSealError(
            "campaign ledger must contain exactly CC-on then CC-off")
    previous = ZERO_SHA256
    for index, (arm, entry) in enumerate(zip(ARMS, entries)):
        entry_fields = {
            "sequence", "run_id", "arm_id", "rate_role", "finished_at",
            "total_campaign_cost_usd", "lifecycle_launch_sha256",
            "lifecycle_record_sha256",
            "absence_artifact_sha256", "exit_code", "previous_entry_sha256",
            "entry_sha256",
        }
        entry_body = {key: value for key, value in entry.items()
                      if key != "entry_sha256"}
        expected_role = "comparator" if arm == "cc-off" else "primary"
        receipt = arm_results[arm]["receipt"]
        receipt_ledger = receipt.get("campaign_ledger", {})
        absence = receipt.get("resource_absence")
        if (not isinstance(entry, Mapping) or set(entry) != entry_fields
                or entry.get("sequence") != index
                or entry.get("arm_id") != arm
                or entry.get("rate_role") != expected_role
                or entry.get("lifecycle_launch_sha256")
                != lifecycle_launch_sha256[arm]
                or entry.get("exit_code") != 0
                or entry.get("previous_entry_sha256") != previous
                or entry.get("entry_sha256")
                != _sha256_bytes(_canonical_bytes(entry_body))
                or entry.get("run_id") != receipt.get("run_id")
                or entry.get("lifecycle_record_sha256")
                != receipt.get("lifecycle_record", {}).get("sha256")
                or not isinstance(absence, Mapping)
                or entry.get("absence_artifact_sha256") != absence.get("sha256")
                or receipt_ledger.get("entry_sha256")
                != entry.get("entry_sha256")
                or Path(str(receipt_ledger.get("path"))).resolve()
                != snapshot.path):
            raise CampaignSealError(f"campaign ledger {arm} entry differs")
        previous = entry["entry_sha256"]
    last_receipt_ledger = arm_results["cc-off"]["receipt"]["campaign_ledger"]
    if (last_receipt_ledger.get("state_sha256") != ledger["ledger_sha256"]
            or last_receipt_ledger.get("file_sha256_at_completion")
            != snapshot.sha256):
        raise CampaignSealError(
            "final lifecycle receipt does not bind the final shared ledger")
    require_unchanged(snapshot)
    return snapshot


def _validate_campaign_document(
    manifest: Mapping[str, Any], *, campaign_public_key_path: str | Path,
    production: bool,
) -> list[FileSnapshot]:
    expected_fields = {
        "schema_version", "protocol_id", "campaign_id", "pair_id",
        "scientific_evidence", "pair_declaration", "pair_declaration_auth",
        "campaign_public_key", "runtime_binding", "active_security_manifest",
        "bootstrap_lifecycle", "active_security_transition",
        "lifecycle_launch_manifests", "lifecycle_controller_public_key",
        "immutable_inputs", "arms", "campaign_ledger", "claim_scope",
    }
    if (set(manifest) != expected_fields
            or manifest.get("schema_version") != CAMPAIGN_SCHEMA
            or manifest.get("protocol_id") != PROTOCOL_ID
            or manifest.get("scientific_evidence") is not False
            or manifest.get("claim_scope") != CAMPAIGN_CLAIM_SCOPE):
        raise CampaignSealError("campaign manifest fields differ")
    campaign_id = _require_identifier(manifest.get("campaign_id"), "campaign_id")
    pair_id = _require_identifier(manifest.get("pair_id"), "pair_id")
    pair, pair_snapshot = _snapshot_binding(
        manifest["pair_declaration"], "pair declaration", json_object=True)
    _, pair_auth_snapshot = _snapshot_binding(
        manifest["pair_declaration_auth"], "pair declaration auth",
        json_object=True)
    _, campaign_key_snapshot = _snapshot_binding(
        manifest["campaign_public_key"], "campaign public key")
    if campaign_key_snapshot.sha256 != _require_ed25519_public_key(
            campaign_public_key_path):
        raise CampaignSealError("campaign public key argument differs")
    validated_pair = validate_pair_declaration(
        pair_snapshot.path, pair_auth_snapshot.path, campaign_key_snapshot.path)
    if (pair != validated_pair or pair.get("campaign_id") != campaign_id
            or pair.get("pair_id") != pair_id):
        raise CampaignSealError("campaign/pair identity differs")
    static_documents, static_snapshots = _validate_static_inputs(
        manifest["immutable_inputs"])
    pair_inputs = pair["immutable_inputs"]
    for name, snapshot in static_snapshots.items():
        if (pair_inputs[name].get("sha256") != snapshot.sha256
                or pair_inputs[name].get("bytes") != snapshot.size):
            raise CampaignSealError(
                f"campaign immutable {name} differs from pair declaration")
    static_hashes = {name: snapshot.sha256
                     for name, snapshot in static_snapshots.items()}
    runtime, runtime_snapshot = _snapshot_binding(
        manifest["runtime_binding"], "runtime binding", json_object=True)
    runtime = validate_active_runtime_binding(
        runtime, static_hashes["deployment_manifest"])
    if (runtime.get("hardware_packet_sha256") != static_hashes["hardware_packet"]
            or runtime.get("workload_config_sha256")
            != static_hashes["workload_config"]):
        raise CampaignSealError("active runtime immutable input binding differs")
    active, active_snapshot = _snapshot_binding(
        manifest["active_security_manifest"], "active-security manifest",
        json_object=True)
    validated_active = require_active_security_manifest(
        active_snapshot.path, active_snapshot.sha256)
    active_inputs = validated_active.get("inputs", {})
    if (active != validated_active
            or active_inputs.get("deployment_manifest_sha256")
            != static_hashes["deployment_manifest"]
            or active_inputs.get("runtime_binding_file_sha256")
            != runtime_snapshot.sha256
            or active_inputs.get("runtime_binding") != runtime):
        raise CampaignSealError("active-security runtime/deployment join differs")
    _, lifecycle_key_snapshot = _snapshot_binding(
        manifest["lifecycle_controller_public_key"],
        "lifecycle controller public key")
    if (pair["public_keys"]["lifecycle_controller"].get("sha256")
            != lifecycle_key_snapshot.sha256):
        raise CampaignSealError(
            "lifecycle controller key differs from pair declaration")
    from .cc_bootstrap_lifecycle import (
        verify_bootstrap_envelope,
        verify_bootstrap_state,
    )
    l0_document, l0_snapshot = _snapshot_binding(
        manifest["bootstrap_lifecycle"], "bootstrap lifecycle", json_object=True)
    transition_document, transition_snapshot = _snapshot_binding(
        manifest["active_security_transition"],
        "active-security transition", json_object=True)
    l0 = verify_bootstrap_envelope(
        l0_snapshot.path, lifecycle_key_snapshot.path)
    transition = verify_bootstrap_state(
        transition_snapshot.path, lifecycle_key_snapshot.path)
    if (not isinstance(l0_document, Mapping)
            or not isinstance(transition_document, Mapping)
            or l0.get("campaign", {}).get("campaign_id") != campaign_id
            or l0.get("campaign", {}).get("pair_id") != pair_id
            or l0.get("pair", {}).get("declaration", {}).get("sha256")
            != pair_snapshot.sha256
            or l0.get("artifacts", {}).get("hardware_packet", {}).get("sha256")
            != static_hashes["hardware_packet"]
            or l0.get("artifacts", {}).get(
                "deployment_manifest", {}).get("sha256")
            != static_hashes["deployment_manifest"]
            or l0.get("artifacts", {}).get("runtime_binding", {}).get("sha256")
            != runtime_snapshot.sha256
            or transition.get("phase") != "active-security-complete"
            or transition.get("campaign_id") != campaign_id
            or transition.get("pair_id") != pair_id
            or transition.get("l0_sha256") != l0_snapshot.sha256
            or transition.get("active_security_manifest_sha256")
            != active_snapshot.sha256):
        raise CampaignSealError("campaign bootstrap/active transition chain differs")
    launch_bindings = manifest.get("lifecycle_launch_manifests")
    if not isinstance(launch_bindings, Mapping) or list(launch_bindings) != list(ARMS):
        raise CampaignSealError("campaign lifecycle launch inventory differs")
    launch_payloads: dict[str, dict[str, Any]] = {}
    launch_snapshots: dict[str, FileSnapshot] = {}
    for arm in ARMS:
        launch, launch_snapshot = _snapshot_binding(
            launch_bindings[arm], f"{arm} lifecycle launch manifest",
            json_object=True)
        try:
            verification_time = datetime.fromisoformat(
                launch["payload"]["issued_at"].replace("Z", "+00:00"))
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise CampaignSealError(
                f"{arm} lifecycle launch issue time is invalid") from exc
        payload = verify_lifecycle_launch_envelope(
            launch_snapshot.path, lifecycle_key_snapshot.path,
            expected_public_key_sha256=lifecycle_key_snapshot.sha256,
            now=verification_time)
        if payload.get("arm", {}).get("protection_mode") != arm:
            raise CampaignSealError(f"{arm} lifecycle launch arm differs")
        launch_payloads[arm] = payload
        launch_snapshots[arm] = launch_snapshot
    expected_launch = {
        "campaign_id": campaign_id,
        "hardware_packet_sha256": static_hashes["hardware_packet"],
        "deployment_manifest_sha256": static_hashes["deployment_manifest"],
        "runtime_binding_sha256": runtime_snapshot.sha256,
        "active_security_manifest_sha256": active_snapshot.sha256,
        "pair_declaration_sha256": pair_snapshot.sha256,
        "comparator_strategy": pair["pairing_contract"]["strategy"],
    }
    for arm, payload in launch_payloads.items():
        if any(payload.get(key) != value for key, value in expected_launch.items()):
            raise CampaignSealError(
                f"{arm} lifecycle launch campaign binding differs")
    if (any(payload.get("campaign_root_sha256") != l0_snapshot.sha256
            or payload.get("active_transition_sha256")
            != transition_snapshot.sha256
            for payload in launch_payloads.values())
            or launch_payloads["cc-on"].get("predecessor") != {
                "kind": "active-transition",
                "sha256": transition_snapshot.sha256,
            }
            or launch_payloads["cc-off"].get("predecessor") != {
                "kind": "cc-on-lifecycle-receipt",
                "sha256": manifest["arms"]["cc-on"][
                    "lifecycle_receipt"]["sha256"],
            }):
        raise CampaignSealError("sequential lifecycle predecessor chain differs")
    arms = manifest.get("arms")
    if not isinstance(arms, Mapping) or list(arms) != list(ARMS):
        raise CampaignSealError("campaign must contain exactly CC-off and CC-on")
    arm_results = {
        arm: _validate_arm_tree(
            arm, arms[arm], pair_id=pair_id,
            pair_declaration_sha256=pair_snapshot.sha256,
            expected_static=static_hashes,
            expected_active_security_sha256=active_snapshot.sha256,
            expected_runtime_binding_sha256=runtime_snapshot.sha256,
            expected_lifecycle_launch_sha256=launch_snapshots[arm].sha256,
            pairing_contract=pair["pairing_contract"],
            production=production,
        ) for arm in ARMS
    }
    if (l0.get("artifacts", {}).get("cc_on_admission", {}).get("sha256")
            != arm_results["cc-on"]["admission_sha256"]
            or l0.get("artifacts", {}).get("cc_on_arrival", {}).get("sha256")
            != arms["cc-on"]["arrival_record"]["sha256"]):
        raise CampaignSealError("bootstrap CC-on evidence binding differs")
    pair_keys = pair["public_keys"]
    if arm_results["cc-on"]["admission_sha256"] != runtime.get(
            "admission_sha256"):
        raise CampaignSealError("active runtime/CC-on admission binding differs")
    for arm in ARMS:
        if (pair_keys[arm].get("sha256")
                != arm_results[arm]["journal_public_key_sha256"]):
            raise CampaignSealError(
                f"{arm} journal key differs from pair declaration")
    strategy = pair["pairing_contract"]["strategy"]
    instance_ids = [
        arm_results[arm]["resource_identity"]["arrival"]["instance_id"]
        for arm in ARMS
    ]
    if (strategy == "same-instance-reboot-mode-toggle"
            and instance_ids[0] != instance_ids[1]):
        raise CampaignSealError(
            "same-instance pair has different instance identities")
    if strategy == "matched-separate-instance" and instance_ids[0] == instance_ids[1]:
        raise CampaignSealError(
            "matched-separate pair reused one instance identity")
    for arm in ARMS:
        result = arm_results[arm]
        expected_resource = result["resource_identity"]["lifecycle"]
        launch_arm = launch_payloads[arm].get("arm")
        if not isinstance(launch_arm, Mapping):
            raise CampaignSealError(f"{arm} lifecycle launch arm is missing")
        expected_authorization_sha256 = runtime.get("admission_sha256")
        expected_authorization_kind = (
            "cc-on-admission" if arm == "cc-on"
            else "paired-cc-on-admission-reference")
        if (launch_arm.get("resource_identity") != expected_resource
                or launch_arm.get("teardown_argv_sha256")
                != result["teardown_argv_sha256"]
                or launch_arm.get("absence_verifier_argv_sha256")
                != result["absence_verifier_argv_sha256"]
                or launch_arm.get("arrival_record_sha256")
                != arms[arm]["arrival_record"]["sha256"]
                or launch_arm.get("authorization_kind")
                != expected_authorization_kind
                or launch_arm.get("authorization_sha256")
                != expected_authorization_sha256
                or launch_arm.get("pair_id") != pair_id
                or launch_arm.get("pair_manifest_sha256")
                != pair_snapshot.sha256
                or result["lifecycle_start"].get("campaign_id")
                != campaign_id):
            raise CampaignSealError(
                f"{arm} lifecycle launch/resource contract differs")
    off_correctness = arm_results["cc-off"]["correctness"]["binding"]
    on_correctness = arm_results["cc-on"]["correctness"]["binding"]
    for field in (
        "pair_id", "config_sha256", "checkpoint_sha256", "panels_sha256",
        "hardware_identity_sha256", "pair_manifest_sha256",
    ):
        if off_correctness.get(field) != on_correctness.get(field):
            raise CampaignSealError(
                f"cross-arm correctness shared binding differs: {field}")
    ledger_snapshot = _validate_ledger(
        manifest["campaign_ledger"], campaign_id=campaign_id,
        packet_sha256=static_hashes["hardware_packet"],
        deployment_sha256=static_hashes["deployment_manifest"],
        campaign_root_sha256=l0_snapshot.sha256,
        pair_declaration_sha256=pair_snapshot.sha256,
        lifecycle_launch_sha256={
            arm: launch_snapshots[arm].sha256 for arm in ARMS},
        arm_results=arm_results,
    )
    for arm, payload in launch_payloads.items():
        declared_ledger = Path(str(payload.get("campaign_ledger_path")))
        if declared_ledger.resolve() != ledger_snapshot.path:
            raise CampaignSealError(
                f"{arm} lifecycle launch shared-ledger path differs")
    snapshots = [
        pair_snapshot, pair_auth_snapshot, campaign_key_snapshot,
        runtime_snapshot, active_snapshot, lifecycle_key_snapshot,
        l0_snapshot, transition_snapshot,
        *launch_snapshots.values(), ledger_snapshot, *static_snapshots.values(),
    ]
    for snapshot in snapshots:
        require_unchanged(snapshot)
    return snapshots


def seal_campaign(
    output_path: str | Path, auth_output_path: str | Path, *,
    pair_declaration_path: str | Path, pair_declaration_auth_path: str | Path,
    campaign_private_key_path: str | Path,
    campaign_public_key_path: str | Path,
    runtime_binding_path: str | Path, active_security_manifest_path: str | Path,
    bootstrap_lifecycle_path: str | Path,
    active_security_transition_path: str | Path,
    lifecycle_launch_manifest_paths: Mapping[str, str | Path],
    lifecycle_controller_public_key_path: str | Path,
    deployment_manifest_path: str | Path, hardware_packet_path: str | Path,
    workload_config_path: str | Path, checkpoint_manifest_path: str | Path,
    checkpoint_artifact_path: str | Path, campaign_ledger_path: str | Path,
    arms: Mapping[str, Mapping[str, Any]], production: bool = True,
) -> tuple[Path, Path]:
    """Validate all bytes first, then exclusively write and sign the campaign."""
    pair = validate_pair_declaration(
        pair_declaration_path, pair_declaration_auth_path,
        campaign_public_key_path)
    if not isinstance(arms, Mapping) or set(arms) != set(ARMS):
        raise CampaignSealError("campaign arms must be exactly cc-off and cc-on")
    if (not isinstance(lifecycle_launch_manifest_paths, Mapping)
            or set(lifecycle_launch_manifest_paths) != set(ARMS)):
        raise CampaignSealError(
            "campaign lifecycle launches must be exactly cc-on and cc-off")
    immutable_paths = {
        "deployment_manifest": deployment_manifest_path,
        "hardware_packet": hardware_packet_path,
        "workload_config": workload_config_path,
        "checkpoint_manifest": checkpoint_manifest_path,
        "checkpoint_artifact": checkpoint_artifact_path,
    }
    arm_documents: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        required = {
            "session_id", "evidence_record", "arrival_record",
            "repetition_set", "repetition_set_auth", "journal_public_key",
            "collector_manifest", "collector_public_key", "lifecycle_receipt",
            "teardown_argv", "absence_verifier_argv",
        }
        if not isinstance(arms[arm], Mapping) or set(arms[arm]) != required:
            raise CampaignSealError(f"{arm} seal input inventory differs")
        evidence_snapshot = read_snapshot(arms[arm]["evidence_record"])
        arrival = load_arrival_record(arms[arm]["arrival_record"])
        arm_documents[arm] = {
            "protection_mode": arm,
            "session_id": arms[arm]["session_id"],
            "admission_sha256": evidence_snapshot.sha256,
            "evidence_kind": (
                "cc-on-admission" if arm == "cc-on"
                else "cc-off-arrival-inventory"),
            "resource_identity": _resource_record(
                arrival, pair["pairing_contract"]),
            **{
                name: _artifact(
                    evidence_snapshot if name == "evidence_record"
                    else read_snapshot(arms[arm][name]))
                for name in required - {"session_id"}
            },
        }
    document = {
        "schema_version": CAMPAIGN_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "campaign_id": pair["campaign_id"],
        "pair_id": pair["pair_id"],
        "scientific_evidence": False,
        "pair_declaration": _artifact(read_snapshot(pair_declaration_path)),
        "pair_declaration_auth": _artifact(
            read_snapshot(pair_declaration_auth_path)),
        "campaign_public_key": _artifact(
            read_snapshot(campaign_public_key_path, max_bytes=1024 * 1024)),
        "runtime_binding": _artifact(read_snapshot(runtime_binding_path)),
        "active_security_manifest": _artifact(
            read_snapshot(active_security_manifest_path)),
        "bootstrap_lifecycle": _artifact(
            read_snapshot(bootstrap_lifecycle_path)),
        "active_security_transition": _artifact(
            read_snapshot(active_security_transition_path)),
        "lifecycle_launch_manifests": {
            arm: _artifact(read_snapshot(lifecycle_launch_manifest_paths[arm]))
            for arm in ARMS
        },
        "lifecycle_controller_public_key": _artifact(read_snapshot(
            lifecycle_controller_public_key_path, max_bytes=1024 * 1024)),
        "immutable_inputs": {
            name: _artifact(read_snapshot(path))
            for name, path in immutable_paths.items()
        },
        "arms": arm_documents,
        "campaign_ledger": _artifact(read_snapshot(campaign_ledger_path)),
        "claim_scope": CAMPAIGN_CLAIM_SCOPE,
    }
    _validate_campaign_document(
        document, campaign_public_key_path=campaign_public_key_path,
        production=production)
    path = write_json_exclusive(output_path, document)
    os.chmod(path, 0o600)
    auth_path = _sign_auth(
        path, auth_output_path, document_schema=CAMPAIGN_SCHEMA,
        campaign_id=pair["campaign_id"], pair_id=pair["pair_id"],
        private_key_path=campaign_private_key_path,
        public_key_path=campaign_public_key_path,
    )
    validate_campaign_manifest(
        path, auth_path, campaign_public_key_path, production=production)
    return path, auth_path


def validate_campaign_manifest(
    manifest_path: str | Path, auth_path: str | Path,
    campaign_public_key_path: str | Path, *, production: bool = True,
) -> dict[str, Any]:
    """Verify the signature, every top-level byte, and both complete arm trees."""
    manifest, manifest_snapshot = read_json_snapshot(manifest_path)
    _validate_campaign_document(
        manifest, campaign_public_key_path=campaign_public_key_path,
        production=production)
    _verify_auth(
        auth_path, manifest_path, campaign_public_key_path,
        document_schema=CAMPAIGN_SCHEMA,
        campaign_id=manifest.get("campaign_id"), pair_id=manifest.get("pair_id"),
    )
    require_unchanged(manifest_snapshot)
    return manifest
