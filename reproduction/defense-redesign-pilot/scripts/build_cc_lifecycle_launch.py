#!/usr/bin/env python3
"""Build one signed, sequential post-active-security lifecycle launch."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys
import uuid

from p4d.cc_admission import load_arrival_record
from p4d.cc_bootstrap_lifecycle import (
    BootstrapLifecycleError,
    transition_after_active_security,
    verify_bootstrap_envelope,
    verify_bootstrap_state,
)
from p4d.cc_campaign import CampaignSealError, validate_pair_declaration
from p4d.cc_lifecycle import (
    LifecycleError,
    create_lifecycle_launch_envelope,
    load_argv_json,
    validate_deployment_lifecycle_binding,
    validate_lifecycle_receipt,
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _object(path: str | Path, label: str) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LifecycleError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise LifecycleError(f"{label} must be a JSON object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("cc-on", "cc-off"), required=True)
    parser.add_argument("--packet", required=True)
    parser.add_argument("--deployment-manifest", required=True)
    parser.add_argument("--runtime-binding", required=True)
    parser.add_argument("--active-security-manifest", required=True)
    parser.add_argument("--pair-declaration", required=True)
    parser.add_argument("--pair-auth", required=True)
    parser.add_argument("--pair-campaign-public-key", required=True)
    parser.add_argument("--bootstrap-lifecycle", required=True)
    parser.add_argument("--active-security-transition", required=True)
    parser.add_argument("--arm-arrival", required=True)
    parser.add_argument("--controller-remote-launch-argv-json", required=True)
    parser.add_argument("--guest-workload-argv-json", required=True)
    parser.add_argument("--teardown-argv-json", required=True)
    parser.add_argument("--absence-verifier-argv-json", required=True)
    parser.add_argument("--launch-nonce", required=True)
    parser.add_argument("--heartbeat-public-key", required=True)
    parser.add_argument("--remote-launch-adapter", required=True)
    parser.add_argument("--cc-on-receipt")
    parser.add_argument("--measurement-ready-output")
    parser.add_argument("--valid-for-minutes", type=int, default=30)
    parser.add_argument("--controller-private-key", required=True)
    parser.add_argument("--controller-public-key", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workdir", default=".")
    args = parser.parse_args()

    try:
        if not 1 <= args.valid_for_minutes <= 240:
            raise LifecycleError("launch validity must be between 1 and 240 minutes")
        if args.arm == "cc-on":
            if not args.measurement_ready_output or args.cc_on_receipt:
                raise LifecycleError(
                    "CC-on requires a ready-state output and no predecessor receipt")
        elif not args.cc_on_receipt or args.measurement_ready_output:
            raise LifecycleError(
                "CC-off requires the CC-on predecessor receipt and no ready output")

        packet = _object(args.packet, "hardware packet")
        runtime = _object(args.runtime_binding, "active runtime binding")
        pair = validate_pair_declaration(
            args.pair_declaration, args.pair_auth,
            args.pair_campaign_public_key)
        bootstrap = verify_bootstrap_envelope(
            args.bootstrap_lifecycle, args.controller_public_key)
        transition = verify_bootstrap_state(
            args.active_security_transition, args.controller_public_key)
        if (transition.get("phase") != "active-security-complete"
                or transition.get("l0_sha256")
                != _sha256(args.bootstrap_lifecycle)
                or transition.get("active_security_manifest_sha256")
                != _sha256(args.active_security_manifest)):
            raise LifecycleError("active-security transition bytes differ")
        arrival = load_arrival_record(args.arm_arrival)
        deployment = arrival.get("deployment", {})
        if (deployment.get("protection_mode") != args.arm
                or deployment.get("pair_id") != pair.get("pair_id")
                or deployment.get("pair_manifest_sha256")
                != _sha256(args.pair_declaration)):
            raise LifecycleError("arm arrival differs from signed pair")

        controller_launch = load_argv_json(
            args.controller_remote_launch_argv_json)
        guest_workload = load_argv_json(args.guest_workload_argv_json)
        teardown = load_argv_json(args.teardown_argv_json)
        absence = load_argv_json(args.absence_verifier_argv_json)
        resource = {
            key: deployment.get(key) for key in (
                "provider", "candidate_id", "sku", "region", "zone",
                "instance_id",
            )
        }
        if any(not isinstance(value, str) or not value for value in resource.values()):
            raise LifecycleError("arm arrival resource identity is incomplete")
        issued = datetime.now(timezone.utc)
        predecessor_kind = (
            "active-transition" if args.arm == "cc-on"
            else "cc-on-lifecycle-receipt")
        if args.arm == "cc-on":
            predecessor_sha256 = _sha256(args.active_security_transition)
        else:
            receipt = validate_lifecycle_receipt(args.cc_on_receipt)
            if receipt.get("arm_id") != "cc-on" or receipt.get("exit_code") != 0:
                raise LifecycleError("CC-off predecessor receipt is not successful")
            predecessor_sha256 = _sha256(args.cc_on_receipt)

        output = create_lifecycle_launch_envelope(
            args.output,
            campaign_id=pair["campaign_id"],
            campaign_ledger_path=bootstrap["campaign"]["ledger_path"],
            packet_sha256=_sha256(args.packet),
            deployment_manifest_sha256=_sha256(args.deployment_manifest),
            runtime_binding_sha256=_sha256(args.runtime_binding),
            active_security_manifest_sha256=_sha256(
                args.active_security_manifest),
            pair_declaration_sha256=_sha256(args.pair_declaration),
            comparator_strategy=pair["pairing_contract"]["strategy"],
            campaign_root_sha256=_sha256(args.bootstrap_lifecycle),
            active_transition_sha256=_sha256(
                args.active_security_transition),
            predecessor_kind=predecessor_kind,
            predecessor_sha256=predecessor_sha256,
            launch_id=str(uuid.uuid4()),
            issued_at=issued.isoformat(timespec="microseconds").replace(
                "+00:00", "Z"),
            expires_at=(issued + timedelta(minutes=args.valid_for_minutes)).isoformat(
                timespec="microseconds").replace("+00:00", "Z"),
            arm={
                "protection_mode": args.arm,
                "arrival_record_sha256": _sha256(args.arm_arrival),
                "authorization_kind": (
                    "cc-on-admission" if args.arm == "cc-on"
                    else "paired-cc-on-admission-reference"),
                "authorization_sha256": runtime["admission_sha256"],
                "pair_id": pair["pair_id"],
                "pair_manifest_sha256": _sha256(args.pair_declaration),
                "resource_identity": resource,
                "controller_remote_launch_argv_sha256": controller_launch.sha256,
                "guest_workload_argv_sha256": guest_workload.sha256,
                "teardown_argv_sha256": teardown.sha256,
                "absence_verifier_argv_sha256": absence.sha256,
                "launch_nonce_sha256": _sha256(args.launch_nonce),
                "heartbeat_public_key_sha256": _sha256(
                    args.heartbeat_public_key),
            },
            private_key_path=args.controller_private_key,
            public_key_path=args.controller_public_key,
        )
        ready_path = None
        if args.arm == "cc-on":
            ready_path = transition_after_active_security(
                args.measurement_ready_output,
                state_path=args.active_security_transition,
                l0_path=args.bootstrap_lifecycle,
                l1_path=output,
                controller_private_key_path=args.controller_private_key,
                controller_public_key_path=args.controller_public_key,
            )
        validate_deployment_lifecycle_binding(
            packet=packet,
            packet_sha256=_sha256(args.packet),
            deployment_manifest_path=args.deployment_manifest,
            workload=controller_launch,
            guest_workload=guest_workload,
            teardown=teardown,
            absence_verifier=absence,
            ledger_path=bootstrap["campaign"]["ledger_path"],
            arm_id=args.arm,
            cwd=args.workdir,
            lifecycle_launch_manifest_path=output,
            runtime_binding_path=args.runtime_binding,
            active_security_manifest_path=args.active_security_manifest,
            controller_public_key_path=args.controller_public_key,
            pair_declaration_path=args.pair_declaration,
            pair_auth_path=args.pair_auth,
            pair_campaign_public_key_path=args.pair_campaign_public_key,
            arm_arrival_path=args.arm_arrival,
            launch_nonce_path=args.launch_nonce,
            heartbeat_public_key_path=args.heartbeat_public_key,
            remote_launch_adapter_path=args.remote_launch_adapter,
            bootstrap_lifecycle_path=args.bootstrap_lifecycle,
            active_transition_path=args.active_security_transition,
            measurement_ready_state_path=ready_path,
            cc_on_receipt_path=args.cc_on_receipt,
        )
    except (BootstrapLifecycleError, CampaignSealError, LifecycleError,
            KeyError, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 1

    print(json.dumps({
        "schema": "p4.cc_lifecycle_launch_build.v2",
        "arm": args.arm,
        "output": str(Path(output).resolve()),
        "sha256": _sha256(output),
        "measurement_ready_state": (
            str(Path(ready_path).resolve()) if ready_path else None),
        "controller_location": "external-control-plane",
        "creates_cloud_resources": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
