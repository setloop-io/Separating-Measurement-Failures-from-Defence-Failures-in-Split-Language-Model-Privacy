#!/usr/bin/env python3
"""Run one shell-free CC engineering command under fail-closed lifecycle controls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from p4d.cc_lifecycle import (
    LifecycleError,
    load_argv_json,
    policy_from_packet,
    run_supervised,
    validate_control_bundle,
    validate_lifecycle_record,
    validate_lifecycle_receipt,
)
from p4d.hardware_preorder import preorder_blockers


DEFAULT_CONTROLS = {
    "ttl": "experiments/controls/cc_ttl_control.json",
    "deadman": "experiments/controls/cc_deadman_control.json",
    "teardown": "experiments/controls/cc_teardown_control.json",
    "budget": "experiments/controls/cc_budget_control.json",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", required=True)
    parser.add_argument("--workload-argv-json", required=True)
    parser.add_argument("--guest-workload-argv-json", required=True)
    parser.add_argument("--teardown-argv-json", required=True)
    parser.add_argument("--absence-verifier-argv-json", required=True)
    parser.add_argument("--deployment-manifest", required=True)
    parser.add_argument("--runtime-binding", required=True)
    parser.add_argument("--active-security-manifest", required=True)
    parser.add_argument("--lifecycle-launch-envelope", required=True)
    parser.add_argument("--lifecycle-controller-public-key", required=True)
    parser.add_argument("--pair-declaration", required=True)
    parser.add_argument("--pair-auth", required=True)
    parser.add_argument("--pair-campaign-public-key", required=True)
    parser.add_argument("--arm-arrival", required=True)
    parser.add_argument("--launch-nonce", required=True)
    parser.add_argument("--heartbeat-public-key", required=True)
    parser.add_argument("--remote-launch-adapter", required=True)
    parser.add_argument("--bootstrap-lifecycle", required=True)
    parser.add_argument("--active-security-transition", required=True)
    parser.add_argument("--measurement-ready-state")
    parser.add_argument("--cc-on-receipt")
    parser.add_argument("--heartbeat-file", required=True)
    parser.add_argument("--record", required=True)
    parser.add_argument("--launch-context", required=True)
    parser.add_argument("--campaign-ledger", required=True)
    parser.add_argument("--absence-artifact", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--workload-output", required=True)
    parser.add_argument("--arm", choices=("cc-off", "cc-on"), required=True)
    parser.add_argument("--workdir", default=".")
    parser.add_argument("--rate-role", choices=("primary", "comparator"), required=True)
    parser.add_argument("--ttl-seconds")
    parser.add_argument("--deadman-seconds")
    parser.add_argument("--budget-check-seconds")
    parser.add_argument("--termination-grace-seconds", default="10")
    parser.add_argument("--teardown-timeout-seconds", default="120")
    parser.add_argument("--ttl-control", default=DEFAULT_CONTROLS["ttl"])
    parser.add_argument("--deadman-control", default=DEFAULT_CONTROLS["deadman"])
    parser.add_argument("--teardown-control", default=DEFAULT_CONTROLS["teardown"])
    parser.add_argument("--budget-control", default=DEFAULT_CONTROLS["budget"])
    args = parser.parse_args()

    try:
        packet_path = Path(args.packet)
        packet = json.loads(packet_path.read_text(encoding="utf-8"))
        controls = {
            "ttl": args.ttl_control,
            "deadman": args.deadman_control,
            "teardown": args.teardown_control,
            "budget": args.budget_control,
        }
        validation = validate_control_bundle(controls, packet_path=packet_path)
        blockers = preorder_blockers(packet)
        if blockers:
            raise LifecycleError("hardware packet is not execution-ready")
        control_digests = {
            name: item["sha256"] for name, item in validation["controls"].items()
        }
        binding_keys = {
            "ttl": "ttl_enforcement_sha256",
            "deadman": "deadman_switch_sha256",
            "teardown": "teardown_runbook_sha256",
            "budget": "budget_enforcement_sha256",
        }
        if any(packet.get("bindings", {}).get(binding_keys[name]) != digest
               for name, digest in control_digests.items()):
            raise LifecycleError("lifecycle controls differ from the hardware packet")
        policy = policy_from_packet(
            packet,
            packet_sha256=_sha256(packet_path),
            rate_role=args.rate_role,
            control_sha256=control_digests,
            ttl_seconds=args.ttl_seconds,
            deadman_seconds=args.deadman_seconds,
            budget_check_seconds=args.budget_check_seconds,
            termination_grace_seconds=args.termination_grace_seconds,
            teardown_timeout_seconds=args.teardown_timeout_seconds,
        )
        result = run_supervised(
            workload=load_argv_json(args.workload_argv_json),
            teardown=load_argv_json(args.teardown_argv_json),
            absence_verifier=load_argv_json(args.absence_verifier_argv_json),
            policy=policy,
            packet=packet,
            deployment_manifest_path=args.deployment_manifest,
            runtime_binding_path=args.runtime_binding,
            active_security_manifest_path=args.active_security_manifest,
            lifecycle_launch_manifest_path=args.lifecycle_launch_envelope,
            controller_public_key_path=args.lifecycle_controller_public_key,
            pair_declaration_path=args.pair_declaration,
            pair_auth_path=args.pair_auth,
            pair_campaign_public_key_path=args.pair_campaign_public_key,
            arm_arrival_path=args.arm_arrival,
            guest_workload=load_argv_json(args.guest_workload_argv_json),
            launch_nonce_path=args.launch_nonce,
            heartbeat_public_key_path=args.heartbeat_public_key,
            remote_launch_adapter_path=args.remote_launch_adapter,
            bootstrap_lifecycle_path=args.bootstrap_lifecycle,
            active_transition_path=args.active_security_transition,
            measurement_ready_state_path=args.measurement_ready_state,
            cc_on_receipt_path=args.cc_on_receipt,
            heartbeat_path=args.heartbeat_file,
            record_path=args.record,
            launch_context_path=args.launch_context,
            ledger_path=args.campaign_ledger,
            absence_artifact_path=args.absence_artifact,
            receipt_path=args.receipt,
            workload_output_path=args.workload_output,
            arm_id=args.arm,
            cwd=args.workdir,
        )
        record = validate_lifecycle_record(args.record)
        receipt = validate_lifecycle_receipt(args.receipt)
    except (LifecycleError, OSError, UnicodeDecodeError, json.JSONDecodeError):
        # Do not echo command arguments, environment, child output, or exception text.
        print("BLOCKED: lifecycle input or evidence validation failed", file=sys.stderr)
        return 1

    print(json.dumps({
        "schema": "p4.cc_lifecycle_run_summary.v1",
        "run_id": result.run_id,
        "reason": result.reason,
        "exit_code": result.exit_code,
        "teardown_exit_code": result.teardown_exit_code,
        "record_sha256": record["record_sha256"],
        "receipt_sha256": receipt["receipt_sha256"],
        "creates_cloud_resources": False,
    }, sort_keys=True))
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
