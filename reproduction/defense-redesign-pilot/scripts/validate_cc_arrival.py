#!/usr/bin/env python3
"""Validate CC arrival inventory or require a separately attested admission."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from p4d.cc_admission import (
    AdmissionFreshnessPolicy,
    AdmissionTrustPolicy,
    ROOT,
    load_arrival_record,
    require_cc_admission,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate inventory-only records or fail closed on CC admission.")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--inventory", type=Path,
                        help="Validate an inventory record; never admits a host.")
    target.add_argument("--admission", type=Path,
                        help="Require a fully passing, evidence-bound admission record.")
    parser.add_argument("--root", type=Path, default=ROOT,
                        help="Root for admission and evidence paths (default: repository root).")
    parser.add_argument("--expected-session-id",
                        help="Externally expected session ID (required with --admission).")
    parser.add_argument("--expected-workload-binding", type=Path,
                        help="Trusted JSON workload binding (required with --admission).")
    parser.add_argument("--trust-policy", type=Path,
                        help="Trusted verifier/deployment policy JSON (required with --admission).")
    parser.add_argument("--challenge-nonce", type=Path,
                        help="Raw 32-byte challenge (required with --admission).")
    parser.add_argument("--nonce-ledger", type=Path,
                        help="Protected replay ledger (required with --admission).")
    parser.add_argument("--max-arrival-age-seconds", type=int, default=1800)
    parser.add_argument("--max-check-age-seconds", type=int, default=900)
    parser.add_argument("--max-admission-age-seconds", type=int, default=300)
    parser.add_argument("--max-future-skew-seconds", type=int, default=30)
    return parser.parse_args()


def load_json_object(path: Path, label: str) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def main() -> int:
    args = parse_args()
    try:
        if args.inventory is not None:
            record = load_arrival_record(args.inventory)
            result = {
                "ok": True,
                "record_type": record["record_type"],
                "schema_version": record["schema_version"],
                "hardware_attested": False,
                "admitted": False,
            }
        else:
            required = {
                "--expected-session-id": args.expected_session_id,
                "--expected-workload-binding": args.expected_workload_binding,
                "--trust-policy": args.trust_policy,
                "--challenge-nonce": args.challenge_nonce,
                "--nonce-ledger": args.nonce_ledger,
            }
            missing = [flag for flag, value in required.items() if value is None]
            if missing:
                raise ValueError(
                    "--admission requires " + ", ".join(missing))
            policy_json = load_json_object(args.trust_policy, "trust policy")
            policy_fields = {
                "verifier_id", "adapter_kind", "adapter_measurement_sha256",
                "public_key_sha256", "expected_deployment_binding", "allow_synthetic",
            }
            if set(policy_json) != policy_fields:
                raise ValueError("trust policy fields differ from the exact contract")
            trust_policy = AdmissionTrustPolicy(**policy_json)
            freshness_policy = AdmissionFreshnessPolicy(
                max_arrival_age_seconds=args.max_arrival_age_seconds,
                max_check_age_seconds=args.max_check_age_seconds,
                max_admission_age_seconds=args.max_admission_age_seconds,
                max_future_skew_seconds=args.max_future_skew_seconds,
            )
            record = require_cc_admission(
                args.admission,
                root=args.root,
                expected_session_id=args.expected_session_id,
                expected_workload_binding=load_json_object(
                    args.expected_workload_binding, "expected workload binding"),
                trust_policy=trust_policy,
                freshness_policy=freshness_policy,
                challenge_nonce_path=args.challenge_nonce,
                nonce_ledger_path=args.nonce_ledger,
            )
            result = {
                "ok": True,
                "record_type": record["record_type"],
                "schema_version": record["schema_version"],
                "admission_id": record["admission_id"],
                "session_id": record["session_id"],
                "verifier_id": record["authentication"]["verifier"]["verifier_id"],
                "verifier_public_key_sha256": record["authentication"]["public_key"]["sha256"],
                "hardware_attested": True,
                "admitted": True,
            }
    except Exception as exc:  # CLI converts every invalid/missing case to a closed gate.
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
