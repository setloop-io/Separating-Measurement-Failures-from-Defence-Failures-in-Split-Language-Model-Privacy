#!/usr/bin/env python3
"""Run the frozen CC engineering canary; never provisions hardware."""

import argparse
import json

from p4d.cc_workload import run_engineering_workload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/configs/cc_engineering_workload.json")
    parser.add_argument("--evidence", required=True,
                        help="CC-on admission record or CC-off arrival inventory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--protection-mode", choices=("cc-on", "cc-off"), required=True)
    parser.add_argument("--device", required=True,
                        help="Exact admitted CUDA ordinal, for example cuda:0")
    parser.add_argument("--hardware-packet", required=True)
    parser.add_argument("--deployment-manifest", required=True)
    parser.add_argument("--active-security-manifest", required=True)
    parser.add_argument("--active-security-manifest-sha256", required=True)
    parser.add_argument("--challenge-nonce", required=True)
    parser.add_argument("--journal-private-key", required=True)
    parser.add_argument("--journal-public-key", required=True)
    parser.add_argument("--collector-manifest", required=True)
    parser.add_argument("--collector-public-key", required=True)
    parser.add_argument("--runtime-environment-manifest", required=True)
    parser.add_argument("--trusted-store-manifest", required=True)
    parser.add_argument("--trusted-store-protection-evidence", required=True)
    parser.add_argument(
        "--collector-argv-json", required=True,
        help="Exact shell-free collector argv as a JSON array; must match the manifest.")
    parser.add_argument("--trusted-store-root", required=True)
    parser.add_argument("--admitted-session-id", required=True)
    parser.add_argument("--admission-nonce-ledger", required=True)
    parser.add_argument("--paired-cc-on-admission",
                        help="Required only for the matched CC-off comparator")
    args = parser.parse_args()
    try:
        collector_argv = json.loads(args.collector_argv_json)
    except json.JSONDecodeError as exc:
        parser.error(f"--collector-argv-json is invalid JSON: {exc}")
    if not isinstance(collector_argv, list) or not all(
            isinstance(item, str) and item for item in collector_argv):
        parser.error("--collector-argv-json must be a nonempty JSON string array")
    summary = run_engineering_workload(args.config, args.evidence, args.output,
                                       args.protection_mode, args.device,
                                       hardware_packet_path=args.hardware_packet,
                                       deployment_manifest_path=args.deployment_manifest,
                                       active_security_manifest_path=(
                                           args.active_security_manifest),
                                       active_security_manifest_sha256=(
                                           args.active_security_manifest_sha256),
                                       challenge_nonce_path=args.challenge_nonce,
                                       journal_private_key_path=args.journal_private_key,
                                       journal_public_key_path=args.journal_public_key,
                                       collector_manifest_path=args.collector_manifest,
                                       collector_argv=collector_argv,
                                       collector_public_key_path=(
                                           args.collector_public_key),
                                       runtime_environment_manifest_path=(
                                           args.runtime_environment_manifest),
                                       trusted_store_manifest_path=(
                                           args.trusted_store_manifest),
                                       trusted_store_protection_evidence_path=(
                                           args.trusted_store_protection_evidence),
                                       trusted_store_root=args.trusted_store_root,
                                       admitted_session_id=args.admitted_session_id,
                                       admission_nonce_ledger_path=(
                                           args.admission_nonce_ledger),
                                       paired_cc_on_admission_path=(
                                           args.paired_cc_on_admission))
    print(f"{args.output}/execution_summary.json: {summary['next_required_action']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
