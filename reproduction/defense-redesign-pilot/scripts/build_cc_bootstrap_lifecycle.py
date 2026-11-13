#!/usr/bin/env python3
"""Build and revalidate signed pre-active lifecycle contract L0; never provision."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from p4d.cc_bootstrap_lifecycle import (
    BootstrapLifecycleError,
    create_bootstrap_contract,
    validate_bootstrap_contract,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-declaration", required=True)
    parser.add_argument("--pair-auth", required=True)
    parser.add_argument("--pair-campaign-public-key", required=True)
    parser.add_argument("--hardware-packet", required=True)
    parser.add_argument("--deployment-manifest", required=True)
    parser.add_argument("--runtime-binding", required=True)
    parser.add_argument("--admission", required=True)
    parser.add_argument("--admission-evidence-root", required=True)
    parser.add_argument("--arrival", required=True)
    parser.add_argument("--active-security-remote-launch-argv", required=True)
    parser.add_argument("--teardown-argv", required=True)
    parser.add_argument("--absence-verifier-argv", required=True)
    parser.add_argument("--remote-launch-adapter", required=True)
    parser.add_argument("--controller-private-key", required=True)
    parser.add_argument("--controller-public-key", required=True)
    parser.add_argument("--campaign-ledger", required=True)
    parser.add_argument("--ttl-control", required=True)
    parser.add_argument("--deadman-control", required=True)
    parser.add_argument("--teardown-control", required=True)
    parser.add_argument("--budget-control", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    controls = {
        "ttl": args.ttl_control,
        "deadman": args.deadman_control,
        "teardown": args.teardown_control,
        "budget": args.budget_control,
    }
    common = {
        "pair_declaration_path": args.pair_declaration,
        "pair_auth_path": args.pair_auth,
        "pair_campaign_public_key_path": args.pair_campaign_public_key,
        "hardware_packet_path": args.hardware_packet,
        "deployment_manifest_path": args.deployment_manifest,
        "runtime_binding_path": args.runtime_binding,
        "admission_path": args.admission,
        "admission_evidence_root": args.admission_evidence_root,
        "arrival_path": args.arrival,
        "active_security_remote_launch_argv_path": (
            args.active_security_remote_launch_argv),
        "teardown_argv_path": args.teardown_argv,
        "absence_verifier_argv_path": args.absence_verifier_argv,
        "remote_launch_adapter_path": args.remote_launch_adapter,
        "controller_public_key_path": args.controller_public_key,
        "campaign_ledger_path": args.campaign_ledger,
        "control_paths": controls,
    }
    try:
        output = create_bootstrap_contract(
            args.output,
            controller_private_key_path=args.controller_private_key,
            **common,
        )
        payload = validate_bootstrap_contract(output, **common)
    except (BootstrapLifecycleError, OSError, ValueError) as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({
        "schema": payload["schema"],
        "path": str(Path(output).resolve()),
        "sha256": _sha256(Path(output)),
        "campaign_id": payload["campaign"]["campaign_id"],
        "pair_id": payload["campaign"]["pair_id"],
        "controller_location": payload["controller_location"],
        "active_security_manifest_sha256": None,
        "measurement_permitted": False,
        "creates_cloud_resources": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
