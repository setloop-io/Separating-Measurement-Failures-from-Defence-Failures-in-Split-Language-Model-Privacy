#!/usr/bin/env python3
"""Advance the signed external-controller bootstrap state without provisioning."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

from p4d.cc_bootstrap_lifecycle import (
    BootstrapLifecycleError,
    record_active_security_success,
    record_l0_failure,
    record_l0_success,
    transition_after_active_security,
    verify_bootstrap_state,
)


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=(
        "l0-success", "l0-failure", "active-success", "transition-failure"),
        required=True)
    parser.add_argument("--l0")
    parser.add_argument("--state")
    parser.add_argument("--active-security-manifest")
    parser.add_argument("--campaign-id")
    parser.add_argument("--pair-id")
    parser.add_argument("--failure-code")
    parser.add_argument("--controller-private-key", required=True)
    parser.add_argument("--controller-public-key", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        if args.action == "l0-success":
            if not args.l0:
                raise BootstrapLifecycleError("l0-success requires --l0")
            output = record_l0_success(
                args.output, l0_path=args.l0,
                controller_private_key_path=args.controller_private_key,
                controller_public_key_path=args.controller_public_key)
        elif args.action == "l0-failure":
            if not all((args.campaign_id, args.pair_id, args.failure_code)):
                raise BootstrapLifecycleError(
                    "l0-failure requires campaign, pair, and failure code")
            output = record_l0_failure(
                args.output, campaign_id=args.campaign_id,
                pair_id=args.pair_id, failure_code=args.failure_code,
                attempted_l0_path=args.l0,
                controller_private_key_path=args.controller_private_key,
                controller_public_key_path=args.controller_public_key)
        elif args.action == "active-success":
            if not all((args.state, args.l0, args.active_security_manifest)):
                raise BootstrapLifecycleError(
                    "active-success requires state, L0, and active manifest")
            output = record_active_security_success(
                args.output, state_path=args.state, l0_path=args.l0,
                active_security_manifest_path=args.active_security_manifest,
                controller_private_key_path=args.controller_private_key,
                controller_public_key_path=args.controller_public_key)
        else:
            if not all((args.state, args.l0, args.failure_code)):
                raise BootstrapLifecycleError(
                    "transition-failure requires state, L0, and failure code")
            output = transition_after_active_security(
                args.output, state_path=args.state, l0_path=args.l0,
                failure_code=args.failure_code,
                controller_private_key_path=args.controller_private_key,
                controller_public_key_path=args.controller_public_key)
        state = verify_bootstrap_state(output, args.controller_public_key)
    except (BootstrapLifecycleError, OSError, ValueError) as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({
        "schema": "p4.cc_bootstrap_state_transition_result.v1",
        "path": str(Path(output).resolve()),
        "sha256": _sha256(output),
        "phase": state["phase"],
        "active_security_permitted": state["active_security_permitted"],
        "measurement_permitted": state["measurement_permitted"],
        "teardown_required": state["teardown_required"],
        "creates_cloud_resources": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
