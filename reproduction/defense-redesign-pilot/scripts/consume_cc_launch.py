#!/usr/bin/env python3
"""Verify a signed CC launch and atomically consume its one-shot guest nonce."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

from p4d.cc_lifecycle import LifecycleError, consume_launch_nonce


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lifecycle-launch", required=True)
    parser.add_argument("--controller-public-key", required=True)
    parser.add_argument("--expected-controller-public-key-sha256", required=True)
    parser.add_argument("--launch-nonce", required=True)
    parser.add_argument("--guest-workload-argv-json", required=True)
    parser.add_argument("--heartbeat-public-key", required=True)
    parser.add_argument("--protection-mode", choices=("cc-on", "cc-off"),
                        required=True)
    parser.add_argument("--instance-id", required=True)
    args = parser.parse_args()
    try:
        record = consume_launch_nonce(
            args.lifecycle_launch,
            controller_public_key_path=args.controller_public_key,
            expected_controller_public_key_sha256=(
                args.expected_controller_public_key_sha256),
            launch_nonce_path=args.launch_nonce,
            guest_workload_argv_path=args.guest_workload_argv_json,
            heartbeat_public_key_path=args.heartbeat_public_key,
            protection_mode=args.protection_mode,
            instance_id=args.instance_id,
        )
    except (LifecycleError, OSError, ValueError) as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({
        "schema": "p4.cc_launch_nonce_consumption_result.v1",
        "record": str(record.resolve()),
        "sha256": _sha256(record),
        "consumed": True,
        "creates_cloud_resources": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
