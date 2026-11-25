#!/usr/bin/env python3
"""Build the post-admission active-runtime binding; never provisions hardware."""

from __future__ import annotations

import argparse
import json
import sys

from p4d.cc_active_security import (
    ActiveSecurityError,
    build_active_runtime_binding,
    file_sha256,
    write_active_runtime_binding,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission", required=True)
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--deployment-manifest", required=True)
    parser.add_argument("--hardware-packet", required=True)
    parser.add_argument("--workload-config", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--challenge-nonce", required=True)
    parser.add_argument("--expected-session-id", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        document = build_active_runtime_binding(
            admission_path=args.admission,
            evidence_root=args.evidence_root,
            deployment_manifest_path=args.deployment_manifest,
            hardware_packet_path=args.hardware_packet,
            workload_config_path=args.workload_config,
            source_manifest_path=args.source_manifest,
            challenge_nonce_path=args.challenge_nonce,
            expected_session_id=args.expected_session_id,
        )
        path = write_active_runtime_binding(args.output, document)
    except (ActiveSecurityError, OSError, PermissionError, ValueError) as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({
        "schema": document["schema"],
        "path": str(path),
        "sha256": file_sha256(path),
        "creates_cloud_resources": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
