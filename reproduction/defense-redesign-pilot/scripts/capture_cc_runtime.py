#!/usr/bin/env python3
"""Capture runtime/store manifests for admission; never verifies or provisions hardware."""

from __future__ import annotations

import argparse
import json
import sys

from p4d.cc_io import write_json_exclusive
from p4d.cc_runtime_attestation import (
    ALLOWED_STORE_PROTECTION,
    RuntimeAttestationError,
    capture_runtime_environment,
    capture_trusted_store,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uv-lock", required=True)
    parser.add_argument("--image-reference", required=True)
    parser.add_argument("--device", required=True, help="Explicit CUDA ordinal, e.g. cuda:0")
    parser.add_argument("--trusted-store-root", required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument(
        "--store-protection", required=True,
        choices=sorted(ALLOWED_STORE_PROTECTION))
    parser.add_argument("--store-protection-evidence", required=True)
    parser.add_argument("--environment-output", required=True)
    parser.add_argument("--store-output", required=True)
    args = parser.parse_args(argv)
    try:
        environment = capture_runtime_environment(
            uv_lock_path=args.uv_lock,
            image_reference=args.image_reference,
            requested_cuda_device=args.device,
        )
        store = capture_trusted_store(
            root=args.trusted_store_root,
            instance_id=args.instance_id,
            protection_type=args.store_protection,
            protection_evidence_path=args.store_protection_evidence,
        )
        write_json_exclusive(args.environment_output, environment)
        write_json_exclusive(args.store_output, store)
    except (RuntimeAttestationError, FileExistsError, OSError, ValueError) as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({
        "runtime_environment_sha256": environment["environment_sha256"],
        "trusted_store_manifest_sha256": store["manifest_sha256"],
        "hardware_verified": False,
        "next_required_action": "independent verifier admission",
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
