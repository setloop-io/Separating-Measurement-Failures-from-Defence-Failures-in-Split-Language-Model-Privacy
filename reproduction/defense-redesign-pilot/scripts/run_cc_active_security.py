#!/usr/bin/env python3
"""Run one frozen LLM-Attacker CC gate; never provisions a resource."""

import argparse
import json
from pathlib import Path
import sys

from p4d.cc_active_security import ActiveSecurityError, run_active_security_gate


def _argv_file(path: str) -> list[str]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise argparse.ArgumentTypeError(f"cannot read argv JSON: {exc}") from exc
    if not isinstance(value, list) or not value \
            or not all(isinstance(item, str) and item for item in value):
        raise argparse.ArgumentTypeError("argv JSON must be a nonempty string array")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attacker-repo", required=True)
    parser.add_argument("--attacker-revision", required=True)
    parser.add_argument("--frozen-plan", required=True)
    parser.add_argument("--frozen-plan-file-sha256", required=True)
    parser.add_argument("--deployment-manifest", required=True)
    parser.add_argument("--deployment-manifest-sha256", required=True)
    parser.add_argument("--runtime-binding", required=True,
                        help="Post-admission p4-cc-active-runtime-binding-0.1 JSON")
    parser.add_argument("--runtime-binding-file-sha256", required=True)
    parser.add_argument("--evidence-bundle", required=True)
    parser.add_argument("--driver-argv-json", required=True, type=_argv_file)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile", choices=("tee", "fhe", "hybrid"), required=True)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--attacker-python", default=sys.executable)
    args = parser.parse_args()
    try:
        result = run_active_security_gate(
            attacker_repo=args.attacker_repo,
            attacker_revision=args.attacker_revision,
            frozen_plan_path=args.frozen_plan,
            frozen_plan_file_sha256=args.frozen_plan_file_sha256,
            deployment_manifest_path=args.deployment_manifest,
            deployment_manifest_sha256=args.deployment_manifest_sha256,
            runtime_binding_path=args.runtime_binding,
            runtime_binding_file_sha256=args.runtime_binding_file_sha256,
            evidence_bundle_dir=args.evidence_bundle,
            driver_argv=args.driver_argv_json,
            output_dir=args.output_dir,
            profile=args.profile,
            timeout_seconds=args.timeout_seconds,
            attacker_python=args.attacker_python,
        )
    except ActiveSecurityError as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
