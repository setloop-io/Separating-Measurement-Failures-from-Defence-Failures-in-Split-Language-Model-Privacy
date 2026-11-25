#!/usr/bin/env python3
"""Run Paper 4's CC workload only after a digest-pinned active gate passes."""

import argparse
import json
from pathlib import Path
import sys

from p4d.cc_active_security import ActiveSecurityError, run_gated_workload


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
    parser.add_argument("--active-security-manifest", required=True)
    parser.add_argument("--active-security-manifest-sha256", required=True)
    parser.add_argument("--workload-argv-json", required=True, type=_argv_file)
    parser.add_argument("--workdir", default=".")
    parser.add_argument("--timeout-seconds", type=int, default=14400)
    args = parser.parse_args()
    try:
        return run_gated_workload(
            active_security_manifest=args.active_security_manifest,
            active_security_manifest_sha256=args.active_security_manifest_sha256,
            workload_argv=args.workload_argv_json,
            timeout_seconds=args.timeout_seconds,
            cwd=args.workdir,
        )
    except ActiveSecurityError as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
