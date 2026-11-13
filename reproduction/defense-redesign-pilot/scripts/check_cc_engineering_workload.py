#!/usr/bin/env python3
"""Report workload-package blockers; never execute the model or contact cloud."""

import argparse
import json
from pathlib import Path

from p4d.cc_workload import validate_workload_config, workload_blockers


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?",
                        default="experiments/configs/cc_engineering_workload.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    validate_workload_config(config)
    blockers = workload_blockers(config)
    report = {
        "schema_version": "p4-cc-workload-readiness-0.1",
        "protocol_id": config["protocol_id"],
        "ready_for_execution_freeze": not blockers,
        "blockers": blockers,
        "executes_workload": False,
        "creates_cloud_resources": False,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0 if not blockers else 1


if __name__ == "__main__":
    raise SystemExit(main())
