#!/usr/bin/env python3
"""Offline validation for CC lifecycle controls or a completed lifecycle record."""

from __future__ import annotations

import argparse
import json
import sys

from p4d.cc_lifecycle import (
    LifecycleError,
    validate_control_bundle,
    validate_lifecycle_record,
    validate_lifecycle_receipt,
    write_validation_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    controls = subparsers.add_parser("controls", help="validate four pre-order controls")
    controls.add_argument("--packet", required=True)
    controls.add_argument("--ttl-control", required=True)
    controls.add_argument("--deadman-control", required=True)
    controls.add_argument("--teardown-control", required=True)
    controls.add_argument("--budget-control", required=True)
    controls.add_argument("--output")
    record = subparsers.add_parser("record", help="validate a completed JSONL journal")
    record.add_argument("--record", required=True)
    receipt = subparsers.add_parser(
        "receipt", help="validate the post-teardown lifecycle receipt and bindings",
    )
    receipt.add_argument("--receipt", required=True)
    args = parser.parse_args()

    try:
        if args.mode == "controls":
            report = validate_control_bundle(
                {
                    "ttl": args.ttl_control,
                    "deadman": args.deadman_control,
                    "teardown": args.teardown_control,
                    "budget": args.budget_control,
                },
                packet_path=args.packet,
            )
            if args.output:
                write_validation_report(report, args.output)
        elif args.mode == "record":
            report = validate_lifecycle_record(args.record)
        else:
            report = validate_lifecycle_receipt(args.receipt)
    except (LifecycleError, OSError):
        print("INVALID: lifecycle artifact validation failed", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
