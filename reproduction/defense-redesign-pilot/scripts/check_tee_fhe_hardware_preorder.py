#!/usr/bin/env python3
"""Validate and report hardware readiness without contacting a provider."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from p4d.hardware_preorder import load_packet, readiness_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "packet",
        nargs="?",
        default="experiments/configs/tee_fhe_hardware_preorder.json",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = readiness_report(load_packet(args.packet))
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0 if report["ready_for_order"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
