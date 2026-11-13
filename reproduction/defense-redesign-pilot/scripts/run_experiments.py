#!/usr/bin/env python3
"""Run P0 or validate a draft scientific matrix; never silently freeze a protocol."""
import argparse
import json
from p4d.experiment import run_matrix

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-label")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--device", help="Override the config device (cpu, mps, cuda[:N])")
    args = parser.parse_args()
    result = run_matrix(args.config, args.run_label, args.validate_only, device=args.device)
    print(json.dumps(result, indent=2))
    if result.get("failures"):
        raise SystemExit(1)
