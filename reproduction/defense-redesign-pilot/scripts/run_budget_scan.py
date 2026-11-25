"""Run the training-budget scan engineering campaign; never a scientific result."""
import argparse
import json
from p4d.budget_scan_scenarios import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-label")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--device", help="Override the config device (cpu, mps, cuda[:N])")
    parser.add_argument("--only-seed", type=int, action="append", dest="only_seeds",
                        help="Execute only this declared control seed (repeatable)")
    args = parser.parse_args()
    result = run(args.config, args.run_label, args.validate_only, device=args.device,
                 only_seeds=args.only_seeds)
    print(json.dumps(result, indent=2))
    if result.get("failed") or ("completed" in result and result["completed"] != result["expected_scenarios"]):
        raise SystemExit(1)
