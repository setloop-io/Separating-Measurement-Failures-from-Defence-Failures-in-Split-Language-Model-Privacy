"""Run the bounded engineering warmup/null-stability campaign."""
import argparse
import json
from p4d.warmup_scenarios import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-label")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    result = run(args.config, args.run_label, args.validate_only)
    print(json.dumps(result, indent=2))
    if result.get("failed") or ("completed" in result and result["completed"] != result["expected_scenarios"]):
        raise SystemExit(1)
