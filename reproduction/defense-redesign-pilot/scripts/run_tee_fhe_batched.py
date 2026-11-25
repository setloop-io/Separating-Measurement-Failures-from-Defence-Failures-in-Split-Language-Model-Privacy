"""Run the batched-matmul corrective and selective-policy engineering stages; never a confidentiality claim."""
import argparse
import json
from p4d.tee_fhe_scenarios import run_batched

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-label")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--device", help="Override the config device (cpu, mps, cuda[:N])")
    parser.add_argument("--only-stage", action="append", dest="only_stages",
                        choices=["stage-c", "stage-d"],
                        help="Execute only this stage (repeatable); recorded in manifest inputs")
    args = parser.parse_args()
    result = run_batched(args.config, args.run_label, args.validate_only, device=args.device,
                         only_stages=args.only_stages)
    print(json.dumps(result, indent=2))
    if result.get("failed") or ("completed" in result and result["completed"] != result["expected_scenarios"]):
        raise SystemExit(1)
