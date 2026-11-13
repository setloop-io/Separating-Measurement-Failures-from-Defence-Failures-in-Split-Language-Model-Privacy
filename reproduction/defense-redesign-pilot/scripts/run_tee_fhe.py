"""Run the fleet-side TEE/FHE engineering stages; never a confidentiality claim."""
import argparse
import json
from p4d.tee_fhe_scenarios import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-label")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--device", help="Override the config device (cpu, mps, cuda[:N])")
    args = parser.parse_args()
    result = run(args.config, args.run_label, args.validate_only, device=args.device)
    print(json.dumps(result, indent=2))
    if result.get("failed") or ("completed" in result and result["completed"] != result["expected_scenarios"]):
        raise SystemExit(1)
