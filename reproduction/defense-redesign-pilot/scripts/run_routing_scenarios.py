"""Run plaintext operator-placement diagnostics, with no TEE/FHE claims."""
import argparse
import json
from p4d.routing_scenarios import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-label")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    result = run(args.config, args.run_label, args.validate_only)
    print(json.dumps(result, indent=2))
    if result.get("failed"):
        raise SystemExit(1)
