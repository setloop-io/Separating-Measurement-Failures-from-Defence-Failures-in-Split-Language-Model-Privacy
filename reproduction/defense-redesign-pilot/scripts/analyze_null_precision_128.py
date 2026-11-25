"""Verify and summarize the matched-null precision campaign at the pilot operating point."""
import argparse
import json
from p4d.null_precision_128_analysis import analyze

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-label", required=True)
    print(json.dumps(analyze(parser.parse_args().run_label), indent=2))
