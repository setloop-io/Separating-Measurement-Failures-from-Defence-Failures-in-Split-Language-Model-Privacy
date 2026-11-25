#!/usr/bin/env python3
"""Compare bound paired-correctness artifacts from CC-off and CC-on arms."""

import argparse
import json

from p4d.cc_correctness import (
    CorrectnessComparisonError, compare_paired_correctness_files,
)
from p4d.cc_io import SecureIOError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cc-off", required=True,
                        help="CC-off p4-cc-paired-correctness-0.1 artifact")
    parser.add_argument("--cc-on", required=True,
                        help="CC-on p4-cc-paired-correctness-0.1 artifact")
    parser.add_argument("--output", required=True,
                        help="New output path; an existing file is never replaced")
    args = parser.parse_args(argv)
    try:
        result = compare_paired_correctness_files(
            args.cc_off, args.cc_on, args.output)
    except (CorrectnessComparisonError, SecureIOError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
