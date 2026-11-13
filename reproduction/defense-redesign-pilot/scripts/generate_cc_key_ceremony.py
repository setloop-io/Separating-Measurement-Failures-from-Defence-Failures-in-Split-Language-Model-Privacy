#!/usr/bin/env python3
"""Generate or validate provider-neutral CC trust roots; never contact cloud."""
from __future__ import annotations

import argparse
import json

from p4d.cc_keys import (
    KeyCeremonyError,
    generate_key_ceremony,
    validate_key_ceremony,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument("--ceremony-id", required=True)
    generate.add_argument("--public-dir", required=True)
    generate.add_argument("--private-dir", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--manifest", required=True)
    args = parser.parse_args()
    try:
        if args.command == "generate":
            path = generate_key_ceremony(
                public_dir=args.public_dir,
                private_dir=args.private_dir,
                ceremony_id=args.ceremony_id,
            )
            report = validate_key_ceremony(path)
        else:
            report = validate_key_ceremony(args.manifest)
    except (KeyCeremonyError, OSError) as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps({
        "valid": True,
        "ceremony_id": report["ceremony_id"],
        "manifest_sha256": report["manifest_sha256"],
        "roles": {
            role: report["roles"][role]["public_key_sha256"]
            for role in sorted(report["roles"])
        },
        "creates_cloud_resources": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
