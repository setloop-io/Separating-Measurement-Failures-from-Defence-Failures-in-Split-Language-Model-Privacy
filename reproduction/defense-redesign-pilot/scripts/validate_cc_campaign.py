#!/usr/bin/env python3
"""Validate a signed CC cross-arm campaign without launching hardware."""
import argparse
import json

from p4d.cc_campaign import validate_campaign_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--auth", required=True)
    parser.add_argument("--campaign-public-key", required=True)
    args = parser.parse_args()
    document = validate_campaign_manifest(
        args.manifest, args.auth, args.campaign_public_key)
    print(json.dumps({
        "valid": True,
        "campaign_id": document["campaign_id"],
        "pair_id": document["pair_id"],
        "arms": list(document["arms"]),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
