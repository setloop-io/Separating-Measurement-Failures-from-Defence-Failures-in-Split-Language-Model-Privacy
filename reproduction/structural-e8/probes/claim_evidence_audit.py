#!/usr/bin/env python3
"""Check a claim-to-artifact manifest used to close issue #47."""

import argparse
import json
from pathlib import Path

ALLOWED = {"confirmatory", "supporting", "fallback_baseline"}


def audit(manifest, root):
    errors = []
    claims = manifest.get("claims", [])
    for i, claim in enumerate(claims):
        cid = claim.get("id", f"claim-{i}")
        if not claim.get("text"):
            errors.append(f"{cid}: missing claim text")
        artifacts = claim.get("artifacts", [])
        if not artifacts:
            errors.append(f"{cid}: no artifacts")
        for rel in artifacts:
            if not (root / rel).is_file():
                errors.append(f"{cid}: missing artifact {rel}")
        if claim.get("evidence_status") not in ALLOWED:
            errors.append(f"{cid}: invalid evidence_status")
        if claim.get("headline") and claim.get("evidence_status") != "confirmatory":
            errors.append(f"{cid}: headline is not confirmatory")
        if "scope" not in claim or "limitations" not in claim:
            errors.append(f"{cid}: missing scope or limitations")
    return errors


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("manifest", nargs="?", default="papers/CLAIM_EVIDENCE_MANIFEST.json")
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    path = Path(args.manifest)
    data = json.loads(path.read_text())
    errors = audit(data, Path(args.root))
    for error in errors:
        print("ERROR", error)
    print(f"claims={len(data.get('claims', []))} errors={len(errors)}")
    raise SystemExit(bool(errors))


if __name__ == "__main__":
    main()
