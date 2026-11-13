#!/usr/bin/env python3
"""Verify staged binary-tier files against the merged package's recorded sha256.

Usage: .venv/bin/python experiments/verify_binary_tier.py <staged-binary-tier-dir>

The expected set and hashes come from build/binary-tier-expected.json, which was
extracted from build/merge-fix-v1/merged/validation.json:
  - 60 shared artifacts  <- top-level "artifacts" map (name -> sha256)
  - 400 defended observations <- cells[].defended_observations {path, sha256}
  - 400 defence provenance   <- defended_victims[].provenance {path, sha256}
Writes <staged-dir>/binary-tier-verification.json with per-file results.
"""
import hashlib
import json
import sys
from pathlib import Path

CHUNK = 1 << 20


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> int:
    staged = Path(sys.argv[1])
    expected = json.loads(Path("build/binary-tier-expected.json").read_text())
    results = {}
    missing, mismatch, ok = [], [], 0
    for rel, want in sorted(expected.items()):
        p = staged / rel
        if not p.is_file():
            missing.append(rel)
            results[rel] = {"status": "missing", "expected_sha256": want}
            continue
        got = sha256_file(p)
        if got != want:
            mismatch.append(rel)
            results[rel] = {"status": "mismatch", "expected_sha256": want, "actual_sha256": got}
        else:
            ok += 1
            results[rel] = {"status": "ok", "sha256": got, "bytes": p.stat().st_size}
    extra = sorted(
        str(q.relative_to(staged))
        for q in staged.rglob("*")
        if q.is_file() and str(q.relative_to(staged)) not in expected
        and q.name != "binary-tier-verification.json"
    )
    report = {
        "expected": len(expected),
        "ok": ok,
        "missing": missing,
        "mismatch": mismatch,
        "extra": extra,
        "files": results,
    }
    out = staged / "binary-tier-verification.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"expected={len(expected)} ok={ok} missing={len(missing)} mismatch={len(mismatch)} extra={len(extra)}")
    return 0 if (not missing and not mismatch) else 1


if __name__ == "__main__":
    sys.exit(main())
