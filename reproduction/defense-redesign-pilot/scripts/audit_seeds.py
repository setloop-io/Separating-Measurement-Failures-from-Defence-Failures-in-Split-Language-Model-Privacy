#!/usr/bin/env python3
"""Machine-checked seed non-overlap audit for scientific cohort reservations.

Verifies that the named scientific cohorts (calibration, pilot, confirmation) do not
overlap the inherited Paper 3 seed list, any Paper 4 engineering namespace, each
other, or any seed already used in a committed manifest. Emits a JSON audit record
suitable for binding into a protocol freeze. Read-only; never mutates anything.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COHORTS = ("calibration", "pilot", "confirmation_proposed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", help="Write the audit record to this path")
    args = parser.parse_args()

    registry_path = ROOT / "experiments/seed_registry.json"
    registry = json.loads(registry_path.read_text())
    registry_sha = hashlib.sha256(registry_path.read_bytes()).hexdigest()

    prior = set(registry["paper3_prior_seeds"])
    engineering = {name: set(seeds) for name, seeds in registry.items()
                   if name not in COHORTS and isinstance(seeds, list) and name != "paper3_prior_seeds"}
    cohorts = {name: set(registry[name]) for name in COHORTS}

    findings = []

    def check(label, overlap):
        overlap = sorted(overlap)
        findings.append({"check": label, "overlap": overlap, "passed": not overlap})

    for cohort, seeds in cohorts.items():
        check(f"{cohort} vs paper3_prior_seeds", seeds & prior)
        for name, eng in engineering.items():
            check(f"{cohort} vs {name}", seeds & eng)
    names = list(cohorts)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            check(f"{a} vs {b}", cohorts[a] & cohorts[b])

    used = {}
    for manifest in sorted((ROOT / "experiments/manifests").glob("p4-*.json")):
        record = json.loads(manifest.read_text())
        seed = record.get("seed", record.get("inputs", {}).get("seed"))
        if seed is not None:
            used.setdefault(int(seed), []).append(record["run_id"])
    for cohort, seeds in cohorts.items():
        check(f"{cohort} vs executed manifests", seeds & set(used))

    audit = {"audit_id": f"p4-seed-audit-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}",
             "created_utc": datetime.now(timezone.utc).isoformat(),
             "registry_path": str(registry_path.relative_to(ROOT)),
             "registry_sha256": registry_sha,
             "cohorts": {k: sorted(v) for k, v in cohorts.items()},
             "manifests_scanned": sum(len(v) for v in used.values()),
             "distinct_executed_seeds": sorted(used),
             "findings": findings,
             "passed": all(f["passed"] for f in findings)}
    text = json.dumps(audit, indent=2, sort_keys=True) + "\n"
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
    print(text)
    raise SystemExit(0 if audit["passed"] else 1)


if __name__ == "__main__":
    main()
