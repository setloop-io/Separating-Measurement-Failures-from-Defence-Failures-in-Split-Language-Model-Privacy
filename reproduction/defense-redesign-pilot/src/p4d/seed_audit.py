"""Importable replication of the scripts/audit_seeds.py non-overlap audit.

Same checks as the committed audit script (scientific cohorts vs Paper 3 seeds,
every Paper 4 engineering namespace, each other, and executed manifests). The
optional exclude_prefix lets a campaign re-audit at execution time without its
own already-recorded per-seed subsets tripping the executed-manifests check.
Read-only; never mutates anything.
"""
import hashlib
import json
from datetime import datetime, timezone

from .experiment import ROOT

COHORTS = ("calibration", "pilot", "confirmation_proposed")


def audit_seeds(root=ROOT, cohorts=COHORTS, exclude_prefix=None):
    registry_path = root / "experiments/seed_registry.json"
    registry = json.loads(registry_path.read_text())
    registry_sha = hashlib.sha256(registry_path.read_bytes()).hexdigest()
    prior = set(registry["paper3_prior_seeds"])
    engineering = {name: set(seeds) for name, seeds in registry.items()
                   if name not in cohorts and isinstance(seeds, list) and name != "paper3_prior_seeds"}
    reserved = {name: set(registry[name]) for name in cohorts}
    findings = []

    def check(label, overlap):
        overlap = sorted(overlap)
        findings.append({"check": label, "overlap": overlap, "passed": not overlap})

    for cohort, seeds in reserved.items():
        check(f"{cohort} vs paper3_prior_seeds", seeds & prior)
        for name, eng in engineering.items():
            check(f"{cohort} vs {name}", seeds & eng)
    names = list(reserved)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            check(f"{a} vs {b}", reserved[a] & reserved[b])
    used = {}
    for manifest in sorted((root / "experiments/manifests").glob("p4-*.json")):
        if exclude_prefix is not None and manifest.stem.startswith(exclude_prefix):
            continue
        record = json.loads(manifest.read_text())
        seed = record.get("seed", record.get("inputs", {}).get("seed"))
        if seed is not None:
            used.setdefault(int(seed), []).append(record["run_id"])
    for cohort, seeds in reserved.items():
        check(f"{cohort} vs executed manifests", seeds & set(used))
    return {"audit_id": f"p4-seed-audit-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "registry_path": str(registry_path.relative_to(root)),
            "registry_sha256": registry_sha,
            "cohorts": {k: sorted(v) for k, v in reserved.items()},
            "manifests_scanned": sum(len(v) for v in used.values()),
            "distinct_executed_seeds": sorted(used),
            "exclude_prefix": exclude_prefix,
            "findings": findings,
            "passed": all(f["passed"] for f in findings)}
