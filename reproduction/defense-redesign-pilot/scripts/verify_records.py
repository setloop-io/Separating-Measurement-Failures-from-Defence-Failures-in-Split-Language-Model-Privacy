#!/usr/bin/env python3
"""CPU-node independent verification of the Paper 4 pilot records (read-only).

Roles:
  manifests — schema-validate every run manifest (soula)
  freeze    — verify freeze-record digests + pilot-scoped seed audit (voula)
  artifacts — verify small-artifact hashes of a campaign (sasa)
"""
import argparse
import json
import sys
from pathlib import Path

import jsonschema

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def role_manifests():
    run_schema = json.loads((ROOT / "experiments/schemas/run_manifest.schema.json").read_text())
    scenario_schema = json.loads((ROOT / "experiments/schemas/scenario_manifest.schema.json").read_text())
    counts, failures = {}, []
    for m in sorted((ROOT / "experiments/manifests").glob("p4-*.json")):
        record = json.loads(m.read_text())
        version = record.get("schema_version", "")
        schema = run_schema if version.startswith("p4-run-") else scenario_schema
        try:
            jsonschema.validate(record, schema)
            key = record["run_id"].split("-")[1] if "-" in record["run_id"] else "other"
            counts[key] = counts.get(key, 0) + 1
        except jsonschema.ValidationError as exc:
            failures.append((record.get("run_id", m.name), exc.message[:120]))
    print("schema-valid counts:", counts)
    if failures:
        print("FAILURES:", failures[:5])
        return 1
    print(f"all {sum(counts.values())} manifests schema-valid on this host "
          f"(run schema v1/v2 + scenario schema)")
    return 0


def role_freeze():
    rec = json.loads((ROOT / "experiments/freeze/p4-defense-pilot-1.0.json").read_text())
    ok = True
    for name, expected, path in (
            ("config", rec["config_sha256"], "experiments/configs/pilot.yaml"),
            ("protocol_doc", rec["protocol_doc_sha256"], "docs/PILOT_FREEZE.md"),
            ("documents", rec["documents_sha256"], "data/gutenberg_p4.manifest.json"),
            ("seed_audit", rec["seed_audit"]["sha256"], "experiments/freeze/pilot-seed-audit-20260922.json")):
        actual = sha(ROOT / path)
        good = actual == expected
        ok &= good
        print(f"freeze {name}: {'OK' if good else 'MISMATCH'}")
    corpus = ROOT / "data/raw/gutenberg_p4.txt"
    if corpus.exists():
        good = sha(corpus) == rec["corpus_sha256"]
        ok &= good
        print(f"freeze corpus: {'OK' if good else 'MISMATCH'}")
    else:
        print("freeze corpus: skipped (corpus not synced to this host)")
    from p4d.seed_audit import audit_seeds
    audit = audit_seeds(cohorts=("pilot",), exclude_prefix="p4-pilot-20260922-a-")
    print("pilot-scoped seed audit passed:", audit["passed"])
    ok &= audit["passed"]
    return 0 if ok else 1


def role_artifacts(label):
    manifests = sorted((ROOT / "experiments/manifests").glob(f"p4-{label}-*.json"))
    checked, bad = 0, []
    for m in manifests:
        record = json.loads(m.read_text())
        for artifact in record["artifacts"]:
            p = ROOT / artifact["path"]
            if p.name in ("captures.pt", "checkpoint.pt"):
                continue  # large tensors verified on the GPU host with the analyzer
            if not p.is_file() or sha(p) != artifact["sha256"]:
                bad.append(artifact["path"])
            else:
                checked += 1
    print(f"{label}: {checked} small artifacts hash-verified on this host; problems: {bad[:5]}")
    return 0 if not bad else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=["manifests", "freeze", "artifacts"])
    parser.add_argument("--label", default="pilot-20260922-a")
    args = parser.parse_args()
    if args.role == "manifests":
        return role_manifests()
    if args.role == "freeze":
        return role_freeze()
    return role_artifacts(args.label)


if __name__ == "__main__":
    sys.exit(main())
