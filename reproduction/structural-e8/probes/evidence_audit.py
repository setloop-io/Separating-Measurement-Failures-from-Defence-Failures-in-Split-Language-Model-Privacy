#!/usr/bin/env python3
"""Audit experiment JSONs and manuscript claims for evidence completeness.

This is intentionally stdlib-only so CI can run it before torch is installed.
It supports two modes:

* corpus audit: report historical provenance/schema coverage without pretending
  missing historical metadata can be reconstructed;
* strict path audit: fail when explicitly supplied new artifacts omit the
  forward-looking evidence contract.
"""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

REQUIRED_PROVENANCE = {
    "dtraining_commit",
    "corpus_sha256",
    "corpus_n_docs",
    "model_manifest_sha256",
    "container_image",
    "hostname",
}
EVIDENCE_STATES = {
    "confirmatory", "supporting", "supporting_with_known_limitation",
    "superseded_metric", "fallback_baseline", "pending", "invalid",
    "diagnostic",
}
MEASUREMENT_KINDS = {"measured", "emulated", "simulated", "analytical"}
PATH_STATES = {"invalid": "invalid", "diagnostic": "diagnostic"}


def load_object(path):
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return None, [f"unreadable_json: {exc}"]
    if not isinstance(value, dict):
        return value, ["top_level_not_object"]
    return value, []


def expected_path_state(path):
    parts = Path(path).parts
    for directory, state in PATH_STATES.items():
        if directory in parts:
            return state
    return None


def audit_object(value, strict=False, path=None):
    errors, warnings = [], []
    if not isinstance(value, dict):
        return ["top_level_not_object"], warnings
    provenance = value.get("provenance")
    if not isinstance(provenance, dict):
        (errors if strict else warnings).append("missing_provenance")
    else:
        missing = sorted(k for k in REQUIRED_PROVENANCE
                         if provenance.get(k) in (None, "", "unknown"))
        if missing:
            (errors if strict else warnings).append(
                "incomplete_provenance:" + ",".join(missing))
    config = value.get("config")
    kind = (config or {}).get("measurement_kind") if isinstance(config, dict) else None
    if kind not in MEASUREMENT_KINDS:
        (errors if strict else warnings).append("missing_or_invalid_measurement_kind")
    state = value.get("evidence_status")
    if state not in EVIDENCE_STATES:
        (errors if strict else warnings).append("missing_or_invalid_evidence_status")
    if "known_limitations" not in value:
        (errors if strict else warnings).append("missing_known_limitations")
    path_state = expected_path_state(path) if path is not None else None
    if path_state and state != path_state:
        (errors if strict else warnings).append(
            f"path_status_mismatch:expected_{path_state}:got_{state}")
    return errors, warnings


def audit_collection_manifest(path, value, strict=False):
    """Audit a collection sidecar and the run summaries it references."""
    errors, warnings = [], []
    target = errors if strict else warnings
    state = value.get("evidence_status")
    if state not in EVIDENCE_STATES:
        target.append("missing_or_invalid_evidence_status")
    if value.get("measurement_kind") not in MEASUREMENT_KINDS:
        target.append("missing_or_invalid_measurement_kind")
    if not isinstance(value.get("known_limitations"), list):
        target.append("missing_known_limitations")
    provenance = value.get("provenance")
    if not isinstance(provenance, dict) or not provenance.get("dtraining_commit"):
        target.append("missing_manifest_provenance")

    runs = value.get("runs")
    if not isinstance(runs, list) or not runs:
        target.append("missing_manifest_runs")
        return errors, warnings
    base = Path(path).parent
    for run in runs:
        if not isinstance(run, dict):
            target.append("invalid_manifest_run")
            continue
        run_dir, manifest_state = run.get("run_dir"), run.get("evidence_status")
        if manifest_state not in EVIDENCE_STATES:
            target.append(f"invalid_manifest_run_status:{run_dir}")
            continue
        summary = base / str(run_dir) / "summary.json"
        artifact, load_errors = load_object(summary)
        if load_errors:
            target.append(f"missing_or_invalid_run_summary:{run_dir}")
            continue
        artifact_state = artifact.get("evidence_status")
        if artifact_state != manifest_state:
            target.append(
                f"manifest_artifact_status_mismatch:{run_dir}:"
                f"manifest={manifest_state}:artifact={artifact_state}")
        if manifest_state == "supporting_with_known_limitation":
            limitations = artifact.get("known_limitations")
            if not isinstance(limitations, list) or not limitations:
                target.append(f"missing_run_known_limitations:{run_dir}")
    return errors, warnings


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def run(paths, strict=False):
    rows = []
    for path in paths:
        value, load_errors = load_object(path)
        if load_errors:
            errors, warnings = [], []
        elif value.get("schema") == "dtraining.collection_manifest.v1":
            errors, warnings = audit_collection_manifest(path, value, strict)
        else:
            errors, warnings = audit_object(value, strict, path)
        rows.append({"path": str(path), "errors": load_errors + errors,
                     "warnings": warnings})
    return {
        "schema": "dtraining.evidence_audit.v1",
        "dtraining_commit": git_commit(),
        "strict": strict,
        "n_artifacts": len(rows),
        "n_pass": sum(not r["errors"] for r in rows),
        "n_error": sum(bool(r["errors"]) for r in rows),
        "rows": rows,
    }


def self_test():
    good = {
        "config": {"measurement_kind": "measured"},
        "evidence_status": "confirmatory",
        "known_limitations": [],
        "provenance": {k: "x" for k in REQUIRED_PROVENANCE},
    }
    bad = {"config": {"measurement_kind": "simulated"}}
    with tempfile.TemporaryDirectory() as td:
        p1, p2 = Path(td) / "good.json", Path(td) / "bad.json"
        p1.write_text(json.dumps(good)); p2.write_text(json.dumps(bad))
        assert run([p1], strict=True)["n_error"] == 0
        assert run([p2], strict=True)["n_error"] == 1
        assert run([p2], strict=False)["n_error"] == 0
        invalid_dir = Path(td) / "collected" / "invalid"
        invalid_dir.mkdir(parents=True)
        p3 = invalid_dir / "wrong.json"
        p3.write_text(json.dumps(good))
        row = run([p3], strict=True)["rows"][0]
        assert any("path_status_mismatch" in e for e in row["errors"])

        collection = Path(td) / "collection"
        (collection / "run1").mkdir(parents=True)
        summary = dict(good, evidence_status="supporting")
        (collection / "run1" / "summary.json").write_text(json.dumps(summary))
        manifest = {
            "schema": "dtraining.collection_manifest.v1",
            "evidence_status": "supporting",
            "measurement_kind": "measured",
            "known_limitations": [],
            "provenance": {"dtraining_commit": "abc"},
            "runs": [{"run_dir": "run1", "evidence_status": "supporting"}],
        }
        pm = collection / "COLLECTION_MANIFEST.json"
        pm.write_text(json.dumps(manifest))
        assert run([pm], strict=True)["n_error"] == 0
        manifest["runs"][0]["evidence_status"] = "confirmatory"
        pm.write_text(json.dumps(manifest))
        row = run([pm], strict=True)["rows"][0]
        assert any("manifest_artifact_status_mismatch" in e
                   for e in row["errors"])
    print("SELF-TEST PASSED")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="*", help="JSON artifacts to audit")
    ap.add_argument("--root", default=None,
                    help="recursively audit *.json under this directory")
    ap.add_argument("--strict", action="store_true",
                    help="fail on missing forward-looking contract fields")
    ap.add_argument("--output", default=None)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test(); return
    paths = [Path(p) for p in args.paths]
    if args.root:
        paths.extend(sorted(Path(args.root).rglob("*.json")))
    report = run(paths, strict=args.strict)
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(text)
    else:
        print(text, end="")
    raise SystemExit(1 if args.strict and report["n_error"] else 0)


if __name__ == "__main__":
    main()
