"""Verify engineering artifacts and generate a report without scientific gates."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

import jsonschema
import numpy as np
import torch

from p4d.experiment import ROOT, CHANNELS, UNEVALUATED, digest, write_json


def verify_campaign(label):
    campaign = ROOT / "outputs/runs" / label
    if campaign.resolve().parent != (ROOT / "outputs/runs").resolve():
        raise ValueError("Invalid campaign label")
    summary = json.loads((campaign / "summary.json").read_text())
    source = json.loads((campaign / "source.json").read_text())
    expected_tree = hashlib.sha256(json.dumps(source["files"], sort_keys=True).encode()).hexdigest()
    if expected_tree != source["source_tree_sha256"]:
        raise ValueError("Source index hash mismatch")
    with tarfile.open(campaign / "source.tar.gz", "r:gz") as archive:
        for name, expected in source["files"].items():
            stream = archive.extractfile(name)
            if stream is None or hashlib.sha256(stream.read()).hexdigest() != expected:
                raise ValueError(f"Archived source mismatch: {name}")
    schema = json.loads((ROOT / "experiments/schemas/run_manifest.schema.json").read_text())
    paths = sorted((ROOT / "experiments/manifests").glob(f"p4-{label}-*.json"))
    if len(paths) != summary["run_count"]:
        raise ValueError("Manifest count differs from validated matrix")
    rows, statuses = [], []
    for path in paths:
        manifest = json.loads(path.read_text())
        jsonschema.validate(manifest, schema)
        if manifest["source_tree_sha256"] != expected_tree:
            raise ValueError("Manifest references a different source tree")
        for artifact in manifest["artifacts"]:
            if digest(ROOT / artifact["path"]) != artifact["sha256"]:
                raise ValueError(f"Artifact hash mismatch: {artifact['path']}")
        statuses.append(manifest["status"])
        if manifest["status"] == "failed":
            rows.append({"run_id": manifest["run_id"], "status": "failed"})
            continue
        out = campaign / manifest["run_id"]
        metric = json.loads((out / "metrics.json").read_text())
        assert metric["scientific_evidence"] is False
        assert metric["privacy_status"] == "undetermined"
        assert metric["active_attack_evaluated"] is False
        assert np.isfinite(metric["final_eval_loss"])
        changed = metric["remote_initial_sha256"] != metric["remote_final_sha256"]
        if metric["mode"] == "trained":
            assert changed, "Trained remote did not change"
        else:
            assert not changed, "Frozen/absent remote checksum changed"
        panels = 0
        if metric["mode"] != "all_local":
            captures = torch.load(out / "captures.pt", weights_only=True)
            assert metric["scores"]["qualified"] is False
            assert metric["scores"]["privacy_outcome"] == "undetermined"
            for panel in ("fit", "test"):
                assert captures[panel], "Empty capture panel"
                panels += 1
                for event in captures[panel]:
                    for channel in CHANNELS:
                        assert event[channel].dtype == torch.float32
                        assert torch.isfinite(event[channel]).all()
                        assert event[channel].shape[:2] == event["next_tokens"].shape
                    assert set(event["quantizer_metadata"]) == {"minimum", "scale"}
        rows.append({"run_id": manifest["run_id"], "cell": metric["cell"],
                     "mode": metric["mode"], "status": "completed", "finite_loss": True,
                     "remote_changed": changed, "capture_panels": panels,
                     "privacy_status": "undetermined"})
    assert statuses.count("completed") == summary["completed"]
    assert statuses.count("failed") == len(summary["failures"])
    report = {"campaign": label, "protocol": summary["protocol"],
              "source_tree_sha256": expected_tree, "source_archive_verified": True,
              "manifest_count": len(paths), "completed": summary["completed"],
              "failed": len(summary["failures"]), "runs": rows,
              "scientific_evidence": False, "privacy_outcome": "undetermined",
              "unevaluated": UNEVALUATED}
    write_json(campaign / "verification.json", report)
    lines = [f"# P0 engineering report: {label}", "",
             "Status: engineering diagnostics only. No scientific hypothesis, instrument",
             "qualification, convergence, or defense-success claim follows from these runs.", "",
             f"- Completed: {report['completed']}/{report['manifest_count']}; failed: {report['failed']}.",
             "- Every recorded artifact hash and every archived source file verified.",
             "- Completed runs have finite losses and the expected remote-update behavior.",
             "- Split runs preserve full-precision forward/returned values, both gradients, and quantizer metadata.",
             "- Probe fitting and evaluation use separate text regions; all diagnostic probes remain unqualified.",
             "- Privacy outcome: **undetermined**; active attacks were not evaluated.", "",
             f"Source tree: `{expected_tree}`. See [source record](source.json),",
             "[matrix summary](summary.json), and [verification details](verification.json).", "",
             "| Cell | Mode | Status | Remote changed | Capture panels |",
             "|---|---|---|---|---|"]
    for row in rows:
        lines.append(f"| {row.get('cell', row['run_id'])} | {row.get('mode', '—')} | "
                     f"{row['status']} | {row.get('remote_changed', '—')} | {row.get('capture_panels', '—')} |")
    lines += ["", "Omitted attacks: " + "; ".join(UNEVALUATED) + ".", "",
              "Next: specify and qualify the calibration controls, resolve the D1 basis/checkpoint",
              "policy, and freeze the draft pilot. The three-step losses and two-epoch probe scores",
              "are retained in per-run metrics solely for debugging; they cannot select a defense.", ""]
    (campaign / "report.md").write_text("\n".join(lines))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-label", required=True)
    result = verify_campaign(parser.parse_args().run_label)
    print(json.dumps({k: v for k, v in result.items() if k != "runs"}, indent=2))
