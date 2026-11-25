"""Verify scenario provenance and generate descriptive engineering reports."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import tarfile

import jsonschema

from p4d.experiment import ROOT, digest, write_json


def analyze(label):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", label):
        raise ValueError("Invalid campaign label")
    campaign = ROOT / "outputs/scenarios" / label
    summary = json.loads((campaign / "summary.json").read_text())
    source = json.loads((campaign / "source.json").read_text())
    tree_hash = hashlib.sha256(json.dumps(source["files"], sort_keys=True).encode()).hexdigest()
    if source["source_tree_sha256"] != tree_hash:
        raise ValueError("Source index hash mismatch")
    with tarfile.open(campaign / "source.tar.gz", "r:gz") as archive:
        for rel, expected in source["files"].items():
            f = archive.extractfile(rel)
            if f is None or hashlib.sha256(f.read()).hexdigest() != expected:
                raise ValueError(f"Source archive mismatch: {rel}")
    schema = json.loads((ROOT / "experiments/schemas/scenario_manifest.schema.json").read_text())
    control_schema = json.loads((ROOT / "experiments/schemas/control_diagnostic.schema.json").read_text())
    controls, routing, statuses = [], [], []
    if len(set(summary["run_ids"])) != len(summary["run_ids"]):
        raise ValueError("Duplicate scenario identity")
    for run_id in summary["run_ids"]:
        manifest = json.loads((ROOT / "experiments/manifests" / f"{run_id}.json").read_text())
        jsonschema.validate(manifest, schema)
        if manifest["source_tree_sha256"] != tree_hash or manifest["git_commit"] != source["git_commit"]:
            raise ValueError("Manifest/source identity mismatch")
        for artifact in manifest["artifacts"]:
            if digest(ROOT / artifact["path"]) != artifact["sha256"]:
                raise ValueError(f"Artifact changed: {artifact['path']}")
        statuses.append(manifest["status"])
        if manifest["status"] != "completed":
            continue
        if summary["kind"] == "calibration":
            result = json.loads((campaign / run_id / "controls.json").read_text())
            jsonschema.validate(result, control_schema)
            for key, value in manifest["inputs"].items():
                if result["binding"].get(key) != value:
                    raise ValueError("Control/source binding mismatch")
            controls.append(result)
        else:
            result = json.loads((campaign / run_id / "metrics.json").read_text())
            if (result["hardware_attested"] is not False or result["ciphertext_only"] is not False
                    or result["scientific_evidence"] is not False or result["privacy_outcome"] != "undetermined"
                    or not result["correctness_passed"]
                    or result["delegated_calls"] != result["expected_delegated_calls"]):
                raise ValueError("Invalid routing outcome or claim flags")
            routing.append(result)
    if (statuses.count("completed") != summary["completed"] or statuses.count("failed") != summary["failed"]
            or len(statuses) != summary["expected_scenarios"]):
        raise ValueError("Campaign counts do not match manifests")
    result = {"label": label, "kind": summary["kind"], "completed": summary["completed"],
              "failed": summary["failed"], "source_archive_and_artifacts_verified": True,
              "git_commit": source["git_commit"], "source_tree_sha256": tree_hash,
              "analysis_script_sha256": digest(__file__), "scientific_evidence": False,
              "privacy_outcome": "undetermined"}
    lines = [f"# Engineering scenario report: {label}", "",
             f"Completed {summary['completed']}/{len(statuses)} scenarios; operational failures: {summary['failed']}.",
             f"Execution code: `{source['git_commit']}`. Source archive and artifact hashes verified.", "",
             "These records are engineering diagnostics. Privacy remains **undetermined**.", ""]
    if controls:
        groups = defaultdict(lambda: {"records": 0, "null_passes": 0, "doses": 0, "positive_passes": 0})
        for c in controls:
            group = groups[(c["binding"]["family"], c["binding"]["channel"])]
            group["records"] += 1
            # The same null is reused for both doses: count it only once.
            check = c["doses"][0]["thresholds"]
            group["null_passes"] += check["null_pipeline_advantage"] is not None and abs(check["null_pipeline_advantage"]) <= check["null_band"]
            for dose in c["doses"]:
                group["doses"] += 1
                check = dose["thresholds"]
                group["positive_passes"] += check["positive_advantage"] is not None and check["positive_advantage"] >= check["positive_floor"]
        result["control_groups"] = [{"family": k[0], "view": k[1], **v} for k, v in sorted(groups.items())]
        result["null_checks_failed"] = sum(v["records"] - v["null_passes"] for v in groups.values())
        result["positive_floor_checks_passed"] = sum(v["positive_passes"] for v in groups.values())
        lines += ["| Family | View | Null band passes | Positive floor passes |",
                  "|---|---|---|---|"]
        for (family, view), g in sorted(groups.items()):
            lines.append(f"| {family} | {view} | {g['null_passes']}/{g['records']} | {g['positive_passes']}/{g['doses']} |")
        lines += ["", "Null checks are counted once per record; positive checks are counted per dose.",
                  "Every record remains unqualified, including records whose numerical checks pass.",
                  "The small P0 panels, synthetic control support, and unfrozen scientific calibration",
                  "prevent qualification. Preserve failed nulls and evaluate larger independent panels",
                  "and a justified control-domain policy before freezing; do not loosen gates post hoc.", "",
                  "Omitted qualification: active/adaptive attacks, state/optimizer, general joint tensors,",
                  "history/cross-run attacks, membership, timing, and unrestricted semantics.", ""]
    else:
        result["maximum_logit_error"] = max(r["maximum_logit_error"] for r in routing) if routing else None
        lines += ["| Policy | Prompt bytes | Delegated calls | Online serialized bytes | Max logit error |",
                  "|---|---|---|---|---|"]
        for r in sorted(routing, key=lambda r: (r["prompt_length"], r["policy"])):
            lines.append(f"| {r['policy']} | {r['prompt_length']} | {r['delegated_calls']} | {r['online_serialized_bytes']} | {r['maximum_logit_error']} |")
        lines += ["", "Setup bytes are recorded separately in each run. This is CPU plaintext loopback",
                  "with full-prefix recomputation, not a network, TEE, FHE, GPU, or KV-cache benchmark.",
                  "The model is an initialized fixture. Correct routing establishes no trained-model",
                  "utility, causal learning gain, confidentiality, or active verification result.", ""]
    lines += ["See [summary](summary.json), [source record](source.json), and [verification](verification.json).", ""]
    write_json(campaign / "verification.json", result)
    (campaign / "report.md").write_text("\n".join(lines))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-label", required=True)
    print(json.dumps(analyze(parser.parse_args().run_label), indent=2))
