"""Verify warmup scenario artifacts and summarize engineering diagnostics."""
import argparse
from collections import defaultdict
import hashlib
import json
import re
import tarfile

import jsonschema
import numpy as np
import torch
from p4d.experiment import ROOT, digest, write_json
from p4d.warmup_scenarios import FLAGS, OMITTED, tensor_hash


def analyze(label):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", label):
        raise ValueError("Invalid campaign label")
    campaign = ROOT / "outputs/scenarios" / label
    summary = json.loads((campaign / "summary.json").read_text())
    source = json.loads((campaign / "source.json").read_text())
    cfg = json.loads((campaign / "config.json").read_text())
    if summary["kind"] != "warmup" or any(summary.get(k) != v for k, v in FLAGS.items()):
        raise ValueError("Invalid engineering claim flags")
    tree_hash = hashlib.sha256(json.dumps(source["files"], sort_keys=True).encode()).hexdigest()
    if tree_hash != source["source_tree_sha256"]:
        raise ValueError("Source index hash mismatch")
    with tarfile.open(campaign / "source.tar.gz", "r:gz") as archive:
        for path, expected in source["files"].items():
            if hashlib.sha256(archive.extractfile(path).read()).hexdigest() != expected:
                raise ValueError(f"Source archive mismatch: {path}")
        import yaml
        archived_cfg = archive.extractfile("experiments/configs/warmup_diagnostics.yaml").read()
        if yaml.safe_load(archived_cfg) != cfg:
            raise ValueError("Config snapshot differs from archived protocol")
    schema = json.loads((ROOT / "experiments/schemas/scenario_manifest.schema.json").read_text())
    setups, branches, statuses, curves = {}, [], [], defaultdict(list)
    if len(set(summary["run_ids"])) != len(summary["run_ids"]):
        raise ValueError("Duplicate scenario IDs")
    for run_id in summary["run_ids"]:
        manifest = json.loads((ROOT / "experiments/manifests" / f"{run_id}.json").read_text())
        jsonschema.validate(manifest, schema)
        if (manifest["run_id"] != run_id or manifest["kind"] != "warmup"
                or manifest["source_tree_sha256"] != tree_hash or manifest["git_commit"] != source["git_commit"]
                or manifest["config_sha256"] != hashlib.sha256(archived_cfg).hexdigest()):
            raise ValueError("Manifest source/config identity mismatch")
        for a in manifest["artifacts"]:
            if digest(ROOT / a["path"]) != a["sha256"]:
                raise ValueError(f"Artifact changed: {a['path']}")
        statuses.append(manifest["status"])
        if manifest["status"] != "completed":
            continue
        out = campaign / run_id
        metrics = json.loads((out / "metrics.json").read_text())
        if any(metrics.get(k) != v for k, v in {**FLAGS, **manifest["inputs"]}.items()):
            raise ValueError("Result flags or input binding mismatch")
        if metrics["stage"] == "shared_setup":
            if metrics["remote_initial_sha256"] != metrics["remote_final_sha256"]:
                raise ValueError("Shared warmup changed frozen weights")
            panel_json = json.loads((out / "panels.json").read_text())
            panels = torch.load(out / "panels.pt", weights_only=True)
            intervals = []
            for name, panel in panels.items():
                declared = panel_json[name]
                if panel["offsets"].tolist() != declared["absolute_offsets"]:
                    raise ValueError("Panel offset declaration mismatch")
                for key, expected in declared["tensor_sha256"].items():
                    if tensor_hash(panel[key]) != expected:
                        raise ValueError("Panel tensor hash mismatch")
                start, stop = panel["region"]
                length = panel["tokens"].shape[1]
                for offset in panel["offsets"].tolist():
                    if not start <= offset or offset + length >= stop:
                        raise ValueError("Panel window outside declared region")
                    intervals.append((offset, offset + length + 1))
            intervals.sort()
            if any(b[0] < a[1] for a, b in zip(intervals, intervals[1:])):
                raise ValueError("Text windows overlap")
            setups[run_id] = metrics
        else:
            if digest(ROOT / metrics["checkpoint_path"]) != metrics["checkpoint_sha256"]:
                raise ValueError("Common checkpoint binding mismatch")
            if digest(campaign / metrics["setup_run_id"] / "panels.pt") != metrics["panel_sha256"]:
                raise ValueError("Panel binding mismatch")
            if not metrics["basis_fixed"]:
                raise ValueError("Actual D1 basis was changed")
            if metrics["mode"] == "frozen_random" and metrics["remote_initial_sha256"] != metrics["remote_final_sha256"]:
                raise ValueError("Frozen remote branch changed")
            branches.append(metrics)
            if metrics["mode"] != "all_local":
                controls = json.loads((out / "stability.json").read_text())
                if any(controls.get(k) != v for k, v in FLAGS.items()):
                    raise ValueError("Invalid diagnostic claim flags")
                if [c["sequences_per_panel"] for c in controls["curve"]] != cfg["panel_sizes"]:
                    raise ValueError("Incomplete panel curve")
                for c in controls["curve"]:
                    values = np.array([r["advantage"] for r in c["nulls"]])
                    if (len(values) != cfg["null_repeats"] or c["null_band"] != .02
                            or c["within_null_band"] != int((abs(values) <= .02).sum())
                            or c["p95_absolute_null_advantage"] != float(np.quantile(abs(values), .95))):
                        raise ValueError("Null diagnostic summary mismatch")
                    curves[c["sequences_per_panel"]].extend(values.tolist())
    for m in branches:
        setup = setups[m["setup_run_id"]]
        if m["seed"] != setup["seed"]:
            raise ValueError("Branch belongs to a different seed")
        if m["mode"] != "all_local":
            if m["trusted_initial_sha256"] != setup["trusted_warm_sha256"]:
                raise ValueError("Branch initial trusted weights mismatch")
            if m["mode"] != "removed_remote" and m["remote_initial_sha256"] != setup["remote_final_sha256"]:
                raise ValueError("Branch initial remote weights mismatch")
            if m["variant"] != "d0":
                name = "initial" if m["variant"] == "d1_initial_basis" else "warm"
                if m["basis_used_sha256"] != setup["bases"][name]["basis_sha256"]:
                    raise ValueError("Branch defense used the wrong basis")
    expected = len(cfg["seeds"]) * (1 + len(cfg["variants"]) * len(cfg["modes"]) + 1)
    if (summary["expected_scenarios"] != expected or len(statuses) != expected
            or statuses.count("completed") != summary["completed"] or statuses.count("failed") != summary["failed"]):
        raise ValueError("Campaign incomplete or counts inconsistent")
    pooled = [{"sequences_per_panel": n, "diagnostic_nulls": len(values),
               "within_null_band": int((abs(np.array(values)) <= .02).sum()),
               "p95_absolute_advantage": float(np.quantile(abs(np.array(values)), .95)),
               "max_absolute_advantage": float(max(abs(np.array(values))))}
              for n, values in sorted(curves.items())]
    result = {"label": label, **FLAGS, "completed": summary["completed"], "failed": summary["failed"],
              "source_archive_and_artifacts_verified": True, "pairing_and_panels_verified": True,
              "git_commit": source["git_commit"], "source_tree_sha256": tree_hash,
              "analysis_script_sha256": digest(__file__), "pooled_null_diagnostics": pooled,
              "unevaluated": OMITTED}
    lines = [f"# Warmup/stability engineering report: {label}", "",
             f"Completed {summary['completed']}/{expected} records; failures: {summary['failed']}.",
             f"Execution commit: `{source['git_commit']}`. Source, artifacts, pairing and panel separation verified.", "",
             "Two shared setups and twenty branch runs; 16 warmup + 16 branch updates per model.",
             "These are engineering diagnostics, not convergence results, defense selection, or instrument qualification.",
             "Privacy remains **undetermined**. Null repeats share text and models; pooling below is descriptive.", "",
             "| Sequences per panel | Null checks within ±0.02 | 95th percentile absolute advantage | Maximum absolute advantage |",
             "|---|---|---|---|"]
    for p in pooled:
        lines.append(f"| {p['sequences_per_panel']} | {p['within_null_band']}/{p['diagnostic_nulls']} | {p['p95_absolute_advantage']:.4f} | {p['max_absolute_advantage']:.4f} |")
    lines += ["", "All repetitions are retained. No null-band change or scientific qualification is inferred.", "",
              "| Seed | Initial–warm basis absolute cosine | Initial basis validation r | Warm basis validation r |",
              "|---|---|---|---|"]
    for s in sorted(setups.values(), key=lambda s: s["seed"]):
        lines.append(f"| {s['seed']} | {s['initial_warm_absolute_cosine']:.4f} | {s['bases']['initial']['validation_r']:.4f} | {s['bases']['warm']['validation_r']:.4f} |")
    lines += ["", "| Seed | Variant | Mode | Final loss (nats/byte) | Final basis cosine to initial | Final basis cosine to warm |",
              "|---|---|---|---|---|---|"]
    for b in sorted(branches, key=lambda b: (b["seed"], b["variant"], b["mode"])):
        d = b.get("post_training_basis")
        initial = f"{d['absolute_cosine_to_initial']:.4f}" if d else "—"
        warm = f"{d['absolute_cosine_to_warm']:.4f}" if d else "—"
        lines.append(f"| {b['seed']} | {b['variant']} | {b['mode']} | {b['final_eval_loss']:.4f} | {initial} | {warm} |")
    lines += ["", "D1 uses a fixed basis throughout each branch. Final re-fitting is diagnostic only.",
              "Removed controls bypass the remote module after shared warmup. All-local has its own warmup.",
              "No inference is made about converged utility, causal-gain eligibility, or defense effectiveness.", "",
              "Unevaluated in this stage: " + "; ".join(OMITTED) + ".", "",
              "See [summary](summary.json), [source](source.json), and [verification](verification.json).", ""]
    write_json(campaign / "verification.json", result)
    (campaign / "report.md").write_text("\n".join(lines))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-label", required=True)
    print(json.dumps(analyze(parser.parse_args().run_label), indent=2))
