"""Verification and descriptive summaries for control-only precision scenarios."""
from collections import defaultdict
import json
import re
import numpy as np
import torch
from .experiment import ROOT, digest, write_json
from .null_precision import ADAPTER
from .scenario_verification import verify_scenario
from .warmup_scenarios import FLAGS, OMITTED, tensor_hash


def check_score(score, counts):
    nulls = np.asarray(score["r_null"], dtype=np.float64)
    if (len(nulls) != max(counts) or not np.isfinite(nulls).all()
            or not np.isfinite(score["r"]) or abs(score["r"]) > 1 or (abs(nulls) > 1).any()
            or [e["matched_null_count"] for e in score["estimators"]] != counts):
        raise ValueError("Invalid or incomplete stored correlations")
    for e in score["estimators"]:
        mean = float(np.mean(nulls[:e["matched_null_count"]]))
        if e["mean_r_null"] != mean or e["advantage"] != score["r"] - mean:
            raise ValueError("Estimator differs from its recorded correlations")


def summarize(rows, positives):
    values = np.array([e["advantage"] for _, e in rows])
    raw = np.array([r for r, _ in rows])
    matched = np.array([e["mean_r_null"] for _, e in rows])
    return {"diagnostic_nulls": len(values), "within_null_band": int((abs(values) <= .02).sum()),
            "above_positive_threshold": int((values > .02).sum()),
            "mean_advantage": float(values.mean()), "sd_advantage": float(values.std(ddof=1)),
            "p95_absolute_advantage": float(np.quantile(abs(values), .95)),
            "sd_fitted_r": float(raw.std(ddof=1)), "sd_mean_matched_null_r": float(matched.std(ddof=1)),
            "positive_floor_passes": sum(v >= .10 for v in positives), "positive_checks": len(positives)}


def analyze(label):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", label):
        raise ValueError("Invalid campaign label")
    path = ROOT / "outputs/scenarios" / label
    summary = json.loads((path / "summary.json").read_text())
    if summary["kind"] != "null_precision" or any(summary.get(k) != v for k, v in FLAGS.items()):
        raise ValueError("Invalid campaign kind/claim flags")
    if len(set(summary["run_ids"])) != len(summary["run_ids"]):
        raise ValueError("Duplicate run IDs")
    grouped, positive, statuses, identities = defaultdict(list), defaultdict(list), [], set()
    max_support_error = 0.
    config_snapshot = json.loads((path / "config.json").read_text())
    source = json.loads((path / "source.json").read_text())
    for run_id in summary["run_ids"]:
        manifest, artifacts, cfg = verify_scenario(f"experiments/manifests/{run_id}.json")
        if (cfg != config_snapshot or manifest["kind"] != "null_precision"
                or manifest["source_record"] != str((path / "source.json").relative_to(ROOT))):
            raise ValueError("Manifest/campaign config mismatch")
        statuses.append(manifest["status"])
        if manifest["status"] != "completed":
            continue
        c = json.loads(artifacts["controls.json"].read_text())
        if (c["binding"] != manifest["inputs"] or c["adapter"] != ADAPTER
                or any(c.get(k) != v for k, v in FLAGS.items())
                or c["normal_candidate_scored"] is not False or c["new_training_updates"] != 0):
            raise ValueError("Control identity or claim-scope mismatch")
        b = c["binding"]
        identities.add((b["source_manifest"], b["control_seed"], b["release_condition"]["id"]))
        if digest(ROOT / b["source_manifest"]) != b["source_manifest_sha256"]:
            raise ValueError("Source manifest changed")
        original, original_artifacts, _ = verify_scenario(b["source_manifest"])
        if (original["run_id"] != b["source_run"] or original["source_tree_sha256"] != b["source_tree_sha256"]
                or digest(original_artifacts["common_checkpoint.pt"]) != b["checkpoint_sha256"]
                or digest(original_artifacts["panels.pt"]) != b["panels_sha256"]):
            raise ValueError("Source checkpoint/panel binding mismatch")
        saved = torch.load(artifacts["control_means.pt"], weights_only=True)
        if [p["sequences_per_panel"] for p in c["panels"]] != cfg["panel_sizes"]:
            raise ValueError("Missing panel size")
        for panel in c["panels"]:
            n = panel["sequences_per_panel"]
            trials = panel["no_signal_trials"]
            if [t["repetition"] for t in trials] != list(range(cfg["outer_null_repeats"])):
                raise ValueError("Missing or duplicated no-signal trial")
            for t in trials:
                check_score(t, cfg["matched_null_counts"])
                for e in t["estimators"]:
                    grouped[(n, e["matched_null_count"])].append((t["r"], e))
            if [d["dose"] for d in panel["positive_controls"]] != [0., *cfg["positive_doses"]]:
                raise ValueError("Missing control dose")
            for dose in panel["positive_controls"]:
                check_score(dose["score"], cfg["matched_null_counts"])
                for key, support in dose["support"].items():
                    features = saved["positive_controls"][f"{n}-{dose['dose']}"][key]
                    if (tensor_hash(features) != dose["features_sha256"][key]
                            or support["q_bits"] != b["release_condition"]["q_bits"]
                            or not support["support_valid"] or not 0 <= support["maximum_normalized_rounding_error"] <= 1e-3
                            or (dose["dose"] == 0 and support["zero_dose_exact"] is not True)):
                        raise ValueError("Invalid control features or quantizer-support record")
                    max_support_error = max(max_support_error, support["maximum_normalized_rounding_error"])
                if dose["dose"] > 0:
                    for e in dose["score"]["estimators"]:
                        positive[(n, e["matched_null_count"])].append(e["advantage"])
    expected_ids = {(s["manifest"], s["control_seed"], c["id"]) for s in config_snapshot["sources"]
                    for c in config_snapshot["release_conditions"]}
    if (len(statuses) != summary["expected_scenarios"] or len(statuses) != len(expected_ids)
            or statuses.count("completed") != summary["completed"] or statuses.count("failed") != summary["failed"]
            or (not summary["failed"] and identities != expected_ids)):
        raise ValueError("Incomplete campaign or mismatched scenario accounting")
    pooled = [{"sequences_per_panel": n, "matched_null_count": k, **summarize(rows, positive[(n, k)])}
              for (n, k), rows in sorted(grouped.items())]
    verified = {"label": label, **FLAGS, "completed": summary["completed"], "failed": summary["failed"],
                "source_archive_and_artifacts_verified": True, "stored_estimates_verified": True,
                "maximum_normalized_rounding_error": max_support_error, "pooled_controls": pooled,
                "git_commit": source["git_commit"], "source_tree_sha256": source["source_tree_sha256"],
                "analysis_module_sha256": digest(__file__), "unevaluated": OMITTED}
    lines = [f"# Matched-null precision engineering report: {label}", "",
             f"Completed {summary['completed']}/{summary['expected_scenarios']} scenarios; failures: {summary['failed']}.",
             f"Execution revision: `{source['git_commit']}`. Source archives, artifact bindings, and stored estimates verified.", "",
             "Two saved common warmup checkpoints, three release conditions each; no new model training.",
             "All scoring uses control copies. No normal candidate was scored or defense selected.",
             "The models were warmed under D0 q8 with frozen remote weights; D1/D3 here change only the diagnostic release.",
             "All instruments remain unqualified and privacy **undetermined**.", "",
             "| Sequences | Matched nulls K | Within ±0.02 | Above +0.02 | 95th percentile absolute advantage | Positive floors passed |",
             "|---|---|---|---|---|---|"]
    for p in pooled:
        lines.append(f"| {p['sequences_per_panel']} | {p['matched_null_count']} | {p['within_null_band']}/{p['diagnostic_nulls']} | {p['above_positive_threshold']}/{p['diagnostic_nulls']} | {p['p95_absolute_advantage']:.4f} | {p['positive_floor_passes']}/{p['positive_checks']} |")
    lines += ["", "Counts pool dependent conditional donor trials. They are not population false-positive rates or independent replications.",
              "Positive checks exclude dose zero and retain both nonzero doses. These floors alone do not qualify an instrument.", "",
              "| Sequences | K | SD of fitted-probe r | SD of mean matched-null r | SD of advantage |",
              "|---|---|---|---|---|"]
    for p in pooled:
        lines.append(f"| {p['sequences_per_panel']} | {p['matched_null_count']} | {p['sd_fitted_r']:.4f} | {p['sd_mean_matched_null_r']:.4f} | {p['sd_advantage']:.4f} |")
    lines += ["", "Increasing K averages only the matched-null term; fitted-probe r remains shared across K in each trial.",
              f"All retained quantizer-support checks passed; maximum normalized rounding error: {max_support_error:.6g}.",
              "Zero-dose controls match ordinary donor releases exactly. Positive signals were injected after defense",
              "and before final quantization. Metadata were recomputed and hashed, but were not scored as an attack view.",
              "Valid codebook support is not a claim of naturalistic control distributions or defense efficacy.", "",
              "Unevaluated: " + "; ".join(OMITTED) + ".", "",
              "See [verification](verification.json), [summary](summary.json), and [source record](source.json).", ""]
    write_json(path / "verification.json", verified)
    (path / "report.md").write_text("\n".join(lines))
    return verified
