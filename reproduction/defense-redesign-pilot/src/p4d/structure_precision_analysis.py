"""Verification and descriptive summaries for the next-byte structure controls."""
from collections import defaultdict
import json
import re
import numpy as np
import torch
from pcr.utils import derive_seed
from .experiment import ROOT, digest, write_json
from .scenario_verification import verify_scenario
from .structure_precision import ADAPTER
from .warmup_scenarios import FLAGS, OMITTED, tensor_hash


def check_structure_score(score, counts):
    nulls = np.asarray(score["acc_null"], dtype=np.float64)
    if (len(nulls) != max(counts) or not np.isfinite(nulls).all()
            or not 0 <= score["accuracy"] <= 1 or ((nulls < 0) | (nulls > 1)).any()
            or [e["matched_null_count"] for e in score["estimators"]] != counts):
        raise ValueError("Invalid or incomplete stored accuracies")
    for e in score["estimators"]:
        mean = float(np.mean(nulls[:e["matched_null_count"]]))
        if e["mean_acc_null"] != mean or e["advantage"] != score["accuracy"] - mean:
            raise ValueError("Estimator differs from its recorded accuracies")


def summarize(rows, positives, floor):
    values = np.array([e["advantage"] for _, e in rows])
    raw = np.array([a for a, _ in rows])
    matched = np.array([e["mean_acc_null"] for _, e in rows])
    return {"diagnostic_nulls": len(values), "within_null_band": int((abs(values) <= .02).sum()),
            "above_positive_threshold": int((values > .02).sum()),
            "mean_advantage": float(values.mean()), "sd_advantage": float(values.std(ddof=1)),
            "p95_absolute_advantage": float(np.quantile(abs(values), .95)),
            "sd_fitted_accuracy": float(raw.std(ddof=1)), "sd_mean_null_accuracy": float(matched.std(ddof=1)),
            "positive_floor_passes": sum(v >= floor for v in positives), "positive_checks": len(positives)}


def check_source_chain(binding):
    """Re-verify the prepare manifest and its warmup-budget chain against the binding."""
    if digest(ROOT / binding["prepare_manifest"]) != binding["prepare_manifest_sha256"]:
        raise ValueError("Source prepare manifest changed")
    prepare_record, prepare_artifacts, _ = verify_scenario(binding["prepare_manifest"])
    inputs = prepare_record["inputs"]
    if (prepare_record["run_id"] != binding["prepare_run_id"]
            or inputs.get("stage") != "control_model_preparation"
            or digest(ROOT / inputs["setup_manifest"]) != binding["setup_manifest_sha256"]):
        raise ValueError("Preparation identity or binding mismatch")
    setup_record, setup_artifacts, _ = verify_scenario(binding["setup_manifest"])
    if (setup_record["run_id"] != binding["setup_run_id"]
            or digest(setup_artifacts["panels.pt"]) != binding["panels_sha256"]):
        raise ValueError("Setup/panel binding mismatch")
    prepare = json.loads(prepare_artifacts["prepare_metrics.json"].read_text())
    replay = prepare["replay_verification"]
    if (replay["verified"] is not True or replay["actual"] != replay["expected"]
            or prepare["control_checkpoint_sha256"] != binding["control_checkpoint_sha256"]
            or digest(prepare_artifacts["control_checkpoint.pt"]) != binding["control_checkpoint_sha256"]
            or prepare["corpus_sha256"] != binding["corpus_sha256"]):
        raise ValueError("Control checkpoint or replay binding mismatch")


def analyze(label):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", label):
        raise ValueError("Invalid campaign label")
    path = ROOT / "outputs/scenarios" / label
    summary = json.loads((path / "summary.json").read_text())
    if summary["kind"] != "structure_precision" or any(summary.get(k) != v for k, v in FLAGS.items()):
        raise ValueError("Invalid campaign kind/claim flags")
    if len(set(summary["run_ids"])) != len(summary["run_ids"]):
        raise ValueError("Duplicate run IDs")
    grouped, positive, statuses, identities = defaultdict(list), defaultdict(list), [], set()
    max_support_error = 0.
    config_snapshot = json.loads((path / "config.json").read_text())
    source = json.loads((path / "source.json").read_text())
    floor = config_snapshot["structure_positive_floor"]
    for run_id in summary["run_ids"]:
        manifest, artifacts, cfg = verify_scenario(f"experiments/manifests/{run_id}.json")
        # The committed config pins device: cpu; execution may record a CLI device
        # override, which must match the per-manifest recorded device instead.
        cfg_base = {k: v for k, v in cfg.items() if k != "device"}
        snap_base = {k: v for k, v in config_snapshot.items() if k != "device"}
        if (cfg_base != snap_base or manifest["kind"] != "structure_precision"
                or manifest["source_record"] != str((path / "source.json").relative_to(ROOT))
                or manifest["inputs"].get("device", "cpu") != config_snapshot.get("device", "cpu")):
            raise ValueError("Manifest/campaign config mismatch")
        statuses.append(manifest["status"])
        if manifest["status"] != "completed":
            continue
        b = manifest["inputs"]
        check_source_chain(b)
        c = json.loads(artifacts["controls.json"].read_text())
        if (c["binding"] != b or c["adapter"] != ADAPTER
                or any(c.get(k) != v for k, v in FLAGS.items())
                or c["normal_candidate_scored"] is not False or c["new_training_updates"] != 0):
            raise ValueError("Control identity or claim-scope mismatch")
        condition = b["release_condition"]
        identities.add((b["prepare_manifest"], b["control_seed"], condition["id"]))
        saved = torch.load(artifacts["control_features.pt"], weights_only=True)
        cosines = np.asarray(c["class_directions"]["basis_cosines"], dtype=np.float64)
        if (not 0 < c["feature_scale"] or not np.isfinite(c["feature_scale"])
                or saved["feature_scale"] != c["feature_scale"]
                or tensor_hash(saved["class_directions"]) != c["class_directions"]["directions_sha256"]
                or c["class_directions"]["seed"] != derive_seed(ADAPTER, b["control_seed"], "class-directions")
                or cosines.ndim != 2 or cosines.shape[0] != 5 or not np.isfinite(cosines).all()
                or cosines.min() < 0 or cosines.max() > 1
                or not np.allclose(saved["basis_cosines"].numpy(), cosines)):
            raise ValueError("Invalid class-direction or feature-scale record")
        for key, expected in c["label_classes_sha256"].items():
            if tensor_hash(saved["label_classes"][key]) != expected:
                raise ValueError("Stored label classes differ from their record")
        if [p["sequences_per_panel"] for p in c["panels"]] != cfg["panel_sizes"]:
            raise ValueError("Missing panel size")
        for panel in c["panels"]:
            n = panel["sequences_per_panel"]
            trials = panel["no_signal_trials"]
            if [t["repetition"] for t in trials] != list(range(cfg["outer_null_repeats"])):
                raise ValueError("Missing or duplicated no-signal trial")
            for t in trials:
                check_structure_score(t, cfg["matched_null_counts"])
                for e in t["estimators"]:
                    grouped[(condition["id"], n, e["matched_null_count"])].append((t["accuracy"], e))
            if [d["dose"] for d in panel["positive_controls"]] != [0., *cfg["positive_doses"]]:
                raise ValueError("Missing control dose")
            for dose in panel["positive_controls"]:
                check_structure_score(dose["score"], cfg["matched_null_counts"])
                for key, support in dose["support"].items():
                    if (support["q_bits"] != condition["q_bits"]
                            or not support["support_valid"] or not 0 <= support["maximum_normalized_rounding_error"] <= 1e-3
                            or (dose["dose"] == 0 and support["zero_dose_exact"] is not True)):
                        raise ValueError("Invalid control features or quantizer-support record")
                    max_support_error = max(max_support_error, support["maximum_normalized_rounding_error"])
                if dose["dose"] > 0:
                    for e in dose["score"]["estimators"]:
                        positive[(condition["id"], n, e["matched_null_count"])].append(e["advantage"])
    expected_ids = {(s["prepare_manifest"], s["control_seed"], c["id"])
                    for s in config_snapshot["sources"] for c in config_snapshot["release_conditions"]}
    if (len(statuses) != summary["expected_scenarios"] or len(statuses) != len(expected_ids)
            or statuses.count("completed") != summary["completed"] or statuses.count("failed") != summary["failed"]
            or (not summary["failed"] and identities != expected_ids)):
        raise ValueError("Incomplete campaign or mismatched scenario accounting")
    pooled = [{"release_condition": c, "sequences_per_panel": n, "matched_null_count": k,
               **summarize(rows, positive[(c, n, k)], floor)}
              for (c, n, k), rows in sorted(grouped.items())]
    for p in pooled:
        p["fraction_within_null_band"] = (p["within_null_band"] / p["diagnostic_nulls"]
                                          if p["diagnostic_nulls"] else None)
    verified = {"label": label, **FLAGS, "completed": summary["completed"], "failed": summary["failed"],
                "source_archive_and_artifacts_verified": True, "replay_checksums_verified": True,
                "stored_estimates_verified": True,
                "maximum_normalized_rounding_error": max_support_error, "pooled_controls": pooled,
                "git_commit": source["git_commit"], "source_tree_sha256": source["source_tree_sha256"],
                "analysis_module_sha256": digest(__file__), "unevaluated": OMITTED}
    lines = [f"# Next-byte structure precision at the pilot operating point (engineering report): {label}", "",
             f"Completed {summary['completed']}/{summary['expected_scenarios']} scenarios; failures: {summary['failed']}.",
             f"Execution revision: `{source['git_commit']}`. Source archives, artifact bindings, replay checksums, and stored estimates verified.", "",
             "Two verified precision-128 control-model preparations; three release conditions per control model.",
             "The scorer is the five-class MLP on per-token released features; labels are the panels' own next bytes.",
             "No normal candidate was scored or defense selected. All instruments remain unqualified and",
             "privacy **undetermined**.", ""]
    for condition in [c["id"] for c in config_snapshot["release_conditions"]]:
        rows = [p for p in pooled if p["release_condition"] == condition]
        lines += [f"## Release condition `{condition}`", "",
                  "| Sequences | Matched nulls K | Within ±0.02 | Above +0.02 | Mean advantage | SD of advantage | Positive floors passed |",
                  "|---|---|---|---|---|---|---|"]
        for p in rows:
            lines.append(f"| {p['sequences_per_panel']} | {p['matched_null_count']} | "
                         f"{p['within_null_band']}/{p['diagnostic_nulls']} ({p['fraction_within_null_band']:.2f}) | "
                         f"{p['above_positive_threshold']}/{p['diagnostic_nulls']} | "
                         f"{p['mean_advantage']:.4f} | {p['sd_advantage']:.4f} | "
                         f"{p['positive_floor_passes']}/{p['positive_checks']} |")
        lines += ["",
                  "| Sequences | K | SD of fitted accuracy | SD of mean null accuracy | 95th percentile absolute advantage |",
                  "|---|---|---|---|---|"]
        for p in rows:
            lines.append(f"| {p['sequences_per_panel']} | {p['matched_null_count']} | {p['sd_fitted_accuracy']:.4f} | "
                         f"{p['sd_mean_null_accuracy']:.4f} | {p['p95_absolute_advantage']:.4f} |")
        lines.append("")
    lines += ["Counts pool dependent conditional donor trials across both setups. They are not population false-positive",
              "rates or independent replications. Positive checks exclude dose zero and retain both nonzero doses;",
              f"the floor is the declared {floor} structure positive floor, not an acceptance gate.",
              "No-signal trials derange the label panel at the sequence level; matched nulls permute fit labels at the",
              "token-row level. Neither reuses an observed association, and neither certifies detection power.",
              f"All retained quantizer-support checks passed; maximum normalized rounding error: {max_support_error:.6g}.",
              "Zero-dose controls match ordinary donor releases exactly. Class-code signals were injected after defense",
              "and before final quantization. Valid codebook support is not a claim of naturalistic control",
              "distributions or defense efficacy.", "",
              "Unevaluated: " + "; ".join(OMITTED) + ".", "",
              "See [verification](verification.json), [summary](summary.json), and [source record](source.json).", ""]
    write_json(path / "verification.json", verified)
    (path / "report.md").write_text("\n".join(lines))
    return verified
