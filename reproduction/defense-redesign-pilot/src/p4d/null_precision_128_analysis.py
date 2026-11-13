"""Verification and descriptive summaries for the 128-token matched-null controls."""
from collections import defaultdict
import json
import re
import torch
from .experiment import ROOT, digest, write_json
from .null_precision_128 import ADAPTER
from .precision_analysis import check_score, summarize
from .scenario_verification import verify_scenario
from .warmup_scenarios import FLAGS, OMITTED, tensor_hash


def check_source_chain(binding):
    """Re-verify the bound warmup-budget setup/branch manifests against the binding."""
    if (digest(ROOT / binding["setup_manifest"]) != binding["setup_manifest_sha256"]
            or digest(ROOT / binding["branch_manifest"]) != binding["branch_manifest_sha256"]):
        raise ValueError("Source manifest changed")
    setup, setup_artifacts, setup_cfg = verify_scenario(binding["setup_manifest"])
    branch, _, _ = verify_scenario(binding["branch_manifest"])
    if (setup["run_id"] != binding["setup_run_id"] or branch["run_id"] != binding["branch_run_id"]
            or setup["source_tree_sha256"] != binding["source_tree_sha256"]
            or branch["inputs"].get("setup_run_id") != binding["setup_run_id"]
            or digest(setup_artifacts["common_checkpoint.pt"]) != binding["checkpoint_sha256"]
            or digest(setup_artifacts["panels.pt"]) != binding["panels_sha256"]):
        raise ValueError("Source checkpoint/panel binding mismatch")
    return setup_cfg


def analyze(label):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", label):
        raise ValueError("Invalid campaign label")
    path = ROOT / "outputs/scenarios" / label
    summary = json.loads((path / "summary.json").read_text())
    if summary["kind"] != "null_precision_128" or any(summary.get(k) != v for k, v in FLAGS.items()):
        raise ValueError("Invalid campaign kind/claim flags")
    if len(set(summary["run_ids"])) != len(summary["run_ids"]):
        raise ValueError("Duplicate run IDs")
    grouped, positive, statuses, identities = defaultdict(list), defaultdict(list), [], set()
    prepares = {}
    deferred = []
    max_support_error = 0.
    config_snapshot = json.loads((path / "config.json").read_text())
    source = json.loads((path / "source.json").read_text())
    # Two passes: control-model preparations first so release conditions can be
    # checked against them regardless of summary run-id ordering.
    for run_id in summary["run_ids"]:
        manifest, artifacts, cfg = verify_scenario(f"experiments/manifests/{run_id}.json")
        # The committed config pins device: cpu; execution may record a CLI device
        # override, which must match the per-manifest recorded device instead.
        cfg_base = {k: v for k, v in cfg.items() if k != "device"}
        snap_base = {k: v for k, v in config_snapshot.items() if k != "device"}
        if (cfg_base != snap_base or manifest["kind"] != "null_precision_128"
                or manifest["source_record"] != str((path / "source.json").relative_to(ROOT))
                or manifest["inputs"].get("device", "cpu") != config_snapshot.get("device", "cpu")):
            raise ValueError("Manifest/campaign config mismatch")
        statuses.append(manifest["status"])
        if manifest["status"] != "completed":
            continue
        if manifest["inputs"].get("stage") == "control_model_preparation":
            b = manifest["inputs"]
            setup_cfg = check_source_chain(b)
            p = json.loads(artifacts["prepare_metrics.json"].read_text())
            replay = p["replay_verification"]
            if (p["binding"] != b or any(p.get(k) != v for k, v in FLAGS.items())
                    or p["normal_candidate_scored"] is not False
                    or p["new_training_updates"] != setup_cfg["branch_steps"]
                    or replay["verified"] is not True or replay["actual"] != replay["expected"]
                    or p["control_checkpoint_sha256"] != digest(artifacts["control_checkpoint.pt"])
                    or p["corpus_sha256"] != setup_cfg["data"]["sha256"]):
                raise ValueError("Control-preparation identity or replay-verification mismatch")
            prepares[b["control_seed"]] = {"run_id": run_id, "path": p["control_checkpoint_path"],
                                           "sha256": p["control_checkpoint_sha256"]}
            identities.add((b["setup_manifest"], b["branch_manifest"], b["control_seed"], None))
        else:
            deferred.append((run_id, manifest, artifacts, cfg))
    for run_id, manifest, artifacts, cfg in deferred:
        b = manifest["inputs"]
        check_source_chain(b)
        c = json.loads(artifacts["controls.json"].read_text())
        if (c["binding"] != b or c["adapter"] != ADAPTER
                or any(c.get(k) != v for k, v in FLAGS.items())
                or c["normal_candidate_scored"] is not False or c["new_training_updates"] != 0):
            raise ValueError("Control identity or claim-scope mismatch")
        condition = b["release_condition"]
        prepared = prepares.get(b["control_seed"])
        if (prepared is None or b["control_checkpoint_path"] != prepared["path"]
                or b["control_checkpoint_sha256"] != prepared["sha256"]):
            raise ValueError("Release condition is not bound to a verified control checkpoint")
        identities.add((b["setup_manifest"], b["branch_manifest"], b["control_seed"], condition["id"]))
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
                    grouped[(condition["id"], n, e["matched_null_count"])].append((t["r"], e))
            if [d["dose"] for d in panel["positive_controls"]] != [0., *cfg["positive_doses"]]:
                raise ValueError("Missing control dose")
            for dose in panel["positive_controls"]:
                check_score(dose["score"], cfg["matched_null_counts"])
                for key, support in dose["support"].items():
                    features = saved["positive_controls"][f"{n}-{dose['dose']}"][key]
                    if (tensor_hash(features) != dose["features_sha256"][key]
                            or support["q_bits"] != condition["q_bits"]
                            or not support["support_valid"] or not 0 <= support["maximum_normalized_rounding_error"] <= 1e-3
                            or (dose["dose"] == 0 and support["zero_dose_exact"] is not True)):
                        raise ValueError("Invalid control features or quantizer-support record")
                    max_support_error = max(max_support_error, support["maximum_normalized_rounding_error"])
                if dose["dose"] > 0:
                    for e in dose["score"]["estimators"]:
                        positive[(condition["id"], n, e["matched_null_count"])].append(e["advantage"])
    expected_prepare = {(s["setup_manifest"], s["branch_manifest"], s["control_seed"])
                        for s in config_snapshot["sources"]}
    expected_ids = {(*p, c["id"]) for p in expected_prepare for c in config_snapshot["release_conditions"]}
    expected_ids |= {(*p, None) for p in expected_prepare}
    if (len(statuses) != summary["expected_scenarios"] or len(statuses) != len(expected_ids)
            or statuses.count("completed") != summary["completed"] or statuses.count("failed") != summary["failed"]
            or (not summary["failed"] and identities != expected_ids)):
        raise ValueError("Incomplete campaign or mismatched scenario accounting")
    pooled = [{"release_condition": c, "sequences_per_panel": n, "matched_null_count": k,
               **summarize(rows, positive[(c, n, k)])}
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
    lines = [f"# Matched-null precision at the pilot operating point (engineering report): {label}", "",
             f"Completed {summary['completed']}/{summary['expected_scenarios']} scenarios; failures: {summary['failed']}.",
             f"Execution revision: `{source['git_commit']}`. Source archives, artifact bindings, replay checksums, and stored estimates verified.", "",
             "Two verified warmup-budget shared setups; each d0-trained branch was replayed inside this campaign",
             "and the replayed remote/trusted checksums matched the bound branch records bit for bit.",
             "Three release conditions per control model. No normal candidate was scored or defense selected.",
             "All instruments remain unqualified and privacy **undetermined**.", ""]
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
                  "| Sequences | K | SD of fitted-probe r | SD of mean matched-null r | 95th percentile absolute advantage |",
                  "|---|---|---|---|---|"]
        for p in rows:
            lines.append(f"| {p['sequences_per_panel']} | {p['matched_null_count']} | {p['sd_fitted_r']:.4f} | "
                         f"{p['sd_mean_matched_null_r']:.4f} | {p['p95_absolute_advantage']:.4f} |")
        lines.append("")
    lines += ["Counts pool dependent conditional donor trials across both setups. They are not population false-positive",
              "rates or independent replications. Positive checks exclude dose zero and retain both nonzero doses.",
              "Increasing K averages only the matched-null term; fitted-probe r remains shared across K in each trial.",
              f"All retained quantizer-support checks passed; maximum normalized rounding error: {max_support_error:.6g}.",
              "Zero-dose controls match ordinary donor releases exactly. Positive signals were injected after defense",
              "and before final quantization. Metadata were recomputed and hashed, but were not scored as an attack view.",
              "Valid codebook support is not a claim of naturalistic control distributions or defense efficacy.", "",
              "Unevaluated: " + "; ".join(OMITTED) + ".", "",
              "See [verification](verification.json), [summary](summary.json), and [source record](source.json).", ""]
    write_json(path / "verification.json", verified)
    (path / "report.md").write_text("\n".join(lines))
    return verified
