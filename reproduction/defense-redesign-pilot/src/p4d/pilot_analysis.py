"""Verification, pairing, and frozen gate evaluation for the exploratory pilot."""
import json
import math
import re
import numpy as np
import torch
from scipy.stats import t as student_t

from .experiment import ROOT, digest, write_json
from .pilot_runs import (ADAPTER, CLAIM_SCOPE, MODES, SCIENTIFIC_FLAGS, UNEVALUATED)
from .qualification_analysis import check_target_controls, recompute_candidates
from .run_verification import verify_run

TARGET_VIEWS = [("property", "mean_forward_values"), ("structure", "token_forward_values")]
T_CRIT = float(student_t.ppf(0.95, 2))


def gate_table(cell_runs, seeds):
    """Frozen utility/gain gate arithmetic for one cell over the recorded pilot seeds."""
    deltas, gains, removed = {}, {}, {}
    for seed in seeds:
        runs = cell_runs[seed]
        trained = runs["trained"]["final_eval_loss"]
        deltas[seed] = trained - runs["all_local"]["final_eval_loss"]
        gains[seed] = runs["frozen_random"]["final_eval_loss"] - trained
        removed[seed] = runs["removed_remote"]["final_eval_loss"] - trained
    dv = np.array([deltas[s] for s in seeds])
    gv = np.array([gains[s] for s in seeds])
    upper = float(dv.mean() + T_CRIT * dv.std(ddof=1) / math.sqrt(3))
    utility = {"delta_u_per_seed": deltas, "mean_delta_u": float(dv.mean()),
               "sd_delta_u": float(dv.std(ddof=1)), "upper_95_limit": upper,
               "interval_rule": "mean + t(0.95, df=2) * sd / sqrt(3)", "t_critical": T_CRIT,
               "mean_limit": .10, "upper_limit": .12,
               "passed": bool(dv.mean() <= .10 and upper <= .12)}
    gain = {"g_r_per_seed": gains, "mean_g_r": float(gv.mean()), "floor": .01,
            "all_pairs_positive": bool((gv > 0).all()),
            "passed": bool(gv.mean() >= .01 and (gv > 0).all())}
    removed_comparison = {"removed_minus_trained_per_seed": removed,
                          "mean_removed_minus_trained": float(np.mean([removed[s] for s in seeds]))}
    return utility, gain, removed_comparison


def privacy_gate(cell_runs, seeds):
    """Worst-seed candidate advantage per required qualified (target, view); never a silent pass."""
    targets = {}
    for target, view in TARGET_VIEWS:
        bindings, unqualified = [], []
        for seed in seeds:
            record = cell_runs[seed]["trained"]["targets"][target]
            qualified = record["binding_qualification"]["qualified"]
            bindings.append({"seed": seed, "qualified": qualified,
                             "candidate_advantage": record["candidate_advantage"]})
            if not qualified:
                unqualified.append(seed)
        if unqualified:
            targets[target] = {"view": view, "status": "undetermined",
                               "reason": f"coverage incomplete: unqualified bindings on seeds {unqualified}",
                               "bindings": bindings}
        else:
            worst = max(b["candidate_advantage"] for b in bindings)
            targets[target] = {"view": view, "status": "complete",
                               "worst_seed_candidate_advantage": worst, "band": .02,
                               "passed": bool(worst <= .02), "bindings": bindings}
    undetermined = [f"{target} x {view}" for target, view in TARGET_VIEWS
                    if targets[target]["status"] != "complete"]
    if undetermined:
        status = "undetermined"
    elif all(targets[target]["passed"] for target, _ in TARGET_VIEWS):
        status = "passed"
    else:
        status = "failed"
    return {"targets": targets, "undetermined_coverage": undetermined, "status": status}


def analyze(label):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", label):
        raise ValueError("Invalid campaign label")
    path = ROOT / "outputs/runs" / label
    summary = json.loads((path / "summary.json").read_text())
    if (summary["protocol_id"] != "p4-defense-pilot-1.0"
            or any(summary.get(k) != v for k, v in SCIENTIFIC_FLAGS.items())):
        raise ValueError("Invalid campaign identity or claim flags")
    if len(set(summary["run_ids"])) != len(summary["run_ids"]):
        raise ValueError("Duplicate run IDs")
    config_snapshot = json.loads((path / "config.json").read_text())
    source = json.loads((path / "source.json").read_text())
    panels = torch.load(path / "panels.pt", weights_only=True)
    if digest(path / "panels.pt") != summary["panels_sha256"]:
        raise ValueError("Campaign panels changed")
    statuses, run_records = [], {}
    for run_id in summary["run_ids"]:
        manifest, artifacts, cfg = verify_run(f"experiments/manifests/{run_id}.json")
        # The committed config pins device: cpu; execution may record a CLI device
        # override, which must match the per-manifest recorded device instead.
        cfg_base = {k: v for k, v in cfg.items() if k != "device"}
        snap_base = {k: v for k, v in config_snapshot.items() if k != "device"}
        if (cfg_base != snap_base or manifest["cohort"] != "exploratory"
                or manifest["scientific_evidence"] is not True
                or manifest["source_record"] != str((path / "source.json").relative_to(ROOT))
                or manifest["freeze_record"] != config_snapshot["freeze_record"]
                or manifest["inputs"].get("device", "cpu") != config_snapshot.get("device", "cpu")
                or manifest["inputs"].get("panels_sha256") != digest(path / "panels.pt")):
            raise ValueError("Manifest/campaign binding mismatch")
        statuses.append(manifest["status"])
        if manifest["status"] != "completed":
            continue
        metrics = json.loads(artifacts["metrics.json"].read_text())
        if (any(metrics.get(k) != v for k, v in SCIENTIFIC_FLAGS.items())
                or metrics["seed"] != manifest["seed"] or metrics["cell"] != manifest["cell"]
                or metrics["mode"] != manifest["mode"]):
            raise ValueError("Metrics identity mismatch")
        record = {"seed": manifest["seed"], "cell": manifest["cell"], "mode": manifest["mode"],
                  "final_eval_loss": metrics["final_eval_loss"], "targets": {}}
        if manifest["mode"] != "all_local":
            controls = metrics["controls"]
            check_target_controls(controls["property"], cfg, cfg["property_matched_nulls"], False)
            check_target_controls(controls["structure"], cfg, cfg["structure_matched_nulls"], True)
            record["recompute"] = recompute_candidates(metrics, artifacts, panels, cfg,
                                                       manifest["inputs"]["cell"], manifest["seed"],
                                                       ADAPTER)
            for target, _ in TARGET_VIEWS:
                record["targets"][target] = {
                    "candidate_advantage": controls[target]["candidate"]["estimators"][0]["advantage"],
                    "binding_qualification": controls[target]["binding_qualification"]}
        run_records[run_id] = record
    required = {(s, c["id"], m) for s in summary["seeds"] for c in config_snapshot["cells"] for m in MODES}
    required |= {(s, "all-local", "all_local") for s in summary["seeds"]}
    actual = {(r["seed"], r["cell"], r["mode"]) for r in run_records.values()}
    missing = sorted(required - actual)
    if (missing or len(statuses) != summary["expected_runs"]
            or statuses.count("completed") != summary["completed"]
            or statuses.count("failed") != summary["failed"] or summary["failed"]):
        raise ValueError(f"Incomplete cohort: missing or failed required runs: "
                         f"{missing or summary['failures']}")
    seeds = list(summary["seeds"])
    cells = [c["id"] for c in config_snapshot["cells"]]
    cell_runs = {(cell, seed): {} for cell in cells for seed in seeds}
    for record in run_records.values():
        if record["cell"] != "all-local":
            cell_runs[(record["cell"], record["seed"])][record["mode"]] = record
        else:
            for cell in cells:
                cell_runs[(cell, record["seed"])]["all_local"] = record
    verdicts = []
    for cell in cells:
        runs = {seed: cell_runs[(cell, seed)] for seed in seeds}
        utility, gain, removed_comparison = gate_table(runs, seeds)
        privacy = privacy_gate(runs, seeds)
        eligible = bool(utility["passed"] and gain["passed"] and privacy["status"] == "passed")
        reasons = []
        if not utility["passed"]:
            reasons.append("utility gate failed")
        if not gain["passed"]:
            reasons.append("causal-gain gate failed")
        if privacy["status"] == "undetermined":
            reasons.append("privacy coverage undetermined")
        elif privacy["status"] == "failed":
            reasons.append("privacy gate failed")
        verdicts.append({"cell": cell, "pilot_eligible": eligible,
                         "reason": "; ".join(reasons) if reasons else "all frozen gates hold at this cell",
                         "utility": utility, "gain": gain,
                         "removed_remote_comparison": removed_comparison,
                         "privacy": privacy})
    advantage_table = [{"seed": r["seed"], "cell": r["cell"], "mode": r["mode"], "target": target,
                        "candidate_advantage": r["targets"][target]["candidate_advantage"],
                        "binding_qualified": r["targets"][target]["binding_qualification"]["qualified"]}
                       for r in sorted(run_records.values(), key=lambda r: (r["seed"], r["cell"], r["mode"]))
                       if r["mode"] != "all_local" for target, _ in TARGET_VIEWS]
    recomputed = sum(1 for r in run_records.values() if "recompute" in r)
    verified = {"label": label, **SCIENTIFIC_FLAGS, "expected_runs": summary["expected_runs"],
                "completed": summary["completed"], "failed": summary["failed"],
                "freeze_record": summary["freeze_record"],
                "source_archive_and_artifacts_verified": True, "captures_hashes_verified": True,
                "candidate_scores_recomputed": recomputed == len(advantage_table) // 2,
                "split_runs_recomputed": recomputed,
                "stored_estimates_verified": True,
                "cell_verdicts": verdicts, "advantage_table": advantage_table,
                "git_commit": source["git_commit"], "source_tree_sha256": source["source_tree_sha256"],
                "analysis_module_sha256": digest(__file__), "unevaluated": UNEVALUATED}
    lines = [f"# Exploratory pilot gate report (p4-defense-pilot-1.0): {label}", "",
             f"Completed {summary['completed']}/{summary['expected_runs']} runs; failures: {summary['failed']}.",
             f"Execution revision: `{source['git_commit']}`. Manifests, source archives, artifact bindings,",
             "capture hashes, stored control scores, and candidate-score recomputation verified.", "",
             "Cohort `exploratory`, scientific_evidence true. Pilot-eligibility is exploratory evidence only:",
             "it unlocks no positive defense claim and privacy remains **undetermined** beyond the named",
             "qualified coverage. A captured or declared channel is not an evaluated attack.", "",
             "| Cell | Pilot-eligible | Mean Δ_U | Upper-95 Δ_U | Mean G_R | All pairs G_R > 0 | Privacy |",
             "|---|---|---|---|---|---|---|"]
    for v in verdicts:
        lines.append(f"| {v['cell']} | {'yes' if v['pilot_eligible'] else 'no'} | "
                     f"{v['utility']['mean_delta_u']:.4f} | {v['utility']['upper_95_limit']:.4f} | "
                     f"{v['gain']['mean_g_r']:.4f} | {v['gain']['all_pairs_positive']} | "
                     f"{v['privacy']['status']} |")
    lines += ["", "Gates (frozen): utility mean Δ_U ≤ 0.10 and mean + t(0.95, df=2)·sd/√3 ≤ 0.12; gain",
              "mean G_R ≥ 0.01 with all seed pairs positive; privacy worst-seed candidate advantage ≤ 0.02",
              "with every required binding qualified. A missing matched partner removes the pair and blocks",
              "the complete-cohort gate; no outcome-based exclusion. Removed-remote comparisons are reported",
              "separately per cell and are not gate inputs.", "",
              "## Candidate advantages per seed/mode/target (descriptive)", "",
              "| Seed | Cell | Mode | Target | Candidate advantage | Binding qualified |",
              "|---|---|---|---|---|---|"]
    for row in advantage_table:
        lines.append(f"| {row['seed']} | {row['cell']} | {row['mode']} | {row['target']} | "
                     f"{row['candidate_advantage']:.4f} | {row['binding_qualified']} |")
    undetermined = [(v["cell"], coverage) for v in verdicts for coverage in v["privacy"]["undetermined_coverage"]]
    lines += ["", "## Undetermined privacy coverage", ""]
    if undetermined:
        for cell, coverage in undetermined:
            lines.append(f"- {cell}: {coverage}")
    else:
        lines.append("- None named; required qualified coverage is complete for every cell.")
    lines += ["", "Named omissions (freeze §5): " + "; ".join(UNEVALUATED) + ".", "",
              CLAIM_SCOPE, "",
              "See [verification](verification.json), [summary](summary.json), and [source record](source.json).", ""]
    write_json(path / "verification.json", verified)
    (path / "report.md").write_text("\n".join(lines))
    return verified
