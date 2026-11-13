"""Verification and aggregation for the calibration-cohort qualification campaign."""
from collections import defaultdict
import json
import re
import numpy as np
import torch
from pcr.privacy_attacks_p4b import next_byte_class, sequence_targets
from pcr.utils import derive_seed

from .devices import resolve_device
from .experiment import ROOT, digest, write_json
from .null_precision import RidgeDesign
from .precision_analysis import check_score
from .qualification_runs import (ADAPTER, CLAIM_SCOPE, SCIENTIFIC_FLAGS, UNEVALUATED)
from .run_verification import verify_run
from .structure_precision import structure_score
from .structure_precision_analysis import check_structure_score
from .warmup_scenarios import TARGET, tensor_hash

TARGETS = ("property", "structure")


def recompute_candidates(metrics, artifacts, panels, cfg, cell, seed, adapter=None):
    """Re-derive candidate scores from saved captures; exact on the run's own device."""
    adapter = adapter or ADAPTER
    captures = torch.load(artifacts["captures.pt"], weights_only=True)
    for key, parts in captures.items():
        for name, tensor in parts.items():
            if tensor_hash(tensor) != metrics["captures_sha256"][key][name]:
                raise ValueError("Saved captures differ from their recorded hashes")
    targets = {key: sequence_targets(panels[name]["tokens"])[TARGET]
               for key, name in (("fit", "probe_fit"), ("test", "evaluation"))}
    design = RidgeDesign(captures["fit"]["means"], captures["test"]["means"], cfg["ridge_penalty"])
    candidate = design.score(targets["fit"], targets["test"],
                             derive_seed(adapter, seed, cell["id"], "property-candidate"),
                             [cfg["property_matched_nulls"]])
    recorded = metrics["controls"]["property"]["candidate"]
    if candidate != recorded:
        raise ValueError("Property candidate differs from the saved captures")
    classes = {key: torch.as_tensor(next_byte_class(panels[name]["next_tokens"].numpy()))
               for key, name in (("fit", "probe_fit"), ("test", "evaluation"))}
    device = resolve_device(metrics["device"])
    mlp_cfg = {"mlp_hidden": cfg["mlp_hidden"], "mlp_lr": cfg["mlp_lr"],
               "mlp_row_batch": cfg["mlp_row_batch"]}
    width = captures["fit"]["tokens"].shape[-1]
    structure_candidate = structure_score(
        captures["fit"]["tokens"].reshape(-1, width).to(device), classes["fit"].reshape(-1).to(device),
        captures["test"]["tokens"].reshape(-1, width).to(device), classes["test"].reshape(-1).to(device),
        [cfg["structure_matched_nulls"]], mlp_cfg,
        derive_seed(adapter, seed, cell["id"], "structure-candidate"), device)
    if structure_candidate != metrics["controls"]["structure"]["candidate"]:
        raise ValueError("Structure candidate differs from the saved captures")
    return {"property_candidate_recomputed": True, "structure_candidate_recomputed": True,
            "recompute_device": str(device)}


def check_target_controls(target_controls, cfg, matched_nulls, structure):
    score_check = check_structure_score if structure else check_score
    score_check(target_controls["candidate"], [matched_nulls])
    trials = target_controls["no_signal_trials"]
    if [t["repetition"] for t in trials] != list(range(cfg["outer_null_repeats"])):
        raise ValueError("Missing or duplicated no-signal trial")
    for trial in trials:
        score_check(trial, [matched_nulls])
    if [d["dose"] for d in target_controls["positive_controls"]] != [0., *cfg["positive_doses"]]:
        raise ValueError("Missing control dose")
    for dose in target_controls["positive_controls"]:
        score_check(dose["score"], [matched_nulls])
        for support in dose["support"].values():
            if (not support["support_valid"] or not 0 <= support["maximum_normalized_rounding_error"] <= 1e-3
                    or (dose["dose"] == 0 and support["zero_dose_exact"] is not True)):
                raise ValueError("Invalid quantizer-support record")
    verdict = target_controls["binding_qualification"]
    band, floor = verdict["null_band"], verdict["positive_floor"]
    over_band = [t for t in trials if abs(t["estimators"][0]["advantage"]) > band]
    below_floor = [d for d in target_controls["positive_controls"]
                   if d["dose"] > 0 and d["score"]["estimators"][0]["advantage"] < floor]
    if verdict["qualified"] != (not over_band and not below_floor):
        raise ValueError("Recorded qualification verdict does not follow from the raw scores")


def analyze(label):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", label):
        raise ValueError("Invalid campaign label")
    path = ROOT / "outputs/runs" / label
    summary = json.loads((path / "summary.json").read_text())
    if (summary["protocol_id"] != "p4-qualification-0.1"
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
        if (cfg_base != snap_base or manifest["cohort"] != "calibration"
                or manifest["scientific_evidence"] is not True
                or manifest["source_record"] != str((path / "source.json").relative_to(ROOT))
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
                                                       manifest["inputs"]["cell"], manifest["seed"])
            for target in TARGETS:
                record["targets"][target] = {
                    "candidate_advantage": controls[target]["candidate"]["estimators"][0]["advantage"],
                    "candidate_score": controls[target]["candidate"]["estimators"][0],
                    "binding_qualification": controls[target]["binding_qualification"]}
        run_records[run_id] = record
    if (len(statuses) != summary["expected_runs"]
            or statuses.count("completed") != summary["completed"]
            or statuses.count("failed") != summary["failed"]):
        raise ValueError("Incomplete campaign or mismatched run accounting")
    verdicts = []
    for cell in config_snapshot["cells"]:
        for target in TARGETS:
            rows = [{"seed": r["seed"], "binding": r["targets"][target]["binding_qualification"],
                     "candidate_advantage": r["targets"][target]["candidate_advantage"]}
                    for r in run_records.values()
                    if r["cell"] == cell["id"] and r["mode"] == "trained" and target in r["targets"]]
            missing = [s for s in config_snapshot["seeds"] if s not in {r["seed"] for r in rows}]
            failed = [r["seed"] for r in rows if not r["binding"]["qualified"]]
            qualified = not missing and not failed
            reasons = []
            if missing:
                reasons.append(f"missing seeds: {missing}")
            if failed:
                reasons.append(f"unqualified bindings on seeds: {failed}")
            verdicts.append({"cell": cell["id"], "target": target, "qualified": qualified,
                             "reason": "; ".join(reasons) if reasons else
                                       "all protocol seeds qualified",
                             "seeds": len(rows),
                             "bindings": [{"seed": r["seed"], "qualified": r["binding"]["qualified"],
                                           "candidate_advantage": r["candidate_advantage"]}
                                          for r in sorted(rows, key=lambda r: r["seed"])]})
    recomputed = sum(1 for r in run_records.values() if "recompute" in r)
    completed_split = sum(1 for r in run_records.values() if r["mode"] != "all_local")
    verified = {"label": label, **SCIENTIFIC_FLAGS, "expected_runs": summary["expected_runs"],
                "completed": summary["completed"], "failed": summary["failed"],
                "source_archive_and_artifacts_verified": True, "captures_hashes_verified": True,
                "candidate_scores_recomputed": recomputed == completed_split,
                "split_runs_recomputed": recomputed,
                "stored_estimates_verified": True,
                "qualification_verdicts": verdicts,
                "git_commit": source["git_commit"], "source_tree_sha256": source["source_tree_sha256"],
                "analysis_module_sha256": digest(__file__), "unevaluated": UNEVALUATED}
    lines = [f"# Instrument qualification dress rehearsal (calibration cohort): {label}", "",
             f"Completed {summary['completed']}/{summary['expected_runs']} runs; failures: {summary['failed']}.",
             f"Execution revision: `{source['git_commit']}`. Manifests, source archives, artifact bindings,",
             "capture hashes, stored control scores, and candidate-score recomputation verified.", "",
             "Cohort `calibration`, scientific_evidence true, scoped to instrument qualification only.",
             "No defense-success, utility, or causal-gain claim; privacy remains **undetermined**.",
             "Candidate advantages on calibration models are exploratory observations, not pilot gate outcomes.", "",
             "| Cell | Target | Seeds | Verdict |",
             "|---|---|---|---|"]
    for v in verdicts:
        lines.append(f"| {v['cell']} | {v['target']} | {v['seeds']}/{len(config_snapshot['seeds'])} | "
                     f"{'qualified' if v['qualified'] else 'unqualified: ' + v['reason']} |")
    lines += ["", "A binding is qualified iff all 16 no-signal trials stay within the ±0.02 band and both",
              "nonzero positive doses clear the target floor (property 0.10, structure 0.02). A cell-target",
              "verdict is qualified iff all four calibration seeds' bindings qualify; unqualified coverage",
              "leaves the corresponding pilot coverage undetermined.", "",
              "## Candidate advantages per seed (descriptive)", "",
              "| Cell | Target | Seed | Candidate advantage |",
              "|---|---|---|---|"]
    for v in verdicts:
        for b in v["bindings"]:
            lines.append(f"| {v['cell']} | {v['target']} | {b['seed']} | {b['candidate_advantage']:.4f} |")
    lines += ["", CLAIM_SCOPE, "", "Unevaluated: " + "; ".join(UNEVALUATED) + ".", "",
              "See [verification](verification.json), [summary](summary.json), and [source record](source.json).", ""]
    write_json(path / "verification.json", verified)
    (path / "report.md").write_text("\n".join(lines))
    return verified
