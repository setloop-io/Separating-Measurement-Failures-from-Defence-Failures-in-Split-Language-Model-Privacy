#!/usr/bin/env python3
"""Retrospective, read-only reanalysis of archived split-LLM privacy evidence.

Run from any directory with Python >=3.10. Writes derived outputs to data/ and
syncs generated LaTeX count macros to paper/results_macros.tex. No training,
attack fitting, capture collection, network access, or imports from archived
experimental implementations are performed.
"""

from __future__ import annotations

import collections
import csv
import datetime as dt
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import statistics


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PAPER = ROOT / "paper"
# Fixed archived conventions, not values estimated or tuned in this reanalysis.
R1_T975_DF3 = 3.182446305284263
P4_T95_DF2 = 2.9199855803537242
TOL = 1e-9


class Sources:
    def __init__(self):
        self.records = {}

    def text(self, relative, level, purpose):
        path = ROOT / relative
        blob = path.read_bytes()
        record = self.records.setdefault(str(relative), {
            "path": str(relative), "sha256": hashlib.sha256(blob).hexdigest(),
            "bytes": len(blob), "source_levels": [], "used_for": [],
        })
        assert record["sha256"] == hashlib.sha256(blob).hexdigest(), f"Input changed: {relative}"
        if level not in record["source_levels"]:
            record["source_levels"].append(level)
        if purpose not in record["used_for"]:
            record["used_for"].append(purpose)
        return blob.decode("utf-8")

    def json(self, relative, level, purpose):
        return json.loads(self.text(relative, level, purpose))


S = Sources()
CHECKS = []


def check(name, condition):
    if not condition:
        raise AssertionError(name)
    CHECKS.append(name)


def close(a, b, tol=TOL):
    return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)


def interval(values, critical):
    n = len(values)
    if n < 2:
        raise ValueError("An interval needs at least two independent seed observations")
    mu = statistics.mean(values)
    sd = statistics.stdev(values)
    half = critical * sd / math.sqrt(n)
    return {"n": n, "mean": mu, "sd": sd, "se": sd / math.sqrt(n),
            "lower": mu - half, "upper": mu + half, "critical_t": critical, "df": n - 1}


def csv_rows(path, level, purpose):
    return list(csv.DictReader(io.StringIO(S.text(path, level, purpose))))


def write_csv(name, rows, fields):
    with (DATA / name).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({key: str(value).lower() if isinstance(value, bool) else value
                          for key, value in row.items()} for row in rows)


def audit_r1():
    prefix = "reproduction/trilemma-campaign/evidence/r1-exploration-20260917/"
    source = prefix + "analysis/exploratory_analysis.json"
    original = S.json(source, "archived_analysis_with_per_seed_losses", "R1 gate recomputation")
    cell_csv = csv_rows("evidence/trilemma-campaign/r1_cell_results.csv",
                        "generated_cell_table", "cross-check recomputed R1 cells")
    seed_csv = csv_rows("evidence/trilemma-campaign/r1_seed_results.csv",
                        "generated_seed_table", "cross-check archived per-seed losses")
    by_csv = {r["cell"]: r for r in cell_csv}
    by_seed_csv = {(r["cell"], int(r["seed"])): r for r in seed_csv}
    qual_source = "reproduction/trilemma-campaign/evidence/r1-qualification-20260916/calibration_analysis_summary.json"
    qual = S.json(qual_source, "archived_qualification_summary", "count frozen qualification coverage")
    S.json("reproduction/trilemma-campaign/evidence/r1-qualification-20260916/attack-spec-frozen.json",
           "frozen_specification", "record qualification freeze provenance")
    power_source = prefix + "confirmation_power.json"
    power = S.json(power_source, "prospective_planning_output", "source-reported conditional power bound")
    S.text("evidence/trilemma-campaign/R1_CONFIRMATION_REVIEW.md", "planning_review",
           "confirmation was proposed and not frozen/admitted")
    check("R1 power input hash matches archived analysis",
          power["input"]["sha256"] == S.records[source]["sha256"])
    cells = []
    for name, cell in original["cells"].items():
        seed_values = []
        for row in cell["seeds"]:
            loss = row["losses"]
            values = {"seed": row["seed"],
                      "utility": loss["trained"] - loss["all_local"],
                      "gain": loss["frozen_random"] - loss["trained"],
                      "removal_benefit": loss["removed_remote"] - loss["trained"]}
            old = by_seed_csv[(name, row["seed"])]
            check(f"R1 {name}/{row['seed']} seed contrasts agree",
                  close(values["utility"], old["replacement_penalty"]) and
                  close(values["gain"], old["gain_against_frozen"]) and
                  close(values["removal_benefit"], old["gain_against_removed"]))
            seed_values.append(values)
        u = interval([r["utility"] for r in seed_values], R1_T975_DF3)
        g = interval([r["gain"] for r in seed_values], R1_T975_DF3)
        b = interval([r["removal_benefit"] for r in seed_values], R1_T975_DF3)
        utility_pass = u["mean"] <= .10 and u["upper"] <= .12
        gain_pass = g["mean"] >= .01 and g["lower"] > 0
        combinations = []
        for key, combo in cell["privacy"]["combinations"].items():
            qualified = combo["qualified_in_frozen_specification"]
            per_seed = combo.get("per_seed", {})
            # Qualification and saved score arrays are consumed as archived
            # observations; the original attack estimator is not refitted here.
            complete = qualified and len(per_seed) == len(seed_values) and all(
                v.get("run_qualified") and v.get("advantage") is not None for v in per_seed.values())
            worst = max((v["advantage"] for v in per_seed.values()), default=None) if complete else None
            verdict = ("BREACH" if worst > combo["limit"] else "PASS") if complete else "UNDETERMINED"
            check(f"R1 {name}/{key} privacy rule agrees", verdict == combo["verdict"])
            combinations.append({"combination": key, "qualified": bool(qualified),
                                 "complete": bool(complete), "worst_seed_advantage": worst,
                                 "limit": combo["limit"], "verdict": verdict})
        breached = any(c["verdict"] == "BREACH" for c in combinations)
        privacy_status = "REJECTED" if breached else (
            "PASS" if all(c["verdict"] == "PASS" for c in combinations) else "UNDETERMINED")
        reasons = (["utility_gate"] if not utility_pass else []) + (["gain_gate"] if not gain_pass else [])
        if breached:
            reasons.append("privacy_qualified_breach")
        verdict = "REJECTED" if reasons else ("PASS" if privacy_status == "PASS" else "UNDETERMINED")
        check(f"R1 {name} verdict/reasons agree", verdict == cell["verdict"] and reasons == cell["reject_reasons"])
        check(f"R1 {name} privacy status agrees", privacy_status == cell["privacy"]["status"])
        check(f"R1 {name} table statistics agree",
              all(close(v, by_csv[name][k]) for k, v in {
                  "utility_mean": u["mean"], "utility_lower": u["lower"], "utility_upper": u["upper"],
                  "gain_mean": g["mean"], "gain_lower": g["lower"], "gain_upper": g["upper"]}.items()))
        cells.append({"cell": name, "verdict": verdict, "reasons": ";".join(reasons),
                      "privacy_status": privacy_status, "n_seeds": len(seed_values),
                      "utility_mean": u["mean"], "utility_lower": u["lower"], "utility_upper": u["upper"],
                      "utility_sd": u["sd"], "utility_pass": utility_pass,
                      "gain_mean": g["mean"], "gain_lower": g["lower"], "gain_upper": g["upper"],
                      "gain_pass": gain_pass, "benefit_over_removal_mean": b["mean"],
                      "benefit_over_removal_lower": b["lower"], "benefit_over_removal_upper": b["upper"],
                      "source_level": "recomputed_from_archived_per_seed_losses_and_probe_scores",
                      "seed_values": seed_values, "privacy_combinations": combinations, "sources": [source]})
    reason_counts = collections.Counter(r for cell in cells for r in cell["reasons"].split(";") if r)
    intersections = collections.Counter(cell["reasons"] for cell in cells)
    undetermined = [c["cell"] for c in cells if c["privacy_status"] == "UNDETERMINED"]
    check("R1 19 cells and overlapping reason counts", len(cells) == 19 and
          dict(reason_counts) == {"privacy_qualified_breach": 13, "utility_gate": 5, "gain_gate": 7})
    check("R1 six privacy-undetermined cells independently rejected", len(undetermined) == 6 and all(
        (not c["utility_pass"] or not c["gain_pass"]) for c in cells if c["cell"] in undetermined))
    span = next(c for c in cells if c["cell"] == "span-6")
    check("R1 span-6 passes both utility conditions", span["utility_pass"])
    fields = ["cell", "verdict", "reasons", "privacy_status", "n_seeds", "utility_mean", "utility_lower",
              "utility_upper", "utility_pass", "gain_mean", "gain_lower", "gain_upper", "gain_pass",
              "benefit_over_removal_mean", "benefit_over_removal_lower", "benefit_over_removal_upper", "source_level"]
    write_csv("r1_cells.csv", cells, fields)
    return {"cohort": "archived_exploratory", "new_independent_validation": False,
            "denominators": {"cells": len(cells), "independent_training_seeds_per_cell": 4,
                             "paired_cell_seed_records": len(seed_csv), "source_reported_completed_runs": original["accounting"]["completed_count"]},
            "reason_counts_overlapping": dict(reason_counts), "reason_intersections": dict(intersections),
            "privacy_undetermined_independently_rejected": undetermined,
            "qualification": {"qualified": len(qual["qualified_combinations"]),
                              "unqualified": len(qual["unqualified_combinations"]),
                              "total": len(qual["qualified_combinations"]) + len(qual["unqualified_combinations"]),
                              "source_level": "counts_from_archived_qualification_summary", "source": qual_source},
            "confidence_and_gates": {"interval": "two-sided 95% Student-t over four paired training-seed contrasts",
                                     "df": 3, "critical_t": R1_T975_DF3,
                                     "utility": "mean <= 0.10 nats AND upper interval endpoint <= 0.12 nats",
                                     "gain": "mean >= 0.01 nats AND lower interval endpoint > 0",
                                     "privacy": original["method"]["privacy_rule"],
                                     "not_in_cell_gate": "benefit over removed remote is reported separately"},
            "power_planning": {"source": power_source, "source_level": "source_reported_not_recomputed",
                               "proposed_n": power["proposed_n"], "endpoints": power["simultaneous_endpoint_count"],
                               "upper_all_19_rejections": power["family_plugin_power_bounds"]["1.0"]["all_19_rejected_probability_upper_bound"],
                               "scope": power["scope"], "cautions": power["planning_cautions"]},
            "span6": {"mean_margin_below_0_10": .10 - span["utility_mean"],
                      "upper_margin_below_0_12": .12 - span["utility_upper"], "utility_pass": True},
            "cells": cells}


def score_advantage(score, target):
    if target == "property":
        value = score["r"] - statistics.mean(score["r_null"])
    else:
        value = score["accuracy"] - statistics.mean(score["acc_null"])
    for estimator in score.get("estimators", []):
        check("P4 saved scalar agrees with saved score-array arithmetic", close(value, estimator["advantage"]))
    return value


def audit_p4():
    base = "reproduction/defense-redesign-pilot/outputs/runs/pilot-20260922-a/"
    verification = S.json(base + "verification.json", "archived_generated_verification", "cross-check defence-pilot gates and scores")
    report = S.text(base + "report.md", "archived_generated_report", "cross-check rounded publication table")
    summary = S.json(base + "summary.json", "archived_execution_summary", "defence-pilot run enumeration")
    S.text("evidence/defense-redesign-pilot/PILOT_DECISION.md", "archived_decision_record", "defence-pilot decision provenance")
    S.text("reproduction/defense-redesign-pilot/src/p4d/pilot_analysis.py", "archived_analysis_implementation", "frozen gate and confidence conventions")
    S.text("reproduction/defense-redesign-pilot/src/p4d/calibration.py", "archived_calibration_implementation", "floor/gate definitions")
    old_cells = {v["cell"]: v for v in verification["cell_verdicts"]}
    rows = {}
    for run_id in summary["run_ids"]:
        relative = base + run_id + "/metrics.json"
        record = S.json(relative, "archived_run_metrics_and_saved_score_arrays", "recompute defence-pilot losses, controls and gates")
        rows[(record["cell"], record["seed"], record["mode"])] = (record, relative)
    seeds = sorted({key[1] for key in rows})
    check("P4 66 records, three seeds", len(rows) == 66 and len(seeds) == 3)
    cells = []
    all_candidate_rows = []
    for name, old in old_cells.items():
        u_values, g_values, source_paths = [], [], []
        targets = {"property": [], "structure": []}
        for seed in seeds:
            trained, p = rows[(name, seed, "trained")]
            frozen, pf = rows[(name, seed, "frozen_random")]
            local, pl = rows[("all-local", seed, "all_local")]
            source_paths += [p, pf, pl]
            u_values.append(trained["final_eval_loss"] - local["final_eval_loss"])
            g_values.append(frozen["final_eval_loss"] - trained["final_eval_loss"])
            for target in targets:
                control = trained["controls"][target]
                advantage = score_advantage(control["candidate"], target)
                nulls = [score_advantage(trial, target) for trial in control["no_signal_trials"]]
                positives = [score_advantage(row["score"], target) for row in control["positive_controls"] if row["dose"] > 0]
                floor = .10 if target == "property" else .02
                qualified = len(nulls) == 16 and len(positives) == 2 and all(abs(x) <= .02 for x in nulls) and all(x >= floor for x in positives)
                check(f"P4 {name}/{seed}/{target} qualification agrees",
                      qualified == trained["binding_qualification"][target]["qualified"])
                targets[target].append({"seed": seed, "candidate_advantage": advantage,
                                        "qualified": qualified, "null_trials": len(nulls),
                                        "max_abs_null_advantage": max(map(abs, nulls)),
                                        "positive_dose_min_advantage": min(positives), "positive_floor": floor})
        u = interval(u_values, P4_T95_DF2)
        g_mean = statistics.mean(g_values)
        utility_pass = u["mean"] <= .10 and u["upper"] <= .12
        gain_pass = g_mean >= .01 and all(x > 0 for x in g_values)
        property_complete = all(b["qualified"] for b in targets["property"])
        structure_complete = all(b["qualified"] for b in targets["structure"])
        privacy = "undetermined" if not (property_complete and structure_complete) else (
            "passed" if all(b["candidate_advantage"] <= .02 for bs in targets.values() for b in bs) else "failed")
        eligible = utility_pass and gain_pass and privacy == "passed"
        check(f"P4 {name} gates agree with archived verification",
              utility_pass == old["utility"]["passed"] and gain_pass == old["gain"]["passed"] and
              privacy == old["privacy"]["status"] and eligible == old["pilot_eligible"] and
              close(u["mean"], old["utility"]["mean_delta_u"]) and close(u["upper"], old["utility"]["upper_95_limit"]))
        check(f"P4 {name} rounded report agrees", f"| {name} | no | {u['mean']:.4f} | {u['upper']:.4f} | {g_mean:.4f} | True | {privacy} |" in report)
        qualified_prop = [b["candidate_advantage"] for b in targets["property"] if b["qualified"]]
        cell = {"cell": name, "n_seeds": len(seeds), "utility_mean": u["mean"], "utility_sd": u["sd"],
                "utility_upper": u["upper"], "utility_pass": utility_pass,
                "gain_mean": g_mean, "gain_all_pairs_positive": all(x > 0 for x in g_values), "gain_pass": gain_pass,
                "privacy_status": privacy, "property_worst_advantage": max(b["candidate_advantage"] for b in targets["property"]),
                "property_worst_qualified_advantage": max(qualified_prop) if qualified_prop else None,
                "property_qualified_bindings": sum(b["qualified"] for b in targets["property"]),
                "structure_worst_advantage": max(b["candidate_advantage"] for b in targets["structure"]),
                "structure_qualified_bindings": sum(b["qualified"] for b in targets["structure"]),
                "pilot_eligible": eligible, "source_level": "recomputed_from_saved_run_losses_and_score_arrays",
                "utility_per_seed": dict(zip(map(str, seeds), u_values)), "gain_per_seed": dict(zip(map(str, seeds), g_values)),
                "targets": targets, "sources": sorted(set(source_paths))}
        cells.append(cell)
    # The complete 126-row score table covers 63 split records, including the
    # frozen/removed controls. Recompute its saved candidate score arithmetic.
    for (cell, seed, mode), (record, source) in rows.items():
        if mode == "all_local":
            continue
        for target in ("property", "structure"):
            advantage = score_advantage(record["controls"][target]["candidate"], target)
            match = next(r for r in verification["advantage_table"] if
                         (r["cell"], r["seed"], r["mode"], r["target"]) == (cell, seed, mode, target))
            check("P4 candidate table agrees with score arrays", close(advantage, match["candidate_advantage"]))
            all_candidate_rows.append({"cell": cell, "seed": seed, "mode": mode, "target": target, "advantage": advantage})
    fields = ["cell", "n_seeds", "utility_mean", "utility_sd", "utility_upper", "utility_pass", "gain_mean",
              "gain_all_pairs_positive", "gain_pass", "privacy_status", "property_worst_advantage",
              "property_worst_qualified_advantage", "property_qualified_bindings", "structure_worst_advantage",
              "structure_qualified_bindings", "pilot_eligible", "source_level"]
    write_csv("p4_cells.csv", cells, fields)
    return {"cohort": "archived_exploratory", "new_independent_validation": False,
            "denominators": {"cells": 7, "independent_training_seeds_per_cell": 3, "run_records": len(rows),
                             "split_run_records": 63, "candidate_score_records": len(all_candidate_rows),
                             "matched_nulls_property_per_score": 16, "matched_nulls_structure_per_score": 4,
                             "qualification_no_signal_trials_per_binding": 16,
                             "source_reported_windows_per_panel": 16384,
                             "note": "Windows, matched nulls and candidate scores do not increase the training-seed n."},
            "counts": {"utility_pass": sum(c["utility_pass"] for c in cells), "utility_fail": sum(not c["utility_pass"] for c in cells),
                       "gain_pass": sum(c["gain_pass"] for c in cells), "privacy": dict(collections.Counter(c["privacy_status"] for c in cells)),
                       "pilot_eligible": sum(c["pilot_eligible"] for c in cells)},
            "confidence_and_gates": {"utility_upper": "one-sided 95% upper Student-t limit over three paired training-seed contrasts",
                                     "df": 2, "critical_t": P4_T95_DF2, "utility": "mean <=0.10 AND upper95 <=0.12 nats",
                                     "gain": "mean >=0.01 nats AND all three paired gains strictly >0",
                                     "privacy": "worst seed <=0.02 on every required target with all bindings qualified; missing qualification yields undetermined",
                                     "property_positive_control_floor": .10, "structure_positive_control_floor": .02,
                                     "null_band": [-.02, .02], "privacy_acceptance": .02,
                                     "units": {"property": "Pearson correlation advantage over matched null (punctuation fraction)",
                                               "structure": "five-class accuracy advantage over matched null (next-byte class)"}},
            "cells": cells, "scope_note": "No positive complete-transcript defence claim; unevaluated channels remain undetermined.",
            "unevaluated": verification["unevaluated"]}


def audit_e8():
    source = "reproduction/structural-e8/paper-data/collected/h100-2/issues/issue_e8_27b_20260807/e8_robustness_27b_bf16_densek_20260807T230529Z.json"
    raw = S.json(source, "archived_attack_summary", "extract 27B crossings and passive-session outcomes")
    S.text("evidence/structural-e8/EXPERIMENT_STORY.md", "archived_experiment_narrative", "orthogonal-wrapper context and scope provenance")
    depths = []
    for depth in raw["config"]["depths"]:
        rows = [r for r in raw["summary"] if r["depth"] == depth]
        baseline = next(r for r in rows if r["attack"] == "0_baseline_no_defense")
        pooled = next(r for r in rows if r["attack"] == "2_session_rotation")
        crossing = raw["oracle_k50"][str(depth)]
        passive = [r for r in rows if r["attack"].startswith(("0_obfuscated", "1a_", "1b_", "2_"))]
        depths.append({"depth": depth, "hidden_width": 5120, "k50": crossing["k50_interpolated"],
                       "k50_bracket": crossing["k50_bracket"], "k50_method": crossing["k50_method"],
                       "k50_threshold_absolute_top1_pct": crossing["k50_threshold_pct"],
                       "k50_over_H": crossing["k50_interpolated"] / 5120,
                       "baseline_top1_pct": baseline["top1_mean"], "baseline_decoder_seed_count": baseline["n_seeds"],
                       "pooled_session_top1_pct": pooled["top1_mean"], "pooled_session_decoder_seed_count": pooled["n_seeds"],
                       "per_session_top1_pct": pooled["per_session_top1"],
                       "passive_summary_cells": [{k: r[k] for k in ("attack", "setting", "top1_mean", "top1_std", "n_seeds")} for r in passive],
                       "source_level": "extracted_archived_results_not_new_attack_execution", "source": source})
    return {"model_source_label": raw["config"]["model"], "threat_model": raw["threat_model"],
            "denominators": {"depths": len(depths), "victim_documents": raw["config"]["victim_docs"],
                             "decoder_seed_count_baseline": len(raw["config"]["seeds"]), "sessions_per_pooled_score": 3,
                             "note": "Decoder seeds and sessions are not independent training/model replications; key seed material is deliberately not copied."},
            "depths": depths, "largest_pooled_passive_session_top1_pct": max(d["pooled_session_top1_pct"] for d in depths),
            "largest_individual_passive_session_top1_pct": max(v for d in depths for v in d["per_session_top1_pct"]),
            "interpretation": "Depth-specific finite attack measurements; not a universal <=3.7% bound or a general K50=0.09H law."}


def audit_er():
    base = "reproduction/structural-e8/paper-data/collected/toula/er_20260811/"
    manifest = S.json(base + "COLLECTION_MANIFEST.json", "accepted_collection_manifest", "latest accepted status and threat scope")
    ledger_source = "evidence/structural-e8/EVIDENCE_EXECUTION_LEDGER.md"
    S.text(ledger_source, "current_evidence_ledger", "distinguish per-session versus per-epoch claims")
    runs = []
    identities = []
    for run in manifest["runs"]:
        relative = base + run["run_dir"] + "/summary.json"
        summary = S.json(relative, "archived_deployment_summary", "extract cadence and accepted session summaries")
        sessions = []
        for session in summary["sessions"]:
            index = session["session"]
            identity_path = base + run["run_dir"] + f"/session_{index}/identity.json"
            identity = S.json(identity_path, "archived_identity_and_throughput_record", "extract matched static and ratchet throughput")
            check("E-R summary identity agrees with identity record", session["C_identity"] == identity["ratchet_C_identity"])
            check("E-R summary rate agrees with identity record", close(session["ratchet_tps"], identity["ratchet_C_tps"]))
            record = {"session": index, "identity_restored": identity["ratchet_C_identity"],
                      "epochs_advanced": identity["epochs_advanced"], "static_restored_tps": identity["static_C_tps"],
                      "ratchet_restored_tps": identity["ratchet_C_tps"],
                      "rate_change_fraction": identity["ratchet_C_tps"] / identity["static_C_tps"] - 1,
                      "baseline_measured_in_session": identity.get("A_tps") is not None,
                      "source_level": "extracted_archived_measurement", "source": identity_path}
            sessions.append(record)
            identities.append(record)
        runs.append({"run": run["run_dir"], "evidence_status": run["evidence_status"],
                     "limitation": run.get("limitation"), "ratchet_n": summary["config"]["ratchet_n"],
                     "budget_b": summary["config"]["budget_b"], "rtt_ms": summary["config"]["rtt_ms"],
                     "sessions": sessions, "source": relative})
    check("E-R 20 of 24 identities restored", sum(s["identity_restored"] for s in identities) == 20 and len(identities) == 24)
    anchor = next(r for r in runs if r["run"] == "er_n16_rtt0_20260810T213415")["sessions"][0]
    check("E-R measured 2.8 versus 7.6 anchor", close(anchor["ratchet_restored_tps"], 2.8) and close(anchor["static_restored_tps"], 7.6))
    return {"source_level": "extracted_archived_measurements_and_accepted_disposition",
            "accepted_evidence_status": manifest["evidence_status"], "scope_note": manifest["scope_note"],
            "known_limitations": manifest["known_limitations"], "runs": runs,
            "denominators": {"runs": len(runs), "sessions": len(identities), "identities_restored": sum(s["identity_restored"] for s in identities),
                             "runs_supporting_with_known_limitation": sum(r["evidence_status"] == "supporting_with_known_limitation" for r in runs),
                             "note": "RTT twins and repeated deterministic runs are not claimed as new independent validation."},
            "n16_rtt0_session1_anchor": {"ratchet_tps": anchor["ratchet_restored_tps"], "static_tps": anchor["static_restored_tps"],
                                         "throughput_reduction_pct": -100 * anchor["rate_change_fraction"],
                                         "source_level": "exact_values_as_recorded_in_raw_identity_json_not_unrounded_hardware_truth",
                                         "source": anchor["source"]},
            "zero_rotation_control_sessions": sum(s["epochs_advanced"] == 0 for s in identities)}


def main():
    DATA.mkdir(exist_ok=True)
    metadata = {"schema_version": "split-llm-privacy-reanalysis-1.0",
                "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "python_version": platform.python_version(),
                "dependencies": "Python standard library only",
                "analysis_type": "retrospective_existing_artifact_secondary_analysis",
                "new_training_or_attack_execution": False,
                "new_independent_validation_cohort": False,
                "input_root": ".",
                "outputs_root": "data",
                "reproduction_claim": "Arithmetic and decision-rule reanalysis of archived records, not a rerun of training, capture collection or attack fitting."}
    result = {"metadata": metadata, "r1": audit_r1(), "p4": audit_p4(), "e8": audit_e8(), "er": audit_er()}
    # These sources support the other case-study anchors. Their experimental
    # results are source-reported, not newly reproduced by this script.
    supporting = {
        "E1": ["evidence/decoy-gradient/paper1-main.tex", "evidence/decoy-gradient/E1_GRADIENT_CHANNEL_2026-08-22.md"],
        "B1": ["evidence/boundary-replacement/independent_evidence_review_2026-09-09/VERDICTS.md",
               "evidence/boundary-replacement/independent_evidence_review_2026-09-09/recomputed_evidence.json"],
    }
    for anchor, paths in supporting.items():
        for path in paths:
            S.text(path, "source_reported_supporting_case_study", f"{anchor} case-study anchor and publication source index")
    result["supporting_document_sources"] = {
        anchor: {"paths": paths, "source_level": "source_reported_not_recomputed_here"}
        for anchor, paths in supporting.items()}
    mi_sources = [
        ("evidence/privacy-contract/privacy-contract-draft.tex", "companion_manuscript",
         "check acknowledged MI lower-bound direction and limitations"),
        ("evidence/privacy-contract/KEY_NUMBERS.md", "companion_key_numbers",
         "source-reported MINE and cross-entropy lower-bound readouts"),
        ("evidence/structural-e8/LATENT_CAMPAIGN_REPORT_V7_V9.md", "archived_ablation_report",
         "finite architecture ablation records"),
    ]
    for path, level, purpose in mi_sources:
        S.text(path, level, purpose)
    result["mi_logic"] = {"source_level": "analytic_counterexample_not_an_experimental_result",
                          "sources": [p for p, _, _ in mi_sources],
                          "premises": ["L <= I", "L=0", "B=0.25"],
                          "compatible_cases": [{"L": 0, "I": 0}, {"L": 0, "I": 100}],
                          "conclusion": "L<=B does not entail I<=B. A near-zero lower bound cannot certify a leakage ceiling."}
    records = [dict(source_id=f"S{index:03d}", **record) for index, record in enumerate(
        sorted(S.records.values(), key=lambda r: r["path"]), 1)]
    macros = {
        "RoneCells": result["r1"]["denominators"]["cells"],
        "RoneQualifiedBreaches": result["r1"]["reason_counts_overlapping"]["privacy_qualified_breach"],
        "RoneUtilityFailures": result["r1"]["reason_counts_overlapping"]["utility_gate"],
        "RoneGainFailures": result["r1"]["reason_counts_overlapping"]["gain_gate"],
        "RoneUndetermined": len(result["r1"]["privacy_undetermined_independently_rejected"]),
        "PfourCells": result["p4"]["denominators"]["cells"],
        "PfourUtilityFailures": result["p4"]["counts"]["utility_fail"],
        "PfourPrivacyFailures": result["p4"]["counts"]["privacy"]["failed"],
        "PfourPrivacyUndetermined": result["p4"]["counts"]["privacy"]["undetermined"],
        "PinnedInputs": len(records),
        "ConsistencyChecks": f"{len(CHECKS):,}",
    }
    macro_text = (
        "% Generated by analysis/recompute.py; retrospective archived-evidence counts.\n" +
        "".join(f"\\newcommand{{\\{name}}}{{{value}}}\n" for name, value in macros.items())
    )
    (DATA / "results_macros.tex").write_text(macro_text, encoding="utf-8")
    if PAPER.is_dir():
        (PAPER / "results_macros.tex").write_text(macro_text, encoding="utf-8")
    result["validation"] = {"checks_passed": len(CHECKS), "checks": CHECKS,
                            "limits": ["No raw tensor captures were loaded or scored.",
                                       "No original experimental implementation was imported or executed.",
                                       "Recorded qualification evidence is checked through saved score arrays; estimator fitting is not repeated.",
                                       "Historical evidence-status conflicts are resolved using the later accepted collection manifest, not by counting duplicate invalid archives."]}
    (DATA / "reanalysis_results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = {"schema_version": "split-llm-privacy-source-manifest-1.0", "input_root": ".",
                "script": {"path": "analysis/recompute.py", "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
                "input_file_count": len(records), "inputs": records,
                "note": "Hashes bind every archived record consumed by analysis/recompute.py and cited in Appendix A; this is not a revalidation of every original run manifest or tensor capture."}
    (DATA / "source_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv("source-index.csv", [{**r, "source_levels": ";".join(r["source_levels"]), "used_for": ";".join(r["used_for"])} for r in records],
              ["source_id", "path", "source_levels", "sha256", "bytes", "used_for"])
    print("Split-LLM privacy reanalysis: retrospective archived-evidence verification")
    print("R1:", result["r1"]["denominators"], result["r1"]["reason_counts_overlapping"])
    print("R1 reason intersections:", result["r1"]["reason_intersections"])
    print("R1 span-6 utility:", result["r1"]["span6"])
    print("P4:", result["p4"]["counts"])
    print("E8 K50:", [(d["depth"], d["k50"], d["k50_over_H"]) for d in result["e8"]["depths"]])
    print("E8 largest pooled / individual session top1:", result["e8"]["largest_pooled_passive_session_top1_pct"], result["e8"]["largest_individual_passive_session_top1_pct"])
    print("E-R:", result["er"]["denominators"], result["er"]["n16_rtt0_session1_anchor"])
    print(f"PASS: {len(CHECKS)} internal arithmetic/disposition checks; {len(records)} input files hashed.")
    print("Outputs:", ", ".join(p.name for p in sorted(DATA.iterdir()) if p.is_file()))


if __name__ == "__main__":
    main()
