"""Fresh-seed validation checker for the selected operating point.

Validates a fresh-seed validation results dir (seeds {44, 55, 77},
same cell format as the bottleneck sweep, plus a white-box addendum
tier) against the mechanical selection record
(llm_attacker.operating_point_selection.v1):

  * FRESH-ONLY rule: the validation plan's seed set must be exactly the
    fresh seeds; selection-seed results are never reused here;
  * EXACT-COVERAGE rule: the fresh plan must cover EXACTLY the selected
    configurations plus the paired controls — every selected split must
    be present, each with the none control, the pca-compress control at
    the selected dim, and the selected (method, dim) cell at every fresh
    seed; anything beyond that (an empty plan, an omitted selected
    split, extra non-selected cells) fails closed;
  * paired undefended controls (method "none") and pca-compress controls
    must be present and COMPLETE for every fresh seed at every selected
    split — missing controls are reported as failures, never skipped;
  * every planned cell is loaded fail-closed (same checks as the sweep
    analyzer: provenance hash, evidence journal, utility, matched
    access, valid arm values, gates); any failed cell makes the overall
    validation outcome "failed";
  * ALL eligibility conditions are re-evaluated on the fresh data only:
    (a) the utility gate (recomputed from raw perplexities vs the paired
        none control),
    (b) PCA-superiority of the STRONGEST OBSERVED leakage across all
        measured arms (base tier + white-box tier, recomputed against
        the fresh pca-compress control, bootstrap AND all-seeds-positive
        conditions), and
    (c) the compute factor (<= 2x the pca-compress cell's train+attack
        seconds across all measured arms);
  * RETEST-ONLY rule: the selection record's chosen configuration is
    loaded verbatim and only that configuration is tested. This tool
    never reopens selection, never ranks candidates, and never changes
    thresholds on fresh data; the emitted `selected` per split is
    byte-identical to the input record's.

Emits llm_attacker.fresh_seed_validation.v1. Like the selector, this
tool never freezes anything.

Usage:
  python experiments/check_fresh_seed_validation.py
      --root <validation-dir> --selection <selection.json>
      --output validation.json [--markdown validation.md]
"""
import argparse
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_bottleneck_sweep import (  # noqa: E402
    load_cells, load_whitebox_cells, check_matched_access)
from analyze_defended_pilot import percentile_bootstrap  # noqa: E402
from select_operating_point import (  # noqa: E402
    PRIMARY_GATE, SECONDARY_GATE, COMPUTE_FACTOR, ACCESS_LABELS,
    V3_RULE, measured_leakage)

SCHEMA = "llm_attacker.fresh_seed_validation.v1"
SELECTION_SCHEMA = "llm_attacker.operating_point_selection.v1"
PLAN_SCHEMA = "llm_attacker.bottleneck_sweep.v1"
FRESH_SEEDS = (44, 55, 77)
CONTROL_METHOD = "none"
PCA_CONTROL = "pca-compress"
COTRAIN_METHOD = "tb-cotrain"

RETEST_ONLY_RULE = (
    "retest only: the selection record's chosen configuration is loaded "
    "verbatim and only that configuration is tested against the fresh "
    "seeds; this checker never reopens selection, never ranks "
    "candidates, and never changes thresholds on fresh data")

COVERAGE_RULE = (
    "exact coverage: the fresh plan must cover EXACTLY the selected "
    "configurations plus the paired none/pca-compress controls (and the "
    "matching white-box addendum cells) at every fresh seed — an empty "
    "plan, an omitted selected split, or any extra cell beyond the "
    "selected configuration plus controls fails closed")

COTRAIN_WB_NA = (
    "tb-cotrain training-phase white-box cells are NOT_EVALUABLE: the "
    "saved checkpoint is the FINAL co-trained encoder, mismatched with "
    "the capture-time encoder; the fresh statistic runs over the "
    "remaining measured arms")


def _strongest_observed(cell, wb_cell):
    """Strongest observed attack result for one fresh cell-seed across
    all measured arms (base tier + white-box tier when evaluable)."""
    row = {"forward": cell["attacks"]["forward"]["accuracy"],
           "sipit": cell["attacks"]["sipit"]["accuracy"],
           "da_forward": cell["attacks"]["defense_aware"]["forward"]
           ["accuracy"],
           "da_sipit": cell["attacks"]["defense_aware"]["sipit"]
           ["accuracy"]}
    if wb_cell is not None:
        arms = wb_cell["attacks"]
        row.update({
            "wb_inverter_forward": arms["whitebox_inverter"]["forward"]
            ["accuracy"],
            "wb_inverter_sipit": arms["whitebox_inverter"]["sipit"]
            ["accuracy"],
            "wb_opt_forward": arms["whitebox_optimization"]["forward"]
            ["accuracy"],
            "wb_opt_snap": arms["whitebox_optimization"]["snap"]
            ["accuracy"]})
    return measured_leakage(row)


def validate(root, selection):
    root = Path(root)
    if selection.get("schema") != SELECTION_SCHEMA:
        raise ValueError(f"unexpected selection schema "
                         f"{selection.get('schema')!r}")
    plan = json.loads((root / "plan.json").read_text())
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError(f"unexpected validation plan schema "
                         f"{plan.get('schema')!r}")
    failures = []
    plan_seeds = sorted(plan.get("seeds") or [])
    if plan_seeds != sorted(FRESH_SEEDS):
        failures.append({"cell": "plan",
                         "reason": f"fresh-only rule violated: plan seeds "
                                   f"{plan_seeds} != fresh seeds "
                                   f"{list(FRESH_SEEDS)}; selection-seed "
                                   "results are never reused"})
    cells, load_failures = load_cells(root, plan)
    failures += load_failures
    if cells:
        try:
            check_matched_access(cells)
        except ValueError as exc:
            failures.append({"cell": "plan", "reason": str(exc)})
    wb_cells, wb_failures, wb_not_evaluable = load_whitebox_cells(root)
    failures += wb_failures
    selected = {}
    for key, entry in (selection.get("splits") or {}).items():
        selected[int(key)] = (entry or {}).get("selected")
    sel_splits = sorted(selected)
    splits = sorted(plan.get("splits") or [])
    if splits != sel_splits:
        failures.append({
            "cell": "plan",
            "reason": f"fresh plan coverage violated: plan splits "
                      f"{splits} != the selection record's splits "
                      f"{sel_splits}; every selected split must be "
                      "retested, nothing omitted"})
    # -- exact coverage: selected configurations + controls, nothing more
    controls = {}
    defended = {}
    for c in cells:
        if c["method"] == CONTROL_METHOD:
            controls.setdefault((c["split"], c["seed"]), []).append(c)
        else:
            defended[(c["split"], c["seed"], c["method"], c["dim"])] = c
    for split in splits:
        for seed in FRESH_SEEDS:
            group = controls.get((split, seed)) or []
            if len(group) != 1:
                failures.append({
                    "cell": f"split{split}/seed-{seed}/{CONTROL_METHOD}",
                    "reason": "paired undefended control missing, "
                              "duplicated, or not COMPLETE (required per "
                              "fresh seed)"})
    expected_defended = set()
    for split in sel_splits:
        sel = selected[split]
        if sel is None:
            continue
        for seed in FRESH_SEEDS:
            expected_defended.add((split, seed, PCA_CONTROL, sel["dim"]))
            expected_defended.add((split, seed, sel["method"],
                                   sel["dim"]))
    for key in sorted(expected_defended - set(defended)):
        failures.append({
            "cell": f"split{key[0]}/seed-{key[1]}/dim-{key[3]}/{key[2]}",
            "reason": "planned fresh cell missing or not COMPLETE; the "
                      "fresh plan must cover every selected "
                      "configuration and its pca-compress control"})
    for key in sorted(set(defended) - expected_defended):
        failures.append({
            "cell": f"split{key[0]}/seed-{key[1]}/dim-{key[3]}/{key[2]}",
            "reason": "cell is beyond the selected configuration plus "
                      "controls (retest-only coverage rule); extra cells "
                      "are never loaded"})
    # -- white-box addendum coverage (needed for the strongest-observed
    # statistic; NOT_EVALUABLE for co-trained encoders)
    wb_plan_present = (root / "whitebox" / "plan.json").is_file()
    wb_by_key = {(c["split"], c["seed"], c["method"], c["dim"]): c
                 for c in wb_cells}
    needs_wb = any(sel is not None and sel["method"] != COTRAIN_METHOD
                   for sel in selected.values())
    has_selection = any(sel is not None for sel in selected.values())
    if not wb_plan_present:
        if needs_wb:
            failures.append({
                "cell": "whitebox/plan",
                "reason": "fresh white-box addendum tier missing; the "
                          "strongest-observed statistic cannot be "
                          "recomputed over all measured arms (fail "
                          "closed)"})
    elif has_selection:
        expected_wb = set()
        for split in sel_splits:
            sel = selected[split]
            if sel is None:
                continue
            for seed in FRESH_SEEDS:
                expected_wb.add((split, seed, PCA_CONTROL, sel["dim"]))
                if sel["method"] != COTRAIN_METHOD:
                    expected_wb.add((split, seed, sel["method"],
                                     sel["dim"]))
        for key in sorted(expected_wb - set(wb_by_key)):
            failures.append({
                "cell": f"whitebox/split{key[0]}/seed-{key[1]}/"
                        f"{key[2]}-{key[3]}",
                "reason": "fresh white-box cell missing or not "
                          "COMPLETE (required for the "
                          "strongest-observed statistic)"})
        for key in sorted(set(wb_by_key) - expected_wb):
            failures.append({
                "cell": f"whitebox/split{key[0]}/seed-{key[1]}/"
                        f"{key[2]}-{key[3]}",
                "reason": "white-box cell is beyond the selected "
                          "configuration plus controls (retest-only "
                          "coverage rule)"})
    split_records = {}
    for split in sel_splits:
        sel = selected[split]
        entry = {"split": split, "selected": sel,
                 "control_coverage": {
                     str(seed): len(controls.get((split, seed)) or [])
                     == 1
                     for seed in FRESH_SEEDS},
                 "fresh_verdict": None}
        if sel is None:
            entry["fresh_verdict"] = ("no operating point selected for "
                                      "this split; controls coverage "
                                      "checked only")
        else:
            dim, method = sel["dim"], sel["method"]
            group = [defended[(split, seed, method, dim)]
                     for seed in FRESH_SEEDS
                     if (split, seed, method, dim) in defended]
            pca = [defended[(split, seed, PCA_CONTROL, dim)]
                   for seed in FRESH_SEEDS
                   if (split, seed, PCA_CONTROL, dim) in defended]
            paired = [c for c in group
                      if len(controls.get((split, c["seed"])) or [])
                      == 1]
            complete = len(group) == len(FRESH_SEEDS) \
                and len(pca) == len(FRESH_SEEDS) \
                and len(paired) == len(FRESH_SEEDS)
            if not complete:
                entry["fresh_verdict"] = (
                    "fresh coverage incomplete for the selected "
                    "configuration; see failures")
            else:
                wb_status = None
                if method == COTRAIN_METHOD:
                    cand_wb_cells = {}
                    wb_status = "NOT_EVALUABLE: " + COTRAIN_WB_NA
                else:
                    cand_wb_cells = {seed: wb_by_key.get(
                        (split, seed, method, dim))
                        for seed in FRESH_SEEDS}
                    if any(v is None for v in cand_wb_cells.values()):
                        cand_wb_cells = {}
                pca_wb_cells = {seed: wb_by_key.get(
                    (split, seed, PCA_CONTROL, dim))
                    for seed in FRESH_SEEDS}
                if any(v is None for v in pca_wb_cells.values()):
                    pca_wb_cells = {} if wb_plan_present else {}
                ctrl_by_seed = {seed: controls[(split, seed)][0]
                                for seed in FRESH_SEEDS}
                ratios = [c["utility_after"]["perplexity"]
                          / ctrl_by_seed[c["seed"]]
                          ["utility_after"]["perplexity"]
                          for c in paired]
                n15 = sum(1 for r in ratios if r <= PRIMARY_GATE)
                n20 = sum(1 for r in ratios if r <= SECONDARY_GATE)
                gate = ("primary" if n15 == len(ratios) else
                        "secondary" if n20 == len(ratios) else "fail")
                verdict = {
                    "gate": gate,
                    "fresh_ppl_ratios": ratios,
                    "pass_1_5x_seeds": n15, "pass_2x_seeds": n20,
                    "selection_label": sel["gate"]}
                if gate == "fail":
                    failures.append({
                        "cell": f"split{split}/dim-{dim}/{method}",
                        "reason": "fresh-seed utility gate FAIL at both "
                                  f"predefined thresholds "
                                  f"({PRIMARY_GATE}x / "
                                  f"{SECONDARY_GATE}x)"})
                # PCA-superiority of the strongest-observed leakage,
                # recomputed on the fresh seeds only
                try:
                    cand_leak = {seed: _strongest_observed(
                        defended[(split, seed, method, dim)],
                        cand_wb_cells.get(seed))
                        for seed in FRESH_SEEDS}
                    ctrl_leak = {seed: _strongest_observed(
                        defended[(split, seed, PCA_CONTROL, dim)],
                        pca_wb_cells.get(seed))
                        for seed in FRESH_SEEDS}
                except (KeyError, ValueError) as exc:
                    cand_leak, ctrl_leak = None, None
                    failures.append({
                        "cell": f"split{split}/dim-{dim}/{method}",
                        "reason": f"fresh strongest-observed statistic "
                                  f"not computable (fail closed): {exc}"})
                if cand_leak is not None:
                    diffs = [ctrl_leak[s] - cand_leak[s]
                             for s in FRESH_SEEDS]
                    ci = percentile_bootstrap(diffs)
                    bootstrap_ok = ci["mean"] is not None \
                        and ci["mean"] > 0 and ci["lower"] is not None \
                        and ci["lower"] > 0
                    all_seeds_ok = all(d > 0 for d in diffs)
                    superiority = {
                        "statistic": "strongest observed attack result "
                                     "across measured arms (max per "
                                     "seed), recomputed on the fresh "
                                     "seeds",
                        "whitebox_tier": wb_status or "measured",
                        "candidate_per_seed": cand_leak,
                        "control_per_seed": ctrl_leak,
                        "control_minus_candidate_diffs": diffs,
                        "point_estimate": ci["mean"],
                        "ci_low": ci["lower"], "ci_high": ci["upper"],
                        "ci_status": ci["status"],
                        "bootstrap_condition": {"passed": bootstrap_ok},
                        "all_seeds_positive_condition": {
                            "per_seed_diffs_positive": {
                                str(s): d > 0 for s, d in
                                zip(FRESH_SEEDS, diffs)},
                            "passed": all_seeds_ok},
                        "passed": bootstrap_ok and all_seeds_ok}
                    verdict["pca_superiority"] = superiority
                    if not superiority["passed"]:
                        failures.append({
                            "cell": f"split{split}/dim-{dim}/{method}",
                            "reason": "fresh PCA-superiority regresses: "
                                      "the selected configuration's "
                                      "strongest observed leakage is "
                                      "NOT below the pca-compress "
                                      "control on the fresh seeds "
                                      "(bootstrap and/or "
                                      "all-seeds-positive condition "
                                      "failed)"})
                # compute factor, recomputed on the fresh cells (all
                # measured arms: base tier + white-box attack seconds)
                cand_wb_attack = statistics.mean(
                    c["attack_seconds"] for c in cand_wb_cells.values()
                    ) if cand_wb_cells else 0.0
                ctrl_wb_attack = statistics.mean(
                    c["attack_seconds"] for c in pca_wb_cells.values()
                    ) if pca_wb_cells else 0.0
                cand_compute = statistics.mean(
                    c["wall_seconds"] + c["attack_seconds"]
                    for c in group) + cand_wb_attack
                ctrl_compute = statistics.mean(
                    c["wall_seconds"] + c["attack_seconds"]
                    for c in pca) + ctrl_wb_attack
                ratio = (cand_compute / ctrl_compute
                         if ctrl_compute > 0 else None)
                verdict["compute_comparison"] = {
                    "candidate_train_plus_attack_seconds": cand_compute,
                    "control_train_plus_attack_seconds": ctrl_compute,
                    "ratio": ratio, "max_factor": COMPUTE_FACTOR,
                    "within_factor": ratio is not None
                    and ratio <= COMPUTE_FACTOR}
                if not verdict["compute_comparison"]["within_factor"]:
                    failures.append({
                        "cell": f"split{split}/dim-{dim}/{method}",
                        "reason": f"fresh compute ratio {ratio} exceeds "
                                  f"the predefined {COMPUTE_FACTOR}x "
                                  "train+attack wall-second factor vs "
                                  "the pca-compress control"})
                entry["fresh_verdict"] = verdict
        split_records[str(split)] = entry
    outcome = "failed" if failures else "passed"
    return {"schema": SCHEMA, "outcome": outcome,
            "fresh_seeds": list(FRESH_SEEDS),
            "gates": {"primary_ppl_ratio": PRIMARY_GATE,
                      "secondary_ppl_ratio": SECONDARY_GATE,
                      "predefined": True},
            "fresh_only_rule": ("gate verdicts re-evaluated on fresh "
                                "seeds only; selection-seed results are "
                                "never reused"),
            "coverage_rule": COVERAGE_RULE,
            "retest_only_rule": RETEST_ONLY_RULE,
            "selected_configurations": {
                s: (selection["splits"].get(s) or {}).get("selected")
                for s in (str(sp) for sp in sel_splits)},
            "access_labels": dict(ACCESS_LABELS),
            "v3_status": "UNFROZEN", "v3_rule": V3_RULE,
            "splits": split_records,
            "failures": failures}


def render_markdown(record):
    lines = ["# Fresh-seed validation of the selected operating point",
             "",
             f"Outcome: **{record['outcome']}**. Fresh seeds "
             f"{record['fresh_seeds']} only (selection-seed results "
             "never reused). Predefined gates: "
             f"{record['gates']['primary_ppl_ratio']}x primary / "
             f"{record['gates']['secondary_ppl_ratio']}x secondary. "
             + record["retest_only_rule"] + ".", "",
             "| Split | Selected | Controls | Fresh gate |",
             "|---:|---|:---:|---|"]
    for split, e in sorted(record["splits"].items(),
                           key=lambda kv: int(kv[0])):
        sel = e["selected"]
        cov = all(e["control_coverage"].values())
        verdict = e["fresh_verdict"]
        gate = verdict["gate"] if isinstance(verdict, dict) else "n/a"
        lines.append(f"| {split} | "
                     + (f"{sel['method']} dim={sel['dim']}"
                        if sel else "NONE")
                     + f" | {'OK' if cov else 'MISSING'} | {gate} |")
    lines += ["", "## Failures", ""]
    if record["failures"]:
        lines += ["", "| Cell | Reason |", "|---|---|"]
        for f in record["failures"]:
            lines.append(f"| {f['cell']} | {f['reason']} |")
    else:
        lines.append("None: all fresh cells complete, paired controls "
                     "present, gates re-evaluated on fresh seeds only.")
    lines += ["", "## Access and approximation limits", ""]
    lines += ["- " + v for v in record["access_labels"].values()]
    lines += ["- " + record["v3_rule"], ""]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True,
                    help="fresh-seed validation results dir")
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--markdown", type=Path, default=None)
    args = ap.parse_args(argv)
    selection = json.loads(args.selection.read_text())
    record = validate(args.root, selection)
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False)
                           + "\n")
    if args.markdown is not None:
        args.markdown.write_text(render_markdown(record))
    print(json.dumps({"schema": SCHEMA, "outcome": record["outcome"],
                      "failures": len(record["failures"])}, indent=2))
    return 0 if record["outcome"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
