"""Analyze the bottleneck dimension/method development sweep.

Fail-closed contracts, mirroring analyze_defended_pilot.py:
  * article-hash disjointness of the development corpus against the
    frozen corpus manifest is re-asserted before any table is written
    (strict pilot-corpus identity check for development_sweep tier);
  * the attacker-access spec must be byte-identical across all cells
    (matched access) — drift aborts the analysis;
  * every planned cell is accounted for: a cell that is missing,
    incomplete, or lacks provenance / utility / access-spec / attack
    results is reported FAILED in the failures-and-nulls section, never
    silently dropped — and any failure makes the analysis raise after
    the report is written. INCOMPLETE cells are never dropped.

Table: dim x method x split rows with forward/SIPIT/defense-aware
recovery, delta-PPL vs the paired undefended control with percentile
bootstrap CIs over the 3 seeds (conditional on this fixed article
sample), and the dual utility-gate verdicts (ppl_after <= 1.5x / 2x the
paired control, per-seed counts and an all-seeds verdict). When the
post-hoc white-box-aware addendum tier (<root>/whitebox/, produced by
run_whitebox_sweep.py) is present, its cells are reported as separate,
clearly labeled rows ("white-box-aware (true encoder granted)") with the
same paired bootstrap CIs and carried utility-gate columns — a strictly
stronger access condition, never merged into the matched-access rows,
and fail-closed like the base cells (FAILED, never dropped). When the
adaptive end-to-end addendum tier (<root>/whitebox-e2e/, produced by
run_whitebox_sweep.py --adaptive-e2e) is present, its cells are likewise
reported as separate, clearly labeled rows ("adaptive end-to-end (true
transform)") — a strictly stronger AND adaptive access condition — with
the same fail-closed rules, and exported as whitebox_e2e_seed_rows so
the operating-point selector can include the e2e arm in its per-seed
strongest-observed max.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from attacker.evidence import file_sha256, verify_evidence  # noqa: E402
from analyze_defended_pilot import (  # noqa: E402
    check_matched_access, percentile_bootstrap)
from run_bottleneck_sweep import CELL_SCHEMA  # noqa: E402
from run_defended_pilot import (  # noqa: E402
    assert_development_corpus_disjoint)

SCHEMA = "llm_attacker.bottleneck_sweep_analysis.v1"
WHITEBOX_PLAN_SCHEMA = "llm_attacker.whitebox_sweep.v1"
WHITEBOX_CELL_SCHEMA = "llm_attacker.whitebox_sweep_cell.v1"
WHITEBOX_REQUIRED_KEYS = ("wall_seconds", "attack_seconds",
                          "utility_after", "utility_gate", "access",
                          "defense_checkpoint", "attacks", "dim",
                          "method", "base_cell")
WHITEBOX_E2E_PLAN_SCHEMA = "llm_attacker.whitebox_sweep_e2e.v1"
WHITEBOX_E2E_CELL_SCHEMA = "llm_attacker.whitebox_sweep_e2e_cell.v1"
WHITEBOX_E2E_ACCESS_SCHEMA = "llm_attacker.whitebox_sweep_e2e_access.v1"
WHITEBOX_E2E_REQUIRED_KEYS = WHITEBOX_REQUIRED_KEYS
REQUIRED_CELL_KEYS = ("wall_seconds", "attack_seconds",
                      "cuda_peak_allocated_bytes", "utility_before",
                      "utility_after", "access", "provenance", "attacks",
                      "dim", "method")


def _valid_accuracy(value):
    """An attack accuracy is a finite float in [0, 1] (bool excluded);
    NaN/inf/out-of-range values are INVALID, never silently averaged."""
    return isinstance(value, (int, float)) \
        and not isinstance(value, bool) \
        and math.isfinite(value) and 0.0 <= value <= 1.0


def _verify_cell_evidence(cell, path):
    """Verify the cell's declared evidence journal (the runner seals
    every run artifact with attacker.evidence.EvidenceLogger). Returns
    None on success, else a failure reason."""
    evidence = cell.get("evidence")
    if not isinstance(evidence, str) or not evidence:
        return "evidence journal pointer missing"
    journal = path.parent / evidence
    if not journal.is_file():
        return f"evidence journal missing: {evidence}"
    ok, problems = verify_evidence(str(journal))
    if not ok:
        return "evidence journal failed verification: " + "; ".join(
            problems)
    return None


def load_cells(root, plan):
    cells, failures = [], []
    for entry in plan["cells"]:
        path = root / entry["path"] / "cell.json"
        label = entry["path"]
        if not path.is_file():
            failures.append({"cell": label, "reason": "cell.json missing"})
            continue
        try:
            cell = json.loads(path.read_text())
        except ValueError as exc:
            failures.append({"cell": label,
                             "reason": f"cell.json invalid: {exc}"})
            continue
        if cell.get("schema") != CELL_SCHEMA:
            failures.append({"cell": label, "reason": "unexpected cell "
                                                      "schema"})
            continue
        if cell.get("status") != "COMPLETE":
            failures.append({"cell": label,
                             "reason": f"status {cell.get('status')!r}"})
            continue
        missing = [k for k in REQUIRED_CELL_KEYS if k not in cell]
        if missing:
            failures.append({"cell": label,
                             "reason": f"missing keys: {missing}"})
            continue
        prov = cell.get("provenance") or {}
        prov_path = path.parent / str(prov.get("path", ""))
        if not prov.get("path") or not prov_path.is_file() or \
                file_sha256(prov_path) != prov.get("sha256"):
            failures.append({"cell": label,
                             "reason": "provenance sidecar missing or "
                                       "hash mismatch"})
            continue
        evidence_problem = _verify_cell_evidence(cell, path)
        if evidence_problem is not None:
            failures.append({"cell": label, "reason": evidence_problem})
            continue
        util_ok = True
        for stage in ("utility_before", "utility_after"):
            util = cell.get(stage) or {}
            ce, ppl = util.get("cross_entropy"), util.get("perplexity")
            if not isinstance(ce, (int, float)) or isinstance(ce, bool) \
                    or not isinstance(ppl, (int, float)) \
                    or isinstance(ppl, bool) or not math.isfinite(ce) \
                    or not math.isfinite(ppl) or ppl <= 0:
                failures.append({"cell": label,
                                 "reason": f"{stage} incomplete"})
                util_ok = False
                break
        if not util_ok:
            continue
        attacks = cell.get("attacks") or {}
        if not all(isinstance(attacks.get(a), dict)
                   and _valid_accuracy(attacks[a].get("accuracy"))
                   for a in ("forward", "sipit")):
            failures.append({"cell": label,
                             "reason": "common forward/SIPIT baselines "
                                       "missing or invalid (finite "
                                       "float in [0, 1] required)"})
            continue
        if cell["method"] != "none":
            da = attacks.get("defense_aware") or {}
            if da.get("defense_aware") is not True or not all(
                    isinstance(da.get(a), dict)
                    and _valid_accuracy(da[a].get("accuracy"))
                    for a in ("forward", "sipit")):
                failures.append({"cell": label,
                                 "reason": "defense-aware arm missing or "
                                           "invalid"})
                continue
            gate = cell.get("utility_gate") or {}
            ratio = gate.get("ratio")
            if not isinstance(ratio, (int, float)) \
                    or isinstance(ratio, bool) \
                    or not math.isfinite(ratio) or ratio <= 0 \
                    or not isinstance(gate.get("pass_1_5x"), bool) \
                    or not isinstance(gate.get("pass_2x"), bool):
                failures.append({"cell": label,
                                 "reason": "utility gate missing or "
                                           "malformed"})
                continue
        cells.append(cell)
    return cells, failures


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def load_whitebox_cells(root):
    """The post-hoc white-box-aware addendum tier (<root>/whitebox/).

    Returns (cells, failures, not_evaluable). Absent tier -> ([], [],
    []). Present tier is fail-closed: every planned whitebox cell must
    exist, be COMPLETE, carry a hash-verified pointer to the base cell's
    true defense checkpoint, both whitebox arms (finite accuracies in
    [0, 1]), a sealed evidence journal, and the carried utility/gate;
    anything else is reported FAILED, never dropped. Cells recorded
    NOT_EVALUABLE by the runner (co-trained encoders: the saved
    checkpoint is the FINAL encoder, mismatched with the capture-time
    one) are a planned exclusion: carried in not_evaluable with the
    reason, never failed, never evaluated. The whitebox access spec is
    a STRICTLY STRONGER condition and is NOT matched against the base
    matrix's access spec.
    """
    root = Path(root)
    plan_path = root / "whitebox" / "plan.json"
    if not plan_path.is_file():
        return [], [], []
    plan = json.loads(plan_path.read_text())
    if plan.get("schema") != WHITEBOX_PLAN_SCHEMA:
        raise ValueError("unexpected whitebox plan schema")
    cells, failures, not_evaluable = [], [], []
    for entry in plan["cells"]:
        label = entry["path"]
        path = root / label / "cell.json"
        if not path.is_file():
            failures.append({"cell": label, "reason": "cell.json missing"})
            continue
        try:
            cell = json.loads(path.read_text())
        except ValueError as exc:
            failures.append({"cell": label,
                             "reason": f"cell.json invalid: {exc}"})
            continue
        if cell.get("schema") != WHITEBOX_CELL_SCHEMA:
            failures.append({"cell": label, "reason": "unexpected cell "
                                                      "schema"})
            continue
        if cell.get("status") == "NOT_EVALUABLE":
            not_evaluable.append({
                "cell": label, "split": cell.get("split"),
                "seed": cell.get("seed"), "method": cell.get("method"),
                "dim": cell.get("dim"),
                "reason": cell.get("not_evaluable_reason")
                or cell.get("failure_reason")
                or "white-box tier not evaluable"})
            continue
        if cell.get("status") != "COMPLETE":
            reason = cell.get("failure_reason") or \
                f"status {cell.get('status')!r}"
            failures.append({"cell": label, "reason": reason})
            continue
        missing = [k for k in WHITEBOX_REQUIRED_KEYS if k not in cell]
        if missing:
            failures.append({"cell": label,
                             "reason": f"missing keys: {missing}"})
            continue
        ckpt = cell.get("defense_checkpoint") or {}
        ckpt_path = root / str(ckpt.get("path", ""))
        if not ckpt.get("path") or not ckpt_path.is_file() or \
                file_sha256(ckpt_path) != ckpt.get("sha256"):
            failures.append({"cell": label,
                             "reason": "true defense checkpoint missing "
                                       "or hash mismatch"})
            continue
        evidence_problem = _verify_cell_evidence(cell, path)
        if evidence_problem is not None:
            failures.append({"cell": label, "reason": evidence_problem})
            continue
        access = cell.get("access") or {}
        if not isinstance((access.get("whitebox_extension") or {})
                          .get("granted"), str):
            failures.append({"cell": label,
                             "reason": "whitebox access extension "
                                       "missing"})
            continue
        attacks = cell.get("attacks") or {}
        inv, opt = attacks.get("whitebox_inverter") or {}, \
            attacks.get("whitebox_optimization") or {}
        ok = _valid_accuracy((inv.get("forward") or {}).get("accuracy")) \
            and _valid_accuracy((inv.get("sipit") or {})
                                .get("accuracy")) \
            and _valid_accuracy((opt.get("forward") or {})
                                .get("accuracy")) \
            and _valid_accuracy((opt.get("snap") or {}).get("accuracy"))
        if not ok:
            failures.append({"cell": label,
                             "reason": "whitebox arms incomplete or "
                                       "invalid (finite float in "
                                       "[0, 1] required)"})
            continue
        util = cell.get("utility_after") or {}
        ppl = util.get("perplexity")
        if not isinstance(ppl, (int, float)) or isinstance(ppl, bool) \
                or not math.isfinite(ppl) or ppl <= 0:
            failures.append({"cell": label,
                             "reason": "carried utility missing"})
            continue
        gate = cell.get("utility_gate") or {}
        ratio = gate.get("ratio")
        if not isinstance(ratio, (int, float)) or isinstance(ratio, bool) \
                or not math.isfinite(ratio) or ratio <= 0 \
                or not isinstance(gate.get("pass_1_5x"), bool) \
                or not isinstance(gate.get("pass_2x"), bool):
            failures.append({"cell": label,
                             "reason": "carried utility gate missing or "
                                       "malformed"})
            continue
        cells.append(cell)
    return cells, failures, not_evaluable


def whitebox_summary_rows(wb_cells, baseline, replicates, seed):
    """dim x method x split rows for the white-box-aware tier; utility is
    the base cell's carried value (never recomputed)."""
    import itertools
    rows = []
    key = lambda c: (c["split"], c["dim"], c["method"])
    for (split, dim, method), group in itertools.groupby(
            sorted(wb_cells, key=key), key=key):
        group = sorted(group, key=lambda c: c["seed"])
        row = {"split": split, "dim": dim, "method": method,
               "tier": "white-box-aware (true encoder granted)",
               "seeds": len(group),
               "wb_inverter_forward_mean": statistics.mean(
                   c["attacks"]["whitebox_inverter"]["forward"]
                   ["accuracy"] for c in group),
               "wb_inverter_sipit_mean": statistics.mean(
                   c["attacks"]["whitebox_inverter"]["sipit"]["accuracy"]
                   for c in group),
               "wb_opt_forward_mean": statistics.mean(
                   c["attacks"]["whitebox_optimization"]["forward"]
                   ["accuracy"] for c in group),
               "wb_opt_snap_mean": statistics.mean(
                   c["attacks"]["whitebox_optimization"]["snap"]
                   ["accuracy"] for c in group),
               "ppl_after_mean": statistics.mean(
                   c["utility_after"]["perplexity"] for c in group),
               "attack_seconds_mean": statistics.mean(
                   c["attack_seconds"] for c in group)}
        ppl_diffs = [c["utility_after"]["perplexity"]
                     - baseline[(split, c["seed"])]["utility_after"]
                     ["perplexity"]
                     for c in group if (split, c["seed"]) in baseline]
        ci = percentile_bootstrap(ppl_diffs, replicates, seed)
        row["ppl_delta_vs_none_mean"] = ci["mean"]
        row["ppl_delta_vs_none_ci_low"] = ci["lower"]
        row["ppl_delta_vs_none_ci_high"] = ci["upper"]
        row["ppl_delta_vs_none_status"] = ci["status"]
        rec_diffs = [baseline[(split, c["seed"])]["attacks"]["forward"]
                     ["accuracy"]
                     - c["attacks"]["whitebox_inverter"]["forward"]
                     ["accuracy"]
                     for c in group if (split, c["seed"]) in baseline]
        ci = percentile_bootstrap(rec_diffs, replicates, seed)
        row["none_minus_wb_inverter_forward_mean"] = ci["mean"]
        row["none_minus_wb_inverter_forward_ci_low"] = ci["lower"]
        row["none_minus_wb_inverter_forward_ci_high"] = ci["upper"]
        row.update(gate_verdict(group))
        rows.append(row)
    return rows


def load_whitebox_e2e_cells(root):
    """The adaptive end-to-end addendum tier (<root>/whitebox-e2e/).

    Returns (cells, failures, not_evaluable). Absent tier -> ([], [],
    []). Present tier is fail-closed: every planned e2e cell must exist,
    be COMPLETE, declare the base cell's true defense checkpoint
    (path + sha256; the hash is verified when the tensor file is
    present locally — the metadata-only sync excludes the .pt tensors),
    carry the adaptive_e2e arm's snap accuracy (finite float in
    [0, 1]), a sealed evidence journal, and the carried utility/gate;
    anything else is reported FAILED, never dropped. Cells recorded
    NOT_EVALUABLE by the runner (tb-cotrain: training-phase captures
    predate the saved final co-trained encoder) are a planned
    exclusion: carried in not_evaluable with the reason, never failed,
    never evaluated. The e2e access spec is a STRICTLY STRONGER and
    ADAPTIVE condition and is NOT matched against the base matrix's
    access spec.
    """
    root = Path(root)
    plan_path = root / "whitebox-e2e" / "plan.json"
    if not plan_path.is_file():
        return [], [], []
    plan = json.loads(plan_path.read_text())
    if plan.get("schema") != WHITEBOX_E2E_PLAN_SCHEMA:
        raise ValueError("unexpected whitebox-e2e plan schema")
    cells, failures, not_evaluable = [], [], []
    for entry in plan["cells"]:
        label = entry["path"]
        path = root / label / "cell.json"
        if not path.is_file():
            failures.append({"cell": label, "reason": "cell.json missing"})
            continue
        try:
            cell = json.loads(path.read_text())
        except ValueError as exc:
            failures.append({"cell": label,
                             "reason": f"cell.json invalid: {exc}"})
            continue
        if cell.get("schema") != WHITEBOX_E2E_CELL_SCHEMA:
            failures.append({"cell": label, "reason": "unexpected cell "
                                                      "schema"})
            continue
        if cell.get("status") == "NOT_EVALUABLE":
            not_evaluable.append({
                "cell": label, "split": cell.get("split"),
                "seed": cell.get("seed"), "method": cell.get("method"),
                "dim": cell.get("dim"),
                "reason": cell.get("not_evaluable_reason")
                or cell.get("failure_reason")
                or "whitebox-e2e tier not evaluable"})
            continue
        if cell.get("status") != "COMPLETE":
            reason = cell.get("failure_reason") or \
                f"status {cell.get('status')!r}"
            failures.append({"cell": label, "reason": reason})
            continue
        missing = [k for k in WHITEBOX_E2E_REQUIRED_KEYS if k not in cell]
        if missing:
            failures.append({"cell": label,
                             "reason": f"missing keys: {missing}"})
            continue
        ckpt = cell.get("defense_checkpoint") or {}
        ckpt_path = root / str(ckpt.get("path", ""))
        if not ckpt.get("path") or not isinstance(ckpt.get("sha256"),
                                                  str):
            failures.append({"cell": label,
                             "reason": "true defense checkpoint pointer "
                                       "missing"})
            continue
        if ckpt_path.is_file() and \
                file_sha256(ckpt_path) != ckpt.get("sha256"):
            failures.append({"cell": label,
                             "reason": "true defense checkpoint hash "
                                       "mismatch"})
            continue
        evidence_problem = _verify_cell_evidence(cell, path)
        if evidence_problem is not None:
            failures.append({"cell": label, "reason": evidence_problem})
            continue
        access = cell.get("access") or {}
        if access.get("schema") != WHITEBOX_E2E_ACCESS_SCHEMA \
                or not isinstance((access.get("whitebox_extension") or {})
                                  .get("granted"), str):
            failures.append({"cell": label,
                             "reason": "whitebox-e2e access spec "
                                       "missing or unexpected"})
            continue
        arm = (cell.get("attacks") or {}).get("adaptive_e2e") or {}
        if arm.get("true_encoder_granted") is not True \
                or not _valid_accuracy((arm.get("snap") or {})
                                       .get("accuracy")):
            failures.append({"cell": label,
                             "reason": "adaptive_e2e arm incomplete or "
                                       "invalid (finite float in "
                                       "[0, 1] required)"})
            continue
        util = cell.get("utility_after") or {}
        ppl = util.get("perplexity")
        if not isinstance(ppl, (int, float)) or isinstance(ppl, bool) \
                or not math.isfinite(ppl) or ppl <= 0:
            failures.append({"cell": label,
                             "reason": "carried utility missing"})
            continue
        gate = cell.get("utility_gate") or {}
        ratio = gate.get("ratio")
        if not isinstance(ratio, (int, float)) or isinstance(ratio, bool) \
                or not math.isfinite(ratio) or ratio <= 0 \
                or not isinstance(gate.get("pass_1_5x"), bool) \
                or not isinstance(gate.get("pass_2x"), bool):
            failures.append({"cell": label,
                             "reason": "carried utility gate missing or "
                                       "malformed"})
            continue
        cells.append(cell)
    return cells, failures, not_evaluable


def whitebox_e2e_summary_rows(e2e_cells, baseline, replicates, seed):
    """dim x method x split rows for the adaptive end-to-end tier;
    utility is the base cell's carried value (never recomputed)."""
    import itertools
    rows = []
    key = lambda c: (c["split"], c["dim"], c["method"])
    for (split, dim, method), group in itertools.groupby(
            sorted(e2e_cells, key=key), key=key):
        group = sorted(group, key=lambda c: c["seed"])
        row = {"split": split, "dim": dim, "method": method,
               "tier": "adaptive end-to-end (true transform)",
               "seeds": len(group),
               "e2e_snap_mean": statistics.mean(
                   c["attacks"]["adaptive_e2e"]["snap"]["accuracy"]
                   for c in group),
               "ppl_after_mean": statistics.mean(
                   c["utility_after"]["perplexity"] for c in group),
               "attack_seconds_mean": statistics.mean(
                   c["attack_seconds"] for c in group)}
        ppl_diffs = [c["utility_after"]["perplexity"]
                     - baseline[(split, c["seed"])]["utility_after"]
                     ["perplexity"]
                     for c in group if (split, c["seed"]) in baseline]
        ci = percentile_bootstrap(ppl_diffs, replicates, seed)
        row["ppl_delta_vs_none_mean"] = ci["mean"]
        row["ppl_delta_vs_none_ci_low"] = ci["lower"]
        row["ppl_delta_vs_none_ci_high"] = ci["upper"]
        row["ppl_delta_vs_none_status"] = ci["status"]
        rec_diffs = [baseline[(split, c["seed"])]["attacks"]["forward"]
                     ["accuracy"]
                     - c["attacks"]["adaptive_e2e"]["snap"]["accuracy"]
                     for c in group if (split, c["seed"]) in baseline]
        ci = percentile_bootstrap(rec_diffs, replicates, seed)
        row["none_minus_e2e_snap_mean"] = ci["mean"]
        row["none_minus_e2e_snap_ci_low"] = ci["lower"]
        row["none_minus_e2e_snap_ci_high"] = ci["upper"]
        row.update(gate_verdict(group))
        rows.append(row)
    return rows


def gate_verdict(group):
    """Per (dim, method, split) gate summary over seeds: per-seed pass
    counts and an all-seeds verdict for both thresholds."""
    gates = [c["utility_gate"] for c in group]
    n15 = sum(1 for g in gates if g["pass_1_5x"])
    n20 = sum(1 for g in gates if g["pass_2x"])
    return {"seeds": len(group),
            "pass_1_5x_seeds": n15,
            "pass_2x_seeds": n20,
            "gate_1_5x": "PASS" if n15 == len(group) else "FAIL",
            "gate_2x": "PASS" if n20 == len(group) else "FAIL",
            "ppl_ratio_mean": statistics.mean(g["ratio"] for g in gates)}


def analyze(root, manifest_path, replicates=10000, seed=2026):
    root = Path(root)
    plan = json.loads((root / "plan.json").read_text())
    if plan.get("schema") != "llm_attacker.bottleneck_sweep.v1":
        raise ValueError("unexpected plan schema")
    corpus = plan["corpus"]["path"]
    corpus_path = Path(corpus)
    if not corpus_path.is_absolute():
        candidate = root / corpus
        corpus_path = candidate if candidate.is_file() else Path(corpus)
    tier = plan.get("tier")
    disjointness = assert_development_corpus_disjoint(
        corpus_path, manifest_path,
        require_excluded_pilot=(tier == "development_sweep"))
    cells, failures = load_cells(root, plan)
    if cells:
        check_matched_access(cells)
    wb_cells, wb_failures, wb_not_evaluable = load_whitebox_cells(root)
    e2e_cells, e2e_failures, e2e_not_evaluable = \
        load_whitebox_e2e_cells(root)
    failures = failures + wb_failures + e2e_failures
    baseline = {(c["split"], c["seed"]): c for c in cells
                if c["method"] == "none"}
    wb_summary = whitebox_summary_rows(wb_cells, baseline, replicates,
                                       seed)
    e2e_summary = whitebox_e2e_summary_rows(e2e_cells, baseline,
                                            replicates, seed)
    e2e_seed_rows = [{
        "split": c["split"], "dim": c["dim"], "method": c["method"],
        "seed": c["seed"],
        "adaptive_e2e_snap":
            c["attacks"]["adaptive_e2e"]["snap"]["accuracy"],
        "ppl_ratio": c["utility_gate"]["ratio"],
        "wall_seconds": c["wall_seconds"],
        "attack_seconds": c["attack_seconds"]} for c in e2e_cells]
    wb_seed_rows = [{
        "split": c["split"], "dim": c["dim"], "method": c["method"],
        "seed": c["seed"],
        "wb_inverter_forward":
            c["attacks"]["whitebox_inverter"]["forward"]["accuracy"],
        "wb_inverter_sipit":
            c["attacks"]["whitebox_inverter"]["sipit"]["accuracy"],
        "wb_opt_forward":
            c["attacks"]["whitebox_optimization"]["forward"]["accuracy"],
        "wb_opt_snap":
            c["attacks"]["whitebox_optimization"]["snap"]["accuracy"],
        "ppl_ratio": c["utility_gate"]["ratio"],
        "wall_seconds": c["wall_seconds"],
        "attack_seconds": c["attack_seconds"]} for c in wb_cells]
    summary = []
    seed_rows = []
    for split in plan["splits"]:
        for dim in plan["dims"]:
            for method in plan["methods"]:
                group = sorted([c for c in cells
                                if c["split"] == split
                                and c["dim"] == dim
                                and c["method"] == method],
                               key=lambda c: c["seed"])
                if not group:
                    continue
                row = {"split": split, "dim": dim, "method": method,
                       "seeds": len(group),
                       "forward_accuracy_mean": statistics.mean(
                           c["attacks"]["forward"]["accuracy"]
                           for c in group),
                       "sipit_accuracy_mean": statistics.mean(
                           c["attacks"]["sipit"]["accuracy"]
                           for c in group),
                       "defense_aware_forward_mean": statistics.mean(
                           c["attacks"]["defense_aware"]["forward"]
                           ["accuracy"] for c in group),
                       "defense_aware_sipit_mean": statistics.mean(
                           c["attacks"]["defense_aware"]["sipit"]
                           ["accuracy"] for c in group),
                       "ppl_after_mean": statistics.mean(
                           c["utility_after"]["perplexity"]
                           for c in group),
                       "attack_seconds_mean": statistics.mean(
                           c["attack_seconds"] for c in group),
                       "wall_seconds_mean": statistics.mean(
                           c["wall_seconds"] for c in group)}
                ppl_diffs = [c["utility_after"]["perplexity"]
                             - baseline[(split, c["seed"])]
                             ["utility_after"]["perplexity"]
                             for c in group
                             if (split, c["seed"]) in baseline]
                ci = percentile_bootstrap(ppl_diffs, replicates, seed)
                row["ppl_delta_vs_none_mean"] = ci["mean"]
                row["ppl_delta_vs_none_ci_low"] = ci["lower"]
                row["ppl_delta_vs_none_ci_high"] = ci["upper"]
                row["ppl_delta_vs_none_status"] = ci["status"]
                rec_diffs = [baseline[(split, c["seed"])]["attacks"]
                             ["forward"]["accuracy"]
                             - c["attacks"]["defense_aware"]["forward"]
                             ["accuracy"]
                             for c in group
                             if (split, c["seed"]) in baseline]
                ci = percentile_bootstrap(rec_diffs, replicates, seed)
                row["none_minus_da_forward_mean"] = ci["mean"]
                row["none_minus_da_forward_ci_low"] = ci["lower"]
                row["none_minus_da_forward_ci_high"] = ci["upper"]
                row.update(gate_verdict(group))
                summary.append(row)
                for c in group:
                    seed_rows.append({
                        "split": split, "dim": dim, "method": method,
                        "seed": c["seed"],
                        "forward_accuracy":
                            c["attacks"]["forward"]["accuracy"],
                        "sipit_accuracy":
                            c["attacks"]["sipit"]["accuracy"],
                        "da_forward_accuracy":
                            c["attacks"]["defense_aware"]["forward"]
                            ["accuracy"],
                        "da_sipit_accuracy":
                            c["attacks"]["defense_aware"]["sipit"]
                            ["accuracy"],
                        "ppl_after": c["utility_after"]["perplexity"],
                        "ppl_ratio": c["utility_gate"]["ratio"],
                        "pass_1_5x": c["utility_gate"]["pass_1_5x"],
                        "pass_2x": c["utility_gate"]["pass_2x"],
                        "wall_seconds": c["wall_seconds"],
                        "attack_seconds": c["attack_seconds"],
                        "cuda_peak_allocated_bytes":
                            c["cuda_peak_allocated_bytes"]})
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    if summary:
        write_csv(out / "summary.csv", summary)
    if seed_rows:
        write_csv(out / "seed_metrics.csv", seed_rows)
    write_csv(out / "failures.csv",
              failures or [{"cell": "", "reason": "none"}])
    lines = ["# Bottleneck dimension/method development sweep", "",
             f"Tier: {tier}. Matrix: methods {' x '.join(plan['methods'])} "
             f"x dims {plan['dims']} x splits {plan['splits']} x seeds "
             f"{plan['seeds']} on "
             + ("the 96 development wikitext-2 articles."
                if tier == "development_sweep" else
                "a SYNTHETIC smoke corpus (engineering tier; not security "
                "evidence)."), "",
             f"Frozen-corpus separation: development corpus sha256 "
             f"`{disjointness['corpus_sha256']}`; article ids disjoint "
             "from every frozen-corpus partition (asserted fail-closed "
             f"against `{disjointness['frozen_manifest']}`); identity "
             "checked against excluded_pilot_sha256: "
             f"{disjointness['identity_checked']}.", "",
             "Attacker access is matched across methods/dims: public "
             "calibration partition, pretrained public head only, fixed "
             "query budgets (candidates 32 / beam 4 / SIPIT 128 by "
             "default); the defense-aware arm additionally knows the "
             "defense CLASS and k and fits a PCA surrogate inverse on "
             "public calibration (BPDA-style). Per-cell access specs "
             "were verified byte-identical.", "",
             "## Recovery and utility (dim x method x split)", "",
             "Accuracy is position-aligned token recovery on captured "
             "training-phase wires. dPPL is defended-minus-undefended "
             "held-out perplexity, paired over seeds with percentile "
             "bootstrap CIs (conditional on this fixed article sample). "
             "Gates: defended ppl_after <= 1.5x / 2x the paired control "
             "(PASS = all seeds).", "",
             "| Split | Dim | Method | Fwd % | SIPIT % | DA fwd % | DA "
             "SIPIT % | PPL after | dPPL [CI] | Gate 1.5x | Gate 2x |",
             "|---|---:|---|---:|---:|---:|---:|---:|---:|:---:|:---:|"]
    for r in summary:
        lo, hi = r["ppl_delta_vs_none_ci_low"], \
            r["ppl_delta_vs_none_ci_high"]
        delta = (f"{r['ppl_delta_vs_none_mean']:.3f} "
                 f"[{lo:.3f}, {hi:.3f}]" if lo is not None else
                 (f"{r['ppl_delta_vs_none_mean']:.3f} [n/a: "
                  f"{r['ppl_delta_vs_none_status']}]"
                  if r["ppl_delta_vs_none_mean"] is not None else "n/a"))
        lines.append(
            f"| {r['split']} | {r['dim']} | {r['method']} | "
            f"{100 * r['forward_accuracy_mean']:.2f} | "
            f"{100 * r['sipit_accuracy_mean']:.2f} | "
            f"{100 * r['defense_aware_forward_mean']:.2f} | "
            f"{100 * r['defense_aware_sipit_mean']:.2f} | "
            f"{r['ppl_after_mean']:.2f} | {delta} | "
            f"{r['gate_1_5x']} ({r['pass_1_5x_seeds']}/{r['seeds']}) | "
            f"{r['gate_2x']} ({r['pass_2x_seeds']}/{r['seeds']}) |")
    if wb_summary:
        write_csv(out / "whitebox_summary.csv", wb_summary)
        lines += ["", "## White-box-aware tier (true encoder granted)", "",
                  "Post-hoc addendum cells under `whitebox/`: the attacker "
                  "is granted each defended cell's ACTUAL trained defense "
                  "checkpoint (encoder weights / PCA basis + provenance) — "
                  "a strictly stronger access condition than the matched "
                  "matrix; these rows are NOT comparable as matched-access "
                  "rows. Utility and gates are carried verbatim from the "
                  "base cell (the attack cannot change training-time "
                  "utility). A strictly stronger access measurement: "
                  "white-box recovery does not constrain black-box "
                  "recovery of the same wire, nor the reverse.", "",
                  "| Split | Dim | Method | WB inv fwd % | WB inv SIPIT % "
                  "| WB opt fwd % | WB opt snap % | PPL after | dPPL [CI] "
                  "| Gate 1.5x | Gate 2x |",
                  "|---|---:|---|---:|---:|---:|---:|---:|---:|:---:|"
                  ":---:|"]
        for r in wb_summary:
            lo, hi = r["ppl_delta_vs_none_ci_low"], \
                r["ppl_delta_vs_none_ci_high"]
            delta = (f"{r['ppl_delta_vs_none_mean']:.3f} "
                     f"[{lo:.3f}, {hi:.3f}]" if lo is not None else "n/a")
            lines.append(
                f"| {r['split']} | {r['dim']} | {r['method']} "
                f"(white-box-aware) | "
                f"{100 * r['wb_inverter_forward_mean']:.2f} | "
                f"{100 * r['wb_inverter_sipit_mean']:.2f} | "
                f"{100 * r['wb_opt_forward_mean']:.2f} | "
                f"{100 * r['wb_opt_snap_mean']:.2f} | "
                f"{r['ppl_after_mean']:.2f} | {delta} | "
                f"{r['gate_1_5x']} ({r['pass_1_5x_seeds']}/{r['seeds']}) | "
                f"{r['gate_2x']} ({r['pass_2x_seeds']}/{r['seeds']}) |")
    if e2e_summary:
        write_csv(out / "whitebox_e2e_summary.csv", e2e_summary)
        lines += ["", "## Adaptive end-to-end tier (true transform)", "",
                  "Post-hoc addendum cells under `whitebox-e2e/`: the "
                  "attacker is granted each defended cell's ACTUAL "
                  "trained defense checkpoint and optimizes each wire's "
                  "token sequence straight through the TRUE transform "
                  "end-to-end (single vocabulary snap readout; no "
                  "inverter/ridge/beam/SIPIT stages) — a strictly "
                  "stronger AND adaptive access condition; these rows "
                  "are NOT comparable as matched-access rows and do not "
                  "bound black-box recovery of the same wire. Utility "
                  "and gates are carried verbatim from the base cell "
                  "(the attack cannot change training-time utility).", "",
                  "| Split | Dim | Method | E2E snap % | PPL after | "
                  "dPPL [CI] | Gate 1.5x | Gate 2x |",
                  "|---|---:|---|---:|---:|---:|:---:|:---:|"]
        for r in e2e_summary:
            lo, hi = r["ppl_delta_vs_none_ci_low"], \
                r["ppl_delta_vs_none_ci_high"]
            delta = (f"{r['ppl_delta_vs_none_mean']:.3f} "
                     f"[{lo:.3f}, {hi:.3f}]" if lo is not None else "n/a")
            lines.append(
                f"| {r['split']} | {r['dim']} | {r['method']} "
                f"(adaptive-e2e) | "
                f"{100 * r['e2e_snap_mean']:.2f} | "
                f"{r['ppl_after_mean']:.2f} | {delta} | "
                f"{r['gate_1_5x']} ({r['pass_1_5x_seeds']}/{r['seeds']}) | "
                f"{r['gate_2x']} ({r['pass_2x_seeds']}/{r['seeds']}) |")
    lines += ["", "## Failures and nulls", ""]
    if failures:
        lines.append("Every planned cell below is counted FAILED, not "
                     "dropped; the analysis exits nonzero.")
        lines += ["", "| Cell | Reason |", "|---|---|"]
        for f in failures:
            lines.append(f"| {f['cell']} | {f['reason']} |")
    else:
        lines.append(f"All {plan['expected_cells']} planned cells "
                     "completed with provenance, utility, matched access "
                     "spec, both common baselines and the defense-aware "
                     "arm. None dropped.")
    if wb_not_evaluable:
        lines += ["", "## White-box cells NOT_EVALUABLE (planned "
                  "exclusion)", ""]
        lines.append("Recorded by the runner, never evaluated and never "
                     "counted as failures:")
        lines += ["", "| Cell | Reason |", "|---|---|"]
        for e in wb_not_evaluable:
            lines.append(f"| {e['cell']} | {e['reason']} |")
    if e2e_not_evaluable:
        lines += ["", "## Adaptive end-to-end cells NOT_EVALUABLE "
                  "(planned exclusion)", ""]
        lines.append("Recorded by the runner, never evaluated and never "
                     "counted as failures:")
        lines += ["", "| Cell | Reason |", "|---|---|"]
        for e in e2e_not_evaluable:
            lines.append(f"| {e['cell']} | {e['reason']} |")
    lines += ["", "## Scope and limitations", ""]
    lines += ["- " + x for x in plan.get("limits", [])]
    lines += ["", "Regenerate: `python experiments/"
              "analyze_bottleneck_sweep.py --root <sweep-dir>`", ""]
    (out / "report.md").write_text("\n".join(lines))
    result = {"schema": SCHEMA,
              "status": "COMPLETE" if not failures else "FAILED_CELLS",
              "cells_ok": len(cells), "cells_failed": len(failures),
              "expected_cells": plan["expected_cells"], "tier": tier,
              "disjointness": disjointness,
              "bootstrap": {"replicates": replicates, "seed": seed,
                            "method": "percentile, paired over seeds"},
              "summary": summary, "whitebox_summary": wb_summary,
              "whitebox_cells_ok": len(wb_cells),
              "whitebox_not_evaluable": wb_not_evaluable,
              "whitebox_e2e_summary": e2e_summary,
              "whitebox_e2e_cells_ok": len(e2e_cells),
              "whitebox_e2e_not_evaluable": e2e_not_evaluable,
              "seed_rows": seed_rows,
              "whitebox_seed_rows": wb_seed_rows,
              "whitebox_e2e_seed_rows": e2e_seed_rows,
              "failures": failures,
              "analysis_script_sha256": file_sha256(__file__),
              "files": {p.name: file_sha256(p) for p in out.iterdir()
                        if p.suffix in (".csv", ".md")}}
    (out / "analysis.json").write_text(json.dumps(result, indent=2,
                                                  allow_nan=False))
    print(json.dumps({"status": result["status"],
                      "cells_ok": len(cells),
                      "cells_failed": len(failures),
                      "report": str(out / "report.md")}, indent=2))
    if failures:
        raise ValueError(f"{len(failures)} planned cells FAILED; see "
                         f"{out / 'report.md'} failures-and-nulls "
                         "section")
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--manifest", type=Path,
                    default=Path("experiments/"
                                 "validation_corpus_manifest.json"))
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args(argv)
    analyze(args.root, args.manifest, args.bootstrap, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
