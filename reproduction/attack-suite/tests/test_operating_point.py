"""Mechanical operating-point selection + fresh-seed validation tests.

Torch-free fixtures: a small synthetic analysis.json (summary rows +
per-seed white-box rows) and a small synthetic fresh-seed validation
results dir (plan + cell.jsons with hash-verified provenance sidecars,
sealed evidence journals, paired none/pca-compress controls and a
white-box addendum tier, matching the sweep analyzer's fail-closed
loaders).
"""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from attacker.evidence import EvidenceLogger

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "experiments" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


selector = load_script("select_operating_point")
fresh = load_script("check_fresh_seed_validation")

SEEDS = (11, 22, 33)
FRESH_SEEDS = (44, 55, 77)
SPLITS = (3, 6)


# ---------------------------------------------------------------------------
# analysis.json fixtures


def summary_row(split, dim, method, g15="PASS", g2="PASS", wall=100.0,
                attack=10.0, da_fwd=0.05):
    return {"split": split, "dim": dim, "method": method, "seeds": 3,
            "forward_accuracy_mean": 0.9, "sipit_accuracy_mean": 0.04,
            "defense_aware_forward_mean": da_fwd,
            "defense_aware_sipit_mean": da_fwd,
            "ppl_after_mean": 120.0, "ppl_ratio_mean": 1.2,
            "wall_seconds_mean": wall, "attack_seconds_mean": attack,
            "gate_1_5x": g15, "gate_2x": g2,
            "pass_1_5x_seeds": 3 if g15 == "PASS" else 0,
            "pass_2x_seeds": 3 if g2 == "PASS" else 0}


def wb_seed_rows(split, dim, method, leaks, seeds=SEEDS, attack=10.0):
    rows = []
    for seed, leak in zip(seeds, leaks):
        rows.append({"split": split, "dim": dim, "method": method,
                     "seed": seed, "wb_inverter_forward": leak,
                     "wb_inverter_sipit": leak, "wb_opt_forward": leak,
                     "wb_opt_snap": leak, "ppl_ratio": 1.2,
                     "wall_seconds": 100.0, "attack_seconds": attack})
    return rows


def base_seed_rows(split, dim, method, seeds=SEEDS, fwd=0.04,
                   sipit=0.04, da_fwd=0.05, da_sipit=0.05,
                   pass_15=True, pass_20=True):
    """Base-tier per-seed rows (low by default so the white-box arms
    dominate the strongest-observed statistic)."""
    return [{"split": split, "dim": dim, "method": method, "seed": seed,
             "forward_accuracy": fwd, "sipit_accuracy": sipit,
             "da_forward_accuracy": da_fwd, "da_sipit_accuracy": da_sipit,
             "ppl_after": 120.0, "ppl_ratio": 1.2, "pass_1_5x": pass_15,
             "pass_2x": pass_20, "wall_seconds": 100.0,
             "attack_seconds": 10.0, "cuda_peak_allocated_bytes": 0}
            for seed in seeds]


def make_analysis(tb_frozen_leaks=(0.10, 0.11, 0.12),
                  tb_frozen_g15="PASS", tb_frozen_g2="PASS",
                  tb_frozen_wall=100.0, tb_cotrain_leaks=(0.15, 0.16,
                                                          0.17),
                  control_leaks=(0.30, 0.31, 0.32), seeds=SEEDS,
                  tb_frozen_base=None, cotrain_wb_na=False):
    summary, wb_rows, base_rows = [], [], []
    for split in SPLITS:
        summary.append(summary_row(split, 64, "pca-compress",
                                   da_fwd=0.06))
        summary.append(summary_row(split, 64, "tb-frozen",
                                   g15=tb_frozen_g15, g2=tb_frozen_g2,
                                   wall=tb_frozen_wall))
        summary.append(summary_row(split, 64, "tb-cotrain"))
        wb_rows += wb_seed_rows(split, 64, "pca-compress", control_leaks,
                                seeds=seeds)
        wb_rows += wb_seed_rows(split, 64, "tb-frozen", tb_frozen_leaks,
                                seeds=seeds)
        if not cotrain_wb_na:
            wb_rows += wb_seed_rows(split, 64, "tb-cotrain",
                                    tb_cotrain_leaks, seeds=seeds)
        base_rows += base_seed_rows(split, 64, "pca-compress",
                                    seeds=seeds)
        base_rows += base_seed_rows(split, 64, "tb-frozen", seeds=seeds,
                                    pass_15=tb_frozen_g15 == "PASS",
                                    pass_20=tb_frozen_g2 == "PASS",
                                    **(tb_frozen_base or {}))
        base_rows += base_seed_rows(split, 64, "tb-cotrain", seeds=seeds)
    analysis = {"schema": "llm_attacker.bottleneck_sweep_analysis.v1",
                "status": "COMPLETE", "summary": summary,
                "whitebox_summary": [], "seed_rows": base_rows,
                "whitebox_seed_rows": wb_rows, "failures": []}
    if cotrain_wb_na:
        analysis["whitebox_not_evaluable"] = [
            {"split": split, "dim": 64, "method": "tb-cotrain",
             "reason": "co-trained encoder: training-phase captures "
                       "predate the saved final checkpoint"}
            for split in SPLITS]
    return analysis


def write_analysis(tmp_path, analysis):
    path = tmp_path / "analysis.json"
    path.write_text(json.dumps(analysis))
    return path


def candidate(record, split, method):
    return [c for c in record["splits"][str(split)]["candidates"]
            if c["method"] == method][0]


def test_primary_selection(tmp_path):
    record = selector.select(make_analysis())
    assert record["schema"] == "llm_attacker.operating_point_selection.v1"
    assert record["v3_status"] == "UNFROZEN"
    assert record["negative_splits"] == []
    for split in ("3", "6"):
        r = record["splits"][split]
        assert r["outcome"] == "primary"
        # both candidates eligible; lowest white-box leakage selected
        assert r["selected"] == {"method": "tb-frozen", "dim": 64,
                                 "gate": "primary"}
        assert all(c["eligible"] for c in r["candidates"])
    # access / approximation labels carried verbatim
    labels = record["access_labels"]
    assert "NOT a faithful decoder" in labels["surrogate_arm"]
    assert "strictly stronger" in labels["whitebox_tier"]
    assert "never freezes" in record["v3_rule"]


def test_secondary_label(tmp_path):
    record = selector.select(make_analysis(tb_frozen_g15="FAIL",
                                           tb_cotrain_leaks=(0.40, 0.41,
                                                             0.42)))
    for split in ("3", "6"):
        r = record["splits"][split]
        assert r["outcome"] == "secondary"
        assert r["selected"]["gate"] == "secondary"
        cotrain = [c for c in r["candidates"]
                   if c["method"] == "tb-cotrain"][0]
        assert not cotrain["eligible"]  # worse than the PCA control


def test_negative_label_retains_result_and_v3_unfrozen(tmp_path):
    record = selector.select(make_analysis(tb_frozen_g15="FAIL",
                                           tb_frozen_g2="FAIL",
                                           tb_cotrain_leaks=(0.40, 0.41,
                                                             0.42)))
    for split in ("3", "6"):
        r = record["splits"][split]
        assert r["outcome"] == "negative"
        assert r["selected"] is None
        gate_fail = [c for c in r["candidates"]
                     if c["method"] == "tb-frozen"][0]
        assert any("predefined thresholds" in reason
                   for reason in gate_fail["reasons"])
    assert record["negative_splits"] == ["3", "6"]
    assert "NEGATIVE" in record["negative_statement"]
    assert "UNFROZEN" in record["negative_statement"]
    assert record["v3_status"] == "UNFROZEN"


def test_threshold_override_refused(tmp_path, capsys):
    analysis = write_analysis(tmp_path, make_analysis())
    rc = selector.main(["--analysis", str(analysis),
                        "--output", str(tmp_path / "sel.json"),
                        "--thresholds", "1.4", "2.2"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "no-retuning" in err and "PREDEFINED" in err
    assert not (tmp_path / "sel.json").exists()


def test_pca_superiority_gating(tmp_path):
    # candidate worse than the PCA control -> ineligible, comparison
    # still recorded
    record = selector.select(make_analysis(tb_frozen_leaks=(0.40, 0.41,
                                                            0.42)))
    cand = candidate(record, 3, "tb-frozen")
    assert not cand["eligible"]
    assert any("NOT below" in r for r in cand["reasons"])
    comp = cand["pca_leakage_comparison"]
    assert comp["point_estimate"] < 0
    assert comp["paired_seeds"] == list(SEEDS)
    assert comp["ci_low"] is not None
    assert record["splits"]["3"]["outcome"] == "primary"  # cotrain OK


def test_mixed_diffs_fail_on_ci(tmp_path):
    # point estimate positive but one paired diff negative -> CI lower
    # bound crosses 0 -> ineligible (fail closed on the interval)
    record = selector.select(make_analysis(tb_frozen_leaks=(0.10, 0.10,
                                                            0.35),
                                           tb_cotrain_leaks=(0.40, 0.40,
                                                             0.40)))
    cand = candidate(record, 3, "tb-frozen")
    comp = cand["pca_leakage_comparison"]
    assert comp["point_estimate"] > 0
    assert not cand["eligible"]
    assert any("includes 0" in r for r in cand["reasons"])
    assert record["splits"]["3"]["outcome"] == "negative"


def test_compute_factor_enforced(tmp_path):
    record = selector.select(make_analysis(tb_frozen_wall=250.0,
                                           tb_cotrain_leaks=(0.40, 0.41,
                                                             0.42)))
    cand = candidate(record, 3, "tb-frozen")
    comp = cand["compute_comparison"]
    # (250 wall + 10 base attack + 10 wb attack) / (100 + 10 + 10)
    assert comp["ratio"] == pytest.approx(270.0 / 120.0)
    assert comp["max_factor"] == 2.0
    assert not comp["within_factor"]
    assert not cand["eligible"]
    assert any("compute ratio" in r for r in cand["reasons"])
    assert record["splits"]["3"]["outcome"] == "negative"


def test_whitebox_attack_compute_included_in_factor(tmp_path):
    # regression: white-box attack compute was excluded from the compute
    # gate; a huge white-box bill must count toward the 2x factor
    analysis = make_analysis(tb_cotrain_leaks=(0.40, 0.41, 0.42))
    for row in analysis["whitebox_seed_rows"]:
        if row["method"] == "tb-frozen":
            row["attack_seconds"] = 300.0
    record = selector.select(analysis)
    cand = candidate(record, 3, "tb-frozen")
    comp = cand["compute_comparison"]
    assert comp["whitebox_attack_seconds_mean"] == pytest.approx(300.0)
    # (100 wall + 10 base attack + 300 wb attack) / (100 + 10 + 10)
    assert comp["ratio"] == pytest.approx(410.0 / 120.0)
    assert not comp["within_factor"]
    assert not cand["eligible"]
    assert any("compute ratio" in r for r in cand["reasons"])


def test_failed_cells_in_analysis_fail_closed(tmp_path):
    # regression: the selector accepted analyses containing FAILED cells
    analysis = make_analysis()
    analysis["status"] = "FAILED_CELLS"
    analysis["failures"] = [{"cell": "split3/seed-11/dim-64/tb-frozen",
                             "reason": "status 'FAILED'"}]
    record = selector.select(analysis)
    assert record["analysis_status"] == "FAILED_CELLS"
    for split in ("3", "6"):
        r = record["splits"][split]
        assert r["outcome"] == "negative"
        assert r["selected"] is None
        assert all(not c["eligible"] for c in r["candidates"])
        assert any("FAILED" in reason
                   for c in r["candidates"] for reason in c["reasons"])
    assert record["negative_splits"] == ["3", "6"]
    assert record["negative_statement"]


def test_utility_gates_recomputed_from_per_seed_rows(tmp_path):
    # regression: summary PASS labels were trusted even when the per-seed
    # rows contradict them; the verdict must be recomputed from seed_rows
    analysis = make_analysis()
    for row in analysis["seed_rows"]:
        if (row["method"], row["split"], row["seed"]) == ("tb-frozen", 3,
                                                          33):
            row["pass_1_5x"] = False
            row["ppl_ratio"] = 1.7
    record = selector.select(analysis)
    cand = candidate(record, 3, "tb-frozen")
    assert not cand["eligible"]
    assert any("per-seed" in r for r in cand["reasons"])
    rec = cand["utility_gate_recomputed"]
    assert rec["gate_1_5x"] == "FAIL" and rec["gate_2x"] == "PASS"
    # split 6's per-seed rows are consistent with the PASS label
    assert candidate(record, 6, "tb-frozen")["eligible"]


def test_nonfinite_arm_value_fails_closed(tmp_path):
    # regression: NaN disappeared inside the max-across-arms
    analysis = make_analysis()
    row = [r for r in analysis["whitebox_seed_rows"]
           if (r["method"], r["split"], r["seed"]) == ("tb-frozen", 3,
                                                       22)][0]
    row["wb_opt_forward"] = float("nan")
    record = selector.select(analysis)
    cand = candidate(record, 3, "tb-frozen")
    assert not cand["eligible"]
    assert any("invalid attack value" in r for r in cand["reasons"])


def test_out_of_range_arm_value_fails_closed(tmp_path):
    analysis = make_analysis()
    row = [r for r in analysis["whitebox_seed_rows"]
           if (r["method"], r["split"], r["seed"]) == ("tb-frozen", 3,
                                                       22)][0]
    row["wb_inverter_forward"] = 1.5  # accuracies live in [0, 1]
    record = selector.select(analysis)
    cand = candidate(record, 3, "tb-frozen")
    assert not cand["eligible"]
    assert any("invalid attack value" in r for r in cand["reasons"])


def test_missing_arm_value_fails_closed_not_crash(tmp_path):
    analysis = make_analysis()
    row = [r for r in analysis["whitebox_seed_rows"]
           if (r["method"], r["split"], r["seed"]) == ("tb-frozen", 3,
                                                       22)][0]
    del row["wb_opt_snap"]
    record = selector.select(analysis)
    cand = candidate(record, 3, "tb-frozen")
    assert not cand["eligible"]
    assert any("invalid attack value" in r or "missing" in r
               for r in cand["reasons"])


def test_cotrain_whitebox_not_evaluable(tmp_path):
    # regression (co-trained white-box mismatch): with the white-box tier
    # recorded NOT_EVALUABLE for tb-cotrain, the strongest-observed
    # statistic is computed over the remaining measured arms and the
    # exclusion is recorded, never silently evaluated against the
    # mismatched final encoder
    record = selector.select(make_analysis(cotrain_wb_na=True))
    cand = candidate(record, 3, "tb-cotrain")
    assert cand["eligible"]
    comp = cand["pca_leakage_comparison"]
    assert comp["whitebox_tier"].startswith("NOT_EVALUABLE")
    # statistic falls back to the measured base-tier arms (max 0.05)
    assert all(v == pytest.approx(0.05)
               for v in comp["candidate_per_seed"].values())
    # the PCA control still has its full white-box tier
    assert all(v == pytest.approx(l) for v, l in zip(
        comp["control_per_seed"].values(), (0.30, 0.31, 0.32)))
    # with the white-box arms excluded, tb-cotrain's measured maximum
    # (0.05) is the lowest and wins the split — eligible, with the
    # exclusion recorded rather than silently evaluated
    assert record["splits"]["3"]["selected"]["method"] == "tb-cotrain"
    assert record["splits"]["3"]["outcome"] == "primary"


def test_missing_whitebox_rows_ineligible(tmp_path):
    analysis = make_analysis()
    analysis["whitebox_seed_rows"] = [
        r for r in analysis["whitebox_seed_rows"]
        if not (r["method"] == "tb-frozen" and r["split"] == 3)]
    record = selector.select(analysis)
    cand = candidate(record, 3, "tb-frozen")
    assert not cand["eligible"]
    assert any("fail closed" in r for r in cand["reasons"])
    # split 6 unaffected
    assert record["splits"]["6"]["outcome"] == "primary"


def test_missing_gate_fields_ineligible(tmp_path):
    analysis = make_analysis()
    for row in analysis["summary"]:
        if row["method"] == "tb-frozen" and row["split"] == 3:
            del row["gate_1_5x"]
    record = selector.select(analysis)
    cand = candidate(record, 3, "tb-frozen")
    assert not cand["eligible"]
    assert any("gate_1_5x" in r for r in cand["reasons"])


def test_missing_per_seed_utility_rows_ineligible(tmp_path):
    analysis = make_analysis()
    analysis["seed_rows"] = [
        r for r in analysis["seed_rows"]
        if not (r["method"] == "tb-frozen" and r["split"] == 3)]
    record = selector.select(analysis)
    cand = candidate(record, 3, "tb-frozen")
    assert not cand["eligible"]


def test_determinism(tmp_path):
    analysis = write_analysis(tmp_path, make_analysis())
    out1, out2 = tmp_path / "a.json", tmp_path / "b.json"
    assert selector.main(["--analysis", str(analysis), "--output",
                          str(out1)]) == 0
    assert selector.main(["--analysis", str(analysis), "--output",
                          str(out2)]) == 0
    assert out1.read_bytes() == out2.read_bytes()


def test_markdown_render(tmp_path):
    analysis = write_analysis(tmp_path, make_analysis())
    md = tmp_path / "sel.md"
    assert selector.main(["--analysis", str(analysis), "--output",
                          str(tmp_path / "sel.json"),
                          "--markdown", str(md)]) == 0
    text = md.read_text()
    assert "Outcome per split" in text
    assert "| 3 | primary | tb-frozen dim=64 (primary gate) |" in text
    assert "NOT a faithful decoder" in text
    assert "strictly stronger" in text
    assert "UNFROZEN" in text


def test_bad_analysis_schema_rejected(tmp_path):
    with pytest.raises(ValueError, match="schema"):
        selector.select({"schema": "bogus"})


def test_bootstrap_and_all_seeds_conditions_reported_separately(tmp_path):
    record = selector.select(make_analysis())
    comp = candidate(record, 3, "tb-frozen")["pca_leakage_comparison"]
    boot = comp["bootstrap_condition"]
    asp = comp["all_seeds_positive_condition"]
    # two SEPARATE mechanical conditions, both recorded, both passing
    assert boot["point_estimate_positive"] is True
    assert boot["ci_low_positive"] is True
    assert boot["passed"] is True
    assert asp["passed"] is True
    assert all(asp["per_seed_diffs_positive"].values())
    assert "eligibility_conditions" in record
    assert "BOTH" in record["eligibility_conditions"]


def test_all_seeds_positive_required_even_when_ci_passes(tmp_path):
    # 4 seeds: CI lower > 0 (bootstrap condition passes) but one per-seed
    # paired diff is exactly 0 -> all-seeds-positive condition FAILS ->
    # INELIGIBLE. The bootstrap condition does NOT imply improvement on
    # every paired seed.
    seeds4 = (11, 22, 33, 44)
    analysis = make_analysis(seeds=seeds4,
                             tb_frozen_leaks=(0.10, 0.10, 0.10, 0.30),
                             tb_cotrain_leaks=(0.10, 0.10, 0.10, 0.31),
                             control_leaks=(0.30, 0.30, 0.30, 0.30))
    record = selector.select(analysis)
    comp = candidate(record, 3, "tb-frozen")["pca_leakage_comparison"]
    assert comp["bootstrap_condition"]["passed"] is True
    assert comp["ci_low"] > 0
    assert comp["all_seeds_positive_condition"]["passed"] is False
    assert comp["all_seeds_positive_condition"][
        "per_seed_diffs_positive"]["44"] is False
    assert not candidate(record, 3, "tb-frozen")["eligible"]
    assert any("every" in r
               for r in candidate(record, 3, "tb-frozen")["reasons"])
    # cotrain has the same single zero diff -> also ineligible -> negative
    assert record["splits"]["3"]["outcome"] == "negative"


def test_strongest_observed_max_across_all_measured_arms(tmp_path):
    # a base-tier arm (defense-aware surrogate) exceeds every white-box
    # arm for the candidate -> it must determine the statistic
    analysis = make_analysis(tb_frozen_base={"da_fwd": 0.25,
                                             "da_sipit": 0.25})
    record = selector.select(analysis)
    comp = candidate(record, 3, "tb-frozen")["pca_leakage_comparison"]
    assert comp["statistic_label"].startswith("strongest observed")
    # candidate per-seed leakage is the base-tier 0.25, not the wb 0.10+
    assert all(v == pytest.approx(0.25)
               for v in comp["candidate_per_seed"].values())
    # and it gates eligibility: candidate now loses to the control on
    # the last seed (0.32 - 0.25 > 0 still)... control base is low, so
    # control stays 0.30/0.31/0.32 and diffs stay positive
    assert comp["all_seeds_positive_condition"]["passed"] is True


def test_base_tier_arm_can_make_candidate_ineligible(tmp_path):
    # base-tier surrogate arm stronger than the control's strongest
    # observed result on every seed -> diffs negative -> ineligible
    analysis = make_analysis(tb_frozen_base={"da_fwd": 0.40,
                                             "da_sipit": 0.40})
    record = selector.select(analysis)
    cand = candidate(record, 3, "tb-frozen")
    assert not cand["eligible"]
    assert any("NOT below" in r for r in cand["reasons"])


def test_no_bound_wording_in_emitted_records(tmp_path):
    # regression: a "bounds ... from above" claim survived the wording
    # audit; white-box rows are stronger-access measurements, not bounds
    record = selector.select(make_analysis())
    text = json.dumps(record).lower()
    assert "bound" not in text
    assert "strongest observed" in text
    md = selector.render_markdown(record).lower()
    assert "bound" not in md
    assert "strongest observed" in md
    # fresh-seed validation record is clean too
    root = make_validation_dir(tmp_path)
    vrec = fresh.validate(root, record)
    assert vrec["outcome"] == "passed"
    vtext = json.dumps(vrec).lower()
    assert "bound" not in vtext
    assert "bound" not in fresh.render_markdown(vrec).lower()


# ---------------------------------------------------------------------------
# fresh-seed validation fixtures

ACCESS = {"schema": "llm_attacker.bottleneck_sweep_access.v1",
          "matched_across_defenses": True}
WB_PLAN_SCHEMA = "llm_attacker.whitebox_sweep.v1"
WB_CELL_SCHEMA = "llm_attacker.whitebox_sweep_cell.v1"


def _seal(path):
    with EvidenceLogger(str(path)):
        pass


def make_cell(root, split, seed, method, ppl_after, dim=64,
              status="COMPLETE", da_fwd=0.05, fwd=None, sipit=None,
              wall=100.0, attack=10.0):
    if fwd is None:
        fwd = 0.9 if method == "none" else 0.05
    if sipit is None:
        sipit = 0.9 if method == "none" else 0.04
    rel = Path(f"split{split}") / f"seed-{seed}" / f"dim-{dim}" / method
    cell_dir = root / rel
    cell_dir.mkdir(parents=True, exist_ok=True)
    prov = cell_dir / "provenance.json"
    prov.write_text(json.dumps({"cell": str(rel)}))
    run_json = cell_dir / "run.json"
    run_json.write_text(json.dumps({"cell": str(rel)}))
    _seal(run_json)
    cell = {"schema": "llm_attacker.bottleneck_sweep_cell.v1",
            "status": status, "split": split, "seed": seed,
            "method": method, "dim": dim,
            "wall_seconds": wall, "attack_seconds": attack,
            "cuda_peak_allocated_bytes": 0,
            "utility_before": {"cross_entropy": 5.0, "perplexity": 150.0},
            "utility_after": {"cross_entropy": 4.8,
                              "perplexity": ppl_after},
            "access": dict(ACCESS),
            "provenance": {"path": "provenance.json",
                           "sha256": hashlib.sha256(prov.read_bytes())
                           .hexdigest()},
            "run_artifact": "run.json",
            "evidence": "run.json.evidence.json",
            "attacks": {"forward": {"accuracy": fwd},
                        "sipit": {"accuracy": sipit}}}
    if method != "none":
        cell["attacks"]["defense_aware"] = {
            "defense_aware": True,
            "forward": {"accuracy": da_fwd},
            "sipit": {"accuracy": da_fwd}}
        ratio = ppl_after / 100.0
        cell["utility_gate"] = {
            "ratio": ratio, "pass_1_5x": ratio <= 1.5,
            "pass_2x": ratio <= 2.0}
    (cell_dir / "cell.json").write_text(json.dumps(cell))
    return str(rel)


def make_wb_cell(root, split, seed, method, ppl_after=120.0, dim=64,
                 leak=0.10, attack=10.0):
    rel = Path("whitebox") / f"split{split}" / f"seed-{seed}" / \
        f"{method}-{dim}"
    cell_dir = root / rel
    cell_dir.mkdir(parents=True, exist_ok=True)
    ckpt = cell_dir / "encoder.pt"
    ckpt.write_bytes(b"fixture true defense checkpoint")
    ratio = ppl_after / 100.0
    cell = {"schema": WB_CELL_SCHEMA, "status": "COMPLETE",
            "tier": "whitebox", "split": split, "seed": seed,
            "method": method, "dim": dim,
            "base_cell": f"split{split}/seed-{seed}/dim-{dim}/{method}",
            "defense_checkpoint": {
                "path": str(ckpt.relative_to(root)),
                "sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest()},
            "wall_seconds": attack, "attack_seconds": attack,
            "utility_after": {"cross_entropy": 4.8,
                              "perplexity": ppl_after},
            "utility_gate": {"ratio": ratio, "pass_1_5x": ratio <= 1.5,
                             "pass_2x": ratio <= 2.0},
            "access": {"schema": "llm_attacker.whitebox_sweep_access.v1",
                       "whitebox_extension": {
                           "granted": "the actual trained defense "
                                      "checkpoint"}},
            "evidence": "cell.json.evidence.json",
            "attacks": {
                "whitebox_inverter": {"forward": {"accuracy": leak},
                                      "sipit": {"accuracy": leak}},
                "whitebox_optimization": {"forward": {"accuracy": leak},
                                          "snap": {"accuracy": leak}}}}
    (cell_dir / "cell.json").write_text(json.dumps(cell))
    _seal(cell_dir / "cell.json")
    return str(rel)


def make_validation_dir(tmp_path, candidate_ppl=120.0, seeds=FRESH_SEEDS,
                        drop_control=None, fail_cell=None,
                        candidate_leak=0.10, candidate_wb_leak=None,
                        control_leak=0.30, candidate_wall=100.0,
                        control_wall=100.0, splits=SPLITS,
                        selected_method="tb-frozen", extra_methods=(),
                        with_whitebox=True):
    """A fresh-seed validation dir covering EXACTLY the selected
    configuration plus the none/pca-compress controls (and the matching
    white-box addendum tier) at every fresh seed."""
    root = tmp_path / "validation"
    paths, wb_paths = [], []
    for split in splits:
        for seed in seeds:
            if (split, seed) != drop_control:
                paths.append(make_cell(root, split, seed, "none", 100.0))
            paths.append(make_cell(root, split, seed, "pca-compress",
                                   120.0, fwd=control_leak,
                                   sipit=control_leak,
                                   da_fwd=control_leak,
                                   wall=control_wall))
            paths.append(make_cell(
                root, split, seed, selected_method, candidate_ppl,
                status="FAILED" if (split, seed) == fail_cell
                else "COMPLETE",
                fwd=candidate_leak, sipit=candidate_leak,
                da_fwd=candidate_leak, wall=candidate_wall))
            for extra in extra_methods:
                paths.append(make_cell(root, split, seed, extra, 110.0))
            if with_whitebox:
                wb_paths.append(make_wb_cell(root, split, seed,
                                             "pca-compress",
                                             leak=control_leak))
                if selected_method != "tb-cotrain":
                    wb_paths.append(make_wb_cell(
                        root, split, seed, selected_method,
                        ppl_after=candidate_ppl,
                        leak=(candidate_wb_leak if candidate_wb_leak
                              is not None else candidate_leak)))
    methods = ["none", "pca-compress", selected_method] + list(
        extra_methods)
    plan = {"schema": "llm_attacker.bottleneck_sweep.v1",
            "tier": "fresh_seed_validation", "seeds": list(seeds),
            "splits": list(splits), "dims": [64],
            "methods": methods,
            "cells": [{"path": p} for p in paths],
            "expected_cells": len(paths),
            "corpus": {"path": "corpus.jsonl"}}
    root.mkdir(exist_ok=True)
    (root / "plan.json").write_text(json.dumps(plan))
    if with_whitebox:
        (root / "whitebox").mkdir(exist_ok=True)
        (root / "whitebox" / "plan.json").write_text(json.dumps(
            {"schema": WB_PLAN_SCHEMA, "tier": "whitebox",
             "cells": [{"path": p} for p in wb_paths]}))
    return root


def selection_record():
    return selector.select(make_analysis())


def cotrain_selection_record():
    return {"schema": "llm_attacker.operating_point_selection.v1",
            "splits": {str(s): {"selected": {"method": "tb-cotrain",
                                             "dim": 64,
                                             "gate": "primary"}}
                       for s in SPLITS}}


def test_fresh_validation_passes(tmp_path):
    root = make_validation_dir(tmp_path)
    record = fresh.validate(root, selection_record())
    assert record["schema"] == "llm_attacker.fresh_seed_validation.v1"
    assert record["outcome"] == "passed"
    assert record["failures"] == []
    for split in ("3", "6"):
        e = record["splits"][split]
        assert all(e["control_coverage"].values())
        v = e["fresh_verdict"]
        assert v["gate"] == "primary"
        assert v["fresh_ppl_ratios"] == [pytest.approx(1.2)] * 3
        # all eligibility conditions re-evaluated on the fresh seeds
        assert v["pca_superiority"]["bootstrap_condition"]["passed"]
        assert v["pca_superiority"]["all_seeds_positive_condition"][
            "passed"]
        assert v["pca_superiority"]["statistic"].startswith(
            "strongest observed")
        assert v["compute_comparison"]["within_factor"]
        assert v["compute_comparison"]["max_factor"] == 2.0


def test_fresh_validation_secondary(tmp_path):
    root = make_validation_dir(tmp_path, candidate_ppl=180.0)
    record = fresh.validate(root, selection_record())
    assert record["outcome"] == "passed"
    for split in ("3", "6"):
        assert record["splits"][split]["fresh_verdict"]["gate"] == \
            "secondary"


def test_fresh_validation_missing_control_fails(tmp_path):
    root = make_validation_dir(tmp_path, drop_control=(6, 77))
    record = fresh.validate(root, selection_record())
    assert record["outcome"] == "failed"
    assert any("paired undefended control" in f["reason"]
               and "split6/seed-77" in f["cell"]
               for f in record["failures"])
    assert record["splits"]["6"]["control_coverage"]["77"] is False


def test_fresh_validation_failed_cell_propagates(tmp_path):
    root = make_validation_dir(tmp_path, fail_cell=(3, 44))
    record = fresh.validate(root, selection_record())
    assert record["outcome"] == "failed"
    assert any("split3/seed-44" in f["cell"] for f in record["failures"])


def test_fresh_only_rule(tmp_path):
    root = make_validation_dir(tmp_path, seeds=(11,) + FRESH_SEEDS)
    record = fresh.validate(root, selection_record())
    assert record["outcome"] == "failed"
    assert any("fresh-only rule" in f["reason"]
               for f in record["failures"])


def test_fresh_gate_regression_fails(tmp_path):
    root = make_validation_dir(tmp_path, candidate_ppl=250.0)
    record = fresh.validate(root, selection_record())
    assert record["outcome"] == "failed"
    assert any("fresh-seed utility gate FAIL" in f["reason"]
               for f in record["failures"])
    assert record["splits"]["3"]["fresh_verdict"]["gate"] == "fail"


def test_fresh_empty_plan_fails_closed(tmp_path):
    # regression: an empty fresh-validation plan passed
    root = tmp_path / "validation"
    root.mkdir()
    plan = {"schema": "llm_attacker.bottleneck_sweep.v1",
            "tier": "fresh_seed_validation", "seeds": list(FRESH_SEEDS),
            "splits": [], "dims": [], "methods": [], "cells": [],
            "expected_cells": 0, "corpus": {"path": "corpus.jsonl"}}
    (root / "plan.json").write_text(json.dumps(plan))
    record = fresh.validate(root, selection_record())
    assert record["outcome"] == "failed"
    assert any("covers" in f["reason"] or "coverage" in f["reason"]
               for f in record["failures"])


def test_fresh_plan_must_cover_every_selected_split(tmp_path):
    # regression: a selected split omitted from the fresh run escaped
    # checking entirely
    root = make_validation_dir(tmp_path, splits=(3,))
    record = fresh.validate(root, selection_record())
    assert record["outcome"] == "failed"
    assert any("6" in f["reason"] for f in record["failures"])


def test_fresh_extra_cells_beyond_controls_fail_closed(tmp_path):
    # regression (retest-only): extra non-selected cells must not even be
    # loaded; the plan covers EXACTLY the selected config plus controls
    root = make_validation_dir(tmp_path, extra_methods=("tb-cotrain",))
    selection = selection_record()
    record = fresh.validate(root, selection)
    assert record["outcome"] == "failed"
    assert any("beyond the selected configuration" in f["reason"]
               for f in record["failures"])
    # and the emitted selection is still byte-identical to the input's
    assert record["selected_configurations"] == {
        s: selection["splits"][s]["selected"] for s in ("3", "6")}


def test_fresh_pca_superiority_retested_on_fresh_seeds(tmp_path):
    # regression: fresh validation only re-checked the utility gate; a
    # candidate whose fresh leakage regresses above the pca-compress
    # control must FAIL even with the utility gate passing
    root = make_validation_dir(tmp_path, candidate_leak=0.35)
    record = fresh.validate(root, selection_record())
    assert record["outcome"] == "failed"
    assert any("pca-compress" in f["reason"] and "leakage" in f["reason"]
               for f in record["failures"])
    v = record["splits"]["3"]["fresh_verdict"]
    assert v["gate"] == "primary"  # utility still passes
    assert not v["pca_superiority"]["passed"]


def test_fresh_strongest_observed_uses_all_eight_arms(tmp_path):
    # regression: the fresh leakage statistic used only the defense-aware
    # forward arm; a white-box arm above the control's max must gate
    root = make_validation_dir(tmp_path, candidate_leak=0.05,
                               candidate_wb_leak=0.45)
    record = fresh.validate(root, selection_record())
    assert record["outcome"] == "failed"
    v = record["splits"]["3"]["fresh_verdict"]
    assert not v["pca_superiority"]["passed"]
    assert all(val == pytest.approx(0.45)
               for val in v["pca_superiority"]
               ["candidate_per_seed"].values())


def test_fresh_compute_factor_retested(tmp_path):
    # regression: the compute factor was never re-checked on fresh data
    root = make_validation_dir(tmp_path, candidate_wall=500.0)
    record = fresh.validate(root, selection_record())
    assert record["outcome"] == "failed"
    assert any("compute" in f["reason"] for f in record["failures"])
    v = record["splits"]["3"]["fresh_verdict"]
    assert not v["compute_comparison"]["within_factor"]


def test_fresh_missing_whitebox_tier_fails_closed(tmp_path):
    # the strongest-observed statistic cannot be recomputed for a
    # tb-frozen selection without the fresh white-box addendum
    root = make_validation_dir(tmp_path, with_whitebox=False)
    record = fresh.validate(root, selection_record())
    assert record["outcome"] == "failed"
    assert any("white-box" in f["reason"] for f in record["failures"])


def test_fresh_cotrain_whitebox_not_evaluable(tmp_path):
    # tb-cotrain's training-phase white-box cells are NOT_EVALUABLE (the
    # saved checkpoint is the final co-trained encoder, mismatched with
    # the capture-time encoder); fresh validation proceeds over the
    # remaining measured arms with the exclusion recorded
    root = make_validation_dir(tmp_path, selected_method="tb-cotrain")
    record = fresh.validate(root, cotrain_selection_record())
    assert record["outcome"] == "passed"
    v = record["splits"]["3"]["fresh_verdict"]
    assert v["pca_superiority"]["whitebox_tier"].startswith(
        "NOT_EVALUABLE")


def test_fresh_validation_cli_and_markdown(tmp_path):
    root = make_validation_dir(tmp_path)
    sel_path = tmp_path / "selection.json"
    sel_path.write_text(json.dumps(selection_record()))
    out, md = tmp_path / "val.json", tmp_path / "val.md"
    assert fresh.main(["--root", str(root), "--selection", str(sel_path),
                       "--output", str(out), "--markdown", str(md)]) == 0
    text = md.read_text()
    assert "Fresh-seed validation" in text
    assert "passed" in text
    assert "selection-seed results never reused" in text
    assert "retest only" in text
    # failing validation exits nonzero
    bad_root = make_validation_dir(tmp_path / "bad",
                                   drop_control=(3, 55))
    assert fresh.main(["--root", str(bad_root), "--selection",
                       str(sel_path), "--output",
                       str(tmp_path / "v2.json")]) == 1


def test_checker_retests_verbatim_selection_even_when_gate_regresses(
        tmp_path):
    # selection claims a secondary-gate operating point; fresh data
    # passes the primary gate. The checker must NOT upgrade or
    # re-derive anything — it reports the fresh verdict against the
    # input record's selection, unchanged.
    root = make_validation_dir(tmp_path, candidate_ppl=120.0)
    selection = selection_record()
    for s in ("3", "6"):
        selection["splits"][s]["selected"]["gate"] = "secondary"
    record = fresh.validate(root, selection)
    assert record["outcome"] == "passed"
    for split in ("3", "6"):
        assert record["splits"][split]["selected"]["gate"] == \
            "secondary"
        v = record["splits"][split]["fresh_verdict"]
        assert v["gate"] == "primary"  # fresh verdict reported as-is
        assert v["selection_label"] == "secondary"


# ---------------------------------------------------------------------------
# adaptive end-to-end (whitebox-e2e) arm in the strongest-observed max


def e2e_seed_rows(split, dim, method, snaps, seeds=SEEDS, attack=5.0):
    return [{"split": split, "dim": dim, "method": method, "seed": seed,
             "adaptive_e2e_snap": snap, "ppl_ratio": 1.2,
             "wall_seconds": 100.0, "attack_seconds": attack}
            for seed, snap in zip(seeds, snaps)]


def add_e2e(analysis, tb_frozen_snaps=(0.20, 0.21, 0.22),
            control_snaps=(0.35, 0.36, 0.37), cotrain_na=True):
    rows = []
    for split in SPLITS:
        rows += e2e_seed_rows(split, 64, "pca-compress", control_snaps)
        rows += e2e_seed_rows(split, 64, "tb-frozen", tb_frozen_snaps)
    analysis["whitebox_e2e_seed_rows"] = rows
    if cotrain_na:
        analysis["whitebox_e2e_not_evaluable"] = [
            {"split": split, "dim": 64, "method": "tb-cotrain",
             "reason": "co-trained encoder: training-phase captures "
                       "predate the saved final checkpoint"}
            for split in SPLITS]
    return analysis


def test_e2e_arm_dominates_per_seed_max(tmp_path):
    # e2e snap above every other candidate arm -> it determines the
    # statistic; still below the control's e2e max -> stays eligible
    record = selector.select(add_e2e(make_analysis()))
    comp = candidate(record, 3, "tb-frozen")["pca_leakage_comparison"]
    assert comp["e2e_tier"] == "measured"
    assert all(comp["candidate_per_seed"][s] == pytest.approx(v)
               for s, v in zip(sorted(comp["candidate_per_seed"]),
                               (0.20, 0.21, 0.22)))
    ctrl = comp["control_per_seed"]
    assert all(ctrl[s] == pytest.approx(0.35 + 0.01 * i)
               for i, s in enumerate(sorted(ctrl)))
    # eligibility still decided by the predefined conditions
    assert candidate(record, 3, "tb-frozen")["eligible"]


def test_e2e_arm_below_max_still_recorded(tmp_path):
    # e2e snap weaker than the white-box arms -> max unchanged, tier
    # still recorded as measured
    analysis = add_e2e(make_analysis(), tb_frozen_snaps=(0.01, 0.01,
                                                         0.01),
                       control_snaps=(0.01, 0.01, 0.01))
    record = selector.select(analysis)
    comp = candidate(record, 3, "tb-frozen")["pca_leakage_comparison"]
    assert comp["e2e_tier"] == "measured"
    assert all(comp["candidate_per_seed"][s] == pytest.approx(v)
               for s, v in zip(sorted(comp["candidate_per_seed"]),
                               (0.10, 0.11, 0.12)))
    ctrl = comp["control_per_seed"]
    assert all(ctrl[s] == pytest.approx(v)
               for s, v in zip(sorted(ctrl), (0.30, 0.31, 0.32)))
    labels = record["access_labels"]
    assert "strictly stronger, adaptive" in labels["whitebox_e2e_tier"]
    assert "adaptive end-to-end" in record["leakage_statistic"]


def test_e2e_absent_tier_backward_compatible(tmp_path):
    # no whitebox_e2e_seed_rows: behavior identical to the pre-e2e tool
    record = selector.select(make_analysis())
    comp = candidate(record, 3, "tb-frozen")["pca_leakage_comparison"]
    assert comp["e2e_tier"] == "absent"
    assert comp["control_e2e_tier"] == "absent"
    for split in ("3", "6"):
        assert record["splits"][split]["outcome"] == "primary"
        assert record["splits"][split]["selected"]["method"] == \
            "tb-frozen"


def test_e2e_not_evaluable_carried_with_reason(tmp_path):
    record = selector.select(add_e2e(make_analysis()))
    comp = candidate(record, 3, "tb-cotrain")["pca_leakage_comparison"]
    assert comp["e2e_tier"].startswith("NOT_EVALUABLE")
    assert "co-trained encoder" in comp["e2e_tier"]
    # the max runs over the remaining measured arms (wb values here)
    assert all(comp["candidate_per_seed"][s] == pytest.approx(v)
               for s, v in zip(sorted(comp["candidate_per_seed"]),
                               (0.15, 0.16, 0.17)))


def test_e2e_invalid_value_fails_closed(tmp_path):
    for bad in (float("nan"), 1.5, -0.1, True, None):
        analysis = add_e2e(make_analysis())
        row = [r for r in analysis["whitebox_e2e_seed_rows"]
               if r["method"] == "tb-frozen" and r["split"] == 3
               and r["seed"] == 11][0]
        row["adaptive_e2e_snap"] = bad
        record = selector.select(analysis)
        cand = candidate(record, 3, "tb-frozen")
        assert not cand["eligible"]
        assert any("invalid adaptive e2e" in r for r in cand["reasons"])


def test_e2e_partial_seed_coverage_fails_closed(tmp_path):
    # e2e rows for only two of three seeds: a partially measured tier
    # is not a legitimate exclusion -> unpairable -> ineligible
    analysis = add_e2e(make_analysis())
    analysis["whitebox_e2e_seed_rows"] = [
        r for r in analysis["whitebox_e2e_seed_rows"]
        if not (r["method"] == "tb-frozen" and r["seed"] == 33)]
    record = selector.select(analysis)
    cand = candidate(record, 3, "tb-frozen")
    assert not cand["eligible"]
    assert any("missing or unpairable" in r for r in cand["reasons"])
