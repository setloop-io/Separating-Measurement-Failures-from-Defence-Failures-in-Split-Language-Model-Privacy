"""Bottleneck-sweep runner/analyzer contracts.

Torch-free layer (runs everywhere): dry-run matrix structure with the
required per-cell files, matched attacker-access spec byte-identity
across methods/dims (and analyzer rejection of drift), frozen-corpus
disjointness firing on a deliberately contaminated manifest fixture, the
utility-gate verdict logic (dual 1.5x/2x thresholds), analyzer fail-closed
on a missing/malformed cell, and the failures-and-nulls section in
report.md.
"""
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


runner = load_script("run_bottleneck_sweep")
analyzer = load_script("analyze_bottleneck_sweep")

DEFENDED = [(s, seed, k, m)
            for s in (3, 6) for seed in (11, 22, 33)
            for k in (64, 128, 256)
            for m in ("pca-compress", "tb-frozen", "tb-cotrain")]
CONTROLS = [(s, seed) for s in (3, 6) for seed in (11, 22, 33)]


def _manifest_fixture(tmp_path, extra_eval_ids=()):
    manifest = json.loads(
        (ROOT / "experiments" / "validation_corpus_manifest.json")
        .read_text())
    manifest["selected_ids"]["evaluation"] = list(
        manifest["selected_ids"]["evaluation"]) + list(extra_eval_ids)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def _corpus_fixture(tmp_path, ids=("wikitext2-train-aaaa",
                                   "wikitext2-train-bbbb")):
    path = tmp_path / "corpus.jsonl"
    path.write_text("".join(json.dumps({"id": i, "text": f"text of {i}"})
                            + "\n" for i in ids))
    return path


def test_dry_run_full_matrix_structure(tmp_path):
    out = tmp_path / "sweep"
    assert runner.main(["--output", str(out), "--dry-run"]) == 0
    plan = json.loads((out / "plan.json").read_text())
    assert plan["expected_cells"] == 60 and len(plan["cells"]) == 60
    assert plan["tier"] == "development_sweep"
    for split, seed in CONTROLS:
        cell = json.loads((out / f"split{split}/seed-{seed}/none"
                           / "cell.json").read_text())
        assert cell["status"] == "PLANNED"
        assert cell["paired_control"] is None
    for split, seed, dim, method in DEFENDED:
        cell_dir = out / f"split{split}/seed-{seed}/dim-{dim}/{method}"
        cell = json.loads((cell_dir / "cell.json").read_text())
        assert cell["status"] == "PLANNED"
        assert cell["paired_control"] == "../../none/cell.json"
        assert cell["dim"] == dim and cell["method"] == method
        assert cell["access"]["schema"] == \
            "llm_attacker.bottleneck_sweep_access.v1"
    assert (out / "access.json").is_file()


def test_dry_run_matched_access_byte_identical(tmp_path):
    out = tmp_path / "sweep"
    runner.main(["--output", str(out), "--dry-run"])
    plan = json.loads((out / "plan.json").read_text())
    rendered = {json.dumps(
        json.loads((out / c["path"] / "cell.json").read_text())["access"],
        sort_keys=True) for c in plan["cells"]}
    assert len(rendered) == 1
    disk = json.loads((out / "access.json").read_text())
    assert json.dumps(disk, sort_keys=True) == rendered.pop()


def test_dry_run_raised_budget_defaults(tmp_path):
    out = tmp_path / "sweep"
    runner.main(["--output", str(out), "--dry-run"])
    plan = json.loads((out / "plan.json").read_text())
    budgets = plan["budgets"]
    assert budgets["forward_candidates"] == 32
    assert budgets["beam_width"] == 4
    assert budgets["sipit_steps"] == 128


def test_disjointness_fires_on_contaminated_manifest(tmp_path):
    corpus = _corpus_fixture(tmp_path)
    contaminated = _manifest_fixture(
        tmp_path, extra_eval_ids=["wikitext2-train-aaaa"])
    with pytest.raises(ValueError, match="overlaps the frozen"):
        runner.assert_development_corpus_disjoint(
            corpus, contaminated, require_excluded_pilot=False)


def test_disjointness_requires_excluded_pilot_identity(tmp_path):
    corpus = _corpus_fixture(tmp_path)  # not the pilot corpus
    manifest = _manifest_fixture(tmp_path)
    with pytest.raises(ValueError, match="not the excluded pilot corpus"):
        runner.assert_development_corpus_disjoint(
            corpus, manifest, require_excluded_pilot=True)


def test_utility_gate_verdict_logic():
    gate = runner.utility_gate(30.0, 20.0)
    assert gate["ratio"] == 1.5
    assert gate["pass_1_5x"] is True and gate["pass_2x"] is True
    gate = runner.utility_gate(30.01, 20.0)
    assert gate["pass_1_5x"] is False and gate["pass_2x"] is True
    gate = runner.utility_gate(41.0, 20.0)
    assert gate["pass_1_5x"] is False and gate["pass_2x"] is False
    # group verdict: all seeds must pass
    cells = [{"utility_gate": runner.utility_gate(p, 20.0)}
             for p in (21.0, 25.0, 31.0)]
    v = analyzer.gate_verdict(cells)
    assert v["gate_1_5x"] == "FAIL" and v["pass_1_5x_seeds"] == 2
    assert v["gate_2x"] == "PASS" and v["pass_2x_seeds"] == 3


def _complete_cell(cell_dir, split, seed, method, dim, access, *,
                   ppl_after=22.0, control_ppl=20.0,
                   drop_provenance_file=False):
    cell_dir.mkdir(parents=True)
    prov = cell_dir / "prov.json"
    prov.write_text(json.dumps({"cell": method}))
    run_json = cell_dir / "run.json"
    run_json.write_text(json.dumps({"cell": method}))
    with EvidenceLogger(str(run_json)):
        pass
    cell = {
        "schema": runner.CELL_SCHEMA, "status": "COMPLETE",
        "measurement_version": 2,
        "observation_location": "body_input_seam",
        "split": split, "seed": seed, "method": method, "dim": dim,
        "wall_seconds": 1.0, "attack_seconds": 0.5,
        "cuda_peak_allocated_bytes": None,
        "utility_before": {"cross_entropy": 4.0, "perplexity": 50.0},
        "utility_after": {"cross_entropy": 3.0, "perplexity": ppl_after},
        "access": access, "run_artifact": "run.json",
        "evidence": "run.json.evidence.json",
        "provenance": {"path": "prov.json",
                       "sha256": __import__("hashlib").sha256(
                           prov.read_bytes()).hexdigest()},
        "attacks": {"forward": {"accuracy": 0.5},
                    "sipit": {"accuracy": 0.4}},
    }
    if method != "none":
        cell["attacks"]["defense_aware"] = {
            "defense_aware": True,
            "forward": {"accuracy": 0.6}, "sipit": {"accuracy": 0.5}}
        cell["utility_gate"] = runner.utility_gate(ppl_after, control_ppl)
    (cell_dir / "cell.json").write_text(json.dumps(cell))
    if drop_provenance_file:
        prov.unlink()


def _synthetic_root(tmp_path, tamper_access=False,
                    drop_provenance_file=False, drop_gate=False):
    root = tmp_path / "sweep"
    root.mkdir()
    access = {"schema": "llm_attacker.bottleneck_sweep_access.v1",
              "matched_across_defenses": True}
    corpus = _corpus_fixture(root)
    cells = []
    rel = "split3/seed-11/none"
    _complete_cell(root / rel, 3, 11, "none", 0, access, ppl_after=20.0)
    cells.append({"path": rel, "split": 3, "seed": 11, "method": "none",
                  "dim": 0})
    for dim in (64, 128):
        for method in runner.METHODS:
            rel = f"split3/seed-11/dim-{dim}/{method}"
            a = dict(access)
            if tamper_access and method == "tb-frozen":
                a["victim_final_head_granted"] = True
            _complete_cell(
                root / rel, 3, 11, method, dim, a,
                drop_provenance_file=(drop_provenance_file
                                      and method == "tb-frozen"))
            if drop_gate and method == "tb-cotrain":
                cell = json.loads((root / rel / "cell.json").read_text())
                del cell["utility_gate"]
                (root / rel / "cell.json").write_text(json.dumps(cell))
            cells.append({"path": rel, "split": 3, "seed": 11,
                          "method": method, "dim": dim})
    plan = {"schema": runner.SCHEMA, "tier": "engineering_smoke",
            "methods": list(runner.METHODS), "dims": [64, 128],
            "splits": [3], "seeds": [11], "expected_cells": len(cells),
            "corpus": {"path": str(corpus)}, "cells": cells,
            "limits": []}
    (root / "plan.json").write_text(json.dumps(plan))
    return root


def test_analyzer_complete_fixture_gates_and_failures_section(tmp_path):
    root = _synthetic_root(tmp_path)
    result = analyzer.analyze(root, _manifest_fixture(tmp_path),
                              replicates=200)
    assert result["status"] == "COMPLETE"
    report = (root / "analysis" / "report.md").read_text()
    assert "## Failures and nulls" in report
    assert "None dropped" in report
    assert "Gate 1.5x" in report
    for name in ("summary.csv", "seed_metrics.csv", "failures.csv",
                 "analysis.json"):
        assert (root / "analysis" / name).is_file()
    summary = (root / "analysis" / "summary.csv").read_text()
    assert "defense_aware_forward_mean" in summary
    assert "ppl_delta_vs_none_mean" in summary
    # fixture defended cells: ppl 22 vs control 20 -> ratio 1.1
    for row in result["summary"]:
        assert row["gate_1_5x"] == "PASS" and row["gate_2x"] == "PASS"


def test_analyzer_fail_closed_on_missing_provenance(tmp_path):
    root = _synthetic_root(tmp_path, drop_provenance_file=True)
    with pytest.raises(ValueError, match="FAILED"):
        analyzer.analyze(root, _manifest_fixture(tmp_path))
    report = (root / "analysis" / "report.md").read_text()
    assert "split3/seed-11/dim-64/tb-frozen" in report
    assert "provenance" in report


def test_analyzer_fail_closed_on_missing_gate(tmp_path):
    root = _synthetic_root(tmp_path, drop_gate=True)
    with pytest.raises(ValueError, match="FAILED"):
        analyzer.analyze(root, _manifest_fixture(tmp_path))
    report = (root / "analysis" / "report.md").read_text()
    assert "utility gate" in report


def test_analyzer_rejects_unmatched_access(tmp_path):
    root = _synthetic_root(tmp_path, tamper_access=True)
    with pytest.raises(ValueError, match="not matched"):
        analyzer.analyze(root, _manifest_fixture(tmp_path))


def test_analyzer_fail_closed_on_missing_evidence(tmp_path):
    # regression: loaders never verified the cell's evidence journal
    root = _synthetic_root(tmp_path)
    (root / "split3/seed-11/dim-64/tb-frozen/run.json.evidence.json"
     ).unlink()
    with pytest.raises(ValueError, match="FAILED"):
        analyzer.analyze(root, _manifest_fixture(tmp_path))
    report = (root / "analysis" / "report.md").read_text()
    assert "split3/seed-11/dim-64/tb-frozen" in report
    assert "evidence" in report


def test_analyzer_fail_closed_on_tampered_evidence(tmp_path):
    root = _synthetic_root(tmp_path)
    events = (root / "split3/seed-11/dim-64/tb-cotrain"
              / "run.json.evidence.jsonl")
    with events.open("a") as handle:
        handle.write('{"seq": 99, "kind": "injected"}\n')
    with pytest.raises(ValueError, match="FAILED"):
        analyzer.analyze(root, _manifest_fixture(tmp_path))
    report = (root / "analysis" / "report.md").read_text()
    assert "evidence" in report


def test_analyzer_fail_closed_on_invalid_attack_value(tmp_path):
    # regression: NaN / out-of-range accuracies passed the loader's
    # isinstance checks and poisoned downstream statistics
    for i, bad in enumerate((float("nan"), float("inf"), 1.5, -0.1)):
        sub = tmp_path / f"bad-{i}"
        sub.mkdir()
        root = _synthetic_root(sub)
        cell_path = root / "split3/seed-11/dim-64/pca-compress/cell.json"
        cell = json.loads(cell_path.read_text())
        cell["attacks"]["forward"]["accuracy"] = bad
        cell_path.write_text(json.dumps(cell))
        with pytest.raises(ValueError, match="FAILED"):
            analyzer.analyze(root, _manifest_fixture(sub))
        report = (root / "analysis" / "report.md").read_text()
        assert "invalid" in report


def test_dry_run_methods_none_plans_controls_only(tmp_path):
    out = tmp_path / "sweep"
    assert runner.main(["--output", str(out), "--dry-run",
                        "--methods", "none"]) == 0
    plan = json.loads((out / "plan.json").read_text())
    assert plan["expected_cells"] == len(CONTROLS)
    assert {(c["split"], c["seed"], c["method"]) for c in plan["cells"]} == \
        {(s, seed, "none") for s, seed in CONTROLS}
    assert not list(out.glob("**/dim-*/**"))


def test_parser_accepts_none_and_min_control_accuracy():
    args = runner.build_parser(
        ["--output", "x", "--methods", "none",
         "--min-control-accuracy", "0.85"])
    assert args.methods == ["none"]
    assert args.min_control_accuracy == 0.85
    args = runner.build_parser(["--output", "x"])
    assert args.min_control_accuracy is None


# ---------------------------------------------------------------------------
# adaptive end-to-end (whitebox-e2e) addendum tier ingestion

E2E_PLAN_SCHEMA = "llm_attacker.whitebox_sweep_e2e.v1"
E2E_CELL_SCHEMA = "llm_attacker.whitebox_sweep_e2e_cell.v1"
E2E_ACCESS = {
    "schema": "llm_attacker.whitebox_sweep_e2e_access.v1",
    "tier": "whitebox-e2e",
    "whitebox_extension": {
        "granted": "the defended cell's ACTUAL trained defense "
                   "checkpoint, used as the exact training-phase "
                   "transform the attack differentiates through"}}


def _e2e_cell(root, split, seed, method, dim, snap=0.02,
              status="COMPLETE"):
    rel = Path("whitebox-e2e") / f"split{split}" / f"seed-{seed}" / \
        f"{method}-{dim}"
    cell_dir = root / rel
    cell_dir.mkdir(parents=True, exist_ok=True)
    cell = {"schema": E2E_CELL_SCHEMA, "status": status,
            "tier": "whitebox-e2e", "split": split, "seed": seed,
            "method": method, "dim": dim,
            "base_cell":
                f"split{split}/seed-{seed}/dim-{dim}/{method}",
            "access": dict(E2E_ACCESS)}
    if status == "COMPLETE":
        cell.update({
            "wall_seconds": 1.0, "attack_seconds": 1.0,
            "defense_checkpoint": {
                "path": f"split{split}/seed-{seed}/dim-{dim}/{method}/"
                        "captures/encoder.pt",  # tensor excluded locally
                "sha256": "0" * 64},
            "utility_after": {"cross_entropy": 3.0, "perplexity": 22.0},
            "utility_gate": {"ratio": 1.1, "pass_1_5x": True,
                             "pass_2x": True},
            "attacks": {"adaptive_e2e": {
                "true_encoder_granted": True, "adaptive": True,
                "snap": {"accuracy": snap}}},
            "evidence": "cell.json.evidence.json"})
        cell_path = cell_dir / "cell.json"
        cell_path.write_text(json.dumps(cell))
        with EvidenceLogger(str(cell_path)):
            pass
    else:
        cell["not_evaluable_reason"] = \
            "co-trained encoder: captures predate the final checkpoint"
        (cell_dir / "cell.json").write_text(json.dumps(cell))
    return {"path": str(rel), "split": split, "seed": seed,
            "method": method, "dim": dim}


def _e2e_tier(root, snap=0.02, not_evaluable=("tb-cotrain",)):
    cells = []
    for dim in (64, 128):
        for method in runner.METHODS:
            status = ("NOT_EVALUABLE" if method in not_evaluable
                      else "COMPLETE")
            cells.append(_e2e_cell(root, 3, 11, method, dim, snap=snap,
                                   status=status))
    plan = {"schema": E2E_PLAN_SCHEMA, "tier": "whitebox-e2e",
            "cells": cells}
    (root / "whitebox-e2e" / "plan.json").write_text(json.dumps(plan))
    return cells


def test_analyzer_whitebox_e2e_tier_ingested(tmp_path):
    root = _synthetic_root(tmp_path)
    _e2e_tier(root)
    result = analyzer.analyze(root, _manifest_fixture(tmp_path),
                              replicates=200)
    assert result["status"] == "COMPLETE"
    assert result["whitebox_e2e_cells_ok"] == 4  # 2 methods x 2 dims
    ne = result["whitebox_e2e_not_evaluable"]
    assert len(ne) == 2  # tb-cotrain dims, carried with reason
    assert all(e["method"] == "tb-cotrain" and "co-trained" in e["reason"]
               for e in ne)
    rows = result["whitebox_e2e_seed_rows"]
    assert len(rows) == 4
    assert all(r["adaptive_e2e_snap"] == 0.02 for r in rows)
    summary = result["whitebox_e2e_summary"]
    assert {r["method"] for r in summary} == {"pca-compress", "tb-frozen"}
    assert all(r["tier"] == "adaptive end-to-end (true transform)"
               for r in summary)
    report = (root / "analysis" / "report.md").read_text()
    assert "## Adaptive end-to-end tier (true transform)" in report
    assert "## Adaptive end-to-end cells NOT_EVALUABLE" in report
    assert "whitebox-e2e/split3/seed-11/tb-cotrain-64" in report
    assert (root / "analysis" / "whitebox_e2e_summary.csv").is_file()


def test_analyzer_whitebox_e2e_absent_tier_backward_compatible(tmp_path):
    root = _synthetic_root(tmp_path)  # no whitebox-e2e/ directory
    result = analyzer.analyze(root, _manifest_fixture(tmp_path),
                              replicates=200)
    assert result["status"] == "COMPLETE"
    assert result["whitebox_e2e_summary"] == []
    assert result["whitebox_e2e_seed_rows"] == []
    assert result["whitebox_e2e_not_evaluable"] == []
    assert result["whitebox_e2e_cells_ok"] == 0
    assert not (root / "analysis" / "whitebox_e2e_summary.csv").exists()
    report = (root / "analysis" / "report.md").read_text()
    assert "Adaptive end-to-end" not in report


def test_analyzer_whitebox_e2e_fail_closed_on_invalid_arm(tmp_path):
    for i, bad in enumerate((float("nan"), 1.5, -0.1, True)):
        sub = tmp_path / f"bad-{i}"
        sub.mkdir()
        root = _synthetic_root(sub)
        _e2e_tier(root)
        cell_path = root / "whitebox-e2e/split3/seed-11/tb-frozen-64" \
            / "cell.json"
        cell = json.loads(cell_path.read_text())
        cell["attacks"]["adaptive_e2e"]["snap"]["accuracy"] = bad
        cell_path.write_text(json.dumps(cell))
        with EvidenceLogger(str(cell_path)):  # re-seal so the failure
            pass                             # is the arm, not evidence
        with pytest.raises(ValueError, match="FAILED"):
            analyzer.analyze(root, _manifest_fixture(sub), replicates=200)
        report = (root / "analysis" / "report.md").read_text()
        assert "whitebox-e2e/split3/seed-11/tb-frozen-64" in report
        assert "invalid" in report


def test_analyzer_whitebox_e2e_fail_closed_on_tampered_evidence(tmp_path):
    root = _synthetic_root(tmp_path)
    _e2e_tier(root)
    events = root / "whitebox-e2e/split3/seed-11/pca-compress-64" \
        / "cell.json.evidence.jsonl"
    with events.open("a") as handle:
        handle.write('{"seq": 99, "kind": "injected"}\n')
    with pytest.raises(ValueError, match="FAILED"):
        analyzer.analyze(root, _manifest_fixture(tmp_path), replicates=200)
    report = (root / "analysis" / "report.md").read_text()
    assert "evidence" in report


def test_analyzer_whitebox_e2e_fail_closed_on_missing_cell(tmp_path):
    root = _synthetic_root(tmp_path)
    cells = _e2e_tier(root)
    # plan a cell that was never written
    cells.append({"path": "whitebox-e2e/split3/seed-11/tb-frozen-512",
                  "split": 3, "seed": 11, "method": "tb-frozen",
                  "dim": 512})
    (root / "whitebox-e2e" / "plan.json").write_text(json.dumps(
        {"schema": E2E_PLAN_SCHEMA, "tier": "whitebox-e2e",
         "cells": cells}))
    with pytest.raises(ValueError, match="FAILED"):
        analyzer.analyze(root, _manifest_fixture(tmp_path), replicates=200)
    report = (root / "analysis" / "report.md").read_text()
    assert "cell.json missing" in report
