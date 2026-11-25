"""Defended-pilot runner/analyzer contracts (Phase 2).

Torch-free layer (runs everywhere): dry-run matrix structure with the
required per-cell files, matched attacker-access spec equality across
defenses (and analyzer rejection of drift), frozen-corpus disjointness
firing on a deliberately contaminated manifest fixture, the pilot-corpus
identity check, analyzer fail-closed on a missing-provenance cell, and
the failures-and-nulls section in report.md.

Torch layer (slow, self-skips without torch/transformers): end-to-end
--quick CPU smoke — full 30-cell matrix with tiny random GPT-2 and
synthetic documents, then the analyzer to COMPLETE.
"""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "experiments" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load_script("run_defended_pilot")
analyzer = load_script("analyze_defended_pilot")

EXPECTED_CELLS = [(s, seed, d)
                  for s in (3, 6) for seed in (11, 22, 33)
                  for d in ("none", "gaussian", "trained-bottleneck",
                            "secret-mixup", "replica-median")]


def _manifest_fixture(tmp_path, extra_eval_ids=()):
    manifest = json.loads(
        (ROOT / "experiments" / "validation_corpus_manifest.json").read_text())
    manifest["selected_ids"]["evaluation"] = list(
        manifest["selected_ids"]["evaluation"]) + list(extra_eval_ids)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def _corpus_fixture(tmp_path, ids=("wikitext2-train-aaaa", "wikitext2-train-bbbb")):
    path = tmp_path / "corpus.jsonl"
    path.write_text("".join(json.dumps({"id": i, "text": f"text of {i}"})
                            + "\n" for i in ids))
    return path


def test_dry_run_full_matrix_structure(tmp_path):
    out = tmp_path / "pilot"
    assert runner.main(["--output", str(out), "--dry-run"]) == 0
    plan = json.loads((out / "plan.json").read_text())
    assert plan["expected_cells"] == 30 and len(plan["cells"]) == 30
    assert plan["tier"] == "development_pilot"
    for split, seed, defense in EXPECTED_CELLS:
        cell_dir = out / f"split{split}/seed-{seed}/{defense}"
        cell = json.loads((cell_dir / "cell.json").read_text())
        assert cell["status"] == "PLANNED"
        assert cell["access"]["schema"] == \
            "llm_attacker.defended_pilot_access.v1"
        if defense != "none":
            assert cell["paired_control"] == "../none/cell.json"
    assert (out / "access.json").is_file()


def test_dry_run_matched_access_identical_across_defenses(tmp_path):
    out = tmp_path / "pilot"
    runner.main(["--output", str(out), "--dry-run"])
    rendered = {json.dumps(spec, sort_keys=True) for spec in
                [json.loads((out / c["path"] / "cell.json").read_text())["access"]
                 for c in json.loads((out / "plan.json").read_text())["cells"]]}
    assert len(rendered) == 1


def test_disjointness_fires_on_contaminated_manifest(tmp_path):
    corpus = _corpus_fixture(tmp_path)
    contaminated = _manifest_fixture(tmp_path,
                                     extra_eval_ids=["wikitext2-train-aaaa"])
    with pytest.raises(ValueError, match="overlaps the frozen"):
        runner.assert_development_corpus_disjoint(corpus, contaminated,
                                                  require_excluded_pilot=False)


def test_disjointness_requires_excluded_pilot_identity(tmp_path):
    corpus = _corpus_fixture(tmp_path)  # not the pilot corpus
    manifest = _manifest_fixture(tmp_path)
    with pytest.raises(ValueError, match="not the excluded pilot corpus"):
        runner.assert_development_corpus_disjoint(corpus, manifest,
                                                  require_excluded_pilot=True)
    # ids-only check passes for a disjoint non-pilot corpus
    rec = runner.assert_development_corpus_disjoint(
        corpus, manifest, require_excluded_pilot=False)
    assert rec["article_ids_disjoint_from_frozen"] is True


def test_real_pilot_corpus_is_disjoint_from_frozen_manifest():
    corpus = ROOT / runner.DEFAULT_CORPUS
    if not corpus.is_file():
        pytest.skip("pilot inputs not present on this host")
    rec = runner.assert_development_corpus_disjoint(
        corpus, ROOT / "experiments" / "validation_corpus_manifest.json")
    assert rec["identity_checked"] is True and rec["articles"] == 96


def _complete_cell(cell_dir, split, seed, defense, access, *,
                   drop_provenance_file=False, drop_provenance_key=False):
    cell_dir.mkdir(parents=True)
    prov = cell_dir / "prov.json"
    prov.write_text(json.dumps({"cell": defense}))
    cell = {
        "schema": runner.CELL_SCHEMA, "status": "COMPLETE",
        "measurement_version": 2, "observation_location": "body_input_seam",
        "split": split, "seed": seed, "defense": defense,
        "wall_seconds": 1.0, "attack_seconds": 0.5,
        "cuda_peak_allocated_bytes": None,
        "utility_before": {"cross_entropy": 4.0, "perplexity": 50.0},
        "utility_after": {"cross_entropy": 3.0, "perplexity": 20.0
                          if defense == "none" else 22.0},
        "access": access, "run_artifact": "run.json",
        "provenance": None if drop_provenance_key else {
            "path": "prov.json",
            "sha256": __import__("hashlib").sha256(
                prov.read_bytes()).hexdigest()},
        "attacks": {"forward": {"accuracy": 0.8 if defense == "none" else 0.5},
                    "sipit": {"accuracy": 0.7 if defense == "none" else 0.4}},
    }
    for stage in ('before','after'):
        cell['utility_' + stage].update(evaluation_contract='matched-final-state-v2',
                                       partition='utility_eval', stage=stage+'_victim_training')
    if defense == "trained-bottleneck":
        cell["attacks"]["representation-privacy"] = {"arms": {}}
    if defense == "secret-mixup":
        cell["attacks"]["view-privacy"] = {"rows": []}
    if defense == "replica-median":
        cell["attacks"]["replica-integrity"] = {"arms": []}
    (cell_dir / "cell.json").write_text(json.dumps(cell))
    if drop_provenance_file:
        prov.unlink()


def _synthetic_root(tmp_path, drop_provenance_file=False,
                    tamper_access=False):
    root = tmp_path / "pilot"
    root.mkdir()
    access = {"schema": "llm_attacker.defended_pilot_access.v1",
              "matched_across_defenses": True}
    corpus = _corpus_fixture(root)
    cells = []
    for split, seed, defense in [(3, 11, d) for d in
                                 ("none", "gaussian", "trained-bottleneck",
                                  "secret-mixup", "replica-median")]:
        rel = f"split{split}/seed-{seed}/{defense}"
        a = dict(access)
        if tamper_access and defense == "gaussian":
            a["victim_final_head_granted"] = True
        _complete_cell(root / rel, split, seed, defense, a,
                       drop_provenance_file=(drop_provenance_file
                                             and defense == "gaussian"))
        cells.append({"path": rel, "split": split, "seed": seed,
                      "defense": defense})
    plan = {"schema": runner.SCHEMA, "tier": "engineering_smoke",
            "defenses": ["none", "gaussian", "trained-bottleneck",
                         "secret-mixup", "replica-median"],
            "splits": [3], "seeds": [11], "expected_cells": len(cells),
            "corpus": {"path": str(corpus)}, "cells": cells, "limits": []}
    (root / "plan.json").write_text(json.dumps(plan))
    return root


def test_analyzer_fail_closed_on_missing_provenance(tmp_path):
    root = _synthetic_root(tmp_path, drop_provenance_file=True)
    with pytest.raises(ValueError, match="FAILED"):
        analyzer.analyze(root, _manifest_fixture(tmp_path))
    report = (root / "analysis" / "report.md").read_text()
    assert "## Failures and nulls" in report
    assert "provenance" in report
    assert "split3/seed-11/gaussian" in report


def test_analyzer_rejects_unmatched_access(tmp_path):
    root = _synthetic_root(tmp_path, tamper_access=True)
    with pytest.raises(ValueError, match="not matched"):
        analyzer.analyze(root, _manifest_fixture(tmp_path))


def test_analyzer_complete_fixture_has_failures_section(tmp_path):
    root = _synthetic_root(tmp_path)
    result = analyzer.analyze(root, _manifest_fixture(tmp_path),
                              replicates=200)
    assert result["status"] == "COMPLETE"
    report = (root / "analysis" / "report.md").read_text()
    assert "## Failures and nulls" in report
    assert "None dropped" in report
    for name in ("summary.csv", "seed_metrics.csv", "utility.csv",
                 "failures.csv", "analysis.json"):
        assert (root / "analysis" / name).is_file()
    summary = (root / "analysis" / "summary.csv").read_text()
    assert "forward_minus_none_mean" in summary


def test_percentile_bootstrap_thresholds():
    assert analyzer.percentile_bootstrap([1.0, 2.0])["status"] == \
        "INSUFFICIENT_EVIDENCE"
    ci = analyzer.percentile_bootstrap([1.0, 2.0, 3.0], replicates=500)
    assert ci["status"] == "ESTIMATED"
    assert ci["lower"] <= ci["mean"] <= ci["upper"]
    assert analyzer.percentile_bootstrap([])["status"] == "MISSING"


# ---------------------------------------------------------------------------
# Torch layer: end-to-end --quick CPU smoke (full matrix, tiny model)


@pytest.mark.slow
def test_quick_end_to_end_cpu(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("numpy")
    out = tmp_path / "pilot-defended-quick"
    assert runner.main(["--output", str(out), "--quick"]) == 0
    plan = json.loads((out / "plan.json").read_text())
    assert plan["status"] == "COMPLETE" and plan["expected_cells"] == 30
    for split, seed, defense in EXPECTED_CELLS:
        cell_dir = out / f"split{split}/seed-{seed}/{defense}"
        cell = json.loads((cell_dir / "cell.json").read_text())
        assert cell["status"] == "COMPLETE"
        for name in ("run.json", "run.json.evidence.json"):
            assert (cell_dir / name).is_file(), name
        assert (cell_dir / "captures" / "canonical.pt").is_file()
        assert cell["wall_seconds"] > 0 and cell["attack_seconds"] > 0
        assert "cuda_peak_allocated_bytes" in cell
        assert cell["utility_after"]["perplexity"] > 1.0
        assert 0.0 <= cell["attacks"]["forward"]["accuracy"] <= 1.0
        assert 0.0 <= cell["attacks"]["sipit"]["accuracy"] <= 1.0
        prov = cell_dir / cell["provenance"]["path"]
        assert prov.is_file()
    for split, seed in ((3, 11), (6, 33)):
        d = out / f"split{split}/seed-{seed}"
        assert (d / "trained-bottleneck" / "representation-privacy.json").is_file()
        assert (d / "secret-mixup" / "view-privacy.json").is_file()
        rep = json.loads((d / "replica-median"
                          / "replica-integrity.json").read_text())
        assert {s["arm"] for s in rep["summary"]} >= \
            {"clean", "one-malicious", "two-colluding"}
    result = analyzer.analyze(out, ROOT / "experiments"
                              / "validation_corpus_manifest.json",
                              replicates=200)
    assert result["status"] == "COMPLETE" and result["cells_ok"] == 30
    assert "## Failures and nulls" in (out / "analysis"
                                       / "report.md").read_text()
