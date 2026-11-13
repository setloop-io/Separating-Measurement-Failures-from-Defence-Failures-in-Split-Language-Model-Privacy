"""White-box-aware addendum tier contracts.

Torch-free layer: dry-run enumeration against a fixture sweep root
(PLANNED vs BLOCKED on checkpoint presence), tier separation (the base
matrix's matched-access cells are untouched), analyzer labeling + CI and
carried utility-gate columns, fail-closed on a missing true defense
checkpoint. Torch-gated layer (tiny CPU fixture from the base sweep's
--quick mode): end-to-end whitebox run and inverter determinism.
"""
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "experiments" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load_script("run_bottleneck_sweep")
whitebox = load_script("run_whitebox_sweep")
analyzer = load_script("analyze_bottleneck_sweep")

HAS_TORCH = importlib.util.find_spec("torch") is not None
needs_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch unavailable")


def _manifest_fixture(tmp_path):
    src = ROOT / "experiments" / "validation_corpus_manifest.json"
    path = tmp_path / "manifest.json"
    path.write_text(src.read_text())
    return path


# ---------------------------------------------------------------------------
# Torch-free: dry-run enumeration + tier separation + fail-closed planning


def _dry_run_root(tmp_path):
    out = tmp_path / "sweep"
    runner.main(["--output", str(out), "--dry-run"])
    return out


def test_dry_run_enumerates_all_defended_cells(tmp_path):
    root = _dry_run_root(tmp_path)
    assert whitebox.main(["--root", str(root), "--dry-run"]) == 0
    plan = json.loads((root / "whitebox" / "plan.json").read_text())
    assert plan["schema"] == "llm_attacker.whitebox_sweep.v1"
    assert len(plan["cells"]) == 54  # 60-cell matrix minus 6 controls
    assert all(c["base_status"] == "PLANNED" for c in plan["cells"])
    for c in plan["cells"]:
        cell = json.loads((root / c["path"] / "cell.json").read_text())
        assert cell["tier"] == "whitebox"
        assert cell["base_cell"] == c["base_cell"]
        if c["method"] == "tb-cotrain":
            # the co-trained encoder evolves during victim training while
            # only the FINAL checkpoint is saved: training-phase captures
            # can never be matched to their capture-time encoder
            assert cell["status"] == "NOT_EVALUABLE"
            assert "encoder" in cell["not_evaluable_reason"]
        else:
            assert cell["status"] == "BLOCKED"  # base dry-run: no ckpts
            assert not c["checkpoint_present"]


def test_dry_run_marks_cells_planned_when_checkpoint_present(tmp_path):
    root = _dry_run_root(tmp_path)
    base = root / "split3/seed-11/dim-64/tb-frozen"
    cell = json.loads((base / "cell.json").read_text())
    cell["status"] = "COMPLETE"
    (base / "cell.json").write_text(json.dumps(cell))
    ckpt = base / "captures" / "trained_bottleneck_encoder.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_bytes(b"fixture")
    whitebox.main(["--root", str(root), "--dry-run"])
    wb = json.loads((root / "whitebox/split3/seed-11/tb-frozen-64"
                     / "cell.json").read_text())
    assert wb["status"] == "PLANNED"
    assert wb["checkpoint_present"] is True
    other = json.loads((root / "whitebox/split3/seed-11/pca-compress-64"
                        / "cell.json").read_text())
    assert other["status"] == "BLOCKED"
    # a co-trained cell stays NOT_EVALUABLE even with a checkpoint on
    # disk: that checkpoint is the FINAL encoder, never the capture-time
    # one
    cotrain = json.loads((root / "whitebox/split3/seed-11/tb-cotrain-64"
                          / "cell.json").read_text())
    assert cotrain["status"] == "NOT_EVALUABLE"


def test_whitebox_access_is_strictly_stronger_labeled(tmp_path):
    base_access = {"schema": "llm_attacker.bottleneck_sweep_access.v1",
                   "matched_across_defenses": True}
    spec = whitebox.whitebox_access_spec(base_access, "base/cell",
                                         "base/cell/captures/ckpt.pt")
    assert spec["schema"] == "llm_attacker.whitebox_sweep_access.v1"
    assert spec["tier"] == "whitebox"
    ext = spec["whitebox_extension"]
    assert "ACTUAL trained defense checkpoint" in ext["granted"]
    assert "strictly stronger" in ext["strength"]
    # stronger-access measurement, never a "bound" claim
    assert "bound" not in json.dumps(spec).lower()
    # the base spec object is not mutated
    assert base_access["schema"].endswith("v1") \
        and "whitebox_extension" not in base_access


# ---------------------------------------------------------------------------
# Torch-gated fixture: one tiny CPU base sweep + whitebox pass per module


@pytest.fixture(scope="module")
def quick_root(tmp_path_factory):
    if not HAS_TORCH:
        pytest.skip("torch unavailable")
    out = tmp_path_factory.mktemp("wb") / "sweep"
    runner.main(["--output", str(out), "--quick", "--seeds", "11",
                 "--splits", "3", "--methods", "pca-compress", "tb-frozen",
                 "--dims", "4"])
    return out


def _attacks_snapshot(root):
    snap = {}
    for path in sorted(root.glob("whitebox/split*/seed-*/*/cell.json")):
        cell = json.loads(path.read_text())
        snap[str(path.relative_to(root))] = {
            arm: {k: v[k]["accuracy"] for k in ("forward", "sipit", "snap")
                  if isinstance(v.get(k), dict) and "accuracy" in v[k]}
            for arm, v in cell["attacks"].items()}
    return snap


@needs_torch
def test_quick_run_arms_and_tier_separation(quick_root, tmp_path):
    base_cells = sorted(quick_root.glob("split*/seed-*/**/cell.json"))
    before = {p: p.read_bytes() for p in base_cells}
    assert whitebox.main(["--root", str(quick_root), "--quick"]) == 0
    # matched-access identity of the base matrix is untouched
    after = {p: p.read_bytes() for p in base_cells}
    assert before == after
    wb_cells = sorted(quick_root.glob("whitebox/split*/seed-*/*/cell.json"))
    assert len(wb_cells) == 4
    for path in wb_cells:
        cell = json.loads(path.read_text())
        assert cell["status"] == "COMPLETE"
        assert cell["access"]["schema"] == \
            "llm_attacker.whitebox_sweep_access.v1"
        assert cell["access"]["whitebox_extension"]["granted"]
        inv = cell["attacks"]["whitebox_inverter"]
        opt = cell["attacks"]["whitebox_optimization"]
        for accuracy in (inv["forward"]["accuracy"],
                         inv["sipit"]["accuracy"],
                         opt["forward"]["accuracy"],
                         opt["snap"]["accuracy"]):
            assert 0.0 <= accuracy <= 1.0
        # utility carried verbatim from the base cell, never recomputed
        base = json.loads((quick_root / cell["base_cell"] / "cell.json")
                          .read_text())
        assert cell["utility_after"] == base["utility_after"]
        assert cell["utility_gate"] == base["utility_gate"]
        ckpt = quick_root / cell["defense_checkpoint"]["path"]
        assert ckpt.is_file()
        # complete cells carry a sealed, verifiable evidence journal
        from attacker.evidence import verify_evidence
        journal = path.parent / cell["evidence"]
        assert journal.is_file()
        ok, problems = verify_evidence(str(journal))
        assert ok, problems


@needs_torch
def test_inverter_determinism(quick_root, tmp_path):
    whitebox.main(["--root", str(quick_root), "--quick"])
    first = _attacks_snapshot(quick_root)
    clone = tmp_path / "clone"
    shutil.copytree(quick_root, clone)
    whitebox.main(["--root", str(clone), "--quick"])
    assert _attacks_snapshot(clone) == first


@needs_torch
def test_analyzer_labels_whitebox_rows(quick_root, tmp_path):
    whitebox.main(["--root", str(quick_root), "--quick"])
    result = analyzer.analyze(quick_root, _manifest_fixture(tmp_path),
                              replicates=200)
    assert result["status"] == "COMPLETE"
    assert result["whitebox_cells_ok"] == 4
    report = (quick_root / "analysis" / "report.md").read_text()
    assert "White-box-aware tier (true encoder granted)" in report
    assert "(white-box-aware)" in report
    # the white-box section is a stronger-access measurement, never a
    # "bound" claim about black-box recovery
    wb_section = report.split("White-box-aware tier")[1]
    wb_section = wb_section.split("## Failures")[0]
    assert "strictly stronger access measurement" in wb_section
    assert "bound" not in wb_section.lower()
    rows = result["whitebox_summary"]
    assert len(rows) == 4
    for row in rows:
        assert row["tier"] == "white-box-aware (true encoder granted)"
        for key in ("wb_inverter_forward_mean", "wb_opt_snap_mean",
                    "ppl_delta_vs_none_mean",
                    "ppl_delta_vs_none_ci_low",
                    "ppl_delta_vs_none_ci_high",
                    "none_minus_wb_inverter_forward_ci_low",
                    "gate_1_5x", "gate_2x", "ppl_ratio_mean"):
            assert key in row
    assert (quick_root / "analysis" / "whitebox_summary.csv").is_file()


@needs_torch
def test_fail_closed_on_missing_checkpoint(quick_root, tmp_path):
    clone = tmp_path / "broken"
    shutil.copytree(quick_root, clone)
    (clone / "split3/seed-11/dim-4/tb-frozen/captures"
     / "trained_bottleneck_encoder.pt").unlink()
    with pytest.raises(ValueError, match="FAILED"):
        whitebox.main(["--root", str(clone), "--quick"])
    cell = json.loads((clone / "whitebox/split3/seed-11/tb-frozen-4"
                       / "cell.json").read_text())
    assert cell["status"] == "FAILED"
    assert "checkpoint missing" in cell["failure_reason"]
    # reported, never dropped: the other cells still completed
    ok = [p for p in clone.glob("whitebox/split*/seed-*/*/cell.json")
          if json.loads(p.read_text())["status"] == "COMPLETE"]
    assert len(ok) == 3
    plan = json.loads((clone / "whitebox" / "plan.json").read_text())
    assert plan["status"] == "FAILED_CELLS"
    with pytest.raises(ValueError, match="FAILED"):
        analyzer.analyze(clone, _manifest_fixture(tmp_path),
                         replicates=200)
    report = (clone / "analysis" / "report.md").read_text()
    assert "whitebox/split3/seed-11/tb-frozen-4" in report
    assert "checkpoint missing" in report


@pytest.fixture(scope="module")
def quick_cotrain_root(tmp_path_factory):
    if not HAS_TORCH:
        pytest.skip("torch unavailable")
    out = tmp_path_factory.mktemp("wbc") / "sweep"
    runner.main(["--output", str(out), "--quick", "--seeds", "11",
                 "--splits", "3", "--methods", "tb-cotrain",
                 "--dims", "4"])
    return out


# ---------------------------------------------------------------------------
# Adaptive end-to-end tier (--adaptive-e2e): opt-in, separate whitebox-e2e
# tree, own access records, same fail-closed / NOT_EVALUABLE conventions


@needs_torch
def test_flag_off_produces_no_e2e_cells(quick_root):
    """Without --adaptive-e2e the runner must not create any
    whitebox-e2e output (behavior bit-identical to before)."""
    assert whitebox.main(["--root", str(quick_root), "--quick"]) == 0
    assert not (quick_root / "whitebox-e2e").exists()
    wb_cells = sorted(quick_root.glob("whitebox/split*/seed-*/*/cell.json"))
    assert len(wb_cells) == 4  # the whitebox tier is untouched


@pytest.fixture(scope="module")
def e2e_root(tmp_path_factory):
    if not HAS_TORCH:
        pytest.skip("torch unavailable")
    out = tmp_path_factory.mktemp("e2e") / "sweep"
    runner.main(["--output", str(out), "--quick", "--seeds", "11",
                 "--splits", "3", "--methods", "pca-compress", "tb-frozen",
                 "--dims", "4"])
    assert whitebox.main(["--root", str(out), "--quick",
                          "--adaptive-e2e"]) == 0
    return out


def _e2e_snapshot(root):
    snap = {}
    for path in sorted(root.glob("whitebox-e2e/split*/seed-*/*/cell.json")):
        cell = json.loads(path.read_text())
        snap[str(path.relative_to(root))] = {
            arm: {"accuracy": v["snap"]["accuracy"],
                  "mse": v["final_encoder_mse"]}
            for arm, v in cell["attacks"].items()}
    return snap


@needs_torch
def test_e2e_cells_complete_with_budget_counters(e2e_root):
    cells = sorted(e2e_root.glob("whitebox-e2e/split*/seed-*/*/cell.json"))
    assert len(cells) == 4
    budgets = json.loads((e2e_root / "plan.json").read_text())["budgets"]
    for path in cells:
        cell = json.loads(path.read_text())
        assert cell["schema"] == "llm_attacker.whitebox_sweep_e2e_cell.v1"
        assert cell["status"] == "COMPLETE"
        assert cell["tier"] == "whitebox-e2e"
        access = cell["access"]
        assert access["schema"] == \
            "llm_attacker.whitebox_sweep_e2e_access.v1"
        ext = access["whitebox_extension"]
        assert "strictly stronger" in ext["strength"]
        assert "budget_mapping" in ext
        assert "bound" not in json.dumps(access).lower()
        arm = cell["attacks"]["adaptive_e2e"]
        assert arm["true_encoder_granted"] and arm["adaptive"] is True
        # budget counter: SIPIT step count mapped to steps per wire
        assert arm["steps_per_wire"] == budgets["sipit_steps"]
        snap = arm["snap"]
        assert 0.0 <= snap["accuracy"] <= 1.0
        assert snap["work"]["vocabulary_dot_products"] > 0
        assert cell["wall_seconds"] > 0
        # utility carried verbatim from the base cell
        base = json.loads((e2e_root / cell["base_cell"] / "cell.json")
                          .read_text())
        assert cell["utility_after"] == base["utility_after"]
        assert cell["utility_gate"] == base["utility_gate"]
        # complete cells carry a sealed, verifiable evidence journal
        from attacker.evidence import verify_evidence
        journal = path.parent / cell["evidence"]
        assert journal.is_file()
        ok, problems = verify_evidence(str(journal))
        assert ok, problems


@needs_torch
def test_e2e_optimization_runs_the_client_head(e2e_root):
    """Regression: the adaptive arm must optimize its variable THROUGH
    the victim's client head before the true encoder.

    It once compared raw token embeddings against the released z, so the
    head never ran: the variable was dragged toward seam space while the
    nearest-embedding readout still read it as a token embedding. GPT-2's
    embedding width equals its hidden width, so nothing raised. Recorded
    head work is the observable proof the head is in the loop.
    """
    cells = sorted(e2e_root.glob("whitebox-e2e/split*/seed-*/*/cell.json"))
    assert len(cells) == 4
    steps = json.loads((e2e_root / "plan.json").read_text())[
        "budgets"]["sipit_steps"]
    for path in cells:
        arm = json.loads(path.read_text())["attacks"]["adaptive_e2e"]
        opt = arm["optimization"]
        n_seq, n_tokens = arm["snap"]["n_sequences"], arm["snap"]["n_tokens"]
        assert (opt["steps"], opt["n_sequences"]) == (steps, n_seq)
        work = opt["work"]
        # exactly one full-sequence head forward + backward per Adam step
        assert work["head_forward_calls"] == steps
        assert work["head_forward_sequences"] == steps * n_seq
        assert work["head_forward_token_positions"] == steps * n_tokens
        assert work["head_backward_calls"] == steps
        assert work["head_backward_token_positions"] == steps * n_tokens
        assert work["wall_seconds"] > 0
        # the readout stays a bare vocabulary snap: still no head calls
        assert arm["snap"]["work"]["head_forward_calls"] == 0
        # and the recorded metadata names the stage it now spends
        assert "head(x)" in arm["objective"]
        assert "head forward" in arm["budget_match"]
        mapping = json.loads(path.read_text())["access"][
            "whitebox_extension"]["budget_mapping"]
        assert "head forward and one head backward" in mapping


@needs_torch
def test_e2e_determinism(e2e_root, tmp_path):
    first = _e2e_snapshot(e2e_root)
    clone = tmp_path / "clone"
    shutil.copytree(e2e_root, clone)
    whitebox.main(["--root", str(clone), "--quick", "--adaptive-e2e"])
    assert _e2e_snapshot(clone) == first


@needs_torch
def test_e2e_fail_closed_on_missing_checkpoint(e2e_root, tmp_path):
    clone = tmp_path / "broken"
    shutil.copytree(e2e_root, clone)
    shutil.rmtree(clone / "whitebox-e2e")
    (clone / "split3/seed-11/dim-4/tb-frozen/captures"
     / "trained_bottleneck_encoder.pt").unlink()
    with pytest.raises(ValueError, match="FAILED"):
        whitebox.main(["--root", str(clone), "--quick", "--adaptive-e2e"])
    cell = json.loads((clone / "whitebox-e2e/split3/seed-11/tb-frozen-4"
                       / "cell.json").read_text())
    assert cell["status"] == "FAILED"
    assert "checkpoint missing" in cell["failure_reason"]
    # reported, never dropped: the other cells still completed
    ok = [p for p in clone.glob("whitebox-e2e/split*/seed-*/*/cell.json")
          if json.loads(p.read_text())["status"] == "COMPLETE"]
    assert len(ok) == 3
    plan = json.loads((clone / "whitebox-e2e" / "plan.json").read_text())
    assert plan["status"] == "FAILED_CELLS"


@needs_torch
def test_e2e_cotrain_cells_not_evaluable(quick_cotrain_root):
    """The encoder-version mismatch exclusion carries over to the
    adaptive e2e tier: tb-cotrain cells are NOT_EVALUABLE (never
    silently evaluated against the FINAL co-trained encoder)."""
    assert whitebox.main(["--root", str(quick_cotrain_root), "--quick",
                          "--adaptive-e2e"]) == 0
    cell = json.loads((quick_cotrain_root
                       / "whitebox-e2e/split3/seed-11/tb-cotrain-4"
                       / "cell.json").read_text())
    assert cell["status"] == "NOT_EVALUABLE"
    assert "attacks" not in cell
    assert "encoder" in cell["not_evaluable_reason"]
    plan = json.loads((quick_cotrain_root / "whitebox-e2e" / "plan.json")
                      .read_text())
    assert plan["status"] == "COMPLETE"  # planned exclusion, not failure
    assert any(e["method"] == "tb-cotrain"
               for e in plan["not_evaluable"])


@needs_torch
def test_cotrain_cells_not_evaluable_end_to_end(quick_cotrain_root,
                                                tmp_path):
    """Regression: co-trained white-box attacks previously loaded the
    FINAL encoder checkpoint and attacked training-phase captures made
    by EARLIER encoder versions. Those cells must now be recorded
    NOT_EVALUABLE (never silently evaluated), without failing the run."""
    assert whitebox.main(["--root", str(quick_cotrain_root),
                          "--quick"]) == 0
    cell = json.loads((quick_cotrain_root
                       / "whitebox/split3/seed-11/tb-cotrain-4"
                       / "cell.json").read_text())
    assert cell["status"] == "NOT_EVALUABLE"
    assert "attacks" not in cell  # never evaluated with a mismatch
    assert "encoder" in cell["not_evaluable_reason"]
    plan = json.loads((quick_cotrain_root / "whitebox" / "plan.json")
                      .read_text())
    assert plan["status"] == "COMPLETE"  # planned exclusion, not failure
    assert any(e["method"] == "tb-cotrain"
               for e in plan["not_evaluable"])
    # the analyzer carries the exclusion instead of failing or dropping
    result = analyzer.analyze(quick_cotrain_root,
                              _manifest_fixture(tmp_path),
                              replicates=200)
    assert result["status"] == "COMPLETE"
    na = result["whitebox_not_evaluable"]
    assert any(e["method"] == "tb-cotrain" and e["reason"]
               for e in na)
    report = (quick_cotrain_root / "analysis" / "report.md").read_text()
    assert "NOT_EVALUABLE" in report
