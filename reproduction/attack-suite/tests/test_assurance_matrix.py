"""Assurance-matrix contracts (Phase 6).

Schema validation, evidence-ladder integrity (fixed enum only), the
no-aggregate-score invariant, fail-closed INCOMPLETE rows on missing
leakage/utility fields, deterministic output from the real pilot
analysis.json, and markdown rendering sanity.
"""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PILOT_ANALYSIS = ROOT / "results" / "pilot-defended-v1" / "analysis" / "analysis.json"
SWEEP_ANALYSIS = (
    ROOT / "results" / "bottleneck-sweep-v1" / "analysis" / "analysis.json"
)
SWEEP_SELECTION = (
    ROOT / "results" / "bottleneck-sweep-v1" / "analysis" / "selection.json"
)


def load_script(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "experiments" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


am = load_script("assurance_matrix")


def minimal_pilot(summary):
    return {
        "schema": "llm_attacker.defended_pilot_analysis.v1",
        "status": "COMPLETE",
        "tier": "development_pilot",
        "cells_ok": len(summary),
        "cells_failed": 0,
        "failures": [],
        "summary": summary,
    }


def full_cell(**overrides):
    cell = {
        "split": 3,
        "defense": "gaussian",
        "seeds": 3,
        "forward_minus_none_mean": 0.007,
        "forward_minus_none_ci_low": 0.002,
        "forward_minus_none_ci_high": 0.017,
        "forward_minus_none_status": "ESTIMATED",
        "ppl_delta_vs_none_mean": 0.346,
        "ppl_delta_vs_none_ci_low": -1.426,
        "ppl_delta_vs_none_ci_high": 1.874,
    }
    cell.update(overrides)
    return cell


def iter_keys(node):
    if isinstance(node, dict):
        for key, value in node.items():
            yield key
            yield from iter_keys(value)
    elif isinstance(node, list):
        for item in node:
            yield from iter_keys(item)


class TestSchema:
    def test_schema_and_shape(self):
        matrix = am.build_from_pilot(minimal_pilot([full_cell()]))
        assert matrix["schema"] == "llm_attacker.assurance_matrix.v1"
        assert set(matrix) == {"schema", "meta", "rows"}
        row = matrix["rows"][0]
        assert row["defense"] == "gaussian"
        assert row["access"]["split"] == 3
        assert row["status"] == "ESTIMATED"
        assert row["leakage"]["paired_effect"] == 0.007
        assert row["utility_cost"]["paired_effect"] == 0.346
        assert row["evidence_strength"] == "defended_pilot_paired"

    def test_rejects_unknown_source_schema(self):
        with pytest.raises(ValueError, match="schema"):
            am.build_from_pilot({"schema": "nope", "summary": []})


class TestEvidenceLadder:
    def test_ladder_is_fixed_enum_in_order(self):
        assert am.EVIDENCE_LADDER == [
            "fixture",
            "live_channel_engineering",
            "defended_pilot_paired",
            "strengthened_attacker_development_sweep",
            "frozen_crossed_bootstrap",
        ]
        assert set(am.EVIDENCE_LADDER_META) == set(am.EVIDENCE_LADDER)

    def test_meta_documents_used_levels(self):
        matrix = am.build_from_pilot(minimal_pilot([full_cell()]))
        levels = [e["level"] for e in matrix["meta"]["evidence_ladder"]]
        # Base levels are always documented; the enum-extension level appears
        # only when rows use it, so pilot-only builds reproduce v1 byte-for-byte.
        assert levels == [
            "fixture",
            "live_channel_engineering",
            "defended_pilot_paired",
            "frozen_crossed_bootstrap",
        ]
        assert all(e["meaning"] for e in matrix["meta"]["evidence_ladder"])
        assert set(matrix["meta"]["access_conditions"]) == {"split-training-wire"}

    def test_rows_only_carry_enum_values(self):
        matrix = am.build_from_pilot(minimal_pilot([full_cell()]))
        for row in matrix["rows"]:
            assert row["evidence_strength"] in am.EVIDENCE_LADDER


class TestNoAggregateScore:
    def test_no_forbidden_field_anywhere(self):
        matrix = am.build_from_pilot(minimal_pilot([full_cell(), full_cell(split=6)]))
        for key in iter_keys(matrix):
            lowered = str(key).lower()
            for forbidden in am.FORBIDDEN_FIELDS:
                assert forbidden not in lowered, key

    def test_builder_refuses_aggregate_score(self):
        with pytest.raises(ValueError, match="aggregate"):
            am._assert_no_forbidden_fields({"rows": [{"risk_score": 0.5}]})
        with pytest.raises(ValueError, match="aggregate"):
            am._assert_no_forbidden_fields({"overall_maturity_level": 2})


class TestFailClosed:
    @pytest.mark.parametrize("missing", [
        "forward_minus_none_mean",
        "forward_minus_none_ci_low",
        "ppl_delta_vs_none_mean",
        "ppl_delta_vs_none_ci_high",
    ])
    def test_missing_field_emits_incomplete_row(self, missing):
        cell = full_cell()
        del cell[missing]
        matrix = am.build_from_pilot(minimal_pilot([cell]))
        assert len(matrix["rows"]) == 1  # never dropped
        row = matrix["rows"][0]
        assert row["status"] == "INCOMPLETE"
        assert missing in row["missing_fields"]
        assert row["leakage"] is None
        assert row["utility_cost"] is None

    def test_missing_identity_is_hard_error(self):
        with pytest.raises(ValueError):
            am.build_from_pilot(minimal_pilot([{"split": 3}]))


class TestRealPilot:
    def test_deterministic_from_real_analysis(self, tmp_path):
        if not PILOT_ANALYSIS.exists():
            pytest.skip("pilot analysis.json not present")
        pilot = json.loads(PILOT_ANALYSIS.read_text())
        first = am.build_from_pilot(pilot)
        second = am.build_from_pilot(json.loads(json.dumps(pilot)))
        assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
        assert len(first["rows"]) == len(pilot["summary"])
        assert all(r["status"] == "ESTIMATED" for r in first["rows"])

    def test_cli_rebuild_matches_committed_artifact(self, tmp_path):
        if not PILOT_ANALYSIS.exists():
            pytest.skip("pilot analysis.json not present")
        out = tmp_path / "matrix.json"
        rc = am.main([
            "--pilot", str(PILOT_ANALYSIS),
            "--output", str(out),
        ])
        assert rc == 0
        committed = ROOT / "experiments" / "assurance_matrix_pilot_v1.json"
        if committed.exists():
            assert json.loads(out.read_text()) == json.loads(committed.read_text())


class TestMarkdown:
    def test_rendering_sanity(self):
        matrix = am.build_from_pilot(minimal_pilot([full_cell(), full_cell(split=6)]))
        md = am.render_markdown(matrix)
        assert "no universal risk score" in md
        assert "maturity model" in md
        assert "`defended_pilot_paired`" in md
        assert md.count("| gaussian |") == 2
        assert "| Defense | Access" in md

    def test_incomplete_row_rendered_not_dropped(self):
        cell = full_cell()
        del cell["ppl_delta_vs_none_mean"]
        md = am.render_markdown(am.build_from_pilot(minimal_pilot([cell])))
        assert "INCOMPLETE" in md


def sweep_cell(**overrides):
    cell = {
        "split": 3,
        "dim": 128,
        "method": "tb-frozen",
        "seeds": 3,
        "forward_accuracy_mean": 0.8,
        "sipit_accuracy_mean": 0.7,
        "defense_aware_forward_mean": 0.06,
        "defense_aware_sipit_mean": 0.03,
        "ppl_delta_vs_none_mean": 41.8,
        "ppl_delta_vs_none_ci_low": 35.0,
        "ppl_delta_vs_none_ci_high": 48.5,
        "ppl_delta_vs_none_status": "ESTIMATED",
        "ppl_ratio_mean": 3.9,
        "gate_1_5x": "FAIL",
        "gate_2x": "FAIL",
    }
    cell.update(overrides)
    return cell


def minimal_sweep(summary, whitebox_summary=None, not_evaluable=None):
    return {
        "schema": "llm_attacker.bottleneck_sweep_analysis.v1",
        "status": "COMPLETE",
        "tier": "development_sweep",
        "cells_ok": len(summary),
        "cells_failed": 0,
        "expected_cells": len(summary),
        "failures": [],
        "summary": summary,
        "whitebox_summary": whitebox_summary or [],
        "whitebox_not_evaluable": not_evaluable or [],
    }


def minimal_selection():
    return {
        "schema": "llm_attacker.operating_point_selection.v1",
        "analysis_status": "COMPLETE",
        "control_method": "pca-compress",
        "candidate_methods": ["tb-frozen", "tb-cotrain"],
        "gates": {"primary_ppl_ratio": 1.5, "secondary_ppl_ratio": 2.0,
                  "predefined": True, "retuning": "refused (no-retuning rule)"},
        "leakage_statistic": "strongest observed across measured arms",
        "eligibility_conditions": "both conditions required",
        "access_labels": {"surrogate_arm": "approximation", "whitebox_tier": "stronger access"},
        "v3_status": "UNFROZEN",
        "v3_rule": "this tool never freezes anything",
        "negative_splits": ["3"],
        "negative_statement": "selected NO operating point",
        "splits": {
            "3": {
                "outcome": "negative",
                "selected": None,
                "candidates": [
                    {"split": 3, "dim": 128, "method": "tb-frozen",
                     "eligible": False, "reasons": ["utility gate FAIL"]},
                ],
            },
        },
    }


class TestSweepSchema:
    def test_sweep_rows_shape(self):
        rows, source, sel = am.build_from_sweep(
            minimal_sweep([sweep_cell()]), minimal_selection())
        assert len(rows) == 1
        row = rows[0]
        assert row["defense"] == "tb-frozen"
        assert row["dim"] == 128
        assert row["access"]["id"] == "split-training-wire"
        assert row["access"]["split"] == 3
        assert row["status"] == "ESTIMATED"
        assert row["evidence_strength"] == "strengthened_attacker_development_sweep"
        assert row["leakage"]["metric"] == (
            "strongest_observed_recovery_across_measured_arms")
        assert row["leakage"]["observed"] == 0.8
        assert row["leakage"]["strongest_arm"] == "base forward"
        assert set(row["leakage"]["arms_considered"]) == {
            "base forward", "base SIPIT",
            "defense-aware surrogate forward", "defense-aware surrogate SIPIT",
        }
        assert row["utility_cost"]["gates"] == {
            "thresholds_predefined": True,
            "retuning": "refused (no-retuning rule)",
            "1.5x": "FAIL",
            "2.0x": "FAIL",
        }
        assert row["selection"]["outcome"] == "negative"
        assert row["selection"]["eligible"] is False
        assert source["kind"] == "bottleneck_sweep"
        assert sel["outcome"] == "NEGATIVE"
        assert sel["v3_status"] == "UNFROZEN"

    def test_control_row_role(self):
        rows, _, _ = am.build_from_sweep(
            minimal_sweep([sweep_cell(method="pca-compress")]),
            minimal_selection())
        assert rows[0]["selection"]["role"] == "control"
        assert "eligible" not in rows[0]["selection"]

    def test_rejects_unknown_schemas(self):
        with pytest.raises(ValueError, match="schema"):
            am.build_from_sweep({"schema": "nope"}, minimal_selection())
        with pytest.raises(ValueError, match="schema"):
            am.build_from_sweep(minimal_sweep([]), {"schema": "nope"})


class TestSweepWhitebox:
    def whitebox_cell(self, **overrides):
        cell = sweep_cell()
        cell.update({
            "tier": "white-box-aware (true encoder granted)",
            "wb_inverter_forward_mean": 0.81,
            "wb_inverter_sipit_mean": 0.48,
            "wb_opt_forward_mean": 0.82,
            "wb_opt_snap_mean": 0.02,
        })
        cell.update(overrides)
        return cell

    def test_whitebox_row_separate_access_condition(self):
        rows, _, _ = am.build_from_sweep(
            minimal_sweep([sweep_cell()], whitebox_summary=[self.whitebox_cell()]),
            minimal_selection())
        assert len(rows) == 2
        wb = rows[1]
        assert wb["access"]["id"] == "split-training-wire-whitebox"
        assert wb["access"]["id"] in am.ACCESS_CONDITIONS
        assert wb["status"] == "ESTIMATED"
        assert wb["leakage"]["metric"] == "strongest_observed_recovery_whitebox_arms"
        assert wb["leakage"]["observed"] == 0.82
        assert wb["leakage"]["strongest_arm"] == "white-box optimized forward"

    def test_not_evaluable_row_emitted_never_dropped(self):
        ne = [{"split": 3, "dim": 128, "method": "tb-cotrain",
               "reason": "final co-trained checkpoint matches no capture"}]
        rows, _, _ = am.build_from_sweep(
            minimal_sweep(
                [sweep_cell(), sweep_cell(method="tb-cotrain")],
                whitebox_summary=[self.whitebox_cell()],
                not_evaluable=ne),
            minimal_selection())
        ne_rows = [r for r in rows if r["status"] == "NOT_EVALUABLE"]
        assert len(ne_rows) == 1
        assert ne_rows[0]["defense"] == "tb-cotrain"
        assert ne_rows[0]["access"]["id"] == "split-training-wire-whitebox"
        assert ne_rows[0]["leakage"] is None
        assert "checkpoint" in ne_rows[0]["reason"]

    def test_missing_whitebox_arm_emits_incomplete(self):
        cell = self.whitebox_cell()
        del cell["wb_opt_forward_mean"]
        rows, _, _ = am.build_from_sweep(
            minimal_sweep([sweep_cell()], whitebox_summary=[cell]),
            minimal_selection())
        wb = rows[1]
        assert wb["status"] == "INCOMPLETE"
        assert "wb_opt_forward_mean" in wb["missing_fields"]
        assert wb["leakage"] is None


class TestSweepFailClosed:
    @pytest.mark.parametrize("missing", [
        "forward_accuracy_mean",
        "defense_aware_sipit_mean",
        "ppl_delta_vs_none_mean",
    ])
    def test_missing_field_emits_incomplete_row(self, missing):
        cell = sweep_cell()
        del cell[missing]
        rows, _, _ = am.build_from_sweep(
            minimal_sweep([cell]), minimal_selection())
        assert len(rows) == 1  # never dropped
        row = rows[0]
        assert row["status"] == "INCOMPLETE"
        assert missing in row["missing_fields"]
        assert row["leakage"] is None


class TestSweepCombinedMatrix:
    def combined(self):
        pilot = json.loads(PILOT_ANALYSIS.read_text())
        analysis = json.loads(SWEEP_ANALYSIS.read_text())
        selection = json.loads(SWEEP_SELECTION.read_text())
        matrix = am.build_from_pilot(pilot)
        return am.extend_with_sweep(matrix, analysis, selection)

    def test_deterministic_from_real_artifacts(self):
        if not (PILOT_ANALYSIS.exists() and SWEEP_ANALYSIS.exists()
                and SWEEP_SELECTION.exists()):
            pytest.skip("real artifacts not present")
        first = self.combined()
        second = self.combined()
        assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
        # 10 pilot rows + 18 base sweep rows + 12 evaluable + 6 NOT_EVALUABLE
        assert len(first["rows"]) == 46
        strengths = {r["evidence_strength"] for r in first["rows"]}
        assert strengths == {
            "defended_pilot_paired", "strengthened_attacker_development_sweep"}
        assert all(r["evidence_strength"] in am.EVIDENCE_LADDER
                   for r in first["rows"])
        ne = [r for r in first["rows"] if r["status"] == "NOT_EVALUABLE"]
        assert len(ne) == 6
        assert all(r["defense"] == "tb-cotrain" for r in ne)
        assert first["meta"]["sweep_selection"]["outcome"] == "NEGATIVE"
        assert first["meta"]["sweep_selection"]["v3_status"] == "UNFROZEN"

    def test_no_forbidden_field_anywhere(self):
        if not (PILOT_ANALYSIS.exists() and SWEEP_ANALYSIS.exists()
                and SWEEP_SELECTION.exists()):
            pytest.skip("real artifacts not present")
        matrix = self.combined()
        for key in iter_keys(matrix):
            lowered = str(key).lower()
            for forbidden in am.FORBIDDEN_FIELDS:
                assert forbidden not in lowered, key

    def test_cli_rebuild_matches_committed_v2_artifact(self, tmp_path):
        if not (PILOT_ANALYSIS.exists() and SWEEP_ANALYSIS.exists()
                and SWEEP_SELECTION.exists()):
            pytest.skip("real artifacts not present")
        out = tmp_path / "matrix.json"
        rc = am.main([
            "--pilot", str(PILOT_ANALYSIS),
            "--sweep-analysis", str(SWEEP_ANALYSIS),
            "--sweep-selection", str(SWEEP_SELECTION),
            "--output", str(out),
        ])
        assert rc == 0
        committed = ROOT / "experiments" / "assurance_matrix_v2.json"
        if committed.exists():
            assert json.loads(out.read_text()) == json.loads(committed.read_text())

    def test_cli_requires_sweep_args_together(self):
        with pytest.raises(SystemExit):
            am.main([
                "--pilot", str(PILOT_ANALYSIS),
                "--sweep-analysis", str(SWEEP_ANALYSIS),
                "--output", "unused.json",
            ])

    def test_extended_markdown_sanity(self):
        if not (PILOT_ANALYSIS.exists() and SWEEP_ANALYSIS.exists()
                and SWEEP_SELECTION.exists()):
            pytest.skip("real artifacts not present")
        md = am.render_markdown(self.combined())
        assert "| Defense | Dim | Access (split) |" in md
        assert "white-box (true checkpoint)" in md
        assert "NOT_EVALUABLE" in md
        assert "NEGATIVE" in md
        assert "UNFROZEN" in md
        assert "`defended_pilot_paired`" in md
        assert "`strengthened_attacker_development_sweep`" in md
