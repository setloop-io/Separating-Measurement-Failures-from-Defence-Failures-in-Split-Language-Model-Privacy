"""Tests for experiments/audit_attack_budgets.py."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments import audit_attack_budgets as audit  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def _work_fwd():
    return {
        "head_forward_calls": 32,
        "head_forward_sequences": 128000,
        "head_forward_token_positions": 2159616,
        "head_backward_calls": 0,
        "head_backward_token_positions": 0,
        "vocabulary_dot_products": 51463168,
    }


def _work_sipit(n_seq=32, steps=128):
    return {
        "head_forward_calls": 2 * n_seq * steps,
        "head_forward_sequences": n_seq * 2 * n_seq * steps,
        "head_forward_token_positions": 4325376,
        "head_backward_calls": n_seq * steps,
        "head_backward_token_positions": 2162688,
        "vocabulary_dot_products": 6587285504,
    }


def _access(cand=32, beam=4, steps=128, seqs=32):
    return {
        "query_budgets": {
            "forward_beam": {"candidate_count": cand, "beam_width": beam},
            "sipit": {"steps_per_token": steps},
            "scored_sequences": seqs,
        }
    }


def _cell(cand=32, beam=4, steps=128, seqs=32, sipit_work=None):
    return {
        "schema": "llm_attacker.bottleneck_sweep_cell.v1",
        "status": "COMPLETE",
        "access": _access(cand, beam, steps, seqs),
        "attacks": {
            "forward": {
                "n_sequences": seqs,
                "candidate_count": cand,
                "beam_width": beam,
                "work": _work_fwd(),
            },
            "sipit": {
                "n_sequences": seqs,
                "steps_per_token": steps,
                "work": sipit_work if sipit_work is not None
                else _work_sipit(seqs, steps),
            },
        },
    }


DECLARED = {"candidate_count": 32, "beam_width": 4,
            "steps_per_token": 128, "n_sequences": 32}


def _write_cell(tmp_path, rel, cell):
    d = tmp_path / rel
    d.mkdir(parents=True)
    (d / "cell.json").write_text(json.dumps(cell))
    return d / "cell.json"


def test_matching_cell_passes(tmp_path):
    p = _write_cell(tmp_path, "split3/seed-11/none", _cell())
    rec = audit.audit_cell(p, DECLARED, "t", tmp_path)
    assert rec["verdict"] == "PASS", rec["failures"]


def test_steps_shortfall_fails(tmp_path):
    cell = _cell(steps=64)  # recorded 64, declared 128
    p = _write_cell(tmp_path, "split3/seed-11/none", cell)
    rec = audit.audit_cell(p, DECLARED, "t", tmp_path)
    assert rec["verdict"] == "FAIL"
    assert any("steps_per_token" in f for f in rec["failures"])


def test_work_counter_mismatch_fails(tmp_path):
    bad = _work_sipit(32, 128)
    bad["head_backward_calls"] = 1000
    p = _write_cell(tmp_path, "split3/seed-11/none",
                    _cell(sipit_work=bad))
    rec = audit.audit_cell(p, DECLARED, "t", tmp_path)
    assert rec["verdict"] == "FAIL"
    assert any("head_backward_calls" in f for f in rec["failures"])


def test_forward_budget_mismatch_fails(tmp_path):
    cell = _cell(cand=8, beam=2)
    p = _write_cell(tmp_path, "split3/seed-11/none", cell)
    rec = audit.audit_cell(p, DECLARED, "t", tmp_path)
    assert rec["verdict"] == "FAIL"
    assert any("candidate_count" in f for f in rec["failures"])
    assert any("beam_width" in f for f in rec["failures"])


def test_not_evaluable_is_uncheckable(tmp_path):
    cell = {"schema": "llm_attacker.whitebox_sweep_cell.v1",
            "status": "NOT_EVALUABLE",
            "not_evaluable_reason": "encoder version mismatch"}
    p = _write_cell(tmp_path, "whitebox/split3/seed-11/tb-cotrain-64", cell)
    rec = audit.audit_cell(p, DECLARED, "t", tmp_path)
    assert rec["verdict"] == "UNCHECKABLE"
    assert rec["notes"]


def test_whitebox_steps_per_wire_mismatch_fails(tmp_path):
    cell = {
        "status": "COMPLETE",
        "access": _access(),
        "attacks": {
            "whitebox_optimization": {
                "true_encoder_granted": True,
                "steps_per_wire": 64,
                "forward": {"n_sequences": 32, "candidate_count": 32,
                            "beam_width": 4, "work": _work_fwd()},
            },
        },
    }
    p = _write_cell(tmp_path, "whitebox/split3/seed-11/tb-frozen-64", cell)
    rec = audit.audit_cell(p, DECLARED, "t", tmp_path)
    assert rec["verdict"] == "FAIL"
    assert any("steps_per_wire" in f for f in rec["failures"])


def test_missing_cell_fails_closed(tmp_path):
    (tmp_path / "plan.json").write_text(json.dumps({
        "schema": "llm_attacker.defended_pilot.v1",
        "budgets": {"forward_candidates": 8, "beam_width": 2,
                    "sipit_steps": 64, "attack_sequences": 32},
        "cells": [{"path": "split3/seed-11/none"}],
    }))
    cells, _ = audit.audit_tree(tmp_path, audit.DEFENDED_PILOT, "pilot")
    assert cells[0]["verdict"] == "FAIL"
    assert any("absent" in f for f in cells[0]["failures"])


def test_main_fail_closed_exit_code(tmp_path):
    (tmp_path / "plan.json").write_text(json.dumps({
        "schema": "llm_attacker.defended_pilot.v1",
        "budgets": {"forward_candidates": 8, "beam_width": 2,
                    "sipit_steps": 64, "attack_sequences": 32},
        "cells": [{"path": "split3/seed-11/none"}],
    }))
    rc = audit.main(["--pilot", str(tmp_path), "--sweep",
                     str(tmp_path / "absent")])
    assert rc == 1


@pytest.mark.skipif(
    not (REPO / "results" / "pilot-defended-v1").exists()
    or not (REPO / "results" / "bottleneck-sweep-v1").exists(),
    reason="results trees not present locally")
def test_real_trees_smoke(tmp_path):
    out = tmp_path / "report.json"
    rc = audit.main([
        "--pilot", str(REPO / "results" / "pilot-defended-v1"),
        "--sweep", str(REPO / "results" / "bottleneck-sweep-v1"),
        "--json-out", str(out),
    ])
    report = json.loads(out.read_text())
    assert report["schema"] == "llm_attacker.budget_audit.v1"
    total = sum(t["cells"] for t in report["trees"])
    accounted = sum(t["pass"] + t["fail"] + t["uncheckable"]
                    for t in report["trees"])
    assert total == accounted and total > 0
    assert rc == 0, [
        (c["cell_id"], c["failures"]) for t in report["trees"]
        for c in t["records"] if c["verdict"] == "FAIL"]
