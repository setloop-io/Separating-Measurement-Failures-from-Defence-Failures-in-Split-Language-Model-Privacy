"""Small CPU tests of the new opt-in policy, independent of model execution."""
from dataclasses import asdict
import importlib.util
import json
import csv
from pathlib import Path
import random

import pytest

from attacker.clustered_assessment import (ClusterPolicy, assess, cluster_records,
    paired_cluster_test, paired_bootstrap, simulation_power)
from attacker.gate import DEFAULT_GATE


CONTRACT = {"model": "fixture", "corpus": "shared", "mode": "inference",
            "surface": "activation", "grant": "public", "budget": {"k": 32},
            "metric": "token_accuracy"}


def records(kind="candidate", seeds=(1, 2, 3), documents=30):
    out = []
    for seed in seeds:
        for d in range(documents):
            difference = (-1, 1)[d % 2]
            correct = {"candidate": 100+difference, "strong": 180+difference,
                       "weak": 100+difference, "below": 40+difference,
                       "zero": 0}[kind]
            out.append({"seed": seed, "document_id": str(d), "n_tokens": 200,
                        "correct": correct, "control_correct": 0 if kind == "zero" else 100})
    return out


def utility(seeds=(1, 2, 3), increase=0.01):
    return {"metric": "held-out CE", "pairs": [{"seed": s,
        "candidate_ce": 1+increase, "reference_ce": 1} for s in seeds]}


def assessment(candidate=None, reference=None, **overrides):
    return assess(candidate if candidate is not None else records(),
        reference_records=reference if reference is not None else records("strong"),
        protocol=CONTRACT, reference_protocol=overrides.pop("reference_protocol", CONTRACT),
        utility=overrides.pop("utility", utility()), **overrides)


def test_supported_case_needs_real_control_and_utility_without_legacy_changes():
    before = asdict(DEFAULT_GATE)
    result = assessment()
    assert result["verdict"] == "PASS_TESTED_CONTRACT"
    assert result["reference"]["lower_one_sided_pp"] > 1
    assert result["utility"]["status"] == "ELIGIBLE"
    assert result["retrospective"] and not result["legacy_gate_and_scorecard_changed"]
    assert asdict(DEFAULT_GATE) == before


@pytest.mark.parametrize("reference", [[], records("weak")])
def test_missing_or_weak_reference_cannot_back_acceptance(reference):
    result = assess(records(), reference_records=reference, protocol=CONTRACT,
                    reference_protocol=CONTRACT, utility=utility())
    assert result["verdict"] == "INCONCLUSIVE"
    assert result["reasons"]


def test_contract_mismatch_and_missing_contract_fields_fail_closed():
    result = assessment(reference_protocol=dict(CONTRACT, grant="exact"))
    assert result["verdict"] == "INCONCLUSIVE"
    assert any("protocol mismatch" in x for x in result["reasons"])
    result = assess(records(), reference_records=records("strong"), protocol={"model": "fixture"},
                    reference_protocol={"model": "fixture"}, utility=utility())
    assert result["verdict"] == "INCONCLUSIVE"


@pytest.mark.parametrize("change", ["seeds", "documents", "denominator", "control"])
def test_matching_labels_do_not_replace_actual_reference_alignment(change):
    reference = records("strong")
    if change == "seeds":
        reference = [dict(r, seed=r["seed"]+10) for r in reference]
    elif change == "documents":
        reference = [dict(r, document_id="other-"+r["document_id"]) for r in reference]
    elif change == "denominator":
        reference[0] = dict(reference[0], n_tokens=201)
    else:
        reference[0] = dict(reference[0], control_correct=99)
    result = assessment(reference=reference)
    assert result["reference"]["lower_one_sided_pp"] > 1  # strong, yet mismatched
    assert result["verdict"] == "INCONCLUSIVE"
    assert not result["reference_alignment"]["matched"]
    assert any("match" in reason for reason in result["reasons"])


def test_upper_margin_is_not_two_sided_equivalence():
    result = assessment(records("below"))
    assert result["verdict"] == "PASS_TESTED_CONTRACT"
    assert result["candidate"]["upper_margin"]["supported_below_margin"]
    assert not result["candidate"]["tost"]["equivalent"]
    assert result["candidate"]["tost"]["p_lower"] > 0.99


def test_zero_samples_do_not_create_zero_width_certainty():
    cl = cluster_records(records("zero"))
    result = paired_cluster_test(cl, margin_pp=1, alpha=0.05)
    bootstrap = paired_bootstrap(cl, replicates=100, seed=7)
    assert bootstrap["degenerate_percentile_distribution"]
    assert bootstrap["lower_pp"] == bootstrap["upper_pp"] == 0
    assert result["zero_variance_guard_used"]
    assert result["upper_one_sided_pp"] > 1
    assert not result["upper_margin"]["supported_below_margin"]
    assert not result["tost"]["equivalent"]


def test_same_document_ids_across_seeds_are_one_cluster_and_order_invariant():
    rows = records()
    shuffled = rows.copy(); random.Random(9).shuffle(shuffled)
    a, b = cluster_records(rows), cluster_records(shuffled)
    assert len(a["labels"]) == len(b["labels"]) == 30  # not90 independent documents
    assert a["tokens"].tolist() == [600]*30
    assert paired_cluster_test(a) == paired_cluster_test(b)


def test_duplicate_seed_document_is_rejected_instead_of_extra_replication():
    rows = records()
    with pytest.raises(ValueError, match="average attack seeds"):
        cluster_records(rows+[rows[0]])


def test_variable_cluster_sizes_preserve_pooled_paired_estimand():
    rows = [{"seed": 1, "document_id": "a", "n_tokens": 10, "correct": 9, "control_correct": 1},
            {"seed": 2, "document_id": "a", "n_tokens": 10, "correct": 8, "control_correct": 2},
            {"seed": 1, "document_id": "b", "n_tokens": 100, "correct": 50, "control_correct": 40}]
    result = paired_cluster_test(cluster_records(rows))
    assert result["mean_excess_pp"] == pytest.approx(100*(8+6+10)/120)
    assert result["clusters"] == 2


def test_serial_blocks_preserve_seed_and_step_and_cannot_authorise_acceptance():
    candidate, reference = records(), records("strong")
    for rows in (candidate, reference):
        for row in rows:
            row["step"] = int(row["document_id"])
    cl = cluster_records(candidate, serial=True, block_length=10)
    assert len(cl["labels"]) == 9
    result = assessment(candidate, reference, serial=True, block_length=10)
    assert result["verdict"] == "INCONCLUSIVE"
    assert any("evolving training" in s for s in result["reasons"])


@pytest.mark.parametrize("u", [None, utility(seeds=(1, 2))])
def test_missing_or_unmatched_utility_is_inconclusive(u):
    assert assessment(utility=u)["verdict"] == "INCONCLUSIVE"


def test_malformed_duplicate_and_overflow_utility_fail_safely():
    u = utility(); u["pairs"][1] = u["pairs"][0]
    assert assessment(utility=u)["verdict"] == "INCONCLUSIVE"
    assert assessment(utility={"metric": "CE", "pairs": [{}]})["verdict"] == "INCONCLUSIVE"
    result = assessment(utility=utility(increase=1000))
    assert result["verdict"] == "NON_VIABLE"
    assert result["utility"]["geometric_ppl_ratio"] is None
    assert result["utility"]["ppl_ratio_overflow"]


def test_utility_collapse_is_nonviable_even_if_recovery_is_low():
    result = assessment(utility=utility(increase=1))
    assert result["candidate"]["upper_margin"]["supported_below_margin"]
    assert result["verdict"] == "NON_VIABLE"


def test_seed_minimum_is_not_satisfied_by_many_documents_from_one_victim():
    result = assessment(records(seeds=(1,)), utility=utility(seeds=(1,)))
    assert result["verdict"] == "INCONCLUSIVE"
    assert any("independent seed" in s for s in result["reasons"])


def test_explicit_one_sided_tost_components_and_family_allocation():
    p = ClusterPolicy()
    result = paired_cluster_test(cluster_records(records()), alpha=p.test_alpha)
    assert p.test_alpha == 0.05/(2*40)
    assert result["tost"]["equivalent"]
    assert result["tost"]["p_tost"] == max(result["tost"]["p_lower"], result["tost"]["p_upper"])
    assert result["central_interval_level"] == 1-2*p.test_alpha


def test_planning_simulation_retains_zero_variance_guard_and_declared_effect():
    profile = {"id": "fixture", "incidence": [[1]*8]*3,
               "tokens_per_document": [32]*8, "baseline_rate": 0}
    sim = {"replicates_per_scenario": 100, "seed": 7, "document_beta_rho": [0],
           "seed_probability_sd": [0], "paired_coupling": [1], "requested_excess_pp": [0, 2]}
    rows = simulation_power(profile, simulation=sim, alpha=0.05)
    assert rows[0]["upper_margin_acceptance_probability"] == 0
    assert rows[0]["tost_equivalence_probability"] == 0
    assert rows[1]["achieved_expected_excess_pp"] == pytest.approx(2)
    assert rows[1]["boundary_clipping_fraction"] == 0


def test_domain_loader_selects_the_declared_defense_in_mixed_bundles(tmp_path):
    script = (Path(__file__).resolve().parents[1]/"artifacts/paper-b-review-round2-20261021"
              /"all-review-closure/cpu-closure-20261022/statistical/run_statistical.py")
    spec = importlib.util.spec_from_file_location("cpu_statistical_fixture", script)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    for corpus in ("code", "clinical", "wikitext-reference"):
        for suffix, defenses in (("", ["none"]), ("-defended", ["none", "gaussian"])):
            results = []
            for defense in defenses:
                for mode in ("training", "inference"):
                    for seed in (1, 2, 3):
                        results.append({"arm": "vmi", "mode": mode, "defense": defense, "seed": seed,
                            "per_sequence": [{"document_id": "shared", "n_tokens": 32,
                                "correct": 1 if defense == "none" else 2, "control_correct": 0}],
                            "utility_after": {"cross_entropy": 3}})
            path = tmp_path/f"results/domain-shift-v1/{corpus}{suffix}/experiment.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"status": "COMPLETE", "results": results}))
    cases = module.load_domain(module.Inputs(tmp_path))
    assert len(cases) == 12
    assert all(len(c["records"]) == 3 for c in cases)
    assert all(c["records"][0]["correct"] == (2 if c["id"].endswith("gaussian") else 1) for c in cases)
    assert all(not c["serial"] for c in cases)


def test_case_tables_include_samples_bounds_controls_utility_and_all_mde_rows(tmp_path):
    script = (Path(__file__).resolve().parents[1]/"artifacts/paper-b-review-round2-20261021"
              /"all-review-closure/cpu-closure-20261022/statistical/run_statistical.py")
    spec = importlib.util.spec_from_file_location("cpu_statistical_tables_fixture", script)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    result = assessment()
    result.update(id="fixture/shared", simulation_profile="design-fixture", observed={
        "victim_seed_identities": 3, "paired_records": 90, "token_observations": 18000,
        "unique_probe_identities": 30, "attack_accuracy_pct": 50, "control_accuracy_pct": 50})
    result["mde_sensitivity_rows"] = [{"profile": "design-fixture", "rho": 0,
        "seed_probability_sd": 0, "paired_coupling": 1,
        "minimum_achieved_excess_pp_detected_above_zero_on_grid": 2,
        "minimum_achieved_excess_pp_establishing_reference_competence_on_grid": None,
        "largest_achieved_excess_pp_accepted_below_margin_on_grid": 0}]
    module.write_tables(tmp_path, [result], [], interval_level=.99875)
    with (tmp_path/"case-assessments.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1 and rows[0]["victim_seed_identities"] == "3"
    assert rows[0]["reference_aligned"] == "True"
    assert rows[0]["competence_not_attained_scenarios"] == "1"
    with (tmp_path/"case-mde-scenarios.csv").open() as f:
        assert len(list(csv.DictReader(f))) == 1
    assert "fixture/shared" in (tmp_path/"case-assessments.tex").read_text()
    assert "2.00--2.00 (0/1 NA)" in (tmp_path/"case-mde.tex").read_text()


def test_horizon_loader_excludes_evidence_sidecars_by_full_filename(tmp_path):
    script = (Path(__file__).resolve().parents[1]/"artifacts/paper-b-review-round2-20261021"
              /"all-review-closure/cpu-closure-20261022/statistical/run_statistical.py")
    spec = importlib.util.spec_from_file_location("cpu_statistical_horizon_fixture", script)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    directory = tmp_path/"results/long-horizon-v1/matrix/artifacts"
    directory.mkdir(parents=True)
    (directory/"seam-inversion_vmi_none_s1_h20.json").write_text(json.dumps({"results": [
        {"step": 0, "per_sequence": [{"document_id": "d", "n_tokens": 32,
            "correct": 1, "control_correct": 0}]}]}))
    (directory/"splitbench_none_s1_h20.json").write_text(json.dumps({"results": [{"mean_loss": 3}]}))
    (directory/"seam-inversion_vmi_none_s1_h20.json.evidence.json").write_text("not a result payload")
    cases = module.load_horizon(module.Inputs(tmp_path))
    assert len(cases) == 1
    assert cases[0]["id"] == "horizon/none/20"
    assert len(cases[0]["records"]) == 1
