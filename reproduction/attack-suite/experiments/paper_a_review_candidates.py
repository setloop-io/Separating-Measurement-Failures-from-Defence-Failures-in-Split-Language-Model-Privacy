"""Descriptive candidate-list audit of the frozen Paper A primary predictions.

Pure stdlib. No model, attack, inference, tokenizer or network operation occurs.
All 16 retained evidence chains and all 40 primary prediction exports must pass
before truth-side scoring. Identical attack-seed exports are counted once.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys


ROOT = Path(__file__).resolve().parents[1]
INPUT_SHA256 = "d79ef4fc3b27dbac21fd731f80873a5afc38481bc7a03acb0ee9fec1587bc21a"
VICTIM_SEEDS = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]
ATTACK_SEEDS = [17, 29]
ARMS = ["forward_k32", "joint_true_k32"]
MEMBERSHIPS = ["both", "forward_only", "joint_only", "neither"]
OUTCOMES = ["both_correct", "joint_rescue", "joint_harm", "both_wrong"]
SCOPES = ["all", "position_0", "positions_1_to_31"]
VOCABULARY = 50257


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def cell_id(victim, attack, arm):
    return f"{victim}-6-public-{attack}-training-{arm}"


def empty_counts():
    fields = ["positions", "forward_truth_in_candidates", "joint_truth_in_candidates",
        "forward_prediction_correct", "joint_prediction_correct",
        "forward_truth_available_but_not_selected", "joint_truth_available_but_not_selected"]
    fields.extend("membership_" + category for category in MEMBERSHIPS)
    fields.extend("outcome_" + category for category in OUTCOMES)
    fields.extend(f"{outcome}__{membership}" for outcome in OUTCOMES for membership in MEMBERSHIPS)
    return Counter({field: 0 for field in fields})


def observe(truth, forward_prediction, joint_prediction, forward_candidates, joint_candidates):
    """One position's observable membership and correctness, with no stream attribution."""
    forward_set, joint_set = set(forward_candidates), set(joint_candidates)
    if forward_prediction not in forward_set or joint_prediction not in joint_set:
        raise ValueError("A selected prediction is absent from its own candidate list")
    fm, jm = truth in forward_set, truth in joint_set
    fc, jc = forward_prediction == truth, joint_prediction == truth
    membership = ("both" if fm and jm else "forward_only" if fm
                  else "joint_only" if jm else "neither")
    outcome = ("both_correct" if fc and jc else "joint_rescue" if jc
               else "joint_harm" if fc else "both_wrong")
    value = empty_counts()
    value.update({"positions": 1, "forward_truth_in_candidates": int(fm),
        "joint_truth_in_candidates": int(jm), "forward_prediction_correct": int(fc),
        "joint_prediction_correct": int(jc),
        "forward_truth_available_but_not_selected": int(fm and not fc),
        "joint_truth_available_but_not_selected": int(jm and not jc),
        "membership_" + membership: 1, "outcome_" + outcome: 1,
        f"{outcome}__{membership}": 1})
    return value


def validate_accounting(counts):
    n = counts["positions"]
    if any(type(value) is not int or value < 0 for value in counts.values()):
        raise ValueError("Candidate accounting contains invalid counts")
    checks = [
        sum(counts["membership_" + m] for m in MEMBERSHIPS) == n,
        sum(counts["outcome_" + o] for o in OUTCOMES) == n,
        counts["forward_truth_in_candidates"] == counts["membership_both"] + counts["membership_forward_only"],
        counts["joint_truth_in_candidates"] == counts["membership_both"] + counts["membership_joint_only"],
        counts["forward_prediction_correct"] == counts["outcome_both_correct"] + counts["outcome_joint_harm"],
        counts["joint_prediction_correct"] == counts["outcome_both_correct"] + counts["outcome_joint_rescue"],
        counts["forward_truth_available_but_not_selected"] + counts["forward_prediction_correct"] == counts["forward_truth_in_candidates"],
        counts["joint_truth_available_but_not_selected"] + counts["joint_prediction_correct"] == counts["joint_truth_in_candidates"],
        counts["joint_prediction_correct"] - counts["forward_prediction_correct"] == counts["outcome_joint_rescue"] - counts["outcome_joint_harm"],
    ]
    for membership in MEMBERSHIPS:
        checks.append(sum(counts[f"{outcome}__{membership}"] for outcome in OUTCOMES)
                      == counts["membership_" + membership])
    for outcome in OUTCOMES:
        checks.append(sum(counts[f"{outcome}__{membership}"] for membership in MEMBERSHIPS)
                      == counts["outcome_" + outcome])
    # Correct output requires truth to have been available to that decoder.
    for membership in MEMBERSHIPS:
        if membership != "both":
            checks.append(counts["both_correct__" + membership] == 0)
        if membership not in ("both", "joint_only"):
            checks.append(counts["joint_rescue__" + membership] == 0)
        if membership not in ("both", "forward_only"):
            checks.append(counts["joint_harm__" + membership] == 0)
    if not all(checks):
        raise ValueError("Candidate membership/correctness accounting failed")


def row(counts, **identity):
    validate_accounting(counts)
    n = counts["positions"]
    if n <= 0:
        raise ValueError("Empty candidate diagnostic stratum")
    result = {**identity, **dict(counts)}
    for arm in ("forward", "joint"):
        result[arm + "_candidate_truth_recall_percent"] = 100 * counts[arm + "_truth_in_candidates"] / n
        result[arm + "_prediction_accuracy_percent"] = 100 * counts[arm + "_prediction_correct"] / n
        result[arm + "_truth_available_but_not_selected_percent_of_positions"] = (
            100 * counts[arm + "_truth_available_but_not_selected"] / n)
        errors = n - counts[arm + "_prediction_correct"]
        result[arm + "_prediction_errors"] = errors
        result[arm + "_truth_absent_from_candidates_errors"] = n - counts[arm + "_truth_in_candidates"]
        result[arm + "_truth_available_but_not_selected_percent_of_errors"] = (
            100 * counts[arm + "_truth_available_but_not_selected"] / errors if errors else None)
    result["joint_rescue_percent_of_positions"] = 100 * counts["outcome_joint_rescue"] / n
    result["joint_harm_percent_of_positions"] = 100 * counts["outcome_joint_harm"] / n
    result["joint_minus_forward_prediction_accuracy_pp"] = (
        100 * (counts["joint_prediction_correct"] - counts["forward_prediction_correct"]) / n)
    return result


def validate_prediction(payload):
    ids, predictions = payload["document_ids"], payload["predictions"]
    if (len(ids) != 256 or len(set(ids)) != 256
            or any(type(identifier) is not str or not identifier for identifier in ids)
            or len(predictions) != 256):
        raise ValueError("Expected 256 distinct ordered document IDs/prediction rows")
    if any(len(tokens) != 32 or any(type(token) is not int or not 0 <= token < VOCABULARY
                                  for token in tokens) for tokens in predictions):
        raise ValueError("Invalid 256-by-32 prediction token matrix")
    batches = payload["diagnostics"]["batches"]
    if len(batches) != 32:
        raise ValueError("Expected 32 candidate-diagnostic batches of eight documents")
    candidates = []
    for batch in batches:
        members, residuals = batch["candidate_ids"], batch["forward_residual_sum"]
        if len(members) != 8 or len(residuals) != 8:
            raise ValueError("Candidate/residual batch document count differs")
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) or value < 0 for value in residuals):
            raise ValueError("Invalid diagnostic forward residual")
        for document in members:
            if len(document) != 32:
                raise ValueError("Candidate diagnostic passage length differs")
            for position in document:
                if (len(position) != 32 or len(set(position)) != 32
                        or any(type(token) is not int or not 0 <= token < VOCABULARY for token in position)):
                    raise ValueError("Candidate list requires exactly 32 distinct vocabulary IDs")
            candidates.append(document)
    if any(token not in candidates[d][p] for d, tokens in enumerate(predictions)
           for p, token in enumerate(tokens)):
        raise ValueError("Prediction token absent from its exported candidate list")
    return candidates


def verify_sources(results, input_path, evidence):
    """Bind retained exports without opening the truth input yet."""
    inventory, selected, seen_cells = [], {}, set()
    common_protocol = None

    def record(path, role, expected=None):
        actual = digest(path)
        if expected is not None and actual != expected:
            raise ValueError("Source SHA256 mismatch: " + str(path))
        inventory.append({"path": str(path.resolve()), "role": role, "sha256": actual,
                          "expected_sha256": expected, "bytes": path.stat().st_size})
        return actual

    record(input_path, "frozen_truth_input_not_yet_read", INPUT_SHA256)
    for shard_index in range(16):
        folder = results / f"validation-heldout-v3-r3-shard{shard_index}"
        report_path = folder / "validation.json"
        seal_path = Path(str(report_path) + ".evidence.json")
        events_path = Path(str(report_path) + ".evidence.jsonl")
        ok, problems = evidence.verify_evidence(seal_path)
        if not ok:
            raise ValueError(f"Original source chain failed for shard {shard_index}: {problems}")
        seal, report = json.loads(seal_path.read_text()), json.loads(report_path.read_text())
        record(seal_path, "verified_original_evidence_manifest")
        record(events_path, "verified_original_hash_chained_journal")
        record(report_path, "verified_original_report", seal["artifact"]["sha256"])
        if (report["status"] != "COMPLETE" or report["input_sha256"] != INPUT_SHA256
                or report.get("supplied_input_sha256") != INPUT_SHA256
                or report.get("evidence_tier") != "held_out_controlled_probe"
                or report["shard"]["shard_index"] != shard_index
                or report["shard"]["shard_count"] != 16):
            raise ValueError("Original report completeness/input/shard identity differs")
        protocol = report["protocol"]
        if (protocol["model"] != "openai-community/gpt2" or protocol["victim_seeds"] != VICTIM_SEEDS
                or protocol["attack_seeds"] != ATTACK_SEEDS or protocol["sequence_length"] != 32
                or protocol["beam_width"] != 2 or protocol["attack_batch_size"] != 8
                or protocol["frozen_inputs"]["tokenized_sha256"] != INPUT_SHA256
                or 6 not in protocol["depths"] or "public" not in protocol["knowledge"]):
            raise ValueError("Original embedded protocol differs from the primary setting")
        if common_protocol is None:
            common_protocol = protocol
        elif common_protocol != protocol:
            raise ValueError("Original shards disagree on their embedded protocol")
        effective = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
        if effective != report["effective_protocol_sha256"]:
            raise ValueError("Original effective protocol digest differs")
        events = [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]
        journal = {}
        for event in events:
            if event["kind"] in ("failure", "error", "cell_failed", "defended_block_failed"):
                raise ValueError("Original complete report contains a failure event")
            if event["kind"] == "cell_complete":
                name = event["payload"]["cell_id"]
                if name in journal:
                    raise ValueError("Duplicate original cell-completion event")
                journal[name] = event["payload"]
        report_ids = [cell["cell_id"] for cell in report["cells"]]
        if (len(report_ids) != len(set(report_ids)) or set(report_ids) & seen_cells
                or sorted(report_ids) != report["shard"]["cell_ids"]):
            raise ValueError("Original full shard cell coverage/uniqueness differs")
        seen_cells.update(report_ids)
        for cell in report["cells"]:
            if not (cell["mode"] == "training" and cell["depth"] == 6
                    and cell["knowledge"] == "public" and cell["arm"] in ARMS):
                continue
            key = cell["victim_seed"], cell["attack_seed"], cell["arm"]
            if (key in selected or key[0] not in VICTIM_SEEDS or key[1] not in ATTACK_SEEDS
                    or cell["cell_id"] != cell_id(*key) or cell["execution_id"] != cell["cell_id"]
                    or cell["measurement_reused"] is not False
                    or cell.get("status") not in (None, "COMPLETE") or len(cell["metrics"]) != 256):
                raise ValueError("Original primary cell identity/reuse/status differs")
            logged = journal.get(cell["cell_id"])
            if logged != {field: value for field, value in cell.items() if field != "metrics"}:
                raise ValueError("Primary record is not bound to its completion journal")
            path = folder / cell["prediction_file"]
            if path.parent.resolve() != folder.resolve():
                raise ValueError("Prediction path leaves its original shard")
            record(path, "journal_bound_primary_prediction", cell["prediction_sha256"])
            payload = json.loads(path.read_text())
            candidates = validate_prediction(payload)
            selected[key] = {"path": path, "sha256": cell["prediction_sha256"],
                "payload": payload, "candidates": candidates, "cell": cell}
    expected = {(victim, attack, arm) for victim in VICTIM_SEEDS for attack in ATTACK_SEEDS for arm in ARMS}
    if set(selected) != expected:
        raise ValueError("Exactly forty original primary prediction cells required")
    deduplication = []
    ids = None
    for victim in VICTIM_SEEDS:
        for arm in ARMS:
            first, duplicate = [selected[victim, attack, arm] for attack in ATTACK_SEEDS]
            if first["sha256"] != duplicate["sha256"] or first["path"].read_bytes() != duplicate["path"].read_bytes():
                raise ValueError("Deterministic attack-seed exports are not byte-identical")
            if ids is None:
                ids = first["payload"]["document_ids"]
            elif ids != first["payload"]["document_ids"]:
                raise ValueError("Original primary documents are not paired across all victims/arms")
            deduplication.append({"victim_seed": victim, "arm": arm,
                "counted_attack_seed": 17, "duplicate_attack_seed": 29,
                "prediction_sha256": first["sha256"], "byte_identical": True})
        forward, joint = [selected[victim, 17, arm] for arm in ARMS]
        if any(forward["candidates"][d][0] != joint["candidates"][d][0] for d in range(256)):
            raise ValueError("Joint position-zero candidates do not exactly equal ordered forward candidates")
    return selected, ids, inventory, deduplication


def analyze(results, input_path, output):
    results, input_path, output = Path(results), Path(input_path), Path(output)
    if output.exists():
        raise ValueError("Candidate audit output must be fresh")
    sys.path.insert(0, str(ROOT))
    from attacker import evidence
    selected, ids, inventory, deduplication = verify_sources(results, input_path, evidence)
    # All immutable exported predictions are bound and validated before truth is read.
    data = json.loads(input_path.read_text())
    docs = data["evaluation"]
    if ({part: len(data[part]) for part in ("calibration", "train", "evaluation")}
            != {"calibration": 32, "train": 128, "evaluation": 256}
            or [doc["id"] for doc in docs] != ids
            or any(len(doc["tokens"]) != 32 or any(type(token) is not int or not 0 <= token < VOCABULARY
                   for token in doc["tokens"]) for doc in docs)):
        raise ValueError("Frozen truth schema/document alignment differs")
    totals = {scope: empty_counts() for scope in SCOPES}
    positions = [empty_counts() for _ in range(32)]
    victim_rows = []
    for victim in VICTIM_SEEDS:
        local = {scope: empty_counts() for scope in SCOPES}
        forward, joint = [selected[victim, 17, arm] for arm in ARMS]
        for d, doc in enumerate(docs):
            for arm, source in zip(ARMS, (forward, joint)):
                predicted = source["payload"]["predictions"][d]
                correct = sum(a == b for a, b in zip(doc["tokens"], predicted))
                metric = source["cell"]["metrics"][d]
                if (metric["document_id"] != doc["id"] or metric["token_count"] != 32
                        or metric["correct_tokens"] != correct or metric["token_accuracy"] != correct / 32
                        or metric["sequence_exact"] is not (correct == 32)):
                    raise ValueError("Original primary token metrics do not independently rescore")
            for p, truth in enumerate(doc["tokens"]):
                counts = observe(truth, forward["payload"]["predictions"][d][p],
                    joint["payload"]["predictions"][d][p], forward["candidates"][d][p], joint["candidates"][d][p])
                scope = "position_0" if p == 0 else "positions_1_to_31"
                for counter in (local["all"], local[scope], totals["all"], totals[scope], positions[p]):
                    counter.update(counts)
        victim_rows.extend(row(local[scope], victim_seed=victim, scope=scope) for scope in SCOPES)
    expected_positions = {"all": 10 * 256 * 32, "position_0": 10 * 256, "positions_1_to_31": 10 * 256 * 31}
    if any(totals[scope]["positions"] != size for scope, size in expected_positions.items()):
        raise ValueError("Deduplicated position totals differ")
    if totals["position_0"]["membership_forward_only"] or totals["position_0"]["membership_joint_only"]:
        raise ValueError("Position-zero candidate membership differs despite identical lists")
    for field in empty_counts():
        if totals["all"][field] != totals["position_0"][field] + totals["positions_1_to_31"][field]:
            raise ValueError("Position split accounting differs")
        if totals["all"][field] != sum(counter[field] for counter in positions):
            raise ValueError("Per-position aggregation differs")
    summaries = [row(totals[scope], scope=scope) for scope in SCOPES]
    position_rows = [row(counter, position_zero_based=p, position_one_based=p + 1)
                     for p, counter in enumerate(positions)]
    membership_rows = [{"scope": scope, "membership": membership,
        "positions": totals[scope]["membership_" + membership],
        **{outcome: totals[scope][f"{outcome}__{membership}"] for outcome in OUTCOMES}}
        for scope in SCOPES for membership in MEMBERSHIPS]
    for item in inventory:
        if digest(item["path"]) != item["sha256"]:
            raise ValueError("Source bytes changed during candidate audit: " + item["path"])
    output.mkdir(parents=True)
    write_csv(output / "summary.csv", summaries)
    write_csv(output / "per-victim.csv", victim_rows)
    write_csv(output / "per-position.csv", position_rows)
    write_csv(output / "membership-correctness.csv", membership_rows)
    dump(output / "input-inventory.json", {"sources": inventory, "rehashed_after_analysis": True,
        "source_script": {"path": str(Path(__file__).resolve()), "sha256": digest(__file__)},
        "evidence_implementation": {"path": str(Path(evidence.__file__).resolve()), "sha256": digest(evidence.__file__)},
        "attack_seed_deduplication": deduplication})
    lines = ["# Original primary candidate-list audit", "",
        "Post-hoc descriptive re-scoring of retained L6/public k=32, beam=2 predictions. All 16 original evidence chains and 40 primary exports were verified. Attack seeds 17 and 29 have byte-identical exports within every victim/arm and are counted once: ten paired victim runs, 256 documents, 32 token positions.", "",
        "| Positions | Count | Forward candidate recall (%) | Joint candidate recall (%) | Forward correct (%) | Joint correct (%) | Rescues | Harms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for value in summaries:
        lines.append(f"| {value['scope']} | {value['positions']} | {value['forward_candidate_truth_recall_percent']:.5f} | {value['joint_candidate_truth_recall_percent']:.5f} | {value['forward_prediction_accuracy_percent']:.5f} | {value['joint_prediction_accuracy_percent']:.5f} | {value['outcome_joint_rescue']} | {value['outcome_joint_harm']} |")
    lines.extend(["", "Candidate membership partitions each truth token as present in both exported lists, forward only, joint only, or neither. A rescue means joint predicts the correct token while forward does not; a harm means the converse. These definitions describe final outputs and candidate membership, without identifying a unique causal proposal stream.", "",
        "| Positions | Truth membership | Count | Both correct | Joint rescue | Joint harm | Both wrong |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |"])
    for value in membership_rows:
        lines.append(f"| {value['scope']} | {value['membership']} | {value['positions']} | {value['both_correct']} | {value['joint_rescue']} | {value['joint_harm']} | {value['both_wrong']} |")
    lines.extend(["", "Truth available but not selected is a decoder error at a position where that decoder's exported candidate list contains the truth token. Summary and per-victim/per-position tables retain these counts, their percentages of positions, and their percentages of each decoder's errors. This is an after-the-fact diagnostic, not an attack with access to labels or a new attack run.",
        "", "Position zero has exactly the same ordered forward and joint candidate lists; it receives no shifted gradient proposal. Wider beam paths can still yield different selected position-zero tokens after the full-sequence objective is evaluated. Positions 1–31 are reported separately.",
        "", "Under the executed construction, joint-only membership relative to the forward top-32 list identifies a token supplied by the gradient top-16 proposals: forward refill cannot supply an ID outside the forward list. This is local candidate provenance, not proof of gradient necessity under larger activation budgets or globally optimal decoding. No new hypothesis test, uncertainty interval, or independent replication from duplicate attack seeds is claimed. Evidence chains establish internal consistency with pinned bytes, not independent execution authenticity.", ""])
    (output / "REPORT.md").write_text("\n".join(lines))
    result = {"schema": "paper-a.candidate-audit.v1", "status": "COMPLETE",
        "utc": datetime.now(timezone.utc).isoformat(), "analysis_type": "post_hoc_descriptive_counts_and_rates_only",
        "setting": {"model": "openai-community/gpt2", "depth": 6, "knowledge": "public",
            "candidate_count": 32, "beam_width": 2, "mode": "training",
            "victim_seeds": VICTIM_SEEDS, "verified_attack_seeds": ATTACK_SEEDS,
            "counted_attack_seed": 17, "deterministic_duplicate_seed": 29,
            "distinct_victim_pairs": 10, "documents_per_victim": 256, "positions_per_document": 32,
            "positions_all": 81920},
        "input_sha256": INPUT_SHA256, "source_script_sha256": digest(__file__),
        "verification": {"original_report_chains": 16, "primary_prediction_exports": 40,
            "paired_byte_identical_seed_checks": 20, "position_zero_ordered_candidates_equal": True,
            "truth_read_after_retained_export_verification": True, "source_bytes_rehashed_after_analysis": True,
            "candidate_dimensions_and_uniqueness": True, "selected_tokens_in_own_candidates": True,
            "stored_primary_token_metrics_independently_rescored": True,
            "membership_correctness_accounting": True},
        "summaries": summaries, "membership_correctness": membership_rows,
        "attack_seed_deduplication": deduplication,
        "model_execution_performed": False, "new_prediction_export_performed": False,
        "hypothesis_tests_performed": False, "software": {"python": platform.python_version(), "dependencies": "stdlib only"},
        "limitations": ["Post-hoc diagnostic on the same examined documents and checkpoints",
            "Joint-only membership identifies gradient proposal provenance within this frozen construction, not gradient necessity for other decoders or budgets",
            "Truth-side availability is an analysis diagnostic, never input to a new attacker",
            "No tests, intervals, causal attribution or global-optimality claim",
            "Internal evidence consistency is not independent execution authentication"],
        "outputs": {path.name: digest(path) for path in sorted(output.iterdir())}}
    artifact = output / "analysis.json"
    with evidence.EvidenceLogger(artifact, node_role="cpu/original-candidate-list-audit",
            extra={"source_script_sha256": digest(__file__), "input_sha256": INPUT_SHA256}) as logger:
        logger.event("candidate_audit_complete", {"verified_reports": 16, "verified_exports": 40,
            "distinct_victim_pairs": 10, "positions": 81920, "hypothesis_tests": False})
        dump(artifact, result)
        logger.finalize(artifact)
    return result


def self_test():
    """Synthetic accounting and schema regressions; never reads retained outcomes."""
    import unittest

    class CandidateAuditTests(unittest.TestCase):
        def test_all_membership_correctness_transitions(self):
            cases = [
                (1, 1, [1, 2], [1, 3], "both", "both_correct"),
                (2, 1, [1, 2], [1, 3], "both", "joint_rescue"),
                (1, 3, [1, 2], [1, 3], "both", "joint_harm"),
                (2, 3, [1, 2], [1, 3], "both", "both_wrong"),
                (1, 3, [1, 2], [3, 4], "forward_only", "joint_harm"),
                (2, 3, [1, 2], [3, 4], "forward_only", "both_wrong"),
                (2, 1, [2, 3], [1, 4], "joint_only", "joint_rescue"),
                (2, 4, [2, 3], [1, 4], "joint_only", "both_wrong"),
                (2, 4, [2, 3], [4, 5], "neither", "both_wrong"),
            ]
            total = empty_counts()
            for fp, jp, forward, joint, membership, outcome in cases:
                counts = observe(1, fp, jp, forward, joint)
                validate_accounting(counts)
                self.assertEqual(counts["membership_" + membership], 1)
                self.assertEqual(counts[f"{outcome}__{membership}"], 1)
                total.update(counts)
            value = row(total, scope="synthetic")
            self.assertEqual(value["positions"], 9)
            self.assertEqual(value["outcome_joint_rescue"], 2)
            self.assertEqual(value["outcome_joint_harm"], 2)
            self.assertEqual(value["forward_truth_available_but_not_selected"], 3)
            self.assertEqual(value["joint_truth_available_but_not_selected"], 3)
            self.assertEqual(value["joint_minus_forward_prediction_accuracy_pp"], 0)

        def test_selected_prediction_outside_candidates_rejected(self):
            with self.assertRaises(ValueError):
                observe(1, 1, 2, [2, 3], [1, 2])

        def test_invalid_accounting_rejected(self):
            counts = observe(1, 2, 1, [1, 2], [1, 3])
            counts["joint_rescue__neither"] += 1
            with self.assertRaises(ValueError):
                validate_accounting(counts)

        def test_prediction_candidate_schema(self):
            import copy
            payload = {"document_ids": [f"synthetic-{d}" for d in range(256)],
                "predictions": [[0] * 32 for _ in range(256)],
                "diagnostics": {"batches": [{"candidate_ids": [[list(range(32)) for _ in range(32)]
                    for _ in range(8)], "forward_residual_sum": [0.] * 8} for _ in range(32)]}}
            candidates = validate_prediction(payload)
            self.assertEqual((len(candidates), len(candidates[0]), len(candidates[0][0])), (256, 32, 32))
            duplicate = copy.deepcopy(payload)
            duplicate["diagnostics"]["batches"][0]["candidate_ids"][0][0][1] = 0
            with self.assertRaises(ValueError):
                validate_prediction(duplicate)
            outside = copy.deepcopy(payload)
            outside["predictions"][0][0] = 32
            with self.assertRaises(ValueError):
                validate_prediction(outside)
            nonfinite = copy.deepcopy(payload)
            nonfinite["diagnostics"]["batches"][0]["forward_residual_sum"][0] = float("nan")
            with self.assertRaises(ValueError):
                validate_prediction(nonfinite)

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CandidateAuditTests))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    arguments = parser.parse_args()
    if arguments.self_test:
        self_test()
    else:
        if any(value is None for value in (arguments.results, arguments.input, arguments.output)):
            parser.error("--results, --input and --output are required unless --self-test is selected")
        result = analyze(arguments.results, arguments.input, arguments.output)
        print(json.dumps({"status": result["status"], "setting": result["setting"],
                          "summaries": result["summaries"]}, indent=2))
