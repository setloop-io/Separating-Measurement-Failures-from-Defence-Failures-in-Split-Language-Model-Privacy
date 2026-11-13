"""Export and verify the original Paper A secondary contrasts; no new bootstrap.

Reads frozen contrast outputs, their released hash bindings, retained matrices,
protocols, source journals and predictions. Never changes original results.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def table(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=ROOT / "artifacts/paper-a-review-round2-20261020/contrast-audit")
    args = ap.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("output must be new or empty")
    sys.path.insert(0, str(ROOT))
    from attacker.evidence import verify_evidence
    from experiments.analyze_validation import contrasts

    original = ROOT / "outputs/validation-v3-review-2026-09-15"
    release = ROOT / "build/hf-release-stage1/paper-a"
    release_manifest = {name: digest for digest, name in
                        (line.split("  ", 1) for line in (release / "MANIFEST.sha256").read_text().splitlines())}
    sources = []

    def record(path, role, expected=None):
        actual = sha(path)
        if expected is not None and actual != expected:
            raise ValueError(f"source digest mismatch: {path}")
        sources.append({"path": str(path.relative_to(ROOT)), "role": role,
                        "sha256": actual, "expected_sha256": expected})
        return actual

    record(release / "MANIFEST.sha256", "retained_release_manifest")
    for name in ("contrasts.csv", "contrasts.json", "audit.json", "token_accuracy_matrices.npz"):
        expected = release_manifest[f"v3-campaign/review/{name}"]
        record(original / name, "original_released_analysis", expected)
        record(release / f"v3-campaign/review/{name}", "byte_identical_release_copy", expected)
    protocol_path = ROOT / "experiments/validation_protocol.v3.json"
    record(protocol_path, "frozen_protocol")
    record(ROOT / "experiments/analyze_validation.py", "original_contrast_family_definition")
    p = json.loads(protocol_path.read_text())
    main_key, secondary_keys = contrasts(p)
    independent_keys = [(depth, copy, "training", f"joint_true_k{k}", right)
        for depth, copy, k in itertools.product(p["depths"], p["knowledge"], p["candidate_counts"])
        for right in (f"forward_k{k}", f"joint_shuffled_k{k}", f"joint_random_k{k}", "backward")]
    if len(independent_keys) != 48 or len(secondary_keys) != 47:
        raise ValueError("expected the original 48-pair grid and 47 secondary pairs")
    if set(independent_keys) - set(secondary_keys) != {main_key} or main_key != (
            6, "public", "training", "joint_true_k32", "forward_k32"):
        raise ValueError("secondary omission is not exactly the primary contrast")

    original_rows = json.loads((original / "contrasts.json").read_text())
    original_csv = list(csv.DictReader((original / "contrasts.csv").open()))
    expected_order = [main_key] + secondary_keys
    row_keys = [tuple(row[f] for f in ("depth", "knowledge", "mode", "left", "right"))
                for row in original_rows]
    if row_keys != expected_order or len(original_csv) != len(original_rows):
        raise ValueError("original contrast files do not exactly enumerate the protocol family")
    for json_row, csv_row in zip(original_rows, original_csv, strict=True):
        for field, value in csv_row.items():
            expected = json_row[field]
            if expected is None:
                valid = value == ""
            elif isinstance(expected, bool):
                valid = value == str(expected)
            elif isinstance(expected, (float, int)):
                valid = float(value) == expected
            else:
                valid = value == expected
            if not valid:
                raise ValueError(f"retained CSV/JSON mismatch: {field}")
    npz = np.load(original / "token_accuracy_matrices.npz", allow_pickle=False)
    matrices = {(int(key[0]), int(key[1]), key[2], int(key[3]), key[4], key[5]): value
                for key, value in zip(npz["keys"], npz["values"], strict=True)}
    needed = {(v, d, copy, a, mode, arm)
              for d, copy, mode, left, right in expected_order
              for v, a, arm in itertools.product(p["victim_seeds"], p["attack_seeds"], (left, right))}
    if len(needed) != 1040 or not needed <= set(matrices):
        raise ValueError("retained matrix does not cover the exact original contrast cell grid")
    audit = json.loads((original / "audit.json").read_text())
    old_shards = {entry["shard"]: entry for entry in audit["sources"]}
    found, selected_work, ids, prediction_checks = set(), {}, None, 0
    for index in range(16):
        shard = ROOT / f"results/validation-heldout-v3-r3-shard{index}"
        evidence_path = shard / "validation.json.evidence.json"
        record(evidence_path, "original_audit_bound_evidence_manifest", old_shards[index]["manifest_sha256"])
        ok, problems = verify_evidence(str(evidence_path))
        if not ok:
            raise ValueError(f"journal evidence verification failed: {index}: {problems}")
        evidence = json.loads(evidence_path.read_text())
        report_path = shard / "validation.json"
        record(report_path, "journal_and_original_audit_bound_report", evidence["artifact"]["sha256"])
        if sha(report_path) != old_shards[index]["report_sha256"]:
            raise ValueError("source report differs from original analysis audit")
        report = json.loads(report_path.read_text())
        if report["protocol"] != p or report["status"] != "COMPLETE":
            raise ValueError("source shard protocol/status mismatch")
        record(shard / "protocol-source.json", "source_protocol", report["protocol_sha256"])
        record(shard / "protocol-effective.json", "effective_protocol", report["effective_protocol_sha256"])
        record(shard / "input.json", "frozen_evaluation_input", p["frozen_inputs"]["tokenized_sha256"])
        data = json.loads((shard / "input.json").read_text())
        current_ids = [doc["id"] for doc in data["evaluation"]]
        if ids is None:
            ids = current_ids
        if current_ids != ids or len(ids) != 256 or len(set(ids)) != 256:
            raise ValueError("source document alignment mismatch")
        truth = np.asarray([doc["tokens"] for doc in data["evaluation"]])
        if truth.shape != (256, 32):
            raise ValueError("unexpected retained truth grid")
        journal_path = shard / "validation.json.evidence.jsonl"
        record(journal_path, "verified_hash_chained_journal")
        journal = {event["payload"]["cell_id"]: event["payload"]
                   for event in map(json.loads, journal_path.read_text().splitlines())
                   if event["kind"] == "cell_complete"}
        for cell in report["cells"]:
            key = tuple(cell[f] for f in ("victim_seed", "depth", "knowledge", "attack_seed", "mode", "arm"))
            if key not in needed:
                continue
            if key in found:
                raise ValueError("duplicate selected source cell")
            found.add(key)
            stored = np.asarray([row["token_accuracy"] for row in cell["metrics"]])
            if not np.array_equal(stored, matrices[key]):
                raise ValueError("released matrix differs from original retained report")
            event = journal[cell["cell_id"]]
            if event["prediction_sha256"] != cell["prediction_sha256"]:
                raise ValueError("prediction hash not journal-bound")
            path = shard / cell["prediction_file"]
            record(path, "journal_bound_retained_prediction", event["prediction_sha256"])
            pred = json.loads(path.read_text())
            if pred["document_ids"] != ids:
                raise ValueError("prediction document order differs")
            tokens = np.asarray(pred["predictions"])
            if tokens.shape != truth.shape:
                raise ValueError("prediction shape differs")
            recomputed = (tokens == truth).mean(axis=1)
            if not np.array_equal(recomputed, stored):
                raise ValueError("token accuracy does not reproduce from predictions")
            prediction_checks += 1
            selected_work[key] = cell["work"]
        print(f"Verified shard {index}: {prediction_checks}/{len(needed)} selected prediction cells", flush=True)
    if found != needed:
        raise ValueError("incomplete source prediction grid")

    verified_rows = []
    for source_row, key in zip(original_rows, expected_order, strict=True):
        d, copy, mode, left, right = key
        matrix = np.asarray([np.mean([matrices[v, d, copy, a, mode, left] -
                                     matrices[v, d, copy, a, mode, right]
                                     for a in p["attack_seeds"]], axis=0)
                             for v in p["victim_seeds"]])
        if float(matrix.mean()) != source_row["difference"] or not np.array_equal(
                matrix.mean(axis=1), np.asarray(source_row["per_victim_differences"])):
            raise ValueError("original contrast point estimate/victim differences do not reproduce")
        primary = key == main_key
        alpha = p["statistics"]["alpha"] / (1 if primary else len(secondary_keys))
        if source_row["primary"] != primary or source_row["alpha"] != alpha:
            raise ValueError("contrast family or interval multiplicity mismatch")
        if source_row["expected_draws_per_tail"] != p["statistics"]["bootstrap_replicates"]*alpha/2:
            raise ValueError("retained Monte Carlo tail count mismatch")
        if not source_row["lower"] <= source_row["difference"] <= source_row["upper"]:
            raise ValueError("invalid retained interval")
        verified_rows.append({**source_row, "k": int(left.rsplit("k", 1)[1]),
            "copy": "trained" if copy == "exact" else copy,
            "baseline_label": {"forward": "Activation-only", "joint_shuffled": "Shuffled gradient",
                               "joint_random": "Random candidates", "backward": "Backward-only"}[
                                   right.rsplit("_k", 1)[0]],
            "difference_pp": 100*source_row["difference"],
            "lower_pp": 100*source_row["lower"], "upper_pp": 100*source_row["upper"],
            "individual_confidence_percent": 100*(1-alpha),
            "interval_origin": "unchanged retained original percentile crossed-bootstrap output",
            "point_estimate_recomputed_from_retained_predictions": True})
    secondary = [row for row in verified_rows if not row["primary"]]
    if len(secondary) != 47 or not all(row["lower"] > 0 for row in secondary):
        raise ValueError("the original 47 positive secondary intervals were not reproduced")

    def mean_accuracy(d, copy, arm):
        return float(100*np.mean([matrices[v, d, copy, a, "training", arm]
                                 for v, a in itertools.product(p["victim_seeds"], p["attack_seeds"])]))

    main_checks = [{"arm": arm, "token_recovery_percent": mean_accuracy(6, "public", arm)}
                   for arm in ("forward_k8", "forward_k16", "forward_k32", "joint_shuffled_k32",
                               "joint_random_k32", "joint_true_k32")]
    ranges = [{"depth": d, "knowledge": copy, "token_recovery_percent": mean_accuracy(d, copy, "forward_k32")}
              for d, copy in itertools.product(p["depths"], p["knowledge"])]
    for v, d, copy, a, n in itertools.product(p["victim_seeds"], p["depths"], p["knowledge"],
                                            p["attack_seeds"], p["candidate_counts"]):
        reference = selected_work[v, d, copy, a, "training", f"forward_k{n}"]
        for family in ("true", "shuffled", "random"):
            work = selected_work[v, d, copy, a, "training", f"joint_{family}_k{n}"]
            for field in ("head_forward_calls", "head_forward_sequences", "head_forward_token_positions"):
                if work[field] != reference[field]:
                    raise ValueError("claimed equal prefix-search work does not hold")
    bounds = [min(r["token_recovery_percent"] for r in ranges), max(r["token_recovery_percent"] for r in ranges)]
    result = {"status": "original_pre_specified_contrasts_exported_without_new_inference",
        "family": {"full_grid_pairs": len(independent_keys), "secondary_pairs": len(secondary),
            "construction": "2 split depths x 2 client copies x 3 candidate budgets x 4 baseline families",
            "excluded_primary": verified_rows[0], "excluded_from_secondary_reason": "separately prespecified primary contrast",
            "baseline_families": ["activation-only", "shuffled-gradient", "random-candidate", "backward-only"],
            "perturbed_gradients_in_family": False, "sip_it_in_family": False,
            "nominal_family_alpha": p["statistics"]["alpha"],
            "per_secondary_interval_alpha": p["statistics"]["alpha"]/len(secondary),
            "individual_secondary_confidence_percent": 100*(1-p["statistics"]["alpha"]/len(secondary)),
            "nominal_family_confidence_percent": 100*(1-p["statistics"]["alpha"]),
            "bootstrap_replicates_original": p["statistics"]["bootstrap_replicates"],
            "bootstrap_seed_original": p["statistics"]["bootstrap_seed"],
            "expected_draws_per_secondary_tail": secondary[0]["expected_draws_per_tail"],
            "new_bootstrap_performed": False}, "secondary": secondary,
        "checks": {"all_csv_json_fields_agree": True, "exact_protocol_family_and_order": True,
            "release_manifest_hash_bindings": True, "all_original_shard_audit_hash_bindings": True,
            "all_source_journal_chains_verified": True, "prediction_cells_verified": prediction_checks,
            "matrix_rows_equal_source_prediction_token_accuracy": True, "all_paired_point_estimates_and_victim_differences_equal_original": True,
            "all_secondary_lower_bounds_positive": True, "equal_prefix_search_work_verified": True},
        "v2_range_checks": {"forward_k32_settings": ranges, "range_percent": bounds,
                            "range_rounded_two_decimals": [round(x, 2) for x in bounds]},
        "primary_setting_budget_and_control_checks": main_checks,
        "limitations": ["Original percentile bootstrap intervals are retained unchanged, not independently re-estimated here.",
            "Bonferroni supplies nominal family-wise coverage conditional on the approximate individual bootstrap intervals; this is not a mathematical guarantee of exact finite-sample coverage.",
            "The primary has its separate unadjusted 95% interval and is excluded from the 47-comparison Bonferroni family.",
            "Training corpus, short passage length and attack-family scope remain those of the original protocol.",
            "Hash/journal checks establish consistency with the retained frozen release, not independent external authenticity."]}
    for source in sources:
        if sha(ROOT / source["path"]) != source["sha256"]:
            raise ValueError("source changed during analysis")
    args.output.mkdir(parents=True, exist_ok=True)
    dump(args.output / "secondary-contrasts.json", result)
    columns = ("depth", "knowledge", "copy", "k", "left", "right", "baseline_label", "difference_pp",
               "lower_pp", "upper_pp", "alpha", "individual_confidence_percent", "positive_lower_bound")
    table(args.output / "secondary-contrasts.csv", [{field: row[field] for field in columns} for row in secondary])
    table(args.output / "budget-control-checks.csv", main_checks)
    tex = ["% Original prespecified secondary contrasts; all numbers in percentage points.",
           "% Columns: depth, client copy, k, baseline, estimate, lower, upper."]
    tex += [f"{row['depth']} & {row['copy']} & {row['k']} & {row['baseline_label']} & "
            f"{row['difference_pp']:.4f} & {row['lower_pp']:.4f} & {row['upper_pp']:.4f} \\\\" for row in secondary]
    (args.output / "secondary-contrasts-rows.tex").write_text("\n".join(tex)+"\n")
    dump(args.output / "source-manifest.json", {"sources": sources, "script_sha256": sha(Path(__file__)),
                                               "all_sources_rehashed_after_export": True})
    f16 = next(r["token_recovery_percent"] for r in main_checks if r["arm"] == "forward_k16")
    control_sentence = "; ".join(f"{r['arm']} {r['token_recovery_percent']:.6f}% "
                                f"({r['token_recovery_percent']-f16:+.6f} pp versus forward k=16)"
                                for r in main_checks if r["arm"] in ("joint_shuffled_k32", "joint_random_k32"))
    report = ["# Original secondary contrast audit", "",
        f"The exact original grid contains {len(independent_keys)} pairs. Removing only the separately "
        f"prespecified primary (L=6, public copy, k=32, joint versus activation-only) leaves {len(secondary)} "
        "secondary comparisons. Perturbed-gradient and SipIt comparisons do not enter this family.", "",
        f"The individual interval alpha is {result['family']['per_secondary_interval_alpha']:.17g}; "
        f"individual confidence is {result['family']['individual_secondary_confidence_percent']:.9f}%, "
        f"for nominal 95% family confidence. All {len(secondary)} retained lower bounds are positive. "
        f"The original {p['statistics']['bootstrap_replicates']:,} crossed-bootstrap replicates used seed "
        f"{p['statistics']['bootstrap_seed']}, with {secondary[0]['expected_draws_per_tail']:.5f} expected draws "
        "per corrected tail. No bootstrap was rerun.", "",
        f"All {prediction_checks} needed prediction cells reproduce the released token-accuracy matrix, "
        "the source reports and all paired point estimates/per-victim differences. Source journal chains "
        "and frozen-release/source-audit hashes match. All shared prefix-search work counters match; "
        "this does not imply equal total compute.", "",
        f"The activation-only k=32 range is {bounds[0]:.8f}–{bounds[1]:.8f}% "
        f"({bounds[0]:.2f}–{bounds[1]:.2f}% rounded). It is correctly budget-qualified in v2. "
        f"At L=6/public, forward k=16 is {f16:.8f}%. The controls are nearby, not identical: "
        + control_sentence + ".", "",
        "Appendix data: [47-row CSV](secondary-contrasts.csv), [complete JSON](secondary-contrasts.json), "
        "[TeX rows](secondary-contrasts-rows.tex). [Source manifest](source-manifest.json) records the existing "
        "release bindings. Original intervals were exported unchanged."]
    (args.output / "report.md").write_text("\n".join(report)+"\n")
    (args.output / "SOURCE-MANIFEST.sha256").write_text("".join(
        f"{row['sha256']}  {row['path']}\n" for row in sources)
        + f"{sha(Path(__file__))}  {Path(__file__).relative_to(ROOT)}\n")
    (args.output / "MANIFEST.sha256").write_text("".join(f"{sha(path)}  {path.name}\n"
        for path in sorted(args.output.iterdir()) if path.is_file() and path.name != "MANIFEST.sha256"))
    print(json.dumps({"output": str(args.output), "family": result["family"], "checks": result["checks"],
                      "budget_control_checks": main_checks}, indent=2), flush=True)


if __name__ == "__main__":
    main()
