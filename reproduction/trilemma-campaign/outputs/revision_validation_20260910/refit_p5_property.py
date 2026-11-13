"""Post-hoc target-resolved verification using unchanged retained P5 scorers.

No model training, new calibration selection, or historical output replacement.
Invoke with the project Python; --root can point to a restored evidence checkout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("p5_property_refit.json"))
    parser.add_argument("--benchmark-structure", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root / "src"))
    import pcr.privacy_attacks_p4b as pb
    from pcr.data import load_corpus, make_probe_batch, split_corpus

    torch.set_num_threads(2)
    start = time.perf_counter()
    index = {r["path"]: r for r in json.loads((root / "evidence/release-20260910-v1.json").read_text())["files"]}
    checked_inputs = {}

    def checked(rel):
        path = root / rel
        got = digest(path)
        if got != index[rel]["sha256"]:
            raise ValueError(f"Retained input does not match release index: {rel}")
        checked_inputs[rel] = {"sha256": got, "bytes": path.stat().st_size}
        return path

    source_before = {str(p.relative_to(root)): digest(p) for p in sorted((root / "src/pcr").glob("*.py"))}
    source_before["scripts/analyze_delegation.py"] = digest(root / "scripts/analyze_delegation.py")
    matrix = yaml.safe_load(checked("experiments/configs/delegation_matrix.yaml").read_text()) if "experiments/configs/delegation_matrix.yaml" in index else yaml.safe_load((root / "experiments/configs/delegation_matrix.yaml").read_text())
    original = json.loads(checked("outputs/delegation/delegation_analysis.json").read_text())
    p4b = json.loads(checked("outputs/privacy_extended_p4b/privacy_extended_p4b.json").read_text())
    qualification = p4b["families"]["sequence_property_v2"]["qualification"]
    qualified_targets = sorted(k for k, v in qualification["per_target"].items() if v["passes_floor"] and v["null_within_band"])
    _, heldout = split_corpus(load_corpus(checked("data/raw/tinyshakespeare.txt")), matrix["fixed"]["corpus_split"]["train_fraction"])
    tokens, _ = make_probe_batch(heldout, matrix["fixed"]["probe_batch_size"], matrix["fixed"]["sequence_length"])
    events = torch.load(checked("outputs/confirmatory/calibration-s919/calibration_pairs.pt"), map_location="cpu", weights_only=True)
    result = {
        "status": "post-hoc target-resolved verification; not a new confirmatory cohort or historical training reproduction",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "numpy": np.__version__, "device": "cpu", "torch_threads": torch.get_num_threads()},
        "scorer": "Unchanged pcr.privacy_attacks_p4b.family1_v2_train/apply; seed=919; original per-defense q; matched_test_null=True; all three original targets retained",
        "qualification": {"source": "outputs/privacy_extended_p4b/privacy_extended_p4b.json", "reused_without_refitting": True, "qualified_targets": qualified_targets, "record": qualification},
        "threshold": 0.02, "records": [], "cells": {},
        "structure_scores": "Stored qualified structure scores are reused, not independently refitted. An optional one-epoch speed benchmark is not a scientific result.",
    }
    banks = {}
    for defense, spec in matrix["factors"]["defenses"].items():
        t0 = time.perf_counter()
        banks[defense] = pb.family1_v2_train(events, spec["q_bits"], seed=919)
        print(f"{defense} property bank: {time.perf_counter() - t0:.2f}s", flush=True)
    if args.benchmark_structure:
        q_events = [{"latents": pb.quantize_latents(e["latents"], 8), "tokens": e["tokens"]} for e in events]
        features, labels = pb.family2_v2_dataset(q_events)
        t0 = time.perf_counter()
        benchmark_model = pb.train_probe(features, labels, features.shape[1], 1, 4096, 3e-3, 919, torch.device("cpu"))
        elapsed = time.perf_counter() - t0
        result["structure_runtime_benchmark"] = {"purpose": "execution-time estimate only; output model discarded and no score used", "examples": len(labels), "epochs": 1, "seconds": elapsed, "full_forward_bank_fits": 4, "frozen_epochs_per_fit": 200, "estimated_full_fit_seconds": elapsed * 800}
        del benchmark_model, features, labels, q_events
        print(f"structure benchmark one epoch {elapsed:.2f}s; projected four 200-epoch fits {elapsed * 800 / 60:.1f} minutes", flush=True)
    for span_key, segment in matrix["factors"]["segment_span"].items():
        for defense, spec in matrix["factors"]["defenses"].items():
            cell_key = f"{span_key}-{defense}"
            cell_records = []
            for seed in matrix["factors"]["seed"]:
                run_id = f"p5-s{segment['span']}-{defense}-trained-s{seed}"
                rel = f"outputs/delegation/{run_id}/transcript_tensors.pt"
                captures = torch.load(checked(rel), map_location="cpu", weights_only=True)
                final = max((s for s in captures if s.get("probe_batch_index", 0) == 0), key=lambda s: s["step"])
                scores = pb.family1_v2_apply(banks[defense], final["forward_values"], tokens, spec["q_bits"], matched_test_null=True)
                row = {"run_id": run_id, "cell": cell_key, "seed": seed, "q_bits": spec["q_bits"], "capture_step": final["step"], "scores": scores, "all_target_max": max(v["r_advantage"] for v in scores.values()), "qualified_target_max": max(scores[k]["r_advantage"] for k in qualified_targets)}
                cell_records.append(row)
                result["records"].append(row)
                del captures, final
            stored = original["cells"][cell_key]["gates"]["privacy"]["per_variant"]["sequence_property_v2@forward@p4b"]["per_seed"]
            recomputed = [r["all_target_max"] for r in sorted(cell_records, key=lambda r: r["run_id"])]
            differences = [abs(a - b) for a, b in zip(stored, recomputed, strict=True)]
            maxima = {target: max(r["scores"][target]["r_advantage"] for r in cell_records) for target in cell_records[0]["scores"]}
            result["cells"][cell_key] = {"per_target_worst_seed_advantage": maxima, "qualified_punctuation_gate_pass": maxima["punctuation_fraction"] <= 0.02, "original_all_target_aggregation": stored, "recomputed_all_target_aggregation": recomputed, "maximum_absolute_aggregation_difference": max(differences), "aggregation_matches_at_1e-10": max(differences) <= 1e-10}
            print(f"{cell_key}: punctuation max={maxima['punctuation_fraction']:.12f}, aggregation delta={max(differences):.3g}", flush=True)
    result["checked_retained_inputs"] = checked_inputs
    result["source_sha256_at_start"] = source_before
    result["source_unchanged"] = {rel: digest(root / rel) == sha for rel, sha in source_before.items()}
    result["input_hashes_unchanged_after_refit"] = all(digest(root / rel) == item["sha256"] for rel, item in checked_inputs.items())
    result["script_sha256"] = digest(Path(__file__))
    result["matrix_sha256"] = digest(root / "experiments/configs/delegation_matrix.yaml")
    result["elapsed_seconds"] = time.perf_counter() - start
    result["finished_at"] = datetime.now(timezone.utc).isoformat()
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}; {result['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
