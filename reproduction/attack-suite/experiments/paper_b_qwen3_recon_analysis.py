"""Statistical analysis for Experiment #4 (PB-R1 Major 4 & Paper A R4 Major 4):
Qwen3-0.6B (GQA, RoPE, QK-Norm, SwiGLU, 28 layers, d=1024, V=151,936)
Controlled Joint Gradient-Plus-Activation Text-Reconstruction Benchmark.

Merges the 8 independent victim seeds (toula: 11, 22, 33, 44; koula: 55, 66, 77, 88)
across split depths L in {1, 7, 14, 21} and knowledge grants {public, exact} on 256
held-out 32-token WikiText-2 passages (qwen_input.json), verifies hash-chained
evidence logs and per-setup prediction sidecar SHA-256 digests, and computes
100,000-replicate crossed two-factor (8 victims x 256 documents) bootstrap confidence
intervals (seed 93281) for per-arm metrics and paired contrasts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _two_way_bootstrap_ci(
    matrix_s_d: np.ndarray,
    replicates: int = 100000,
    seed: int = 93281,
) -> dict:
    """Two-way crossed bootstrap over S victim seeds x D documents."""
    a = np.asarray(matrix_s_d, dtype=float)
    n_s, n_d = a.shape
    mean_val = float(a.mean())
    seed_means = a.mean(axis=1)
    seed_sd = float(seed_means.std(ddof=1)) if n_s > 1 else 0.0
    doc_means = a.mean(axis=0)
    doc_sd = float(doc_means.std(ddof=1)) if n_d > 1 else 0.0

    rng = np.random.default_rng(seed)
    s_idx = rng.integers(0, n_s, size=(replicates, n_s))
    d_idx = rng.integers(0, n_d, size=(replicates, n_d))
    doc_boot = np.empty((replicates, n_s), dtype=float)
    for s in range(n_s):
        doc_boot[:, s] = a[s, d_idx].mean(axis=1)
    draws = np.take_along_axis(doc_boot, s_idx, axis=1).mean(axis=1)
    ci95 = np.quantile(draws, [0.025, 0.975])
    p_gt_zero = float((draws > 0.0).mean())
    return {
        "mean": mean_val,
        "seed_sd": seed_sd,
        "doc_sd": doc_sd,
        "ci95": [float(ci95[0]), float(ci95[1])],
        "prob_gt_zero": p_gt_zero,
    }


def analyze_qwen3_recon(toula_dir: Path, koula_dir: Path, output_dir: Path) -> dict:
    from attacker.evidence import verify_evidence

    output_dir.mkdir(parents=True, exist_ok=True)
    host_reports = {}
    for host, hdir in [("toula", toula_dir), ("koula", koula_dir)]:
        rpath = hdir / "qwen3_recon_replay.json"
        ok, problems = verify_evidence(str(rpath) + ".evidence.json")
        if not ok:
            raise ValueError(f"Evidence verification failed for {host}: {problems}")
        r = json.loads(rpath.read_text())
        if r.get("status") != "COMPLETE":
            raise ValueError(f"Host {host} status is not COMPLETE: {r.get('status')}")
        kv_checks = r.get("kv_cache_verifications", [])
        if not kv_checks or not all(k.get("exact_discrete_tokens", False) for k in kv_checks):
            raise ValueError(f"Host {host} KV-cache verification failed: {kv_checks}")
        host_reports[host] = (hdir, r)

    victims = {}
    cells = {}
    sidecar_cache = {}
    preds = {}
    for host, (hdir, r) in host_reports.items():
        for v in r["victims"]:
            seed = int(v["seed"])
            if seed in victims:
                raise ValueError(f"Duplicate victim seed {seed}")
            victims[seed] = v
        for cell in r["cells"]:
            key = (int(cell["depth"]), str(cell["knowledge"]), str(cell["arm"]), int(cell["victim_seed"]))
            if key in cells:
                raise ValueError(f"Duplicate cell {key}")
            sc_rel = cell["sidecar_file"]
            sc_path = hdir / sc_rel
            if sc_path not in sidecar_cache:
                actual_sha = _digest(sc_path)
                if actual_sha != cell["sidecar_sha256"]:
                    raise ValueError(f"Sidecar SHA-256 mismatch for {sc_path}: {actual_sha} != {cell['sidecar_sha256']}")
                sidecar_cache[sc_path] = json.loads(sc_path.read_text())
            cells[key] = cell
            preds[key] = sidecar_cache[sc_path]["arms"][str(cell["arm"])]

    seeds = [11, 22, 33, 44, 55, 66, 77, 88]
    if sorted(victims.keys()) != seeds:
        raise ValueError(f"Expected victim seeds {seeds}, got {sorted(victims.keys())}")

    depths = sorted({k[0] for k in cells.keys()})
    by_depth_knowledge_arm: dict[str, dict[str, dict[str, dict]]] = {}
    matrices: dict[tuple[int, str, str], dict[str, np.ndarray]] = {}

    for depth in depths:
        d_key = str(depth)
        by_depth_knowledge_arm[d_key] = {}
        for knowledge in ("public", "exact"):
            arms = [
                a for a in ("backward", "forward_k32", "joint_true_k32", "joint_shuffled_k32", "forward_k128", "joint_true_k128")
                if (depth, knowledge, a, seeds[0]) in cells
            ]
            by_depth_knowledge_arm[d_key][knowledge] = {}
            for arm_idx, arm in enumerate(arms):
                tok_mat = np.empty((8, 256), dtype=float)
                pos0_mat = np.empty((8, 256), dtype=float)
                pos1_31_mat = np.empty((8, 256), dtype=float)
                exact_mat = np.empty((8, 256), dtype=float)
                span8_mat = np.empty((8, 256), dtype=float)
                lrun_mat = np.empty((8, 256), dtype=float)

                wall_vals = []
                cand_overall = []
                cand_pos0 = []
                cand_pos1_31 = []

                for s_idx, seed in enumerate(seeds):
                    c = cells[(depth, knowledge, arm, seed)]
                    p = preds[(depth, knowledge, arm, seed)]
                    docs = p["per_document"]
                    if len(docs) != 256:
                        raise ValueError(f"Expected 256 docs in {(depth, knowledge, arm, seed)}, got {len(docs)}")
                    for d_idx, d in enumerate(docs):
                        tok_mat[s_idx, d_idx] = float(d["token_accuracy"])
                        pos0_mat[s_idx, d_idx] = float(d["pos0_acc"])
                        pos1_31_mat[s_idx, d_idx] = float(d["pos1_31_acc"])
                        exact_mat[s_idx, d_idx] = 1.0 if d["sequence_exact"] else 0.0
                        span8_mat[s_idx, d_idx] = float(d["span8_rate"] or 0.0)
                        lrun_mat[s_idx, d_idx] = float(d["longest_exact_run"])

                    wall_vals.append(float(c["wall_seconds"]))
                    if c["candidate_recall"] is not None:
                        cand_overall.append(float(c["candidate_recall"]["overall"]))
                        cand_pos0.append(float(c["candidate_recall"]["pos0"]))
                        cand_pos1_31.append(float(c["candidate_recall"]["pos1_31"]))

                matrices[(depth, knowledge, arm)] = {
                    "token_accuracy": tok_mat,
                    "pos0_accuracy": pos0_mat,
                    "pos1_31_accuracy": pos1_31_mat,
                    "exact_match": exact_mat,
                    "span8_rate": span8_mat,
                    "longest_exact_run": lrun_mat,
                }

                base_seed = 93281 + depth * 1000 + (0 if knowledge == "public" else 500) + arm_idx * 10
                by_depth_knowledge_arm[d_key][knowledge][arm] = {
                    "depth": depth,
                    "knowledge": knowledge,
                    "arm": arm,
                    "n_seeds": 8,
                    "n_documents": 256,
                    "total_evaluated_documents": 8 * 256,
                    "total_evaluated_tokens": 8 * 256 * 32,
                    "token_accuracy": _two_way_bootstrap_ci(tok_mat, replicates=100000, seed=base_seed),
                    "pos0_accuracy": _two_way_bootstrap_ci(pos0_mat, replicates=20000, seed=base_seed + 1),
                    "pos1_31_accuracy": _two_way_bootstrap_ci(pos1_31_mat, replicates=20000, seed=base_seed + 2),
                    "exact_match": _two_way_bootstrap_ci(exact_mat, replicates=100000, seed=base_seed + 3),
                    "span8_rate": _two_way_bootstrap_ci(span8_mat, replicates=20000, seed=base_seed + 4),
                    "longest_exact_run": _two_way_bootstrap_ci(lrun_mat, replicates=20000, seed=base_seed + 5),
                    "candidate_recall_mean": {
                        "overall": float(np.mean(cand_overall)) if cand_overall else None,
                        "pos0": float(np.mean(cand_pos0)) if cand_pos0 else None,
                        "pos1_31": float(np.mean(cand_pos1_31)) if cand_pos1_31 else None,
                    },
                    "wall_seconds_mean": float(np.mean(wall_vals)),
                    "amortized_seconds_per_document_mean": float(np.mean(wall_vals)) / 256.0,
                }

    # Paired contrasts
    paired_contrasts = []
    for depth in depths:
        for knowledge in ("public", "exact"):
            pairs = [
                ("joint_true_k32", "forward_k32", "joint_k32_minus_forward_k32"),
                ("joint_true_k32", "joint_shuffled_k32", "joint_k32_minus_shuffled_k32"),
                ("joint_true_k32", "backward", "joint_k32_minus_backward"),
                ("joint_shuffled_k32", "forward_k32", "shuffled_k32_minus_forward_k32"),
            ]
            if (depth, knowledge, "forward_k128") in matrices:
                pairs.extend([
                    ("joint_true_k128", "forward_k128", "joint_k128_minus_forward_k128"),
                    ("forward_k128", "forward_k32", "forward_k128_minus_forward_k32"),
                    ("joint_true_k32", "forward_k128", "joint_k32_minus_forward_k128"),
                    ("joint_true_k128", "joint_true_k32", "joint_k128_minus_joint_k32"),
                ])
            for c_idx, (left_arm, right_arm, cname) in enumerate(pairs):
                m_left = matrices[(depth, knowledge, left_arm)]
                m_right = matrices[(depth, knowledge, right_arm)]
                diff_tok = m_left["token_accuracy"] - m_right["token_accuracy"]
                diff_pos1 = m_left["pos1_31_accuracy"] - m_right["pos1_31_accuracy"]
                diff_ex = m_left["exact_match"] - m_right["exact_match"]
                diff_sp8 = m_left["span8_rate"] - m_right["span8_rate"]
                c_seed = 93281 + depth * 2000 + (0 if knowledge == "public" else 1000) + c_idx * 20
                paired_contrasts.append({
                    "depth": depth,
                    "knowledge": knowledge,
                    "contrast": cname,
                    "left": left_arm,
                    "right": right_arm,
                    "token_accuracy_diff": _two_way_bootstrap_ci(diff_tok, replicates=100000, seed=c_seed),
                    "pos1_31_accuracy_diff": _two_way_bootstrap_ci(diff_pos1, replicates=20000, seed=c_seed + 1),
                    "exact_match_diff": _two_way_bootstrap_ci(diff_ex, replicates=100000, seed=c_seed + 2),
                    "span8_rate_diff": _two_way_bootstrap_ci(diff_sp8, replicates=20000, seed=c_seed + 3),
                })

    # Knowledge grant contrasts (exact - public) for k=32 arms
    knowledge_contrasts = []
    for depth in depths:
        for a_idx, arm in enumerate(("backward", "forward_k32", "joint_true_k32", "joint_shuffled_k32")):
            m_ex = matrices[(depth, "exact", arm)]
            m_pub = matrices[(depth, "public", arm)]
            diff_tok = m_ex["token_accuracy"] - m_pub["token_accuracy"]
            diff_doc = m_ex["exact_match"] - m_pub["exact_match"]
            k_seed = 193281 + depth * 500 + a_idx * 10
            knowledge_contrasts.append({
                "depth": depth,
                "arm": arm,
                "contrast": "exact_minus_public",
                "token_accuracy_diff": _two_way_bootstrap_ci(diff_tok, replicates=50000, seed=k_seed),
                "exact_match_diff": _two_way_bootstrap_ci(diff_doc, replicates=50000, seed=k_seed + 1),
            })

    victim_list = [victims[s] for s in seeds]
    nll_before = [float(v["evaluation_nll_before"]) for v in victim_list]
    nll_after = [float(v["evaluation_nll_after"]) for v in victim_list]
    ppl_before = [math.exp(x) for x in nll_before]
    ppl_after = [math.exp(x) for x in nll_after]

    result = {
        "schema": "paper-b.qwen3-recon-analysis.v1",
        "model": "Qwen/Qwen3-0.6B",
        "architecture": {
            "num_hidden_layers": 28,
            "hidden_size": 1024,
            "vocab_size": 151936,
            "attention": "GQA (16Q / 8KV heads, head_dim=128, QK-Norm, RoPE)",
            "mlp": "SwiGLU",
            "norm": "RMSNorm",
        },
        "protocol": {
            "seeds": seeds,
            "n_seeds": 8,
            "n_calibration_docs": 32,
            "n_train_docs": 128,
            "n_eval_docs": 256,
            "seq_len": 32,
            "split_depths": depths,
            "bootstrap_replicates": 100000,
            "bootstrap_seed": 93281,
            "n_total_cells": len(cells),
            "n_sidecars_verified": len(sidecar_cache),
        },
        "hosts": {
            h: {
                "elapsed_seconds": r["elapsed_seconds"],
                "kv_cache_verifications": r["kv_cache_verifications"],
                "victim_seeds": r["victim_seeds"],
                "n_cells": len(r["cells"]),
            }
            for h, (_, r) in host_reports.items()
        },
        "victims": {
            "nll_before_mean": float(np.mean(nll_before)),
            "nll_after_mean": float(np.mean(nll_after)),
            "nll_after_sd": float(np.std(nll_after, ddof=1)),
            "perplexity_before_mean": float(np.mean(ppl_before)),
            "perplexity_after_mean": float(np.mean(ppl_after)),
            "perplexity_after_sd": float(np.std(ppl_after, ddof=1)),
            "per_seed": victim_list,
        },
        "by_depth_knowledge_arm": by_depth_knowledge_arm,
        "paired_contrasts": paired_contrasts,
        "knowledge_contrasts": knowledge_contrasts,
    }

    out_json = output_dir / "qwen3_recon_summary.json"
    out_json.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    lines = [
        "# Qwen3-0.6B Controlled Joint Text-Reconstruction Benchmark (`PB-R1 Major 4` & `R4 Major 4`)",
        "",
        "- **Model**: `Qwen/Qwen3-0.6B` (28 layers, $d=1024$, $V=151,936$, GQA 16Q/8KV, QK-Norm, RoPE, SwiGLU, RMSNorm)",
        f"- **Protocol**: 8 independent victim seeds (`11..88`), 256 held-out 32-token WikiText-2 passages (`2,048` document evaluations = `65,536` token positions per cell), {len(cells)} total attack cells verified (`{len(sidecar_cache)}/{len(sidecar_cache)}` sidecar SHA-256 + hash-chained evidence logs).",
        f"- **Victim Fine-Tuning Utility**: Eval NLL reduced from `{result['victims']['nll_before_mean']:.4f}` (PPL `{result['victims']['perplexity_before_mean']:.2f}`) to `{result['victims']['nll_after_mean']:.4f} ± {result['victims']['nll_after_sd']:.4f}` (PPL `{result['victims']['perplexity_after_mean']:.2f} ± {result['victims']['perplexity_after_sd']:.2f}`).",
        "",
        "## 1. Per-Arm Token Accuracy and Exact-Document Recovery",
        "",
        "| Depth $L$ (% Depth) | Grant | `backward` Tok% | `forward_k32` Tok% (Doc%) | `joint_true_k32` Tok% (Doc%) | `joint_shuffled_k32` Tok% (Doc%) | `forward_k128` Tok% (Doc%) | `joint_true_k128` Tok% (Doc%) |",
        "|---:|:---|---:|---:|---:|---:|---:|---:|",
    ]
    for depth in depths:
        pct_d = int(round(100.0 * depth / 28.0))
        for knowledge in ("public", "exact"):
            row = by_depth_knowledge_arm[str(depth)][knowledge]
            bwd_t = 100.0 * row["backward"]["token_accuracy"]["mean"]
            f32_t = 100.0 * row["forward_k32"]["token_accuracy"]["mean"]
            f32_d = 100.0 * row["forward_k32"]["exact_match"]["mean"]
            j32_t = 100.0 * row["joint_true_k32"]["token_accuracy"]["mean"]
            j32_d = 100.0 * row["joint_true_k32"]["exact_match"]["mean"]
            s32_t = 100.0 * row["joint_shuffled_k32"]["token_accuracy"]["mean"]
            s32_d = 100.0 * row["joint_shuffled_k32"]["exact_match"]["mean"]
            if "forward_k128" in row:
                f128_t = 100.0 * row["forward_k128"]["token_accuracy"]["mean"]
                f128_d = 100.0 * row["forward_k128"]["exact_match"]["mean"]
                j128_t = 100.0 * row["joint_true_k128"]["token_accuracy"]["mean"]
                j128_d = 100.0 * row["joint_true_k128"]["exact_match"]["mean"]
                f128_str = f"{f128_t:.2f}% ({f128_d:.2f}%)"
                j128_str = f"**{j128_t:.2f}%** (**{j128_d:.2f}%**)"
            else:
                f128_str = "—"
                j128_str = "—"
            lines.append(
                f"| $L={depth}$ ({pct_d}%) | `{knowledge}` | {bwd_t:.2f}% | {f32_t:.2f}% ({f32_d:.2f}%) | **{j32_t:.2f}%** (**{j32_d:.2f}%**) | {s32_t:.2f}% ({s32_d:.2f}%) | {f128_str} | {j128_str} |"
            )

    lines.extend([
        "",
        "## 2. Paired Two-Way Bootstrap Contrasts (100,000 Replicates, Seed 93281)",
        "",
        "| Depth $L$ | Grant | Contrast | Token Diff (pp) [95% CI] | Pos 1..31 Diff (pp) [95% CI] | Exact-Doc Diff (pp) [95% CI] |",
        "|---:|:---|:---|---:|---:|---:|",
    ])
    for c in paired_contrasts:
        td = c["token_accuracy_diff"]
        p1 = c["pos1_31_accuracy_diff"]
        ed = c["exact_match_diff"]
        lines.append(
            f"| $L={c['depth']}$ | `{c['knowledge']}` | `{c['contrast']}` | "
            f"`{100*td['mean']:+.2f}` [`{100*td['ci95'][0]:+.2f}`, `{100*td['ci95'][1]:+.2f}`] | "
            f"`{100*p1['mean']:+.2f}` [`{100*p1['ci95'][0]:+.2f}`, `{100*p1['ci95'][1]:+.2f}`] | "
            f"`{100*ed['mean']:+.2f}` [`{100*ed['ci95'][0]:+.2f}`, `{100*ed['ci95'][1]:+.2f}`] |"
        )

    out_md = output_dir / "qwen3_recon_report.md"
    out_md.write_text("\n".join(lines) + "\n")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--toula", type=Path, required=True)
    ap.add_argument("--koula", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    analyze_qwen3_recon(args.toula, args.koula, args.output)
    print(json.dumps({"status": "QWEN3_RECON_ANALYSIS_COMPLETE", "output": str(args.output / "qwen3_recon_summary.json")}, indent=2))


if __name__ == "__main__":
    main()
