"""Statistical analysis for Qwen3-0.6B & GPT-2 Ridge-Calibration Sample-Size Scaling
and Exact-Grant k=128 Completion (Closing XSci PoPETs Major 1 and reviewer3.com #2 & #3).

Verifies hash-chained .evidence.json logs and sidecar SHA-256 digests from toula and koula,
verifies 100% bit-identity on anchor cells against output/paper-b-qwen3-recon-20261020/,
merges the newly completed exact-grant k=128 cells (L in {1, 7, 14}) with the 304-cell
Qwen3-0.6B benchmark, and computes 100,000-replicate paired two-way bootstrap CIs (seed 93281).
"""
from __future__ import annotations

import argparse
import hashlib
import json
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


def analyze_cal_scale(
    toula_dir: Path,
    koula_dir: Path,
    base_recon_dir: Path,
    output_dir: Path,
) -> dict:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from attacker.evidence import verify_evidence

    output_dir.mkdir(parents=True, exist_ok=True)
    host_reports = {}
    for host, hdir in [("toula", toula_dir), ("koula", koula_dir)]:
        rpath = hdir / "qwen3_cal_scale_replay.json"
        ok, problems = verify_evidence(str(rpath) + ".evidence.json")
        if not ok:
            raise ValueError(f"Evidence verification failed for {host}: {problems}")
        r = json.loads(rpath.read_text())
        if r.get("status") != "COMPLETE":
            raise ValueError(f"Host {host} status is not COMPLETE: {r.get('status')}")
        host_reports[host] = (hdir, r)

    # 1. Load base 304-cell Qwen3-0.6B predictions for C=32 reference & anchor check
    base_preds = {}
    base_cells = {}
    for host in ("toula", "koula"):
        bdir = base_recon_dir / host
        br = json.loads((bdir / "qwen3_recon_replay.json").read_text())
        for c in br["cells"]:
            key = (int(c["depth"]), str(c["knowledge"]), 32, str(c["arm"]), int(c["victim_seed"]))
            sc = json.loads((bdir / c["sidecar_file"]).read_text())
            base_cells[key] = c
            base_preds[key] = sc["arms"][str(c["arm"])]

    # 2. Load new calibration-scale curves and beam cells
    qwen_curve = []
    gpt2_curve = []
    gpt2_beam = {}
    new_qwen_cells = {}
    new_qwen_preds = {}
    sidecar_cache = {}
    anchor_checks = []

    for host, (hdir, r) in host_reports.items():
        qwen_curve.extend(r["qwen3_cal_curve"])
        gpt2_curve.extend(r["gpt2_cal_curve"])
        for gb in r["gpt2_beam_cells"]:
            gkey = (int(gb["depth"]), str(gb["knowledge"]), int(gb["cal_passages"]), str(gb["arm"]), int(gb["victim_seed"]))
            gpt2_beam[gkey] = gb
        for c in r["qwen3_beam_cells"]:
            key = (int(c["depth"]), str(c["knowledge"]), int(c["cal_passages"]), str(c["arm"]), int(c["victim_seed"]))
            sc_path = hdir / c["sidecar_file"]
            if sc_path not in sidecar_cache:
                actual_sha = _digest(sc_path)
                if actual_sha != c["sidecar_sha256"]:
                    raise ValueError(f"Sidecar SHA-256 mismatch for {sc_path}")
                sidecar_cache[sc_path] = json.loads(sc_path.read_text())
            pred_obj = sidecar_cache[sc_path]["arms"][str(c["arm"])]
            if c.get("is_anchor"):
                ref_pred = base_preds[key]
                exact_tok_match = (pred_obj["predicted_tokens"] == ref_pred["predicted_tokens"])
                if not exact_tok_match:
                    raise ValueError(f"Anchor token mismatch on {key}!")
                anchor_checks.append({
                    "host": host,
                    "cell_id": c["cell_id"],
                    "exact_token_match_vs_frozen_304cell": exact_tok_match,
                    "token_accuracy_mean": c["summary"]["token_accuracy_mean"],
                })
            else:
                new_qwen_cells[key] = c
                new_qwen_preds[key] = pred_obj

    all_qwen_cells = dict(base_cells)
    all_qwen_cells.update(new_qwen_cells)
    all_qwen_preds = dict(base_preds)
    all_qwen_preds.update(new_qwen_preds)

    qwen_seeds = [11, 22, 33, 44, 55, 66, 77, 88]
    gpt2_seeds = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]

    # 3. Aggregate Qwen3-0.6B Calibration Curve across C in {32, 64, 128, 256, 384}
    qwen_curve_idx = {
        (int(x["depth"]), str(x["knowledge"]), int(x["cal_passages"]), int(x["victim_seed"])): x
        for x in qwen_curve
    }
    qwen_curve_summary = {}
    for depth in (1, 7, 14, 21):
        qwen_curve_summary[str(depth)] = {}
        for know in ("public", "exact"):
            qwen_curve_summary[str(depth)][know] = {}
            for C in (32, 64, 128, 256, 384):
                rows = [qwen_curve_idx[(depth, know, C, s)] for s in qwen_seeds]
                qwen_curve_summary[str(depth)][know][str(C)] = {
                    "cal_passages": C,
                    "cal_tokens": C * 32,
                    "cal_unique_tokens": int(rows[0]["cal_unique_tokens"]),
                    "ridge_top1_fp32_mean": float(np.mean([r["ridge_top1_token_acc_fp32"] for r in rows])),
                    "ridge_top1_fp64_mean": float(np.mean([r["ridge_top1_token_acc_fp64"] for r in rows])),
                    "cand_recall_fp32": {
                        arm: {
                            slice_k: float(np.mean([r["candidate_recall_fp32"][arm][slice_k] for r in rows]))
                            for slice_k in ("overall", "pos0", "pos1_31")
                        }
                        for arm in ("forward_k32", "joint_true_k32", "forward_k128", "joint_true_k128")
                    },
                    "cand_recall_fp64": {
                        arm: {
                            slice_k: float(np.mean([r["candidate_recall_fp64"][arm][slice_k] for r in rows]))
                            for slice_k in ("overall", "pos0", "pos1_31")
                        }
                        for arm in ("forward_k32", "joint_true_k32", "forward_k128", "joint_true_k128")
                    },
                }

    # 4. Aggregate GPT-2 Calibration Curve & Beam Search at L=6 public across 10 seeds
    gpt2_curve_idx = {
        (int(x["depth"]), str(x["knowledge"]), int(x["cal_passages"]), int(x["victim_seed"])): x
        for x in gpt2_curve
    }
    gpt2_curve_summary = {}
    for C in (32, 64, 128, 256, 384):
        rows = [gpt2_curve_idx[(6, "public", C, s)] for s in gpt2_seeds]
        entry = {
            "cal_passages": C,
            "cal_tokens": C * 32,
            "cal_unique_tokens": int(rows[0]["cal_unique_tokens"]),
            "ridge_top1_fp32_mean": float(np.mean([r["ridge_top1_token_acc_fp32"] for r in rows])),
            "ridge_top1_fp64_mean": float(np.mean([r["ridge_top1_token_acc_fp64"] for r in rows])),
            "cand_recall_fp64": {
                arm: float(np.mean([r["candidate_recall_fp64"][arm]["overall"] for r in rows]))
                for arm in ("forward_k32", "joint_true_k32", "forward_k128", "joint_true_k128")
            },
            "beam_search": {},
        }
        for arm in ("forward_k32", "joint_true_k32", "forward_k128", "joint_true_k128"):
            if (6, "public", C, arm, gpt2_seeds[0]) in gpt2_beam:
                b_rows = [gpt2_beam[(6, "public", C, arm, s)] for s in gpt2_seeds]
                entry["beam_search"][arm] = {
                    "token_accuracy_mean": float(np.mean([b["summary"]["token_accuracy_mean"] for b in b_rows])),
                    "token_accuracy_seed_sd": float(np.std([b["summary"]["token_accuracy_mean"] for b in b_rows], ddof=1)),
                    "exact_match_mean": float(np.mean([b["summary"]["exact_match_mean"] for b in b_rows])),
                    "span8_rate_mean": float(np.mean([b["summary"]["span8_rate_mean"] for b in b_rows])),
                }
        gpt2_curve_summary[str(C)] = entry

    # 5. Aggregate Qwen3-0.6B Beam Search across (depth, knowledge, C, arm) with 100k two-way bootstrap CIs
    qwen_matrices = {}
    qwen_beam_summary = {}
    for (depth, know, C, arm, seed) in sorted(all_qwen_cells.keys()):
        if seed != qwen_seeds[0]:
            continue
        tok_mat = np.empty((8, 256), dtype=float)
        pos1_mat = np.empty((8, 256), dtype=float)
        doc_mat = np.empty((8, 256), dtype=float)
        span8_mat = np.empty((8, 256), dtype=float)
        cand_vals = []
        wall_vals = []
        for s_i, s in enumerate(qwen_seeds):
            c = all_qwen_cells[(depth, know, C, arm, s)]
            p = all_qwen_preds[(depth, know, C, arm, s)]
            for d_i, d in enumerate(p["per_document"]):
                tok_mat[s_i, d_i] = float(d["token_accuracy"])
                pos1_mat[s_i, d_i] = float(d["pos1_31_acc"])
                doc_mat[s_i, d_i] = 1.0 if d["sequence_exact"] else 0.0
                span8_mat[s_i, d_i] = float(d["span8_rate"] or 0.0)
            wall_vals.append(float(c["wall_seconds"]))
            if c["candidate_recall"] is not None:
                cand_vals.append(float(c["candidate_recall"]["overall"]))

        qwen_matrices[(depth, know, C, arm)] = {
            "tok": tok_mat,
            "pos1": pos1_mat,
            "doc": doc_mat,
            "span8": span8_mat,
        }
        qwen_beam_summary[f"L{depth}_{know}_C{C}_{arm}"] = {
            "depth": depth,
            "knowledge": know,
            "cal_passages": C,
            "cal_tokens": C * 32,
            "arm": arm,
            "token_accuracy": _two_way_bootstrap_ci(tok_mat, seed=93281 + depth * 100 + C),
            "pos1_31_accuracy": _two_way_bootstrap_ci(pos1_mat, seed=94281 + depth * 100 + C),
            "exact_match": _two_way_bootstrap_ci(doc_mat, seed=95281 + depth * 100 + C),
            "span8_rate": _two_way_bootstrap_ci(span8_mat, seed=96281 + depth * 100 + C),
            "candidate_recall_mean": float(np.mean(cand_vals)) if cand_vals else None,
            "wall_seconds_mean": float(np.mean(wall_vals)),
        }

    # 6. Compute Paired Bootstrap Contrasts for:
    # (a) Exact-grant k=128 at L in {1, 7, 14} (C=32)
    # (b) Calibration scaling at L=14 (C in {32, 64, 128, 256})
    contrasts = {}
    for depth in (1, 7, 14):
        for know in ("public", "exact"):
            for cname, left_k, right_k in [
                ("joint_k128_minus_forward_k128", (depth, know, 32, "joint_true_k128"), (depth, know, 32, "forward_k128")),
                ("forward_k128_minus_forward_k32", (depth, know, 32, "forward_k128"), (depth, know, 32, "forward_k32")),
                ("joint_k32_minus_forward_k128", (depth, know, 32, "joint_true_k32"), (depth, know, 32, "forward_k128")),
                ("joint_k128_minus_joint_k32", (depth, know, 32, "joint_true_k128"), (depth, know, 32, "joint_true_k32")),
            ]:
                if left_k in qwen_matrices and right_k in qwen_matrices:
                    contrasts[f"L{depth}_{know}_C32_{cname}"] = {
                        "token_diff": _two_way_bootstrap_ci(qwen_matrices[left_k]["tok"] - qwen_matrices[right_k]["tok"], seed=97001 + depth),
                        "exact_doc_diff": _two_way_bootstrap_ci(qwen_matrices[left_k]["doc"] - qwen_matrices[right_k]["doc"], seed=98001 + depth),
                    }

    for know in ("public", "exact"):
        for C in (64, 128, 256):
            for arm in ("forward_k32", "joint_true_k32", "forward_k128", "joint_true_k128"):
                k_c = (14, know, C, arm)
                k_32 = (14, know, 32, arm)
                if k_c in qwen_matrices and k_32 in qwen_matrices:
                    contrasts[f"L14_{know}_{arm}_C{C}_minus_C32"] = {
                        "token_diff": _two_way_bootstrap_ci(qwen_matrices[k_c]["tok"] - qwen_matrices[k_32]["tok"], seed=99001 + C),
                        "exact_doc_diff": _two_way_bootstrap_ci(qwen_matrices[k_c]["doc"] - qwen_matrices[k_32]["doc"], seed=99501 + C),
                    }
            if (14, know, C, "joint_true_k32") in qwen_matrices and (14, know, C, "forward_k32") in qwen_matrices:
                contrasts[f"L14_{know}_C{C}_joint_k32_minus_forward_k32"] = {
                    "token_diff": _two_way_bootstrap_ci(
                        qwen_matrices[(14, know, C, "joint_true_k32")]["tok"] - qwen_matrices[(14, know, C, "forward_k32")]["tok"],
                        seed=99701 + C,
                    ),
                    "exact_doc_diff": _two_way_bootstrap_ci(
                        qwen_matrices[(14, know, C, "joint_true_k32")]["doc"] - qwen_matrices[(14, know, C, "forward_k32")]["doc"],
                        seed=99801 + C,
                    ),
                }
            if (14, know, C, "joint_true_k128") in qwen_matrices and (14, know, C, "forward_k128") in qwen_matrices:
                contrasts[f"L14_{know}_C{C}_joint_k128_minus_forward_k128"] = {
                    "token_diff": _two_way_bootstrap_ci(
                        qwen_matrices[(14, know, C, "joint_true_k128")]["tok"] - qwen_matrices[(14, know, C, "forward_k128")]["tok"],
                        seed=99901 + C,
                    ),
                    "exact_doc_diff": _two_way_bootstrap_ci(
                        qwen_matrices[(14, know, C, "joint_true_k128")]["doc"] - qwen_matrices[(14, know, C, "forward_k128")]["doc"],
                        seed=99951 + C,
                    ),
                }

    summary_payload = {
        "schema": "paper-a-b.qwen3-cal-scale-summary.v1",
        "anchor_verifications": anchor_checks,
        "qwen3_cal_curve_summary": qwen_curve_summary,
        "gpt2_L6_public_cal_curve_summary": gpt2_curve_summary,
        "qwen3_beam_summary": qwen_beam_summary,
        "paired_bootstrap_contrasts": contrasts,
    }
    (output_dir / "qwen3_cal_scale_summary.json").write_text(json.dumps(summary_payload, indent=2) + "\n")
    return summary_payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--toula", type=Path, required=True)
    ap.add_argument("--koula", type=Path, required=True)
    ap.add_argument("--base-recon", type=Path, default=Path("output/paper-b-qwen3-recon-20261020"))
    ap.add_argument("--output", type=Path, default=Path("output/paper-a-qwen3-cal-scale-20261021"))
    args = ap.parse_args()
    res = analyze_cal_scale(args.toula, args.koula, args.base_recon, args.output)
    print("Analysis complete. Anchors verified:", len(res["anchor_verifications"]))


if __name__ == "__main__":
    main()
