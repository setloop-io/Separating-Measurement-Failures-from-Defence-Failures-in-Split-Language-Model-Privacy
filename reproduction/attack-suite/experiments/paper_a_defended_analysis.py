"""Statistical analysis for Paper A's defended joint and adaptive attack evaluation.

Analyses:
1. Primary Held-Out Checkpoint Defences (10 victim seeds 101..1010 x 256 passages = 2,560
   paired observations at L=6, public client grant, k=32, beam_width=2) across:
   - replica-median (K=3 coordinate-median channels)
   - secret-mixup (lambda=0.75 recycled-decoy activation mixup with victim-side gradient correction)
   - trained-bottleneck (d=768 -> k=64 learned bottleneck encoder with public cloud decoder)
   Uses a two-way crossed bootstrap (100,000 replicates, seed 20261020) over victim seeds
   and evaluation passages with Holm-Bonferroni and Bonferroni family-wise adjustments.
2. 500-Step Live Training Trajectory Defence under mid-lite (seeds 11, 22, 33 x 500 steps
   = 1,500 step observations at L=6, public client grant, k=32, beam_width=2), paired
   step-for-step with Experiment #1 (convergence-none-{11,22,33}).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "experiments/paper_a_stronger_frozen"
sys.path[:0] = [str(FROZEN / "experiments"), str(FROZEN)]
from attacker.evidence import verify_evidence


VICTIM_SEEDS = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]
TRAJ_SEEDS = [11, 22, 33]
HELDOUT_ARMS = [
    ("replica-median", "backward@replica-median"),
    ("replica-median", "forward_k32@replica-median"),
    ("replica-median", "joint_true_k32@replica-median"),
    ("secret-mixup", "backward@secret-mixup"),
    ("secret-mixup", "forward_k32@secret-mixup"),
    ("secret-mixup", "joint_true_k32@secret-mixup"),
    ("secret-mixup", "adaptive_demix_forward_k32@secret-mixup"),
    ("secret-mixup", "adaptive_demix_joint_k32@secret-mixup"),
    ("trained-bottleneck", "backward@trained-bottleneck"),
    ("trained-bottleneck", "forward_k32@trained-bottleneck"),
    ("trained-bottleneck", "joint_true_k32@trained-bottleneck"),
    ("trained-bottleneck", "adaptive_decode_backward@trained-bottleneck"),
    ("trained-bottleneck", "adaptive_decode_forward_k32@trained-bottleneck"),
    ("trained-bottleneck", "adaptive_decode_joint_k32@trained-bottleneck"),
]
HELDOUT_CONTRASTS = [
    ("replica_median_joint_minus_forward", "joint_true_k32@replica-median", "forward_k32@replica-median"),
    ("replica_median_forward_minus_backward", "forward_k32@replica-median", "backward@replica-median"),
    ("secret_mixup_joint_minus_forward", "joint_true_k32@secret-mixup", "forward_k32@secret-mixup"),
    ("secret_mixup_adaptive_forward_minus_forward", "adaptive_demix_forward_k32@secret-mixup", "forward_k32@secret-mixup"),
    ("secret_mixup_adaptive_joint_minus_forward", "adaptive_demix_joint_k32@secret-mixup", "forward_k32@secret-mixup"),
    ("secret_mixup_adaptive_joint_minus_joint", "adaptive_demix_joint_k32@secret-mixup", "joint_true_k32@secret-mixup"),
    ("secret_mixup_adaptive_joint_minus_adaptive_forward", "adaptive_demix_joint_k32@secret-mixup", "adaptive_demix_forward_k32@secret-mixup"),
    ("trained_bottleneck_raw_joint_minus_raw_forward", "joint_true_k32@trained-bottleneck", "forward_k32@trained-bottleneck"),
    ("trained_bottleneck_adaptive_forward_minus_raw_forward", "adaptive_decode_forward_k32@trained-bottleneck", "forward_k32@trained-bottleneck"),
    ("trained_bottleneck_adaptive_joint_minus_raw_forward", "adaptive_decode_joint_k32@trained-bottleneck", "forward_k32@trained-bottleneck"),
    ("trained_bottleneck_adaptive_joint_minus_adaptive_forward", "adaptive_decode_joint_k32@trained-bottleneck", "adaptive_decode_forward_k32@trained-bottleneck"),
]
MID_LITE_ARMS = [
    "backward@mid-lite",
    "forward_k32@mid-lite",
    "joint_true_k32@mid-lite",
]
WINDOWS = [
    ("steps_000_099", 0, 100),
    ("steps_100_199", 100, 200),
    ("steps_200_299", 200, 300),
    ("steps_300_399", 300, 400),
    ("steps_400_499", 400, 500),
]


def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _clopper_pearson_upper(k, n, alpha=0.05):
    if k == 0:
        return float(1.0 - (alpha / 2.0) ** (1.0 / n))
    return None


def _crossed_bootstrap_1d(mat, n_boot=100000, seed=20261020, alpha=0.05):
    """Two-way crossed bootstrap over (n_seeds, n_docs) matrix."""
    rng = np.random.default_rng(seed)
    n_s, n_d = mat.shape
    point = float(mat.mean())
    # Generate bootstrap means in chunks to keep memory small
    chunk_size = 5000
    means = np.empty(n_boot, dtype=np.float64)
    for start in range(0, n_boot, chunk_size):
        stop = min(n_boot, start + chunk_size)
        b = stop - start
        s_idx = rng.integers(0, n_s, size=(b, n_s))
        d_idx = rng.integers(0, n_d, size=(b, n_d))
        # First average over resampled seeds, or use crossed weights:
        # w_s[b, i] = count of seed i in replicate b; w_d[b, j] = count of doc j in replicate b
        w_s = np.zeros((b, n_s), dtype=np.float64)
        for col in range(n_s):
            np.add.at(w_s, (np.arange(b), s_idx[:, col]), 1.0 / n_s)
        w_d = np.zeros((b, n_d), dtype=np.float64)
        for col in range(n_d):
            np.add.at(w_d, (np.arange(b), d_idx[:, col]), 1.0 / n_d)
        # Replicate mean = sum_{i,j} w_s[b,i] * mat[i,j] * w_d[b,j]
        means[start:stop] = np.einsum("bi,ij,bj->b", w_s, mat, w_d)
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    se = float(means.std(ddof=1))
    p_less = float(np.mean(means <= 0.0))
    p_greater = float(np.mean(means >= 0.0))
    p_two = float(min(1.0, 2.0 * min(p_less, p_greater)))
    return {
        "estimate": point,
        "se": se,
        "ci95": [float(lo), float(hi)],
        "p_two_sided": p_two,
        "_replicates": means,
    }


def _stratified_step_bootstrap(mat, n_boot=100000, seed=20261020, alpha=0.05):
    """Stratified bootstrap over (3 seeds, n_steps) trajectory matrix."""
    rng = np.random.default_rng(seed)
    n_s, n_t = mat.shape
    point = float(mat.mean())
    chunk_size = 5000
    means = np.empty(n_boot, dtype=np.float64)
    for start in range(0, n_boot, chunk_size):
        stop = min(n_boot, start + chunk_size)
        b = stop - start
        acc = np.zeros(b, dtype=np.float64)
        for s in range(n_s):
            idx = rng.integers(0, n_t, size=(b, n_t))
            acc += mat[s][idx].mean(axis=1)
        means[start:stop] = acc / n_s
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    se = float(means.std(ddof=1))
    p_less = float(np.mean(means <= 0.0))
    p_greater = float(np.mean(means >= 0.0))
    p_two = float(min(1.0, 2.0 * min(p_less, p_greater)))
    return {
        "estimate": point,
        "se": se,
        "ci95": [float(lo), float(hi)],
        "p_two_sided": p_two,
        "_replicates": means,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--toula", type=Path, required=True)
    ap.add_argument("--koula", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    for host_name, path in (("toula", args.toula), ("koula", args.koula)):
        ev_path = Path(str(path) + ".evidence.json")
        ok, errors = verify_evidence(ev_path)
        if not ok:
            raise ValueError(f"Evidence verification failed for {host_name}: {errors}")

    r_toula = json.loads(args.toula.read_text())
    r_koula = json.loads(args.koula.read_text())
    if r_toula.get("status") != "DEFENDED_REPLAY_COMPLETE" or r_koula.get("status") != "DEFENDED_REPLAY_COMPLETE":
        raise ValueError("Both host reports must have status DEFENDED_REPLAY_COMPLETE")

    # Verify anchor checks
    anchors = r_toula.get("anchor_verifications", []) + r_koula.get("anchor_verifications", [])
    if len(anchors) != 4:
        raise ValueError(f"Expected 4 anchor verifications across hosts, got {len(anchors)}")
    for ac in anchors:
        if not ac.get("exact_predictions") or not ac.get("exact_candidates"):
            raise ValueError(f"Anchor verification failed: {ac}")

    # Collect held-out cells: (arm, seed) -> cell
    heldout_by_arm_seed = {}
    for cell in r_toula["heldout_cells"] + r_koula["heldout_cells"]:
        if cell.get("status") != "COMPLETE":
            raise ValueError(f"Incomplete held-out cell: {cell.get('cell_id')}")
        key = (cell["arm"], int(cell["victim_seed"]))
        if key in heldout_by_arm_seed:
            raise ValueError(f"Duplicate held-out cell: {key}")
        heldout_by_arm_seed[key] = cell

    for defense, arm in HELDOUT_ARMS:
        for seed in VICTIM_SEEDS:
            if (arm, seed) not in heldout_by_arm_seed:
                raise ValueError(f"Missing held-out cell: ({arm}, {seed})")

    # Build (10, 256) matrices per arm for token_accuracy, exact_match, named_entity_f1, number_f1, candidate_recall
    arm_matrices = {}
    heldout_summaries = {}
    for defense, arm in HELDOUT_ARMS:
        tok_mat = np.zeros((10, 256), dtype=np.float64)
        pos0_vals = []
        pos1_31_vals = []
        exact_mat = np.zeros((10, 256), dtype=np.float64)
        ent_mat = np.zeros((10, 256), dtype=np.float64)
        num_mat = np.zeros((10, 256), dtype=np.float64)
        cand_mat = np.zeros((10, 256), dtype=np.float64)
        has_cand = True
        wall_s = []
        fwd_seqs = []

        for s_idx, seed in enumerate(VICTIM_SEEDS):
            cell = heldout_by_arm_seed[(arm, seed)]
            docs = cell["metrics"]
            if len(docs) != 256:
                raise ValueError(f"Expected 256 docs in {cell['cell_id']}")
            for d_idx, d in enumerate(docs):
                tok_mat[s_idx, d_idx] = float(d["token_accuracy"])
                exact_mat[s_idx, d_idx] = 1.0 if d["sequence_exact"] else 0.0
                ent_mat[s_idx, d_idx] = float(d["longest_exact_run"])
                num_mat[s_idx, d_idx] = float(d["contiguous_spans"]["8"]["rate"] or 0.0)
            pos0_vals.append(float(cell["summary"]["pos0_accuracy_mean"]))
            pos1_31_vals.append(float(cell["summary"]["pos1_31_accuracy_mean"]))
            if cell.get("candidate_recall") and "per_document" in cell["candidate_recall"]:
                cand_mat[s_idx, :] = np.array(cell["candidate_recall"]["per_document"], dtype=np.float64)
            else:
                has_cand = False
            if cell["work"].get("wall_seconds", 0.0) > 0.0:
                wall_s.append(float(cell["work"]["wall_seconds"]))
            if cell["work"].get("head_forward_sequences", 0) > 0:
                fwd_seqs.append(int(cell["work"]["head_forward_sequences"]))

        arm_matrices[arm] = {
            "token_accuracy": tok_mat,
            "exact_match": exact_mat,
            "longest_exact_run": ent_mat,
            "span8_rate": num_mat,
            "candidate_recall": cand_mat if has_cand else None,
        }
        b_tok = _crossed_bootstrap_1d(tok_mat, n_boot=100000, seed=20261020)
        b_exact = _crossed_bootstrap_1d(exact_mat, n_boot=100000, seed=20261020)
        b_ent = _crossed_bootstrap_1d(ent_mat, n_boot=100000, seed=20261020)
        b_num = _crossed_bootstrap_1d(num_mat, n_boot=100000, seed=20261020)
        b_cand = _crossed_bootstrap_1d(cand_mat, n_boot=100000, seed=20261020) if has_cand else None

        exact_hits = int(round(exact_mat.sum()))
        heldout_summaries[arm] = {
            "defense": defense,
            "arm": arm,
            "n_victims": 10,
            "n_documents": 256,
            "n_observations": 2560,
            "token_accuracy": {k: v for k, v in b_tok.items() if k != "_replicates"},
            "pos0_accuracy_mean": float(np.mean(pos0_vals)),
            "pos1_31_accuracy_mean": float(np.mean(pos1_31_vals)),
            "candidate_recall": {k: v for k, v in b_cand.items() if k != "_replicates"} if b_cand else None,
            "exact_match": {
                **{k: v for k, v in b_exact.items() if k != "_replicates"},
                "exact_hits": exact_hits,
                "clopper_pearson_95_upper": _clopper_pearson_upper(exact_hits, 2560),
            },
            "longest_exact_run": {k: v for k, v in b_ent.items() if k != "_replicates"},
            "span8_rate": {k: v for k, v in b_num.items() if k != "_replicates"},
            "per_seed_token_accuracy": {str(seed): float(tok_mat[i].mean()) for i, seed in enumerate(VICTIM_SEEDS)},
            "mean_wall_seconds": float(np.mean(wall_s)) if wall_s else None,
            "mean_head_forward_sequences": float(np.mean(fwd_seqs)) if fwd_seqs else 0.0,
        }

    # Held-out contrasts (m = len(HELDOUT_CONTRASTS))
    m_contrasts = len(HELDOUT_CONTRASTS)
    alpha_bonf = 0.05 / m_contrasts
    heldout_contrasts = []
    for cid, left_arm, right_arm in HELDOUT_CONTRASTS:
        diff_tok = arm_matrices[left_arm]["token_accuracy"] - arm_matrices[right_arm]["token_accuracy"]
        res_tok = _crossed_bootstrap_1d(diff_tok, n_boot=100000, seed=20261020)
        reps = res_tok["_replicates"]
        lo_bonf, hi_bonf = np.quantile(reps, [alpha_bonf / 2.0, 1.0 - alpha_bonf / 2.0])

        diff_exact = arm_matrices[left_arm]["exact_match"] - arm_matrices[right_arm]["exact_match"]
        res_exact = _crossed_bootstrap_1d(diff_exact, n_boot=100000, seed=20261020)

        cand_diff_summary = None
        if arm_matrices[left_arm]["candidate_recall"] is not None and arm_matrices[right_arm]["candidate_recall"] is not None:
            diff_cand = arm_matrices[left_arm]["candidate_recall"] - arm_matrices[right_arm]["candidate_recall"]
            res_cand = _crossed_bootstrap_1d(diff_cand, n_boot=100000, seed=20261020)
            cand_diff_summary = {k: v for k, v in res_cand.items() if k != "_replicates"}

        seed_diffs = diff_tok.mean(axis=1)
        heldout_contrasts.append({
            "contrast_id": cid,
            "left_arm": left_arm,
            "right_arm": right_arm,
            "token_accuracy_diff": {
                "estimate": res_tok["estimate"],
                "se": res_tok["se"],
                "ci95_unadjusted": res_tok["ci95"],
                "ci95_bonferroni": [float(lo_bonf), float(hi_bonf)],
                "p_unadjusted": res_tok["p_two_sided"],
                "p_bonferroni": float(min(1.0, m_contrasts * res_tok["p_two_sided"])),
                "positive_seeds": int(np.sum(seed_diffs > 0)),
                "per_seed_diff": {str(seed): float(seed_diffs[i]) for i, seed in enumerate(VICTIM_SEEDS)},
            },
            "exact_match_diff": {k: v for k, v in res_exact.items() if k != "_replicates"},
            "candidate_recall_diff": cand_diff_summary,
        })

    # Holm-Bonferroni step-down p-values for heldout_contrasts
    order = np.argsort([c["token_accuracy_diff"]["p_unadjusted"] for c in heldout_contrasts])
    running_max = 0.0
    for rank, idx in enumerate(order):
        p_raw = heldout_contrasts[idx]["token_accuracy_diff"]["p_unadjusted"]
        p_holm = min(1.0, (m_contrasts - rank) * p_raw)
        running_max = max(running_max, p_holm)
        heldout_contrasts[idx]["token_accuracy_diff"]["p_holm"] = float(running_max)

    # Utility summary per defense across the 10 victim seeds
    utility_by_defense = {}
    for dv in r_toula["defended_victims"] + r_koula["defended_victims"]:
        dname = dv["defense"]
        utility_by_defense.setdefault(dname, []).append(dv)
    defense_utility_summary = {}
    for dname, rows in utility_by_defense.items():
        rows_sorted = sorted(rows, key=lambda x: x["seed"])
        ce_before = [float(r["utility"]["utility_before"]["cross_entropy"]) for r in rows_sorted]
        ce_after = [float(r["utility"]["utility_after"]["cross_entropy"]) for r in rows_sorted]
        ppl_after = [float(r["utility"]["utility_after"]["perplexity"]) for r in rows_sorted]
        train_s = [float(r["training_seconds"]) for r in rows_sorted]
        extra_info = {}
        if "fresh_vs_ref_wire_max_abs_diff" in rows_sorted[0]:
            extra_info["max_fresh_vs_ref_wire_abs_diff"] = float(max(r["fresh_vs_ref_wire_max_abs_diff"] for r in rows_sorted))
        if "public_decoder_vs_encoder_cal_top1" in rows_sorted[0]:
            extra_info["max_public_decoder_vs_encoder_cal_top1_diff"] = float(
                max(r["public_decoder_vs_encoder_cal_top1"]["abs_diff"] for r in rows_sorted)
            )
        defense_utility_summary[dname] = {
            "n_seeds": len(rows_sorted),
            "cross_entropy_before_mean": float(np.mean(ce_before)),
            "cross_entropy_after_mean": float(np.mean(ce_after)),
            "cross_entropy_after_sd": float(np.std(ce_after, ddof=1)),
            "perplexity_after_mean": float(np.mean(ppl_after)),
            "training_seconds_mean": float(np.mean(train_s)),
            **extra_info,
        }

    # =========================================================================
    # PART 2: 500-STEP LIVE TRAINING TRAJECTORY UNDER mid-lite (seeds 11, 22, 33)
    # =========================================================================
    ml_cells = {}
    for cell in r_toula["mid_lite_cells"] + r_koula["mid_lite_cells"]:
        key = (cell["arm"], int(cell["victim_seed"]), cell["host"])
        ml_cells[key] = cell

    ml_arm_step_tok = {}
    ml_arm_step_exact = {}
    ml_arm_step_cand = {}
    ml_summaries = {}

    for arm in MID_LITE_ARMS:
        tok_mat = np.zeros((3, 500), dtype=np.float64)
        exact_mat = np.zeros((3, 500), dtype=np.float64)
        cand_mat = np.zeros((3, 500), dtype=np.float64)
        has_cand = True
        pos0_vals = []
        pos1_31_vals = []
        wall_total = 0.0

        for s_idx, tseed in enumerate(TRAJ_SEEDS):
            c_t = ml_cells[(arm, tseed, "toula")]
            c_k = ml_cells[(arm, tseed, "koula")]
            steps_merged = c_t["step_metrics"] + c_k["step_metrics"]
            if len(steps_merged) != 500:
                raise ValueError(f"Expected 500 merged steps for {arm} seed {tseed}")
            for s, row in enumerate(steps_merged):
                if int(row["step"]) != s:
                    raise ValueError(f"Step alignment mismatch at {arm} seed {tseed} step {s}")
                tok_mat[s_idx, s] = float(row["token_accuracy"])
                exact_mat[s_idx, s] = 1.0 if row["sequence_exact"] else 0.0
                if "candidate_recall" in row:
                    cand_mat[s_idx, s] = float(row["candidate_recall"])
                else:
                    has_cand = False
            pos0_vals.append((c_t["summary"]["pos0_accuracy_mean"] * 248 + c_k["summary"]["pos0_accuracy_mean"] * 252) / 500.0)
            pos1_31_vals.append((c_t["summary"]["pos1_31_accuracy_mean"] * 248 + c_k["summary"]["pos1_31_accuracy_mean"] * 252) / 500.0)
            wall_total += float(c_t["work"]["wall_seconds"]) + float(c_k["work"]["wall_seconds"])

        ml_arm_step_tok[arm] = tok_mat
        ml_arm_step_exact[arm] = exact_mat
        ml_arm_step_cand[arm] = cand_mat if has_cand else None

        b_tok = _stratified_step_bootstrap(tok_mat, n_boot=100000, seed=20261020)
        b_exact = _stratified_step_bootstrap(exact_mat, n_boot=100000, seed=20261020)
        b_cand = _stratified_step_bootstrap(cand_mat, n_boot=100000, seed=20261020) if has_cand else None

        win_summary = {}
        for wname, w0, w1 in WINDOWS:
            win_summary[wname] = {
                "token_accuracy_mean": float(tok_mat[:, w0:w1].mean()),
                "exact_match_mean": float(exact_mat[:, w0:w1].mean()),
                "candidate_recall_mean": float(cand_mat[:, w0:w1].mean()) if has_cand else None,
            }

        ml_summaries[arm] = {
            "arm": arm,
            "n_seeds": 3,
            "n_steps_per_seed": 500,
            "n_observations": 1500,
            "token_accuracy": {k: v for k, v in b_tok.items() if k != "_replicates"},
            "pos0_accuracy_mean": float(np.mean(pos0_vals)),
            "pos1_31_accuracy_mean": float(np.mean(pos1_31_vals)),
            "candidate_recall": {k: v for k, v in b_cand.items() if k != "_replicates"} if b_cand else None,
            "exact_match": {
                **{k: v for k, v in b_exact.items() if k != "_replicates"},
                "exact_hits": int(round(exact_mat.sum())),
            },
            "per_seed_token_accuracy": {str(tseed): float(tok_mat[i].mean()) for i, tseed in enumerate(TRAJ_SEEDS)},
            "windows": win_summary,
            "early_0_49_mean": float(tok_mat[:, :50].mean()),
            "late_450_499_mean": float(tok_mat[:, 450:].mean()),
            "late_minus_early": float(tok_mat[:, 450:].mean() - tok_mat[:, :50].mean()),
            "total_wall_seconds": wall_total,
        }

    ml_contrasts = []
    for cid, left_arm, right_arm in (
        ("mid_lite_joint_minus_forward", "joint_true_k32@mid-lite", "forward_k32@mid-lite"),
        ("mid_lite_backward_minus_forward", "backward@mid-lite", "forward_k32@mid-lite"),
        ("mid_lite_joint_minus_backward", "joint_true_k32@mid-lite", "backward@mid-lite"),
    ):
        diff_mat = ml_arm_step_tok[left_arm] - ml_arm_step_tok[right_arm]
        res = _stratified_step_bootstrap(diff_mat, n_boot=100000, seed=20261020)
        win_diffs = {wname: float(diff_mat[:, w0:w1].mean()) for wname, w0, w1 in WINDOWS}
        ml_contrasts.append({
            "contrast_id": cid,
            "left_arm": left_arm,
            "right_arm": right_arm,
            "token_accuracy_diff": {k: v for k, v in res.items() if k != "_replicates"},
            "per_seed_diff": {str(tseed): float(diff_mat[i].mean()) for i, tseed in enumerate(TRAJ_SEEDS)},
            "windows_diff": win_diffs,
        })

    args.output.mkdir(parents=True, exist_ok=True)
    out_file = args.output / "defended_analysis.json"
    result = {
        "schema": "paper-a.defended-expansion-analysis.v1",
        "utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "toula_report": str(args.toula),
            "toula_sha256": _sha256(args.toula),
            "koula_report": str(args.koula),
            "koula_sha256": _sha256(args.koula),
        },
        "anchor_verifications": anchors,
        "defense_utility_summary": defense_utility_summary,
        "heldout_summaries": heldout_summaries,
        "heldout_contrasts": heldout_contrasts,
        "mid_lite_trajectory_summaries": ml_summaries,
        "mid_lite_trajectory_contrasts": ml_contrasts,
    }
    out_file.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {out_file}", flush=True)


if __name__ == "__main__":
    main()
