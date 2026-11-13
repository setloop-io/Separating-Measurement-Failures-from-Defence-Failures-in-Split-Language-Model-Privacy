"""Statistical analysis for Paper A Experiment #3:
- Part A (R4 Major 3 & R6 Major 1): Fixed-Boundary Training-Placement Intervention on GPT-2
  (Regime A at L_obs = 1, b_train in {1, 3, 5, 7}; Regime B at L_obs = 6, b_train in {0, 2, 4, 6};
   8 victim seeds 11..88; D0 and D1; plus 8 all_local controls).
- Part B (R6 Minor 10): Qwen3-0.6B D1 8-Arm Sub-Component Isolation Factorial
  (seeds 66, 11, 22, 33 + cross-host verification on seed 66).
- Part C (R6 Major 3): Independent Longer-Passage Reconstruction at 32, 64, and 128 Tokens
  (255 held-out WikiText-2 articles >= 128 tokens, 10 victim seeds 101..1010, L = 6, public client grant,
   backward, forward_k32, joint_true_k32, forward_k128, joint_true_k128).
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


def _r2_start(matrix_8x4: np.ndarray) -> float:
    """Proportion of total variance across 8 seeds x 4 starts explained by start placement."""
    a = np.asarray(matrix_8x4, dtype=float)
    grand = a.mean(keepdims=True)
    full = a.mean(axis=0, keepdims=True)
    sse0 = float(((a - grand) ** 2).sum())
    sse1 = float(((a - full) ** 2).sum())
    return max(0.0, (sse0 - sse1) / sse0) if sse0 > 1e-24 else 0.0


def _within_seed_r2_start(matrix_8x4: np.ndarray) -> float:
    """Proportion of within-seed variance across 8 seeds x 4 starts explained by start placement."""
    a = np.asarray(matrix_8x4, dtype=float)
    seed_means = a.mean(axis=1, keepdims=True)
    centered = a - seed_means
    start_effect = centered.mean(axis=0, keepdims=True)
    sse0 = float((centered ** 2).sum())
    sse1 = float(((centered - start_effect) ** 2).sum())
    return max(0.0, (sse0 - sse1) / sse0) if sse0 > 1e-24 else 0.0


def _variance_test_1span(matrix_8x4: np.ndarray, replicates: int = 10000, seed: int = 20260917, alpha: float = 0.0125) -> dict:
    a = np.asarray(matrix_8x4, dtype=float)
    rng = np.random.default_rng(seed)
    actual_r2 = _r2_start(a)
    actual_within_r2 = _within_seed_r2_start(a)
    boot_r2 = np.empty(replicates)
    boot_within = np.empty(replicates)
    null_r2 = np.empty(replicates)

    for i in range(replicates):
        idx = rng.integers(len(a), size=len(a))
        boot_r2[i] = _r2_start(a[idx])
        boot_within[i] = _within_seed_r2_start(a[idx])
        perm = np.argsort(rng.random(a.shape), axis=1)
        null_r2[i] = _r2_start(np.take_along_axis(a, perm, axis=1))

    lo, hi = np.quantile(boot_r2, [0.025, 0.975])
    w_lo, w_hi = np.quantile(boot_within, [0.025, 0.975])
    p_val = (1 + int((null_r2 >= actual_r2 - 1e-12).sum())) / (replicates + 1)
    return {
        "partial_r2": float(actual_r2),
        "seed_bootstrap_95ci": [float(lo), float(hi)],
        "within_seed_r2": float(actual_within_r2),
        "within_seed_bootstrap_95ci": [float(w_lo), float(w_hi)],
        "permutation_p": float(p_val),
        "alpha_adjusted": float(alpha),
        "start_matters": bool(actual_r2 >= 0.10 and p_val < alpha),
    }


def _heldout_rmse_1span(matrix_8x4: np.ndarray, starts: list[int], seeds: list[int]) -> dict:
    a = np.asarray(matrix_8x4, dtype=float)
    fit = [i for i, s in enumerate(seeds) if s in [11, 22, 33, 44, 55, 66]]
    test = [i for i, s in enumerate(seeds) if s in [77, 88]]
    st = np.asarray(starts, dtype=float)
    errors = []
    for include_start in (False, True):
        x = np.column_stack([np.ones(len(st))] + ([st] if include_start else []))
        train_x = np.tile(x, (len(fit), 1))
        y = a[fit].reshape(-1)
        penalty = np.eye(x.shape[1]) * 1e-3
        penalty[0, 0] = 0.0
        coef = np.linalg.solve(train_x.T @ train_x + penalty, train_x.T @ y)
        pred = x @ coef
        rmse = float(np.sqrt(np.mean((a[test] - pred[None, :]) ** 2)))
        errors.append(rmse)
    return {
        "intercept_only_rmse": errors[0],
        "plus_start_rmse": errors[1],
        "ten_percent_improvement": bool(errors[1] <= 0.9 * errors[0] and errors[0] > 0),
    }


def _two_way_bootstrap_ci(matrix_s_d: np.ndarray, replicates: int = 20000, seed: int = 20261021) -> dict:
    """Two-way cluster bootstrap over S victim seeds x D documents."""
    a = np.asarray(matrix_s_d, dtype=float)
    n_s, n_d = a.shape
    mean_val = float(a.mean())
    seed_means = a.mean(axis=1)
    seed_sd = float(seed_means.std(ddof=1)) if n_s > 1 else 0.0
    rng = np.random.default_rng(seed)
    s_idx = rng.integers(0, n_s, size=(replicates, n_s))
    d_idx = rng.integers(0, n_d, size=(replicates, n_d))
    # Efficient two-way mean: average over sampled documents per seed, then over sampled seeds
    doc_boot = np.empty((replicates, n_s), dtype=float)
    for s in range(n_s):
        doc_boot[:, s] = a[s, d_idx].mean(axis=1)
    draws = np.take_along_axis(doc_boot, s_idx, axis=1).mean(axis=1)
    ci95 = np.quantile(draws, [0.025, 0.975])
    return {
        "mean": mean_val,
        "seed_sd": seed_sd,
        "ci95": [float(ci95[0]), float(ci95[1])],
    }


def analyze_exp3(toula_dir: Path, koula_dir: Path, output_dir: Path) -> dict:
    from attacker.evidence import verify_evidence

    output_dir.mkdir(parents=True, exist_ok=True)
    host_reports = {}
    for host, hdir in [("toula", toula_dir), ("koula", koula_dir)]:
        rpath = hdir / "exp3_replay.json"
        evpath = hdir / "exp3_replay.json.evidence.json"
        ok, problems = verify_evidence(evpath)
        if not ok:
            raise ValueError(f"Evidence verification failed for {host}: {problems}")
        r = json.loads(rpath.read_text())
        if r.get("status") != "EXP3_REPLAY_COMPLETE":
            raise ValueError(f"Host {host} status is not EXP3_REPLAY_COMPLETE: {r.get('status')}")
        host_reports[host] = (hdir, r)

    # =========================================================================
    # 1. PART A ANALYSIS: FIXED-BOUNDARY TRAINING-PLACEMENT INTERVENTION
    # =========================================================================
    part_a_controls = {}
    part_a_cells = {}
    part_a_sidecars = {}
    for host, (hdir, r) in host_reports.items():
        for ctrl in r["part_a_controls"]:
            part_a_controls[ctrl["seed"]] = ctrl
        for cell in r["part_a_cells"]:
            key = (cell["regime"], cell["l_obs"], cell["b_train"], cell["defense"], cell["seed"])
            if key in part_a_cells:
                raise ValueError(f"Duplicate Part A cell: {key}")
            sc_path = hdir / cell["sidecar_file"]
            if _digest(sc_path) != cell["sidecar_sha256"]:
                raise ValueError(f"Part A sidecar SHA256 mismatch: {sc_path}")
            part_a_cells[key] = cell
            part_a_sidecars[key] = json.loads(sc_path.read_text())

    seeds_a = [11, 22, 33, 44, 55, 66, 77, 88]
    if sorted(part_a_controls.keys()) != seeds_a:
        raise ValueError(f"Missing Part A controls: {sorted(part_a_controls.keys())}")

    regimes_spec = [
        ("Regime_A_L1_server_placement", 1, [1, 3, 5, 7]),
        ("Regime_B_L6_cross_seam_placement", 6, [0, 2, 4, 6]),
    ]
    part_a_summary = {}
    for regime_name, l_obs, starts in regimes_spec:
        part_a_summary[regime_name] = {"l_obs": l_obs, "starts": starts, "defenses": {}}
        for defense in ("D0", "D1"):
            metrics_names = [
                "delta_utility",
                "vmi_excess",
                "property_leakage",
                "normalised_leakage_index",
                "backward_token_accuracy",
                "forward_k32_token_accuracy",
                "joint_true_k32_token_accuracy",
            ]
            arrays = {m: np.empty((8, 4), dtype=float) for m in metrics_names}
            per_start_summary = []

            for k_idx, b_train in enumerate(starts):
                du_docs = np.empty((8, 256), dtype=float)
                vmi_docs = np.empty((8, 256), dtype=float)
                bwd_docs = np.empty((8, 256), dtype=float)
                fwd_docs = np.empty((8, 256), dtype=float)
                jnt_docs = np.empty((8, 256), dtype=float)
                prop_vals = []
                nli_vals = []
                ppl_vals = []

                for s_idx, seed in enumerate(seeds_a):
                    key = (regime_name, l_obs, b_train, defense, seed)
                    c = part_a_cells[key]
                    sc = part_a_sidecars[key]
                    for m in metrics_names:
                        arrays[m][s_idx, k_idx] = float(c[m])
                    du_docs[s_idx] = sc["delta_utility_per_doc"]
                    vmi_docs[s_idx] = sc["vmi_excess_per_doc"]
                    bwd_docs[s_idx] = sc["section4_reconstruction"]["backward"]["per_document_token_accuracy"]
                    fwd_docs[s_idx] = sc["section4_reconstruction"]["forward_k32"]["per_document_token_accuracy"]
                    jnt_docs[s_idx] = sc["section4_reconstruction"]["joint_true_k32"]["per_document_token_accuracy"]
                    prop_vals.append(float(c["property_leakage"]))
                    nli_vals.append(float(c["normalised_leakage_index"]))
                    ppl_vals.append(float(c["utility_after_perplexity"]))

                rng_p = np.random.default_rng(20260918 + l_obs * 100 + b_train * 10 + (0 if defense == "D0" else 1))
                prop_arr = np.asarray(prop_vals)
                nli_arr = np.asarray(nli_vals)
                s_boot = rng_p.integers(0, 8, size=(10000, 8))
                prop_ci = np.quantile(prop_arr[s_boot].mean(axis=1), [0.025, 0.975]).tolist()
                nli_ci = np.quantile(nli_arr[s_boot].mean(axis=1), [0.025, 0.975]).tolist()

                per_start_summary.append({
                    "b_train": b_train,
                    "trainable_range": f"{b_train},{b_train + 4}",
                    "delta_utility": _two_way_bootstrap_ci(du_docs, replicates=10000, seed=20260918 + k_idx),
                    "perplexity_mean": float(np.mean(ppl_vals)),
                    "vmi_excess": _two_way_bootstrap_ci(vmi_docs, replicates=10000, seed=20260928 + k_idx),
                    "property_leakage": {
                        "mean": float(prop_arr.mean()),
                        "seed_sd": float(prop_arr.std(ddof=1)),
                        "ci95": prop_ci,
                    },
                    "normalised_leakage_index": {
                        "mean": float(nli_arr.mean()),
                        "seed_sd": float(nli_arr.std(ddof=1)),
                        "ci95": nli_ci,
                    },
                    "backward_token_accuracy": _two_way_bootstrap_ci(bwd_docs, replicates=10000, seed=20260938 + k_idx),
                    "forward_k32_token_accuracy": _two_way_bootstrap_ci(fwd_docs, replicates=10000, seed=20260948 + k_idx),
                    "joint_true_k32_token_accuracy": _two_way_bootstrap_ci(jnt_docs, replicates=10000, seed=20260958 + k_idx),
                })

            variance_tests = {}
            for m in metrics_names:
                variance_tests[m] = {
                    "variance": _variance_test_1span(arrays[m], replicates=10000, seed=20260917, alpha=0.05 / 4),
                    "heldout_prediction": _heldout_rmse_1span(arrays[m], starts, seeds_a),
                }
            part_a_summary[regime_name]["defenses"][defense] = {
                "per_start": per_start_summary,
                "variance_and_prediction": variance_tests,
            }

    # =========================================================================
    # 2. PART B ANALYSIS: QWEN3-0.6B D1 SUB-COMPONENT ISOLATION FACTORIAL
    # =========================================================================
    part_b_by_host_cell = {}
    part_b_traces = {}
    for host, (hdir, r) in host_reports.items():
        for cell in r["part_b_cells"]:
            key = (host, cell["arm"], cell["seed"])
            tpath = hdir / cell["trace_file"]
            if _digest(tpath) != cell["trace_sha256"]:
                raise ValueError(f"Part B trace SHA256 mismatch: {tpath}")
            part_b_by_host_cell[key] = cell
            part_b_traces[key] = json.loads(tpath.read_text())

    # Cross-host verification on seed 66 for D1_full and D1_ste_fix
    cross_host_qwen_checks = []
    for arm_check in ("D1_full", "D1_ste_fix"):
        c_toula = part_b_by_host_cell[("toula", arm_check, 66)]
        c_koula = part_b_by_host_cell[("koula", arm_check, 66)]
        cross_host_qwen_checks.append({
            "arm": arm_check,
            "seed": 66,
            "toula_all_finite": c_toula["all_finite"],
            "koula_all_finite": c_koula["all_finite"],
            "toula_first_nonfinite_step": c_toula["first_nonfinite_step"],
            "koula_first_nonfinite_step": c_koula["first_nonfinite_step"],
            "toula_first_nonfinite_tensor": c_toula["first_nonfinite_tensor"],
            "koula_first_nonfinite_tensor": c_koula["first_nonfinite_tensor"],
            "toula_eval_ppl": c_toula["eval_perplexity"],
            "koula_eval_ppl": c_koula["eval_perplexity"],
        })

    qwen_arms_order = [
        "D0_baseline",
        "quant4_fwd_only",
        "quant4_fwd_bwd",
        "quant4_clip1_nonoise",
        "quant4_noclip_noise01",
        "quant8_clip1_noise01",
        "D1_full",
        "D1_ste_fix",
    ]
    qwen_seeds = [66, 11, 22, 33]
    part_b_arm_summaries = []
    for arm_name in qwen_arms_order:
        per_seed = []
        for s in qwen_seeds:
            host_owner = "toula" if s in (66, 11) else "koula"
            c = part_b_by_host_cell[(host_owner, arm_name, s)]
            per_seed.append(c)
        finite_count = sum(1 for c in per_seed if c["all_finite"])
        ppls = [c["eval_perplexity"] for c in per_seed if c["eval_perplexity"] is not None]
        part_b_arm_summaries.append({
            "arm": arm_name,
            "quant_bits": per_seed[0]["quant_bits"],
            "quant_grad": per_seed[0]["quant_grad"],
            "grad_clip": per_seed[0]["grad_clip"],
            "grad_noise": per_seed[0]["grad_noise"],
            "ste_fix": per_seed[0]["ste_fix"],
            "n_seeds": len(qwen_seeds),
            "n_finite": finite_count,
            "seed_66_all_finite": per_seed[0]["all_finite"],
            "seed_66_first_nonfinite_step": per_seed[0]["first_nonfinite_step"],
            "seed_66_first_nonfinite_stage": per_seed[0]["first_nonfinite_stage"],
            "seed_66_first_nonfinite_tensor": per_seed[0]["first_nonfinite_tensor"],
            "seed_66_max_down_proj_grad_norm_by_layer": per_seed[0]["max_down_proj_grad_norm_by_layer"],
            "seed_66_eval_perplexity": per_seed[0]["eval_perplexity"],
            "finite_eval_perplexity_mean": float(np.mean(ppls)) if ppls else None,
            "per_seed": per_seed,
        })

    # Extract step 95..102 trajectory on seed 66 across arms for diagnostic inspection
    seed66_step_window = {}
    for arm_name in qwen_arms_order:
        tr = part_b_traces[("toula", arm_name, 66)]["step_trace"]
        seed66_step_window[arm_name] = [row for row in tr if 94 <= row["step"] <= 102]

    # =========================================================================
    # 3. PART C ANALYSIS: INDEPENDENT LONGER-PASSAGE RECONSTRUCTION (32, 64, 128)
    # =========================================================================
    part_c_cells = {}
    part_c_preds = {}
    for host, (hdir, r) in host_reports.items():
        for cell in r["part_c_cells"]:
            key = (cell["seq_len"], cell["arm"], cell["victim_seed"])
            if key in part_c_cells:
                raise ValueError(f"Duplicate Part C cell: {key}")
            ppath = hdir / cell["prediction_file"]
            if _digest(ppath) != cell["prediction_sha256"]:
                raise ValueError(f"Part C prediction SHA256 mismatch: {ppath}")
            part_c_cells[key] = cell
            part_c_preds[key] = json.loads(ppath.read_text())

    seeds_c = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]
    arms_c = ["backward", "forward_k32", "joint_true_k32", "forward_k128", "joint_true_k128"]
    seq_lens_c = [32, 64, 128]

    part_c_by_len_arm = {}
    part_c_matrices = {}
    for sl in seq_lens_c:
        part_c_by_len_arm[str(sl)] = {}
        for arm in arms_c:
            tok_mat = np.empty((10, 255), dtype=float)
            pos0_mat = np.empty((10, 255), dtype=float)
            pos1_31_mat = np.empty((10, 255), dtype=float)
            pos0_31_mat = np.empty((10, 255), dtype=float)
            pos32_63_mat = np.empty((10, 255), dtype=float) if sl >= 64 else None
            pos64_127_mat = np.empty((10, 255), dtype=float) if sl >= 128 else None
            exact_mat = np.empty((10, 255), dtype=float)
            pref32_mat = np.empty((10, 255), dtype=float)
            span8_mat = np.empty((10, 255), dtype=float)
            lrun_mat = np.empty((10, 255), dtype=float)

            wall_vals = []
            cand_overall = []
            name_rates = []
            num_rates = []

            for s_idx, seed in enumerate(seeds_c):
                c = part_c_cells[(sl, arm, seed)]
                p = part_c_preds[(sl, arm, seed)]
                docs = p["per_document"]
                if len(docs) != 255:
                    raise ValueError(f"Expected 255 documents in Part C cell {(sl, arm, seed)}, got {len(docs)}")
                for d_idx, d in enumerate(docs):
                    tok_mat[s_idx, d_idx] = float(d["token_accuracy"])
                    pos0_mat[s_idx, d_idx] = float(d["pos0_acc"])
                    pos1_31_mat[s_idx, d_idx] = float(d["pos1_31_acc"])
                    pos0_31_mat[s_idx, d_idx] = float(d["pos0_31_acc"])
                    if sl >= 64:
                        pos32_63_mat[s_idx, d_idx] = float(d["pos32_63_acc"])
                    if sl >= 128:
                        pos64_127_mat[s_idx, d_idx] = float(d["pos64_127_acc"])
                    exact_mat[s_idx, d_idx] = 1.0 if d["sequence_exact"] else 0.0
                    pref32_mat[s_idx, d_idx] = 1.0 if d["prefix32_exact"] else 0.0
                    span8_mat[s_idx, d_idx] = float(d["contiguous_spans"]["8"]["rate"] or 0.0)
                    lrun_mat[s_idx, d_idx] = float(d["longest_exact_run"])

                wall_vals.append(float(c["work"]["wall_seconds"]))
                if c["candidate_recall"] is not None:
                    cand_overall.append(float(c["candidate_recall"]["overall"]))
                if c["summary"]["name_span_rate"] is not None:
                    name_rates.append(float(c["summary"]["name_span_rate"]))
                if c["summary"]["number_span_rate"] is not None:
                    num_rates.append(float(c["summary"]["number_span_rate"]))

            part_c_matrices[(sl, arm)] = {
                "token_accuracy": tok_mat,
                "pos0_31": pos0_31_mat,
                "pos32_63": pos32_63_mat,
                "pos64_127": pos64_127_mat,
                "exact_match": exact_mat,
            }

            part_c_by_len_arm[str(sl)][arm] = {
                "seq_len": sl,
                "arm": arm,
                "n_seeds": 10,
                "n_documents": 255,
                "total_tokens_per_seed": 255 * sl,
                "token_accuracy": _two_way_bootstrap_ci(tok_mat, replicates=20000, seed=20261021 + sl),
                "pos0_accuracy": _two_way_bootstrap_ci(pos0_mat, replicates=10000, seed=20261121 + sl),
                "pos1_31_accuracy": _two_way_bootstrap_ci(pos1_31_mat, replicates=10000, seed=20261221 + sl),
                "pos0_31_accuracy": _two_way_bootstrap_ci(pos0_31_mat, replicates=10000, seed=20261321 + sl),
                "pos32_63_accuracy": _two_way_bootstrap_ci(pos32_63_mat, replicates=10000, seed=20261421 + sl) if sl >= 64 else None,
                "pos64_127_accuracy": _two_way_bootstrap_ci(pos64_127_mat, replicates=10000, seed=20261521 + sl) if sl >= 128 else None,
                "exact_match": _two_way_bootstrap_ci(exact_mat, replicates=20000, seed=20261621 + sl),
                "prefix32_exact": _two_way_bootstrap_ci(pref32_mat, replicates=10000, seed=20261721 + sl),
                "span8_rate": _two_way_bootstrap_ci(span8_mat, replicates=10000, seed=20261821 + sl),
                "longest_exact_run": _two_way_bootstrap_ci(lrun_mat, replicates=10000, seed=20261921 + sl),
                "name_span_rate_mean": float(np.mean(name_rates)) if name_rates else None,
                "number_span_rate_mean": float(np.mean(num_rates)) if num_rates else None,
                "candidate_recall_mean": float(np.mean(cand_overall)) if cand_overall else None,
                "wall_seconds_mean": float(np.mean(wall_vals)),
                "amortized_seconds_per_document_mean": float(np.mean(wall_vals)) / 255.0,
            }

    # Compute paired Part C contrasts across sequence lengths
    part_c_contrasts = []
    contrast_pairs = [
        ("joint_true_k32", "forward_k32", "joint_k32_minus_forward_k32"),
        ("joint_true_k128", "forward_k128", "joint_k128_minus_forward_k128"),
        ("forward_k128", "forward_k32", "forward_k128_minus_forward_k32"),
        ("joint_true_k32", "forward_k128", "joint_k32_minus_forward_k128"),
    ]
    for sl in seq_lens_c:
        for left_arm, right_arm, cname in contrast_pairs:
            diff_tok = part_c_matrices[(sl, left_arm)]["token_accuracy"] - part_c_matrices[(sl, right_arm)]["token_accuracy"]
            diff_ex = part_c_matrices[(sl, left_arm)]["exact_match"] - part_c_matrices[(sl, right_arm)]["exact_match"]
            part_c_contrasts.append({
                "seq_len": sl,
                "contrast": cname,
                "left": left_arm,
                "right": right_arm,
                "token_accuracy_diff": _two_way_bootstrap_ci(diff_tok, replicates=20000, seed=20262000 + sl),
                "exact_match_diff": _two_way_bootstrap_ci(diff_ex, replicates=20000, seed=20262100 + sl),
            })

    # Position-slice drift contrasts at seq_len = 64 and 128
    position_drift_contrasts = []
    for sl in (64, 128):
        for arm in ("forward_k32", "joint_true_k32", "forward_k128", "joint_true_k128"):
            m = part_c_matrices[(sl, arm)]
            d_32_63 = m["pos32_63"] - m["pos0_31"]
            rec = {
                "seq_len": sl,
                "arm": arm,
                "pos32_63_minus_pos0_31": _two_way_bootstrap_ci(d_32_63, replicates=10000, seed=20263000 + sl),
            }
            if sl == 128:
                d_64_127 = m["pos64_127"] - m["pos0_31"]
                rec["pos64_127_minus_pos0_31"] = _two_way_bootstrap_ci(d_64_127, replicates=10000, seed=20263100 + sl)
            position_drift_contrasts.append(rec)

    result = {
        "schema": "paper-a.exp3-closure-analysis.v1",
        "hosts": {
            h: {
                "elapsed_seconds": r["elapsed_seconds"],
                "anchor_verifications": r["anchor_verifications"],
                "n_part_a_controls": len(r["part_a_controls"]),
                "n_part_a_cells": len(r["part_a_cells"]),
                "n_part_b_cells": len(r["part_b_cells"]),
                "n_part_c_cells": len(r["part_c_cells"]),
            }
            for h, (_, r) in host_reports.items()
        },
        "part_a_fixed_boundary_placement": {
            "controls_eval_loss_mean": float(np.mean([c["eval_loss_mean"] for c in part_a_controls.values()])),
            "controls_perplexity_mean": float(np.mean([c["utility_after"]["perplexity"] for c in part_a_controls.values()])),
            "regimes": part_a_summary,
        },
        "part_b_qwen3_d1_subcomponent_isolation": {
            "cross_host_seed66_checks": cross_host_qwen_checks,
            "arms": part_b_arm_summaries,
            "seed66_step94_to_102_window": seed66_step_window,
        },
        "part_c_longer_passages_32_64_128": {
            "by_length_and_arm": part_c_by_len_arm,
            "paired_contrasts": part_c_contrasts,
            "position_drift_contrasts": position_drift_contrasts,
        },
    }

    out_json = output_dir / "exp3_analysis.json"
    out_json.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--toula", type=Path, required=True)
    ap.add_argument("--koula", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    res = analyze_exp3(args.toula, args.koula, args.output)
    print(json.dumps({"status": "EXP3_ANALYSIS_COMPLETE", "output": str(args.output / "exp3_analysis.json")}, indent=2))


if __name__ == "__main__":
    main()
