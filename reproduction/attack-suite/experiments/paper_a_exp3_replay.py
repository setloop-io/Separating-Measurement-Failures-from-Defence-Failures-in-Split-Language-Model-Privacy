"""Experiment #3 Replay for Paper A:
1. Part A (R4 Major 3 & R6 Major 1): Fixed-Boundary Training-Placement Intervention on GPT-2
   - Regime A (L_obs = 1 fixed, s = 4, server-segment starts b_train in {1, 3, 5, 7}, D0 & D1, 8 seeds 11..88)
   - Regime B (L_obs = 6 fixed, s = 4, cross-boundary starts b_train in {0, 2, 4, 6}, D0 & D1, 8 seeds 11..88)
   - Paired with 8 all_local full-model controls (seeds 11..88) and anchor-verified against original P6 v2 cells.
   - Measures both P6 leakage metrics (vmi_excess, property_auc, normalised_leakage_index, delta_utility)
     and Section 4 reconstruction metrics (backward_acc, forward_k32_acc, joint_true_k32_acc).
2. Part B (R6 Minor 10): Qwen3-0.6B D1 Sub-Component Isolation Factorial
   - 8-arm sub-component isolation on Qwen/Qwen3-0.6B (s = 4, b = 0, split = 1, trainable_range = "0,4",
     128 steps, FP32, seeds 66, 11, 22, 33):
     D0_baseline, quant4_fwd_only, quant4_fwd_bwd, quant4_clip1_nonoise,
     quant4_noclip_noise01, quant8_clip1_noise01, D1_full, D1_ste_fix.
   - Records per-step loss, seam activation max-abs, wire gradient norms, per-layer mlp.down_proj gradient norms,
     and exact first non-finite step/tensor.
3. Part C (R6 Major 3): Independent Longer-Passage Evaluation (32, 64, and 128 tokens)
   - Evaluates all 255 held-out WikiText-2 evaluation articles with >= 128 tokens at sequence lengths
     32, 64, and 128 across all 10 trained undefended victim checkpoints (seeds 101..1010, L = 6, public client grant).
   - Evaluates backward, forward_k32 (k=32, B=2), joint_true_k32 (k=32, B=2), forward_k128 (k=128, B=2),
     and joint_true_k128 (k=128, B=2), with position-slice accuracy (pos 0, pos 1..31, pos 32..63, pos 64..127),
     exact passage match, contiguous 8-token span rate, and named-entity/numeric span recovery.

Host partitioning:
- toula:
  * Part A: seeds [11, 22, 33, 44] (anchor seed 11 verified against p6_gpt2_v2_reference.json)
  * Part B: seeds [66, 11] across all 8 Qwen3 sub-component arms (16 runs)
  * Part C: victim seeds [101, 202, 303, 404, 505] (anchor seed 101 verified against frozen 32-token baseline)
- koula:
  * Part A: seeds [55, 66, 77, 88] (anchor seed 55 verified against p6_gpt2_v2_reference.json)
  * Part B: seeds [22, 33] across all 8 Qwen3 sub-component arms + seed 66 D1_full cross-host check (17 runs)
  * Part C: victim seeds [606, 707, 808, 909, 1010] (anchor seed 606 verified against frozen 32-token baseline)
"""
from __future__ import annotations

import copy
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import json
import math
from pathlib import Path
import platform
import random
import socket
import sys
import time
import traceback


HOSTS = ["toula", "koula"]
PART_A_HOST_SEEDS = {
    "toula": [11, 22, 33, 44],
    "koula": [55, 66, 77, 88],
}
PART_A_ANCHOR_SEED = {
    "toula": 11,
    "koula": 55,
}
PART_B_HOST_SEEDS = {
    "toula": [66, 11],
    "koula": [22, 33],
}
PART_C_HOST_SEEDS = {
    "toula": [101, 202, 303, 404, 505],
    "koula": [606, 707, 808, 909, 1010],
}
PART_C_ANCHOR_SEED = {
    "toula": 101,
    "koula": 606,
}

PROPERTY_TOKENS = [262, 11, 286]

QWEN_SUBCOMPONENT_ARMS = [
    {"arm": "D0_baseline", "quant_bits": 8, "quant_grad": True, "grad_clip": 0.0, "grad_noise": 0.0, "ste_fix": False},
    {"arm": "quant4_fwd_only", "quant_bits": 4, "quant_grad": False, "grad_clip": 0.0, "grad_noise": 0.0, "ste_fix": False},
    {"arm": "quant4_fwd_bwd", "quant_bits": 4, "quant_grad": True, "grad_clip": 0.0, "grad_noise": 0.0, "ste_fix": False},
    {"arm": "quant4_clip1_nonoise", "quant_bits": 4, "quant_grad": True, "grad_clip": 1.0, "grad_noise": 0.0, "ste_fix": False},
    {"arm": "quant4_noclip_noise01", "quant_bits": 4, "quant_grad": True, "grad_clip": 0.0, "grad_noise": 0.1, "ste_fix": False},
    {"arm": "quant8_clip1_noise01", "quant_bits": 8, "quant_grad": True, "grad_clip": 1.0, "grad_noise": 0.1, "ste_fix": False},
    {"arm": "D1_full", "quant_bits": 4, "quant_grad": True, "grad_clip": 1.0, "grad_noise": 0.1, "ste_fix": False},
    {"arm": "D1_ste_fix", "quant_bits": 4, "quant_grad": True, "grad_clip": 1.0, "grad_noise": 0.1, "ste_fix": True},
]


def _dump(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _score_long_predictions(evaluation_rows, predictions, metrics_module):
    """Compute per-document and position-sliced metrics for length 32, 64, or 128."""
    per_doc = []
    n = len(evaluation_rows)
    seq_len = len(evaluation_rows[0]["tokens"])
    pos0_hits = 0.0
    pos1_31_hits = 0.0
    pos0_31_hits = 0.0
    pos32_63_hits = 0.0
    pos64_127_hits = 0.0
    prefix32_exact = 0.0

    for row, pred in zip(evaluation_rows, predictions):
        truth = row["tokens"]
        m = metrics_module.reconstruction_metrics(truth, pred, row.get("annotations", []))
        p0 = 1.0 if truth[0] == pred[0] else 0.0
        p1_31 = sum(1.0 if a == b else 0.0 for a, b in zip(truth[1:32], pred[1:32])) / 31.0
        p0_31 = sum(1.0 if a == b else 0.0 for a, b in zip(truth[:32], pred[:32])) / 32.0
        pref32_ex = 1.0 if truth[:32] == pred[:32] else 0.0
        pos0_hits += p0
        pos1_31_hits += p1_31
        pos0_31_hits += p0_31
        prefix32_exact += pref32_ex

        doc_slices = {
            "pos0_acc": p0,
            "pos1_31_acc": p1_31,
            "pos0_31_acc": p0_31,
            "prefix32_exact": bool(truth[:32] == pred[:32]),
        }
        if seq_len >= 64:
            p32_63 = sum(1.0 if a == b else 0.0 for a, b in zip(truth[32:64], pred[32:64])) / 32.0
            pos32_63_hits += p32_63
            doc_slices["pos32_63_acc"] = p32_63
        if seq_len >= 128:
            p64_127 = sum(1.0 if a == b else 0.0 for a, b in zip(truth[64:128], pred[64:128])) / 64.0
            pos64_127_hits += p64_127
            doc_slices["pos64_127_acc"] = p64_127

        per_doc.append({
            "document_id": row["id"],
            "index_in_256": row.get("index_in_256"),
            **m,
            **doc_slices,
        })

    tok_acc = sum(d["token_accuracy"] for d in per_doc) / n
    exact = sum(1.0 if d["sequence_exact"] else 0.0 for d in per_doc) / n
    longest_run = sum(float(d["longest_exact_run"]) for d in per_doc) / n
    span8 = sum(float(d["contiguous_spans"]["8"]["rate"] or 0.0) for d in per_doc) / n
    name_exact = sum(int(d["annotated_spans"]["name"]["exact"]) for d in per_doc)
    name_total = sum(int(d["annotated_spans"]["name"]["total"]) for d in per_doc)
    num_exact = sum(int(d["annotated_spans"]["number"]["exact"]) for d in per_doc)
    num_total = sum(int(d["annotated_spans"]["number"]["total"]) for d in per_doc)

    summary = {
        "n_documents": n,
        "seq_len": seq_len,
        "token_accuracy_mean": tok_acc,
        "pos0_accuracy_mean": pos0_hits / n,
        "pos1_31_accuracy_mean": pos1_31_hits / n,
        "pos0_31_accuracy_mean": pos0_31_hits / n,
        "pos32_63_accuracy_mean": (pos32_63_hits / n) if seq_len >= 64 else None,
        "pos64_127_accuracy_mean": (pos64_127_hits / n) if seq_len >= 128 else None,
        "prefix32_exact_mean": prefix32_exact / n,
        "exact_match_mean": exact,
        "sequence_exact_mean": exact,
        "longest_exact_run_mean": longest_run,
        "span8_rate_mean": span8,
        "name_span_rate": (name_exact / name_total) if name_total > 0 else None,
        "number_span_rate": (num_exact / num_total) if num_total > 0 else None,
        "name_spans_total": name_total,
        "number_spans_total": num_total,
    }
    return per_doc, summary


def _candidate_recall_long(batches_details, truth_tensor):
    import torch
    cands = [torch.tensor(b["candidate_ids"], dtype=torch.long) for b in batches_details if "candidate_ids" in b]
    if not cands:
        return None
    all_cands = torch.cat(cands, dim=0)
    truth_cpu = truth_tensor.detach().cpu()
    hits = (all_cands == truth_cpu.unsqueeze(-1)).any(dim=-1).to(dtype=torch.float32)
    seq_len = hits.shape[1]
    out = {
        "overall": float(hits.mean().item()),
        "pos0": float(hits[:, 0].mean().item()),
        "pos1_31": float(hits[:, 1:32].mean().item()),
        "pos0_31": float(hits[:, :32].mean().item()),
        "pos32_63": float(hits[:, 32:64].mean().item()) if seq_len >= 64 else None,
        "pos64_127": float(hits[:, 64:128].mean().item()) if seq_len >= 128 else None,
    }
    return out


class _NullCaptureWriter:
    """In-memory stand-in for CaptureWriter during training when step captures are not needed."""
    def __init__(self, directory):
        self.dir = str(directory)
        self.count = 0

    def write_training(self, tensor, mb_id, phase, step, epoch):
        self.count += 1


def run_exp3_replay(plan, root, output, resolve_asset, digest):
    """Execute Parts A, B, and C of Experiment #3 on the assigned host."""
    root, output = Path(root), Path(output)
    host = socket.gethostname().split(".")[0].lower()
    if host not in HOSTS:
        raise ValueError(f"Unexpected execution host: {host}")

    part_a_seeds = PART_A_HOST_SEEDS[host]
    part_a_anchor = PART_A_ANCHOR_SEED[host]
    part_b_seeds = PART_B_HOST_SEEDS[host]
    part_c_seeds = PART_C_HOST_SEEDS[host]
    part_c_anchor = PART_C_ANCHOR_SEED[host]

    output.mkdir(parents=True, exist_ok=False)
    (output / "part_a_predictions").mkdir(parents=True, exist_ok=True)
    (output / "part_b_traces").mkdir(parents=True, exist_ok=True)
    (output / "part_c_predictions").mkdir(parents=True, exist_ok=True)
    (output / "scratch").mkdir(parents=True, exist_ok=True)
    report_path = output / "exp3_replay.json"
    started = time.monotonic()

    # Ensure ROOT comes first on sys.path so attacker.splitbench includes P6 quant/Qwen3 support
    for mod_name in list(sys.modules.keys()):
        if mod_name == "attacker" or mod_name.startswith("attacker.") or mod_name == "validate_reconstruction":
            del sys.modules[mod_name]
    sys.path[:0] = [str(root), str(root / "experiments")]

    from attacker.evidence import EvidenceLogger, verify_evidence

    report = {
        "schema": "paper-a.exp3-closure-replay.v1",
        "status": "RUNNING",
        "phase": plan["phase"],
        "host": host,
        "part_a_seeds": part_a_seeds,
        "part_a_anchor_seed": part_a_anchor,
        "part_b_seeds": part_b_seeds,
        "part_c_seeds": part_c_seeds,
        "part_c_anchor_seed": part_c_anchor,
        "utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": plan["source_commit"],
        "runner_sha256": digest(__file__),
        "plan": plan,
        "anchor_verifications": [],
        "part_a_controls": [],
        "part_a_cells": [],
        "part_b_cells": [],
        "part_c_cells": [],
        "failed_cells": [],
    }
    _dump(report_path, report)

    with EvidenceLogger(report_path, node_role=host + "/gpu", extra={"study": plan["study"], "phase": plan["phase"]}) as ev:
        try:
            import torch
            import torch.nn.functional as F
            import transformers
            from transformers import AutoModelForCausalLM, GPT2LMHeadModel
            from attacker import splitbench as sb
            from attacker.attacks import seam_inversion as si
            from attacker.attacks.label_scoring import _sequence_norms_torch
            from attacker.attacks.lira_membership import rank_auc
            import attacker.reconstruction_baselines as baselines
            import attacker.reconstruction_metrics as metrics_module
            from experiments.pilot_measurements import evaluation_state, utility as measure_utility
            from validate_reconstruction import attack_observations

            report["module_Paths_and_sha256"] = {
                "splitbench": {"path": str(Path(sb.__file__).resolve()), "sha256": digest(sb.__file__)},
                "seam_inversion": {"path": str(Path(si.__file__).resolve()), "sha256": digest(si.__file__)},
                "reconstruction_baselines": {"path": str(Path(baselines.__file__).resolve()), "sha256": digest(baselines.__file__)},
                "reconstruction_metrics": {"path": str(Path(metrics_module.__file__).resolve()), "sha256": digest(metrics_module.__file__)},
            }

            torch.set_num_threads(1)
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable for Experiment #3 replay")

            report["environment"] = {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "memory_total_bytes": torch.cuda.get_device_properties(0).total_memory,
            }

            exp3_dir = Path(plan["exp3_directory"])
            exp3_pins_path = exp3_dir / "PINS.json"
            exp3_pins = json.loads(exp3_pins_path.read_text())
            report["exp3_pins_sha256"] = digest(exp3_pins_path)
            for fname, sha in exp3_pins.items():
                fpath = exp3_dir / fname
                if digest(fpath) != sha:
                    raise ValueError(f"Exp3 asset hash mismatch for {fname}: {digest(fpath)} != {sha}")

            input_path = Path(resolve_asset(plan, "input.json", plan["input_sha256"]))
            if digest(input_path) != plan["input_sha256"]:
                raise ValueError("Primary input.json hash mismatch")
            gpt2_data = json.loads(input_path.read_text())
            qwen_data = json.loads((exp3_dir / "qwen_input.json").read_text())
            p6_ref = json.loads((exp3_dir / "p6_gpt2_v2_reference.json").read_text())
            longer_data = json.loads((exp3_dir / "longer_passages_32_64_128.json").read_text())

            gpt2_snap = Path.home() / ".cache/huggingface/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e"
            qwen_snap = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
            if not gpt2_snap.is_dir():
                raise FileNotFoundError(f"Missing GPT-2 snapshot: {gpt2_snap}")
            if not qwen_snap.is_dir():
                raise FileNotFoundError(f"Missing Qwen3-0.6B snapshot: {qwen_snap}")

            ev.event("exp3_replay_start", {k: v for k, v in report.items() if not isinstance(v, list)})
            _dump(report_path, report)

            # =========================================================================
            # PART A: GPT-2 FIXED-BOUNDARY TRAINING-PLACEMENT INTERVENTION
            # (R4 Major 3 & R6 Major 1)
            # =========================================================================
            # Keep PyTorch default TF32 state (matmul False, cudnn True) during Part A & Part B to match run_p6_corrected.py
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = True

            base_gpt2_eager = AutoModelForCausalLM.from_pretrained(
                str(gpt2_snap), torch_dtype=torch.float32, attn_implementation="eager", local_files_only=True
            )
            cal_ids_gpt2 = torch.tensor([r["tokens"] for r in gpt2_data["calibration"]], dtype=torch.long, device="cuda")
            eval_rows_gpt2 = gpt2_data["evaluation"]
            eval_ids_gpt2 = torch.tensor([r["tokens"] for r in eval_rows_gpt2], dtype=torch.long, device="cuda")
            eval_batches_gpt2 = [torch.tensor([r["tokens"]], dtype=torch.long) for r in eval_rows_gpt2]
            majority_gpt2 = si.majority_control_token(cal_ids_gpt2.tolist())
            prop_labels_gpt2 = [[t in r["tokens"] for t in PROPERTY_TOKENS] for r in eval_rows_gpt2]

            # Precompute public heads and Ridge decoders at L_obs = 1 and L_obs = 6 for Part A
            pub_heads_part_a = {}
            for l_obs in (1, 6):
                pub_model = copy.deepcopy(base_gpt2_eager).to("cuda").eval()
                pub_model.requires_grad_(False)
                pub_split = sb.build_split(pub_model, sb.SplitPlan.parse(str(l_obs), n_layers=12))
                pub_E = si._embedding_of_model(pub_model).detach().to("cuda")
                with torch.no_grad():
                    h_cal = si._head_rows(pub_split, cal_ids_gpt2).reshape(-1, pub_E.shape[1])
                    e_cal = pub_E[cal_ids_gpt2.reshape(-1)]
                    W_dec, lam_used = si._ridge_fit(h_cal, e_cal, 0.0)
                # Switch attacker model to sdpa after eager calibration (matching Section 4 validate_reconstruction.py)
                pub_model.set_attn_implementation("sdpa")
                metered = baselines.MeteredHead(pub_split)
                pub_heads_part_a[l_obs] = {
                    "split": pub_split,
                    "E": pub_E,
                    "W_dec": W_dec,
                    "ridge_lambda": lam_used,
                    "head": metered,
                }
            forward_k32_frozen_cache = {}

            orig_beam_reconstruct = baselines.beam_reconstruct

            @torch.no_grad()
            def _kv_beam_reconstruct(head, h, candidates, beam_width=2):
                split = head.split
                start_blk, end_blk = split.ranges[0]
                blocks = split.blocks[start_blk:end_blk]
                n_blocks = len(blocks)
                n, length, count = candidates.shape
                if h.shape[:2] != (n, length) or beam_width < 1:
                    raise ValueError("invalid beam shape/budget")
                prefixes = torch.empty(n, 1, 0, dtype=torch.long, device=h.device)
                scores = torch.zeros(n, 1, device=h.device)
                past_k = [None] * n_blocks
                past_v = [None] * n_blocks
                for pos in range(length):
                    width = prefixes.shape[1]
                    seq_count = n * width * count
                    head.work.head_forward_calls += 1
                    head.work.head_forward_sequences += seq_count
                    head.work.head_forward_token_positions += seq_count * (pos + 1)
                    cand_tok = candidates[:, None, pos, :].expand(n, width, count).reshape(seq_count, 1)
                    pos_id = torch.tensor([[pos]], dtype=torch.long, device=h.device)
                    x = split.drop(split.wte(cand_tok) + split.wpe(pos_id))
                    new_k_list = []
                    new_v_list = []
                    for l_idx, blk in enumerate(blocks):
                        attn = blk.attn
                        n_heads = attn.num_heads
                        head_dim = attn.head_dim
                        ln1_x = blk.ln_1(x)
                        q, k, v = attn.c_attn(ln1_x).split(attn.split_size, dim=2)
                        q_h = q.view(n * width, count, n_heads, head_dim).transpose(1, 2)
                        k_h = k.view(n * width, count, n_heads, head_dim).transpose(1, 2)
                        v_h = v.view(n * width, count, n_heads, head_dim).transpose(1, 2)
                        if past_k[l_idx] is None:
                            attn_out = v_h.transpose(1, 2).reshape(seq_count, 1, n_heads * head_dim)
                        else:
                            pk = past_k[l_idx].reshape(n * width, n_heads, pos, head_dim)
                            pv = past_v[l_idx].reshape(n * width, n_heads, pos, head_dim)
                            scale = 1.0 / (head_dim ** 0.5)
                            scores_past = torch.matmul(q_h, pk.transpose(-1, -2)) * scale
                            scores_self = (q_h * k_h).sum(dim=-1, keepdim=True) * scale
                            attn_weights = torch.softmax(torch.cat([scores_past, scores_self], dim=-1), dim=-1)
                            out_past = torch.matmul(attn_weights[..., :pos], pv)
                            out_self = attn_weights[..., pos:] * v_h
                            attn_out = (out_past + out_self).transpose(1, 2).reshape(seq_count, 1, n_heads * head_dim)
                        x = x + attn.resid_dropout(attn.c_proj(attn_out))
                        x = x + blk.mlp(blk.ln_2(x))
                        new_k_list.append(k_h.transpose(1, 2).reshape(n, width * count, n_heads, 1, head_dim))
                        new_v_list.append(v_h.transpose(1, 2).reshape(n, width * count, n_heads, 1, head_dim))
                    hidden = x.reshape(n, width * count, -1)
                    residual = (hidden - h[:, None, pos]).square().sum(-1)
                    total = residual + scores.repeat_interleave(count, dim=1)
                    scores, chosen = total.topk(min(beam_width, width * count), largest=False, dim=1)
                    expanded = prefixes[:, :, None, :].expand(n, width, count, pos)
                    tokens = candidates[:, None, pos, :, None].expand(n, width, count, 1)
                    extensions = torch.cat((expanded, tokens), dim=-1).reshape(n, width * count, pos + 1)
                    prefixes = extensions.gather(1, chosen[..., None].expand(-1, -1, pos + 1))
                    for l_idx in range(n_blocks):
                        nk = new_k_list[l_idx]
                        nv = new_v_list[l_idx]
                        gather_idx = chosen[:, :, None, None, None].expand(n, chosen.shape[1], n_heads, 1, head_dim)
                        ck = nk.gather(1, gather_idx)
                        cv = nv.gather(1, gather_idx)
                        if past_k[l_idx] is None:
                            past_k[l_idx] = ck
                            past_v[l_idx] = cv
                        else:
                            parent_idx = (chosen // count)[:, :, None, None, None].expand(n, chosen.shape[1], n_heads, pos, head_dim)
                            past_k[l_idx] = torch.cat([past_k[l_idx].gather(1, parent_idx), ck], dim=3)
                            past_v[l_idx] = torch.cat([past_v[l_idx].gather(1, parent_idx), cv], dim=3)
                return prefixes[:, 0], {"forward_residual_sum": scores[:, 0].tolist()}

            # Verify on-GPU mathematical equivalence of _kv_beam_reconstruct vs orig_beam_reconstruct at L=6
            with torch.no_grad():
                p6_info = pub_heads_part_a[6]
                test_ids = eval_ids_gpt2[:8]
                test_h = p6_info["split"].head_forward(test_ids)
                test_fwd = baselines.forward_scores(test_h, p6_info["E"], p6_info["W_dec"], p6_info["head"])
                test_cands = baselines.candidate_ids(test_fwd, None, 32, "forward")
                p_orig, d_orig = orig_beam_reconstruct(p6_info["head"], test_h, test_cands, beam_width=2)
                p_kv, d_kv = _kv_beam_reconstruct(p6_info["head"], test_h, test_cands, beam_width=2)
                max_res_diff = max(abs(a - b) for a, b in zip(d_orig["forward_residual_sum"], d_kv["forward_residual_sum"]))
                if not bool(torch.equal(p_orig, p_kv)) or max_res_diff > 1e-2:
                    raise RuntimeError(f"KV beam equivalence check failed: equal={torch.equal(p_orig, p_kv)}, max_res_diff={max_res_diff}")
                kv_eq_rec = {
                    "part": "KV_Beam_Equivalence",
                    "l_obs": 6,
                    "n_docs": 8,
                    "predictions_exact_equal": True,
                    "max_residual_abs_diff": max_res_diff,
                    "verified": True,
                }
                report["anchor_verifications"].append(kv_eq_rec)
                ev.event("kv_beam_equivalence_verified", kv_eq_rec)
            baselines.beam_reconstruct = _kv_beam_reconstruct

            def run_part_a_unit(kind, regime, l_obs, span, b_train, defense, seed):
                t0 = time.perf_counter()
                control = (kind == "all_local")
                cfg = sb.BenchConfig(
                    model=str(gpt2_snap),
                    split="1" if control else str(l_obs),
                    mode="finetune",
                    steps=128,
                    seq_len=32,
                    batch_size=1,
                    lr=5e-5,
                    seed=seed,
                    device="cuda",
                    dtype="fp32",
                    defense="none" if control else "quant",
                    quant_bits=8 if defense == "D0" else 4,
                    quant_grad=not control,
                    grad_clip=1.0 if (not control and defense == "D1") else 0.0,
                    grad_noise=0.1 if (not control and defense == "D1") else 0.0,
                    trainable_range=None if control else f"{b_train},{b_train + span}",
                    capture_dir=str(output / "scratch"),
                    output=str(output / "scratch" / "tmp.json"),
                    save_canonical=False,
                )
                cfg.validate()
                torch.manual_seed(seed)
                model = copy.deepcopy(base_gpt2_eager).to("cuda")
                split = sb.build_split(model, sb.SplitPlan.parse(cfg.split, n_layers=12))
                perm = torch.randperm(len(gpt2_data["train"]), generator=torch.Generator().manual_seed(seed)).tolist()
                batches = [torch.tensor([gpt2_data["train"][i]["tokens"]], dtype=torch.long) for i in perm]

                measured = {}
                def measure_cb(stage, s_split, s_cfg, s_state):
                    measured[stage] = measure_utility(s_split, s_cfg, s_state, eval_batches_gpt2, [])
                    if stage == "after":
                        losses_doc = []
                        norms_doc = []
                        wires_fp16 = []
                        obs_fp16 = []
                        with evaluation_state(s_split, s_state):
                            torch.manual_seed(s_cfg.seed + 90317)
                            for i, row in enumerate(eval_rows_gpt2):
                                ids = torch.tensor([row["tokens"]], dtype=torch.long, device=s_split.device)
                                with torch.no_grad():
                                    hidden = s_split.head_forward(ids)
                                    sent = sb._apply_seam_defense(hidden, s_cfg, s_state)
                                wire = sent.detach().requires_grad_(True)
                                logits = sb._server_rest_forward(s_split, wire, s_cfg, s_state)
                                loss_val = F.cross_entropy(
                                    logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                                    ids[:, 1:].reshape(-1),
                                )
                                grad = torch.autograd.grad(loss_val, wire)[0]
                                # Apply return-path defense on CPU so torch.manual_seed(seed+90317) CPU RNG stream matches run_p6_corrected.py
                                observed = sb._apply_grad_defense(grad.detach().cpu(), s_cfg)
                                w16 = wire.detach().to("cpu", torch.float16)
                                o16 = observed.detach().to("cpu", torch.float16)
                                norm_val = _sequence_norms_torch(o16)[0]
                                losses_doc.append(float(loss_val.detach().item()))
                                norms_doc.append(float(norm_val))
                                wires_fp16.append(w16)
                                obs_fp16.append(o16)
                        measured["losses"] = losses_doc
                        measured["norms"] = norms_doc
                        measured["wires_fp32"] = torch.cat(wires_fp16, dim=0).to(device="cuda", dtype=torch.float32)
                        measured["obs_fp32"] = torch.cat(obs_fp16, dim=0).to(device="cuda", dtype=torch.float32)

                dummy_writer = _NullCaptureWriter(output / "scratch")
                orig_apply_seam = sb._apply_seam_defense
                if l_obs == 6 and not control:
                    def _ste_seam_defense(hidden_t, c_cfg, c_state=None):
                        q_t = orig_apply_seam(hidden_t, c_cfg, c_state)
                        return (hidden_t + (q_t - hidden_t).detach()) if hidden_t.requires_grad else q_t
                    sb._apply_seam_defense = _ste_seam_defense
                try:
                    train_stats = sb.finetune(split, cfg, batches, dummy_writer, on_step=None, measurement=measure_cb)
                finally:
                    sb._apply_seam_defense = orig_apply_seam

                # Compute VMI excess and property AUC exactly as in analyze_p6_corrected.py
                pub_info = pub_heads_part_a[1 if control else l_obs]
                rows_h = measured["wires_fp32"]
                rows_g = measured["obs_fp32"]
                with torch.no_grad():
                    ehat = rows_h.reshape(-1, pub_info["E"].shape[1]) @ pub_info["W_dec"]
                    pred_vmi = si._cosine_nn_torch(ehat, pub_info["E"]).reshape(len(eval_rows_gpt2), 32).cpu().tolist()

                vmi_excess_list = []
                vmi_acc_list = []
                for row, p_seq in zip(eval_rows_gpt2, pred_vmi):
                    t_seq = row["tokens"]
                    corr = sum(1 for a, b in zip(p_seq, t_seq) if a == b)
                    ctl_corr = sum(1 for b in t_seq if b == majority_gpt2)
                    vmi_excess_list.append((corr - ctl_corr) / 32.0)
                    vmi_acc_list.append(corr / 32.0)

                aucs = []
                for j in range(len(PROPERTY_TOKENS)):
                    present = [x for x, l in zip(measured["norms"], prop_labels_gpt2) if l[j]]
                    absent = [x for x, l in zip(measured["norms"], prop_labels_gpt2) if not l[j]]
                    aucs.append(float(rank_auc(present, absent)))

                vmi_excess_mean = sum(vmi_excess_list) / len(vmi_excess_list)
                vmi_acc_mean = sum(vmi_acc_list) / len(vmi_acc_list)
                prop_leakage = max(2.0 * abs(x - 0.5) for x in aucs)
                nli = max(0.0, vmi_excess_mean, prop_leakage)

                # Also run Section 4 causal reconstruction arms (backward, forward_k32, joint_true_k32)
                recon_summary = {}
                if not control:
                    rt_cfg = {"attack_batch_size": 32, "beam_width": 2}
                    for arm in ("backward", "forward_k32", "joint_true_k32"):
                        cache_key = (l_obs, defense)
                        if arm == "forward_k32" and b_train >= l_obs and cache_key in forward_k32_frozen_cache:
                            cached_entry = forward_k32_frozen_cache[cache_key]
                            if torch.equal(rows_h, cached_entry["rows_h"]):
                                recon_summary[arm] = copy.deepcopy(cached_entry["summary"])
                                continue
                        preds_arm, _ = attack_observations(
                            pub_info["head"], rows_h, rows_g, pub_info["E"], pub_info["W_dec"],
                            majority_gpt2, arm, rt_cfg, 17,
                        )
                        tok_hits = [
                            sum(1.0 if a == b else 0.0 for a, b in zip(r["tokens"], p)) / 32.0
                            for r, p in zip(eval_rows_gpt2, preds_arm)
                        ]
                        ex_hits = [1.0 if r["tokens"] == p else 0.0 for r, p in zip(eval_rows_gpt2, preds_arm)]
                        recon_summary[arm] = {
                            "token_accuracy_mean": sum(tok_hits) / len(tok_hits),
                            "exact_match_mean": sum(ex_hits) / len(ex_hits),
                            "per_document_token_accuracy": tok_hits,
                        }
                        if arm == "forward_k32" and b_train >= l_obs and cache_key not in forward_k32_frozen_cache:
                            forward_k32_frozen_cache[cache_key] = {
                                "rows_h": rows_h.detach().clone(),
                                "summary": copy.deepcopy(recon_summary[arm]),
                            }

                del model, split, measured["wires_fp32"], measured["obs_fp32"]
                gc.collect()
                torch.cuda.empty_cache()

                return {
                    "kind": kind,
                    "regime": regime,
                    "l_obs": 1 if control else l_obs,
                    "span": span,
                    "b_train": b_train,
                    "trainable_range": None if control else f"{b_train},{b_train + span}",
                    "defense": defense,
                    "seed": seed,
                    "train_stats": train_stats,
                    "utility_before": measured["before"],
                    "utility_after": measured["after"],
                    "eval_loss_mean": sum(measured["losses"]) / len(measured["losses"]),
                    "eval_losses_per_doc": measured["losses"],
                    "vmi_accuracy_mean": vmi_acc_mean,
                    "vmi_excess_mean": vmi_excess_mean,
                    "vmi_excess_per_doc": vmi_excess_list,
                    "property_auc": aucs,
                    "property_leakage": prop_leakage,
                    "normalised_leakage_index": nli,
                    "section4_reconstruction": recon_summary,
                    "wall_seconds": time.perf_counter() - t0,
                }

            # 1. Run all_local controls for host seeds
            control_by_seed = {}
            for seed in part_a_seeds:
                ctrl_res = run_part_a_unit("all_local", "control", 1, 4, 0, "D0", seed)
                control_by_seed[seed] = ctrl_res
                ctrl_summary = {k: v for k, v in ctrl_res.items() if k not in ("eval_losses_per_doc", "vmi_excess_per_doc")}
                report["part_a_controls"].append(ctrl_summary)
                ev.event("part_a_control_complete", ctrl_summary)
                _dump(report_path, report)
                print(json.dumps({"event": "part_a_control", "seed": seed, "eval_loss": ctrl_res["eval_loss_mean"], "wall_s": round(ctrl_res["wall_seconds"], 2)}), flush=True)

            # 2. Run Regime A (L_obs = 1 fixed, b_train in {1, 3, 5, 7}) and Regime B (L_obs = 6 fixed, b_train in {0, 2, 4, 6})
            part_a_regimes = [
                ("Regime_A_L1_server_placement", 1, [1, 3, 5, 7]),
                ("Regime_B_L6_cross_seam_placement", 6, [0, 2, 4, 6]),
            ]
            p6_ref_lookup = {
                (c["span"], c["start"], c["defense"], c["seed"]): c
                for c in p6_ref["cells"]
            }

            for regime_name, l_obs, starts_list in part_a_regimes:
                for b_train in starts_list:
                    for defense in ("D0", "D1"):
                        for seed in part_a_seeds:
                            cell_res = run_part_a_unit("trained", regime_name, l_obs, 4, b_train, defense, seed)
                            ctrl_losses = control_by_seed[seed]["eval_losses_per_doc"]
                            delta_u_per_doc = [m_l - c_l for m_l, c_l in zip(cell_res["eval_losses_per_doc"], ctrl_losses)]
                            delta_u_mean = sum(delta_u_per_doc) / len(delta_u_per_doc)
                            cell_res["delta_utility"] = delta_u_mean
                            cell_res["delta_utility_per_doc"] = delta_u_per_doc

                            # Save per-document vectors to sidecar file
                            cell_id = f"{regime_name}_L{l_obs}_s4_b{b_train}_{defense}_seed{seed}"
                            sidecar_path = output / "part_a_predictions" / f"{cell_id}.json"
                            _dump(sidecar_path, {
                                "cell_id": cell_id,
                                "delta_utility_per_doc": delta_u_per_doc,
                                "vmi_excess_per_doc": cell_res["vmi_excess_per_doc"],
                                "section4_reconstruction": cell_res["section4_reconstruction"],
                            })

                            # Check anchor cells (Regime A, L_obs=1, b_train=1, seed == part_a_anchor) against p6_gpt2_v2_reference.json
                            if l_obs == 1 and b_train == 1 and seed == part_a_anchor:
                                ref_cell = p6_ref_lookup[(4, 1, defense, seed)]
                                du_diff = abs(delta_u_mean - ref_cell["delta_utility"])
                                vmi_diff = abs(cell_res["vmi_excess_mean"] - ref_cell["vmi_excess"])
                                auc_diff = max(abs(a - b) for a, b in zip(cell_res["property_auc"], ref_cell["property_auc"]))
                                if vmi_diff > 1e-5 or du_diff > 0.01 or (defense == "D0" and auc_diff > 0.15):
                                    raise ValueError(
                                        f"Part A anchor mismatch on {cell_id}: du_diff={du_diff}, vmi_diff={vmi_diff}, auc_diff={auc_diff}"
                                    )
                                anchor_rec = {
                                    "part": "Part_A",
                                    "cell_id": cell_id,
                                    "delta_utility_abs_diff": du_diff,
                                    "vmi_excess_abs_diff": vmi_diff,
                                    "max_property_auc_abs_diff": auc_diff,
                                    "verified": True,
                                }
                                report["anchor_verifications"].append(anchor_rec)
                                ev.event("part_a_anchor_verified", anchor_rec)

                            cell_compact = {
                                "cell_id": cell_id,
                                "regime": regime_name,
                                "l_obs": l_obs,
                                "span": 4,
                                "b_train": b_train,
                                "trainable_range": cell_res["trainable_range"],
                                "defense": defense,
                                "seed": seed,
                                "delta_utility": delta_u_mean,
                                "eval_loss_mean": cell_res["eval_loss_mean"],
                                "utility_after_perplexity": cell_res["utility_after"]["perplexity"],
                                "vmi_accuracy_mean": cell_res["vmi_accuracy_mean"],
                                "vmi_excess": cell_res["vmi_excess_mean"],
                                "property_auc": cell_res["property_auc"],
                                "property_leakage": cell_res["property_leakage"],
                                "normalised_leakage_index": cell_res["normalised_leakage_index"],
                                "backward_token_accuracy": cell_res["section4_reconstruction"]["backward"]["token_accuracy_mean"],
                                "forward_k32_token_accuracy": cell_res["section4_reconstruction"]["forward_k32"]["token_accuracy_mean"],
                                "joint_true_k32_token_accuracy": cell_res["section4_reconstruction"]["joint_true_k32"]["token_accuracy_mean"],
                                "sidecar_file": f"part_a_predictions/{sidecar_path.name}",
                                "sidecar_sha256": digest(sidecar_path),
                                "wall_seconds": cell_res["wall_seconds"],
                            }
                            report["part_a_cells"].append(cell_compact)
                            ev.event("part_a_cell_complete", cell_compact)
                            _dump(report_path, report)
                            print(json.dumps({
                                "event": "part_a_cell",
                                "cell_id": cell_id,
                                "delta_u": round(delta_u_mean, 4),
                                "vmi_ex": round(cell_compact["vmi_excess"], 4),
                                "prop_leak": round(cell_compact["property_leakage"], 4),
                                "fwd_k32": round(cell_compact["forward_k32_token_accuracy"], 4),
                                "jnt_k32": round(cell_compact["joint_true_k32_token_accuracy"], 4),
                                "wall_s": round(cell_compact["wall_seconds"], 2),
                            }), flush=True)

            del pub_heads_part_a, base_gpt2_eager
            gc.collect()
            torch.cuda.empty_cache()

            # =========================================================================
            # PART B: QWEN3-0.6B D1 SUB-COMPONENT ISOLATION FACTORIAL (R6 Minor 10)
            # =========================================================================
            base_qwen = AutoModelForCausalLM.from_pretrained(
                str(qwen_snap), torch_dtype=torch.float32, attn_implementation="eager", local_files_only=True
            )
            qwen_eval_batches = [torch.tensor([r["tokens"]], dtype=torch.long) for r in qwen_data["evaluation"]]

            def run_part_b_qwen_arm(arm_spec, seed):
                t0 = time.perf_counter()
                arm_name = arm_spec["arm"]
                cfg = sb.BenchConfig(
                    model=str(qwen_snap),
                    split="1",
                    mode="finetune",
                    steps=128,
                    seq_len=32,
                    batch_size=1,
                    lr=5e-5,
                    seed=seed,
                    device="cuda",
                    dtype="fp32",
                    defense="quant",
                    quant_bits=arm_spec["quant_bits"],
                    quant_grad=arm_spec["quant_grad"],
                    grad_clip=arm_spec["grad_clip"],
                    grad_noise=arm_spec["grad_noise"],
                    trainable_range="0,4",
                    capture_dir=str(output / "scratch"),
                    output=str(output / "scratch" / "tmp_qwen.json"),
                    save_canonical=False,
                )
                cfg.validate()
                torch.manual_seed(seed)
                model = copy.deepcopy(base_qwen).to("cuda")
                split = sb.build_split(model, sb.SplitPlan.parse("1", n_layers=sb._n_layers_of(model)))
                perm = torch.randperm(len(qwen_data["train"]), generator=torch.Generator().manual_seed(seed)).tolist()
                batches = [torch.tensor([qwen_data["train"][i]["tokens"]], dtype=torch.long) for i in perm]

                # Exact sb.finetune RNG & step sequence with per-step sub-component telemetry
                torch.manual_seed(cfg.seed)
                model.train()
                for p in model.parameters():
                    p.requires_grad_(False)
                for blk in split.blocks[0:4]:
                    for p in blk.parameters():
                        p.requires_grad_(True)
                params = [p for p in model.parameters() if p.requires_grad]
                opt = torch.optim.AdamW(params, lr=cfg.lr)
                n_b = len(batches)

                step_trace = []
                first_nonfinite_step = None
                first_nonfinite_stage = None
                first_nonfinite_tensor = None
                max_down_proj_grad_norm_by_layer = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
                max_h_star_abs = 0.0
                max_g_wire_raw_norm = 0.0

                for step in range(cfg.steps):
                    ids = batches[step % n_b].to(split.device)
                    h_star = split.head_forward(ids)
                    if arm_spec["ste_fix"]:
                        h_q = sb._quantize_absmax(h_star, cfg.quant_bits)
                        h_sent = h_star + (h_q - h_star).detach()
                    else:
                        h_sent = sb._apply_seam_defense(h_star, cfg, {})
                    h_wire = h_sent.detach().requires_grad_(True)
                    logits = sb._server_rest_forward(split, h_wire, cfg, {})
                    loss = F.cross_entropy(
                        logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                        ids[:, 1:].reshape(-1),
                    )
                    loss.backward()
                    g_wire_raw = h_wire.grad
                    g_wire = sb._apply_grad_defense(g_wire_raw, cfg)
                    if h_sent.requires_grad:
                        h_sent.backward(g_wire.detach())

                    # Telemetry before optimizer step
                    loss_val = float(loss.detach().item())
                    h_abs_val = float(h_star.detach().abs().max().item())
                    g_raw_val = float(g_wire_raw.detach().norm().item())
                    g_post_val = float(g_wire.detach().norm().item())

                    if math.isfinite(h_abs_val) and h_abs_val > max_h_star_abs:
                        max_h_star_abs = h_abs_val
                    if math.isfinite(g_raw_val) and g_raw_val > max_g_wire_raw_norm:
                        max_g_wire_raw_norm = g_raw_val

                    layer_grad_norms = {}
                    for l_idx in range(4):
                        dp_grad = split.blocks[l_idx].mlp.down_proj.weight.grad
                        gn = float(dp_grad.detach().norm().item()) if dp_grad is not None else 0.0
                        layer_grad_norms[str(l_idx)] = gn
                        if math.isfinite(gn) and gn > max_down_proj_grad_norm_by_layer[l_idx]:
                            max_down_proj_grad_norm_by_layer[l_idx] = gn

                    if first_nonfinite_step is None:
                        if not math.isfinite(loss_val):
                            first_nonfinite_step = step
                            first_nonfinite_stage = "forward_loss"
                            first_nonfinite_tensor = "loss"
                        elif not bool(torch.isfinite(g_wire_raw).all().item()):
                            first_nonfinite_step = step
                            first_nonfinite_stage = "seam_backward_raw"
                            first_nonfinite_tensor = "h_wire.grad"
                        elif not bool(torch.isfinite(g_wire).all().item()):
                            first_nonfinite_step = step
                            first_nonfinite_stage = "seam_backward_defended"
                            first_nonfinite_tensor = "g_wire"
                        else:
                            for l_idx in range(4):
                                for pname, p in split.blocks[l_idx].named_parameters():
                                    if p.grad is not None and not bool(torch.isfinite(p.grad).all().item()):
                                        first_nonfinite_step = step
                                        first_nonfinite_stage = "parameter_grad"
                                        first_nonfinite_tensor = f"blocks.{l_idx}.{pname}.grad"
                                        break
                                if first_nonfinite_step is not None:
                                    break

                    opt.step()
                    opt.zero_grad()

                    if first_nonfinite_step is None:
                        for l_idx in range(4):
                            for pname, p in split.blocks[l_idx].named_parameters():
                                if not bool(torch.isfinite(p).all().item()):
                                    first_nonfinite_step = step
                                    first_nonfinite_stage = "parameter_weight_post_step"
                                    first_nonfinite_tensor = f"blocks.{l_idx}.{pname}"
                                    break
                            if first_nonfinite_step is not None:
                                break

                    step_trace.append({
                        "step": step,
                        "loss": loss_val if math.isfinite(loss_val) else None,
                        "loss_finite": math.isfinite(loss_val),
                        "h_star_max_abs": h_abs_val if math.isfinite(h_abs_val) else None,
                        "g_wire_raw_norm": g_raw_val if math.isfinite(g_raw_val) else None,
                        "g_wire_post_norm": g_post_val if math.isfinite(g_post_val) else None,
                        "down_proj_grad_norms": {
                            k: (v if math.isfinite(v) else None) for k, v in layer_grad_norms.items()
                        },
                    })

                all_finite = (first_nonfinite_step is None)
                eval_util = None
                if all_finite:
                    eval_util = measure_utility(split, cfg, {}, qwen_eval_batches, [])

                cell_id = f"qwen3_s4_b0_{arm_name}_seed{seed}"
                trace_file = output / "part_b_traces" / f"{cell_id}.json"
                _dump(trace_file, {
                    "cell_id": cell_id,
                    "arm_spec": arm_spec,
                    "seed": seed,
                    "all_finite": all_finite,
                    "first_nonfinite_step": first_nonfinite_step,
                    "first_nonfinite_stage": first_nonfinite_stage,
                    "first_nonfinite_tensor": first_nonfinite_tensor,
                    "step_trace": step_trace,
                })

                del model, split, opt, params
                gc.collect()
                torch.cuda.empty_cache()

                return {
                    "cell_id": cell_id,
                    "arm": arm_name,
                    "quant_bits": arm_spec["quant_bits"],
                    "quant_grad": arm_spec["quant_grad"],
                    "grad_clip": arm_spec["grad_clip"],
                    "grad_noise": arm_spec["grad_noise"],
                    "ste_fix": arm_spec["ste_fix"],
                    "seed": seed,
                    "all_finite": all_finite,
                    "first_nonfinite_step": first_nonfinite_step,
                    "first_nonfinite_stage": first_nonfinite_stage,
                    "first_nonfinite_tensor": first_nonfinite_tensor,
                    "final_train_loss": step_trace[-1]["loss"],
                    "max_h_star_abs": max_h_star_abs,
                    "max_g_wire_raw_norm": max_g_wire_raw_norm,
                    "max_down_proj_grad_norm_by_layer": {str(k): v for k, v in max_down_proj_grad_norm_by_layer.items()},
                    "eval_cross_entropy": eval_util["cross_entropy"] if eval_util else None,
                    "eval_perplexity": eval_util["perplexity"] if eval_util else None,
                    "trace_file": f"part_b_traces/{trace_file.name}",
                    "trace_sha256": digest(trace_file),
                    "wall_seconds": time.perf_counter() - t0,
                }

            part_b_schedule = [(arm_spec, seed) for seed in part_b_seeds for arm_spec in QWEN_SUBCOMPONENT_ARMS]
            if host == "koula":
                # Also run D1_full and D1_ste_fix on seed 66 on koula for cross-host overflow & STE verification
                for extra_arm in ("D1_full", "D1_ste_fix"):
                    spec = next(a for a in QWEN_SUBCOMPONENT_ARMS if a["arm"] == extra_arm)
                    part_b_schedule.append((spec, 66))

            for arm_spec, seed in part_b_schedule:
                b_res = run_part_b_qwen_arm(arm_spec, seed)
                report["part_b_cells"].append(b_res)
                ev.event("part_b_cell_complete", b_res)
                _dump(report_path, report)
                print(json.dumps({
                    "event": "part_b_cell",
                    "cell_id": b_res["cell_id"],
                    "all_finite": b_res["all_finite"],
                    "first_nonfinite_step": b_res["first_nonfinite_step"],
                    "first_nonfinite_tensor": b_res["first_nonfinite_tensor"],
                    "eval_ppl": round(b_res["eval_perplexity"], 2) if b_res["eval_perplexity"] else None,
                    "wall_s": round(b_res["wall_seconds"], 2),
                }), flush=True)

            del base_qwen
            gc.collect()
            torch.cuda.empty_cache()

            # =========================================================================
            # PART C: INDEPENDENT LONGER-PASSAGE EVALUATION (32, 64, and 128 TOKENS)
            # (R6 Major 3)
            # =========================================================================
            # Strictly disable TF32 to match Section 4 / stronger-search replay contract
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

            pins = json.loads((root / "experiments/v4_adaptation_expected.json").read_text())
            train_tensor_c = torch.tensor([r["tokens"] for r in gpt2_data["train"]], device="cuda", dtype=torch.long)
            eval_by_len = {
                int(sl): longer_data["evaluation_by_length"][sl]
                for sl in ("32", "64", "128")
            }
            eval_tensors_by_len = {
                sl: torch.tensor([r["tokens"] for r in rows], device="cuda", dtype=torch.long)
                for sl, rows in eval_by_len.items()
            }
            eligible_indices_in_256 = [r["index_in_256"] for r in eval_by_len[32]]

            for seed in part_c_seeds:
                t_v = time.perf_counter()
                torch.manual_seed(seed)
                base_c = GPT2LMHeadModel.from_pretrained(str(gpt2_snap), local_files_only=True)
                victim = copy.deepcopy(base_c).to("cuda").train()
                optim = torch.optim.AdamW(victim.parameters(), lr=5e-5)
                order = list(range(len(train_tensor_c)))
                random.Random(seed).shuffle(order)
                for step in range(128):
                    ids = train_tensor_c[order[step % len(order)]][None]
                    optim.zero_grad(set_to_none=True)
                    loss = victim(ids, labels=ids).loss
                    loss.backward()
                    optim.step()
                victim.eval()
                victim.requires_grad_(False)
                vs = sb.build_split(victim, sb.SplitPlan.parse("6", n_layers=12))

                # Load pinned public head and fit Ridge calibration decoder
                head_name = f"public-{seed}-6.pt"
                head_path = Path(resolve_asset(plan, head_name, pins[head_name]))
                dsplit, dembedding = si._load_head_pt(str(head_path), "cuda")
                dsplit.model.set_attn_implementation("sdpa")
                dsplit.model.eval().requires_grad_(False)
                dembedding = dembedding.detach()
                dhead = baselines.MeteredHead(dsplit)
                with torch.no_grad():
                    dhead(ids=cal_ids_gpt2[:1])
                    dhc = dhead(ids=cal_ids_gpt2).reshape(-1, dembedding.shape[-1])
                    ddecoder, _ = si._ridge_fit(dhc, dembedding[cal_ids_gpt2.flatten()], 0)

                # Verify that our trained victim's 32-token observations match the retained observations-{seed}-6.pt
                obs_name = f"observations-{seed}-6.pt"
                obs_path = Path(resolve_asset(plan, obs_name, pins[obs_name]))
                retained_obs = torch.load(obs_path, map_location="cuda", weights_only=True)
                ret_h_255 = retained_obs["h"][eligible_indices_in_256].to(dtype=torch.float32)
                ret_g_255 = retained_obs["g"][eligible_indices_in_256].to(dtype=torch.float32)

                for seq_len in (32, 64, 128):
                    ev_rows = eval_by_len[seq_len]
                    ev_tensor = eval_tensors_by_len[seq_len]
                    hs, gs = [], []
                    for chunk in ev_tensor.split(8):
                        with torch.no_grad():
                            h = vs.head_forward(chunk).detach()
                        leaf = h.clone().requires_grad_(True)
                        logits = vs.rest_forward(leaf)
                        loss = F.cross_entropy(
                            logits[:, :-1].reshape(-1, logits.shape[-1]),
                            chunk[:, 1:].reshape(-1),
                            reduction="sum",
                        ) / (len(ev_tensor) * (seq_len - 1))
                        gs.append(torch.autograd.grad(loss, leaf)[0].detach())
                        hs.append(h)
                        del logits, leaf, loss
                    hidden_sl = torch.cat(hs)
                    gradient_sl = torch.cat(gs)

                    if seq_len == 32:
                        # Note: in validate_reconstruction.py, gradient is normalized by 1/(256*31),
                        # whereas backward_scores normalizes each row F.normalize(-g[:, :-1], dim=-1),
                        # which is scale-invariant. Verify hidden activation match against retained_obs!
                        max_h_diff = float((hidden_sl - ret_h_255).abs().max().item())
                        if max_h_diff > 1e-4:
                            raise ValueError(f"Part C victim {seed} 32-token activation mismatch: {max_h_diff}")
                        if seed == part_c_anchor:
                            anchor_c = {
                                "part": "Part_C",
                                "seed": seed,
                                "seq_len": 32,
                                "max_hidden_abs_diff_vs_retained": max_h_diff,
                                "verified": True,
                            }
                            report["anchor_verifications"].append(anchor_c)
                            ev.event("part_c_anchor_verified", anchor_c)

                    arms_c = [
                        ("backward", None, 2, 32),
                        ("forward_k32", 32, 2, 32),
                        ("joint_true_k32", 32, 2, 32),
                        ("forward_k128", 128, 2, 16),
                        ("joint_true_k128", 128, 2, 16),
                    ]
                    for arm_name, k_val, b_val, batch_sz in arms_c:
                        rt_c = {"attack_batch_size": batch_sz, "beam_width": b_val}
                        torch.cuda.synchronize()
                        with dhead.measure() as work_c:
                            preds_c, diag_c = attack_observations(
                                dhead, hidden_sl, gradient_sl, dembedding, ddecoder,
                                majority_gpt2, arm_name, rt_c, 17,
                            )
                        per_doc_c, summary_c = _score_long_predictions(ev_rows, preds_c, metrics_module)
                        cand_rec_c = _candidate_recall_long(diag_c.get("batches", []), ev_tensor)
                        cell_id_c = f"long_seed{seed}_L6_public_len{seq_len}_{arm_name}"
                        pred_path_c = output / "part_c_predictions" / f"{cell_id_c}.json"
                        _dump(pred_path_c, {
                            "cell_id": cell_id_c,
                            "victim_seed": seed,
                            "seq_len": seq_len,
                            "arm": arm_name,
                            "predictions": preds_c,
                            "document_ids": [r["id"] for r in ev_rows],
                            "per_document": per_doc_c,
                        })
                        cell_rec_c = {
                            "cell_id": cell_id_c,
                            "status": "COMPLETE",
                            "victim_seed": seed,
                            "depth": 6,
                            "knowledge": "public",
                            "seq_len": seq_len,
                            "arm": arm_name,
                            "k": k_val,
                            "beam_width": b_val,
                            "work": asdict(work_c),
                            "amortized_seconds_per_document": work_c.wall_seconds / len(ev_rows),
                            "candidate_recall": cand_rec_c,
                            "summary": summary_c,
                            "prediction_file": f"part_c_predictions/{pred_path_c.name}",
                            "prediction_sha256": digest(pred_path_c),
                        }
                        report["part_c_cells"].append(cell_rec_c)
                        ev.event("part_c_cell_complete", cell_rec_c)
                        _dump(report_path, report)
                        print(json.dumps({
                            "event": "part_c_cell",
                            "cell_id": cell_id_c,
                            "tok_acc": round(summary_c["token_accuracy_mean"], 4),
                            "pos0_31": round(summary_c["pos0_31_accuracy_mean"], 4),
                            "pos32_63": round(summary_c["pos32_63_accuracy_mean"], 4) if summary_c["pos32_63_accuracy_mean"] is not None else None,
                            "pos64_127": round(summary_c["pos64_127_accuracy_mean"], 4) if summary_c["pos64_127_accuracy_mean"] is not None else None,
                            "exact": round(summary_c["exact_match_mean"], 4),
                            "wall_s": round(work_c.wall_seconds, 2),
                        }), flush=True)
                    del hidden_sl, gradient_sl

                del victim, vs, base_c, dsplit, dhead, dembedding, ddecoder, retained_obs
                gc.collect()
                torch.cuda.empty_cache()

            report["status"] = "EXP3_REPLAY_COMPLETE"
            report["elapsed_seconds"] = time.monotonic() - started
            _dump(report_path, report)
            ev.event("exp3_replay_complete", {
                "host": host,
                "n_part_a_controls": len(report["part_a_controls"]),
                "n_part_a_cells": len(report["part_a_cells"]),
                "n_part_b_cells": len(report["part_b_cells"]),
                "n_part_c_cells": len(report["part_c_cells"]),
                "elapsed_seconds": report["elapsed_seconds"],
            })
        except BaseException as error:
            report.update(
                status="FAILED",
                error=repr(error),
                traceback=traceback.format_exc(),
                elapsed_seconds=time.monotonic() - started,
            )
            _dump(report_path, report)
            ev.event("exp3_replay_failed", {"error": repr(error)})
            raise

    ok, problems = verify_evidence(report_path.with_name(report_path.name + ".evidence.json"))
    if not ok:
        raise RuntimeError(f"Evidence verification failed on {host}: {problems}")
    return report
