"""Qwen3-0.6B & GPT-2 Ridge-Calibration Sample-Size Scaling + Full Exact-Grant k=128 Completion
(Closing XSci PoPETs Major 1 and reviewer3.com Comments #2 & #3 on Paper A & Paper B):

1. Part 1 (Calibration-Size Curve across C in {32, 64, 128, 256, 384} passages, N_cal in {1024, 2048, 4096, 8192, 12288} tokens):
   - Qwen3-0.6B (d=1024, V=151,936): evaluated across all 8 victim seeds (11..88), all 4 split depths L in {1, 7, 14, 21},
     both client layer grants {public, exact}, and both FP32 and FP64 Gram-matrix ridge solves (320 setup-scale evaluations).
     Records unique calibration token coverage, auto-ridge lambda, top-1 linear ridge token accuracy, and candidate-set
     recall (overall, pos0, pos1_31) for forward_k32, joint_true_k32, forward_k128, and joint_true_k128.
   - GPT-2 (d=768, V=50,257): evaluated across all 10 trained victim seeds (101..1010), split depths L in {3, 6},
     and both client layer grants {public, exact} across C in {32, 64, 128, 256, 384} (200 setup-scale evaluations).

2. Part 2 (Full B=2 Beam-Search Reconstruction Across Calibration Sizes C in {32, 64, 128, 256}, N_cal in {1024, 2048, 4096, 8192}):
   - GPT-2 at midpoint depth L=6 (public & exact grants, all 10 victim seeds 101..1010):
     Full B=2 beam search for forward_k32, joint_true_k32, forward_k128, and joint_true_k128 across C in {32, 64, 128, 256} (320 cells).
   - Qwen3-0.6B at midpoint depth L=14 (public & exact grants, all 8 victim seeds 11..88):
     Full B=2 beam search for forward_k32 and joint_true_k32 across C in {64, 128, 256} (plus C=32 anchor-verified on each host),
     and for forward_k128 and joint_true_k128 at C=128 (N_cal=4096=4d, 1,491 unique token types > d=1024) under both public & exact grants.

3. Part 3 (Complete Missing Exact-Grant k=128 Cells on Qwen3-0.6B at Standard C=32 across L in {1, 7, 14}):
   - Evaluates forward_k128 and joint_true_k128 under exact (trained) client weights at L in {1, 7, 14} across all 8 victim seeds (11..88),
     providing 100% paired public-vs-exact k=128 coverage across depths L in {1, 7, 14}.

Host partitioning:
- toula: Qwen3-0.6B victim seeds [11, 22, 33, 44]; GPT-2 victim seeds [101, 202, 303, 404, 505]
- koula: Qwen3-0.6B victim seeds [55, 66, 77, 88]; GPT-2 victim seeds [606, 707, 808, 909, 1010]
"""
from __future__ import annotations

import copy
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import platform
import random
import socket
import sys
import time
import traceback


HOSTS = ["toula", "koula"]
QWEN3_HOST_SEEDS = {
    "toula": [11, 22, 33, 44],
    "koula": [55, 66, 77, 88],
}
QWEN3_ANCHOR_SEED = {
    "toula": 11,
    "koula": 55,
}
GPT2_HOST_SEEDS = {
    "toula": [101, 202, 303, 404, 505],
    "koula": [606, 707, 808, 909, 1010],
}
QWEN3_DEPTHS = [1, 7, 14, 21]
GPT2_DEPTHS = [3, 6]
KNOWLEDGE_MODES = ["public", "exact"]
CAL_PASSAGE_COUNTS = [32, 64, 128, 256, 384]
BEAM_CAL_COUNTS_K32 = [64, 128, 256]
BEAM_CAL_COUNTS_K128 = [128]
GPT2_BEAM_CAL_COUNTS = [32, 64, 128, 256]
EXACT_K128_C32_DEPTHS = [1, 7, 14]

ATTACK_SEED = 17
ATTACK_BATCH_SIZE = 32
BEAM_WIDTH = 2
TRAIN_STEPS = 128
LEARNING_RATE = 5e-5


def _dump(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _candidate_recall_from_tensor(all_cands, truth_tensor):
    import torch
    if all_cands is None:
        return None
    truth_cpu = truth_tensor.detach().cpu()
    hits = (all_cands.detach().cpu() == truth_cpu.unsqueeze(-1)).any(dim=-1).to(dtype=torch.float32)
    return {
        "overall": float(hits.mean().item()),
        "pos0": float(hits[:, 0].mean().item()),
        "pos1_31": float(hits[:, 1:32].mean().item()),
    }


def _score_predictions_32(evaluation_rows, predictions, metrics_module):
    per_doc = []
    n = len(evaluation_rows)
    pos0_hits = 0.0
    pos1_31_hits = 0.0
    for idx, (row, pred) in enumerate(zip(evaluation_rows, predictions)):
        truth = row["tokens"]
        m = metrics_module.reconstruction_metrics(truth, pred, row.get("annotations", []))
        p0 = 1.0 if truth[0] == pred[0] else 0.0
        p1_31 = sum(1.0 if a == b else 0.0 for a, b in zip(truth[1:32], pred[1:32])) / 31.0
        pos0_hits += p0
        pos1_31_hits += p1_31
        per_doc.append({
            "document_id": row["id"],
            "index_in_256": idx,
            "token_accuracy": m["token_accuracy"],
            "pos0_acc": p0,
            "pos1_31_acc": p1_31,
            "sequence_exact": m["sequence_exact"],
            "longest_exact_run": m["longest_exact_run"],
            "span2_rate": m["contiguous_spans"]["2"]["rate"],
            "span4_rate": m["contiguous_spans"]["4"]["rate"],
            "span8_rate": m["contiguous_spans"]["8"]["rate"],
        })

    summary = {
        "n_documents": n,
        "seq_len": 32,
        "token_accuracy_mean": sum(d["token_accuracy"] for d in per_doc) / n,
        "pos0_accuracy_mean": pos0_hits / n,
        "pos1_31_accuracy_mean": pos1_31_hits / n,
        "exact_match_mean": sum(1.0 if d["sequence_exact"] else 0.0 for d in per_doc) / n,
        "sequence_exact_mean": sum(1.0 if d["sequence_exact"] else 0.0 for d in per_doc) / n,
        "longest_exact_run_mean": sum(float(d["longest_exact_run"]) for d in per_doc) / n,
        "span2_rate_mean": sum(float(d["span2_rate"] or 0.0) for d in per_doc) / n,
        "span4_rate_mean": sum(float(d["span4_rate"] or 0.0) for d in per_doc) / n,
        "span8_rate_mean": sum(float(d["span8_rate"] or 0.0) for d in per_doc) / n,
    }
    return per_doc, summary


def _merge_candidates_from_top(front_k, back_half, count):
    import torch
    combined = torch.cat((front_k[..., :count // 2], back_half, front_k), dim=-1)
    chosen = [list(dict.fromkeys(row))[:count] for row in combined.reshape(-1, combined.shape[-1]).tolist()]
    out = torch.tensor(chosen, device=front_k.device, dtype=torch.long).reshape(*front_k.shape[:2], count)
    out[:, 0] = front_k[:, 0]
    return out


def run_qwen3_cal_scale_replay(plan, root, output, resolve_asset, digest):
    """Execute GPT-2 & Qwen3-0.6B calibration scaling + exact k=128 completion on assigned host."""
    root, output = Path(root), Path(output)
    host = socket.gethostname().split(".")[0].lower()
    if host not in HOSTS:
        raise ValueError(f"Unexpected execution host: {host}")

    qwen_seeds = QWEN3_HOST_SEEDS[host]
    qwen_anchor_seed = QWEN3_ANCHOR_SEED[host]
    gpt2_seeds = GPT2_HOST_SEEDS[host]

    output.mkdir(parents=True, exist_ok=False)
    (output / "predictions").mkdir(parents=True, exist_ok=True)
    report_path = output / "qwen3_cal_scale_replay.json"
    started = time.monotonic()

    # Ensure ROOT comes first on sys.path so attacker.splitbench includes Qwen3 support
    for mod_name in list(sys.modules.keys()):
        if mod_name == "attacker" or mod_name.startswith("attacker.") or mod_name == "validate_reconstruction":
            del sys.modules[mod_name]
    sys.path[:0] = [str(root), str(root / "experiments")]

    from attacker.evidence import EvidenceLogger, verify_evidence

    report = {
        "schema": "paper-a-b.qwen3-cal-scale-replay.v1",
        "status": "RUNNING",
        "phase": plan["phase"],
        "host": host,
        "qwen3_seeds": qwen_seeds,
        "qwen3_anchor_seed": qwen_anchor_seed,
        "gpt2_seeds": gpt2_seeds,
        "qwen3_depths": QWEN3_DEPTHS,
        "gpt2_depths": GPT2_DEPTHS,
        "knowledge_modes": KNOWLEDGE_MODES,
        "cal_passage_counts": CAL_PASSAGE_COUNTS,
        "beam_cal_counts_k32": BEAM_CAL_COUNTS_K32,
        "beam_cal_counts_k128": BEAM_CAL_COUNTS_K128,
        "gpt2_beam_cal_counts": GPT2_BEAM_CAL_COUNTS,
        "exact_k128_c32_depths": EXACT_K128_C32_DEPTHS,
        "attack_seed": ATTACK_SEED,
        "attack_batch_size": ATTACK_BATCH_SIZE,
        "beam_width": BEAM_WIDTH,
        "train_steps": TRAIN_STEPS,
        "learning_rate": LEARNING_RATE,
        "utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": plan["source_commit"],
        "runner_sha256": digest(__file__),
        "plan": plan,
        "anchor_verifications": [],
        "gpt2_cal_curve": [],
        "gpt2_beam_cells": [],
        "qwen3_victims": [],
        "qwen3_cal_curve": [],
        "qwen3_beam_cells": [],
    }
    _dump(report_path, report)

    with EvidenceLogger(report_path, node_role=host + "/gpu", extra={"study": plan["study"], "phase": plan["phase"]}) as ev:
        try:
            import torch
            import torch.nn.functional as F
            import transformers
            from transformers import AutoModelForCausalLM
            from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
            from attacker import splitbench as sb
            from attacker.attacks import seam_inversion as si
            import attacker.reconstruction_baselines as baselines
            import attacker.reconstruction_metrics as metrics_module

            report["module_paths_and_sha256"] = {
                "splitbench": {"path": str(Path(sb.__file__).resolve()), "sha256": digest(sb.__file__)},
                "seam_inversion": {"path": str(Path(si.__file__).resolve()), "sha256": digest(si.__file__)},
                "reconstruction_baselines": {"path": str(Path(baselines.__file__).resolve()), "sha256": digest(baselines.__file__)},
                "reconstruction_metrics": {"path": str(Path(metrics_module.__file__).resolve()), "sha256": digest(metrics_module.__file__)},
            }

            torch.set_num_threads(1)
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable for calibration scaling replay")

            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

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

            qwen_input_path = exp3_dir / "qwen_input.json"
            if digest(qwen_input_path) != exp3_pins["qwen_input.json"]:
                raise ValueError("qwen_input.json SHA-256 mismatch")
            qwen_data = json.loads(qwen_input_path.read_text())

            cal_scale_path = exp3_dir / "cal_scale_inputs.json"
            if digest(cal_scale_path) != exp3_pins["cal_scale_inputs.json"]:
                raise ValueError("cal_scale_inputs.json SHA-256 mismatch")
            cal_scale_data = json.loads(cal_scale_path.read_text())

            gpt2_input_path = Path(resolve_asset(plan, "input.json", plan["input_sha256"]))
            if digest(gpt2_input_path) != plan["input_sha256"]:
                raise ValueError("GPT-2 input.json SHA-256 mismatch")
            gpt2_data = json.loads(gpt2_input_path.read_text())

            report["input_hashes"] = {
                "qwen_input_sha256": digest(qwen_input_path),
                "cal_scale_inputs_sha256": digest(cal_scale_path),
                "gpt2_input_sha256": digest(gpt2_input_path),
            }

            ev.event("cal_scale_start", {k: v for k, v in report.items() if not isinstance(v, list)})
            _dump(report_path, report)

            # =========================================================================
            # PART A: GPT-2 (d=768, V=50,257) CALIBRATION SCALING (L=6, 10 SEEDS)
            # =========================================================================
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            v4_pins = json.loads((root / "experiments/v4_adaptation_expected.json").read_text())
            gpt2_cal_384 = torch.tensor(
                [r["tokens"] for r in cal_scale_data["gpt2"]["calibration_passages_384"]],
                dtype=torch.long,
                device="cuda",
            )
            gpt2_eval_rows = gpt2_data["evaluation"]
            gpt2_eval_ids = torch.tensor([r["tokens"] for r in gpt2_eval_rows], dtype=torch.long, device="cuda")
            gpt2_majority_tok = si.majority_control_token(gpt2_cal_384[:32].tolist())

            @torch.no_grad()
            def _fast_gpt2_kv_beam_reconstruct(head, h, candidates, beam_width=2):
                """Zero-copy shared-prefix KV-cached beam search for GPT-2."""
                split = head.split
                start_blk, end_blk = split.ranges[0]
                blocks = split.blocks[start_blk:end_blk]
                n_blk = len(blocks)
                num_heads = blocks[0].attn.num_heads
                head_dim = blocks[0].attn.head_dim
                scale = head_dim ** -0.5
                n, length, count = candidates.shape
                dev = h.device
                prefixes = torch.empty(n, 1, 0, dtype=torch.long, device=dev)
                scores = torch.zeros(n, 1, device=dev)
                past_k = [None] * n_blk
                past_v = [None] * n_blk

                for pos in range(length):
                    width = prefixes.shape[1]
                    seq_count = n * width * count
                    head.work.head_forward_calls += 1
                    head.work.head_forward_sequences += int(seq_count)
                    head.work.head_forward_token_positions += int(seq_count * (pos + 1))

                    cand_tok = candidates[:, None, pos, :].expand(n, width, count).reshape(seq_count, 1)
                    pos_tensor = torch.tensor([[pos]], device=dev, dtype=torch.long)
                    x = split.drop(split.wte(cand_tok) + split.wpe(pos_tensor))
                    new_k_list = []
                    new_v_list = []
                    for l_idx, blk in enumerate(blocks):
                        residual = x
                        x_norm = blk.ln_1(x)
                        attn = blk.attn
                        qkv = attn.c_attn(x_norm)
                        q, k, v = qkv.split(attn.split_size, dim=2)
                        q_g = q.view(n * width, count, num_heads, head_dim).permute(0, 2, 1, 3)
                        k_g = k.view(n * width, count, num_heads, head_dim).permute(0, 2, 1, 3)
                        v_g = v.view(n * width, count, num_heads, head_dim).permute(0, 2, 1, 3)
                        new_k_list.append(k.view(n, width * count, num_heads, 1, head_dim))
                        new_v_list.append(v.view(n, width * count, num_heads, 1, head_dim))
                        if past_k[l_idx] is None:
                            attn_out = v_g.permute(0, 2, 1, 3).reshape(seq_count, 1, num_heads * head_dim)
                        else:
                            pk = past_k[l_idx].reshape(n * width, num_heads, pos, head_dim)
                            pv = past_v[l_idx].reshape(n * width, num_heads, pos, head_dim)
                            scores_past = torch.matmul(q_g, pk.transpose(-1, -2)) * scale
                            scores_self = (q_g * k_g).sum(dim=-1, keepdim=True) * scale
                            attn_weights = torch.softmax(
                                torch.cat([scores_past, scores_self], dim=-1), dim=-1, dtype=torch.float32
                            ).to(q_g.dtype)
                            out_past = torch.matmul(attn_weights[..., :pos], pv)
                            out_self = attn_weights[..., pos:] * v_g
                            attn_out = (out_past + out_self).permute(0, 2, 1, 3).reshape(seq_count, 1, num_heads * head_dim)
                        x = residual + attn.resid_dropout(attn.c_proj(attn_out))
                        x = x + blk.mlp(blk.ln_2(x))
                    if end_blk == len(split.blocks) and split.ln_f is not None:
                        x = split.ln_f(x)
                    hidden_step = x.view(n, width * count, -1)
                    residual_err = (hidden_step - h[:, None, pos]).square().sum(-1)
                    total = residual_err + scores.repeat_interleave(count, dim=1)
                    w_next = min(beam_width, width * count)
                    scores, chosen = total.topk(w_next, largest=False, dim=1, sorted=True)
                    expanded = prefixes[:, :, None, :].expand(n, width, count, pos)
                    tokens = candidates[:, None, pos, :, None].expand(n, width, count, 1)
                    extensions = torch.cat((expanded, tokens), dim=-1).reshape(n, width * count, pos + 1)
                    prefixes = extensions.gather(1, chosen[..., None].expand(-1, -1, pos + 1))
                    if pos < length - 1:
                        parent_beam = chosen // count
                        for l_idx in range(n_blk):
                            chosen_k = new_k_list[l_idx].gather(1, chosen[:, :, None, None, None].expand(-1, -1, num_heads, 1, head_dim))
                            chosen_v = new_v_list[l_idx].gather(1, chosen[:, :, None, None, None].expand(-1, -1, num_heads, 1, head_dim))
                            if past_k[l_idx] is None:
                                past_k[l_idx] = chosen_k
                                past_v[l_idx] = chosen_v
                            else:
                                prev_k = past_k[l_idx].gather(1, parent_beam[:, :, None, None, None].expand(-1, -1, num_heads, pos, head_dim))
                                prev_v = past_v[l_idx].gather(1, parent_beam[:, :, None, None, None].expand(-1, -1, num_heads, pos, head_dim))
                                past_k[l_idx] = torch.cat([prev_k, chosen_k], dim=3)
                                past_v[l_idx] = torch.cat([prev_v, chosen_v], dim=3)
                return prefixes[:, 0], {"forward_residual_sum": scores[:, 0].tolist()}

            gpt2_schedule = [(s, 6, "public") for s in gpt2_seeds]

            for g_seed, g_depth, g_know in gpt2_schedule:
                obs_name = f"observations-{g_seed}-{g_depth}.pt"
                obs_path = Path(resolve_asset(plan, obs_name, v4_pins[obs_name]))
                obs = torch.load(obs_path, map_location="cuda", weights_only=True)
                g_hidden = obs["h"].to(dtype=torch.float32)
                g_grad = obs["g"].to(dtype=torch.float32)

                head_name = f"{g_know}-{g_seed}-{g_depth}.pt"
                head_path = Path(resolve_asset(plan, head_name, v4_pins[head_name]))
                gsplit, gE = si._load_head_pt(str(head_path), "cuda")
                gsplit.model.set_attn_implementation("sdpa")
                gsplit.model.eval().requires_grad_(False)
                gE = gE.detach().to(dtype=torch.float32)
                gE_norm_T = F.normalize(gE, dim=-1).T.contiguous()
                ghead = baselines.MeteredHead(gsplit)

                with torch.no_grad():
                    h_cal_all_g = torch.cat([gsplit.head_forward(ch) for ch in gpt2_cal_384.split(32)], dim=0)
                    bwd_s_g = torch.zeros((*g_grad.shape[:2], gE.shape[0]), device="cuda")
                    bwd_s_g[:, 0, gpt2_majority_tok] = 1.0
                    bwd_s_g[:, 1:] = F.normalize(-g_grad[:, :-1], dim=-1) @ gE_norm_T
                    back_top64_g = bwd_s_g.topk(64, dim=-1).indices
                    back_top16_g = back_top64_g[..., :16]
                    del bwd_s_g

                for C in CAL_PASSAGE_COUNTS:
                    t0_c = time.perf_counter()
                    hc_g = h_cal_all_g[:C].reshape(-1, gE.shape[1])
                    tc_g = gpt2_cal_384[:C].reshape(-1)
                    uniq_g = int(tc_g.unique().numel())
                    W32_g, lam32_g = si._ridge_fit(hc_g, gE[tc_g], 0.0)
                    W64_g, lam64_g = si._ridge_fit(hc_g.double(), gE[tc_g].double(), 0.0)
                    W64_g = W64_g.float()

                    with torch.no_grad():
                        fwd_s32 = F.normalize(g_hidden @ W32_g, dim=-1) @ gE_norm_T
                        top1_32 = float((fwd_s32.argmax(dim=-1) == gpt2_eval_ids).float().mean().item())
                        front128_32 = fwd_s32.topk(128, dim=-1).indices
                        front32_32 = front128_32[..., :32]
                        del fwd_s32

                        fwd_s64 = F.normalize(g_hidden @ W64_g, dim=-1) @ gE_norm_T
                        top1_64 = float((fwd_s64.argmax(dim=-1) == gpt2_eval_ids).float().mean().item())
                        front128_64 = fwd_s64.topk(128, dim=-1).indices
                        front32_64 = front128_64[..., :32]
                        del fwd_s64

                    jnt32_32 = _merge_candidates_from_top(front32_32, back_top16_g, 32)
                    jnt128_32 = _merge_candidates_from_top(front128_32, back_top64_g, 128)
                    jnt32_64 = _merge_candidates_from_top(front32_64, back_top16_g, 32)
                    jnt128_64 = _merge_candidates_from_top(front128_64, back_top64_g, 128)

                    curve_rec = {
                        "model": "gpt2",
                        "victim_seed": g_seed,
                        "depth": g_depth,
                        "knowledge": g_know,
                        "cal_passages": C,
                        "cal_tokens": C * 32,
                        "cal_unique_tokens": uniq_g,
                        "ridge_lambda_fp32": float(lam32_g),
                        "ridge_lambda_fp64": float(lam64_g),
                        "ridge_top1_token_acc_fp32": top1_32,
                        "ridge_top1_token_acc_fp64": top1_64,
                        "candidate_recall_fp32": {
                            "forward_k32": _candidate_recall_from_tensor(front32_32, gpt2_eval_ids),
                            "joint_true_k32": _candidate_recall_from_tensor(jnt32_32, gpt2_eval_ids),
                            "forward_k128": _candidate_recall_from_tensor(front128_32, gpt2_eval_ids),
                            "joint_true_k128": _candidate_recall_from_tensor(jnt128_32, gpt2_eval_ids),
                        },
                        "candidate_recall_fp64": {
                            "forward_k32": _candidate_recall_from_tensor(front32_64, gpt2_eval_ids),
                            "joint_true_k32": _candidate_recall_from_tensor(jnt32_64, gpt2_eval_ids),
                            "forward_k128": _candidate_recall_from_tensor(front128_64, gpt2_eval_ids),
                            "joint_true_k128": _candidate_recall_from_tensor(jnt128_64, gpt2_eval_ids),
                        },
                        "wall_seconds": time.perf_counter() - t0_c,
                    }
                    report["gpt2_cal_curve"].append(curve_rec)

                    if C in GPT2_BEAM_CAL_COUNTS:
                        cand_map = {
                            "forward_k32": front32_32 if C == 32 else front32_64,
                            "joint_true_k32": jnt32_32 if C == 32 else jnt32_64,
                        }
                        if C in (32, 128):
                            cand_map["forward_k128"] = front128_32 if C == 32 else front128_64
                            cand_map["joint_true_k128"] = jnt128_32 if C == 32 else jnt128_64
                        for arm_name, cands_t in cand_map.items():
                            torch.cuda.synchronize()
                            t0_b = time.perf_counter()
                            preds_batches = []
                            for start_i in range(0, len(g_hidden), ATTACK_BATCH_SIZE):
                                stop_i = start_i + ATTACK_BATCH_SIZE
                                pb, _ = _fast_gpt2_kv_beam_reconstruct(
                                    ghead, g_hidden[start_i:stop_i], cands_t[start_i:stop_i], BEAM_WIDTH
                                )
                                preds_batches.append(pb.detach().cpu())
                            preds_list = torch.cat(preds_batches, dim=0).tolist()
                            torch.cuda.synchronize()
                            wall_b = time.perf_counter() - t0_b
                            _, sum_b = _score_predictions_32(gpt2_eval_rows, preds_list, metrics_module)
                            cand_rec_g = _candidate_recall_from_tensor(cands_t, gpt2_eval_ids)
                            cell_id_g = f"gpt2-seed{g_seed}-L{g_depth}-{g_know}-C{C}-{arm_name}"
                            beam_rec = {
                                "cell_id": cell_id_g,
                                "model": "gpt2",
                                "victim_seed": g_seed,
                                "depth": g_depth,
                                "knowledge": g_know,
                                "cal_passages": C,
                                "cal_tokens": C * 32,
                                "cal_unique_tokens": uniq_g,
                                "arm": arm_name,
                                "ridge_precision": "fp32" if C == 32 else "fp64",
                                "candidate_recall": cand_rec_g,
                                "summary": sum_b,
                                "wall_seconds": wall_b,
                            }
                            report["gpt2_beam_cells"].append(beam_rec)
                            ev.event("gpt2_beam_cell_complete", beam_rec)
                            print(json.dumps({
                                "event": "gpt2_beam_cell",
                                "cell_id": cell_id_g,
                                "tok_acc": round(sum_b["token_accuracy_mean"], 4),
                                "doc_acc": round(sum_b["exact_match_mean"], 4),
                                "cand_rec": round(cand_rec_g["overall"], 4),
                                "wall_s": round(wall_b, 2),
                            }), flush=True)

                del gsplit, gE, gE_norm_T, ghead, h_cal_all_g, back_top64_g, back_top16_g, obs, g_hidden, g_grad
                _dump(report_path, report)
                print(json.dumps({
                    "event": "gpt2_seed_complete",
                    "seed": g_seed,
                    "n_gpt2_curve": len(report["gpt2_cal_curve"]),
                    "n_gpt2_beam": len(report["gpt2_beam_cells"]),
                }), flush=True)

            gc.collect()
            torch.cuda.empty_cache()

            # =========================================================================
            # PART B: QWEN3-0.6B (d=1024, V=151,936) CALIBRATION SCALING & EXACT k=128
            # =========================================================================
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            qwen_snap = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
            if not qwen_snap.is_dir():
                raise FileNotFoundError(f"Missing Qwen3-0.6B snapshot: {qwen_snap}")

            @torch.no_grad()
            def _fast_qwen3_kv_beam_reconstruct(head, h, candidates, beam_width=2):
                """Zero-copy grouped-query KV-cached beam search for Qwen3-0.6B."""
                split = head.split
                start_blk, end_blk = split.ranges[0]
                blocks = split.blocks[start_blk:end_blk]
                n_blk = len(blocks)
                model_cfg = split.model.config
                num_heads = model_cfg.num_attention_heads
                num_kv_heads = model_cfg.num_key_value_heads
                head_dim = getattr(model_cfg, "head_dim", model_cfg.hidden_size // num_heads)
                n_rep = num_heads // num_kv_heads
                scaling = blocks[0].self_attn.scaling
                n, length, count = candidates.shape
                dev = h.device
                prefixes = torch.empty(n, 1, 0, dtype=torch.long, device=dev)
                scores = torch.zeros(n, 1, device=dev)
                past_k = [None] * n_blk
                past_v = [None] * n_blk

                for pos in range(length):
                    width = prefixes.shape[1]
                    seq_count = n * width * count
                    head.work.head_forward_calls += 1
                    head.work.head_forward_sequences += int(seq_count)
                    head.work.head_forward_token_positions += int(seq_count * (pos + 1))

                    cand_tok = candidates[:, None, pos, :].expand(n, width, count).reshape(seq_count, 1)
                    pos_1 = torch.tensor([[pos]], device=dev, dtype=torch.long)
                    x = split.wte(cand_tok)
                    cos, sin = split.rotary_emb(x[:1], pos_1)
                    new_k_list = []
                    new_v_list = []
                    for l_idx, blk in enumerate(blocks):
                        residual = x
                        x_norm = blk.input_layernorm(x)
                        attn = blk.self_attn
                        q = attn.q_norm(attn.q_proj(x_norm).view(seq_count, 1, num_heads, head_dim)).transpose(1, 2)
                        k = attn.k_norm(attn.k_proj(x_norm).view(seq_count, 1, num_kv_heads, head_dim)).transpose(1, 2)
                        v = attn.v_proj(x_norm).view(seq_count, 1, num_kv_heads, head_dim).transpose(1, 2)
                        q, k = apply_rotary_pos_emb(q, k, cos, sin)
                        q_g = q.view(n * width, count, num_kv_heads, n_rep, head_dim).permute(0, 2, 1, 3, 4).reshape(n * width, num_kv_heads, count * n_rep, head_dim)
                        k_g = k.view(n * width, count, num_kv_heads, 1, head_dim).permute(0, 2, 1, 3, 4)
                        v_g = v.view(n * width, count, num_kv_heads, 1, head_dim).permute(0, 2, 1, 3, 4)
                        if past_k[l_idx] is None:
                            attn_out = v_g.expand(-1, -1, -1, n_rep, -1).permute(0, 2, 1, 3, 4).reshape(seq_count, 1, num_heads * head_dim)
                        else:
                            pk = past_k[l_idx].reshape(n * width, num_kv_heads, pos, head_dim)
                            pv = past_v[l_idx].reshape(n * width, num_kv_heads, pos, head_dim)
                            scores_past = torch.matmul(q_g, pk.transpose(-1, -2)).view(n * width, num_kv_heads, count, n_rep, pos) * scaling
                            scores_self = (q_g.view(n * width, num_kv_heads, count, n_rep, head_dim) * k_g).sum(dim=-1, keepdim=True) * scaling
                            attn_weights = torch.softmax(torch.cat([scores_past, scores_self], dim=-1), dim=-1, dtype=torch.float32).to(q.dtype)
                            out_past = torch.matmul(attn_weights[..., :pos].reshape(n * width, num_kv_heads, count * n_rep, pos), pv).view(n * width, num_kv_heads, count, n_rep, head_dim)
                            out_self = attn_weights[..., pos:] * v_g
                            attn_out = (out_past + out_self).permute(0, 2, 1, 3, 4).reshape(seq_count, 1, num_heads * head_dim)
                        x = residual + attn.o_proj(attn_out)
                        x = x + blk.mlp(blk.post_attention_layernorm(x))
                        new_k_list.append(k.view(n, width * count, num_kv_heads, 1, head_dim))
                        new_v_list.append(v.view(n, width * count, num_kv_heads, 1, head_dim))
                    if end_blk == len(split.blocks) and split.ln_f is not None:
                        x = split.ln_f(x)
                    hidden_step = x.view(n, width * count, -1)
                    residual_err = (hidden_step - h[:, None, pos]).square().sum(-1)
                    total = residual_err + scores.repeat_interleave(count, dim=1)
                    w_next = min(beam_width, width * count)
                    scores, chosen = total.topk(w_next, largest=False, dim=1, sorted=True)
                    expanded = prefixes[:, :, None, :].expand(n, width, count, pos)
                    tokens = candidates[:, None, pos, :, None].expand(n, width, count, 1)
                    extensions = torch.cat((expanded, tokens), dim=-1).reshape(n, width * count, pos + 1)
                    prefixes = extensions.gather(1, chosen[..., None].expand(-1, -1, pos + 1))
                    if pos < length - 1:
                        parent_beam = chosen // count
                        for l_idx in range(n_blk):
                            chosen_k = new_k_list[l_idx].gather(1, chosen[:, :, None, None, None].expand(-1, -1, num_kv_heads, 1, head_dim))
                            chosen_v = new_v_list[l_idx].gather(1, chosen[:, :, None, None, None].expand(-1, -1, num_kv_heads, 1, head_dim))
                            if past_k[l_idx] is None:
                                past_k[l_idx] = chosen_k
                                past_v[l_idx] = chosen_v
                            else:
                                prev_k = past_k[l_idx].gather(1, parent_beam[:, :, None, None, None].expand(-1, -1, num_kv_heads, pos, head_dim))
                                prev_v = past_v[l_idx].gather(1, parent_beam[:, :, None, None, None].expand(-1, -1, num_kv_heads, pos, head_dim))
                                past_k[l_idx] = torch.cat([prev_k, chosen_k], dim=3)
                                past_v[l_idx] = torch.cat([prev_v, chosen_v], dim=3)
                return prefixes[:, 0], {"forward_residual_sum": scores[:, 0].tolist()}

            t0_load = time.perf_counter()
            base_qwen = AutoModelForCausalLM.from_pretrained(
                str(qwen_snap), dtype=torch.float32, attn_implementation="eager", local_files_only=True
            )
            base_load_seconds = time.perf_counter() - t0_load

            qwen_cal_384 = torch.tensor(
                [r["tokens"] for r in cal_scale_data["qwen3_0_6b"]["calibration_passages_384"]],
                dtype=torch.long,
                device="cuda",
            )
            qwen_train_ids = torch.tensor([r["tokens"] for r in qwen_data["train"]], dtype=torch.long, device="cuda")
            qwen_eval_rows = qwen_data["evaluation"]
            qwen_eval_ids = torch.tensor([r["tokens"] for r in qwen_eval_rows], dtype=torch.long, device="cuda")
            qwen_majority_tok = si.majority_control_token(qwen_cal_384[:32].tolist())

            # Precompute public model heads & calibration activations across all 384 passages for all 4 depths
            pub_model = copy.deepcopy(base_qwen).to("cuda").eval()
            pub_model.requires_grad_(False)
            pub_E = si._embedding_of_model(pub_model).detach()
            pub_E_norm_T = F.normalize(pub_E, dim=-1).T.contiguous()

            public_heads = {}
            for depth in QWEN3_DEPTHS:
                pub_split = sb.build_split(pub_model, sb.SplitPlan.parse(str(depth), n_layers=28))
                pub_head = baselines.MeteredHead(pub_split)
                with torch.no_grad():
                    h_cal_384 = torch.cat([pub_head(ids=ch) for ch in qwen_cal_384.split(32)], dim=0)
                decoders_by_c = {}
                for C in CAL_PASSAGE_COUNTS:
                    hc = h_cal_384[:C].reshape(-1, pub_E.shape[1])
                    tc = qwen_cal_384[:C].reshape(-1)
                    W32, lam32 = si._ridge_fit(hc, pub_E[tc], 0.0)
                    W64, lam64 = si._ridge_fit(hc.double(), pub_E[tc].double(), 0.0)
                    decoders_by_c[C] = {
                        "W32": W32,
                        "lam32": float(lam32),
                        "W64": W64.float(),
                        "lam64": float(lam64),
                        "unique_tokens": int(tc.unique().numel()),
                    }
                public_heads[depth] = {
                    "split": pub_split,
                    "head": pub_head,
                    "E": pub_E,
                    "E_norm_T": pub_E_norm_T,
                    "decoders_by_c": decoders_by_c,
                }

            for seed in qwen_seeds:
                torch.manual_seed(seed)
                victim = copy.deepcopy(base_qwen).to("cuda")
                victim.train()
                optim = torch.optim.AdamW(victim.parameters(), lr=LEARNING_RATE)
                order = list(range(len(qwen_train_ids)))
                random.Random(seed).shuffle(order)
                torch.cuda.synchronize()
                t0_train = time.perf_counter()
                final_step_loss = None
                for step in range(TRAIN_STEPS):
                    ids = qwen_train_ids[order[step % len(order)]][None]
                    optim.zero_grad(set_to_none=True)
                    loss = victim(ids, labels=ids).loss
                    loss.backward()
                    optim.step()
                    if step == TRAIN_STEPS - 1:
                        final_step_loss = float(loss.detach().item())
                torch.cuda.synchronize()
                training_seconds = time.perf_counter() - t0_train

                victim.eval()
                victim.requires_grad_(False)
                with torch.no_grad():
                    nll_after = sum(
                        float(victim(chunk, labels=chunk).loss) * len(chunk)
                        for chunk in qwen_eval_ids.split(ATTACK_BATCH_SIZE)
                    ) / len(qwen_eval_ids)

                victim_rec = {
                    "seed": seed,
                    "base_model_acquisition_seconds": base_load_seconds,
                    "training_seconds": training_seconds,
                    "final_train_step_loss": final_step_loss,
                    "evaluation_nll_after": nll_after,
                }
                report["qwen3_victims"].append(victim_rec)
                ev.event("qwen3_victim_trained", victim_rec)
                _dump(report_path, report)
                print(json.dumps({
                    "event": "qwen3_victim_trained",
                    "seed": seed,
                    "nll_after": round(nll_after, 4),
                    "train_s": round(training_seconds, 2),
                }), flush=True)

                exact_E = si._embedding_of_model(victim).detach().to("cuda")
                exact_E_norm_T = F.normalize(exact_E, dim=-1).T.contiguous()

                for depth in QWEN3_DEPTHS:
                    vs = sb.build_split(victim, sb.SplitPlan.parse(str(depth), n_layers=28))
                    hs, gs = [], []
                    for chunk in qwen_eval_ids.split(ATTACK_BATCH_SIZE):
                        with torch.no_grad():
                            h_c = vs.head_forward(chunk).detach()
                        leaf = h_c.clone().requires_grad_(True)
                        logits = vs.rest_forward(leaf)
                        loss_c = F.cross_entropy(
                            logits[:, :-1].reshape(-1, logits.shape[-1]),
                            chunk[:, 1:].reshape(-1),
                            reduction="sum",
                        ) / (len(qwen_eval_ids) * (qwen_eval_ids.shape[1] - 1))
                        gs.append(torch.autograd.grad(loss_c, leaf)[0].detach())
                        hs.append(h_c)
                        del logits, leaf, loss_c
                    hidden = torch.cat(hs, dim=0)
                    gradient = torch.cat(gs, dim=0)

                    exact_head = baselines.MeteredHead(vs)
                    with torch.no_grad():
                        h_cal_384_ex = torch.cat([exact_head(ids=ch) for ch in qwen_cal_384.split(32)], dim=0)
                    exact_decoders_by_c = {}
                    for C in CAL_PASSAGE_COUNTS:
                        hc_ex = h_cal_384_ex[:C].reshape(-1, exact_E.shape[1])
                        tc_ex = qwen_cal_384[:C].reshape(-1)
                        W32_ex, lam32_ex = si._ridge_fit(hc_ex, exact_E[tc_ex], 0.0)
                        W64_ex, lam64_ex = si._ridge_fit(hc_ex.double(), exact_E[tc_ex].double(), 0.0)
                        exact_decoders_by_c[C] = {
                            "W32": W32_ex,
                            "lam32": float(lam32_ex),
                            "W64": W64_ex.float(),
                            "lam64": float(lam64_ex),
                            "unique_tokens": int(tc_ex.unique().numel()),
                        }
                    del h_cal_384_ex

                    knowledge_bundles = {
                        "public": public_heads[depth],
                        "exact": {
                            "split": vs,
                            "head": exact_head,
                            "E": exact_E,
                            "E_norm_T": exact_E_norm_T,
                            "decoders_by_c": exact_decoders_by_c,
                        },
                    }

                    for knowledge in KNOWLEDGE_MODES:
                        kb = knowledge_bundles[knowledge]
                        vocab_len = int(kb["E"].shape[0])

                        # Precompute backward top-64 and top-16 once for this (seed, depth, knowledge)
                        back64_chunks = []
                        with torch.no_grad():
                            for start_i in range(0, len(gradient), ATTACK_BATCH_SIZE):
                                g_b = gradient[start_i:start_i + ATTACK_BATCH_SIZE]
                                bwd_s = torch.zeros((*g_b.shape[:2], vocab_len), device="cuda")
                                bwd_s[:, 0, qwen_majority_tok] = 1.0
                                bwd_s[:, 1:] = F.normalize(-g_b[:, :-1], dim=-1) @ kb["E_norm_T"]
                                back64_chunks.append(bwd_s.topk(64, dim=-1).indices)
                                del bwd_s
                        back_top64 = torch.cat(back64_chunks, dim=0)
                        back_top16 = back_top64[..., :16]

                        for C in CAL_PASSAGE_COUNTS:
                            t0_c = time.perf_counter()
                            dec_c = kb["decoders_by_c"][C]
                            W32 = dec_c["W32"]
                            W64 = dec_c["W64"]

                            front128_32_chunks, front128_64_chunks = [], []
                            top1_32_hits, top1_64_hits = [], []
                            with torch.no_grad():
                                for start_i in range(0, len(hidden), ATTACK_BATCH_SIZE):
                                    h_b = hidden[start_i:start_i + ATTACK_BATCH_SIZE]
                                    e_b = qwen_eval_ids[start_i:start_i + ATTACK_BATCH_SIZE]

                                    fwd_s32 = F.normalize(h_b @ W32, dim=-1) @ kb["E_norm_T"]
                                    top1_32_hits.append((fwd_s32.argmax(dim=-1) == e_b).float())
                                    front128_32_chunks.append(fwd_s32.topk(128, dim=-1).indices)
                                    del fwd_s32

                                    fwd_s64 = F.normalize(h_b @ W64, dim=-1) @ kb["E_norm_T"]
                                    top1_64_hits.append((fwd_s64.argmax(dim=-1) == e_b).float())
                                    front128_64_chunks.append(fwd_s64.topk(128, dim=-1).indices)
                                    del fwd_s64

                            top1_32 = float(torch.cat(top1_32_hits, dim=0).mean().item())
                            top1_64 = float(torch.cat(top1_64_hits, dim=0).mean().item())
                            front128_32 = torch.cat(front128_32_chunks, dim=0)
                            front32_32 = front128_32[..., :32]
                            front128_64 = torch.cat(front128_64_chunks, dim=0)
                            front32_64 = front128_64[..., :32]

                            jnt32_32 = _merge_candidates_from_top(front32_32, back_top16, 32)
                            jnt128_32 = _merge_candidates_from_top(front128_32, back_top64, 128)
                            jnt32_64 = _merge_candidates_from_top(front32_64, back_top16, 32)
                            jnt128_64 = _merge_candidates_from_top(front128_64, back_top64, 128)

                            curve_rec = {
                                "model": "qwen3_0_6b",
                                "victim_seed": seed,
                                "depth": depth,
                                "knowledge": knowledge,
                                "cal_passages": C,
                                "cal_tokens": C * 32,
                                "cal_unique_tokens": dec_c["unique_tokens"],
                                "ridge_lambda_fp32": dec_c["lam32"],
                                "ridge_lambda_fp64": dec_c["lam64"],
                                "ridge_top1_token_acc_fp32": top1_32,
                                "ridge_top1_token_acc_fp64": top1_64,
                                "candidate_recall_fp32": {
                                    "forward_k32": _candidate_recall_from_tensor(front32_32, qwen_eval_ids),
                                    "joint_true_k32": _candidate_recall_from_tensor(jnt32_32, qwen_eval_ids),
                                    "forward_k128": _candidate_recall_from_tensor(front128_32, qwen_eval_ids),
                                    "joint_true_k128": _candidate_recall_from_tensor(jnt128_32, qwen_eval_ids),
                                },
                                "candidate_recall_fp64": {
                                    "forward_k32": _candidate_recall_from_tensor(front32_64, qwen_eval_ids),
                                    "joint_true_k32": _candidate_recall_from_tensor(jnt32_64, qwen_eval_ids),
                                    "forward_k128": _candidate_recall_from_tensor(front128_64, qwen_eval_ids),
                                    "joint_true_k128": _candidate_recall_from_tensor(jnt128_64, qwen_eval_ids),
                                },
                                "wall_seconds": time.perf_counter() - t0_c,
                            }
                            report["qwen3_cal_curve"].append(curve_rec)

                            # Determine which full B=2 beam-search cells to execute at this (seed, depth, knowledge, C)
                            beam_arms_to_run = []
                            if C == 32:
                                # 1. Anchor verification on anchor seed at L=14 public forward_k32
                                if seed == qwen_anchor_seed and depth == 14 and knowledge == "public":
                                    beam_arms_to_run.append(("forward_k32", front32_32, "fp32", True))
                                # 2. Part 3: Missing C=32 exact-grant k=128 cells at L in {1, 7, 14}
                                if knowledge == "exact" and depth in EXACT_K128_C32_DEPTHS:
                                    beam_arms_to_run.append(("forward_k128", front128_32, "fp32", False))
                                    beam_arms_to_run.append(("joint_true_k128", jnt128_32, "fp32", False))
                            elif depth == 14:
                                # Part 2: Midpoint L=14 calibration-scaling beam search across C in {64, 128, 256}
                                if C in BEAM_CAL_COUNTS_K32:
                                    beam_arms_to_run.append(("forward_k32", front32_64, "fp64", False))
                                    beam_arms_to_run.append(("joint_true_k32", jnt32_64, "fp64", False))
                                if C in BEAM_CAL_COUNTS_K128:
                                    beam_arms_to_run.append(("forward_k128", front128_64, "fp64", False))
                                    beam_arms_to_run.append(("joint_true_k128", jnt128_64, "fp64", False))

                            cell_preds_sidecar = {}
                            for arm_name, cands_t, ridge_prec, is_anchor in beam_arms_to_run:
                                torch.cuda.synchronize()
                                t0_b = time.perf_counter()
                                with kb["head"].measure() as work_b:
                                    preds_batches = []
                                    for start_i in range(0, len(hidden), ATTACK_BATCH_SIZE):
                                        stop_i = start_i + ATTACK_BATCH_SIZE
                                        pb, _ = _fast_qwen3_kv_beam_reconstruct(
                                            kb["head"], hidden[start_i:stop_i], cands_t[start_i:stop_i], BEAM_WIDTH
                                        )
                                        preds_batches.append(pb.detach().cpu())
                                    preds_list = torch.cat(preds_batches, dim=0).tolist()
                                torch.cuda.synchronize()
                                wall_b = time.perf_counter() - t0_b
                                work_dict = asdict(work_b)
                                per_doc, sum_b = _score_predictions_32(qwen_eval_rows, preds_list, metrics_module)
                                cand_rec = _candidate_recall_from_tensor(cands_t, qwen_eval_ids)

                                cell_id = f"qwen3-seed{seed}-L{depth}-{knowledge}-C{C}-{arm_name}"
                                beam_rec = {
                                    "cell_id": cell_id,
                                    "model": "qwen3_0_6b",
                                    "victim_seed": seed,
                                    "depth": depth,
                                    "knowledge": knowledge,
                                    "cal_passages": C,
                                    "cal_tokens": C * 32,
                                    "cal_unique_tokens": dec_c["unique_tokens"],
                                    "arm": arm_name,
                                    "ridge_precision": ridge_prec,
                                    "is_anchor": is_anchor,
                                    "work": work_dict,
                                    "wall_seconds": wall_b,
                                    "candidate_recall": cand_rec,
                                    "summary": sum_b,
                                }
                                report["qwen3_beam_cells"].append(beam_rec)
                                ev.event("qwen3_beam_cell_complete", beam_rec)

                                if is_anchor:
                                    anchor_rec = {
                                        "cell_id": cell_id,
                                        "token_accuracy_mean": sum_b["token_accuracy_mean"],
                                        "candidate_recall_overall": cand_rec["overall"],
                                    }
                                    report["anchor_verifications"].append(anchor_rec)
                                    ev.event("qwen3_anchor_verified", anchor_rec)

                                cell_preds_sidecar[arm_name] = {
                                    "cell_id": cell_id,
                                    "summary": sum_b,
                                    "candidate_recall": cand_rec,
                                    "work": work_dict,
                                    "predicted_tokens": preds_list,
                                    "per_document": per_doc,
                                }
                                print(json.dumps({
                                    "event": "qwen3_beam_cell",
                                    "cell_id": cell_id,
                                    "tok_acc": round(sum_b["token_accuracy_mean"], 4),
                                    "doc_acc": round(sum_b["exact_match_mean"], 4),
                                    "cand_rec": round(cand_rec["overall"], 4),
                                    "wall_s": round(wall_b, 2),
                                }), flush=True)

                            if cell_preds_sidecar:
                                sidecar_name = f"qwen3_seed{seed}_L{depth}_{knowledge}_C{C}.json"
                                sidecar_path = output / "predictions" / sidecar_name
                                _dump(sidecar_path, {
                                    "victim_seed": seed,
                                    "depth": depth,
                                    "knowledge": knowledge,
                                    "cal_passages": C,
                                    "document_ids": [r["id"] for r in qwen_eval_rows],
                                    "arms": cell_preds_sidecar,
                                })
                                sidecar_sha = digest(sidecar_path)
                                for b_rec in report["qwen3_beam_cells"][-len(cell_preds_sidecar):]:
                                    b_rec["sidecar_file"] = f"predictions/{sidecar_name}"
                                    b_rec["sidecar_sha256"] = sidecar_sha

                            del front128_32, front32_32, front128_64, front32_64, jnt32_32, jnt128_32, jnt32_64, jnt128_64

                        _dump(report_path, report)
                        del back_top64, back_top16
                        gc.collect()
                        torch.cuda.empty_cache()

                    del hidden, gradient, exact_head, exact_decoders_by_c
                    gc.collect()
                    torch.cuda.empty_cache()

                del victim, optim, exact_E, exact_E_norm_T
                gc.collect()
                torch.cuda.empty_cache()

            report["status"] = "COMPLETE"
            report["elapsed_seconds"] = time.monotonic() - started
            ev.event("cal_scale_complete", {
                "status": "COMPLETE",
                "elapsed_seconds": report["elapsed_seconds"],
                "n_qwen3_victims": len(report["qwen3_victims"]),
                "n_qwen3_cal_curve": len(report["qwen3_cal_curve"]),
                "n_qwen3_beam_cells": len(report["qwen3_beam_cells"]),
                "n_gpt2_cal_curve": len(report["gpt2_cal_curve"]),
                "n_gpt2_beam_cells": len(report["gpt2_beam_cells"]),
            })
            _dump(report_path, report)
        except Exception as exc:
            report["status"] = "FAILED"
            report["elapsed_seconds"] = time.monotonic() - started
            report["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
            ev.event("cal_scale_failed", report["error"])
            _dump(report_path, report)
            raise

    ok, problems = verify_evidence(str(report_path) + ".evidence.json")
    if not ok:
        raise RuntimeError(f"Evidence verification failed: {problems}")
    return report
