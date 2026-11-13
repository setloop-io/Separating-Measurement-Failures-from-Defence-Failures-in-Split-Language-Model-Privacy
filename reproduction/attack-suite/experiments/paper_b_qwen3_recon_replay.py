"""Experiment #4 Replay (Paper B PB-R1 Major 4 & Paper A R4 Major 4 External Validity):
Qwen3-0.6B (GQA, RoPE, QK-Norm, SwiGLU, 28 layers, d=1024, V=151,936) Joint Gradient-Plus-Activation
Text-Reconstruction Benchmark across Split Depths L in {1, 7, 14, 21} and Client Layer Grants {public, exact}.

Host partitioning (8 independent victim models, 256 held-out 32-token passages each):
- toula: victim seeds [11, 22, 33, 44]
- koula: victim seeds [55, 66, 77, 88]

Arms evaluated per (victim_seed, depth, knowledge) cell:
- Public grant (L in {1, 7, 14, 21}): backward, forward_k32, joint_true_k32, joint_shuffled_k32, forward_k128, joint_true_k128
- Exact/trained grant (L in {1, 7, 14, 21}): backward, forward_k32, joint_true_k32, joint_shuffled_k32
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
HOST_SEEDS = {
    "toula": [11, 22, 33, 44],
    "koula": [55, 66, 77, 88],
}
DEPTHS = [1, 7, 14, 21]
K128_PUBLIC_DEPTHS = [1, 7, 14]
KNOWLEDGE_MODES = ["public", "exact"]
ARMS_BY_KNOWLEDGE = {
    "public": [
        "backward",
        "forward_k32",
        "joint_true_k32",
        "joint_shuffled_k32",
        "forward_k128",
        "joint_true_k128",
    ],
    "exact": [
        "backward",
        "forward_k32",
        "joint_true_k32",
        "joint_shuffled_k32",
    ],
}


def _arms_for_cell(depth: int, knowledge: str) -> list[str]:
    if knowledge == "public" and depth in K128_PUBLIC_DEPTHS:
        return list(ARMS_BY_KNOWLEDGE["public"])
    return list(ARMS_BY_KNOWLEDGE["exact"])
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


def run_qwen3_recon_replay(plan, root, output, resolve_asset, digest):
    """Execute Qwen3-0.6B joint text-reconstruction benchmark on the assigned host."""
    root, output = Path(root), Path(output)
    host = socket.gethostname().split(".")[0].lower()
    if host not in HOSTS:
        raise ValueError(f"Unexpected execution host: {host}")

    seeds = HOST_SEEDS[host]
    output.mkdir(parents=True, exist_ok=False)
    (output / "predictions").mkdir(parents=True, exist_ok=True)
    report_path = output / "qwen3_recon_replay.json"
    started = time.monotonic()

    # Ensure ROOT comes first on sys.path so attacker.splitbench includes Qwen3 support
    for mod_name in list(sys.modules.keys()):
        if mod_name == "attacker" or mod_name.startswith("attacker.") or mod_name == "validate_reconstruction":
            del sys.modules[mod_name]
    sys.path[:0] = [str(root), str(root / "experiments")]

    from attacker.evidence import EvidenceLogger, verify_evidence

    report = {
        "schema": "paper-b.qwen3-recon-replay.v1",
        "status": "RUNNING",
        "phase": plan["phase"],
        "host": host,
        "victim_seeds": seeds,
        "depths": DEPTHS,
        "k128_public_depths": K128_PUBLIC_DEPTHS,
        "knowledge_modes": KNOWLEDGE_MODES,
        "arms_by_knowledge": ARMS_BY_KNOWLEDGE,
        "attack_seed": ATTACK_SEED,
        "attack_batch_size": ATTACK_BATCH_SIZE,
        "beam_width": BEAM_WIDTH,
        "train_steps": TRAIN_STEPS,
        "learning_rate": LEARNING_RATE,
        "utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": plan["source_commit"],
        "runner_sha256": digest(__file__),
        "plan": plan,
        "kv_cache_verifications": [],
        "victims": [],
        "setup": [],
        "cells": [],
        "failed_cells": [],
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
            from attacker.validation_controls import transform_gradient

            report["module_paths_and_sha256"] = {
                "splitbench": {"path": str(Path(sb.__file__).resolve()), "sha256": digest(sb.__file__)},
                "seam_inversion": {"path": str(Path(si.__file__).resolve()), "sha256": digest(si.__file__)},
                "reconstruction_baselines": {"path": str(Path(baselines.__file__).resolve()), "sha256": digest(baselines.__file__)},
                "reconstruction_metrics": {"path": str(Path(metrics_module.__file__).resolve()), "sha256": digest(metrics_module.__file__)},
            }

            torch.set_num_threads(1)
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable for Qwen3-0.6B reconstruction replay")

            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
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
                "allow_tf32": True,
            }

            exp3_dir = Path(plan["exp3_directory"])
            exp3_pins_path = exp3_dir / "PINS.json"
            exp3_pins = json.loads(exp3_pins_path.read_text())
            qwen_input_path = exp3_dir / "qwen_input.json"
            if digest(qwen_input_path) != exp3_pins["qwen_input.json"]:
                raise ValueError("qwen_input.json SHA-256 mismatch")
            report["qwen_input_sha256"] = digest(qwen_input_path)
            qwen_data = json.loads(qwen_input_path.read_text())

            qwen_snap = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
            if not qwen_snap.is_dir():
                raise FileNotFoundError(f"Missing Qwen3-0.6B snapshot: {qwen_snap}")

            ev.event("qwen3_recon_start", {k: v for k, v in report.items() if not isinstance(v, list)})
            _dump(report_path, report)

            orig_beam_reconstruct = baselines.beam_reconstruct

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
                    pos_tensor = torch.tensor([[pos]], device=dev, dtype=torch.long).expand(seq_count, 1)
                    x = split.wte(cand_tok)
                    cos, sin = split.rotary_emb(x, pos_tensor)
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

            # Load base Qwen3-0.6B once in CPU memory
            t0_load = time.perf_counter()
            base_qwen = AutoModelForCausalLM.from_pretrained(
                str(qwen_snap), torch_dtype=torch.float32, attn_implementation="eager", local_files_only=True
            )
            base_load_seconds = time.perf_counter() - t0_load

            cal_ids = torch.tensor([r["tokens"] for r in qwen_data["calibration"]], dtype=torch.long, device="cuda")
            train_ids = torch.tensor([r["tokens"] for r in qwen_data["train"]], dtype=torch.long, device="cuda")
            eval_rows = qwen_data["evaluation"]
            eval_ids = torch.tensor([r["tokens"] for r in eval_rows], dtype=torch.long, device="cuda")
            majority_tok = si.majority_control_token(cal_ids.tolist())

            # Share a single public GPU model across all split depths
            torch.cuda.synchronize()
            t0_pub_model = time.perf_counter()
            pub_model = copy.deepcopy(base_qwen).to("cuda").eval()
            pub_model.requires_grad_(False)
            torch.cuda.synchronize()
            pub_model_load_s = time.perf_counter() - t0_pub_model

            public_heads = {}
            pub_E_shared = si._embedding_of_model(pub_model).detach()
            pub_E_norm_T = F.normalize(pub_E_shared, dim=-1).T.contiguous()

            for depth in DEPTHS:
                torch.cuda.synchronize()
                t0_pub = time.perf_counter()
                pub_split = sb.build_split(pub_model, sb.SplitPlan.parse(str(depth), n_layers=28))
                pub_head = baselines.MeteredHead(pub_split)
                with torch.no_grad():
                    pub_head(ids=cal_ids[:1])
                torch.cuda.synchronize()
                pub_load_s = pub_model_load_s + (time.perf_counter() - t0_pub)

                with pub_head.measure() as fit_work, torch.no_grad():
                    h_cal = pub_head(ids=cal_ids).reshape(-1, pub_E_shared.shape[1])
                    W_dec, lam_used = si._ridge_fit(h_cal, pub_E_shared[cal_ids.reshape(-1)], 0.0)
                fit_work_dict = asdict(fit_work)

                # Verify _fast_qwen3_kv_beam_reconstruct against orig_beam_reconstruct at midpoint L=14
                if depth == 14:
                    with torch.no_grad():
                        h_test = pub_split.head_forward(cal_ids[:2]).detach()
                        fwd_test = baselines.forward_scores(h_test, pub_E_shared, W_dec, pub_head)
                        cands_test = baselines.candidate_ids(fwd_test, None, 32, "forward")
                        pred_orig, det_orig = orig_beam_reconstruct(pub_head, h_test, cands_test, BEAM_WIDTH)
                        pred_kv, det_kv = _fast_qwen3_kv_beam_reconstruct(pub_head, h_test, cands_test, BEAM_WIDTH)
                        exact_tokens = bool(torch.equal(pred_orig, pred_kv))
                        max_res_diff = max(abs(a - b) for a, b in zip(det_orig["forward_residual_sum"], det_kv["forward_residual_sum"]))
                        if not exact_tokens or max_res_diff > 1e-1:
                            raise RuntimeError(
                                f"KV-cache verification failed at depth L={depth}: exact={exact_tokens}, max_res_diff={max_res_diff}"
                            )
                        kv_rec = {
                            "depth": depth,
                            "exact_discrete_tokens": exact_tokens,
                            "max_forward_residual_sum_abs_diff": max_res_diff,
                        }
                        report["kv_cache_verifications"].append(kv_rec)
                        ev.event("kv_cache_verified", kv_rec)

                public_heads[depth] = {
                    "split": pub_split,
                    "E": pub_E_shared,
                    "E_norm_T": pub_E_norm_T,
                    "W_dec": W_dec,
                    "ridge_lambda": lam_used,
                    "head": pub_head,
                    "loading_seconds": pub_load_s,
                    "calibration_work": fit_work_dict,
                }

            baselines.beam_reconstruct = _fast_qwen3_kv_beam_reconstruct

            try:
                for seed in seeds:
                    torch.manual_seed(seed)
                    victim = copy.deepcopy(base_qwen).to("cuda")
                    with torch.no_grad():
                        victim.eval()
                        nll_before = sum(
                            float(victim(chunk, labels=chunk).loss) * len(chunk)
                            for chunk in eval_ids.split(ATTACK_BATCH_SIZE)
                        ) / len(eval_ids)

                    victim.train()
                    optim = torch.optim.AdamW(victim.parameters(), lr=LEARNING_RATE)
                    order = list(range(len(train_ids)))
                    random.Random(seed).shuffle(order)
                    torch.cuda.synchronize()
                    t0_train = time.perf_counter()
                    final_step_loss = None
                    for step in range(TRAIN_STEPS):
                        ids = train_ids[order[step % len(order)]][None]
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
                            for chunk in eval_ids.split(ATTACK_BATCH_SIZE)
                        ) / len(eval_ids)

                    victim_rec = {
                        "seed": seed,
                        "base_model_acquisition_seconds": base_load_seconds,
                        "training_seconds": training_seconds,
                        "final_train_step_loss": final_step_loss,
                        "evaluation_nll_before": nll_before,
                        "evaluation_nll_after": nll_after,
                    }
                    report["victims"].append(victim_rec)
                    ev.event("victim_trained", victim_rec)
                    _dump(report_path, report)
                    print(json.dumps({
                        "event": "victim_trained",
                        "seed": seed,
                        "nll_before": round(nll_before, 4),
                        "nll_after": round(nll_after, 4),
                        "train_s": round(training_seconds, 2),
                    }), flush=True)

                    exact_E = si._embedding_of_model(victim).detach().to("cuda")
                    exact_E_norm_T = F.normalize(exact_E, dim=-1).T.contiguous()

                    for depth in DEPTHS:
                        # Capture seam hidden states h and true seam gradients g at depth L
                        vs = sb.build_split(victim, sb.SplitPlan.parse(str(depth), n_layers=28))
                        hs, gs = [], []
                        for chunk in eval_ids.split(ATTACK_BATCH_SIZE):
                            with torch.no_grad():
                                h_c = vs.head_forward(chunk).detach()
                            leaf = h_c.clone().requires_grad_(True)
                            logits = vs.rest_forward(leaf)
                            loss_c = F.cross_entropy(
                                logits[:, :-1].reshape(-1, logits.shape[-1]),
                                chunk[:, 1:].reshape(-1),
                                reduction="sum",
                            ) / (len(eval_ids) * (eval_ids.shape[1] - 1))
                            gs.append(torch.autograd.grad(loss_c, leaf)[0].detach())
                            hs.append(h_c)
                            del logits, leaf, loss_c
                        hidden = torch.cat(hs, dim=0)
                        gradient = torch.cat(gs, dim=0)
                        g_shuf, _ = transform_gradient(gradient, "shuffled", ATTACK_SEED)

                        # Build exact (trained) client head & ridge decoder on public calibration set
                        torch.cuda.synchronize()
                        t0_ex = time.perf_counter()
                        exact_head = baselines.MeteredHead(vs)
                        with torch.no_grad():
                            exact_head(ids=cal_ids[:1])
                        torch.cuda.synchronize()
                        exact_load_s = time.perf_counter() - t0_ex

                        with exact_head.measure() as exact_fit_work, torch.no_grad():
                            h_cal_ex = exact_head(ids=cal_ids).reshape(-1, exact_E.shape[1])
                            exact_W_dec, exact_lam = si._ridge_fit(h_cal_ex, exact_E[cal_ids.reshape(-1)], 0.0)
                        exact_fit_dict = asdict(exact_fit_work)

                        knowledge_bundles = {
                            "public": public_heads[depth],
                            "exact": {
                                "split": vs,
                                "E": exact_E,
                                "E_norm_T": exact_E_norm_T,
                                "W_dec": exact_W_dec,
                                "ridge_lambda": exact_lam,
                                "head": exact_head,
                                "loading_seconds": exact_load_s,
                                "calibration_work": exact_fit_dict,
                            },
                        }

                        for knowledge in KNOWLEDGE_MODES:
                            t0_setup = time.perf_counter()
                            kb = knowledge_bundles[knowledge]
                            arms_list = _arms_for_cell(depth, knowledge)
                            max_k = 128 if "forward_k128" in arms_list else 32
                            max_bwd_k = 64 if "joint_true_k128" in arms_list else 16

                            setup_id = f"qwen3-seed{seed}-L{depth}-{knowledge}"
                            setup_rec = {
                                "id": setup_id,
                                "victim_seed": seed,
                                "depth": depth,
                                "knowledge": knowledge,
                                "loading_seconds": kb["loading_seconds"],
                                "calibration_work": kb["calibration_work"],
                                "ridge_lambda": kb["ridge_lambda"],
                            }
                            report["setup"].append(setup_rec)

                            # Precompute top candidate indices once across all batches for this (depth, knowledge)
                            torch.cuda.synchronize()
                            t0_cand = time.perf_counter()
                            front_max_chunks = []
                            back_true_max_chunks = []
                            back_shuf_16_chunks = []
                            bwd_argmax_chunks = []
                            vocab_len = int(kb["E"].shape[0])
                            with torch.no_grad():
                                for start in range(0, len(hidden), ATTACK_BATCH_SIZE):
                                    stop = start + ATTACK_BATCH_SIZE
                                    h_b = hidden[start:stop]
                                    g_b = gradient[start:stop]
                                    gs_b = g_shuf[start:stop]

                                    fwd_s = F.normalize(h_b @ kb["W_dec"], dim=-1) @ kb["E_norm_T"]
                                    front_max_chunks.append(fwd_s.topk(max_k, dim=-1).indices)
                                    del fwd_s

                                    bwd_s = torch.zeros((*g_b.shape[:2], vocab_len), device=g_b.device)
                                    bwd_s[:, 0, majority_tok] = 1.0
                                    bwd_s[:, 1:] = F.normalize(-g_b[:, :-1], dim=-1) @ kb["E_norm_T"]
                                    bwd_argmax_chunks.append(bwd_s.argmax(dim=-1))
                                    back_true_max_chunks.append(bwd_s.topk(max_bwd_k, dim=-1).indices)
                                    del bwd_s

                                    bwd_shuf_s = torch.zeros((*gs_b.shape[:2], vocab_len), device=gs_b.device)
                                    bwd_shuf_s[:, 0, majority_tok] = 1.0
                                    bwd_shuf_s[:, 1:] = F.normalize(-gs_b[:, :-1], dim=-1) @ kb["E_norm_T"]
                                    back_shuf_16_chunks.append(bwd_shuf_s.topk(16, dim=-1).indices)
                                    del bwd_shuf_s
                            torch.cuda.synchronize()
                            cand_precomp_s = time.perf_counter() - t0_cand

                            front_max_all = torch.cat(front_max_chunks, dim=0)
                            back_true_max_all = torch.cat(back_true_max_chunks, dim=0)
                            back_shuf_16_all = torch.cat(back_shuf_16_chunks, dim=0)
                            bwd_argmax_all = torch.cat(bwd_argmax_chunks, dim=0)

                            front32_all = front_max_all[..., :32]
                            back16_all = back_true_max_all[..., :16]
                            cand_sets = {
                                "backward": None,
                                "forward_k32": front32_all,
                                "joint_true_k32": _merge_candidates_from_top(front32_all, back16_all, 32),
                                "joint_shuffled_k32": _merge_candidates_from_top(front32_all, back_shuf_16_all, 32),
                            }
                            if max_k == 128:
                                cand_sets["forward_k128"] = front_max_all
                                cand_sets["joint_true_k128"] = _merge_candidates_from_top(front_max_all, back_true_max_all, 128)

                            cell_predictions = {}
                            for arm in arms_list:
                                t0_arm = time.perf_counter()
                                with kb["head"].measure() as work:
                                    if arm == "backward":
                                        work.vocabulary_dot_products += int(len(hidden) * (hidden.shape[1] - 1) * vocab_len)
                                        preds = bwd_argmax_all.detach().cpu().tolist()
                                    else:
                                        cands_arm = cand_sets[arm]
                                        work.vocabulary_dot_products += int(len(hidden) * hidden.shape[1] * vocab_len)
                                        if arm.startswith("joint"):
                                            work.vocabulary_dot_products += int(len(hidden) * (hidden.shape[1] - 1) * vocab_len)
                                        preds_batches = []
                                        for start in range(0, len(hidden), ATTACK_BATCH_SIZE):
                                            stop = start + ATTACK_BATCH_SIZE
                                            p_b, _ = _fast_qwen3_kv_beam_reconstruct(
                                                kb["head"],
                                                hidden[start:stop],
                                                cands_arm[start:stop],
                                                BEAM_WIDTH,
                                            )
                                            preds_batches.append(p_b.detach().cpu())
                                        preds = torch.cat(preds_batches, dim=0).tolist()
                                torch.cuda.synchronize()
                                wall_s = time.perf_counter() - t0_arm + (cand_precomp_s / len(arms_list))
                                work_dict = asdict(work)

                                per_doc, summary = _score_predictions_32(eval_rows, preds, metrics_module)
                                cand_rec = _candidate_recall_from_tensor(cand_sets[arm], eval_ids)
                                setup_cost = kb["loading_seconds"] + (
                                    kb["calibration_work"]["wall_seconds"] if arm.startswith(("forward", "joint")) else 0.0
                                )

                                cell_id = f"qwen3-seed{seed}-L{depth}-{knowledge}-{arm}"
                                cell_rec = {
                                    "cell_id": cell_id,
                                    "setup_id": setup_id,
                                    "victim_seed": seed,
                                    "depth": depth,
                                    "knowledge": knowledge,
                                    "attack_seed": ATTACK_SEED,
                                    "arm": arm,
                                    "work": work_dict,
                                    "wall_seconds": wall_s,
                                    "cold_seconds": setup_cost + wall_s,
                                    "amortized_seconds_per_document": wall_s / len(eval_rows),
                                    "candidate_recall": cand_rec,
                                    "summary": summary,
                                }
                                report["cells"].append(cell_rec)
                                ev.event("cell_complete", cell_rec)

                                cell_predictions[arm] = {
                                    "cell_id": cell_id,
                                    "summary": summary,
                                    "candidate_recall": cand_rec,
                                    "work": work_dict,
                                    "predicted_tokens": preds,
                                    "per_document": per_doc,
                                }

                            sidecar_name = f"qwen3_seed{seed}_L{depth}_{knowledge}.json"
                            sidecar_path = output / "predictions" / sidecar_name
                            _dump(sidecar_path, {
                                "setup_id": setup_id,
                                "victim_seed": seed,
                                "depth": depth,
                                "knowledge": knowledge,
                                "document_ids": [r["id"] for r in eval_rows],
                                "arms": cell_predictions,
                            })
                            sidecar_sha = digest(sidecar_path)
                            for c_rec in report["cells"][-len(arms_list):]:
                                c_rec["sidecar_file"] = f"predictions/{sidecar_name}"
                                c_rec["sidecar_sha256"] = sidecar_sha

                            _dump(report_path, report)
                            log_msg = {
                                "event": "qwen3_setup_complete",
                                "seed": seed,
                                "depth": depth,
                                "knowledge": knowledge,
                                "bwd": round(cell_predictions["backward"]["summary"]["token_accuracy_mean"], 4),
                                "fwd_k32": round(cell_predictions["forward_k32"]["summary"]["token_accuracy_mean"], 4),
                                "jnt_k32": round(cell_predictions["joint_true_k32"]["summary"]["token_accuracy_mean"], 4),
                                "shuf_k32": round(cell_predictions["joint_shuffled_k32"]["summary"]["token_accuracy_mean"], 4),
                                "wall_s": round(time.perf_counter() - t0_setup, 2),
                            }
                            if "forward_k128" in cell_predictions:
                                log_msg["fwd_k128"] = round(cell_predictions["forward_k128"]["summary"]["token_accuracy_mean"], 4)
                                log_msg["jnt_k128"] = round(cell_predictions["joint_true_k128"]["summary"]["token_accuracy_mean"], 4)
                            print(json.dumps(log_msg), flush=True)

                            del front_max_all, back_true_max_all, back_shuf_16_all, bwd_argmax_all, cand_sets
                            gc.collect()
                            torch.cuda.empty_cache()

                        del hidden, gradient, g_shuf, exact_head, exact_W_dec
                        gc.collect()
                        torch.cuda.empty_cache()

                    del victim, optim, exact_E, exact_E_norm_T
                    gc.collect()
                    torch.cuda.empty_cache()
            finally:
                baselines.beam_reconstruct = orig_beam_reconstruct

            report["status"] = "COMPLETE"
            report["elapsed_seconds"] = time.monotonic() - started
            ev.event("qwen3_recon_complete", {
                "status": "COMPLETE",
                "elapsed_seconds": report["elapsed_seconds"],
                "n_victims": len(report["victims"]),
                "n_cells": len(report["cells"]),
            })
            _dump(report_path, report)
        except Exception as exc:
            report["status"] = "FAILED"
            report["elapsed_seconds"] = time.monotonic() - started
            report["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
            ev.event("qwen3_recon_failed", report["error"])
            _dump(report_path, report)
            raise

    ok, problems = verify_evidence(str(report_path) + ".evidence.json")
    if not ok:
        raise RuntimeError(f"Evidence verification failed: {problems}")
    return report
