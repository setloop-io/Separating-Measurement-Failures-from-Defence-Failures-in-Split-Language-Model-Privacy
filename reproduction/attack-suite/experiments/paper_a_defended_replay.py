"""Defended joint and public-grant adaptive attack replay for Paper A.

Addresses R6 Major 4 and R4 Major 2 across two complementary regimes:
1. Primary Held-Out Checkpoint Defences (L=6, public client grant, 10 victim seeds
   101..1010, 256 evaluation passages, k=32, beam_width=2):
   - replica-median (K=3 coordinate-median channels):
     backward@replica-median, forward_k32@replica-median (verified bit-exact on
     host anchor seed against frozen v3 transcripts), joint_true_k32@replica-median.
   - secret-mixup (lambda=0.75 recycled-decoy activation mixup with victim-side
     gradient correction):
     backward@secret-mixup, forward_k32@secret-mixup (verified bit-exact on host
     anchor seed against frozen v3 transcripts), joint_true_k32@secret-mixup,
     adaptive_demix_forward_k32@secret-mixup (public-lambda calibration-mean
     demixing), and adaptive_demix_joint_k32@secret-mixup (demixing + joint
     candidates + gradient-assisted beam scoring).
   - trained-bottleneck (d=768 -> k=64 learned adversarial/MINE bottleneck encoder
     with public cloud decoder):
     non-adaptive raw-wire (backward@trained-bottleneck,
     forward_k32@trained-bottleneck, joint_true_k32@trained-bottleneck) and
     defence-aware public-decoder (adaptive_decode_backward@trained-bottleneck,
     adaptive_decode_forward_k32@trained-bottleneck,
     adaptive_decode_joint_k32@trained-bottleneck).
2. 500-Step Live Training Trajectory Defence (convergence-mid-lite-{11,22,33},
   L=6, public client grant, d=768 -> k=64 random projection, 3 seeds x 500 steps):
   - backward@mid-lite, forward_k32@mid-lite, joint_true_k32@mid-lite, paired
     step-for-step with Experiment #1 (convergence-none-{11,22,33}).

Host partitioning:
- toula: victim seeds [101, 202, 303, 404, 505] (anchor 101) + mid-lite steps 0..247
- koula: victim seeds [606, 707, 808, 909, 1010] (anchor 606) + mid-lite steps 248..499
"""
from __future__ import annotations

import copy
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import random
import socket
import time
import traceback


HOSTS = ["toula", "koula"]
HOST_SEEDS = {
    "toula": [101, 202, 303, 404, 505],
    "koula": [606, 707, 808, 909, 1010],
}
HOST_ANCHOR_SEED = {
    "toula": 101,
    "koula": 606,
}
HOST_STEP_RANGES = {
    "toula": (0, 248),
    "koula": (248, 500),
}
TRAJ_SEEDS = [11, 22, 33]


def _dump(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _candidate_recall(batches_details, truth_tensor):
    """Compute fraction of (document, position) tokens present in the top-k candidate set."""
    import torch
    cands = [torch.tensor(b["candidate_ids"], dtype=torch.long) for b in batches_details if "candidate_ids" in b]
    if not cands:
        return None
    all_cands = torch.cat(cands, dim=0)
    truth_cpu = truth_tensor.detach().cpu()
    hits = (all_cands == truth_cpu.unsqueeze(-1)).any(dim=-1).to(dtype=torch.float32)
    return {
        "overall": float(hits.mean().item()),
        "pos0": float(hits[:, 0].mean().item()),
        "pos1_31": float(hits[:, 1:].mean().item()),
        "per_document": [float(x) for x in hits.mean(dim=-1).tolist()],
    }


def _beam_reconstruct_custom(head, target_hidden, candidates, beam_width, bwd_scores=None, grad_weight=0.0, proj_basis=None):
    """Autoregressive beam search with optional backward-gradient scoring and subspace projection.

    When bwd_scores is provided with grad_weight > 0, each token step combines the
    forward activation residual with the standardized backward token score for that
    candidate token. When proj_basis is provided (e.g. [768, 64] public decoder basis
    for trained-bottleneck), activation residuals are evaluated in the 64-d decoded
    subspace where the wire signal lives.
    """
    import torch
    bsz, seq_len, k = candidates.shape
    device = target_hidden.device
    if proj_basis is not None:
        target_proj = target_hidden @ proj_basis
    else:
        target_proj = target_hidden

    if bwd_scores is not None and grad_weight > 0.0:
        cand_bwd = torch.gather(bwd_scores, -1, candidates)
        bwd_mean = cand_bwd.mean(dim=-1, keepdim=True)
        bwd_std = cand_bwd.std(dim=-1, keepdim=True).clamp_min(1e-6)
        cand_bwd_z = (cand_bwd - bwd_mean) / bwd_std
    else:
        cand_bwd_z = None

    beams = candidates[:, 0, :beam_width].unsqueeze(-1)
    if proj_basis is not None:
        pred_0 = head(ids=beams.reshape(-1, 1)).reshape(bsz, beam_width, 1, -1) @ proj_basis
    else:
        pred_0 = head(ids=beams.reshape(-1, 1)).reshape(bsz, beam_width, 1, -1)
    res_0 = (pred_0[:, :, 0] - target_proj[:, 0:1]).pow(2).mean(dim=-1)
    if cand_bwd_z is not None:
        res_0 = res_0 - grad_weight * cand_bwd_z[:, 0, :beam_width]

    for pos in range(1, seq_len):
        b_cur = beams.shape[1]
        cand_pos = candidates[:, pos, :]
        expanded = torch.cat([
            beams.unsqueeze(2).expand(bsz, b_cur, k, pos),
            cand_pos.unsqueeze(1).expand(bsz, b_cur, k).unsqueeze(-1),
        ], dim=-1)
        flat = expanded.reshape(bsz * b_cur * k, pos + 1)
        pred = head(ids=flat)[:, -1, :].reshape(bsz, b_cur, k, -1)
        if proj_basis is not None:
            pred = pred @ proj_basis
        step_res = (pred - target_proj[:, pos:pos + 1].unsqueeze(1)).pow(2).mean(dim=-1)
        if cand_bwd_z is not None:
            step_res = step_res - grad_weight * cand_bwd_z[:, pos, :].unsqueeze(1)
        total_res = res_0.unsqueeze(-1) + step_res
        flat_res = total_res.reshape(bsz, b_cur * k)
        best_res, best_idx = torch.topk(flat_res, k=min(beam_width, b_cur * k), dim=-1, largest=False, sorted=True)
        gather_idx = best_idx.unsqueeze(-1).expand(bsz, best_idx.shape[1], pos + 1)
        beams = torch.gather(expanded.reshape(bsz, b_cur * k, pos + 1), 1, gather_idx)
        res_0 = best_res

    best_seq = beams[:, 0, :]
    final_pred = head(ids=best_seq)
    if proj_basis is not None:
        final_res = float(((final_pred @ proj_basis) - target_proj).pow(2).sum().item())
    else:
        final_res = float((final_pred - target_hidden).pow(2).sum().item())
    return best_seq, {"forward_residual_sum": final_res, "beam_width": int(beam_width)}


def _attack_custom(head, hidden_for_fwd, hidden_for_beam, gradient, embedding, decoder, majority,
                   arm_kind, k, beam_width, attack_batch_size, grad_weight=0.0, proj_basis=None):
    """ Batched attack helper supporting adaptive demixing and public-decoder subspace scoring."""
    import torch
    from attacker.reconstruction_baselines import forward_scores, backward_scores, candidate_ids
    predictions = []
    batches = []
    for start in range(0, len(hidden_for_fwd), attack_batch_size):
        stop = start + attack_batch_size
        h_fwd = hidden_for_fwd[start:stop]
        h_beam = hidden_for_beam[start:stop]
        with torch.no_grad():
            if arm_kind == "backward":
                bwd = backward_scores(gradient[start:stop], embedding, head, majority)
                pred = bwd.argmax(dim=-1)
                batch_details = {}
            else:
                fwd = forward_scores(h_fwd, embedding, decoder, head)
                if arm_kind == "forward":
                    bwd = None
                    cands = candidate_ids(fwd, None, k, "forward")
                elif arm_kind == "joint":
                    bwd = backward_scores(gradient[start:stop], embedding, head, majority)
                    cands = candidate_ids(fwd, bwd, k, "joint")
                else:
                    raise ValueError(f"Unsupported arm_kind: {arm_kind}")
                pred, batch_details = _beam_reconstruct_custom(
                    head, h_beam, cands, beam_width,
                    bwd_scores=bwd, grad_weight=grad_weight, proj_basis=proj_basis,
                )
                batch_details["candidate_ids"] = cands.cpu().tolist()
        predictions.extend(pred.detach().cpu().tolist())
        batches.append(batch_details)
    return predictions, {"batches": batches, "grad_weight": grad_weight, "subspace_projected": proj_basis is not None}


def _score_predictions(evaluation_rows, predictions, metrics_module):
    """Compute per-document and summary metrics across evaluation rows."""
    per_doc = [
        {"document_id": row["id"], **metrics_module.reconstruction_metrics(row["tokens"], pred, row.get("annotations", []))}
        for row, pred in zip(evaluation_rows, predictions)
    ]
    n = len(per_doc)
    tok_acc = sum(d["token_accuracy"] for d in per_doc) / n
    pos0_acc = sum(1.0 if row["tokens"][0] == pred[0] else 0.0 for row, pred in zip(evaluation_rows, predictions)) / n
    pos1_31_acc = sum(
        sum(1.0 if a == b else 0.0 for a, b in zip(row["tokens"][1:], pred[1:])) / 31.0
        for row, pred in zip(evaluation_rows, predictions)
    ) / n
    exact = sum(1.0 if d["sequence_exact"] else 0.0 for d in per_doc) / n
    longest_run = sum(float(d["longest_exact_run"]) for d in per_doc) / n
    span8 = sum(float(d["contiguous_spans"]["8"]["rate"] or 0.0) for d in per_doc) / n
    name_exact = sum(int(d["annotated_spans"]["name"]["exact"]) for d in per_doc)
    name_total = sum(int(d["annotated_spans"]["name"]["total"]) for d in per_doc)
    num_exact = sum(int(d["annotated_spans"]["number"]["exact"]) for d in per_doc)
    num_total = sum(int(d["annotated_spans"]["number"]["total"]) for d in per_doc)
    return per_doc, {
        "n_documents": n,
        "token_accuracy_mean": tok_acc,
        "pos0_accuracy_mean": pos0_acc,
        "pos1_31_accuracy_mean": pos1_31_acc,
        "exact_match_mean": exact,
        "sequence_exact_mean": exact,
        "longest_exact_run_mean": longest_run,
        "span8_rate_mean": span8,
        "name_span_rate": (name_exact / name_total) if name_total > 0 else None,
        "number_span_rate": (num_exact / num_total) if num_total > 0 else None,
    }


def run_defended_replay(plan, root, output, resolve_asset, digest):
    """Execute the host's assigned defended held-out and 500-step mid-lite cells."""
    root, output = Path(root), Path(output)
    host = socket.gethostname().split(".")[0].lower()
    if host not in HOSTS:
        raise ValueError(f"Unexpected execution host: {host}")
    host_seeds = HOST_SEEDS[host]
    anchor_seed = HOST_ANCHOR_SEED[host]
    step_start, step_stop = HOST_STEP_RANGES[host]

    output.mkdir(parents=True, exist_ok=False)
    (output / "predictions").mkdir(parents=True, exist_ok=True)
    (output / "mid_lite_predictions").mkdir(parents=True, exist_ok=True)
    (output / "scratch_captures").mkdir(parents=True, exist_ok=True)
    report_path = output / "defended_replay.json"
    started = time.monotonic()

    report = {
        "schema": "paper-a.defended-expansion-replay.v1",
        "status": "RUNNING",
        "phase": plan["phase"],
        "host": host,
        "victim_seeds": host_seeds,
        "anchor_seed": anchor_seed,
        "mid_lite_step_range": [step_start, step_stop],
        "utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": plan["source_commit"],
        "runner_sha256": digest(__file__),
        "plan": plan,
        "defended_victims": [],
        "anchor_verifications": [],
        "heldout_cells": [],
        "mid_lite_setups": [],
        "mid_lite_cells": [],
        "failed_cells": [],
    }
    _dump(report_path, report)

    from attacker.evidence import EvidenceLogger, verify_evidence
    with EvidenceLogger(report_path, node_role=host + "/gpu", extra={"study": plan["study"], "phase": plan["phase"]}) as ev:
        try:
            import torch
            import transformers
            from transformers import GPT2LMHeadModel
            from attacker import splitbench as sb
            from attacker import trained_defenses as td
            from attacker.attacks import seam_inversion as si
            import attacker.reconstruction_baselines as baselines
            import attacker.reconstruction_metrics as metrics_module
            import validate_reconstruction as validation_module

            frozen = root / "experiments/paper_a_stronger_frozen"
            source = json.loads((frozen / "SOURCE.json").read_text())
            if source["source_commit"] != plan["source_commit"]:
                raise ValueError("Frozen source commit differs from replay plan")
            for name, sha in source["files"].items():
                if digest(frozen / name) != sha:
                    raise ValueError("Frozen source bytes differ: " + name)
            for module in (sb, td, si, baselines, metrics_module, validation_module):
                if not Path(module.__file__).resolve().is_relative_to(frozen.resolve()):
                    raise ValueError("Replay imported non-frozen module: " + module.__name__)

            torch.set_num_threads(1)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.manual_seed(93281)
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable for defended expansion replay")

            report["environment"] = {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "memory_total_bytes": torch.cuda.get_device_properties(0).total_memory,
                "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                "tf32_cudnn": torch.backends.cudnn.allow_tf32,
            }

            # Verify all staged defended expansion pins before starting
            def_dir = Path(plan["defended_directory"])
            def_pins_path = def_dir / "PINS.json"
            def_pins = json.loads(def_pins_path.read_text())
            report["defended_pins_sha256"] = digest(def_pins_path)
            for fname, sha in def_pins["observations"].items():
                p_obs = def_dir / "observations" / fname
                if digest(p_obs) != sha:
                    raise ValueError(f"Observation hash mismatch: {fname}")
            for fname, sha in def_pins["transcripts"].items():
                p_tr = def_dir / "transcripts" / fname
                if digest(p_tr) != sha:
                    raise ValueError(f"Transcript hash mismatch: {fname}")
            for rel, sha in def_pins["mid_lite"].items():
                p_ml = def_dir / "mid-lite-500step" / rel
                if digest(p_ml) != sha:
                    raise ValueError(f"mid-lite hash mismatch: {rel}")

            pins_path = root / "experiments/v4_adaptation_expected.json"
            pins = json.loads(pins_path.read_text())
            report["asset_pins_sha256"] = digest(pins_path)

            input_path = Path(resolve_asset(plan, "input.json", plan["input_sha256"]))
            if digest(input_path) != plan["input_sha256"]:
                raise ValueError("Input hash mismatch")
            data = json.loads(input_path.read_text())
            calibration_rows = data["calibration"]
            train_rows = data["train"]
            evaluation_rows = data["evaluation"]
            if len(calibration_rows) != 32 or len(train_rows) != 128 or len(evaluation_rows) != 256:
                raise ValueError("Unexpected partition counts in input.json")

            calibration = torch.tensor([r["tokens"] for r in calibration_rows], device="cuda", dtype=torch.long)
            train_tensor = torch.tensor([r["tokens"] for r in train_rows], device="cuda", dtype=torch.long)
            evaluation = torch.tensor([r["tokens"] for r in evaluation_rows], device="cuda", dtype=torch.long)

            hf_snap = Path.home() / ".cache/huggingface/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e"
            if not hf_snap.is_dir():
                raise FileNotFoundError(f"Offline GPT-2 snapshot missing: {hf_snap}")

            ev.event("defended_replay_start", {k: v for k, v in report.items() if k not in ("heldout_cells", "mid_lite_cells")})
            _dump(report_path, report)

            depth = 6
            knowledge = "public"
            attack_seed = 17
            k_cand = 32
            beam_width = 2
            attack_batch_size = 8
            runtime_cfg = {"attack_batch_size": attack_batch_size, "beam_width": beam_width}

            # =========================================================================
            # PART 1-3: HELD-OUT CHECKPOINT DEFENCES (replica-median, secret-mixup, trained-bottleneck)
            # =========================================================================
            for seed in host_seeds:
                torch.manual_seed(seed)
                base_model = GPT2LMHeadModel.from_pretrained(str(hf_snap), local_files_only=True)
                order = list(range(len(train_tensor)))
                random.Random(seed).shuffle(order)
                train_batches = [train_tensor[i:i + 1] for i in order]

                head_name = f"public-{seed}-{depth}.pt"
                head_path = Path(resolve_asset(plan, head_name, pins[head_name]))
                dsplit, dembedding = si._load_head_pt(str(head_path), "cuda")
                dsplit.model.set_attn_implementation("sdpa")
                dsplit.model.eval().requires_grad_(False)
                dembedding = dembedding.detach()
                dhead = baselines.MeteredHead(dsplit)
                with torch.no_grad():
                    dhead(ids=calibration[:1])
                    dhc_3d = dhead(ids=calibration).detach()
                    dhc_2d = dhc_3d.reshape(-1, dembedding.shape[-1])
                    ddecoder, ridge_lambda = si._ridge_fit(dhc_2d, dembedding[calibration.flatten()], 0)
                    mu_cal = dhc_2d.mean(dim=0).view(1, 1, -1).detach()
                dmajority = si.majority_control_token(calibration.tolist())

                def record_heldout_cell(defense, arm, predictions, diagnostics, work_dict, extra=None):
                    cell_name = f"{seed}-{depth}-{knowledge}-{attack_seed}-training-{arm}"
                    pred_file = output / "predictions" / f"{cell_name}.json"
                    _dump(pred_file, {
                        "cell_id": cell_name,
                        "victim_seed": seed,
                        "depth": depth,
                        "knowledge": knowledge,
                        "attack_seed": attack_seed,
                        "defense": defense,
                        "arm": arm,
                        "predictions": predictions,
                        "document_ids": [r["id"] for r in evaluation_rows],
                    })
                    per_doc, summary = _score_predictions(evaluation_rows, predictions, metrics_module)
                    cand_rec = _candidate_recall(diagnostics.get("batches", []), evaluation)
                    cell = {
                        "cell_id": cell_name,
                        "status": "COMPLETE",
                        "victim_seed": seed,
                        "depth": depth,
                        "knowledge": knowledge,
                        "attack_seed": attack_seed,
                        "defense": defense,
                        "arm": arm,
                        "k": k_cand if "k32" in arm else None,
                        "beam_width": beam_width,
                        "work": work_dict,
                        "amortized_seconds_per_document": work_dict["wall_seconds"] / len(evaluation_rows),
                        "prediction_file": f"predictions/{pred_file.name}",
                        "prediction_sha256": digest(pred_file),
                        "candidate_recall": cand_rec,
                        "summary": summary,
                        "metrics": per_doc,
                    }
                    if extra:
                        cell.update(extra)
                    report["heldout_cells"].append(cell)
                    ev.event("heldout_cell_complete", {
                        "cell_id": cell_name,
                        "defense": defense,
                        "arm": arm,
                        "token_accuracy_mean": summary["token_accuracy_mean"],
                        "exact_match_mean": summary["exact_match_mean"],
                        "candidate_recall": cand_rec["overall"] if cand_rec else None,
                        "wall_seconds": work_dict["wall_seconds"],
                    })
                    _dump(report_path, report)
                    print(json.dumps({
                        "cell_id": cell_name,
                        "token_accuracy": round(summary["token_accuracy_mean"], 6),
                        "exact_match": round(summary["exact_match_mean"], 6),
                        "candidate_recall": round(cand_rec["overall"], 6) if cand_rec else None,
                        "wall_s": round(work_dict["wall_seconds"], 2),
                    }), flush=True)

                # -----------------------------------------------------------------
                # 1. replica-median & 2. secret-mixup
                # -----------------------------------------------------------------
                for defense in ("replica-median", "secret-mixup"):
                    obs_name = f"observations-{defense}-{seed}-{depth}.pt"
                    obs_path = def_dir / "observations" / obs_name
                    ref_h = torch.load(obs_path, map_location="cuda", weights_only=True)["h"].to(dtype=torch.float32)
                    if ref_h.shape != (256, 32, 768):
                        raise ValueError(f"Unexpected observation shape for {obs_name}: {ref_h.shape}")

                    tr_name = f"{seed}-{depth}-public-17-training-forward_k32@{defense}.json"
                    ref_tr = json.loads((def_dir / "transcripts" / tr_name).read_text())

                    # Train defended victim with identical RNG seed and batch order as validate_reconstruction.py
                    torch.manual_seed(seed)
                    victim = copy.deepcopy(base_model).to("cuda")
                    split = sb.build_split(victim, sb.SplitPlan.parse(str(depth), n_layers=victim.config.n_layer))
                    cap_dir = output / "scratch_captures" / f"defended-{defense}-{seed}-{depth}"
                    cap_dir.mkdir(parents=True, exist_ok=True)
                    cfg = sb.BenchConfig(
                        split=str(depth), mode="finetune", steps=128, seq_len=32, batch_size=1,
                        lr=5e-5, defense=defense, seed=seed, save_canonical=True, device="cuda",
                        capture_dir=str(cap_dir / "captures"), output=str(cap_dir / "run.json"),
                    ).validate()
                    writer = sb.CaptureWriter(cfg.capture_dir, "training", f"defended-exp-{defense}-{seed}-{depth}")
                    retained_state = {}
                    measured_utility = {}

                    def retain(stage, current_split, current_cfg, state):
                        from experiments.pilot_measurements import utility as matched_utility
                        measured_utility[stage] = matched_utility(
                            current_split, current_cfg, state,
                            [row[None] for row in evaluation], train_batches,
                        )
                        if stage == "after":
                            retained_state.update(state)

                    torch.cuda.synchronize()
                    t_train0 = time.perf_counter()
                    try:
                        sb.execute(cfg, split, writer, train_batches, measurement=retain, capture_replica_input=True)
                    finally:
                        writer.close()
                    torch.cuda.synchronize()
                    train_seconds = time.perf_counter() - t_train0

                    victim.eval()
                    dstate = retained_state
                    with torch.no_grad():
                        if defense == "secret-mixup":
                            for ids in train_batches:
                                dstate["mixup_ids"] = ids
                                sb._apply_seam_defense(split.head_forward(ids.to("cuda")), cfg, dstate)
                    victim.requires_grad_(False)

                    # Compute freshly captured wire (to check agreement with frozen v3 ref_h)
                    # and true seam backward gradient g wrt leaf = ref_h_chunk
                    fresh_wires = []
                    gs = []
                    for idx, chunk in enumerate(evaluation.split(attack_batch_size)):
                        s0 = idx * attack_batch_size
                        s1 = s0 + len(chunk)
                        with torch.no_grad():
                            h0 = split.head_forward(chunk).detach()
                            if defense == "replica-median":
                                w_chunk = h0
                            else:
                                dstate["mixup_ids"] = chunk
                                w_chunk = sb._apply_seam_defense(h0, cfg, dstate)
                            fresh_wires.append(w_chunk.detach())

                        leaf = ref_h[s0:s1].detach().clone().requires_grad_(True)
                        if defense == "replica-median":
                            outs = [f(leaf) for f in dstate["replica"].channel_forwards]
                            median = td.coordinate_median(outs)
                            logits = split.lm_head(split.ln_f(median))
                        else:
                            logits = sb._server_rest_forward(split, leaf, cfg, dstate)
                        loss = torch.nn.functional.cross_entropy(
                            logits[:, :-1].reshape(-1, logits.shape[-1]),
                            chunk[:, 1:].reshape(-1),
                            reduction="sum",
                        ) / (len(evaluation) * (evaluation.shape[1] - 1))
                        g_chunk = torch.autograd.grad(loss, leaf)[0].detach()
                        gs.append(g_chunk)
                        del logits, leaf, loss

                    fresh_h = torch.cat(fresh_wires, dim=0)
                    g = torch.cat(gs, dim=0)
                    wire_max_diff = float((fresh_h - ref_h).abs().max().item())

                    report["defended_victims"].append({
                        "seed": seed,
                        "depth": depth,
                        "defense": defense,
                        "training_seconds": train_seconds,
                        "evaluation_nll": measured_utility["after"]["cross_entropy"],
                        "utility": {
                            "utility_before": measured_utility["before"],
                            "utility_after": measured_utility["after"],
                        },
                        "ref_observation_sha256": def_pins["observations"][obs_name],
                        "fresh_vs_ref_wire_max_abs_diff": wire_max_diff,
                        "grad_norm_mean": float(g.reshape(len(evaluation), -1).norm(dim=-1).mean().item()),
                    })

                    # Arm 1: backward@{defense}
                    torch.cuda.synchronize()
                    with dhead.measure() as work_bwd:
                        pred_bwd, diag_bwd = validation_module.attack_observations(
                            dhead, ref_h, g, dembedding, ddecoder, dmajority,
                            f"backward@{defense}", runtime_cfg, attack_seed,
                        )
                    record_heldout_cell(defense, f"backward@{defense}", pred_bwd, diag_bwd, asdict(work_bwd))

                    # Arm 2: forward_k32@{defense} (execute & verify bit-exact on anchor_seed; score verified v3 transcript on all seeds)
                    if seed == anchor_seed:
                        torch.cuda.synchronize()
                        with dhead.measure() as work_fwd:
                            pred_fwd_exec, diag_fwd_exec = validation_module.attack_observations(
                                dhead, ref_h, None, dembedding, ddecoder, dmajority,
                                f"forward_k32@{defense}", runtime_cfg, attack_seed,
                            )
                        exact_preds = (pred_fwd_exec == ref_tr["predictions"])
                        ref_cands = [b["candidate_ids"] for b in ref_tr["diagnostics"]["batches"]]
                        exec_cands = [b["candidate_ids"] for b in diag_fwd_exec["batches"]]
                        exact_cands = (exec_cands == ref_cands)
                        max_res_diff = max(
                            abs(float(va) - float(vb))
                            for a, b in zip(diag_fwd_exec["batches"], ref_tr["diagnostics"]["batches"])
                            for va, vb in zip(a["forward_residual_sum"], b["forward_residual_sum"])
                        )
                        anchor_check = {
                            "seed": seed,
                            "defense": defense,
                            "arm": f"forward_k32@{defense}",
                            "transcript_sha256": def_pins["transcripts"][tr_name],
                            "exact_predictions": exact_preds,
                            "exact_candidates": exact_cands,
                            "max_residual_abs_diff": max_res_diff,
                            "wall_seconds": work_fwd.wall_seconds,
                        }
                        report["anchor_verifications"].append(anchor_check)
                        if not exact_preds or not exact_cands:
                            raise ValueError(f"Anchor verification failed for {defense} on seed {seed}: {anchor_check}")
                        fwd_work_dict = asdict(work_fwd)
                        fwd_extra = {"anchor_verified_bit_exact": True, "transcript_sha256": def_pins["transcripts"][tr_name]}
                    else:
                        fwd_work_dict = {"wall_seconds": 0.0, "forward_calls": 0, "tokens_processed": 0}
                        fwd_extra = {"verified_v3_transcript_reused": True, "transcript_sha256": def_pins["transcripts"][tr_name]}

                    record_heldout_cell(
                        defense, f"forward_k32@{defense}",
                        ref_tr["predictions"], ref_tr["diagnostics"],
                        fwd_work_dict, extra=fwd_extra,
                    )

                    # Arm 3: joint_true_k32@{defense}
                    torch.cuda.synchronize()
                    with dhead.measure() as work_jnt:
                        pred_jnt, diag_jnt = validation_module.attack_observations(
                            dhead, ref_h, g, dembedding, ddecoder, dmajority,
                            f"joint_true_k32@{defense}", runtime_cfg, attack_seed,
                        )
                    record_heldout_cell(defense, f"joint_true_k32@{defense}", pred_jnt, diag_jnt, asdict(work_jnt))

                    # Public-grant defence-aware adaptive arms for secret-mixup
                    if defense == "secret-mixup":
                        mix_lam = 0.75
                        h_demix = ((ref_h - (1.0 - mix_lam) * mu_cal) / mix_lam).detach()

                        # Arm 4: adaptive_demix_forward_k32@secret-mixup
                        torch.cuda.synchronize()
                        with dhead.measure() as work_afwd:
                            pred_afwd, diag_afwd = _attack_custom(
                                dhead, h_demix, h_demix, None, dembedding, ddecoder, dmajority,
                                arm_kind="forward", k=k_cand, beam_width=beam_width,
                                attack_batch_size=attack_batch_size, grad_weight=0.0, proj_basis=None,
                            )
                        record_heldout_cell(
                            defense, "adaptive_demix_forward_k32@secret-mixup",
                            pred_afwd, diag_afwd, asdict(work_afwd),
                            extra={"adaptive_mechanism": "public_lambda_calibration_mean_demix", "mixup_lambda": mix_lam},
                        )

                        # Arm 5: adaptive_demix_joint_k32@secret-mixup
                        torch.cuda.synchronize()
                        with dhead.measure() as work_ajnt:
                            pred_ajnt, diag_ajnt = _attack_custom(
                                dhead, h_demix, h_demix, g, dembedding, ddecoder, dmajority,
                                arm_kind="joint", k=k_cand, beam_width=beam_width,
                                attack_batch_size=attack_batch_size, grad_weight=0.25, proj_basis=None,
                            )
                        record_heldout_cell(
                            defense, "adaptive_demix_joint_k32@secret-mixup",
                            pred_ajnt, diag_ajnt, asdict(work_ajnt),
                            extra={"adaptive_mechanism": "public_lambda_demix_plus_grad_beam", "mixup_lambda": mix_lam, "grad_weight": 0.25},
                        )

                    del victim, split, dstate, fresh_h, g
                    torch.cuda.empty_cache()

                # -----------------------------------------------------------------
                # 3. trained-bottleneck (L=6, public, k=64 learned bottleneck)
                # -----------------------------------------------------------------
                defense = "trained-bottleneck"
                torch.manual_seed(seed)
                victim = copy.deepcopy(base_model).to("cuda")
                split = sb.build_split(victim, sb.SplitPlan.parse(str(depth), n_layers=victim.config.n_layer))
                cap_dir = output / "scratch_captures" / f"defended-{defense}-{seed}-{depth}"
                cap_dir.mkdir(parents=True, exist_ok=True)
                cfg = sb.BenchConfig(
                    split=str(depth), mode="finetune", steps=128, seq_len=32, batch_size=1,
                    lr=5e-5, defense=defense, seed=seed, save_canonical=False, device="cuda",
                    tb_dim=64, tb_minimax_steps=200, tb_mi_weight=0.1,
                    capture_dir=str(cap_dir / "captures"), output=str(cap_dir / "run.json"),
                ).validate()
                writer = sb.CaptureWriter(cfg.capture_dir, "training", f"defended-exp-{defense}-{seed}-{depth}")
                retained_state = {}
                measured_utility = {}

                def retain_tb(stage, current_split, current_cfg, state):
                    from experiments.pilot_measurements import utility as matched_utility
                    measured_utility[stage] = matched_utility(
                        current_split, current_cfg, state,
                        [row[None] for row in evaluation], train_batches,
                    )
                    if stage == "after":
                        retained_state.update(state)

                torch.cuda.synchronize()
                t_tb0 = time.perf_counter()
                try:
                    sb.execute(cfg, split, writer, train_batches, measurement=retain_tb)
                finally:
                    writer.close()
                torch.cuda.synchronize()
                tb_train_seconds = time.perf_counter() - t_tb0

                victim.eval()
                victim.requires_grad_(False)
                tb = retained_state["tb"]
                basis = tb.geometry["basis"].detach().to(device="cuda", dtype=torch.float32)
                tb_mean = tb.geometry["mean"].detach().to(device="cuda", dtype=torch.float32)

                hs_tb, gs_tb = [], []
                for chunk in evaluation.split(attack_batch_size):
                    with torch.no_grad():
                        h0 = split.head_forward(chunk).detach()
                        w_chunk = tb.apply(h0).detach()
                    leaf = w_chunk.clone().requires_grad_(True)
                    h_srv = tb.decode(leaf)
                    logits = split.rest_forward(h_srv)
                    loss = torch.nn.functional.cross_entropy(
                        logits[:, :-1].reshape(-1, logits.shape[-1]),
                        chunk[:, 1:].reshape(-1),
                        reduction="sum",
                    ) / (len(evaluation) * (evaluation.shape[1] - 1))
                    g_chunk = torch.autograd.grad(loss, leaf)[0].detach()
                    hs_tb.append(w_chunk)
                    gs_tb.append(g_chunk)
                    del logits, leaf, loss

                h_wire_tb = torch.cat(hs_tb, dim=0)
                g_wire_tb = torch.cat(gs_tb, dim=0)

                # Decoded 768-d representations using public cloud decoder tb.decode
                h_dec_tb = tb.decode(h_wire_tb).detach()
                g_dec_tb = (g_wire_tb[..., :tb.bottleneck_dim] @ basis.T).detach()

                # Fit bottleneck-aware ridge decoders on the 32 public calibration passages:
                # (a) dec_pca: strictly public-grant, uses ONLY the public cloud decoder (basis, mean)
                # (b) dec_tb: diagnostic check passing calibration through tb.decode(tb.apply(hc))
                with torch.no_grad():
                    hc_pca = ((dhc_3d - tb_mean) @ basis @ basis.T + tb_mean).reshape(-1, 768)
                    dec_pca, _ = si._ridge_fit(hc_pca, dembedding[calibration.flatten()], 0)
                    hc_tb = tb.decode(tb.apply(dhc_3d)).reshape(-1, 768)
                    dec_tb, _ = si._ridge_fit(hc_tb, dembedding[calibration.flatten()], 0)
                    fwd_pca_top1 = float((baselines.forward_scores(h_dec_tb, dembedding, dec_pca, dhead).argmax(-1) == evaluation).to(torch.float32).mean().item())
                    fwd_tb_top1 = float((baselines.forward_scores(h_dec_tb, dembedding, dec_tb, dhead).argmax(-1) == evaluation).to(torch.float32).mean().item())

                report["defended_victims"].append({
                    "seed": seed,
                    "depth": depth,
                    "defense": defense,
                    "training_seconds": tb_train_seconds,
                    "evaluation_nll": measured_utility["after"]["cross_entropy"],
                    "utility": {
                        "utility_before": measured_utility["before"],
                        "utility_after": measured_utility["after"],
                    },
                    "bottleneck_dim": int(tb.bottleneck_dim),
                    "grad_norm_mean": float(g_wire_tb.reshape(len(evaluation), -1).norm(dim=-1).mean().item()),
                    "public_decoder_vs_encoder_cal_top1": {
                        "dec_pca_top1": fwd_pca_top1,
                        "dec_tb_top1": fwd_tb_top1,
                        "abs_diff": abs(fwd_pca_top1 - fwd_tb_top1),
                    },
                })

                # Non-adaptive raw-wire arms on trained-bottleneck
                torch.cuda.synchronize()
                with dhead.measure() as work_tb_bwd:
                    pred_tb_bwd, diag_tb_bwd = validation_module.attack_observations(
                        dhead, h_wire_tb, g_wire_tb, dembedding, ddecoder, dmajority,
                        "backward@trained-bottleneck", runtime_cfg, attack_seed,
                    )
                record_heldout_cell(defense, "backward@trained-bottleneck", pred_tb_bwd, diag_tb_bwd, asdict(work_tb_bwd))

                torch.cuda.synchronize()
                with dhead.measure() as work_tb_fwd:
                    pred_tb_fwd, diag_tb_fwd = validation_module.attack_observations(
                        dhead, h_wire_tb, None, dembedding, ddecoder, dmajority,
                        "forward_k32@trained-bottleneck", runtime_cfg, attack_seed,
                    )
                record_heldout_cell(defense, "forward_k32@trained-bottleneck", pred_tb_fwd, diag_tb_fwd, asdict(work_tb_fwd))

                torch.cuda.synchronize()
                with dhead.measure() as work_tb_jnt:
                    pred_tb_jnt, diag_tb_jnt = validation_module.attack_observations(
                        dhead, h_wire_tb, g_wire_tb, dembedding, ddecoder, dmajority,
                        "joint_true_k32@trained-bottleneck", runtime_cfg, attack_seed,
                    )
                record_heldout_cell(defense, "joint_true_k32@trained-bottleneck", pred_tb_jnt, diag_tb_jnt, asdict(work_tb_jnt))

                # Strictly public-grant defence-aware adaptive arms on trained-bottleneck
                # Uses ONLY the public cloud decoder (basis, tb_mean) and dec_pca (zero queries to secret tb.encode)
                # 1. adaptive_decode_backward@trained-bottleneck (projects wire gradient via public basis.T)
                torch.cuda.synchronize()
                with dhead.measure() as work_atb_bwd:
                    pred_atb_bwd, diag_atb_bwd = _attack_custom(
                        dhead, h_dec_tb, h_dec_tb, g_dec_tb, dembedding, dec_pca, dmajority,
                        arm_kind="backward", k=k_cand, beam_width=beam_width,
                        attack_batch_size=attack_batch_size,
                    )
                record_heldout_cell(
                    defense, "adaptive_decode_backward@trained-bottleneck",
                    pred_atb_bwd, diag_atb_bwd, asdict(work_atb_bwd),
                    extra={"adaptive_mechanism": "public_cloud_decoder_gradient_projection"},
                )

                # 2. adaptive_decode_forward_k32@trained-bottleneck
                torch.cuda.synchronize()
                with dhead.measure() as work_atb_fwd:
                    pred_atb_fwd, diag_atb_fwd = _attack_custom(
                        dhead, h_dec_tb, h_dec_tb, None, dembedding, dec_pca, dmajority,
                        arm_kind="forward", k=k_cand, beam_width=beam_width,
                        attack_batch_size=attack_batch_size, grad_weight=0.0, proj_basis=basis,
                    )
                record_heldout_cell(
                    defense, "adaptive_decode_forward_k32@trained-bottleneck",
                    pred_atb_fwd, diag_atb_fwd, asdict(work_atb_fwd),
                    extra={"adaptive_mechanism": "strict_public_cloud_decoder_plus_subspace_beam"},
                )

                # 3. adaptive_decode_joint_k32@trained-bottleneck
                torch.cuda.synchronize()
                with dhead.measure() as work_atb_jnt:
                    pred_atb_jnt, diag_atb_jnt = _attack_custom(
                        dhead, h_dec_tb, h_dec_tb, g_dec_tb, dembedding, dec_pca, dmajority,
                        arm_kind="joint", k=k_cand, beam_width=beam_width,
                        attack_batch_size=attack_batch_size, grad_weight=0.25, proj_basis=basis,
                    )
                record_heldout_cell(
                    defense, "adaptive_decode_joint_k32@trained-bottleneck",
                    pred_atb_jnt, diag_atb_jnt, asdict(work_atb_jnt),
                    extra={"adaptive_mechanism": "strict_public_cloud_decoder_plus_subspace_grad_beam", "grad_weight": 0.25},
                )

                del victim, split, tb, h_wire_tb, g_wire_tb, h_dec_tb, g_dec_tb
                del dsplit, dhead, dembedding, ddecoder, dhc_3d, dhc_2d, base_model
                torch.cuda.empty_cache()

            # =========================================================================
            # PART 4: 500-STEP LIVE TRAINING TRAJECTORY UNDER mid-lite (seeds 11, 22, 33)
            # =========================================================================
            pub101_path = Path(resolve_asset(plan, "public-101-6.pt", pins["public-101-6.pt"]))
            tsplit, tembedding = si._load_head_pt(str(pub101_path), "cuda")
            tsplit.model.set_attn_implementation("sdpa")
            tsplit.model.eval().requires_grad_(False)
            tembedding = tembedding.detach()
            thead = baselines.MeteredHead(tsplit)
            with torch.no_grad():
                thead(ids=calibration[:1])
                thc = thead(ids=calibration).reshape(-1, 768)
                tdecoder, tridge_lambda = si._ridge_fit(thc, tembedding[calibration.flatten()], 0)
            del thc
            tmajority = si.majority_control_token(calibration.tolist())

            cal_set = {tuple(r["tokens"]) for r in calibration_rows}
            cal_grams = [{tuple(r["tokens"][i:i + 5]) for i in range(28)} for r in calibration_rows]

            ml_root = def_dir / "mid-lite-500step"
            for tseed in TRAJ_SEEDS:
                seed_dir = ml_root / f"convergence-mid-lite-{tseed}"
                cap_ev = seed_dir / "capture.json.evidence.json"
                can_pt = seed_dir / "captures/canonical.pt"
                ok, errors = verify_evidence(cap_ev)
                if not ok:
                    raise ValueError(f"mid-lite evidence verification failed for seed {tseed}: {errors}")

                canonical_all = torch.load(can_pt, map_location="cpu", weights_only=True)
                if canonical_all.shape != (500, 32) or canonical_all.dtype != torch.int64:
                    raise ValueError(f"Unexpected canonical.pt shape/dtype for mid-lite seed {tseed}")

                all_rows = [tuple(r) for r in canonical_all.tolist()]
                if any(r in cal_set for r in all_rows):
                    raise ValueError(f"Exact calibration overlap in mid-lite seed {tseed}")
                for r in all_rows:
                    g5 = {tuple(r[i:i + 5]) for i in range(28)}
                    if any(len(g5 & cg) / len(g5 | cg) >= 0.8 for cg in cal_grams):
                        raise ValueError(f"Near-duplicate calibration overlap in mid-lite seed {tseed}")

                step_indices = list(range(step_start, step_stop))
                doc_ids = [f"seed{tseed}-step{s:04d}" for s in step_indices]
                hs_ml, gs_ml, h_norms, g_norms = [], [], [], []
                for s in step_indices:
                    f_meta = json.loads((seed_dir / "captures" / f"wire_{2 * s:05d}.json").read_text())
                    b_meta = json.loads((seed_dir / "captures" / f"wire_{2 * s + 1:05d}.json").read_text())
                    if f_meta.get("phase") != "fwd" or f_meta.get("step") != s:
                        raise ValueError(f"mid-lite fwd metadata mismatch at seed {tseed} step {s}")
                    if b_meta.get("phase") != "bwd" or b_meta.get("step") != s:
                        raise ValueError(f"mid-lite bwd metadata mismatch at seed {tseed} step {s}")
                    h_t = torch.load(seed_dir / "captures" / f"wire_{2 * s:05d}.pt", map_location="cpu", weights_only=True).to(dtype=torch.float32)
                    g_t = torch.load(seed_dir / "captures" / f"wire_{2 * s + 1:05d}.pt", map_location="cpu", weights_only=True).to(dtype=torch.float32)
                    if h_t.shape != (1, 32, 768) or g_t.shape != (1, 32, 768):
                        raise ValueError(f"mid-lite wire shape mismatch at seed {tseed} step {s}")
                    hs_ml.append(h_t)
                    gs_ml.append(g_t)
                    h_norms.append(float(h_t.norm().item()))
                    g_norms.append(float(g_t.norm().item()))

                hidden_ml = torch.cat(hs_ml, dim=0).to("cuda")
                gradient_ml = torch.cat(gs_ml, dim=0).to("cuda")
                truth_slice = canonical_all[step_start:step_stop]
                truth_rows = [{"id": doc_id, "tokens": tok, "annotations": []} for doc_id, tok in zip(doc_ids, truth_slice.tolist())]

                report["mid_lite_setups"].append({
                    "victim_seed": tseed,
                    "defense": "mid-lite",
                    "step_range": [step_start, step_stop],
                    "n_steps": len(step_indices),
                    "forward_wire_norm_mean": sum(h_norms) / len(h_norms),
                    "backward_wire_norm_mean": sum(g_norms) / len(g_norms),
                })

                for arm in ("backward@mid-lite", "forward_k32@mid-lite", "joint_true_k32@mid-lite"):
                    cid = f"traj-mid-lite-{tseed}-6-public-17-{host}-{arm}-beam2"
                    torch.cuda.synchronize()
                    with thead.measure() as work_ml:
                        pred_ml, diag_ml = validation_module.attack_observations(
                            thead, hidden_ml, gradient_ml if not arm.startswith("forward") else None,
                            tembedding, tdecoder, tmajority, arm, runtime_cfg, attack_seed,
                        )
                    torch.cuda.synchronize()
                    pred_file = output / "mid_lite_predictions" / f"{cid}.json"
                    _dump(pred_file, {
                        "cell_id": cid,
                        "victim_seed": tseed,
                        "defense": "mid-lite",
                        "host": host,
                        "step_range": [step_start, step_stop],
                        "arm": arm,
                        "document_ids": doc_ids,
                        "predictions": pred_ml,
                    })
                    per_doc_ml, summary_ml = _score_predictions(truth_rows, pred_ml, metrics_module)
                    cand_rec_ml = _candidate_recall(diag_ml.get("batches", []), truth_slice)
                    for idx_s, s in enumerate(step_indices):
                        per_doc_ml[idx_s]["step"] = s
                        per_doc_ml[idx_s]["forward_wire_norm"] = h_norms[idx_s]
                        per_doc_ml[idx_s]["backward_wire_norm"] = g_norms[idx_s]
                        if cand_rec_ml:
                            per_doc_ml[idx_s]["candidate_recall"] = cand_rec_ml["per_document"][idx_s]

                    cell_ml = {
                        "cell_id": cid,
                        "status": "COMPLETE",
                        "victim_seed": tseed,
                        "defense": "mid-lite",
                        "host": host,
                        "step_range": [step_start, step_stop],
                        "n_steps": len(step_indices),
                        "arm": arm,
                        "work": asdict(work_ml),
                        "amortized_seconds_per_step": work_ml.wall_seconds / len(step_indices),
                        "prediction_file": f"mid_lite_predictions/{pred_file.name}",
                        "prediction_sha256": digest(pred_file),
                        "candidate_recall": {k: v for k, v in cand_rec_ml.items() if k != "per_document"} if cand_rec_ml else None,
                        "summary": summary_ml,
                        "step_metrics": per_doc_ml,
                    }
                    report["mid_lite_cells"].append(cell_ml)
                    ev.event("mid_lite_cell_complete", {
                        "cell_id": cid,
                        "victim_seed": tseed,
                        "arm": arm,
                        "token_accuracy_mean": summary_ml["token_accuracy_mean"],
                        "exact_match_mean": summary_ml["exact_match_mean"],
                        "wall_seconds": work_ml.wall_seconds,
                    })
                    _dump(report_path, report)
                    print(json.dumps({
                        "cell_id": cid,
                        "token_accuracy": round(summary_ml["token_accuracy_mean"], 6),
                        "exact_match": round(summary_ml["exact_match_mean"], 6),
                        "wall_s": round(work_ml.wall_seconds, 2),
                    }), flush=True)

                del hidden_ml, gradient_ml
                torch.cuda.empty_cache()

            report["status"] = "DEFENDED_REPLAY_COMPLETE"
            report["elapsed_seconds"] = time.monotonic() - started
            _dump(report_path, report)
            ev.finalize(report_path)
            return report
        except BaseException as error:
            report.update(
                status="FAILED",
                error=repr(error),
                traceback=traceback.format_exc(),
                elapsed_seconds=time.monotonic() - started,
            )
            _dump(report_path, report)
            ev.event("failure", {"error": repr(error)})
            ev.finalize(report_path)
            raise
