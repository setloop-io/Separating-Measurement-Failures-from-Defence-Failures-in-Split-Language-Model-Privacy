"""Training loop, model initialization, and the single-cell runner.

Determinism contract:
  * trusted init (backbone, encoder/decoder) is drawn from a torch.Generator
    keyed by the run seed — identical for every run sharing that seed;
  * remote (surrogate) init is drawn from a generator keyed by
    pairing_block_id — bit-identical across trained/frozen pairs of one block,
    different across blocks;
  * the data stream is keyed by seed only (see data.py), so paired runs see
    identical batches;
  * fp32 everywhere; all init happens on CPU before moving to the device.

Optimizer: AdamW (lr 3e-4, betas (0.9, 0.95), wd 0.1 with no decay on
norms/biases), linear warmup 100 steps, cosine decay to 3e-5, grad clip 1.0.
Frozen parameters never enter the optimizer.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import (
    CanaryBatchSampler,
    draw_canary_strings,
    load_corpus,
    make_eval_batches,
    make_probe_batches,
    make_train_sampler,
    split_corpus,
)
from .instrument import RemoteInstrument
from .modes import apply_mode, has_split, remote_parameters, trainable_parameters
from .model import Backbone, BackboneConfig, init_module_
from .split import AffineQuantizer, RemotePath, SplitBackbone, SplitConfig, build_model
from .transcript import (
    TRANSCRIPT_SCHEMA_VERSION,
    TranscriptCapture,
    build_inventory,
    validate_inventory,
)
from .utils import derive_seed, parameters_checksum, resolve_device, sha256_json


class OptimConfig:
    def __init__(
        self,
        lr: float = 3e-4,
        min_lr: float = 3e-5,
        betas: tuple = (0.9, 0.95),
        weight_decay: float = 0.1,
        warmup_steps: int = 100,
        grad_clip: float = 1.0,
        steps: int = 2000,
    ):
        self.lr = lr
        self.min_lr = min_lr
        self.betas = tuple(betas)
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.grad_clip = grad_clip
        self.steps = steps

    def lr_at(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.lr * (step + 1) / self.warmup_steps
        progress = (step - self.warmup_steps) / max(1, self.steps - self.warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return self.min_lr + (self.lr - self.min_lr) * cosine


def build_initialized_model(
    backbone_cfg: BackboneConfig,
    mode: str,
    split: SplitConfig | None,
    seed: int,
    pairing_block_id: str | None,
) -> torch.nn.Module:
    """Build on CPU and initialize from the seed/pairing generators."""
    model = build_model(backbone_cfg, mode, split)
    run_gen = torch.Generator().manual_seed(derive_seed("run-init", seed))
    if isinstance(model, Backbone):
        init_module_(model, run_gen, backbone_cfg.n_layers, backbone_cfg.init_std)
    else:
        assert isinstance(model, SplitBackbone)
        remote_gen = torch.Generator().manual_seed(
            derive_seed("remote-init", pairing_block_id)
        )
        init_module_(model.tok_emb, run_gen, backbone_cfg.n_layers, backbone_cfg.init_std)
        init_module_(model.pos_emb, run_gen, backbone_cfg.n_layers, backbone_cfg.init_std)
        for block in model.blocks:
            if isinstance(block, RemotePath):
                # trusted side: keyed by run seed; remote side: keyed by pairing block id
                init_module_(block.encoder_ln, run_gen, backbone_cfg.n_layers, backbone_cfg.init_std)
                init_module_(block.encoder, run_gen, backbone_cfg.n_layers, backbone_cfg.init_std)
                init_module_(block.decoder, run_gen, backbone_cfg.n_layers, backbone_cfg.init_std)
                init_module_(block.remote, remote_gen, 1, backbone_cfg.init_std)
            else:
                init_module_(block, run_gen, backbone_cfg.n_layers, backbone_cfg.init_std)
        init_module_(model.ln_f, run_gen, backbone_cfg.n_layers, backbone_cfg.init_std)
        init_module_(model.head, run_gen, backbone_cfg.n_layers, backbone_cfg.init_std)
        # additive (P5): attach the dedicated grad-defense noise stream when the
        # split config asks for it; no effect on init draws or existing configs
        if split is not None and split.grad_defense is not None:
            for block in model.blocks:
                if isinstance(block, RemotePath):
                    block._grad_defense_gen = torch.Generator().manual_seed(
                        derive_seed("grad-defense", seed)
                    )
    apply_mode(model, mode)
    return model


def build_optimizer(model: torch.nn.Module, cfg: OptimConfig) -> torch.optim.AdamW:
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue  # frozen remote params never enter the optimizer
        (decay if p.ndim >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=cfg.betas)


def evaluate(model, eval_batches, device) -> float:
    total, count = 0.0, 0
    with torch.no_grad():
        for x, y in eval_batches:
            logits = model(x.to(device))
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1).to(device))
            total += loss.item()
            count += 1
    return total / max(1, count)


class CalibrationCapture(TranscriptCapture):
    """Calibration-cohort capture: stores raw (pre-quantization) latent/token pairs."""

    wants_raw_latents = True

    def __init__(self, quantizer: AffineQuantizer, capture_every: int, max_pairs: int):
        super().__init__(quantizer)
        self.capture_every = capture_every
        self.max_pairs = max_pairs
        self.events: list[dict] = []
        self.n_pairs = 0
        self.current_step = 0
        self.current_tokens = None

    def on_raw_latents(self, z_pre: torch.Tensor) -> None:
        if self.current_step % self.capture_every != 0 or self.n_pairs >= self.max_pairs:
            return
        latents = z_pre.detach().to("cpu", torch.float32)
        tokens = self.current_tokens.detach().to("cpu")
        self.events.append({"step": self.current_step, "latents": latents, "tokens": tokens})
        self.n_pairs += latents.numel() // latents.shape[-1]


def train_model(
    model,
    sampler,
    optim_cfg: OptimConfig,
    device: torch.device,
    capture: TranscriptCapture | None = None,
    instrument: RemoteInstrument | None = None,
    eval_batches=None,
    eval_every: int = 100,
    probe_batch=None,
    transcript_sample_every: int | None = None,
    ema_beta: float = 0.95,
) -> dict:
    """Core loop shared by run_cell and the P0 self-tests."""
    optimizer = build_optimizer(model, optim_cfg)
    tokens_per_step = sampler.batch_size * sampler.seq_len
    curve: list[dict] = []
    probe_samples: list[dict] = []
    remote_state_checksums: list[str] = []
    smoothed: float | None = None
    t0 = time.perf_counter()

    def curve_row(step: int, eval_loss: float | None) -> dict:
        row = {
            "step": step,
            "train_loss_smoothed": smoothed,
            "eval_loss": eval_loss,
            "cumulative_remote_flops": instrument.snapshot() if instrument else {"fwd_only": 0, "fwd_bwd": 0},
            "cumulative_bytes": dict(capture.total_bytes) if capture else {},
            "wall_time": time.perf_counter() - t0,
        }
        curve.append(row)
        if capture is not None:
            remote = remote_parameters(model)
            remote_state_checksums.append(parameters_checksum(remote) if remote else None)
        return row

    curve_row(0, evaluate(model, eval_batches, device) if eval_batches else None)

    for step in range(optim_cfg.steps):
        for group in optimizer.param_groups:
            group["lr"] = optim_cfg.lr_at(step)
        if capture is not None:
            if hasattr(capture, "current_step"):
                capture.current_step = step
            capture.begin_step()
        x, y = sampler.batch(step)
        if capture is not None and getattr(capture, "wants_raw_latents", False):
            capture.current_tokens = x
        x, y = x.to(device), y.to(device)
        logits = model(x, capture=capture)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_parameters(model), optim_cfg.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if capture is not None:
            capture.end_step()
        if instrument is not None:
            instrument.record_step(tokens_per_step)
        loss_value = loss.item()
        smoothed = loss_value if smoothed is None else ema_beta * smoothed + (1 - ema_beta) * loss_value

        steps_done = step + 1
        if probe_batch is not None and transcript_sample_every and steps_done % transcript_sample_every == 0:
            # additive: probe_batch may be a list (capture_probe_batches > 1);
            # batch 0 is identical to the single-batch case
            batches = probe_batch if isinstance(probe_batch, (list, tuple)) else [probe_batch]
            for batch_index, pb in enumerate(batches):
                sample = capture_probe_tensors(model, pb, capture, device, steps_done)
                sample["probe_batch_index"] = batch_index
                probe_samples.append(sample)
        if eval_batches and (steps_done % eval_every == 0 or steps_done == optim_cfg.steps):
            curve_row(steps_done, evaluate(model, eval_batches, device))

    return {
        "curve": curve,
        "probe_samples": probe_samples,
        "remote_state_checksums": remote_state_checksums,
        "wall_time_seconds": time.perf_counter() - t0,
        "final_train_loss_smoothed": smoothed,
    }


def capture_probe_tensors(model, probe_batch, capture, device, step: int) -> dict:
    """Re-run forward/backward on the fixed probe batch; measurement only."""
    sink: dict = {}
    x, y = probe_batch
    x, y = x.to(device), y.to(device)
    was_counting = capture is not None
    if capture is not None:
        capture.begin_sink(sink)
    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        logits = model(x, capture=capture)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        loss.backward()
    if capture is not None:
        capture.end_sink()
    model.zero_grad(set_to_none=True)
    sample = {"step": step, "probe_loss": loss.item()}
    sample.update(sink)
    return sample


def run_cell(resolved: dict) -> dict:
    """Execute one resolved config end-to-end; write artifacts and manifest."""
    from .manifest import begin_run, write_manifest  # deferred

    begin_run(resolved)
    device = resolve_device(resolved.get("device", "auto"))
    out_dir = Path(resolved["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    backbone_cfg = BackboneConfig(**resolved["backbone"])
    split_cfg = SplitConfig(**resolved["split"]) if resolved["split"] else None
    mode = resolved["mode"]
    seed = resolved["seed"]
    steps = resolved["optim"]["steps"]

    data_cfg = resolved["data"]
    corpus = load_corpus(data_cfg["path"])
    train_region, heldout_region = split_corpus(corpus, data_cfg["train_fraction"])
    sampler = make_train_sampler(train_region, data_cfg["batch_size"], data_cfg["seq_len"], seed)
    canary_cfg = resolved.get("canary")
    canary_strings: tuple | None = None
    if canary_cfg:
        canary_strings = draw_canary_strings(
            canary_cfg["count"], canary_cfg["length"], canary_cfg["seed"]
        )
        if canary_cfg.get("inject"):
            sampler = CanaryBatchSampler(
                train_region, data_cfg["batch_size"], data_cfg["seq_len"], seed,
                label="train", canary=canary_cfg,
            )
    eval_batches = make_eval_batches(
        heldout_region, data_cfg["eval_batches"], data_cfg["batch_size"], data_cfg["seq_len"], seed
    )
    probe_batch = None
    if has_split(mode):
        # additive: capture_probe_batches (default 1) covers N batches of the
        # fixed probe stream per sampled step; measurement only, no RNG or
        # optimizer interaction (train_model consumes none for captures)
        probe_batch = make_probe_batches(
            heldout_region, resolved["logging"]["probe_batch_size"], data_cfg["seq_len"],
            resolved["logging"].get("capture_probe_batches", 1),
        )

    model = build_initialized_model(
        backbone_cfg, mode, split_cfg, seed, resolved.get("pairing_block_id")
    ).to(device)

    capture = None
    if has_split(mode):
        quantizer = AffineQuantizer(split_cfg.q_bits)
        if resolved.get("cohort") == "calibration":
            cal = resolved.get("calibration", {})
            capture = CalibrationCapture(
                quantizer, cal.get("capture_every", 5), cal.get("capture_max_pairs", 131072)
            )
        else:
            capture = TranscriptCapture(quantizer)
    instrument = RemoteInstrument(
        split_cfg.width if split_cfg else 0, data_cfg["seq_len"], mode,
        remote_depth=split_cfg.remote_depth if split_cfg else 1,
        model_shape={"n_layers": backbone_cfg.n_layers, "d_model": backbone_cfg.d_model,
                     "d_ff": backbone_cfg.d_ff,
                     "span": split_cfg.span if split_cfg else 0},
    )

    optim_cfg = OptimConfig(
        lr=resolved["optim"]["lr"],
        min_lr=resolved["optim"]["min_lr"],
        betas=tuple(resolved["optim"]["betas"]),
        weight_decay=resolved["optim"]["weight_decay"],
        warmup_steps=resolved["optim"]["warmup_steps"],
        grad_clip=resolved["optim"]["grad_clip"],
        steps=steps,
    )

    result = train_model(
        model,
        sampler,
        optim_cfg,
        device,
        capture=capture,
        instrument=instrument,
        eval_batches=eval_batches,
        eval_every=resolved["logging"]["eval_every"],
        probe_batch=probe_batch,
        transcript_sample_every=resolved["logging"]["transcript_sample_every"],
        ema_beta=resolved["logging"].get("ema_beta", 0.95),
    )

    # ---- artifacts -------------------------------------------------------
    config_path = out_dir / "config.resolved.json"
    config_path.write_text(json.dumps(resolved, indent=2, sort_keys=True))

    curve_path = out_dir / "curve.jsonl"
    with curve_path.open("w") as handle:
        for row in result["curve"]:
            handle.write(json.dumps(row) + "\n")

    artifacts = [config_path, curve_path]
    captured_channels: list[str] = []

    if has_split(mode):
        tensors_path = out_dir / "transcript_tensors.pt"
        torch.save(result["probe_samples"], tensors_path)
        artifacts.append(tensors_path)

        inventory = build_inventory(
            capture.total_bytes,
            result["remote_state_checksums"],
            has_remote_state=bool(remote_parameters(model)),
        )
        validate_inventory(inventory)
        inventory_path = out_dir / "transcript_inventory.json"
        inventory_path.write_text(json.dumps(inventory, indent=2, sort_keys=True))
        artifacts.append(inventory_path)
        captured_channels = [
            ch for ch, flag in inventory["channels"].items() if flag and ch != "timing"
        ]

        steps_path = out_dir / "transcript_bytes_per_step.json"
        steps_path.write_text(json.dumps(capture.step_bytes))
        artifacts.append(steps_path)

        if isinstance(capture, CalibrationCapture):
            pairs_path = out_dir / "calibration_pairs.pt"
            torch.save(capture.events, pairs_path)
            artifacts.append(pairs_path)

    # additive (P4a): when a canary block is present, save the member and
    # held-out non-member strings plus their final-model boundary latents
    # (one no-grad probe pass; no effect on training)
    if canary_strings is not None:
        members, non_members = canary_strings
        latents = None
        if has_split(mode) and capture is not None:
            sink: dict = {}
            capture.begin_sink(sink)
            with torch.no_grad():
                model(torch.cat([members, non_members]).long().to(device), capture=capture)
            capture.end_sink()
            latents = sink["forward_values"].cpu()
        canary_path = out_dir / "canaries.pt"
        torch.save(
            {
                "members": members,
                "non_members": non_members,
                "member_latents": None if latents is None else latents[: len(members)],
                "non_member_latents": None if latents is None else latents[len(members):],
            },
            canary_path,
        )
        artifacts.append(canary_path)

    total_tokens = steps * data_cfg["batch_size"] * data_cfg["seq_len"]
    byte_totals = capture.total_bytes if capture else {}
    checksums = result["remote_state_checksums"]
    remote_initial = checksums[0] if checksums else None
    remote_final = checksums[-1] if checksums else None
    summary = {
        "run_id": resolved["run_id"],
        "cohort": resolved.get("cohort", "exploratory"),
        "mode": mode,
        "seed": seed,
        "pairing_block_id": resolved.get("pairing_block_id"),
        "steps": steps,
        "device": str(device),
        "final_eval_loss": result["curve"][-1]["eval_loss"],
        "final_train_loss_smoothed": result["final_train_loss_smoothed"],
        "eval_series": [
            {"step": r["step"], "eval_loss": r["eval_loss"]} for r in result["curve"]
        ],
        "wall_time_seconds": result["wall_time_seconds"],
        "cumulative_remote_flops": instrument.snapshot(),
        "cumulative_bytes": byte_totals,
        "bits_per_token": {
            ch: byte_totals[ch] * 8 / total_tokens for ch in byte_totals
        },
        "bits_per_token_total": sum(byte_totals.values()) * 8 / total_tokens if byte_totals else 0.0,
        "remote_param_sha256_initial": remote_initial,
        "remote_param_sha256_final": remote_final,
        "transcript_schema_version": TRANSCRIPT_SCHEMA_VERSION,
        "note": (f"Recorded {resolved.get('cohort', 'exploratory')} cohort; qualification and "
                 "claim support require the corresponding protocol and evidence review."),
    }
    summary_path = out_dir / "metrics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    artifacts.append(summary_path)

    # additive opt-in checkpoint for downstream probes (P3); no effect on
    # training math, RNG consumption, or any other artifact
    if resolved.get("save_final_state"):
        state_path = out_dir / "final_state.pt"
        torch.save(model.state_dict(), state_path)
        artifacts.append(state_path)

    manifest_path = write_manifest(
        resolved,
        status="completed",
        artifact_paths=artifacts,
        captured_channels=captured_channels,
        extra={
            "remote_param_sha256_initial": remote_initial,
            "remote_param_sha256_final": remote_final,
        },
    )
    summary["manifest"] = str(manifest_path)
    return summary
