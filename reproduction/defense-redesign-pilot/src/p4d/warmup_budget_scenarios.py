"""Shared-warmup budget engineering on the pinned Paper 4 corpus; engineering only.

Protocol `p4-warmup-budget-0.1` (docs/WARMUP_BUDGET_PROTOCOL.md): one shared
frozen-random warmup per seed, then paired trained/frozen/removed/local branches
at the full 2,000-step pilot budget. `full_budget: false` exists only so tests can
run the pipeline on tiny synthetic corpora; the protocol config sets it true and
validation then pins every budget constant. Anchor split, seeds, protocol ID, and
branch list are always enforced.
"""
from copy import deepcopy
from dataclasses import replace
import time

import numpy as np
import torch
import yaml
from pcr.data import BatchSampler
from pcr.model import BackboneConfig
from pcr.modes import apply_mode, remote_parameters, trainable_parameters
from pcr.privacy_attacks_p4b import pearson_r, ridge_fit, ridge_predict, sequence_targets
from pcr.split import RemovedRemote, SplitConfig
from pcr.train import OptimConfig, build_initialized_model, build_optimizer
from pcr.utils import derive_seed, parameters_checksum

from .calibration import estimate_property_directions
from .corpus import (contiguous_region, document_panel_record, document_text_panel,
                     load_document_index, load_pinned_corpus)
from .devices import resolve_device
from .experiment import ROOT, digest, evaluate, measurement, write_json
from .model import attach_defense
from .scenario_io import Campaign, require_engineering, validate_seed_namespace
from .warmup_scenarios import (FLAGS, OMITTED, TARGET, basis_cosine, panel_batches,
                               stability_curve, tensor_hash, trusted_checksum)

PROTOCOL_ID = "p4-warmup-budget-0.1"
SEEDS = [410501, 410502]
ANCHOR = {"start": 2, "span": 4, "width": 256, "remote_depth": 1, "q_bits": 8}
CORPUS_SHA256 = "af3d0245c0ec27d7021f5fd29256bc845bb20cf9c562b37bbe23663b6ebe7436"
BRANCHES = [{"branch": "d0", "mode": "trained"},
            {"branch": "d1_s0.1", "mode": "trained"},
            {"branch": "d1_s1.0", "mode": "trained"},
            {"branch": "d3_q1", "mode": "trained"},
            {"branch": "d0", "mode": "frozen_random"},
            {"branch": "d0", "mode": "removed_remote"},
            {"branch": "all_local", "mode": "all_local"}]
BUDGET = {"warmup_steps": 500, "branch_steps": 1500, "local_steps": 2000,
          "basis_windows": 4096, "probe_windows": 16384,
          "panel_sizes": [1024, 4096, 16384], "null_repeats": 16,
          "noise_scales": {"d1_s0.1": 0.1, "d1_s1.0": 1.0}}
OPTIMIZER = {"lr": 0.0003, "min_lr": 0.00003, "warmup_steps": 100,
             "weight_decay": 0.1, "grad_clip": 1.0, "betas": [0.9, 0.95]}


def boundary_means(model, panel, batch_size, seed, device, released=True):
    """Exact selected forward view; do not execute the remote module or suffix."""
    means = []
    with measurement(model, seed), torch.no_grad():
        for x, _ in panel_batches(panel, batch_size):
            x = x.to(device)
            h = model.tok_emb(x) + model.pos_emb(torch.arange(x.shape[1], device=x.device))
            for block in model.blocks[:model.split.start]:
                h = block(h)
            path = model.remote_path
            z = path.encoder(path.encoder_ln(h))
            if released:
                z, _ = path.quantizer(path.defense(z))
            means.append(z.mean(1).cpu())
    return torch.cat(means)


def basis_snapshot(model, fit, validation, batch_size, seed, device):
    x = boundary_means(model, fit, batch_size, seed, device, released=False).numpy()
    xt = boundary_means(model, validation, batch_size, seed, device, released=False).numpy()
    y = sequence_targets(fit["tokens"])[TARGET]
    yt = sequence_targets(validation["tokens"])[TARGET]
    basis = estimate_property_directions(x, {TARGET: y})
    probe = ridge_fit(x, y)
    return basis, {"basis_sha256": tensor_hash(basis), "rank": len(basis),
                   "validation_r": pearson_r(ridge_predict(probe, xt), yt),
                   "fit_sequences": len(y), "validation_sequences": len(yt)}


def train_steps(model, sampler, optimizer_cfg, steps, device, offset=0):
    """A new optimizer is an explicit intervention at every warmup/branch boundary."""
    config = OptimConfig(**optimizer_cfg, steps=steps)
    optimizer = build_optimizer(model, config)
    curve = []
    model.train()
    for step in range(steps):
        for group in optimizer.param_groups:
            group["lr"] = config.lr_at(step)
        x, y = sampler.batch(offset + step)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(x).reshape(-1, 256), y.reshape(-1))
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_parameters(model), config.grad_clip)
        optimizer.step()
        curve.append({"step": offset + step + 1, "train_loss": float(loss.detach())})
    model.zero_grad(set_to_none=True)
    return curve


def basis_split(documents, seed):
    """Seeded disjoint document halves of the calibration region."""
    order = np.random.default_rng(derive_seed("warmup-budget-basis-split", seed)).permutation(len(documents))
    half = len(documents) // 2
    return (sorted((documents[i] for i in order[:half]), key=lambda d: d["offset"]),
            sorted((documents[i] for i in order[half:]), key=lambda d: d["offset"]))


def build_panels(cfg, index, corpus, seed):
    seq = cfg["data"]["sequence_length"]
    fit_docs, validation_docs = basis_split(index["calibration"], seed)
    docs = {"basis_fit": fit_docs, "basis_validation": validation_docs,
            "probe_fit": index["probe_fit"], "probe_test": index["evaluation"]}
    counts = {"basis_fit": cfg["basis_windows"], "basis_validation": cfg["basis_windows"],
              "probe_fit": cfg["probe_windows"], "probe_test": cfg["probe_windows"]}
    panels = {name: document_text_panel(corpus, docs[name], counts[name], seq,
                                        derive_seed("warmup-budget-panel", seed, name), region=name)
              for name in docs}
    declaration = {"basis_split": {name: [d["pg_id"] for d in docs[name]]
                                   for name in ("basis_fit", "basis_validation")},
                   "panels": {}}
    for name, panel in panels.items():
        record = document_panel_record(panel)
        record["documents"] = [d["pg_id"] for d in docs[name]]
        record["region_bounds"] = [docs[name][0]["offset"],
                                   docs[name][-1]["offset"] + docs[name][-1]["bytes"]]
        declaration["panels"][name] = record
    return panels, declaration


def fork(common, spec, bases, cfg, seed, device):
    """Deep-copy the warm checkpoint; apply the branch mode and boundary intervention."""
    branch, mode = spec["branch"], spec["mode"]
    model = deepcopy(common)
    intervention = None
    if mode == "removed_remote":
        model.remote_path.remote = RemovedRemote(model.split.remote_depth)
    apply_mode(model, mode)
    if branch == "d3_q1":
        model.split = replace(model.split, q_bits=1)
        model.remote_path.cfg = model.split
        intervention = {"description": "release quantizer rebuilt at fork", "q_bits": 1}
    basis = bases["warm"] if branch.startswith("d1") else None
    defense = ({"name": "none"} if basis is None else
               {"name": "property_noise", "noise_scale": cfg["noise_scales"][branch]})
    attach_defense(model, defense, basis, derive_seed("warmup-budget-defense", seed))
    model.to(device)
    if trusted_checksum(model) != trusted_checksum(common):
        raise RuntimeError("Branch trusted parameters do not match the common checkpoint")
    return model, intervention


def release_support(model, panel, batch_size, seed, device):
    """Released values must stay on the declared q-bit codebook."""
    path = model.remote_path
    qmax = 2 ** path.quantizer.q_bits - 1
    with measurement(model, seed), torch.no_grad():
        x, _ = next(iter(panel_batches(panel, batch_size)))
        x = x.to(device)
        h = model.tok_emb(x) + model.pos_emb(torch.arange(x.shape[1], device=x.device))
        for block in model.blocks[:model.split.start]:
            h = block(h)
        values, meta = path.quantizer(path.defense(path.encoder(path.encoder_ln(h))))
        codes = (values - meta["minimum"]) / meta["scale"]
        error = float((codes - codes.round()).abs().max())
        if (not torch.isfinite(values).all() or float(codes.min()) < -1e-3
                or float(codes.max()) > qmax + 1e-3 or error > 1e-3):
            raise RuntimeError("Released values violate the declared codebook range")
    return {"q_bits": path.quantizer.q_bits, "support_valid": True,
            "max_normalized_rounding_error": error, "windows": int(x.shape[0])}


def validate(cfg, only_seeds=None):
    require_engineering(cfg, "warmup_budget")
    if cfg.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Protocol ID differs from the warmup-budget engineering protocol")
    validate_seed_namespace(cfg.get("seeds"), "warmup_budget_engineering")
    if cfg.get("split") != ANCHOR:
        raise ValueError("Warmup-budget engineering requires the fixed Paper 4 anchor")
    if cfg.get("branches") != BRANCHES:
        raise ValueError("Branch list differs from the engineering protocol")
    executed = cfg["seeds"] if only_seeds is None else list(only_seeds)
    if not executed or any(s not in cfg["seeds"] for s in executed):
        raise ValueError("Executed seed subset must be nonempty and declared by the protocol")
    expected = len(executed) * (len(BRANCHES) + 1)
    if not cfg.get("full_budget", False):
        return {"expected_scenarios": expected, "full_budget": False, **FLAGS}
    if any(s not in SEEDS for s in cfg["seeds"]) or sorted(cfg["seeds"]) != SEEDS:
        raise ValueError("The full budget declares exactly the two reserved seeds")
    if any(cfg.get(k) != v for k, v in BUDGET.items()) or cfg.get("optimizer") != OPTIMIZER:
        raise ValueError("Changed engineering budget constants require a new protocol")
    data = cfg.get("data", {})
    if data.get("sequence_length") != 128 or data.get("batch_size") != 32:
        raise ValueError("Changed sequence/batch budget requires a new protocol")
    if data.get("sha256") != CORPUS_SHA256 or not data.get("path") or not data.get("documents"):
        raise ValueError("Warmup-budget engineering requires the pinned Gutenberg corpus")
    return {"expected_scenarios": expected, "full_budget": True, **FLAGS}


def run(config_path, label=None, validate_only=False, device=None, only_seeds=None):
    cfg = yaml.safe_load(open(config_path))
    if device is not None:
        cfg["device"] = device
    validation = validate(cfg, only_seeds=only_seeds)
    if validate_only:
        return validation
    device = resolve_device(cfg.get("device", "cpu"))
    corpus = load_pinned_corpus(cfg["data"])
    index = load_document_index(cfg["data"])
    campaign = Campaign(cfg, config_path, label)
    batch, seq = cfg["data"]["batch_size"], cfg["data"]["sequence_length"]
    panel_batch = cfg.get("panel_batch_size", 128)
    train_region = contiguous_region(corpus, index["train"])
    seeds = cfg["seeds"] if only_seeds is None else list(only_seeds)
    for seed in seeds:
        out, record = campaign.start(f"setup-s{seed}",
                                     {"seed": seed, "stage": "shared_setup", "device": str(device),
                                      "protocol_seeds": cfg["seeds"]})
        try:
            panels, declaration = build_panels(cfg, index, corpus, seed)
            torch.save(panels, out / "panels.pt")
            write_json(out / "panels.json", declaration)
            sampler = BatchSampler(train_region, batch, seq, seed, "warmup-training")
            common = build_initialized_model(BackboneConfig(**cfg["backbone"]), "frozen_random",
                                             SplitConfig(**cfg["split"]), seed,
                                             f"{cfg['protocol_id']}-{seed}")
            attach_defense(common, {"name": "none"})
            common.to(device)
            basis_args = (panels["basis_fit"], panels["basis_validation"], panel_batch, seed, device)
            bases, basis_stats = {}, {}
            bases["initial"], basis_stats["initial"] = basis_snapshot(common, *basis_args)
            remote_initial = parameters_checksum(remote_parameters(common))
            curve = train_steps(common, sampler, cfg["optimizer"], cfg["warmup_steps"], device)
            if remote_initial != parameters_checksum(remote_parameters(common)):
                raise RuntimeError("Common warmup changed frozen remote parameters")
            bases["warm"], basis_stats["warm"] = basis_snapshot(common, *basis_args)
            torch.save({"state_dict": {k: v.detach().cpu() for k, v in common.state_dict().items()},
                        "bases": bases}, out / "common_checkpoint.pt")
            write_json(out / "metrics.json", {**FLAGS, "seed": seed, "stage": "shared_setup",
                       "device": str(device),
                       "remote_initial_sha256": remote_initial,
                       "remote_final_sha256": parameters_checksum(remote_parameters(common)),
                       "trusted_warm_sha256": trusted_checksum(common), "bases": basis_stats,
                       "initial_warm_absolute_cosine": basis_cosine(bases["initial"], bases["warm"]),
                       "curve": curve})
            setup_path = str((out / "common_checkpoint.pt").relative_to(ROOT))
            setup_sha = digest(ROOT / setup_path)
            setup_run_id = record["run_id"]
            panels_sha = digest(out / "panels.pt")
        except Exception as error:
            campaign.finish(out, record, error)
            # Preserve a failed setup and do not create unbound dependent branches.
            continue
        campaign.finish(out, record)
        eval_batches = list(panel_batches(panels["probe_test"], batch))
        for spec in cfg["branches"]:
            branch, mode = spec["branch"], spec["mode"]
            inputs = {"seed": seed, "stage": "branch", "branch": branch, "mode": mode,
                      "device": str(device), "setup_run_id": setup_run_id,
                      "protocol_seeds": cfg["seeds"],
                      "checkpoint_path": setup_path, "checkpoint_sha256": setup_sha,
                      "panels_sha256": panels_sha}
            out, record = campaign.start(f"{branch}-{mode}-s{seed}", inputs)
            try:
                start = time.perf_counter()
                if mode == "all_local":
                    model = build_initialized_model(BackboneConfig(**cfg["backbone"]), mode,
                                                    None, seed, None)
                    model.to(device)
                    steps, offset, intervention = cfg["local_steps"], 0, None
                else:
                    model, intervention = fork(common, spec, bases, cfg, seed, device)
                    steps, offset = cfg["branch_steps"], cfg["warmup_steps"]
                remote_before = parameters_checksum(remote_parameters(model))
                trusted_before = trusted_checksum(model)
                used_basis = (tensor_hash(model.remote_path.defense.directions)
                              if branch.startswith("d1") else None)
                curve = train_steps(model, sampler, cfg["optimizer"], steps, device, offset)
                remote_after = parameters_checksum(remote_parameters(model))
                if mode == "frozen_random" and remote_before != remote_after:
                    raise RuntimeError("Frozen branch remote parameters changed")
                if (used_basis is not None
                        and used_basis != tensor_hash(model.remote_path.defense.directions)):
                    raise RuntimeError("D1 basis changed during training")
                metrics = {**FLAGS, **inputs, "trusted_initial_sha256": trusted_before,
                           "trusted_final_sha256": trusted_checksum(model),
                           "remote_initial_sha256": remote_before,
                           "remote_final_sha256": remote_after,
                           "basis_used_sha256": used_basis, "basis_fixed": True,
                           "final_eval_loss": evaluate(model, eval_batches,
                                                       derive_seed("warmup-budget-eval", seed), device),
                           "curve": curve, "unevaluated": OMITTED}
                if intervention is not None:
                    metrics["intervention"] = intervention
                    metrics["release_support"] = release_support(
                        model, panels["probe_fit"], panel_batch,
                        derive_seed("warmup-budget-support", seed), device)
                if mode != "all_local":
                    if mode == "trained":
                        diagnostic_basis, diagnostic_stats = basis_snapshot(model, *basis_args)
                        metrics["post_training_basis"] = {
                            **diagnostic_stats,
                            "absolute_cosine_to_initial": basis_cosine(diagnostic_basis, bases["initial"]),
                            "absolute_cosine_to_warm": basis_cosine(diagnostic_basis, bases["warm"])}
                    fit = boundary_means(model, panels["probe_fit"], panel_batch,
                                         derive_seed("warmup-budget-fit", seed), device)
                    test = boundary_means(model, panels["probe_test"], panel_batch,
                                          derive_seed("warmup-budget-test", seed), device)
                    torch.save({"fit": fit, "test": test}, out / "means.pt")
                    controls = stability_curve(fit, test, panels["probe_fit"]["tokens"],
                                               panels["probe_test"]["tokens"],
                                               cfg["panel_sizes"], cfg["null_repeats"], seed)
                    write_json(out / "stability.json", controls)
                metrics["wall_seconds"] = time.perf_counter() - start
                write_json(out / "metrics.json", metrics)
            except Exception as error:
                campaign.finish(out, record, error)
            else:
                campaign.finish(out, record)
    return campaign.summarize({"expected_scenarios": validation["expected_scenarios"],
                               "unevaluated": OMITTED, "qualified": False})
