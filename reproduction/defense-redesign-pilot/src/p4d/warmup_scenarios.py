"""Shared-warmup and property-null diagnostics; scientific execution is disabled."""
from copy import deepcopy
import hashlib
import time

import numpy as np
import torch
import yaml
from pcr.data import BatchSampler, load_corpus
from pcr.model import BackboneConfig
from pcr.modes import apply_mode, remote_parameters, trainable_parameters
from pcr.privacy_attacks_p4b import pearson_r, ridge_fit, ridge_predict, sequence_targets
from pcr.split import RemovedRemote, SplitConfig
from pcr.train import OptimConfig, build_initialized_model, build_optimizer
from pcr.utils import derive_seed, parameters_checksum

from .calibration import NULL_BAND, estimate_property_directions
from .experiment import ROOT, CORPUS_SHA256, digest, evaluate, measurement, write_json
from .model import attach_defense
from .scenario_io import Campaign, require_engineering, validate_seed_namespace

TARGET = "punctuation_fraction"
ADAPTER = "p4-primary-property-null-diagnostic-0.1"
FLAGS = {"scientific_evidence": False, "qualified": False, "privacy_outcome": "undetermined"}
OMITTED = ["next-byte classes", "returned values", "quantizer metadata",
           "full-precision inbound/outbound gradients", "evolving remote state", "optimizer state",
           "timing", "active/adaptive attackers", "within-run joint tensors", "cross-run attacks",
           "history accumulation", "canary membership", "natural-sequence membership",
           "unrestricted semantics"]


def tensor_hash(tensor):
    t = tensor.detach().cpu().contiguous()
    return hashlib.sha256(str((str(t.dtype), list(t.shape))).encode() + t.numpy().tobytes()).hexdigest()


def text_panel(corpus, start, stop, count, sequence_length, seed):
    """Nonoverlapping windows include the final next-byte target in their support."""
    stride = sequence_length + 1
    if not 0 <= start < stop <= len(corpus) or count < 4 or sequence_length < 1:
        raise ValueError("Invalid panel region, size, or sequence length")
    grid = np.arange(start, stop - stride + 1, stride, dtype=np.int64)
    if count > len(grid):
        raise ValueError("Insufficient disjoint windows in panel region")
    offsets = torch.tensor(np.random.default_rng(seed).permutation(grid)[:count])
    windows = torch.stack([corpus[int(i):int(i) + stride].long() for i in offsets])
    return {"offsets": offsets, "tokens": windows[:, :-1].contiguous(),
            "next_tokens": windows[:, 1:].contiguous(), "region": [start, stop]}


def panel_batches(panel, batch_size):
    for i in range(0, len(panel["tokens"]), batch_size):
        yield panel["tokens"][i:i + batch_size], panel["next_tokens"][i:i + batch_size]


def panel_record(panel):
    return {"region": panel["region"], "sequences": len(panel["tokens"]),
            "sequence_length": panel["tokens"].shape[1],
            "absolute_offsets": panel["offsets"].tolist(),
            "tensor_sha256": {k: tensor_hash(panel[k]) for k in ("offsets", "tokens", "next_tokens")}}


def boundary_means(model, panel, batch_size, seed, released=True):
    """Exact selected forward view; do not execute the remote module or suffix."""
    means = []
    with measurement(model, seed), torch.no_grad():
        for x, _ in panel_batches(panel, batch_size):
            h = model.tok_emb(x) + model.pos_emb(torch.arange(x.shape[1], device=x.device))
            for block in model.blocks[:model.split.start]:
                h = block(h)
            path = model.remote_path
            z = path.encoder(path.encoder_ln(h))
            if released:
                z, _ = path.quantizer(path.defense(z))
            means.append(z.mean(1).cpu())
    return torch.cat(means)


def basis_snapshot(model, fit, validation, batch_size, seed):
    x = boundary_means(model, fit, batch_size, seed, released=False).numpy()
    xt = boundary_means(model, validation, batch_size, seed, released=False).numpy()
    y = sequence_targets(fit["tokens"])[TARGET]
    yt = sequence_targets(validation["tokens"])[TARGET]
    basis = estimate_property_directions(x, {TARGET: y})
    probe = ridge_fit(x, y)
    return basis, {"basis_sha256": tensor_hash(basis), "rank": len(basis),
                   "validation_r": pearson_r(ridge_predict(probe, xt), yt),
                   "fit_sequences": len(y), "validation_sequences": len(yt)}


def basis_cosine(a, b):
    if a.shape != b.shape or a.ndim != 2 or len(a) != 1:
        raise ValueError("This diagnostic compares rank-one bases only")
    return float(torch.abs(torch.nn.functional.cosine_similarity(a.double(), b.double())).clamp(max=1))


def trusted_checksum(model):
    remote = {id(p) for p in remote_parameters(model)}
    return parameters_checksum(p for p in model.parameters() if id(p) not in remote)


def fork_branch(common, variant, mode, bases, noise_scale, seed):
    if variant not in ("d0", "d1_initial_basis", "d1_warm_basis"):
        raise ValueError("Unknown basis schedule")
    if mode not in ("trained", "frozen_random", "removed_remote"):
        raise ValueError("Unknown branch mode")
    model = deepcopy(common)
    if mode == "removed_remote":
        model.remote_path.remote = RemovedRemote(model.split.remote_depth)
    apply_mode(model, mode)
    basis = None if variant == "d0" else bases["initial" if variant == "d1_initial_basis" else "warm"]
    attach_defense(model, {"name": "none" if basis is None else "property_noise",
                           "noise_scale": noise_scale}, basis, seed)
    if trusted_checksum(model) != trusted_checksum(common):
        raise RuntimeError("Branch trusted parameters do not match the common checkpoint")
    return model


def train_steps(model, sampler, optimizer_cfg, steps, offset=0):
    """A new optimizer is an explicit intervention at every warmup/branch boundary."""
    config = OptimConfig(**optimizer_cfg, steps=steps)
    optimizer = build_optimizer(model, config)
    curve = []
    model.train()
    for step in range(steps):
        for group in optimizer.param_groups:
            group["lr"] = config.lr_at(step)
        x, y = sampler.batch(offset + step)
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


def property_score(x, y, xt, yt, seed):
    """Existing single-target r-r_null statistic, with an explicit independent RNG."""
    x, xt, y, yt = [np.asarray(a) for a in (x, xt, y, yt)]
    if (x.ndim != 2 or xt.ndim != 2 or x.shape[1] != xt.shape[1]
            or y.shape != (len(x),) or yt.shape != (len(xt),)
            or min(len(x), len(xt)) < 4 or np.std(y) == 0 or np.std(yt) == 0
            or not all(np.isfinite(a).all() for a in (x, xt, y, yt))):
        raise ValueError("Need finite, aligned, varying property panels")
    rng = np.random.default_rng(seed)
    fitted = ridge_fit(x, y)
    null = ridge_fit(x, rng.permutation(y))
    r = pearson_r(ridge_predict(fitted, xt), yt)
    rn = pearson_r(ridge_predict(null, xt[rng.permutation(len(xt))]), yt)
    result = {"r": r, "r_null": rn, "advantage": r - rn}
    if not all(np.isfinite(v) for v in result.values()):
        raise ValueError("Nonfinite property score")
    return result


def derangement(n, seed):
    if n < 2:
        raise ValueError("Need at least two donors")
    order = np.random.default_rng(seed).permutation(n)
    donors = np.empty(n, dtype=np.int64)
    donors[order] = np.roll(order, 1)
    return donors


def stability_curve(fit_means, test_means, fit_tokens, test_tokens, sizes, repeats, seed):
    y, yt = sequence_targets(fit_tokens)[TARGET], sequence_targets(test_tokens)[TARGET]
    results = []
    for n in sizes:
        if n > min(len(y), len(yt)) or n < 4 or repeats < 1:
            raise ValueError("Invalid stability panel size/repetition count")
        x, xt = fit_means[:n].numpy(), test_means[:n].numpy()
        actual = property_score(x, y[:n], xt, yt[:n], derive_seed(ADAPTER, seed, n, "actual"))
        rows = []
        for repetition in range(repeats):
            seeds = {key: derive_seed(ADAPTER, seed, n, repetition, key)
                     for key in ("fit-donors", "test-donors", "scorer")}
            df, dt = derangement(n, seeds["fit-donors"]), derangement(n, seeds["test-donors"])
            rows.append({"repetition": repetition, "seeds": seeds,
                         **property_score(x[df], y[:n], xt[dt], yt[:n], seeds["scorer"])})
        advantages = np.array([r["advantage"] for r in rows])
        results.append({"sequences_per_panel": n, "observed": actual, "nulls": rows,
                        "mean_null_advantage": float(advantages.mean()),
                        "p95_absolute_null_advantage": float(np.quantile(abs(advantages), .95)),
                        "max_absolute_null_advantage": float(abs(advantages).max()),
                        "within_null_band": int((abs(advantages) <= NULL_BAND).sum()),
                        "null_band": NULL_BAND, "repeats": repeats})
    return {"adapter": ADAPTER, "target": TARGET, "view": "mean_forward_values",
            **FLAGS, "unevaluated": OMITTED, "curve": results}


def validate(cfg):
    require_engineering(cfg, "warmup")
    validate_seed_namespace(cfg["seeds"], "warmup_engineering")
    if cfg["split"] != dict(start=2, span=4, width=256, remote_depth=1, q_bits=8):
        raise ValueError("Warmup diagnostic requires the fixed Paper 4 anchor")
    if cfg["data"]["fractions"] != [0, .80, .85, .90, .95, 1.0]:
        raise ValueError("Warmup diagnostic requires five disjoint text regions")
    if cfg["variants"] != ["d0", "d1_initial_basis", "d1_warm_basis"] or cfg["modes"] != ["trained", "frozen_random", "removed_remote"]:
        raise ValueError("All paired basis variants and controls are required")
    if (cfg["panel_sizes"] != [16, 64, 256, 1024] or cfg["null_repeats"] != 16
            or cfg["noise_scale"] != .1 or cfg["basis_sequences"] != 512
            or cfg["warmup_steps"] != 16 or cfg["branch_steps"] != 16):
        raise ValueError("Changing the declared engineering budget requires a new protocol")
    return {"expected_scenarios": len(cfg["seeds"]) * 11, **FLAGS}


def run(config_path, label=None, validate_only=False):
    cfg = yaml.safe_load(open(config_path))
    validation = validate(cfg)
    if validate_only:
        return validation
    data_path = ROOT / cfg["data"]["path"]
    if digest(data_path) != cfg["data"]["sha256"] or cfg["data"]["sha256"] != CORPUS_SHA256:
        raise ValueError("Corpus differs from pinned public data")
    corpus = load_corpus(data_path)
    campaign = Campaign(cfg, config_path, label)
    batch_size, seq = cfg["data"]["batch_size"], cfg["data"]["sequence_length"]
    cuts = [int(len(corpus) * f) for f in cfg["data"]["fractions"]]
    for seed in cfg["seeds"]:
        out, record = campaign.start(f"setup-s{seed}", {"seed": seed, "stage": "shared_setup"})
        try:
            panels = {name: text_panel(corpus, cuts[i], cuts[i + 1], count, seq,
                                      derive_seed("warmup-panel", seed, name))
                      for i, name, count in [(1, "basis_fit", cfg["basis_sequences"]),
                                             (2, "basis_validation", cfg["basis_sequences"]),
                                             (3, "probe_fit", max(cfg["panel_sizes"])),
                                             (4, "probe_test", max(cfg["panel_sizes"]))]}
            torch.save(panels, out / "panels.pt")
            write_json(out / "panels.json", {k: panel_record(v) for k, v in panels.items()})
            sampler = BatchSampler(corpus[:cuts[1]], batch_size, seq, seed, "warmup-training")
            common = build_initialized_model(BackboneConfig(**cfg["backbone"]), "frozen_random",
                                             SplitConfig(**cfg["split"]), seed, f"{cfg['protocol_id']}-{seed}")
            attach_defense(common, {"name": "none"})
            basis_args = (panels["basis_fit"], panels["basis_validation"], batch_size, seed)
            bases, basis_stats = {}, {}
            bases["initial"], basis_stats["initial"] = basis_snapshot(common, *basis_args)
            remote_initial = parameters_checksum(remote_parameters(common))
            curve = train_steps(common, sampler, cfg["optimizer"], cfg["warmup_steps"])
            if remote_initial != parameters_checksum(remote_parameters(common)):
                raise RuntimeError("Common warmup changed frozen remote parameters")
            bases["warm"], basis_stats["warm"] = basis_snapshot(common, *basis_args)
            torch.save({"state_dict": common.state_dict(), "bases": bases}, out / "common_checkpoint.pt")
            write_json(out / "metrics.json", {**FLAGS, "seed": seed, "stage": "shared_setup",
                       "remote_initial_sha256": remote_initial,
                       "remote_final_sha256": parameters_checksum(remote_parameters(common)),
                       "trusted_warm_sha256": trusted_checksum(common), "bases": basis_stats,
                       "initial_warm_absolute_cosine": basis_cosine(bases["initial"], bases["warm"]),
                       "curve": curve})
            setup_path = str((out / "common_checkpoint.pt").relative_to(ROOT))
            setup_sha = digest(ROOT / setup_path)
            setup_run_id = record["run_id"]
        except Exception as error:
            campaign.finish(out, record, error)
            # Preserve a failed setup and do not create unbound dependent branches.
            continue
        campaign.finish(out, record)
        jobs = [(v, m) for v in cfg["variants"] for m in cfg["modes"]] + [("all_local", "all_local")]
        eval_batches = list(panel_batches(panels["probe_test"], batch_size))[:cfg["eval_batches"]]
        for variant, mode in jobs:
            inputs = {"seed": seed, "stage": "branch", "variant": variant, "mode": mode,
                      "setup_run_id": setup_run_id, "checkpoint_path": setup_path,
                      "checkpoint_sha256": setup_sha, "panel_sha256": digest(campaign.path / setup_run_id / "panels.pt")}
            out, record = campaign.start(f"{variant}-{mode}-s{seed}", inputs)
            try:
                start = time.perf_counter()
                if mode == "all_local":
                    model = build_initialized_model(BackboneConfig(**cfg["backbone"]), mode, None, seed, None)
                    train_steps(model, sampler, cfg["optimizer"], cfg["warmup_steps"])
                else:
                    model = fork_branch(common, variant, mode, bases, cfg["noise_scale"], derive_seed("warmup-defense", seed))
                remote_before = parameters_checksum(remote_parameters(model))
                trusted_before = trusted_checksum(model)
                used_basis = None if variant in ("d0", "all_local") else tensor_hash(model.remote_path.defense.directions)
                curve = train_steps(model, sampler, cfg["optimizer"], cfg["branch_steps"], cfg["warmup_steps"])
                remote_after = parameters_checksum(remote_parameters(model))
                if mode == "frozen_random" and remote_before != remote_after:
                    raise RuntimeError("Frozen branch remote parameters changed")
                if used_basis and used_basis != tensor_hash(model.remote_path.defense.directions):
                    raise RuntimeError("D1 basis changed during training")
                metrics = {**FLAGS, **inputs, "trusted_initial_sha256": trusted_before,
                           "remote_initial_sha256": remote_before, "remote_final_sha256": remote_after,
                           "basis_used_sha256": used_basis, "basis_fixed": True,
                           "final_eval_loss": evaluate(model, eval_batches, derive_seed("warmup-eval", seed)),
                           "curve": curve, "unevaluated": OMITTED}
                if mode != "all_local":
                    diagnostic_basis, diagnostic_stats = basis_snapshot(model, *basis_args)
                    metrics["post_training_basis"] = {**diagnostic_stats,
                        "absolute_cosine_to_initial": basis_cosine(diagnostic_basis, bases["initial"]),
                        "absolute_cosine_to_warm": basis_cosine(diagnostic_basis, bases["warm"])}
                    fit = boundary_means(model, panels["probe_fit"], batch_size, derive_seed("warmup-fit", seed))
                    test = boundary_means(model, panels["probe_test"], batch_size, derive_seed("warmup-test", seed))
                    torch.save({"fit": fit, "test": test, "diagnostic_basis": diagnostic_basis}, out / "means.pt")
                    controls = stability_curve(fit, test, panels["probe_fit"]["tokens"], panels["probe_test"]["tokens"],
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
