"""Calibration-cohort instrument qualification dress rehearsal (p4-qualification-0.1).

Protocol `p4-qualification-0.1` (docs/QUALIFICATION_PROTOCOL.md): train models under
the exact candidate pilot regime on the reserved calibration seeds and run the exact
frozen control/scoring procedure per binding, producing instrument-qualification
records. This is the first scientific-cohort runner: records carry cohort
"calibration" and scientific_evidence true, scoped to instrument qualification only;
no defense-success, utility, or causal-gain claim is made anywhere, and
privacy_outcome stays undetermined. `full_budget: false` exists only so tests can run
the pipeline on tiny synthetic corpora; the protocol config sets it true and
validation then pins every constant. Protocol ID, cohort/status, cell set, anchor
split, and the reserved calibration seed namespace are always enforced.

Training regime: 2,000 steps from initialization, no shared warmup; per-seed paired
trusted/remote initialization across all cells (pairing block f"{protocol_id}-{seed}",
verified per cell). D1 cells refit the rank-1 punctuation basis on the frozen
schedule (step 0 and every 250 steps through 2,000) using the shared
document-disjoint basis panels; refits run inside measurement contexts, replace only
the defense directions buffer, and never perturb the defense RNG or training
batches. One shared panel set (protocol-ID-derived seeds) serves the whole campaign.
"""
from datetime import datetime, timezone
from pathlib import Path
import json
import math
import re
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
import jsonschema
from pcr.data import BatchSampler
from pcr.model import BackboneConfig
from pcr.modes import remote_parameters, trainable_parameters
from pcr.privacy_attacks_p4b import next_byte_class, sequence_targets
from pcr.split import SplitConfig
from pcr.train import OptimConfig, build_initialized_model, build_optimizer
from pcr.utils import derive_seed, parameters_checksum

from .corpus import (contiguous_region, document_panel_record, document_text_panel,
                     load_document_index, load_pinned_corpus)
from .devices import resolve_device
from .experiment import (ROOT, digest, evaluate, source_provenance,
                         verify_inherited_source, write_json)
from .model import attach_defense
from .null_precision import (RidgeDesign, no_signal_trials, prequant_capture,
                             quantized_control, signal_spec)
from .scenario_io import validate_seed_namespace
from .structure_precision import (class_directions, inject_and_release,
                                  quantized_release, structure_score)
from .warmup_budget_scenarios import (OPTIMIZER, basis_snapshot, basis_split,
                                      boundary_means, train_steps)
from .warmup_scenarios import TARGET, derangement, panel_batches, tensor_hash, trusted_checksum

PROTOCOL_ID = "p4-qualification-0.1"
ADAPTER = "p4-qualification-0.1"
CORPUS_SHA256 = "af3d0245c0ec27d7021f5fd29256bc845bb20cf9c562b37bbe23663b6ebe7436"
SEEDS = [419101, 419102, 419103, 419104]
SPLIT = {"start": 2, "span": 4, "width": 256, "remote_depth": 1}
CELLS = [{"id": "d0-q8", "q_bits": 8, "defense": {"name": "none"}},
         {"id": "d1-s0.1-q8", "q_bits": 8, "defense": {"name": "property_noise", "noise_scale": 0.1}},
         {"id": "d1-s0.5-q8", "q_bits": 8, "defense": {"name": "property_noise", "noise_scale": 0.5}},
         {"id": "d1-s1.0-q8", "q_bits": 8, "defense": {"name": "property_noise", "noise_scale": 1.0}},
         {"id": "d3-q1", "q_bits": 1, "defense": {"name": "aggressive_rate"}},
         {"id": "d3-q2", "q_bits": 2, "defense": {"name": "aggressive_rate"}},
         {"id": "d3-q3", "q_bits": 3, "defense": {"name": "aggressive_rate"}}]
BUDGET = {"train_steps": 2000, "basis_windows": 4096, "refit_interval": 250,
          "probe_windows": 16384, "eval_batch_size": 32, "panel_batch_size": 128,
          "capture_batch_size": 64, "property_matched_nulls": 16, "structure_matched_nulls": 4,
          "outer_null_repeats": 16, "positive_doses": [.5, 2.], "ridge_penalty": 1.,
          "null_band": .02, "property_positive_floor": .10, "structure_positive_floor": .02,
          "mlp_hidden": 64, "mlp_lr": 3e-3, "mlp_row_batch": 8192, "cpu_threads": 2}
SCIENTIFIC_FLAGS = {"cohort": "calibration", "scientific_evidence": True,
                    "privacy_outcome": "undetermined"}
UNEVALUATED = ["returned values", "quantizer metadata", "full-precision inbound/outbound gradients",
               "evolving remote state", "optimizer state", "timing", "active/adaptive attackers",
               "within-run joint tensors", "cross-run attacks", "history accumulation",
               "canary membership", "natural-sequence membership",
               "semantics beyond five-class byte structure", "token inversion"]
CLAIM_SCOPE = ("Instrument qualification records under the exact candidate pilot regime, "
               "calibration cohort only; no defense-success, utility, or causal-gain claim; "
               "privacy undetermined. Candidate scores on calibration models are "
               "calibration-cohort exploratory observations, not pilot gate outcomes.")


def validate(cfg, only_seeds=None):
    if cfg.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Protocol ID differs from the qualification protocol")
    if cfg.get("cohort") != "calibration" or cfg.get("status") != "frozen":
        raise ValueError("Qualification requires the frozen calibration cohort configuration")
    seeds = cfg.get("seeds")
    validate_seed_namespace(seeds, "calibration")
    if cfg.get("cells") != CELLS:
        raise ValueError("Cell set differs from the qualification protocol")
    if cfg.get("split") != SPLIT:
        raise ValueError("Qualification requires the fixed Paper 4 anchor")
    executed = list(seeds) if only_seeds is None else list(only_seeds)
    if not executed or any(s not in seeds for s in executed):
        raise ValueError("Executed seed subset must be nonempty and declared by the protocol")
    full = bool(cfg.get("full_budget", False))
    if full:
        if sorted(seeds or []) != SEEDS:
            raise ValueError("The frozen campaign declares exactly the four calibration seeds")
        if cfg.get("optimizer") != OPTIMIZER:
            raise ValueError("Changed optimizer constants require a new protocol")
        if any(cfg.get(k) != v for k, v in BUDGET.items()):
            raise ValueError("Changed training/scoring budget requires a new protocol")
        data = cfg.get("data", {})
        if data.get("sequence_length") != 128 or data.get("batch_size") != 32:
            raise ValueError("Changed sequence/batch budget requires a new protocol")
        if data.get("sha256") != CORPUS_SHA256 or not data.get("path") or not data.get("documents"):
            raise ValueError("Qualification requires the pinned Gutenberg corpus")
    expected = len(executed) * (len(cfg.get("cells") or []) + 1)
    return {"expected_runs": expected, "full_budget": full, **SCIENTIFIC_FLAGS}


def build_panels(cfg, index, corpus, protocol_id=None):
    """One shared panel set for the whole campaign; panel seeds derive from the protocol ID."""
    protocol_id = protocol_id or PROTOCOL_ID
    seq = cfg["data"]["sequence_length"]
    fit_docs, validation_docs = basis_split(index["calibration"], derive_seed(protocol_id, "basis-split"))
    docs = {"basis_fit": fit_docs, "basis_validation": validation_docs,
            "probe_fit": index["probe_fit"], "evaluation": index["evaluation"]}
    counts = {"basis_fit": cfg["basis_windows"], "basis_validation": cfg["basis_windows"],
              "probe_fit": cfg["probe_windows"], "evaluation": cfg["probe_windows"]}
    panels = {name: document_text_panel(corpus, docs[name], counts[name], seq,
                                        derive_seed(protocol_id, "panel", name), region=name)
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


def replace_defense_basis(defense, basis, expected_sha256):
    if (basis.shape != defense.directions.shape or len(basis) != 1
            or not torch.isfinite(basis).all()
            or not torch.allclose(basis @ basis.T, torch.eye(1), atol=1e-5)):
        raise ValueError("Invalid refit basis")
    with torch.no_grad():
        defense.directions.copy_(basis.to(defense.directions.device))
    if tensor_hash(defense.directions) != expected_sha256:
        raise RuntimeError("Basis replacement mismatch")


def train_with_basis_refits(model, sampler, optimizer_cfg, steps, refit_interval,
                            fit_panel, validation_panel, panel_batch, refit_seed, device):
    """train_steps plus the frozen D1 basis-refit schedule; identical optimizer semantics."""
    config = OptimConfig(**optimizer_cfg, steps=steps)
    optimizer = build_optimizer(model, config)
    curve, basis_log = [], []

    def refit(step):
        basis, stats = basis_snapshot(model, fit_panel, validation_panel, panel_batch, refit_seed, device)
        if stats["rank"] != 1:
            raise RuntimeError("Basis refit did not produce the rank-one direction")
        replace_defense_basis(model.remote_path.defense, basis, stats["basis_sha256"])
        basis_log.append({"step": step, "basis_sha256": stats["basis_sha256"],
                          "validation_r": stats["validation_r"]})

    model.train()
    refit(0)
    for step in range(steps):
        for group in optimizer.param_groups:
            group["lr"] = config.lr_at(step)
        x, y = sampler.batch(step)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x).reshape(-1, 256), y.reshape(-1))
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_parameters(model), config.grad_clip)
        optimizer.step()
        curve.append({"step": step + 1, "train_loss": float(loss.detach())})
        if (step + 1) % refit_interval == 0:
            refit(step + 1)
    model.zero_grad(set_to_none=True)
    return curve, basis_log


def binding_verdict(null_trials, positive_rows, band, floor):
    """Qualified iff every no-signal trial stays in the band and nonzero doses clear the floor."""
    advantages = [t["estimators"][0]["advantage"] for t in null_trials]
    over_band = [i for i, a in enumerate(advantages) if abs(a) > band]
    positive = {row["dose"]: row["score"]["estimators"][0]["advantage"]
                for row in positive_rows if row["dose"] > 0}
    below_floor = {dose: advantage for dose, advantage in positive.items() if advantage < floor}
    qualified = not over_band and not below_floor
    reasons = []
    if over_band:
        reasons.append(f"{len(over_band)} no-signal trials outside the ±{band} band")
    if below_floor:
        reasons.append(f"positive doses below the {floor} floor: {sorted(below_floor)}")
    return {"qualified": qualified, "null_band": band, "positive_floor": floor,
            "reason": "; ".join(reasons) if reasons else "within the null band and above the positive floor",
            "max_absolute_null_advantage": max(map(abs, advantages)),
            "positive_advantages": positive}


def score_property(model, panels, pre, released, cfg, cell, seed, device, adapter=None):
    """Ridge sequence-mean matched-null scoring, mirroring the engineering control structure."""
    adapter = adapter or ADAPTER
    batch = cfg["capture_batch_size"]
    quantizer = model.remote_path.quantizer
    k = cfg["property_matched_nulls"]
    calibration = boundary_means(model, panels["basis_fit"], batch,
                                 derive_seed(adapter, seed, cell["id"], "calibration"),
                                 device, released=False)
    spec = signal_spec(calibration, panels["basis_fit"]["tokens"],
                       derive_seed(adapter, seed, cell["id"], "signal"))
    spec_device = {**spec, "direction": spec["direction"].to(device)}
    targets = {key: sequence_targets(panels[name]["tokens"])[TARGET]
               for key, name in (("fit", "probe_fit"), ("test", "evaluation"))}
    targets_device = {key: torch.as_tensor(targets[key], dtype=torch.float32, device=device)
                      for key in targets}
    normal = {key: released[key].mean(1).cpu() for key in released}
    y, yt = targets["fit"], targets["test"]
    design = RidgeDesign(normal["fit"], normal["test"], cfg["ridge_penalty"])
    candidate = design.score(y, yt, derive_seed(adapter, seed, cell["id"], "property-candidate"), [k])
    null_trials = no_signal_trials(design, y, yt, cfg["outer_null_repeats"], [k],
                                   derive_seed(adapter, seed, cell["id"], "property-nulls"))
    donor_seeds = {key: derive_seed(adapter, seed, cell["id"], "property-positive-donors", key)
                   for key in pre}
    donors = {key: derangement(len(y), donor_seeds[key]) for key in pre}
    positive_rows = []
    for dose in [0., *cfg["positive_doses"]]:
        features, checks = {}, {}
        for key in pre:
            means, checks[key] = quantized_control(pre[key], targets_device[key], donors[key],
                                                   spec_device, dose, quantizer, batch)
            features[key] = means.cpu()
            if dose == 0 and not torch.equal(features[key], normal[key][donors[key]]):
                raise RuntimeError("Zero-dose means do not equal the released donor means")
        control_design = RidgeDesign(features["fit"], features["test"], cfg["ridge_penalty"])
        score = control_design.score(y, yt, derive_seed(adapter, seed, cell["id"], "property-positive-scorer"), [k])
        positive_rows.append({"dose": dose, "support": checks, "score": score,
                              "features_sha256": {kk: tensor_hash(v) for kk, v in features.items()}})
        del features
    return {"target": TARGET, "view": "mean_forward_values", "matched_null_count": k,
            "candidate": candidate, "no_signal_trials": null_trials,
            "positive_donor_seeds": donor_seeds, "positive_controls": positive_rows,
            "signal_spec": {kk: v.tolist() if isinstance(v, torch.Tensor) else v for kk, v in spec.items()},
            "binding_qualification": binding_verdict(null_trials, positive_rows,
                                                     cfg["null_band"], cfg["property_positive_floor"])}


def score_structure(model, panels, pre, released, cfg, cell, seed, device, adapter=None):
    """Five-class MLP per-token matched-null scoring, mirroring the structure engineering controls."""
    adapter = adapter or ADAPTER
    batch = cfg["capture_batch_size"]
    quantizer = model.remote_path.quantizer
    k = cfg["structure_matched_nulls"]
    mlp_cfg = {"mlp_hidden": cfg["mlp_hidden"], "mlp_lr": cfg["mlp_lr"],
               "mlp_row_batch": cfg["mlp_row_batch"]}
    classes = {key: torch.as_tensor(next_byte_class(panels[name]["next_tokens"].numpy()), device=device)
               for key, name in (("fit", "probe_fit"), ("test", "evaluation"))}
    width = pre["fit"].shape[-1]
    directions = class_directions(width, derive_seed(adapter, seed, cell["id"], "class-directions"))
    basis = model.remote_path.defense.directions if cell["defense"]["name"] == "property_noise" else None
    cosines = (directions @ basis.cpu().T).abs().clamp(max=1).tolist() if basis is not None else None
    feature_scale = float(pre["fit"].std(unbiased=False)) * math.sqrt(width)
    if not math.isfinite(feature_scale) or feature_scale <= 0:
        raise ValueError("Degenerate fit-panel feature scale")
    directions_device = directions.to(device)
    rows = {key: released[key].reshape(-1, width) for key in released}
    labels = {key: classes[key].reshape(-1) for key in classes}
    candidate = structure_score(rows["fit"], labels["fit"], rows["test"], labels["test"], [k], mlp_cfg,
                                derive_seed(adapter, seed, cell["id"], "structure-candidate"), device)
    null_trials = []
    for repetition in range(cfg["outer_null_repeats"]):
        donor_seeds = {key: derive_seed(adapter, seed, cell["id"], repetition, "structure-null-donors", key)
                       for key in pre}
        null_labels = {key: classes[key][derangement(len(classes[key]), donor_seeds[key])].reshape(-1)
                       for key in pre}
        score = structure_score(rows["fit"], null_labels["fit"], rows["test"], null_labels["test"],
                                [k], mlp_cfg,
                                derive_seed(adapter, seed, cell["id"], repetition, "structure-null-scorer"),
                                device)
        null_trials.append({"repetition": repetition, "seeds": donor_seeds, **score})
    donor_seeds = {key: derive_seed(adapter, seed, cell["id"], "structure-positive-donors", key)
                   for key in pre}
    donors = {key: derangement(len(classes[key]), donor_seeds[key]) for key in pre}
    positive_rows = []
    for dose in [0., *cfg["positive_doses"]]:
        features, checks = {}, {}
        for key in pre:
            features[key], checks[key] = inject_and_release(pre[key], classes[key], donors[key],
                                                            directions_device, feature_scale, dose,
                                                            quantizer, batch)
            if dose == 0 and not torch.equal(features[key], released[key][donors[key]]):
                raise RuntimeError("Zero-dose release does not equal the plain donor release")
        score = structure_score(features["fit"].reshape(-1, width), labels["fit"],
                                features["test"].reshape(-1, width), labels["test"], [k], mlp_cfg,
                                derive_seed(adapter, seed, cell["id"], "structure-positive-scorer"), device)
        positive_rows.append({"dose": dose, "support": checks, "score": score,
                              "features_sha256": {kk: tensor_hash(v) for kk, v in features.items()}})
        del features
    return {"target": "next_byte_class", "view": "token_forward_values", "matched_null_count": k,
            "candidate": candidate, "no_signal_trials": null_trials,
            "positive_donor_seeds": donor_seeds, "positive_controls": positive_rows,
            "feature_scale": feature_scale,
            "feature_scale_definition": ("population standard deviation over every token feature "
                                         "value of the full fit prequant tensor, times sqrt(width)"),
            "class_directions": {"seed": derive_seed(adapter, seed, cell["id"], "class-directions"),
                                 "directions_sha256": tensor_hash(directions),
                                 "basis_cosines": cosines},
            "binding_qualification": binding_verdict(null_trials, positive_rows,
                                                     cfg["null_band"], cfg["structure_positive_floor"])}


def capture_and_score(model, panels, cfg, cell, seed, device, adapter=None):
    """Released post-defense quantized values plus the frozen per-binding scoring procedure."""
    adapter = adapter or ADAPTER
    batch = cfg["capture_batch_size"]
    quantizer = model.remote_path.quantizer
    # Prequant tensors are computed on the fly for the positive controls and are
    # never persisted; released tensors stay on the execution device until saved.
    pre = {key: prequant_capture(model, panels[name], batch,
                                 derive_seed(adapter, seed, cell["id"], key), device)
           for key, name in (("fit", "probe_fit"), ("test", "evaluation"))}
    released, release_support = {}, {}
    for key in pre:
        released[key], release_support[key] = quantized_release(pre[key], quantizer, batch)
    captures = {key: {"means": released[key].mean(1).cpu(), "tokens": released[key].cpu()}
                for key in released}
    controls = {"normal_release_support": release_support,
                "property": score_property(model, panels, pre, released, cfg, cell, seed, device, adapter),
                "structure": score_structure(model, panels, pre, released, cfg, cell, seed, device, adapter)}
    return captures, controls


def run_one(cfg, panels, sampler, cell, mode, seed, device, pairing, out, adapter=None):
    adapter = adapter or ADAPTER
    start = time.perf_counter()
    d1 = cell["defense"]["name"] == "property_noise"
    basis_log = None
    if mode == "all_local":
        model = build_initialized_model(BackboneConfig(**cfg["backbone"]), "all_local",
                                        None, seed, None)
        model.to(device)
        remote_initial = parameters_checksum(remote_parameters(model))
        trusted_initial = trusted_checksum(model)
        curve = train_steps(model, sampler, cfg["optimizer"], cfg["train_steps"], device)
    else:
        model = build_initialized_model(BackboneConfig(**cfg["backbone"]), "trained",
                                        SplitConfig(**cfg["split"], q_bits=cell["q_bits"]), seed,
                                        f"{cfg['protocol_id']}-{seed}")
        basis = torch.eye(cfg["split"]["width"])[:1] if d1 else None
        attach_defense(model, cell["defense"], basis,
                       derive_seed(adapter, seed, cell["id"], "defense"))
        model.to(device)
        remote_initial = parameters_checksum(remote_parameters(model))
        trusted_initial = trusted_checksum(model)
        if (pairing.setdefault("remote_initial_sha256", remote_initial) != remote_initial
                or pairing.setdefault("trusted_initial_sha256", trusted_initial) != trusted_initial):
            raise ValueError("Paired initialization differs across the seed's cells")
        if d1:
            curve, basis_log = train_with_basis_refits(
                model, sampler, cfg["optimizer"], cfg["train_steps"], cfg["refit_interval"],
                panels["basis_fit"], panels["basis_validation"], cfg["panel_batch_size"],
                derive_seed(adapter, seed, cell["id"], "basis-refit"), device)
        else:
            curve = train_steps(model, sampler, cfg["optimizer"], cfg["train_steps"], device)
    eval_batches = list(panel_batches(panels["evaluation"], cfg["eval_batch_size"]))
    final_loss = evaluate(model, eval_batches, derive_seed(adapter, seed, cell["id"], "eval"), device)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, out / "checkpoint.pt")
    write_json(out / "curve.json", curve)
    write_json(out / "resolved.json", {"config": cfg, "cell": cell, "mode": mode, "seed": seed})
    metrics = {**SCIENTIFIC_FLAGS, "seed": seed, "mode": mode, "cell": cell["id"],
               "device": str(device), "protocol_seeds": cfg["seeds"],
               "remote_initial_sha256": remote_initial,
               "remote_final_sha256": parameters_checksum(remote_parameters(model)),
               "trusted_initial_sha256": trusted_initial,
               "trusted_final_sha256": trusted_checksum(model),
               "final_eval_loss": final_loss,
               "basis_policy": "rank-1 punctuation basis fit at step 0, refit every "
                               f"{cfg['refit_interval']} steps" if d1 else None,
               "basis_log": basis_log,
               "claim_scope": CLAIM_SCOPE, "unevaluated": UNEVALUATED,
               "wall_seconds": time.perf_counter() - start}
    if mode != "all_local":
        captures, controls = capture_and_score(model, panels, cfg, cell, seed, device)
        torch.save(captures, out / "captures.pt")
        metrics["captures_sha256"] = {key: {name: tensor_hash(tensor) for name, tensor in parts.items()}
                                      for key, parts in captures.items()}
        metrics["controls"] = controls
        metrics["binding_qualification"] = {
            "property": controls["property"]["binding_qualification"],
            "structure": controls["structure"]["binding_qualification"]}
    write_json(out / "metrics.json", metrics)
    return metrics


def run(config_path, label=None, validate_only=False, device=None, only_seeds=None):
    cfg = yaml.safe_load(Path(config_path).read_text())
    if device is not None:
        cfg["device"] = device
    validation = validate(cfg, only_seeds=only_seeds)
    if validate_only:
        return validation
    device = resolve_device(cfg.get("device", "cpu"))
    verify_inherited_source()
    torch.set_num_threads(cfg.get("cpu_threads", 2))
    torch.use_deterministic_algorithms(not cfg.get("allow_nondeterministic_kernels", False))
    corpus = load_pinned_corpus(cfg["data"])
    index = load_document_index(cfg["data"])
    label = label or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", label):
        raise ValueError("Invalid run label")
    campaign = ROOT / "outputs/runs" / label
    campaign.mkdir(parents=True, exist_ok=False)
    source = source_provenance(campaign, device=cfg.get("device", "cpu"),
                               allow_nondeterministic=cfg.get("allow_nondeterministic_kernels", False))
    write_json(campaign / "config.json", cfg)
    panels, declaration = build_panels(cfg, index, corpus)
    torch.save(panels, campaign / "panels.pt")
    write_json(campaign / "panels.json", declaration)
    panels_sha = digest(campaign / "panels.pt")
    train_region = contiguous_region(corpus, index["train"])
    schema = json.loads((ROOT / "experiments/schemas/run_manifest.schema.json").read_text())
    seeds = list(cfg["seeds"]) if only_seeds is None else list(only_seeds)
    results, failures, run_ids = [], [], []
    for seed in seeds:
        sampler = BatchSampler(train_region, cfg["data"]["batch_size"],
                               cfg["data"]["sequence_length"], seed, "qualification-training")
        jobs = [({"id": "all-local", "q_bits": None, "defense": {"name": "all_local"}}, "all_local")]
        jobs += [(cell, "trained") for cell in cfg["cells"]]
        pairing = {}
        for cell, mode in jobs:
            run_id = f"p4-{label}-{cell['id']}-{mode}-s{seed}"
            out = campaign / run_id
            out.mkdir(exist_ok=False)
            inputs = {"seed": seed, "mode": mode, "cell": cell, "device": str(device),
                      "protocol_seeds": cfg["seeds"], "panels_sha256": panels_sha}
            manifest = {"schema_version": "p4-run-0.2", "run_id": run_id,
                        "protocol_id": cfg["protocol_id"], "cohort": "calibration",
                        "scientific_evidence": True, "seed": seed, "mode": mode,
                        "cell": cell["id"], "git_commit": source["git_commit"],
                        "source_tree_sha256": source["source_tree_sha256"],
                        "source_record": str((campaign / "source.json").relative_to(ROOT)),
                        "corpus_sha256": cfg["data"]["sha256"],
                        "matrix_sha256": digest(config_path), "freeze_record": None,
                        "inputs": inputs,
                        "started_at": datetime.now(timezone.utc).isoformat()}
            try:
                metrics = run_one(cfg, panels, sampler, cell, mode, seed, device, pairing, out)
                inputs["checkpoint"] = {"path": str((out / "checkpoint.pt").relative_to(ROOT)),
                                        "sha256": digest(out / "checkpoint.pt")}
                results.append({"run_id": run_id, "seed": seed, "cell": cell["id"], "mode": mode})
                manifest["status"] = "completed"
            except Exception as exc:
                manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                failures.append({"run_id": run_id, "error": manifest["error"]})
            manifest["artifacts"] = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)}
                                     for p in sorted(out.iterdir()) if p.is_file()]
            manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
            jsonschema.validate(manifest, schema)
            write_json(ROOT / "experiments/manifests" / f"{run_id}.json", manifest)
            run_ids.append(run_id)
            print(f"{run_id}: {manifest['status']}", flush=True)
    summary = {**SCIENTIFIC_FLAGS, "protocol_id": cfg["protocol_id"], "label": label,
               "device": str(device), "seeds": seeds, "protocol_seeds": cfg["seeds"],
               "panels_sha256": panels_sha,
               "expected_runs": validation["expected_runs"], "completed": len(results),
               "failed": len(failures), "failures": failures, "run_ids": run_ids,
               "claim_scope": CLAIM_SCOPE, "unevaluated": UNEVALUATED}
    write_json(campaign / "summary.json", summary)
    return summary
