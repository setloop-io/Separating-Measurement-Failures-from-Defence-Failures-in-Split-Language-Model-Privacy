"""Matched-null precision at the pilot operating point; engineering only.

Protocol `p4-null-precision-128-0.1` (docs/NULL_PRECISION_128_PROTOCOL.md): score
control conditions at sequence length 128 on the warmup-budget panels, using control
models re-trained inside this campaign from the bound shared-setup checkpoints under
the identical sampler, optimizer, seed, and device policy. A replayed d0 branch whose
remote/trusted checksums differ from the bound warmup-budget branch record fails the
preparation scenario and invalidates its dependent release conditions; the failure is
preserved, never patched over. `full_budget: false` exists only so tests can run the
pipeline against tiny synthetic source campaigns; the protocol config sets it true and
validation then pins every budget constant. Control seeds, protocol ID, and the two
distinct verified sources are always enforced. Estimator internals are unchanged from
`null_precision`; only the adapter label and the device-aware captures differ.
"""
from dataclasses import replace
from pathlib import Path
import json
import time

import numpy as np
import torch
import yaml
from pcr.data import BatchSampler
from pcr.model import BackboneConfig
from pcr.modes import apply_mode, remote_parameters
from pcr.privacy_attacks_p4b import sequence_targets
from pcr.split import SplitConfig
from pcr.train import build_initialized_model
from pcr.utils import derive_seed, parameters_checksum

from .corpus import contiguous_region, load_document_index, load_pinned_corpus
from .devices import resolve_device
from .experiment import ROOT, digest, write_json
from .model import attach_defense
from .null_precision import RidgeDesign, no_signal_trials, prequant_capture, quantized_control, signal_spec
from .scenario_io import Campaign, require_engineering, validate_seed_namespace
from .scenario_verification import verify_scenario
from .warmup_budget_scenarios import OPTIMIZER, boundary_means, train_steps
from .warmup_scenarios import FLAGS, OMITTED, TARGET, derangement, tensor_hash, trusted_checksum

PROTOCOL_ID = "p4-null-precision-128-0.1"
ADAPTER = "p4-ensemble-property-control-128-0.1"
CORPUS_SHA256 = "af3d0245c0ec27d7021f5fd29256bc845bb20cf9c562b37bbe23663b6ebe7436"
RELEASE_CONDITIONS = [{"id": "d0-q8", "q_bits": 8, "defense": {"name": "none"}},
                      {"id": "d1-warm-s1.0-q8", "q_bits": 8,
                       "defense": {"name": "property_noise", "noise_scale": 1.0}},
                      {"id": "d3-q1", "q_bits": 1, "defense": {"name": "aggressive_rate"}}]
BUDGET = {"panel_sizes": [8192, 16384], "matched_null_counts": [16, 64], "outer_null_repeats": 32,
          "positive_doses": [.5, 2.], "ridge_penalty": 1., "null_band": .02,
          "candidate_acceptance_threshold": .02, "property_positive_floor": .10,
          "capture_batch_size": 64, "cpu_threads": 2}


def validate(cfg, only_seeds=None):
    require_engineering(cfg, "null_precision_128")
    if cfg.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Protocol ID differs from the matched-null 128 engineering protocol")
    sources = cfg.get("sources")
    if not isinstance(sources, list) or len(sources) != 2:
        raise ValueError("Exactly two verified warmup-budget sources are required")
    for source in sources:
        if not isinstance(source, dict) or set(source) != {"control_seed", "setup_manifest", "branch_manifest"}:
            raise ValueError("Each source declares control_seed, setup_manifest, and branch_manifest")
    seeds = [s["control_seed"] for s in sources]
    validate_seed_namespace(seeds, "null_precision_128_engineering")
    setups = {s["setup_manifest"] for s in sources}
    branches = {s["branch_manifest"] for s in sources}
    if len(setups) != 2 or len(branches) != 2:
        raise ValueError("Two distinct warmup-budget setup/branch sources are required")
    executed = seeds if only_seeds is None else list(only_seeds)
    if not executed or any(s not in seeds for s in executed):
        raise ValueError("Executed seed subset must be nonempty and declared by the protocol")
    full = bool(cfg.get("full_budget", False))
    if full:
        if cfg.get("release_conditions") != RELEASE_CONDITIONS:
            raise ValueError("Release conditions differ from the engineering protocol")
        if any(cfg.get(k) != v for k, v in BUDGET.items()):
            raise ValueError("Changed scoring budget/threshold requires a new protocol")
    expected = len(executed) * (1 + len(cfg.get("release_conditions") or []))
    return {"expected_scenarios": expected, "full_budget": full, **FLAGS}


def load_setup(setup_manifest_path, branch_manifest_path):
    """Verify both source manifests, then load the bound checkpoint, panels, and metrics."""
    record, artifacts, cfg = verify_scenario(setup_manifest_path)
    if (record["status"] != "completed" or record["kind"] != "warmup_budget"
            or record["inputs"].get("stage") != "shared_setup"):
        raise ValueError("Need a completed warmup-budget shared setup")
    branch_record, branch_artifacts, branch_cfg = verify_scenario(branch_manifest_path)
    branch_inputs = branch_record["inputs"]
    if (branch_record["status"] != "completed" or branch_record["kind"] != "warmup_budget"
            or branch_inputs.get("stage") != "branch" or branch_inputs.get("branch") != "d0"
            or branch_inputs.get("mode") != "trained"):
        raise ValueError("Need a completed trained d0 branch record")
    if (branch_inputs.get("setup_run_id") != record["run_id"]
            or branch_inputs.get("seed") != record["inputs"]["seed"]):
        raise ValueError("Branch is not bound to the supplied shared setup")
    if branch_cfg != cfg:
        raise ValueError("Setup and branch were not executed under one source configuration")
    metrics = json.loads(artifacts["metrics.json"].read_text())
    state = torch.load(artifacts["common_checkpoint.pt"], weights_only=True)
    panels = torch.load(artifacts["panels.pt"], weights_only=True)
    declarations = json.loads(artifacts["panels.json"].read_text())
    if (metrics["seed"] != record["inputs"]["seed"]
            or metrics["remote_initial_sha256"] != metrics["remote_final_sha256"]):
        raise ValueError("Invalid shared setup identity or frozen state")
    for name, basis in state["bases"].items():
        if tensor_hash(basis) != metrics["bases"][name]["basis_sha256"]:
            raise ValueError("Checkpoint basis mismatch")
    for name, panel in panels.items():
        for key, expected in declarations["panels"][name]["tensor_sha256"].items():
            if tensor_hash(panel[key]) != expected:
                raise ValueError("Panel tensor mismatch")
    checkpoint_sha = digest(artifacts["common_checkpoint.pt"])
    panels_sha = digest(artifacts["panels.pt"])
    if (branch_inputs.get("checkpoint_sha256") != checkpoint_sha
            or branch_inputs.get("panels_sha256") != panels_sha):
        raise ValueError("Branch is not bound to the supplied setup artifacts")
    branch = {"record": branch_record,
              "metrics": json.loads(branch_artifacts["metrics.json"].read_text())}
    binding = {"setup_manifest": setup_manifest_path,
               "setup_manifest_sha256": digest(ROOT / setup_manifest_path),
               "branch_manifest": branch_manifest_path,
               "branch_manifest_sha256": digest(ROOT / branch_manifest_path),
               "setup_run_id": record["run_id"], "branch_run_id": branch_record["run_id"],
               "source_tree_sha256": record["source_tree_sha256"],
               "checkpoint_sha256": checkpoint_sha, "panels_sha256": panels_sha,
               "training_seed": metrics["seed"], "training_protocol": record["protocol_id"],
               "corpus_sha256": cfg["data"]["sha256"]}
    return state, panels, metrics, cfg, branch, binding


def prepare_control_model(state, panels, setup_metrics, source_cfg, branch, device):
    """Replay the bound d0-trained branch; any checksum divergence is a hard failure."""
    seed = setup_metrics["seed"]
    model = build_initialized_model(BackboneConfig(**source_cfg["backbone"]), "frozen_random",
                                    SplitConfig(**source_cfg["split"]), seed, "checkpoint-restore-only")
    attach_defense(model, {"name": "none"})
    model.load_state_dict(state["state_dict"], strict=True)
    if (trusted_checksum(model) != setup_metrics["trusted_warm_sha256"]
            or parameters_checksum(remote_parameters(model)) != setup_metrics["remote_final_sha256"]):
        raise ValueError("Restored model parameters differ from the bound shared setup")
    apply_mode(model, "trained")
    attach_defense(model, {"name": "none"})
    model.to(device)
    expected = {key: branch["metrics"][key] for key in
                ("remote_initial_sha256", "trusted_initial_sha256",
                 "remote_final_sha256", "trusted_final_sha256")}
    actual = {"remote_initial_sha256": parameters_checksum(remote_parameters(model)),
              "trusted_initial_sha256": trusted_checksum(model)}
    if (actual["remote_initial_sha256"] != expected["remote_initial_sha256"]
            or actual["trusted_initial_sha256"] != expected["trusted_initial_sha256"]):
        raise RuntimeError("Replay start state differs from the bound d0 branch record")
    corpus = load_pinned_corpus(source_cfg["data"])
    train_region = contiguous_region(corpus, load_document_index(source_cfg["data"])["train"])
    sampler = BatchSampler(train_region, source_cfg["data"]["batch_size"],
                           source_cfg["data"]["sequence_length"], seed, "warmup-training")
    steps, offset = source_cfg["branch_steps"], source_cfg["warmup_steps"]
    curve = train_steps(model, sampler, source_cfg["optimizer"], steps, device, offset)
    actual.update(remote_final_sha256=parameters_checksum(remote_parameters(model)),
                  trusted_final_sha256=trusted_checksum(model))
    if (actual["remote_final_sha256"] != expected["remote_final_sha256"]
            or actual["trusted_final_sha256"] != expected["trusted_final_sha256"]):
        raise RuntimeError("Replayed d0 branch differs from the bound warmup-budget record")
    checkpoint = {"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                  "bases": state["bases"]}
    metrics = {**FLAGS, "seed": seed, "stage": "control_model_preparation", "device": str(device),
               "replay_verification": {"expected": expected, "actual": actual, "verified": True},
               "new_training_updates": steps, "normal_candidate_scored": False,
               "bases": {name: tensor_hash(basis) for name, basis in state["bases"].items()},
               "corpus_sha256": source_cfg["data"]["sha256"],
               "curve": curve, "unevaluated": OMITTED}
    return {"model": model, "checkpoint": checkpoint, "metrics": metrics}


def restore_control_model(checkpoint_path, prepare_metrics, source_cfg, condition, seed, device):
    """Restore the just-saved control checkpoint and apply one release condition."""
    if digest(ROOT / checkpoint_path) != prepare_metrics["control_checkpoint_sha256"]:
        raise ValueError("Control checkpoint digest differs from the preparation record")
    saved = torch.load(ROOT / checkpoint_path, weights_only=True)
    model = build_initialized_model(BackboneConfig(**source_cfg["backbone"]), "frozen_random",
                                    SplitConfig(**source_cfg["split"]), prepare_metrics["seed"],
                                    "checkpoint-restore-only")
    attach_defense(model, {"name": "none"})
    model.load_state_dict(saved["state_dict"], strict=True)
    replayed = prepare_metrics["replay_verification"]["actual"]
    if (trusted_checksum(model) != replayed["trusted_final_sha256"]
            or parameters_checksum(remote_parameters(model)) != replayed["remote_final_sha256"]):
        raise ValueError("Restored control model differs from the prepared checkpoint")
    model.split = replace(model.split, q_bits=condition["q_bits"])
    model.remote_path.cfg = model.split
    basis = saved["bases"]["warm"] if condition["defense"]["name"] == "property_noise" else None
    model = attach_defense(model, condition["defense"], basis, derive_seed(ADAPTER, seed, "defense"))
    model.to(device)
    return model


def exercise(model, panels, cfg, seed, binding, out, device):
    start = time.perf_counter()
    batch = cfg["capture_batch_size"]
    quantizer = model.remote_path.quantizer
    calibration = boundary_means(model, panels["basis_fit"], batch, seed, device, released=False)
    spec = signal_spec(calibration, panels["basis_fit"]["tokens"], derive_seed(ADAPTER, seed, "signal"))
    # Prequant tensors stay on the execution device; only the small injected-signal
    # operands move there and only per-window means return to the host.
    spec_device = {**spec, "direction": spec["direction"].to(device)}
    pre = {key: prequant_capture(model, panels[name], batch, derive_seed(ADAPTER, seed, key), device)
           for key, name in (("fit", "probe_fit"), ("test", "probe_test"))}
    targets = {key: sequence_targets(panels[name]["tokens"])[TARGET]
               for key, name in (("fit", "probe_fit"), ("test", "probe_test"))}
    targets_device = {key: torch.as_tensor(targets[key], dtype=torch.float32, device=device)
                      for key in targets}
    # Retain released means for exact reproduction of the control-only calculations.
    normal, support = {}, {}
    for key in pre:
        means, support[key] = quantized_control(pre[key], targets_device[key], np.arange(len(pre[key])),
                                                spec_device, 0., quantizer, batch)
        normal[key] = means.cpu()
    stored = {"normal_means": normal, "targets": {k: torch.tensor(v) for k, v in targets.items()},
              "signal_spec": spec, "positive_controls": {}}
    panels_result = []
    for n in cfg["panel_sizes"]:
        y, yt = targets["fit"][:n], targets["test"][:n]
        design = RidgeDesign(normal["fit"][:n], normal["test"][:n], cfg["ridge_penalty"])
        null_trials = no_signal_trials(design, y, yt, cfg["outer_null_repeats"], cfg["matched_null_counts"], seed)
        donor_seeds = {k: derive_seed(ADAPTER, seed, n, "positive-donors", k) for k in ("fit", "test")}
        donors = {k: derangement(n, s) for k, s in donor_seeds.items()}
        positive_rows = []
        for dose in [0., *cfg["positive_doses"]]:
            features, checks = {}, {}
            for key in pre:
                means, checks[key] = quantized_control(pre[key][:n], targets_device[key][:n], donors[key],
                                                       spec_device, dose, quantizer, batch)
                features[key] = means.cpu()
                if dose == 0 and not torch.equal(features[key], normal[key][:n][donors[key]]):
                    raise RuntimeError("Zero-dose means do not equal the released donor means")
            control_design = RidgeDesign(features["fit"], features["test"], cfg["ridge_penalty"])
            score = control_design.score(y, yt, derive_seed(ADAPTER, seed, n, "positive-scorer"), cfg["matched_null_counts"])
            positive_rows.append({"dose": dose, "support": checks, "score": score,
                                  "features_sha256": {k: tensor_hash(v) for k, v in features.items()}})
            stored["positive_controls"][f"{n}-{dose}"] = features
        panels_result.append({"sequences_per_panel": n, "no_signal_trials": null_trials,
                              "positive_donor_seeds": donor_seeds, "positive_controls": positive_rows})
    torch.save(stored, out / "control_means.pt")
    return {"adapter": ADAPTER, "target": TARGET, "view": "mean_forward_values", **FLAGS,
            "binding": binding, "normal_candidate_scored": False, "new_training_updates": 0,
            "signal_spec": {k: v.tolist() if isinstance(v, torch.Tensor) else v for k, v in spec.items()},
            "normal_release_support": support, "panels": panels_result,
            "unevaluated": OMITTED, "wall_seconds": time.perf_counter() - start}


def require_full_budget_source(source_cfg):
    data = source_cfg["data"]
    if (data["sha256"] != CORPUS_SHA256 or source_cfg["optimizer"] != OPTIMIZER
            or source_cfg["branch_steps"] != 1500 or source_cfg["warmup_steps"] != 500
            or data["batch_size"] != 32 or data["sequence_length"] != 128):
        raise ValueError("Full-budget control replay requires the pinned warmup-budget constants")


def run(config_path, label=None, validate_only=False, device=None, only_seeds=None):
    cfg = yaml.safe_load(Path(config_path).read_text())
    if device is not None:
        cfg["device"] = device
    validation = validate(cfg, only_seeds=only_seeds)
    if validate_only:
        return validation
    device = resolve_device(cfg.get("device", "cpu"))
    sources = []
    for s in cfg["sources"]:
        if only_seeds is not None and s["control_seed"] not in only_seeds:
            continue
        state, panels, metrics, source_cfg, branch, binding = load_setup(s["setup_manifest"], s["branch_manifest"])
        if cfg.get("full_budget"):
            require_full_budget_source(source_cfg)
        sources.append((s["control_seed"], state, panels, metrics, source_cfg, branch, binding))
    campaign = Campaign(cfg, config_path, label)
    for seed, state, panels, setup_metrics, source_cfg, branch, source_binding in sources:
        inputs = {**source_binding, "control_seed": seed, "device": str(device),
                  "protocol_seeds": [s["control_seed"] for s in cfg["sources"]],
                  "stage": "control_model_preparation"}
        out, record = campaign.start(f"prepare-s{seed}", inputs)
        try:
            prepared = prepare_control_model(state, panels, setup_metrics, source_cfg, branch, device)
            torch.save(prepared["checkpoint"], out / "control_checkpoint.pt")
            checkpoint_path = str((out / "control_checkpoint.pt").relative_to(ROOT))
            checkpoint_sha = digest(out / "control_checkpoint.pt")
            write_json(out / "prepare_metrics.json",
                       {**prepared["metrics"], "binding": inputs,
                        "control_checkpoint_path": checkpoint_path,
                        "control_checkpoint_sha256": checkpoint_sha})
        except Exception as error:
            campaign.finish(out, record, error)
            # A failed replay invalidates the dependent release conditions; never proceed.
            continue
        campaign.finish(out, record)
        prepare_metrics = json.loads((out / "prepare_metrics.json").read_text())
        for condition in cfg["release_conditions"]:
            binding = {**source_binding, "control_seed": seed, "release_condition": condition,
                       "control_checkpoint_path": checkpoint_path,
                       "control_checkpoint_sha256": checkpoint_sha, "device": str(device),
                       "basis_sha256": (tensor_hash(state["bases"]["warm"])
                                        if condition["id"].startswith("d1-") else None)}
            out, record = campaign.start(f"{condition['id']}-s{seed}", binding)
            error = None
            try:
                model = restore_control_model(checkpoint_path, prepare_metrics, source_cfg,
                                              condition, seed, device)
                result = exercise(model, panels, cfg, seed, binding, out, device)
                write_json(out / "controls.json", result)
            except Exception as exc:
                error = exc
            campaign.finish(out, record, error)
    return campaign.summarize({"expected_scenarios": validation["expected_scenarios"],
                               "unevaluated": OMITTED, "qualified": False})
