"""Bound checkpoint control experiments; no normal candidate or scientific scoring."""
from dataclasses import replace
from pathlib import Path
import json
import time
import numpy as np
import torch
import yaml
from pcr.model import BackboneConfig
from pcr.modes import remote_parameters
from pcr.privacy_attacks_p4b import sequence_targets
from pcr.split import SplitConfig
from pcr.train import build_initialized_model
from pcr.utils import derive_seed, parameters_checksum

from .experiment import ROOT, digest, write_json
from .model import attach_defense
from .null_precision import ADAPTER, RidgeDesign, no_signal_trials, prequant_capture, quantized_control, signal_spec
from .scenario_io import Campaign, require_engineering, validate_seed_namespace
from .scenario_verification import verify_scenario
from .warmup_scenarios import FLAGS, OMITTED, TARGET, boundary_means, derangement, tensor_hash, trusted_checksum


def validate(cfg):
    require_engineering(cfg, "null_precision")
    seeds = [s["control_seed"] for s in cfg["sources"]]
    validate_seed_namespace(seeds, "null_precision_engineering")
    manifests = [s["manifest"] for s in cfg["sources"]]
    if len(manifests) != 2 or len(set(manifests)) != 2:
        raise ValueError("Two distinct engineering warmup sources required")
    expected = [{"id": "d0-q8", "q_bits": 8, "defense": {"name": "none"}},
                {"id": "d1-warm-s01-q8", "q_bits": 8, "defense": {"name": "property_noise", "noise_scale": .1}},
                {"id": "d3-q2", "q_bits": 2, "defense": {"name": "aggressive_rate"}}]
    if cfg["release_conditions"] != expected:
        raise ValueError("Release conditions differ from the engineering protocol")
    fixed = {"panel_sizes": [256, 1024], "matched_null_counts": [1, 16, 64],
             "outer_null_repeats": 32, "positive_doses": [.5, 2.], "ridge_penalty": 1.,
             "null_band": .02, "candidate_acceptance_threshold": .02, "property_positive_floor": .10,
             "capture_batch_size": 8, "cpu_threads": 2}
    if any(cfg[k] != v for k, v in fixed.items()):
        raise ValueError("Changed scoring budget/threshold requires a new protocol")
    return {"expected_scenarios": 6, **FLAGS}


def load_setup(path):
    record, artifacts, cfg = verify_scenario(path)
    if (record["status"] != "completed" or record["kind"] != "warmup"
            or record["inputs"].get("stage") != "shared_setup"):
        raise ValueError("Need a completed engineering shared-warmup checkpoint")
    metrics = json.loads(artifacts["metrics.json"].read_text())
    state = torch.load(artifacts["common_checkpoint.pt"], weights_only=True)
    panels = torch.load(artifacts["panels.pt"], weights_only=True)
    declarations = json.loads(artifacts["panels.json"].read_text())
    if (metrics["seed"] != record["inputs"]["seed"]
            or metrics["remote_initial_sha256"] != metrics["remote_final_sha256"]):
        raise ValueError("Invalid shared warmup identity or frozen state")
    for name, basis in state["bases"].items():
        if tensor_hash(basis) != metrics["bases"][name]["basis_sha256"]:
            raise ValueError("Checkpoint basis mismatch")
    for name, panel in panels.items():
        for key, expected in declarations[name]["tensor_sha256"].items():
            if tensor_hash(panel[key]) != expected:
                raise ValueError("Panel tensor mismatch")
    binding = {"source_manifest": path, "source_manifest_sha256": digest(ROOT / path),
               "source_run": record["run_id"], "source_tree_sha256": record["source_tree_sha256"],
               "checkpoint_sha256": digest(artifacts["common_checkpoint.pt"]),
               "panels_sha256": digest(artifacts["panels.pt"]), "training_seed": metrics["seed"],
               "training_protocol": record["protocol_id"], "warmup_steps": cfg["warmup_steps"],
               "training_treatment": "D0 q8, remote frozen during common warmup"}
    return state, panels, metrics, cfg, binding


def restore_model(state, metrics, cfg, condition, seed):
    model = build_initialized_model(BackboneConfig(**cfg["backbone"]), "frozen_random",
                                    SplitConfig(**cfg["split"]), metrics["seed"], "checkpoint-restore-only")
    attach_defense(model, {"name": "none"})
    model.load_state_dict(state["state_dict"], strict=True)
    if (trusted_checksum(model) != metrics["trusted_warm_sha256"]
            or parameters_checksum(remote_parameters(model)) != metrics["remote_final_sha256"]):
        raise ValueError("Restored model parameters differ from the saved common checkpoint")
    model.split = replace(model.split, q_bits=condition["q_bits"])
    model.remote_path.cfg = model.split
    basis = state["bases"]["warm"] if condition["defense"]["name"] == "property_noise" else None
    return attach_defense(model, condition["defense"], basis, derive_seed(ADAPTER, seed, "defense"))


def exercise(model, panels, cfg, seed, binding, out):
    start = time.perf_counter()
    batch = cfg["capture_batch_size"]
    quantizer = model.remote_path.quantizer
    calibration = boundary_means(model, panels["basis_fit"], batch, seed, released=False)
    spec = signal_spec(calibration, panels["basis_fit"]["tokens"], derive_seed(ADAPTER, seed, "signal"))
    pre = {key: prequant_capture(model, panels[name], batch, derive_seed(ADAPTER, seed, key))
           for key, name in (("fit", "probe_fit"), ("test", "probe_test"))}
    targets = {key: sequence_targets(panels[name]["tokens"])[TARGET]
               for key, name in (("fit", "probe_fit"), ("test", "probe_test"))}
    # Retain released means for exact reproduction of the control-only calculations.
    normal, support = {}, {}
    for key in pre:
        normal[key], support[key] = quantized_control(pre[key], targets[key], np.arange(len(pre[key])), spec, 0., quantizer, batch)
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
                features[key], checks[key] = quantized_control(pre[key][:n], targets[key][:n], donors[key], spec, dose, quantizer, batch)
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


def run(config_path, label=None, validate_only=False):
    cfg = yaml.safe_load(Path(config_path).read_text())
    valid = validate(cfg)
    if validate_only:
        return valid
    sources = [(s["control_seed"], load_setup(s["manifest"])) for s in cfg["sources"]]
    campaign = Campaign(cfg, config_path, label)
    for seed, (state, panels, metrics, source_cfg, source_binding) in sources:
        for condition in cfg["release_conditions"]:
            binding = {**source_binding, "control_seed": seed, "release_condition": condition,
                       "basis_sha256": tensor_hash(state["bases"]["warm"]) if condition["id"].startswith("d1-") else None}
            out, manifest = campaign.start(f"{condition['id']}-s{seed}", binding)
            error = None
            try:
                model = restore_model(state, metrics, source_cfg, condition, seed)
                result = exercise(model, panels, cfg, seed, binding, out)
                write_json(out / "controls.json", result)
            except Exception as exc:
                error = exc
            campaign.finish(out, manifest, error)
    return campaign.summarize({"expected_scenarios": valid["expected_scenarios"],
                               "qualified": False, "unevaluated": OMITTED})
