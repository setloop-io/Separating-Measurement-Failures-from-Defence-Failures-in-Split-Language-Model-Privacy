"""Training-budget scan engineering on bound warmup-budget setups; engineering only.

Protocol `p4-budget-scan-0.1` (docs/BUDGET_SCAN_PROTOCOL.md): resume each bound
shared warmup checkpoint and continue paired branches to longer total budgets at
the unchanged operating point. The d0 trained branch replays the source campaign's
recorded 1,500-step checksum before continuing; a mismatch fails the scenario.
`full_budget: false` exists only so tests can run tiny budgets; the protocol
config sets it true and validation then pins every constant.
"""
import json
from pathlib import Path
import time

import torch
import yaml
from pcr.data import BatchSampler
from pcr.modes import apply_mode, remote_parameters
from pcr.model import BackboneConfig
from pcr.split import SplitConfig
from pcr.train import build_initialized_model
from pcr.utils import derive_seed, parameters_checksum

from .corpus import contiguous_region, load_document_index, load_pinned_corpus
from .devices import resolve_device
from .experiment import ROOT, digest, evaluate, write_json
from .model import attach_defense
from .scenario_io import Campaign, require_engineering, validate_seed_namespace
from .scenario_verification import verify_scenario
from .warmup_budget_scenarios import OPTIMIZER, train_steps
from .warmup_scenarios import FLAGS, OMITTED, panel_batches, trusted_checksum

PROTOCOL_ID = "p4-budget-scan-0.1"
CONTROL_SEEDS = [410503, 410504]
CORPUS_SHA256 = "af3d0245c0ec27d7021f5fd29256bc845bb20cf9c562b37bbe23663b6ebe7436"
BRANCHES = [{"branch": "d0", "mode": "trained", "budget": 4000},
            {"branch": "d0", "mode": "frozen_random", "budget": 4000},
            {"branch": "all_local", "mode": "all_local", "budget": 4000},
            {"branch": "d0", "mode": "trained", "budget": 8000},
            {"branch": "d0", "mode": "frozen_random", "budget": 8000},
            {"branch": "all_local", "mode": "all_local", "budget": 8000}]
WARMUP_STEPS = 500
BRANCH_STEPS = 1500


def validate(cfg, only_seeds=None):
    require_engineering(cfg, "warmup_budget")
    if cfg.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Protocol ID differs from the budget-scan engineering protocol")
    seeds = [s["control_seed"] for s in cfg.get("sources", [])]
    validate_seed_namespace(seeds, "budget_scan_engineering")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Sources must declare distinct control seeds")
    for source in cfg["sources"]:
        if set(source) != {"control_seed", "setup_manifest", "trained_manifest"}:
            raise ValueError("Each source binds a control seed and two manifests")
    executed = seeds if only_seeds is None else list(only_seeds)
    if not executed or any(s not in seeds for s in executed):
        raise ValueError("Executed seed subset must be nonempty and declared by the protocol")
    expected = len(executed) * len(cfg["branches"])
    if not cfg.get("full_budget", False):
        return {"expected_scenarios": expected, "full_budget": False, **FLAGS}
    if len(seeds) != 2 or seeds != CONTROL_SEEDS:
        raise ValueError("Changed scan control seeds require a new protocol")
    if cfg.get("branches") != BRANCHES:
        raise ValueError("Changed scan seeds or branch budgets require a new protocol")
    if cfg.get("warmup_steps") != WARMUP_STEPS or cfg.get("optimizer") != OPTIMIZER:
        raise ValueError("Changed warmup/optimizer constants require a new protocol")
    if cfg.get("reference_branch_steps") != BRANCH_STEPS:
        raise ValueError("The replay reference must equal the source campaign's branch steps")
    data = cfg.get("data", {})
    if data.get("sequence_length") != 128 or data.get("batch_size") != 32:
        raise ValueError("Changed sequence/batch budget requires a new protocol")
    if data.get("sha256") != CORPUS_SHA256 or not data.get("path") or not data.get("documents"):
        raise ValueError("Budget-scan engineering requires the pinned Gutenberg corpus")
    return {"expected_scenarios": expected, "full_budget": True, **FLAGS}


def load_setup(manifest_path, trained_manifest_path):
    record, artifacts, source_cfg = verify_scenario(manifest_path)
    if (record["status"] != "completed" or record["kind"] != "warmup_budget"
            or record["inputs"].get("stage") != "shared_setup"):
        raise ValueError("Need a completed warmup-budget shared setup")
    trained, trained_artifacts, _ = verify_scenario(trained_manifest_path)
    if (trained["status"] != "completed" or trained["kind"] != "warmup_budget"
            or trained["inputs"].get("stage") != "branch"
            or trained["inputs"].get("branch") != "d0" or trained["inputs"].get("mode") != "trained"
            or trained["inputs"].get("setup_run_id") != record["run_id"]
            or trained["inputs"].get("seed") != record["inputs"].get("seed")):
        raise ValueError("Need the paired completed d0 trained branch of the same setup")
    metrics = json.loads(artifacts["metrics.json"].read_text())
    trained_metrics = json.loads(trained_artifacts["metrics.json"].read_text())
    state = torch.load(artifacts["common_checkpoint.pt"], weights_only=True)
    panels = torch.load(artifacts["panels.pt"], weights_only=True)
    if metrics["remote_initial_sha256"] != metrics["remote_final_sha256"]:
        raise ValueError("Invalid frozen shared warmup")
    binding = {"source_manifest": manifest_path, "source_manifest_sha256": digest(ROOT / manifest_path),
               "trained_manifest": trained_manifest_path,
               "trained_manifest_sha256": digest(ROOT / trained_manifest_path),
               "source_run": record["run_id"], "trained_run": trained["run_id"],
               "source_tree_sha256": record["source_tree_sha256"],
               "checkpoint_sha256": digest(artifacts["common_checkpoint.pt"]),
               "panels_sha256": digest(artifacts["panels.pt"]),
               "training_seed": metrics["seed"], "training_protocol": record["protocol_id"],
               "reference_remote_final_sha256": trained_metrics["remote_final_sha256"],
               "reference_trusted_final_sha256": trained_metrics["trusted_final_sha256"]}
    return state, panels, metrics, source_cfg, binding


def run(config_path, label=None, validate_only=False, device=None, only_seeds=None):
    cfg = yaml.safe_load(Path(config_path).read_text())
    if device is not None:
        cfg["device"] = device
    validation = validate(cfg, only_seeds=only_seeds)
    if validate_only:
        return validation
    device = resolve_device(cfg.get("device", "cpu"))
    corpus = load_pinned_corpus(cfg["data"])
    index = load_document_index(cfg["data"])
    train_region = contiguous_region(corpus, index["train"])
    campaign = Campaign(cfg, config_path, label)
    batch, seq = cfg["data"]["batch_size"], cfg["data"]["sequence_length"]
    sources = {s["control_seed"]: s for s in cfg["sources"]}
    executed = [s["control_seed"] for s in cfg["sources"]] if only_seeds is None else list(only_seeds)
    for control_seed in executed:
        source = sources[control_seed]
        state, panels, setup_metrics, source_cfg, binding = load_setup(
            source["setup_manifest"], source["trained_manifest"])
        training_seed = binding["training_seed"]
        sampler = BatchSampler(train_region, batch, seq, training_seed, "warmup-training")
        eval_batches = list(panel_batches(panels["probe_test"], batch))
        warmup_steps, reference_steps = source_cfg["warmup_steps"], source_cfg["branch_steps"]
        if warmup_steps != cfg["warmup_steps"] or reference_steps != cfg["reference_branch_steps"]:
            raise ValueError("Source campaign schedule differs from the scan declaration")
        inputs_base = {**binding, "control_seed": control_seed, "device": str(device),
                       "protocol_seeds": [s["control_seed"] for s in cfg["sources"]]}
        for spec in cfg["branches"]:
            name = f"{spec['branch']}-{spec['mode']}-b{spec['budget']}-s{control_seed}"
            inputs = {**inputs_base, "stage": "branch", **spec}
            out, record = campaign.start(name, inputs)
            error = None
            try:
                start = time.perf_counter()
                replay = None
                if spec["mode"] == "all_local":
                    model = build_initialized_model(BackboneConfig(**source_cfg["backbone"]),
                                                    "all_local", None, training_seed, None)
                    model.to(device)
                    curve = train_steps(model, sampler, cfg["optimizer"], spec["budget"], device, 0)
                else:
                    model = build_initialized_model(BackboneConfig(**source_cfg["backbone"]),
                                                    "frozen_random", SplitConfig(**source_cfg["split"]),
                                                    training_seed, "checkpoint-restore-only")
                    attach_defense(model, {"name": "none"})
                    model.load_state_dict(state["state_dict"], strict=True)
                    model.to(device)
                    apply_mode(model, spec["mode"])
                    if spec["branch"] == "d0" and spec["mode"] == "trained":
                        first = train_steps(model, sampler, cfg["optimizer"],
                                            reference_steps, device, warmup_steps)
                        replay = {"steps": reference_steps,
                                  "remote_sha256": parameters_checksum(remote_parameters(model)),
                                  "trusted_sha256": trusted_checksum(model),
                                  "reference_remote_sha256": binding["reference_remote_final_sha256"],
                                  "reference_trusted_sha256": binding["reference_trusted_final_sha256"]}
                        if (replay["remote_sha256"] != replay["reference_remote_sha256"]
                                or replay["trusted_sha256"] != replay["reference_trusted_sha256"]):
                            raise RuntimeError("Budget-scan replay differs from the recorded reference branch")
                        replay["matches_reference"] = True
                        remaining = spec["budget"] - warmup_steps - reference_steps
                        if remaining < 0:
                            raise ValueError("Scan budget must exceed warmup plus reference steps")
                        curve = first + (train_steps(model, sampler, cfg["optimizer"], remaining,
                                                     device, warmup_steps + reference_steps)
                                         if remaining else [])
                    else:
                        curve = train_steps(model, sampler, cfg["optimizer"],
                                            spec["budget"] - warmup_steps, device, warmup_steps)
                remote_before_after = parameters_checksum(remote_parameters(model))
                if (spec["mode"] == "frozen_random"
                        and remote_before_after != setup_metrics["remote_final_sha256"]):
                    raise RuntimeError("Frozen scan branch remote parameters changed")
                metrics = {**FLAGS, **inputs,
                           "remote_final_sha256": remote_before_after,
                           "trusted_final_sha256": trusted_checksum(model),
                           "budget_steps": spec["budget"],
                           "final_eval_loss": evaluate(model, eval_batches,
                                                       derive_seed("warmup-budget-eval", training_seed), device),
                           "curve_tail": curve[-5:], "replay": replay,
                           "unevaluated": OMITTED,
                           "wall_seconds": time.perf_counter() - start}
                write_json(out / "metrics.json", metrics)
            except Exception as exc:  # noqa: BLE001 - recorded as a failed scenario
                error = exc
            campaign.finish(out, record, error)
    return campaign.summarize({"expected_scenarios": validation["expected_scenarios"],
                               "unevaluated": OMITTED, "qualified": False})
