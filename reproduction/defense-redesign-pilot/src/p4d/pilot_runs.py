"""Exploratory pilot runner for Paper 4 defense pilot 1.0 (freeze-bound).

Protocol `p4-defense-pilot-1.0` (docs/PILOT_FREEZE.md): the exploratory pilot matrix —
7 cells x modes {trained, frozen_random, removed_remote} + all_local per seed, 22 runs
per seed — executed only under a committed freeze record (config hash, protocol,
seeds, corpus, and code commit all verified before anything runs; post-freeze edits
fail loudly and require a new freeze record by amendment). The training regime,
panels, estimators, and per-binding control scoring are identical to the calibration
qualification campaign (src/p4d/qualification_runs.py); this module reuses that
machinery with the pilot's adapter label, seed namespace, mode matrix, and the
freeze document's named omissions. Records carry cohort "exploratory" and
scientific_evidence true; privacy stays undetermined and no gate value appears in any
per-run record (pairing happens only in the analyzer). `full_budget: false` exists
only so tests can run the pipeline on tiny synthetic corpora with a reduced cell
list; the protocol config sets it true and validation then pins every constant.
"""
from datetime import datetime, timezone
from pathlib import Path
import json
import re
import time

import torch
import yaml
import jsonschema
from pcr.data import BatchSampler
from pcr.model import BackboneConfig
from pcr.modes import remote_parameters
from pcr.split import SplitConfig
from pcr.train import build_initialized_model
from pcr.utils import derive_seed, parameters_checksum

from .corpus import contiguous_region, load_document_index, load_pinned_corpus
from .devices import resolve_device
from .experiment import (ROOT, digest, evaluate, source_provenance,
                         verify_inherited_source, write_json)
from .freeze import require_freeze_record
from .model import attach_defense
from .qualification_runs import (BUDGET, CELLS, OPTIMIZER, SPLIT,
                                 build_panels, capture_and_score,
                                 train_with_basis_refits)
from .scenario_io import validate_seed_namespace
from .seed_audit import audit_seeds
from .warmup_budget_scenarios import train_steps
from .warmup_scenarios import panel_batches, tensor_hash, trusted_checksum

PROTOCOL_ID = "p4-defense-pilot-1.0"
ADAPTER = "p4-defense-pilot-1.0"
SEEDS = [411101, 411102, 411103]
MODES = ["trained", "frozen_random", "removed_remote"]
SCIENTIFIC_FLAGS = {"cohort": "exploratory", "scientific_evidence": True,
                    "privacy_outcome": "undetermined"}
UNEVALUATED = ["timing", "optimizer state beyond the remote module",
               "full evolving remote state", "active/adaptive attackers",
               "within-run joint tensors", "cross-run joint attacks",
               "natural-sequence membership", "canary membership",
               "semantics beyond byte-structure classes", "returned values",
               "full-precision inbound/outbound gradients", "quantizer metadata attacks",
               "history accumulation", "token inversion (secondary, unqualified here)"]
CLAIM_SCOPE = ("Exploratory pilot records under the frozen p4-defense-pilot-1.0 protocol. "
               "Pilot-eligibility, when evaluated by the analyzer, is exploratory evidence "
               "only and unlocks no positive defense claim; privacy remains undetermined. "
               "A captured or declared channel is not an evaluated attack.")


def validate(cfg, only_seeds=None):
    if cfg.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Protocol ID differs from the pilot freeze")
    if cfg.get("cohort") != "exploratory" or cfg.get("status") != "frozen":
        raise ValueError("The pilot requires the frozen exploratory cohort configuration")
    seeds = cfg.get("seeds")
    validate_seed_namespace(seeds, "pilot")
    cells = cfg.get("cells")
    if (not isinstance(cells, list) or not cells
            or len({c.get("id") for c in cells if isinstance(c, dict)}) != len(cells)
            or any(c not in CELLS for c in cells)):
        raise ValueError("Cells must be drawn from the frozen pilot cell set")
    if cfg.get("modes") != MODES:
        raise ValueError("Mode list differs from the pilot freeze")
    if cfg.get("split") != SPLIT:
        raise ValueError("The pilot requires the fixed Paper 4 anchor")
    if not isinstance(cfg.get("freeze_record"), str) or not cfg["freeze_record"]:
        raise ValueError("The frozen pilot config must name its freeze record")
    executed = list(seeds) if only_seeds is None else list(only_seeds)
    if not executed or any(s not in seeds for s in executed):
        raise ValueError("Executed seed subset must be nonempty and declared by the protocol")
    full = bool(cfg.get("full_budget", False))
    if full:
        if sorted(seeds or []) != SEEDS:
            raise ValueError("The frozen pilot declares exactly the three pilot seeds")
        if cells != CELLS:
            raise ValueError("Cell set differs from the pilot freeze")
        if cfg.get("optimizer") != OPTIMIZER:
            raise ValueError("Changed optimizer constants require a new freeze")
        if any(cfg.get(k) != v for k, v in BUDGET.items()):
            raise ValueError("Changed training/scoring budget requires a new freeze")
        data = cfg.get("data", {})
        if data.get("sequence_length") != 128 or data.get("batch_size") != 32:
            raise ValueError("Changed sequence/batch budget requires a new freeze")
        if data.get("sha256") != "af3d0245c0ec27d7021f5fd29256bc845bb20cf9c562b37bbe23663b6ebe7436" \
                or not data.get("path") or not data.get("documents"):
            raise ValueError("The pilot requires the pinned Gutenberg corpus")
    expected = len(executed) * (len(cells) * len(cfg.get("modes") or []) + 1)
    return {"expected_runs": expected, "full_budget": full, **SCIENTIFIC_FLAGS}


def run_one(cfg, panels, sampler, cell, mode, seed, device, pairing, out):
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
        model = build_initialized_model(BackboneConfig(**cfg["backbone"]), mode,
                                        SplitConfig(**cfg["split"], q_bits=cell["q_bits"]), seed,
                                        f"{cfg['protocol_id']}-{seed}")
        basis = torch.eye(cfg["split"]["width"])[:1] if d1 else None
        # Paired defense draws per seed across cells: one defense seed per seed.
        attach_defense(model, cell["defense"], basis, derive_seed(ADAPTER, seed, "defense"))
        model.to(device)
        remote_initial = parameters_checksum(remote_parameters(model))
        trusted_initial = trusted_checksum(model)
        if pairing.setdefault("trusted_initial_sha256", trusted_initial) != trusted_initial:
            raise ValueError("Paired trusted initialization differs across the seed's cells")
        if mode != "removed_remote" and pairing.setdefault(
                "remote_initial_sha256", remote_initial) != remote_initial:
            raise ValueError("Paired remote initialization differs across the seed's cells")
        if d1:
            curve, basis_log = train_with_basis_refits(
                model, sampler, cfg["optimizer"], cfg["train_steps"], cfg["refit_interval"],
                panels["basis_fit"], panels["basis_validation"], cfg["panel_batch_size"],
                derive_seed(ADAPTER, seed, cell["id"], "basis-refit"), device)
        else:
            curve = train_steps(model, sampler, cfg["optimizer"], cfg["train_steps"], device)
        if mode == "frozen_random" and remote_initial != parameters_checksum(remote_parameters(model)):
            raise RuntimeError("Frozen branch remote parameters changed")
    eval_batches = list(panel_batches(panels["evaluation"], cfg["eval_batch_size"]))
    final_loss = evaluate(model, eval_batches, derive_seed(ADAPTER, seed, cell["id"], "eval"), device)
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
        captures, controls = capture_and_score(model, panels, cfg, cell, seed, device, ADAPTER)
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
    freeze_record = require_freeze_record(cfg, config_path)
    device = resolve_device(cfg.get("device", "cpu"))
    verify_inherited_source()
    torch.set_num_threads(cfg.get("cpu_threads", 2))
    torch.use_deterministic_algorithms(not cfg.get("allow_nondeterministic_kernels", False))
    corpus = load_pinned_corpus(cfg["data"])
    index = load_document_index(cfg["data"])
    label = label or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", label):
        raise ValueError("Invalid run label")
    audit = audit_seeds(ROOT, cohorts=("pilot",), exclude_prefix=f"p4-{label}-")
    if not audit["passed"]:
        failed = [f["check"] for f in audit["findings"] if not f["passed"]]
        raise ValueError(f"Seed audit failed at execution time: {failed}")
    campaign = ROOT / "outputs/runs" / label
    campaign.mkdir(parents=True, exist_ok=False)
    write_json(campaign / "seed_audit.json", audit)
    source = source_provenance(campaign, device=cfg.get("device", "cpu"),
                               allow_nondeterministic=cfg.get("allow_nondeterministic_kernels", False))
    write_json(campaign / "config.json", cfg)
    panels, declaration = build_panels(cfg, index, corpus, PROTOCOL_ID)
    torch.save(panels, campaign / "panels.pt")
    write_json(campaign / "panels.json", declaration)
    panels_sha = digest(campaign / "panels.pt")
    train_region = contiguous_region(corpus, index["train"])
    schema = json.loads((ROOT / "experiments/schemas/run_manifest.schema.json").read_text())
    seeds = list(cfg["seeds"]) if only_seeds is None else list(only_seeds)
    results, failures, run_ids = [], [], []
    for seed in seeds:
        sampler = BatchSampler(train_region, cfg["data"]["batch_size"],
                               cfg["data"]["sequence_length"], seed, "pilot-training")
        jobs = [({"id": "all-local", "q_bits": None, "defense": {"name": "all_local"}}, "all_local")]
        jobs += [(cell, mode) for cell in cfg["cells"] for mode in cfg["modes"]]
        pairing = {}
        for cell, mode in jobs:
            run_id = f"p4-{label}-{cell['id']}-{mode}-s{seed}"
            out = campaign / run_id
            out.mkdir(exist_ok=False)
            inputs = {"seed": seed, "mode": mode, "cell": cell, "device": str(device),
                      "protocol_seeds": cfg["seeds"], "panels_sha256": panels_sha}
            manifest = {"schema_version": "p4-run-0.2", "run_id": run_id,
                        "protocol_id": cfg["protocol_id"], "cohort": "exploratory",
                        "scientific_evidence": True, "seed": seed, "mode": mode,
                        "cell": cell["id"], "git_commit": source["git_commit"],
                        "source_tree_sha256": source["source_tree_sha256"],
                        "source_record": str((campaign / "source.json").relative_to(ROOT)),
                        "corpus_sha256": cfg["data"]["sha256"],
                        "matrix_sha256": digest(config_path),
                        "freeze_record": cfg["freeze_record"],
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
               "panels_sha256": panels_sha, "freeze_record": cfg["freeze_record"],
               "expected_runs": validation["expected_runs"], "completed": len(results),
               "failed": len(failures), "failures": failures, "run_ids": run_ids,
               "claim_scope": CLAIM_SCOPE, "unevaluated": UNEVALUATED}
    write_json(campaign / "summary.json", summary)
    return summary
