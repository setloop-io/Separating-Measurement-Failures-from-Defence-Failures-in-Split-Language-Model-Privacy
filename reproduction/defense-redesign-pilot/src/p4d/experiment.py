"""Manifest-producing engineering runner. Draft scientific matrices cannot execute."""
from __future__ import annotations
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.metadata import distributions
from pathlib import Path
import hashlib
import json
import os
import platform
import subprocess
import tarfile
import time

import numpy as np
import torch
import yaml
import jsonschema
from pcr.data import BatchSampler, load_corpus
from pcr.model import BackboneConfig
from pcr.split import SplitConfig
from pcr.train import build_initialized_model, build_optimizer, OptimConfig
from pcr.modes import remote_parameters, trainable_parameters
from pcr.utils import derive_seed, parameters_checksum
from pcr.privacy_attacks_p4b import sequence_targets
from .calibration import estimate_property_directions, score_properties, score_structure
from .defenses import noise_stream
from .devices import device_record, resolve_device
from .model import attach_defense, BoundaryCapture

ROOT = Path(__file__).resolve().parents[2]
CORPUS_SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
CHANNELS = ("forward_values", "returned_values", "outbound_gradients", "inbound_gradients")
UNEVALUATED = ["timing", "optimizer state", "full evolving remote state", "active attackers",
               "adaptive attackers", "within-run joint tensors", "cross-run joint attacks",
               "history accumulation", "canary membership", "natural-sequence membership",
               "unrestricted semantic recovery", "quantizer metadata attacks",
               "returned values attacks", "full-precision inbound/outbound gradient attacks"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def source_files():
    paths = []
    for p in ROOT.rglob("*"):
        rel = p.relative_to(ROOT)
        if any(part in (".git", ".venv", "__pycache__", ".pytest_cache", "build", "dist")
               or part.endswith(".egg-info") for part in rel.parts):
            continue
        if rel.parts[0] in ("outputs", "data") or rel.parts[:2] == ("experiments", "manifests"):
            continue
        if p.is_file() and (p.suffix in (".py", ".md", ".toml", ".lock", ".yaml", ".json", ".sh", ".tex", ".bib")
                            or p.name in ("LICENSE", "Makefile", ".gitignore")):
            paths.append(p)
    return sorted(paths)


def source_provenance(output, device="cpu", allow_nondeterministic=False):
    files = {str(p.relative_to(ROOT)): digest(p) for p in source_files()}
    tree_hash = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                            capture_output=True, check=False)
    execution = {**device_record(resolve_device(device)),
                 "torch_threads": torch.get_num_threads(),
                 "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
    if allow_nondeterministic:
        execution["allow_nondeterministic_kernels"] = True
    record = {"git_commit": result.stdout.strip() if result.returncode == 0 else None,
              "source_tree_sha256": tree_hash, "files": files,
              "paper3_source": json.loads((ROOT / "vendor/paper3/SOURCE.json").read_text()),
              "environment": sorted({f"{d.metadata['Name']}=={d.version}" for d in distributions()}),
              "python": __import__("sys").version,
              "host": {"platform": platform.platform(), "machine": platform.machine(),
                       "processor": platform.processor(), "logical_cpu_count": os.cpu_count()},
              "execution": execution}
    write_json(output / "source.json", record)
    with tarfile.open(output / "source.tar.gz", "w:gz") as archive:
        for rel in files:
            archive.add(ROOT / rel, arcname=rel)
    return record


def verify_inherited_source():
    root = ROOT / "vendor/paper3"
    record = json.loads((root / "SOURCE.json").read_text())
    mismatches = [name for name, expected in record["files"].items()
                  if not (root / name).is_file() or digest(root / name) != expected]
    if mismatches:
        raise ValueError(f"Inherited source differs from pinned snapshot: {mismatches}")
    import pcr
    installed = Path(pcr.__file__).resolve().parent
    for name, expected in record["files"].items():
        if name.startswith("src/pcr/") and digest(installed / Path(name).name) != expected:
            raise ValueError(f"Installed pcr differs from pinned source: {name}")
    return {"revision": record["revision"], "verified_files": len(record["files"]),
            "installed_pcr": str(installed)}


def validate_matrix(cfg, execute=False):
    if not cfg["protocol_id"].startswith("p4-"):
        raise ValueError("Paper 4 run IDs/protocols must be distinct from Paper 3")
    if execute and (cfg["cohort"] != "engineering" or cfg["status"] != "engineering_ready"):
        raise ValueError("Only engineering diagnostics execute; pilot/confirmation require a future freeze")
    if len(cfg["seeds"]) != len(set(cfg["seeds"])):
        raise ValueError("Duplicate training seed")
    if any(s in cfg.get("reserved_prior_seeds", []) for s in cfg["seeds"]):
        raise ValueError("Training seeds overlap inherited Paper 3 seeds")
    if len({c["id"] for c in cfg["cells"]}) != len(cfg["cells"]):
        raise ValueError("Duplicate cell ID")
    if (cfg["split"]["start"], cfg["split"]["span"], cfg["split"]["width"]) != (2, 4, 256):
        raise ValueError("Initial study requires the span4@2 width256 anchor")
    for c in cfg["cells"]:
        if c["defense"]["name"] not in ("none", "property_noise", "aggressive_rate"):
            raise ValueError("Unimplemented or diagnostic-only defense")
        if not 1 <= c["q_bits"] <= 32:
            raise ValueError("Invalid precision")
        if c["defense"]["name"] == "property_noise" and c["q_bits"] != 8:
            raise ValueError("Initial D1 comparison keeps q=8; combined doses need a new matrix")
    cuts = cfg["data"]["fractions"]
    if len(cuts) != 5 or cuts[0] != 0 or cuts[-1] != 1 or any(b <= a for a,b in zip(cuts,cuts[1:])):
        raise ValueError("Need four nonoverlapping ordered data regions")
    if cfg["modes"] != ["trained", "frozen_random", "removed_remote"]:
        raise ValueError("Initial study requires all three paired split controls")
    return {"protocol": cfg["protocol_id"], "status": cfg["status"],
            "run_count": len(cfg["seeds"]) * (len(cfg["cells"]) * len(cfg["modes"]) + 1),
            "execution_enabled": cfg["cohort"] == "engineering" and cfg["status"] == "engineering_ready"}


@contextmanager
def measurement(model, seed):
    training = model.training
    try:
        model.eval()
        with noise_stream(model, seed):
            yield
    finally:
        model.train(training)


def evaluate(model, batches, seed, device=None):
    with measurement(model, seed), torch.no_grad():
        losses = []
        for x, y in batches:
            if device is not None:
                x, y = x.to(device), y.to(device)
            losses.append(float(torch.nn.functional.cross_entropy(
                model(x).reshape(-1, 256), y.reshape(-1))))
    return float(np.mean(losses))


def fit_initial_basis(model, batches, device=None):
    """P0 wiring diagnostic only: same-model initial encoder, separate calibration text.

    A useful post-warmup schedule is intentionally not inferred from this test.
    """
    latents, targets = [], []
    def capture(_module, _args, output):
        latents.append(output.detach().mean(1).cpu().numpy())
    hook = model.remote_path.encoder.register_forward_hook(capture)
    try:
        with torch.no_grad():
            for x, _ in batches:
                if device is not None:
                    x = x.to(device)
                model(x)
                targets.append(sequence_targets(x.cpu())["punctuation_fraction"])
    finally:
        hook.remove()
    return estimate_property_directions(np.concatenate(latents),
                                        {"punctuation_fraction": np.concatenate(targets)})


def capture_panel(model, batches, seed, device=None):
    events = []
    with measurement(model, seed):
        for x, y in batches:
            if device is not None:
                x, y = x.to(device), y.to(device)
            model.zero_grad(set_to_none=True)
            capture = BoundaryCapture()
            logits = model(x, capture=capture)
            torch.nn.functional.cross_entropy(logits.reshape(-1, 256), y.reshape(-1)).backward()
            if any(k not in capture.values for k in CHANNELS):
                raise RuntimeError("Missing forward/backward capture")
            events.append({**capture.values, "tokens": x.cpu().clone(), "next_tokens": y.cpu().clone(),
                           "latents": capture.values["forward_values"]})
    model.zero_grad(set_to_none=True)
    return events


def run_one(cfg, cell, mode, seed, regions, out):
    out.mkdir(parents=True, exist_ok=False)
    device = resolve_device(cfg.get("device", "cpu"))
    seq, batch = cfg["data"]["sequence_length"], cfg["data"]["batch_size"]
    samplers = {key: BatchSampler(region, batch, seq, seed, f"p4-{key}")
                for key, region in regions.items()}
    model_cfg = BackboneConfig(**cfg["backbone"])
    split_cfg = None if mode == "all_local" else SplitConfig(**cfg["split"], q_bits=cell["q_bits"])
    pairing = f"{cfg['protocol_id']}-{seed}"  # Same initialization across strength variants.
    model = build_initialized_model(model_cfg, mode, split_cfg, seed, pairing).to(device)
    basis = None
    if mode != "all_local":
        if cell["defense"]["name"] == "property_noise":
            basis = fit_initial_basis(model, samplers["calibration"].batches(cfg["calibration_batches"]), device)
        attach_defense(model, cell["defense"], basis, derive_seed("p4-defense", seed))
        model.to(device)
    model.train()
    opt_cfg = OptimConfig(**cfg["optimizer"])
    optimizer = build_optimizer(model, opt_cfg)
    initial_remote = parameters_checksum(remote_parameters(model))
    eval_batches = samplers["evaluation"].batches(cfg["eval_batches"])
    start = time.perf_counter()
    curve = [{"step": 0, "eval_loss": evaluate(model, eval_batches, derive_seed("p4-eval", seed), device)}]
    for step in range(opt_cfg.steps):
        for group in optimizer.param_groups:
            group["lr"] = opt_cfg.lr_at(step)
        x, y = samplers["train"].batch(step)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(x).reshape(-1, 256), y.reshape(-1))
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_parameters(model), opt_cfg.grad_clip)
        optimizer.step()
        curve.append({"step": step + 1, "train_loss": float(loss.detach())})
    final_loss = evaluate(model, eval_batches, derive_seed("p4-eval", seed), device)
    curve[-1]["eval_loss"] = final_loss
    final_remote = parameters_checksum(remote_parameters(model))
    if mode == "frozen_random" and initial_remote != final_remote:
        raise RuntimeError("Frozen remote parameters changed")
    scores = None
    if mode != "all_local":
        fit = capture_panel(model, samplers["probe_fit"].batches(cfg["probe_batches"]),
                            derive_seed("p4-probe-fit", seed), device)
        test = capture_panel(model, samplers["evaluation"].batches(cfg["probe_batches"]),
                             derive_seed("p4-probe-test", seed), device)
        torch.save({"fit": fit, "test": test}, out / "captures.pt")
        scores = {"property": score_properties(fit, test, seed),
                  "structure": score_structure(fit, test, seed, epochs=cfg["probe_epochs"]),
                  "qualified": False, "privacy_outcome": "undetermined",
                  "note": "Engineering scorer smoke test; no per-cell instrument qualification."}
    rows = batch * seq
    quant = None if mode == "all_local" else model.remote_path.quantizer
    metrics = {"mode": mode, "seed": seed, "cell": cell["id"],
               "cohort": "engineering", "scientific_evidence": False,
               "final_eval_loss": final_loss, "wall_seconds": time.perf_counter() - start,
               "remote_initial_sha256": initial_remote, "remote_final_sha256": final_remote,
               "property_basis": None if basis is None else basis.tolist(),
               "basis_policy": "same-model-initial-diagnostic-only" if basis is not None else None,
               "analytic_bytes_per_training_step": 0 if quant is None else
                   quant.serialized_nbytes(torch.empty(batch, seq, cfg["split"]["width"])) +
                   3 * rows * cfg["split"]["width"] * 4,
               "byte_accounting": "analytic forward packed codes plus public min/scale; other tensors fp32",
               "scores": scores, "privacy_status": "undetermined", "active_attack_evaluated": False}
    inventory = {"captured": [] if mode == "all_local" else list(CHANNELS) + ["quantizer_metadata"],
                 "evaluated": [], "unqualified_diagnostic_probes": [] if scores is None else
                    ["forward punctuation", "forward secondary properties", "forward next-byte classes"],
                 "unevaluated": UNEVALUATED,
                 "remote_state": "initial/final integrity checksums only",
                 "private_labels": "tokens/next_tokens are offline evaluator labels, not transmitted fields"}
    write_json(out / "resolved.json", {"matrix": cfg, "cell": cell, "mode": mode, "seed": seed})
    write_json(out / "curve.json", curve)
    write_json(out / "metrics.json", metrics)
    write_json(out / "transcript_inventory.json", inventory)
    return metrics


def run_matrix(config_path, label=None, validate_only=False, device=None):
    cfg = yaml.safe_load(Path(config_path).read_text())
    if device is not None:
        cfg["device"] = device
    validation = validate_matrix(cfg, execute=not validate_only)
    if validate_only:
        return validation
    device = resolve_device(cfg.get("device", "cpu"))
    verify_inherited_source()
    torch.set_num_threads(cfg.get("cpu_threads", 2))
    allow_nondeterministic = bool(cfg.get("allow_nondeterministic_kernels", False))
    torch.use_deterministic_algorithms(not allow_nondeterministic)
    corpus_path = ROOT / cfg["data"]["path"]
    if digest(corpus_path) != cfg["data"]["sha256"]:
        raise ValueError("Corpus hash mismatch")
    corpus = load_corpus(corpus_path)
    edges = [int(len(corpus) * f) for f in cfg["data"]["fractions"]]
    names = ("train", "calibration", "probe_fit", "evaluation")
    regions = {name: corpus[a:b] for name, a, b in zip(names, edges, edges[1:])}
    label = label or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if not label or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in label):
        raise ValueError("Invalid run label")
    campaign = ROOT / "outputs/runs" / label
    campaign.mkdir(parents=True, exist_ok=False)
    source = source_provenance(campaign, device=device, allow_nondeterministic=allow_nondeterministic)
    schema = json.loads((ROOT / "experiments/schemas/run_manifest.schema.json").read_text())
    results, failures = [], []
    for seed in cfg["seeds"]:
        jobs = [({"id": "all-local", "defense": {"name": "none"}}, "all_local")]
        jobs += [(cell, mode) for cell in cfg["cells"] for mode in cfg["modes"]]
        for cell, mode in jobs:
            run_id = f"p4-{label}-{cell['id']}-{mode}-s{seed}"
            out = campaign / run_id
            manifest = {"schema_version": "p4-run-0.1", "run_id": run_id,
                        "protocol_id": cfg["protocol_id"], "cohort": "engineering",
                        "scientific_evidence": False, "seed": seed, "mode": mode,
                        "cell": cell["id"], "git_commit": source["git_commit"],
                        "source_tree_sha256": source["source_tree_sha256"],
                        "source_record": str((campaign / "source.json").relative_to(ROOT)),
                        "corpus_sha256": cfg["data"]["sha256"],
                        "matrix_sha256": digest(config_path), "started_at": datetime.now(timezone.utc).isoformat()}
            try:
                metrics = run_one(cfg, cell, mode, seed, regions, out)
                results.append(metrics)
                manifest["status"] = "completed"
            except Exception as exc:
                manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                failures.append({"run_id": run_id, "error": manifest["error"]})
            manifest["artifacts"] = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)}
                                     for p in sorted(out.glob("*")) if p.is_file()]
            manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
            jsonschema.validate(manifest, schema)
            write_json(ROOT / "experiments/manifests" / f"{run_id}.json", manifest)
            print(f"{run_id}: {manifest['status']}", flush=True)
    summary = {**validation, "label": label, "device": str(device), "completed": len(results),
               "failures": failures,
               "scientific_evidence": False, "privacy_outcome": "undetermined",
               "note": "P0 engineering diagnostics only; no convergence or defense-success conclusion."}
    if allow_nondeterministic:
        summary["allow_nondeterministic_kernels"] = True
    write_json(campaign / "summary.json", summary)
    return summary
