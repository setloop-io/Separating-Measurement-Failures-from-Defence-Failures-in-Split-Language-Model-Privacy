"""Real non-scientific R1 loopback CPU validation."""

from __future__ import annotations

import copy
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch
import torch.nn.functional as F

from . import manifest as manifest_state
from . import provenance
from .artifacts import (
    validate_attack_report,
    validate_capture,
    validate_config,
    validate_curve,
    validate_inventory_file,
    validate_metrics,
    validate_remote_checkpoint,
    validate_state_dict,
    validate_training_checkpoint,
    validate_wire_audit,
    validation_record,
)
from .data import make_train_sampler
from .distributed_split import attach_remote_client, capture_measurement, evaluate, train_step
from .local_remote import LocalRemoteService
from .manifest import begin_run, write_manifest
from .modes import trainable_parameters
from .model import BackboneConfig
from .r1 import build_initialized_model, component_state_checksums, enumerate_runs, load_matrix, now
from .r1_attacks import AttackSpecification, run_attack_suite, save_attack_banks
from .r1_panels import (
    REGIONS,
    build_panel_manifest,
    panel_batch_records,
    panel_batches,
    save_panel_manifest,
)
from .split import SplitConfig
from .train import OptimConfig, build_optimizer, evaluate as evaluate_local
from .transcript import build_inventory, validate_inventory
from .utils import sha256_file, sha256_json, state_dict_checksum

CHANNELS = ("forward_values", "returned_values", "inbound_gradients", "outbound_gradients")


@contextmanager
def manifest_namespace(root: Path, directory: Path) -> Iterator[None]:
    previous_root, previous_dir = manifest_state.REPO_ROOT, manifest_state.MANIFEST_DIR
    manifest_state.REPO_ROOT, manifest_state.MANIFEST_DIR = root.resolve(), directory.resolve()
    try:
        yield
    finally:
        manifest_state.REPO_ROOT, manifest_state.MANIFEST_DIR = previous_root, previous_dir


def _optim(resolved: Mapping[str, Any]) -> OptimConfig:
    value = resolved["optim"]
    return OptimConfig(lr=value["lr"], min_lr=value["min_lr"], betas=tuple(value["betas"]),
                       weight_decay=value["weight_decay"], warmup_steps=value["warmup_steps"],
                       grad_clip=value["grad_clip"], steps=value["steps"])


def _regions(corpus: torch.Tensor, panel: Mapping[str, Any]) -> torch.Tensor:
    bounds = panel["regions"]["model_train"]
    return corpus[bounds["start"] : bounds["end"]]


def _initialization(model: torch.nn.Module, resolved: Mapping[str, Any], output_dir: Path,
                    scientific_evidence: bool = False) -> Path:
    path = output_dir / "provenance/r1_initial_state.pt"
    torch.save(model.state_dict(), path)
    split = None if resolved["split"] is None else SplitConfig(**resolved["split"])
    trusted = (model.backbone_state_dict() if hasattr(model, "backbone_state_dict")
               else model.state_dict())
    record = {"schema_version": "r1-initialization-2.0", "run_id": resolved["run_id"],
              "initialization": "r1-named-streams-v1",
              "mode": resolved["mode"], "seed": resolved["seed"],
              "pairing_block_id": resolved.get("pairing_block_id"),
              "components": component_state_checksums(model, resolved["mode"], split),
              "model_state_sha256": state_dict_checksum(model.state_dict()),
              "trusted_state_sha256": state_dict_checksum(trusted),
              "remote_initialization_key": None if split is None else {
                  "seed": resolved["seed"], "width": split.width,
                  "depth": split.remote_depth, "cell_precision_and_defense_excluded": True},
              "state_file": path.name, "state_file_sha256": sha256_file(str(path)),
              "retained_state_file_sha256": sha256_file(str(path)),
              "parameter_shapes": {name: list(value.shape) for name, value in model.state_dict().items()},
              "trusted_parameter_shapes": {
                  name: list(value.shape) for name, value in
                  (model.backbone_state_dict() if hasattr(model, "backbone_state_dict")
                   else model.state_dict()).items()
              },
              "scientific_evidence": scientific_evidence}
    record_path = output_dir / "provenance/r1_initialization.json"
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record_path


def _write_curve(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def _checkpoint_local(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                      path: Path, steps: int) -> None:
    torch.save({"schema_version": "r1-training-checkpoint-1.0", "completed_steps": steps,
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "rng": {"cpu": torch.random.get_rng_state()}}, path)


def _train_local(model: torch.nn.Module, resolved: Mapping[str, Any], sampler,
                 eval_batches) -> tuple[list[dict[str, Any]], torch.optim.Optimizer, float]:
    optim = _optim(resolved)
    device = torch.device(resolved["device"])
    optimizer = build_optimizer(model, optim)
    started = time.perf_counter()
    rows = [{"step": 0, "train_loss_smoothed": None,
             "eval_loss": evaluate_local(model, eval_batches, device),
             "wall_time": 0.0}]
    smoothed = None
    for step in range(optim.steps):
        tokens, targets = sampler.batch(step)
        tokens, targets = tokens.to(device), targets.to(device)
        logits = model(tokens)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_parameters(model), optim.grad_clip)
        for group in optimizer.param_groups:
            group["lr"] = optim.lr_at(step)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        loss_value = float(loss.detach())
        smoothed = loss_value if smoothed is None else 0.95 * smoothed + 0.05 * loss_value
        rows.append({"step": step + 1, "train_loss_smoothed": smoothed,
                     "eval_loss": evaluate_local(model, eval_batches, device),
                     "wall_time": time.perf_counter() - started})
    return rows, optimizer, time.perf_counter() - started


def _channel_payloads(records) -> dict[str, int]:
    result = {name: 0 for name in CHANNELS}
    mapping = {("FORWARD", "sent"): "forward_values",
               ("FORWARD", "received"): "returned_values",
               ("BACKWARD", "sent"): "inbound_gradients",
               ("BACKWARD", "received"): "outbound_gradients"}
    for record in records:
        channel = mapping.get((record.message_type, record.direction))
        if channel is not None:
            result[channel] += record.payload_bytes
    return result


def _metrics(resolved: Mapping[str, Any], rows: list[dict[str, Any]], wall: float,
             wire_bytes: int = 0,
             remote_checkpoint: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {"run_id": resolved["run_id"], "mode": resolved["mode"], "seed": resolved["seed"],
            "cohort": "engineering", "steps": resolved["optim"]["steps"],
            "device": resolved["device"],
            "final_eval_loss": rows[-1]["eval_loss"],
            "final_train_loss_smoothed": rows[-1]["train_loss_smoothed"],
            "wall_time_seconds": wall, "actual_wire_bytes": wire_bytes,
            "remote_device": None if remote_checkpoint is None else remote_checkpoint.get("device"),
            "remote_cuda_peak_allocated_bytes": (None if remote_checkpoint is None else
                                                   remote_checkpoint.get("cuda_peak_allocated_bytes")),
            "remote_cuda_peak_reserved_bytes": (None if remote_checkpoint is None else
                                                  remote_checkpoint.get("cuda_peak_reserved_bytes")),
            "scientific_evidence": False,
            "note": "Real two-process CPU validation only; not scientific R1 evidence."}


def _validated_manifest(resolved: dict[str, Any], output_dir: Path,
                        captured_channels: list[str]) -> Path:
    files = sorted(path for path in output_dir.rglob("*") if path.is_file())
    record = validation_record(files, output_dir)
    semantic_path = output_dir / "semantic_validation.json"
    semantic_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    files.append(semantic_path)
    return write_manifest(resolved, "completed", files, captured_channels,
                          {"semantic_validation_sha256": sha256_file(str(semantic_path)),
                           "wire_protocol": "r1-wire-1.0" if resolved["split"] else None})


def _verify_local_manifest(root: Path, output_dir: Path,
                           manifest_path: Path, resolved: Mapping[str, Any]) -> None:
    recorded = json.loads(manifest_path.read_text())
    manifest_state.validate_manifest(recorded)
    for key in ("run_id", "protocol_version", "seed", "cohort", "pairing_block_id"):
        if recorded.get(key) != resolved.get(key):
            raise ValueError(f"local validation manifest differs in {key}")
    if recorded.get("status") != "completed" or recorded.get("resolved_config_sha256") != sha256_json(resolved):
        raise ValueError("local validation manifest configuration differs")
    original_output = Path(resolved["output_dir"]).resolve()
    bound = set()
    for item in recorded["artifacts"]:
        original = Path(item["path"])
        original = (original if original.is_absolute() else root / original).resolve()
        if not original.is_relative_to(original_output):
            raise ValueError("local validation artifact escaped its recorded output")
        relative = original.relative_to(original_output)
        current = (output_dir / relative).resolve()
        if (not current.is_relative_to(output_dir.resolve()) or not current.is_file() or
                sha256_file(str(current)) != item["sha256"]):
            raise ValueError("local validation artifact is missing or changed")
        bound.add(relative.as_posix())
    required = manifest_state._required_artifacts(dict(resolved), prospective=True)
    if not required.issubset(bound):
        raise ValueError("local validation manifest omits required artifacts")
    execution_provenance = recorded.get("execution_provenance")
    if not isinstance(execution_provenance, dict):
        raise ValueError("local validation manifest has no execution provenance")
    provenance.verify_snapshot(root, output_dir, execution_provenance)
    manifest_state._verify_semantics(output_dir, dict(resolved))


def _validate_common(output_dir: Path, resolved: Mapping[str, Any]) -> None:
    validate_config(output_dir / "config.resolved.json", resolved)
    validate_curve(output_dir / "curve.jsonl", int(resolved["optim"]["steps"]))
    validate_metrics(output_dir / "metrics_summary.json", resolved)
    initialization = json.loads((output_dir / "provenance/r1_initialization.json").read_text())
    shapes = (initialization["parameter_shapes"] if resolved["split"] is None
              else initialization["trusted_parameter_shapes"])
    validate_state_dict(output_dir / "final_state.pt", shapes)
    validate_training_checkpoint(output_dir / "training_checkpoint.pt",
                                 int(resolved["optim"]["steps"]), shapes,
                                 require_remote=resolved["split"] is not None)


def _run_all_local(resolved: dict[str, Any], corpus: torch.Tensor,
                   panel: Mapping[str, Any], panel_path: Path) -> Path:
    begin_run(resolved)
    output_dir = Path(resolved["output_dir"])
    model = build_initialized_model(BackboneConfig(**resolved["backbone"]), "all_local",
                                    None, resolved["seed"], None).to(resolved["device"])
    _initialization(model, resolved, output_dir)
    sampler = make_train_sampler(_regions(corpus, panel), resolved["data"]["batch_size"],
                                 resolved["data"]["seq_len"], resolved["seed"])
    eval_batches = panel_batches(manifest_state.REPO_ROOT / resolved["data"]["path"], panel, "heldout",
                                 resolved["data"]["batch_size"])
    rows, optimizer, wall = _train_local(model, resolved, sampler, eval_batches[:2])
    _write_curve(output_dir / "curve.jsonl", rows)
    (output_dir / "metrics_summary.json").write_text(json.dumps(_metrics(resolved, rows, wall), indent=2) + "\n")
    torch.save(model.state_dict(), output_dir / "final_state.pt")
    _checkpoint_local(model, optimizer, output_dir / "training_checkpoint.pt", resolved["optim"]["steps"])
    (output_dir / "panel_binding.json").write_text(json.dumps({"path": os.path.relpath(panel_path, output_dir),
        "sha256": sha256_file(str(panel_path))}, indent=2) + "\n")
    _validate_common(output_dir, resolved)
    return _validated_manifest(resolved, output_dir, [])


def _measurement_panels(distributed, corpus_path: Path, panel: Mapping[str, Any],
                        step: int, batch_size: int = 4) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    captures, grouped = [], {}
    for region in ("attacker_auxiliary", "qualification", "heldout"):
        events = []
        for index, batch in enumerate(panel_batch_records(corpus_path, panel, region, batch_size)):
            capture = capture_measurement(
                distributed, batch["tokens"], batch["targets"], step, f"{region}-{index}",
            )
            capture.update({key: value for key, value in batch.items()
                            if key not in {"tokens", "targets"}})
            capture["checkpoint_id"] = f"step-{step}"
            capture["channel_metadata"] = {
                channel: {"shape": list(capture[channel].shape),
                          "dtype": str(capture[channel].dtype),
                          "processing": ("quantized-wire-roundtrip" if channel == "forward_values"
                                         else "full-precision-wire")}
                for channel in CHANNELS
            }
            captures.append(capture)
            events.append(capture)
        grouped[region] = events
    return captures, grouped


def _execute_distributed(root: Path, resolved: dict[str, Any], model: torch.nn.Module,
                         split: SplitConfig, sampler, eval_batches,
                         panel: Mapping[str, Any]) -> dict[str, Any]:
    output_dir = Path(resolved["output_dir"])
    started = time.perf_counter()
    with LocalRemoteService(root, output_dir / "remote-process", resolved["run_id"],
                            split.q_bits) as service:
        client = service.connect()
        distributed = attach_remote_client(
            model, client, resolved["seed"], split.grad_defense, _optim(resolved),
            remote_device=resolved["device"],
        )
        rows = [{"step": 0, "train_loss_smoothed": None,
                 "eval_loss": evaluate(distributed, eval_batches[:2], 0), "wall_time": 0.0}]
        smoothed = None
        for step in range(resolved["optim"]["steps"]):
            tokens, targets = sampler.batch(step)
            result = train_step(distributed, tokens, targets, step)
            smoothed = (result["loss"] if smoothed is None
                        else 0.95 * smoothed + 0.05 * result["loss"])
            rows.append({"step": step + 1, "train_loss_smoothed": smoothed,
                         "eval_loss": evaluate(distributed, eval_batches[:2], step + 1),
                         "wall_time": time.perf_counter() - started})
        captures, grouped = _measurement_panels(
            distributed, root / resolved["data"]["path"], panel, resolved["optim"]["steps"],
        )
        remote_checkpoint = client.checkpoint(resolved["optim"]["steps"])
        distributed.save_checkpoint(output_dir / "training_checkpoint.pt",
                                    resolved["optim"]["steps"], remote_checkpoint)
        client.close(resolved["optim"]["steps"])
        client.write_audit(output_dir / "client-wire-audit.json")
    return {"distributed": distributed, "rows": rows, "captures": captures,
            "grouped": grouped, "remote_checkpoint": remote_checkpoint,
            "remote_output_dir": service.output_dir, "client_audit": client.audit,
            "wall_time": time.perf_counter() - started}


def _run_split(root: Path, resolved: dict[str, Any], corpus: torch.Tensor,
               panel: Mapping[str, Any], panel_path: Path,
               run_attacks: bool = True) -> Path:
    begin_run(resolved)
    output_dir = Path(resolved["output_dir"])
    split = SplitConfig(**resolved["split"])
    model = build_initialized_model(BackboneConfig(**resolved["backbone"]), resolved["mode"],
                                    split, resolved["seed"], resolved["pairing_block_id"]).to(resolved["device"])
    _initialization(model, resolved, output_dir)
    sampler = make_train_sampler(_regions(corpus, panel), resolved["data"]["batch_size"],
                                 resolved["data"]["seq_len"], resolved["seed"])
    eval_batches = panel_batches(manifest_state.REPO_ROOT / resolved["data"]["path"], panel, "heldout",
                                 resolved["data"]["batch_size"])
    execution = _execute_distributed(root, resolved, model, split, sampler, eval_batches, panel)
    distributed = execution["distributed"]
    rows = execution["rows"]
    captures = execution["captures"]
    grouped = execution["grouped"]
    remote_checkpoint = execution["remote_checkpoint"]
    remote_output_dir = execution["remote_output_dir"]
    client_audit = execution["client_audit"]
    client_checkpoint = output_dir / "training_checkpoint.pt"
    _write_curve(output_dir / "curve.jsonl", rows)
    (output_dir / "metrics_summary.json").write_text(json.dumps(
        _metrics(resolved, rows, execution["wall_time"], client_audit.total_bytes,
                 remote_checkpoint), indent=2) + "\n")
    torch.save(distributed.model.state_dict(), output_dir / "final_state.pt")
    torch.save(captures, output_dir / "transcript_tensors.pt")
    channels = _channel_payloads(client_audit.records)
    inventory = build_inventory(channels, [remote_checkpoint["model_state_sha256"]], True,
                                "Actual loopback frames under r1-wire-1.0; payload and framing bytes retained separately.")
    validate_inventory(inventory)
    (output_dir / "transcript_inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    (output_dir / "transcript_bytes_per_step.json").write_text(json.dumps(client_audit.to_dict(), indent=2) + "\n")
    (output_dir / "panel_binding.json").write_text(json.dumps({"path": os.path.relpath(panel_path, output_dir),
        "sha256": sha256_file(str(panel_path))}, indent=2) + "\n")
    if resolved["mode"] == "trained" and run_attacks:
        bindings = {"panel": panel["panel_sha256"],
                    "client_checkpoint": sha256_file(str(client_checkpoint)),
                    "remote_checkpoint": sha256_file(str(remote_output_dir / remote_checkpoint["path"])),
                    "capture": sha256_file(str(output_dir / "transcript_tensors.pt"))}
        specification = AttackSpecification()
        defense = "none" if split.grad_defense is None else "gradient-defense"
        bank_path = output_dir / "attack_banks.pt"
        bank_digests = save_attack_banks(
            grouped["attacker_auxiliary"], specification, resolved["seed"], defense, bank_path,
        )
        report = run_attack_suite(grouped["attacker_auxiliary"], grouped["qualification"],
                                  grouped["heldout"], specification, resolved["seed"],
                                  defense, bindings, sha256_file(str(bank_path)))
        if {key: value["attack_bank_sha256"] for key, value in report["results"].items()} != bank_digests:
            raise ValueError("retained attack banks differ from the scored attack suite")
        attack_path = output_dir / "attack_suite.json"
        attack_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        validate_attack_report(attack_path)
    _validate_common(output_dir, resolved)
    validate_capture(output_dir / "transcript_tensors.pt", CHANNELS, split.width,
                     resolved["data"]["seq_len"])
    validate_inventory_file(output_dir / "transcript_inventory.json")
    validate_wire_audit(output_dir / "client-wire-audit.json")
    validate_wire_audit(remote_output_dir / "remote-wire-audit.json")
    validate_remote_checkpoint(remote_output_dir / remote_checkpoint["path"], resolved["optim"]["steps"])
    return _validated_manifest(resolved, output_dir, list(CHANNELS))


def run_distributed_engineering_cell(root: Path, resolved: dict[str, Any],
                                     panel: Mapping[str, Any], panel_path: Path) -> Path:
    corpus_path = root / resolved["data"]["path"]
    corpus = torch.tensor(list(corpus_path.read_bytes()), dtype=torch.uint8)
    if resolved["mode"] == "all_local":
        return _run_all_local(resolved, corpus, panel, panel_path)
    return _run_split(root, resolved, corpus, panel, panel_path, run_attacks=False)


def run_local_validation(root: Path, matrix_path: Path, output_root: Path,
                         execution_id: str) -> dict[str, Any]:
    root, output_root = root.resolve(), output_root.resolve()
    if output_root.exists():
        raise ValueError("local validation output already exists")
    matrix = load_matrix(matrix_path)
    runs = enumerate_runs(matrix, "local_validation", output_root / "runs")
    planned_runs = copy.deepcopy(runs)
    output_root.mkdir(parents=True)
    corpus_path = root / matrix["local_validation"]["base"]["data"]["path"]
    counts = {name: matrix["local_validation"]["panel"]["windows_per_region"] for name in REGIONS}
    panel = build_panel_manifest(
        corpus_path, matrix["local_validation"]["base"]["data"]["seq_len"], counts,
        matrix["local_validation"]["panel"]["selection_seed"],
        corpus_label=matrix["local_validation"]["base"]["data"]["path"],
    )
    panel_path = output_root / "panel_manifest.json"
    save_panel_manifest(panel_path, panel)
    corpus = torch.tensor(list(corpus_path.read_bytes()), dtype=torch.uint8)
    manifests = []
    records = output_root / "execution_records"
    records.mkdir()
    with manifest_namespace(root, output_root / "manifests"):
        for resolved in runs:
            resolved["run_id"] += f"--e{execution_id}"
            resolved["output_dir"] = str(output_root / "runs" / resolved["run_id"])
            resolved["data"]["path"] = str(corpus_path.relative_to(root))
            resolved["r1"].update({"stage": "local_validation", "execution_id": execution_id,
                                   "panel_sha256": panel["panel_sha256"],
                                   "transport": "loopback-tcp-r1-wire-1.0",
                                   "semantic_validation": True,
                                   "attack_validation": True})
            run_id = resolved["run_id"]
            start_path = records / f"{run_id}.start.json"
            start_path.write_text(json.dumps({"run_id": run_id, "started_at_utc": now(),
                "config_sha256": sha256_json(resolved)}, indent=2) + "\n")
            try:
                path = (_run_all_local(resolved, corpus, panel, panel_path) if resolved["mode"] == "all_local"
                        else _run_split(root, resolved, corpus, panel, panel_path))
                end = {"run_id": run_id, "status": "completed", "finished_at_utc": now(),
                       "start_sha256": sha256_file(str(start_path)),
                       "manifest": str(path.relative_to(output_root)),
                       "manifest_sha256": sha256_file(str(path))}
                (records / f"{run_id}.end.json").write_text(json.dumps(end, indent=2) + "\n")
                manifests.append(str(path))
            except BaseException as exc:
                failure = {"run_id": run_id, "status": "operational_failure",
                           "failed_at_utc": now(), "error_type": type(exc).__name__,
                           "error": str(exc), "start_sha256": sha256_file(str(start_path))}
                (records / f"{run_id}.failure.json").write_text(json.dumps(failure, indent=2) + "\n")
                raise
    report = analyze_local_validation(root, output_root, planned_runs, execution_id)
    return {"output_root": str(output_root), "run_count": len(runs), "manifests": manifests,
            "analysis": report, "scientific_evidence": False}


def analyze_local_validation(root: Path, output_root: Path,
                             planned_runs: list[dict[str, Any]],
                             execution_id: str) -> dict[str, Any]:
    rows, failures, interrupted, not_started = [], [], [], []
    records = output_root / "execution_records"
    with manifest_namespace(root, output_root / "manifests"):
        for base in planned_runs:
            run_id = f"{base['run_id']}--e{execution_id}"
            start = records / f"{run_id}.start.json"
            end = records / f"{run_id}.end.json"
            failure = records / f"{run_id}.failure.json"
            if failure.is_file():
                failures.append(json.loads(failure.read_text()))
                continue
            if not start.is_file():
                not_started.append(run_id)
                continue
            if not end.is_file():
                interrupted.append(run_id)
                continue
            output_dir = output_root / "runs" / run_id
            resolved = json.loads((output_dir / "config.resolved.json").read_text())
            completion = json.loads(end.read_text())
            if completion.get("start_sha256") != sha256_file(str(start)):
                raise ValueError("local validation start/end binding differs")
            manifest_value = Path(completion["manifest"])
            manifest_path = (manifest_value if manifest_value.is_absolute()
                             else output_root / manifest_value).resolve()
            if completion.get("manifest_sha256") != sha256_file(str(manifest_path)):
                raise ValueError("local validation manifest binding differs")
            _verify_local_manifest(root, output_dir, manifest_path, resolved)
            _validate_common(output_dir, resolved)
            initialization = json.loads((output_dir / "provenance/r1_initialization.json").read_text())
            metrics = json.loads((output_dir / "metrics_summary.json").read_text())
            rows.append({"run_id": run_id, "cell_id": base["r1"]["cell_id"], "seed": base["seed"],
                         "mode": base["mode"], "loss": metrics["final_eval_loss"],
                         "components": initialization["components"],
                         "actual_wire_bytes": metrics["actual_wire_bytes"]})
    complete = not failures and not interrupted and not not_started and len(rows) == len(planned_runs)
    if complete:
        _verify_initialization(rows)
    contrasts = _contrasts(rows)
    report = {"schema_version": "r1-local-validation-analysis-1.0", "run_count": len(rows),
              "planned_count": len(planned_runs), "failures": failures,
              "interrupted_runs": interrupted, "not_started_runs": not_started,
              "contrasts": contrasts, "privacy": "ENGINEERING_ONLY",
              "scientific_evidence": False}
    (output_root / "analysis.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _contrasts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(row["cell_id"], row["seed"], row["mode"]): row for row in rows}
    results = []
    for row in rows:
        if row["mode"] != "trained":
            continue
        local = by_key.get(("all-local", row["seed"], "all_local"))
        frozen = by_key.get((row["cell_id"], row["seed"], "frozen_random"))
        removed = by_key.get((row["cell_id"], row["seed"], "removed_remote"))
        if local is None or frozen is None or removed is None:
            continue
        results.append({"cell_id": row["cell_id"], "seed": row["seed"],
                        "replacement_penalty": row["loss"] - local["loss"],
                        "gain_against_frozen": frozen["loss"] - row["loss"],
                        "gain_against_removed": removed["loss"] - row["loss"]})
    return results


def _verify_initialization(rows: list[dict[str, Any]]) -> None:
    by_key = {(row["cell_id"], row["seed"], row["mode"]): row for row in rows}
    for row in rows:
        if row["mode"] != "trained":
            continue
        frozen = by_key[(row["cell_id"], row["seed"], "frozen_random")]
        removed = by_key[(row["cell_id"], row["seed"], "removed_remote")]
        local = by_key[("all-local", row["seed"], "all_local")]
        if row["components"] != frozen["components"]:
            raise ValueError("trained/frozen initialization differs")
        shared = {key: value for key, value in row["components"].items() if not key.startswith("remote/")}
        removed_shared = {key: value for key, value in removed["components"].items() if not key.startswith("remote/")}
        if shared != removed_shared:
            raise ValueError("trained/removed trusted initialization differs")
        for key, value in local["components"].items():
            if key in row["components"] and row["components"][key] != value:
                raise ValueError("split/all-local common initialization differs")
