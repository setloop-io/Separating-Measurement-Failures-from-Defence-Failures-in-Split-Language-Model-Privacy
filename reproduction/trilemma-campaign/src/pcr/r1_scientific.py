"""Admitted R1 scientific stage machinery: calibration and exploratory execution.

Nothing in this module executes a scientific run by itself. ``freeze_scientific``
binds the protocol document, attack specification, budget, environment review and
matrix state into one single-use admission; ``execute_scientific`` consumes exactly
one such admission. The engineering substrate (bundle, panels, wire protocol,
attack suite, manifests, semantic validators) is reused unchanged; records produced
inside an admitted execution carry ``scientific_evidence: true`` only because the
admitted executor threads the marker through.
"""

from __future__ import annotations

import json
import math
import os
import platform
import re
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from jsonschema import validate

from . import manifest as manifest_state
from .artifacts import (
    ArtifactValidationError,
    validate_attack_report,
    validate_capture,
    validate_capture_pair,
    validate_inventory_file,
    validate_remote_checkpoint,
    validate_wire_audit,
    validation_record,
)
from .data import make_train_sampler
from .distributed_split import (
    DistributedModel,
    attach_remote_client,
    capture_measurement,
    evaluate,
    train_step,
)
from .local_remote import LocalRemoteService
from .manifest import begin_run, write_manifest
from .model import BackboneConfig
from .r1 import (
    build_initialized_model,
    claim_admission,
    enumerate_runs,
    environment_record,
    inside,
    load_matrix,
    now,
    output_size,
    read_json,
    safe_name,
    stage_spec,
    verify_bundle,
    wall_clock_guard,
    write_new,
)
from .r1_attacks import AttackSpecification, run_attack_suite, save_attack_banks
from .r1_local_validation import (
    CHANNELS,
    _channel_payloads,
    _initialization,
    _measurement_panels,
    _optim,
    _regions,
    _train_local,
    _validate_common,
    _write_curve,
    manifest_namespace,
)
from .r1_panels import REGIONS, build_panel_manifest, panel_batches, save_panel_manifest
from .split import SplitConfig
from .transcript import build_inventory, validate_inventory
from .utils import derive_seed, sha256_bytes, sha256_file, sha256_json, state_dict_checksum

SCIENTIFIC_STAGES = ("calibration", "exploratory")
SCHEMA_ROOT = Path(__file__).resolve().parents[2] / "experiments/schemas"
ATTACK_SPEC_SCHEMA_PATH = SCHEMA_ROOT / "r1_attack_spec.schema.json"
ADMISSION_SCHEMA_PATH = SCHEMA_ROOT / "r1_scientific_admission.schema.json"


@dataclass(frozen=True)
class TrainingCaptureSchedule:
    """Sparse training-frame grid: step 0, every cadence and the final step."""

    steps: int
    every_steps: int
    max_frames: int
    max_examples: int

    def __post_init__(self) -> None:
        for name in ("steps", "every_steps", "max_frames", "max_examples"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"capture schedule {name} must be a positive integer")

    def grid(self) -> tuple[int, ...]:
        points = {0, self.steps, *range(0, self.steps + 1, self.every_steps)}
        ordered = tuple(sorted(points))
        if len(ordered) > self.max_frames:
            raise ValueError(f"capture schedule needs {len(ordered)} frames; the budget allows {self.max_frames}")
        return ordered


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    return sha256_bytes(str(tuple(tensor.shape)).encode() + str(tensor.dtype).encode()
                        + tensor.numpy().tobytes())


def capture_training_frame(distributed: DistributedModel, tokens: torch.Tensor,
                           targets: torch.Tensor, step: int,
                           batch_identity: Mapping[str, Any]) -> dict[str, Any]:
    """Measurement-only capture of one training-stream batch at a committed step.

    Runs under the same measurement semantics as ``capture_measurement``: model
    state is checksummed before and after, the inbound and remote outbound defense
    streams are restored, and no optimizer interaction occurs.
    """
    capture = capture_measurement(distributed, tokens, targets, step, f"training-{step}",
                                  capture_kind="training_frame")
    capture["checkpoint_id"] = f"step-{step}"
    capture["sampler_batch"] = dict(batch_identity)
    capture["channel_metadata"] = {
        channel: {"shape": list(capture[channel].shape),
                  "dtype": str(capture[channel].dtype),
                  "processing": ("quantized-wire-roundtrip" if channel == "forward_values"
                                 else "full-precision-wire")}
        for channel in CHANNELS
    }
    return capture


def validate_training_frames(path: Path, *, width: int, seq_len: int,
                             expected_steps: Sequence[int], max_examples: int) -> None:
    try:
        events = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ArtifactValidationError(f"training frame capture is not loadable: {exc}") from exc
    if not isinstance(events, list) or not events:
        raise ArtifactValidationError("training frame capture must contain at least one event")
    steps = []
    for index, event in enumerate(events):
        if not isinstance(event, dict) or type(event.get("step")) is not int or event["step"] < 0:
            raise ArtifactValidationError(f"training frame {index} has no nonnegative integer step")
        steps.append(event["step"])
        if (event.get("label") != f"training-{event['step']}" or
                event.get("checkpoint_id") != f"step-{event['step']}"):
            raise ArtifactValidationError("training frame identity is invalid")
        tokens, targets = event.get("tokens"), event.get("targets")
        for name, value in (("tokens", tokens), ("targets", targets)):
            if not isinstance(value, torch.Tensor) or value.dtype != torch.long or value.ndim != 2:
                raise ArtifactValidationError(f"training frame {index} {name} is invalid")
        if tokens.shape != targets.shape or tokens.shape[1] != seq_len or not 0 < tokens.shape[0] <= max_examples:
            raise ArtifactValidationError("training frame batch shape exceeds its declared budget")
        batch = tokens.shape[0]
        identity = event.get("sampler_batch")
        if (not isinstance(identity, dict) or identity.get("stream") != "train" or
                type(identity.get("seed")) is not int or identity.get("step") != event["step"] or
                type(identity.get("batch_size")) is not int or identity["batch_size"] < batch or
                identity.get("examples") != batch or identity.get("seq_len") != seq_len or
                identity.get("tokens_sha256") != _tensor_sha256(tokens) or
                identity.get("targets_sha256") != _tensor_sha256(targets)):
            raise ArtifactValidationError("training frame sampler batch identity is invalid")
        metadata = event.get("channel_metadata")
        if not isinstance(metadata, dict) or not set(CHANNELS).issubset(metadata):
            raise ArtifactValidationError("training frame channel metadata is incomplete")
        for channel in CHANNELS:
            value = event.get(channel)
            if (not isinstance(value, torch.Tensor) or not torch.isfinite(value).all() or
                    tuple(value.shape) != (batch, seq_len, width)):
                raise ArtifactValidationError(f"training frame channel {channel} is invalid")
            item = metadata[channel]
            if (not isinstance(item, dict) or item.get("shape") != list(value.shape) or
                    item.get("dtype") != str(value.dtype) or
                    item.get("processing") not in {"quantized-wire-roundtrip", "full-precision-wire"}):
                raise ArtifactValidationError(f"training frame channel metadata differs for {channel}")
    if steps != sorted(steps) or steps != [int(step) for step in expected_steps]:
        raise ArtifactValidationError("training frame steps differ from the bound capture schedule")


def _normalize_checkpoint_rng(path: Path, device: str) -> None:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict) or not isinstance(value.get("rng"), dict):
        raise ValueError("training checkpoint lacks an RNG section")
    if device == "cuda":
        value["rng"]["cuda"] = torch.cuda.get_rng_state()
    else:
        value["rng"].pop("cuda", None)
    torch.save(value, path)


def _checkpoint_scientific_local(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                                 path: Path, steps: int, device: str) -> None:
    torch.save({"schema_version": "r1-training-checkpoint-1.0", "completed_steps": steps,
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "rng": {"cpu": torch.random.get_rng_state()}}, path)
    _normalize_checkpoint_rng(path, device)


def _save_remote_snapshot(state: Mapping[str, torch.Tensor], path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: tensor.detach().cpu() for name, tensor in state.items()}
    torch.save(payload, path)
    return {"path": path.name, "sha256": sha256_file(str(path)),
            "state_dict_checksum": state_dict_checksum(payload),
            "parameter_count": sum(int(tensor.numel()) for tensor in payload.values())}


def _scientific_metrics(resolved: Mapping[str, Any], rows: list[dict[str, Any]], wall: float,
                        *, wire_bytes: int = 0,
                        remote_checkpoint: Mapping[str, Any] | None = None,
                        scientific_evidence: bool) -> dict[str, Any]:
    return {"run_id": resolved["run_id"], "mode": resolved["mode"], "seed": resolved["seed"],
            "cohort": resolved["cohort"], "steps": resolved["optim"]["steps"],
            "device": resolved["device"],
            "final_eval_loss": rows[-1]["eval_loss"],
            "final_train_loss_smoothed": rows[-1]["train_loss_smoothed"],
            "wall_time_seconds": wall, "actual_wire_bytes": wire_bytes,
            "remote_device": None if remote_checkpoint is None else remote_checkpoint.get("device"),
            "remote_cuda_peak_allocated_bytes": (None if remote_checkpoint is None else
                                                 remote_checkpoint.get("cuda_peak_allocated_bytes")),
            "remote_cuda_peak_reserved_bytes": (None if remote_checkpoint is None else
                                                remote_checkpoint.get("cuda_peak_reserved_bytes")),
            "scientific_evidence": scientific_evidence,
            "note": "Admitted R1 scientific stage execution under a local pre-execution binding; "
                    "not an independently witnessed preregistration."}


def _frame_at(distributed: DistributedModel, sampler, step: int,
              schedule: TrainingCaptureSchedule) -> dict[str, Any]:
    tokens, targets = sampler.batch(step)
    tokens, targets = tokens[: schedule.max_examples], targets[: schedule.max_examples]
    identity = {"stream": sampler.label, "seed": sampler.seed, "step": step,
                "batch_size": sampler.batch_size, "examples": int(tokens.shape[0]),
                "seq_len": sampler.seq_len,
                "tokens_sha256": _tensor_sha256(tokens),
                "targets_sha256": _tensor_sha256(targets)}
    return capture_training_frame(distributed, tokens, targets, step, identity)


def _execute_scientific_distributed(root: Path, resolved: dict[str, Any], model: torch.nn.Module,
                                    split: SplitConfig, sampler, eval_batches,
                                    panel: Mapping[str, Any], schedule: TrainingCaptureSchedule,
                                    panel_batch_size: int) -> dict[str, Any]:
    output_dir = Path(resolved["output_dir"])
    steps = int(resolved["optim"]["steps"])
    started = time.perf_counter()
    frames: list[dict[str, Any]] = []
    with LocalRemoteService(Path(root), output_dir / "remote-process", resolved["run_id"],
                            split.q_bits) as service:
        client = service.connect()
        distributed = attach_remote_client(
            model, client, resolved["seed"], split.grad_defense, _optim(resolved),
            remote_device=resolved["device"],
        )
        rows = [{"step": 0, "train_loss_smoothed": None,
                 "eval_loss": evaluate(distributed, eval_batches[:2], 0), "wall_time": 0.0}]
        grid = set(schedule.grid())
        if 0 in grid:
            frames.append(_frame_at(distributed, sampler, 0, schedule))
        smoothed = None
        for step in range(steps):
            tokens, targets = sampler.batch(step)
            result = train_step(distributed, tokens, targets, step)
            smoothed = (result["loss"] if smoothed is None
                        else 0.95 * smoothed + 0.05 * result["loss"])
            rows.append({"step": step + 1, "train_loss_smoothed": smoothed,
                         "eval_loss": evaluate(distributed, eval_batches[:2], step + 1),
                         "wall_time": time.perf_counter() - started})
            completed = step + 1
            if completed in grid:
                frames.append(_frame_at(distributed, sampler, completed, schedule))
        captures, grouped = _measurement_panels(
            distributed, Path(root) / resolved["data"]["path"], panel, steps, panel_batch_size,
        )
        remote_checkpoint = client.checkpoint(steps)
        distributed.save_checkpoint(output_dir / "training_checkpoint.pt", steps, remote_checkpoint)
        _normalize_checkpoint_rng(output_dir / "training_checkpoint.pt", resolved["device"])
        client.close(steps)
        client.write_audit(output_dir / "client-wire-audit.json")
    return {"distributed": distributed, "rows": rows, "frames": frames, "captures": captures,
            "grouped": grouped, "remote_checkpoint": remote_checkpoint,
            "remote_output_dir": service.output_dir, "client_audit": client.audit,
            "wall_time": time.perf_counter() - started}


def _attack_spec_binding(resolved: Mapping[str, Any], attack_spec: AttackSpecification,
                         frozen_spec_path: Path | None, root: Path,
                         *, scientific_evidence: bool) -> dict[str, Any]:
    document = None
    if frozen_spec_path is not None:
        path = Path(frozen_spec_path)
        document = {"sha256": sha256_file(str(path))}
        try:
            document["path"] = str(path.resolve().relative_to(Path(root).resolve()))
        except ValueError:
            document["path"] = str(path)
    return {"schema_version": "r1-attack-spec-binding-1.0", "run_id": resolved["run_id"],
            "specification_sha256": sha256_json(_specification_fields(attack_spec)),
            "status": "draft" if frozen_spec_path is None else "frozen",
            "frozen_document": document, "scientific_evidence": scientific_evidence,
            "note": "Calibration binds the draft specification; exploratory binds the frozen admitted document."}


def _validated_scientific_manifest(resolved: dict[str, Any], output_dir: Path,
                                   captured_channels: list[str],
                                   *, scientific_evidence: bool) -> Path:
    files = sorted(path for path in output_dir.rglob("*") if path.is_file())
    record = validation_record(files, output_dir)
    # semantic_validation.json keeps scientific_evidence: false; the existing
    # validate_semantic_record pins that value and is reused unchanged.
    semantic_path = output_dir / "semantic_validation.json"
    semantic_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    files.append(semantic_path)
    return write_manifest(resolved, "completed", files, captured_channels,
                          {"semantic_validation_sha256": sha256_file(str(semantic_path)),
                           "wire_protocol": "r1-wire-1.0" if resolved["split"] else None,
                           "scientific_evidence": scientific_evidence})


def _run_scientific_all_local(root: Path, resolved: dict[str, Any], corpus: torch.Tensor,
                              panel: Mapping[str, Any], panel_path: Path,
                              *, scientific_evidence: bool) -> Path:
    begin_run(resolved)
    output_dir = Path(resolved["output_dir"])
    model = build_initialized_model(BackboneConfig(**resolved["backbone"]), "all_local",
                                    None, resolved["seed"], None).to(resolved["device"])
    _initialization(model, resolved, output_dir, scientific_evidence=scientific_evidence)
    sampler = make_train_sampler(_regions(corpus, panel), resolved["data"]["batch_size"],
                                 resolved["data"]["seq_len"], resolved["seed"])
    eval_batches = panel_batches(Path(root) / resolved["data"]["path"], panel, "heldout",
                                 resolved["data"]["batch_size"])
    rows, optimizer, wall = _train_local(model, resolved, sampler, eval_batches[:2])
    _write_curve(output_dir / "curve.jsonl", rows)
    (output_dir / "metrics_summary.json").write_text(json.dumps(
        _scientific_metrics(resolved, rows, wall, scientific_evidence=scientific_evidence),
        indent=2) + "\n")
    torch.save(model.state_dict(), output_dir / "final_state.pt")
    _checkpoint_scientific_local(model, optimizer, output_dir / "training_checkpoint.pt",
                                 int(resolved["optim"]["steps"]), resolved["device"])
    (output_dir / "panel_binding.json").write_text(json.dumps(
        {"path": os.path.relpath(panel_path, output_dir),
         "sha256": sha256_file(str(panel_path))}, indent=2) + "\n")
    _validate_common(output_dir, resolved)
    return _validated_scientific_manifest(resolved, output_dir, [],
                                          scientific_evidence=scientific_evidence)


def _run_scientific_split(root: Path, resolved: dict[str, Any], corpus: torch.Tensor,
                          panel: Mapping[str, Any], panel_path: Path,
                          attack_spec: AttackSpecification, frozen_spec_path: Path | None,
                          *, scientific_evidence: bool) -> Path:
    begin_run(resolved)
    output_dir = Path(resolved["output_dir"])
    split = SplitConfig(**resolved["split"])
    steps = int(resolved["optim"]["steps"])
    model = build_initialized_model(BackboneConfig(**resolved["backbone"]), resolved["mode"],
                                    split, resolved["seed"],
                                    resolved["pairing_block_id"]).to(resolved["device"])
    _initialization(model, resolved, output_dir, scientific_evidence=scientific_evidence)
    snapshots_dir = output_dir / "remote_snapshots"
    initial_snapshot = _save_remote_snapshot(model.remote_path.remote.state_dict(),
                                             snapshots_dir / "remote-initial.pt")
    sampler = make_train_sampler(_regions(corpus, panel), resolved["data"]["batch_size"],
                                 resolved["data"]["seq_len"], resolved["seed"])
    eval_batches = panel_batches(Path(root) / resolved["data"]["path"], panel, "heldout",
                                 resolved["data"]["batch_size"])
    capture_cfg = resolved.get("r1", {}).get("training_capture")
    if not isinstance(capture_cfg, dict) or type(resolved["r1"].get("attack_panel_batch_size")) is not int:
        raise ValueError("resolved run lacks the bound scientific capture settings")
    schedule = TrainingCaptureSchedule(steps, int(capture_cfg["every_steps"]),
                                       int(capture_cfg["max_frames"]), int(capture_cfg["max_examples"]))
    execution = _execute_scientific_distributed(root, resolved, model, split, sampler,
                                                eval_batches, panel, schedule,
                                                int(resolved["r1"]["attack_panel_batch_size"]))
    distributed = execution["distributed"]
    rows = execution["rows"]
    frames = execution["frames"]
    captures = execution["captures"]
    grouped = execution["grouped"]
    remote_checkpoint = execution["remote_checkpoint"]
    remote_output_dir = execution["remote_output_dir"]
    client_audit = execution["client_audit"]
    remote_checkpoint_path = remote_output_dir / remote_checkpoint["path"]
    client_checkpoint = output_dir / "training_checkpoint.pt"
    _write_curve(output_dir / "curve.jsonl", rows)
    (output_dir / "metrics_summary.json").write_text(json.dumps(
        _scientific_metrics(resolved, rows, execution["wall_time"],
                            wire_bytes=client_audit.total_bytes,
                            remote_checkpoint=remote_checkpoint,
                            scientific_evidence=scientific_evidence), indent=2) + "\n")
    torch.save(distributed.model.state_dict(), output_dir / "final_state.pt")
    torch.save(captures, output_dir / "transcript_tensors.pt")
    frames_path = output_dir / "training_captures.pt"
    torch.save(frames, frames_path)
    validate_training_frames(frames_path, width=split.width,
                             seq_len=int(resolved["data"]["seq_len"]),
                             expected_steps=schedule.grid(), max_examples=schedule.max_examples)
    if not remote_checkpoint.get("training_captures_path"):
        raise ValueError("remote side did not retain the training frame captures")
    remote_training = remote_output_dir / remote_checkpoint["training_captures_path"]
    validate_capture_pair(frames_path, remote_training, CHANNELS)
    frames_index = {"schema_version": "r1-training-captures-1.0", "run_id": resolved["run_id"],
                    "scientific_evidence": scientific_evidence,
                    "capture_file": frames_path.name,
                    "capture_sha256": sha256_file(str(frames_path)),
                    "schedule": {"steps": schedule.steps, "every_steps": schedule.every_steps,
                                 "max_frames": schedule.max_frames,
                                 "max_examples": schedule.max_examples,
                                 "grid": list(schedule.grid())},
                    "frames": [{"step": frame["step"], "checkpoint_id": frame["checkpoint_id"],
                                "sampler_batch": frame["sampler_batch"]} for frame in frames],
                    "remote_captures": {"path": str(remote_training.relative_to(output_dir)),
                                        "sha256": sha256_file(str(remote_training))},
                    "note": "Training-stream measurement frames at the scheduled steps; capture is measurement-only."}
    (output_dir / "training_captures.json").write_text(
        json.dumps(frames_index, indent=2, sort_keys=True) + "\n")
    if resolved.get("cohort") == "calibration":
        # _required_artifacts binds calibration_pairs.pt for split calibration
        # cohorts; retain the scheduled raw pre-quantization latent/token pairs.
        torch.save([{"step": frame["step"],
                     "latents": frame["forward_values_pre_transport"].to(torch.float32),
                     "tokens": frame["tokens"]} for frame in frames],
                   output_dir / "calibration_pairs.pt")
    channels = _channel_payloads(client_audit.records)
    inventory = build_inventory(channels, [remote_checkpoint["model_state_sha256"]], True,
                                "Actual loopback frames under r1-wire-1.0; payload and framing bytes retained separately.")
    validate_inventory(inventory)
    (output_dir / "transcript_inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    (output_dir / "transcript_bytes_per_step.json").write_text(json.dumps(client_audit.to_dict(), indent=2) + "\n")
    (output_dir / "panel_binding.json").write_text(json.dumps(
        {"path": os.path.relpath(panel_path, output_dir),
         "sha256": sha256_file(str(panel_path))}, indent=2) + "\n")
    remote_final_state = torch.load(remote_checkpoint_path, map_location="cpu", weights_only=True)["model"]
    final_snapshot = _save_remote_snapshot(remote_final_state, snapshots_dir / "remote-final.pt")
    initialization_record = json.loads((output_dir / "provenance/r1_initialization.json").read_text())
    remote_component = f"remote/{split.width}:{split.remote_depth}"
    if initial_snapshot["state_dict_checksum"] != initialization_record["components"][remote_component]:
        raise ValueError("initial remote snapshot differs from the initialization record")
    if final_snapshot["state_dict_checksum"] != remote_checkpoint["model_state_sha256"]:
        raise ValueError("final remote snapshot differs from the remote checkpoint")
    snapshot_record = {"schema_version": "r1-remote-snapshots-1.0", "run_id": resolved["run_id"],
                       "scientific_evidence": scientific_evidence,
                       "remote_component": remote_component,
                       "initial": {**initial_snapshot,
                                   "initialization_component_sha256":
                                       initialization_record["components"][remote_component]},
                       "final": {**final_snapshot,
                                 "remote_checkpoint_id": remote_checkpoint["checkpoint_id"],
                                 "remote_checkpoint_sha256": sha256_file(str(remote_checkpoint_path))},
                       "note": "Remote module state retained at initialization (before step 0) "
                               "and at the final committed step only."}
    (snapshots_dir / "remote-snapshots.json").write_text(
        json.dumps(snapshot_record, indent=2, sort_keys=True) + "\n")
    if resolved["mode"] == "trained":
        bindings = {"panel": panel["panel_sha256"],
                    "client_checkpoint": sha256_file(str(client_checkpoint)),
                    "remote_checkpoint": sha256_file(str(remote_checkpoint_path)),
                    "capture": sha256_file(str(output_dir / "transcript_tensors.pt"))}
        defense = "none" if split.grad_defense is None else "gradient-defense"
        bank_path = output_dir / "attack_banks.pt"
        bank_digests = save_attack_banks(
            grouped["attacker_auxiliary"], attack_spec, resolved["seed"], defense, bank_path,
        )
        report = run_attack_suite(grouped["attacker_auxiliary"], grouped["qualification"],
                                  grouped["heldout"], attack_spec, resolved["seed"],
                                  defense, bindings, sha256_file(str(bank_path)),
                                  scientific_evidence=scientific_evidence)
        if {key: value["attack_bank_sha256"] for key, value in report["results"].items()} != bank_digests:
            raise ValueError("retained attack banks differ from the scored attack suite")
        attack_path = output_dir / "attack_suite.json"
        attack_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        validate_attack_report(attack_path)
        (output_dir / "attack_spec_binding.json").write_text(json.dumps(
            _attack_spec_binding(resolved, attack_spec, frozen_spec_path, Path(root),
                                 scientific_evidence=scientific_evidence),
            indent=2, sort_keys=True) + "\n")
    _validate_common(output_dir, resolved)
    validate_capture(output_dir / "transcript_tensors.pt", CHANNELS, split.width,
                     int(resolved["data"]["seq_len"]))
    validate_inventory_file(output_dir / "transcript_inventory.json")
    validate_wire_audit(output_dir / "client-wire-audit.json")
    validate_wire_audit(remote_output_dir / "remote-wire-audit.json")
    validate_remote_checkpoint(remote_checkpoint_path, steps)
    return _validated_scientific_manifest(resolved, output_dir, list(CHANNELS),
                                          scientific_evidence=scientific_evidence)


def run_scientific_cell(root: Path, resolved: dict[str, Any], panel: Mapping[str, Any],
                        panel_path: Path, attack_spec: AttackSpecification,
                        frozen_spec_path: Path | None = None, *,
                        scientific_evidence: bool = False) -> Path:
    """Execute one admitted scientific cell; the marker is threaded from the admitted executor."""
    if not isinstance(attack_spec, AttackSpecification):
        raise ValueError("a bound attack specification is required")
    if scientific_evidence is not bool(resolved.get("r1", {}).get("scientific_evidence")):
        raise ValueError("resolved run marker differs from the admitted executor flag")
    if frozen_spec_path is not None:
        document = read_json(Path(frozen_spec_path))
        if document.get("status") != "frozen" or _specification_from_document(document) != attack_spec:
            raise ValueError("bound frozen attack specification differs from the admitted document")
    corpus_path = Path(root) / resolved["data"]["path"]
    corpus = torch.tensor(list(corpus_path.read_bytes()), dtype=torch.uint8)
    if resolved["mode"] == "all_local":
        return _run_scientific_all_local(root, resolved, corpus, panel, panel_path,
                                         scientific_evidence=scientific_evidence)
    return _run_scientific_split(root, resolved, corpus, panel, panel_path, attack_spec,
                                 frozen_spec_path, scientific_evidence=scientific_evidence)


def scientific_stage_enabled(matrix: Mapping[str, Any], stage: str) -> None:
    if stage not in SCIENTIFIC_STAGES:
        raise ValueError(f"unsupported scientific stage: {stage}")
    proposed = matrix.get("proposed_scientific_matrix", {})
    enabled = (proposed.get("admission_enabled") if stage == "exploratory"
               else proposed.get("calibration", {}).get("admission_enabled"))
    if enabled is not True:
        raise ValueError(f"the matrix does not enable {stage} scientific admission")


def _stage_execution_device(spec: Mapping[str, Any]) -> tuple[str, str]:
    declared = spec["base"]["device"]
    if declared != "cuda":
        raise ValueError("R1 scientific stages must declare the cuda workload")
    effective = spec.get("execution_device", declared)
    if effective not in ("cpu", "cuda"):
        raise ValueError("scientific execution_device must be cpu or cuda")
    return declared, effective


def _scientific_runs(matrix: dict, stage: str, output_root: Path) -> list[dict]:
    spec = stage_spec(matrix, stage)
    declared, effective = _stage_execution_device(spec)
    runs = enumerate_runs(matrix, stage, output_root)
    if effective != declared:
        for resolved in runs:
            resolved["device"] = effective
    return runs


def scientific_limits(matrix: Mapping[str, Any], stage: str) -> Mapping[str, Any]:
    if stage not in SCIENTIFIC_STAGES:
        raise ValueError(f"unsupported scientific stage: {stage}")
    value = matrix.get("scientific_limits", {}).get(stage)
    if not isinstance(value, dict):
        raise ValueError(f"matrix lacks scientific limits for {stage}")
    for key in ("wall_clock_seconds", "output_bytes"):
        if type(value.get(key)) is not int or value[key] <= 0:
            raise ValueError(f"invalid scientific resource cap: {key}")
    return value


def _scientific_panel(root: Path, matrix: dict, stage: str) -> tuple[Path, dict, int]:
    spec = stage_spec(matrix, stage)
    panels = matrix.get("scientific_panels", {}).get(stage)
    if not isinstance(panels, dict):
        raise ValueError(f"matrix lacks a scientific panel declaration for {stage}")
    data = spec["base"]["data"]
    corpus_path = inside(Path(root), data["path"])
    counts = {name: int(panels["windows_per_region"]) for name in REGIONS}
    batch_size = int(panels["batch_size"])
    if batch_size <= 0 or any(count <= 0 for count in counts.values()):
        raise ValueError("scientific panel windows and batch size must be positive")
    seed = int(panels.get("selection_seed",
                          derive_seed("r1-scientific-panel-v1", matrix["study_id"], stage)))
    panel = build_panel_manifest(corpus_path, int(data["seq_len"]), counts, seed,
                                 corpus_label=data["path"])
    return corpus_path, panel, batch_size


def _capture_schedule(matrix: dict, stage: str) -> TrainingCaptureSchedule:
    spec = stage_spec(matrix, stage)
    budget = matrix.get("proposed_scientific_matrix", {}).get("capture_budget")
    if not isinstance(budget, dict):
        raise ValueError("matrix lacks the scientific capture budget")
    if list(budget.get("full_precision_tensor_views", [])) != list(CHANNELS):
        raise ValueError("capture budget must retain exactly the four full-precision tensor views")
    if budget.get("include_initial_and_final") is not True:
        raise ValueError("capture budget must include initial and final frames")
    schedule = TrainingCaptureSchedule(int(spec["base"]["optim"]["steps"]),
                                       int(budget["training_capture_every_steps"]),
                                       int(budget["maximum_training_capture_frames_per_run"]),
                                       int(budget["maximum_examples_per_frame"]))
    schedule.grid()
    return schedule


def _specification_fields(specification: AttackSpecification) -> dict[str, Any]:
    fields = asdict(specification)
    fields["positive_doses"] = [float(value) for value in specification.positive_doses]
    return fields


def _specification_from_document(document: Mapping[str, Any]) -> AttackSpecification:
    try:
        validate(instance=dict(document), schema=json.loads(ATTACK_SPEC_SCHEMA_PATH.read_text()))
    except Exception as exc:
        raise ValueError(f"attack specification document is invalid: {exc}") from exc
    fields = document["specification"]
    return AttackSpecification(
        ridge_lambda=float(fields["ridge_lambda"]),
        positive_doses=tuple(float(value) for value in fields["positive_doses"]),
        positive_floor_correlation=float(fields["positive_floor_correlation"]),
        positive_floor_accuracy=float(fields["positive_floor_accuracy"]),
        null_band=float(fields["null_band"]),
        privacy_limit_correlation=float(fields["privacy_limit_correlation"]),
        privacy_limit_accuracy=float(fields["privacy_limit_accuracy"]),
    )


def load_attack_specification(path: Path | None, stage: str) -> tuple[AttackSpecification, dict[str, Any]]:
    if path is None:
        if stage == "exploratory":
            raise ValueError("exploratory admission requires a frozen attack specification document")
        document: dict[str, Any] = {"schema_version": "r1-attack-spec-1.0", "status": "draft",
                                    "specification": _specification_fields(AttackSpecification()),
                                    "origin": "built-in draft default for calibration"}
    else:
        document = read_json(Path(path))
    specification = _specification_from_document(document)
    status = document["status"]
    if stage == "exploratory" and status != "frozen":
        raise ValueError("exploratory admission requires the attack specification to carry status: frozen")
    if stage == "calibration" and status not in ("draft", "frozen"):
        raise ValueError("calibration admission binds the draft attack specification")
    return specification, document


def validate_scientific_seed_disjointness(matrix: Mapping[str, Any]) -> None:
    proposed = matrix.get("proposed_scientific_matrix", {})
    groups = {
        "engineering": (list(matrix.get("seeds", []))
                        + list(matrix.get("performance_calibration", {}).get("seeds", []))
                        + list(matrix.get("local_validation", {}).get("seeds", []))),
        "calibration": list(proposed.get("calibration", {}).get("seeds", [])),
        "exploratory": list(proposed.get("seeds", [])),
        "reserved_confirmation": list(proposed.get("reserved_confirmation", {}).get("seeds", [])),
        "legacy_observed": list(matrix.get("seed_audit", {}).get("legacy_observed_seed_union", [])),
    }
    owners: dict[int, str] = {}
    for group, seeds in groups.items():
        for seed in seeds:
            if seed in owners:
                raise ValueError(f"seed {seed} overlaps between {owners[seed]} and {group}")
            owners[seed] = group


def validate_scientific_budget(budget: Mapping[str, Any], *, bundle_sha256: str,
                               study: str, stage: str, count: int) -> None:
    for key, value in (("schema_version", "r1-budget-1.0"), ("status", "approved_for_scientific"),
                       ("study_id", study), ("stage", stage), ("bundle_sha256", bundle_sha256)):
        if budget.get(key) != value:
            raise ValueError(f"scientific budget review {key} is absent or does not match")
    if stage not in SCIENTIFIC_STAGES:
        raise ValueError("not a scientific stage")
    if not budget.get("reviewed_by") or not budget.get("approval_reference") or not budget.get("reviewed_at_utc"):
        raise ValueError("an explicit scientific budget review is required")
    if type(budget.get("max_runs")) is not int or not (count <= budget["max_runs"] and count > 0):
        raise ValueError("requested run count exceeds the reviewed budget")
    for key in ("max_wall_seconds", "max_total_usd", "hourly_usd"):
        if not isinstance(budget.get(key), (int, float)) or not math.isfinite(budget[key]) or budget[key] <= 0:
            raise ValueError(f"budget {key} must be a finite positive number")
    if budget["max_wall_seconds"] * budget["hourly_usd"] / 3600 > budget["max_total_usd"]:
        raise ValueError("wall-time allowance exceeds reviewed compute dollars")


def validate_scientific_environment_review(review: Mapping[str, Any], environment: Mapping[str, Any],
                                           bundle_sha256: str) -> None:
    for key, value in (("schema_version", "r1-environment-review-1.0"), ("status", "approved_for_scientific"),
                       ("bundle_sha256", bundle_sha256), ("environment_sha256", sha256_json(environment)),
                       ("pip_freeze_sha256", sha256_bytes(environment["pip_freeze"].encode()))):
        if review.get(key) != value:
            raise ValueError(f"scientific environment review {key} is absent or mismatched")
    digest = review.get("container_image_digest")
    if not isinstance(digest, str) or not re.fullmatch(r"sha256:[a-f0-9]{64}", digest):
        raise ValueError("review must identify the immutable container image digest")
    if not all(review.get(key) for key in ("reviewed_by", "reviewed_at_utc", "approval_reference")):
        raise ValueError("explicit scientific environment acceptance is required")


def _validate_admission_document(admission: Mapping[str, Any]) -> None:
    try:
        validate(instance=dict(admission), schema=json.loads(ADMISSION_SCHEMA_PATH.read_text()))
    except Exception as exc:
        raise ValueError(f"scientific admission document is invalid: {exc}") from exc


def freeze_scientific(root: Path, stage: str, protocol_path: Path | None,
                      attack_spec_path: Path | None, budget_path: Path,
                      environment_review_path: Path, admission_path: Path) -> dict:
    """Bind a reviewed protocol/attack/budget/environment packet into one single-use admission."""
    root = root.resolve()
    bundle = verify_bundle(root)
    if stage not in SCIENTIFIC_STAGES:
        raise ValueError("freeze_scientific admits only calibration or exploratory stages")
    matrix = load_matrix(root / bundle["matrix"])
    scientific_stage_enabled(matrix, stage)
    spec = stage_spec(matrix, stage)
    declared_device, execution_device = _stage_execution_device(spec)
    plan = read_json(root / "r1_plan.json")["stages"][stage]
    runs = _scientific_runs(matrix, stage, Path("outputs") / bundle["study_id"] / stage)
    if len(runs) != plan["run_count"]:
        raise ValueError("scientific run enumeration differs from the prepared plan")
    _capture_schedule(matrix, stage)
    limits = scientific_limits(matrix, stage)
    digest = sha256_file(str(root / "r1_bundle.json"))
    budget = read_json(budget_path)
    validate_scientific_budget(budget, bundle_sha256=digest, study=bundle["study_id"],
                               stage=stage, count=len(runs))
    if budget["max_wall_seconds"] > limits["wall_clock_seconds"]:
        raise ValueError("reviewed duration exceeds the stage scientific cap")
    if protocol_path is None or not Path(protocol_path).is_file():
        raise ValueError("a protocol document with a bound hash is required")
    protocol_sha256 = sha256_file(str(protocol_path))
    if sha256_file(str(root / bundle["protocol"])) != protocol_sha256:
        raise ValueError("protocol document differs from the bundle-retained protocol")
    _, attack_document = load_attack_specification(attack_spec_path, stage)
    attack_source = None if attack_spec_path is None else {
        "path": str(attack_spec_path), "sha256": sha256_file(str(attack_spec_path))}
    validate_scientific_seed_disjointness(matrix)
    environment = environment_record()
    if execution_device == "cuda" and not environment["cuda_available"]:
        raise ValueError("acceptance requires the selected CUDA environment; CPU/MPS fallback is prohibited")
    environment_review = read_json(environment_review_path)
    validate_scientific_environment_review(environment_review, environment, digest)
    admission = {"schema_version": "r1-scientific-admission-1.0", "created_at_utc": now(),
                 "study_id": bundle["study_id"], "stage": stage, "bundle_sha256": digest,
                 "run_count": len(runs),
                 "matrix_declared_device": declared_device, "execution_device": execution_device,
                 "protocol": bundle["protocol"], "protocol_sha256": protocol_sha256,
                 "protocol_status_declared": matrix.get("protocol_status"),
                 "attack_spec_document": attack_document,
                 "attack_spec_sha256": sha256_json(attack_document),
                 "attack_spec_status": attack_document["status"],
                 "attack_spec_source": attack_source,
                 "budget": budget, "budget_sha256": sha256_json(budget),
                 "environment": environment, "environment_sha256": sha256_json(environment),
                 "environment_review": environment_review,
                 "environment_review_sha256": sha256_json(environment_review),
                 "limits": dict(limits), "scientific_evidence": True,
                 "scope": "Local pre-execution binding; no independently witnessed preregistration asserted.",
                 "note": "Admits exactly one local execution of the bound scientific stage; "
                         "a failure consumes the underlying budget approval."}
    _validate_admission_document(admission)
    write_new(admission_path, admission)
    return admission


def _bind_scientific_runs(root: Path, runs: list[dict], execution_root: Path,
                          panel: Mapping[str, Any], execution_id: str,
                          admission: Mapping[str, Any], panel_batch_size: int,
                          schedule: TrainingCaptureSchedule) -> None:
    for resolved in runs:
        resolved["run_id"] += f"--e{execution_id}"
        resolved["output_dir"] = str(execution_root / "runs" / resolved["run_id"])
        data_path = inside(root, resolved["data"]["path"])
        resolved["data"]["path"] = str(data_path.relative_to(root))
        resolved["r1"].update({"transport": "loopback-tcp-r1-wire-1.0",
                               "semantic_validation": True,
                               "attack_validation": (resolved["split"] is not None
                                                     and resolved["mode"] == "trained"),
                               "panel_sha256": panel["panel_sha256"],
                               "scientific_evidence": True,
                               "protocol_sha256": admission["protocol_sha256"],
                               "attack_spec_sha256": admission["attack_spec_sha256"],
                               "attack_spec_status": admission["attack_spec_status"],
                               "attack_panel_batch_size": panel_batch_size,
                               "training_capture": {"every_steps": schedule.every_steps,
                                                    "max_frames": schedule.max_frames,
                                                    "max_examples": schedule.max_examples}})


def execute_scientific(root: Path, admission_path: Path, execution_id: str) -> dict:
    """Consume one scientific admission exactly once, driving every admitted cell."""
    root = root.resolve()
    if Path(__file__).resolve().parents[2] != root:
        raise ValueError("execute the copied bundle's scripts using its own source tree")
    bundle = verify_bundle(root)
    admission = read_json(admission_path)
    stage = admission.get("stage")
    if admission.get("schema_version") != "r1-scientific-admission-1.0" or stage not in SCIENTIFIC_STAGES:
        raise ValueError("not a scientific admission")
    if (admission.get("study_id") != bundle["study_id"] or
            admission.get("scientific_evidence") is not True or
            admission["bundle_sha256"] != sha256_file(str(root / "r1_bundle.json"))):
        raise ValueError("admission identity mismatch")
    if (sha256_json(admission["budget"]) != admission["budget_sha256"] or
            sha256_json(admission["environment"]) != admission["environment_sha256"] or
            sha256_json(admission["environment_review"]) != admission["environment_review_sha256"] or
            sha256_json(admission["attack_spec_document"]) != admission["attack_spec_sha256"]):
        raise ValueError("admission contents differ from their digests")
    _validate_admission_document(admission)
    validate_scientific_environment_review(admission["environment_review"], admission["environment"],
                                           admission["bundle_sha256"])
    if sha256_json(environment_record()) != admission["environment_sha256"]:
        raise ValueError("environment changed after acceptance")
    if sha256_file(str(root / bundle["protocol"])) != admission["protocol_sha256"]:
        raise ValueError("bundle protocol differs from the admitted document")
    matrix = load_matrix(root / bundle["matrix"])
    scientific_stage_enabled(matrix, stage)
    spec = stage_spec(matrix, stage)
    declared_device, execution_device = _stage_execution_device(spec)
    if [declared_device, execution_device] != [admission["matrix_declared_device"],
                                               admission["execution_device"]]:
        raise ValueError("matrix device declaration changed after admission")
    attack_spec = _specification_from_document(admission["attack_spec_document"])
    if admission["attack_spec_status"] != admission["attack_spec_document"]["status"]:
        raise ValueError("attack specification status differs from the admission")
    if stage == "exploratory" and admission["attack_spec_status"] != "frozen":
        raise ValueError("exploratory execution requires a frozen attack specification")
    safe_name(execution_id)
    execution_root = root / "outputs" / bundle["study_id"] / f"{stage}--{execution_id}"
    runs = _scientific_runs(matrix, stage, execution_root / "runs")
    if len(runs) != admission["run_count"]:
        raise ValueError("run enumeration differs from the admission")
    _, panel, panel_batch_size = _scientific_panel(root, matrix, stage)
    schedule = _capture_schedule(matrix, stage)
    _bind_scientific_runs(root, runs, execution_root, panel, execution_id, admission,
                          panel_batch_size, schedule)
    validate_scientific_budget(admission["budget"], bundle_sha256=admission["bundle_sha256"],
                               study=bundle["study_id"], stage=stage, count=len(runs))
    limits = scientific_limits(matrix, stage)
    if admission["budget"]["max_wall_seconds"] > limits["wall_clock_seconds"]:
        raise ValueError("reviewed duration exceeds the stage scientific cap")
    claim = claim_admission(root, admission, execution_id, scientific_evidence=True)
    execution_root.mkdir(parents=True, exist_ok=False)
    spec_path = execution_root / "attack_spec.json"
    write_new(spec_path, admission["attack_spec_document"])
    frozen_spec_path = spec_path if admission["attack_spec_status"] == "frozen" else None
    panel_path = execution_root / "panel_manifest.json"
    save_panel_manifest(panel_path, panel)
    write_new(execution_root / "admission.json", admission)
    write_new(execution_root / "admission_claim.json", claim)
    launch = {"schema_version": "r1-scientific-launch-1.0", "study_id": bundle["study_id"],
              "stage": stage, "execution_id": execution_id, "launch_uuid": str(uuid.uuid4()),
              "started_at_utc": now(), "pid": os.getpid(), "hostname": platform.node(),
              "argv": list(sys.argv), "cwd": str(Path.cwd()),
              "bundle_root_at_execution": str(root),
              "bundle_sha256": admission["bundle_sha256"],
              "admission_sha256": sha256_file(str(execution_root / "admission.json")),
              "admission_claim_sha256": sha256_file(str(execution_root / "admission_claim.json")),
              "panel_manifest": "panel_manifest.json", "panel_sha256": panel["panel_sha256"],
              "attack_spec": "attack_spec.json",
              "attack_spec_sha256": admission["attack_spec_sha256"],
              "attack_spec_status": admission["attack_spec_status"],
              "protocol": bundle["protocol"], "protocol_sha256": admission["protocol_sha256"],
              "matrix_declared_device": declared_device, "execution_device": execution_device,
              "runs": runs, "scientific_evidence": True}
    write_new(execution_root / "launch.json", launch)
    start = time.monotonic()
    completed = []
    failure = None
    try:
        with manifest_namespace(root, execution_root / "manifests"):
            with wall_clock_guard(admission["budget"]["max_wall_seconds"]):
                for resolved in runs:
                    if time.monotonic() - start >= admission["budget"]["max_wall_seconds"]:
                        raise RuntimeError("scientific wall-time allowance reached before next run")
                    verify_bundle(root)
                    if output_size(execution_root) >= limits["output_bytes"]:
                        raise RuntimeError("scientific output cap reached before next run")
                    record_root = execution_root / "execution_records"
                    run_id = resolved["run_id"]
                    begin = {"schema_version": "r1-process-start-1.0",
                             "launch_uuid": launch["launch_uuid"],
                             "launch_sha256": sha256_file(str(execution_root / "launch.json")),
                             "pid": os.getpid(), "run_id": run_id, "started_at_utc": now(),
                             "config_sha256": sha256_json(resolved), "scientific_evidence": True}
                    write_new(record_root / f"{run_id}.start.json", begin)
                    try:
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                            torch.cuda.reset_peak_memory_stats()
                        run_start_monotonic = time.monotonic()
                        manifest_path = run_scientific_cell(root, resolved, panel, panel_path,
                                                            attack_spec, frozen_spec_path,
                                                            scientific_evidence=True)
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        manifest_state.completed_manifest_exists(run_id, resolved=resolved)
                        metrics = read_json(Path(resolved["output_dir"]) / "metrics_summary.json")
                        if output_size(execution_root) > limits["output_bytes"]:
                            raise RuntimeError("scientific output cap exceeded; preserve artifacts and stop")
                        end = {"schema_version": "r1-process-end-1.0", "run_id": run_id,
                               "launch_uuid": launch["launch_uuid"], "pid": os.getpid(),
                               "finished_at_utc": now(), "status": "completed",
                               "start_sha256": sha256_file(str(record_root / f"{run_id}.start.json")),
                               "manifest": str(manifest_path.relative_to(root)),
                               "manifest_sha256": sha256_file(str(manifest_path)),
                               "scientific_evidence": True,
                               "resource_measurements": {
                                   "complete_run_wall_seconds": time.monotonic() - run_start_monotonic,
                                   "cuda_peak_allocated_bytes": (torch.cuda.max_memory_allocated()
                                                                 if torch.cuda.is_available() else None),
                                   "cuda_peak_reserved_bytes": (torch.cuda.max_memory_reserved()
                                                                if torch.cuda.is_available() else None),
                                   "remote_cuda_peak_allocated_bytes": metrics.get("remote_cuda_peak_allocated_bytes"),
                                   "remote_cuda_peak_reserved_bytes": metrics.get("remote_cuda_peak_reserved_bytes"),
                                   "actual_wire_bytes": metrics.get("actual_wire_bytes", 0),
                                   "output_bytes_including_source": output_size(Path(resolved["output_dir"])),
                                   "scope": "Two-process loopback split execution; not a two-host network latency measurement."}}
                        write_new(record_root / f"{run_id}.end.json", end)
                        completed.append(run_id)
                    except BaseException as exc:
                        write_new(record_root / f"{run_id}.failure.json", {"run_id": run_id,
                                  "launch_uuid": launch["launch_uuid"], "failed_at_utc": now(),
                                  "error_type": type(exc).__name__, "error": str(exc),
                                  "start_sha256": sha256_file(str(record_root / f"{run_id}.start.json")),
                                  "scientific_evidence": True})
                        raise
    except BaseException as exc:
        failure = {"type": type(exc).__name__, "detail": str(exc)}
        raise
    finally:
        write_new(execution_root / "execution_summary.json", {"finished_at_utc": now(),
                  "launch_sha256": sha256_file(str(execution_root / "launch.json")),
                  "completed": completed, "failure": failure, "wall_seconds": time.monotonic() - start,
                  "scientific_evidence": True,
                  "budget_note": "Python wall-time watchdog is supplemented by before-run checks; external provider watchdog must enforce billing cap.",
                  "storage_note": "Output bytes are checked before and after each run. This is a stop threshold and may overshoot during a run; it is not a filesystem quota."})
    return {"execution_root": str(execution_root), "completed": len(completed),
            "scientific_evidence": True}
