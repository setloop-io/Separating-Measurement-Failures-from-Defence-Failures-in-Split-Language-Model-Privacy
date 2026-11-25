"""Semantic validation for prospective R1 artifacts."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from jsonschema import validate

from .transcript import validate_inventory
from .utils import sha256_file, state_dict_checksum

SCHEMA_ROOT = Path(__file__).resolve().parents[2] / "experiments/schemas"


class ArtifactValidationError(RuntimeError):
    """An artifact is present but cannot support its declared semantics."""


def _finite(value: Any, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ArtifactValidationError(f"{field} must be finite")
    return float(value)


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(f"invalid JSON artifact {path}: {exc}") from exc


def validate_config(path: Path, resolved: Mapping[str, Any]) -> None:
    if _json(path) != dict(resolved):
        raise ArtifactValidationError("resolved configuration bytes do not match the run")


def validate_curve(path: Path, expected_steps: int) -> None:
    rows = []
    try:
        for line in path.read_text().splitlines():
            rows.append(json.loads(line))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(f"invalid curve artifact: {exc}") from exc
    if not rows or rows[0].get("step") != 0 or rows[-1].get("step") != expected_steps:
        raise ArtifactValidationError("curve does not cover the declared run")
    steps = [row.get("step") for row in rows]
    if any(type(step) is not int or step < 0 for step in steps) or steps != sorted(set(steps)):
        raise ArtifactValidationError("curve steps are invalid or duplicated")
    for row in rows:
        for field in ("train_loss_smoothed", "eval_loss", "wall_time"):
            if row.get(field) is not None:
                _finite(row[field], f"curve.{field}")


def validate_metrics(path: Path, resolved: Mapping[str, Any]) -> None:
    value = _json(path)
    for field in ("run_id", "mode", "seed"):
        if value.get(field) != resolved.get(field):
            raise ArtifactValidationError(f"metrics {field} differs from resolved run")
    _finite(value.get("final_eval_loss"), "metrics.final_eval_loss")
    if _finite(value.get("wall_time_seconds"), "metrics.wall_time_seconds") < 0:
        raise ArtifactValidationError("metrics wall time cannot be negative")


def validate_inventory_file(path: Path) -> None:
    value = _json(path)
    try:
        validate_inventory(value)
    except Exception as exc:
        raise ArtifactValidationError(f"invalid transcript inventory: {exc}") from exc
    if any(type(count) is not int or count < 0 for count in value["accumulated_bytes"].values()):
        raise ArtifactValidationError("transcript byte counts must be nonnegative integers")


def _tensor(value: Any, field: str, shape: Sequence[int] | None = None) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
        raise ArtifactValidationError(f"{field} must be a finite tensor")
    if shape is not None and tuple(value.shape) != tuple(shape):
        raise ArtifactValidationError(f"{field} has shape {tuple(value.shape)}, expected {tuple(shape)}")
    return value


def validate_capture(path: Path, required_channels: Sequence[str],
                     width: int, seq_len: int) -> None:
    try:
        events = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ArtifactValidationError(f"capture is not loadable: {exc}") from exc
    if not isinstance(events, list) or not events:
        raise ArtifactValidationError("capture must contain at least one event")
    steps, window_ids, panel_ids, checkpoint_ids = [], set(), set(), set()
    for index, event in enumerate(events):
        if not isinstance(event, dict) or type(event.get("step")) is not int or event["step"] < 0:
            raise ArtifactValidationError(f"capture event {index} has no nonnegative integer step")
        steps.append(event["step"])
        tokens = _tensor(event.get("tokens"), f"capture[{index}].tokens")
        targets = _tensor(event.get("targets"), f"capture[{index}].targets")
        if (tokens.ndim != 2 or targets.ndim != 2 or tokens.shape != targets.shape or
                tokens.shape[1] != seq_len or tokens.dtype != torch.long or targets.dtype != torch.long):
            raise ArtifactValidationError("capture token and target panels are inconsistent")
        batch = tokens.shape[0]
        region = event.get("panel_region")
        panel_sha = event.get("panel_sha256")
        checkpoint_id = event.get("checkpoint_id")
        indices = event.get("window_indices")
        inputs = event.get("window_input_sha256")
        target_hashes = event.get("window_target_sha256")
        if (region not in {"attacker_auxiliary", "qualification", "heldout"} or
                not isinstance(panel_sha, str) or re.fullmatch(r"[a-f0-9]{64}", panel_sha) is None or
                not isinstance(checkpoint_id, str) or not checkpoint_id or
                not all(isinstance(value, list) and len(value) == batch
                        for value in (indices, inputs, target_hashes))):
            raise ArtifactValidationError("capture panel binding is incomplete")
        panel_ids.add(panel_sha)
        checkpoint_ids.add(checkpoint_id)
        for window_index, input_hash, target_hash in zip(indices, inputs, target_hashes):
            if (type(window_index) is not int or window_index < 0 or
                    not isinstance(input_hash, str) or re.fullmatch(r"[a-f0-9]{64}", input_hash) is None or
                    not isinstance(target_hash, str) or re.fullmatch(r"[a-f0-9]{64}", target_hash) is None):
                raise ArtifactValidationError("capture window identity is invalid")
            identity = (panel_sha, region, window_index, input_hash, target_hash)
            if identity in window_ids:
                raise ArtifactValidationError("capture reuses a panel window")
            window_ids.add(identity)
        metadata = event.get("channel_metadata")
        if not isinstance(metadata, dict) or not set(required_channels).issubset(metadata):
            raise ArtifactValidationError("capture channel metadata is incomplete")
        for channel in required_channels:
            value = _tensor(event.get(channel), f"capture[{index}].{channel}")
            if value.ndim != 3 or value.shape != (batch, seq_len, width):
                raise ArtifactValidationError(f"capture channel {channel} has an invalid shape")
            item = metadata[channel]
            if (not isinstance(item, dict) or item.get("shape") != list(value.shape) or
                    item.get("dtype") != str(value.dtype) or
                    item.get("processing") not in {"quantized-wire-roundtrip", "full-precision-wire"}):
                raise ArtifactValidationError(f"capture channel metadata differs for {channel}")
    if steps != sorted(steps):
        raise ArtifactValidationError("capture events are not ordered by step")
    if len(panel_ids) != 1 or len(checkpoint_ids) != 1:
        raise ArtifactValidationError("capture events do not share one panel and checkpoint")


def _state_dict(value: Any, field: str) -> Mapping[str, torch.Tensor]:
    if not isinstance(value, dict) or not value:
        raise ArtifactValidationError(f"{field} must be a nonempty state dictionary")
    for name, tensor in value.items():
        if not isinstance(name, str):
            raise ArtifactValidationError(f"{field} has a non-string key")
        _tensor(tensor, f"{field}.{name}")
    return value


def validate_state_dict(path: Path, expected_shapes: Mapping[str, Sequence[int]] | None = None) -> None:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ArtifactValidationError(f"model state is not loadable: {exc}") from exc
    state = _state_dict(value, "model")
    if expected_shapes is not None:
        observed = {name: list(tensor.shape) for name, tensor in state.items()}
        if observed != {name: list(shape) for name, shape in expected_shapes.items()}:
            raise ArtifactValidationError("model parameter shapes differ from initialization")


def validate_capture_panel(path: Path, panel: Mapping[str, Any]) -> None:
    try:
        events = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ArtifactValidationError(f"capture panel binding is not loadable: {exc}") from exc
    expected = {(item["region"], item["index"]):
                (item["input_sha256"], item["target_sha256"])
                for item in panel["windows"]}
    seen = set()
    for event in events:
        region = event["panel_region"]
        for index, input_hash, target_hash in zip(
                event["window_indices"], event["window_input_sha256"],
                event["window_target_sha256"]):
            key = (region, index)
            if key in seen or expected.get(key) != (input_hash, target_hash):
                raise ArtifactValidationError("capture window differs from the bound panel")
            seen.add(key)
    required = {key for key in expected if key[0] in {"attacker_auxiliary", "qualification", "heldout"}}
    if seen != required:
        raise ArtifactValidationError("capture does not cover the declared attack panels exactly")


def validate_capture_pair(client_path: Path, remote_path: Path,
                          channels: Sequence[str]) -> None:
    try:
        client = torch.load(client_path, map_location="cpu", weights_only=True)
        remote = torch.load(remote_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ArtifactValidationError(f"capture pair is not loadable: {exc}") from exc
    if not isinstance(client, list) or not isinstance(remote, list) or len(client) != len(remote):
        raise ArtifactValidationError("client and remote capture counts differ")
    for index, (local_event, remote_event) in enumerate(zip(client, remote)):
        if local_event.get("step") != remote_event.get("step"):
            raise ArtifactValidationError("client and remote capture steps differ")
        for channel in channels:
            local = _tensor(local_event.get(channel), f"client_capture[{index}].{channel}")
            observed = _tensor(remote_event.get(channel), f"remote_capture[{index}].{channel}")
            if not torch.equal(local, observed):
                raise ArtifactValidationError(f"client and remote capture channel differs: {channel}")


def _optimizer_state(value: Any, field: str) -> Mapping[str, Any]:
    if (not isinstance(value, dict) or set(value) != {"state", "param_groups"} or
            not isinstance(value["state"], dict) or not isinstance(value["param_groups"], list) or
            not value["param_groups"]):
        raise ArtifactValidationError(f"{field} is not a reloadable optimizer state")
    return value


def validate_training_checkpoint(path: Path, expected_step: int,
                                 expected_shapes: Mapping[str, Sequence[int]] | None = None,
                                 require_remote: bool = False) -> None:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ArtifactValidationError(f"training checkpoint is not loadable: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != "r1-training-checkpoint-1.0":
        raise ArtifactValidationError("unsupported training checkpoint schema")
    if value.get("completed_steps") != expected_step:
        raise ArtifactValidationError("training checkpoint step differs")
    model = _state_dict(value.get("model"), "checkpoint.model")
    if expected_shapes is not None:
        observed = {name: list(tensor.shape) for name, tensor in model.items()}
        if observed != {name: list(shape) for name, shape in expected_shapes.items()}:
            raise ArtifactValidationError("checkpoint model shapes differ from initialization")
    _optimizer_state(value.get("optimizer"), "checkpoint.optimizer")
    if require_remote:
        for field in ("remote_state_sha256", "remote_checkpoint_sha256"):
            digest = value.get(field)
            if not isinstance(digest, str) or re.fullmatch(r"[a-f0-9]{64}", digest) is None:
                raise ArtifactValidationError(f"training checkpoint {field} is invalid")
        if not isinstance(value.get("remote_checkpoint_id"), str):
            raise ArtifactValidationError("training checkpoint has no remote checkpoint identifier")
        transaction = value.get("last_transaction_id")
        if ((expected_step == 0 and transaction is not None) or
                (expected_step > 0 and (not isinstance(transaction, str) or
                                        re.fullmatch(r"[a-f0-9]{64}", transaction) is None))):
            raise ArtifactValidationError("training checkpoint transaction identity is invalid")
    rng = value.get("rng")
    if not isinstance(rng, dict) or "cpu" not in rng:
        raise ArtifactValidationError("training checkpoint lacks RNG state")
    cpu_rng = _tensor(rng["cpu"], "checkpoint.rng.cpu")
    if cpu_rng.ndim != 1 or cpu_rng.dtype != torch.uint8:
        raise ArtifactValidationError("training checkpoint CPU RNG state is invalid")


def validate_remote_checkpoint(path: Path, expected_step: int) -> None:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ArtifactValidationError(f"remote checkpoint is not loadable: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != "r1-remote-checkpoint-1.0":
        raise ArtifactValidationError("unsupported remote checkpoint schema")
    if value.get("completed_steps") != expected_step:
        raise ArtifactValidationError("remote checkpoint step differs")
    transaction = value.get("last_transaction_id")
    if ((expected_step == 0 and transaction is not None) or
            (expected_step > 0 and (not isinstance(transaction, str) or
                                    re.fullmatch(r"[a-f0-9]{64}", transaction) is None))):
        raise ArtifactValidationError("remote checkpoint transaction identity is invalid")
    config = value.get("config")
    if not isinstance(config, dict) or config.get("mode") not in {"trained", "frozen_random", "removed_remote"}:
        raise ArtifactValidationError("remote checkpoint configuration is invalid")
    model = value.get("model")
    if config["mode"] == "removed_remote":
        if model != {}:
            raise ArtifactValidationError("removed remote checkpoint contains model parameters")
    else:
        from .split import Surrogate

        state = _state_dict(model, "remote.model")
        try:
            Surrogate(int(config["width"]), int(config["depth"])).load_state_dict(state, strict=True)
        except Exception as exc:
            raise ArtifactValidationError(f"remote checkpoint model is not reloadable: {exc}") from exc
    if config["mode"] == "trained":
        _optimizer_state(value.get("optimizer"), "remote.optimizer")
    elif value.get("optimizer") is not None:
        raise ArtifactValidationError("non-trained remote checkpoint contains optimizer state")
    outbound_rng = _tensor(value.get("outbound_rng"), "remote.outbound_rng")
    if outbound_rng.ndim != 1 or outbound_rng.dtype != torch.uint8:
        raise ArtifactValidationError("remote checkpoint RNG state is invalid")


def validate_checkpoint_pair(client_path: Path, remote_path: Path,
                             expected_step: int) -> None:
    try:
        client = torch.load(client_path, map_location="cpu", weights_only=True)
        remote = torch.load(remote_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ArtifactValidationError(f"checkpoint pair is not loadable: {exc}") from exc
    if (client.get("completed_steps") != expected_step or remote.get("completed_steps") != expected_step or
            client.get("last_transaction_id") != remote.get("last_transaction_id") or
            client.get("remote_state_sha256") != state_dict_checksum(remote.get("model", {})) or
            client.get("remote_checkpoint_id") != remote_path.name or
            client.get("remote_checkpoint_sha256") != sha256_file(str(remote_path))):
        raise ArtifactValidationError("client and remote checkpoints are not a coordinated pair")


def validate_wire_audit(path: Path) -> None:
    from .wire import MESSAGE_TYPES, PREFIX

    value = _json(path)
    records = value.get("records")
    if value.get("schema_version") != "r1-wire-audit-1.0" or not isinstance(records, list) or not records:
        raise ArtifactValidationError("invalid wire audit schema")
    total, sessions, grouped = 0, set(), {}
    for record in records:
        for field in ("header_bytes", "payload_bytes", "wire_bytes", "step", "sequence"):
            if type(record.get(field)) is not int or record[field] < 0:
                raise ArtifactValidationError(f"wire record has invalid {field}")
        if (record.get("direction") not in {"sent", "received"} or
                record.get("message_type") not in MESSAGE_TYPES or
                not isinstance(record.get("session_id"), str) or not record["session_id"] or
                not isinstance(record.get("request_id"), str) or not record["request_id"] or
                not isinstance(record.get("payload_sha256"), str) or
                re.fullmatch(r"[a-f0-9]{64}", record["payload_sha256"]) is None):
            raise ArtifactValidationError("wire record identity is invalid")
        if record["wire_bytes"] != PREFIX.size + record["header_bytes"] + record["payload_bytes"]:
            raise ArtifactValidationError("wire record byte total is inconsistent")
        total += record["wire_bytes"]
        sessions.add(record["session_id"])
        grouped.setdefault(record["sequence"], []).append(record)
    if value.get("total_bytes") != total or len(sessions) != 1:
        raise ArtifactValidationError("wire audit aggregate or session identity differs")
    sequences = sorted(grouped)
    if sequences != list(range(sequences[-1] + 1)):
        raise ArtifactValidationError("wire audit sequences are not contiguous")
    for sequence, group in grouped.items():
        sent = [record for record in group if record["direction"] == "sent"]
        received = [record for record in group if record["direction"] == "received"]
        if not sent or len(sent) != len(received):
            raise ArtifactValidationError("wire audit request/response pairing differs")
        identity = {(record["message_type"], record["request_id"], record["step"])
                    for record in group}
        if len(identity) != 1:
            raise ArtifactValidationError("wire audit request/response identity differs")
    if records[-1]["message_type"] != "CLOSE":
        raise ArtifactValidationError("wire audit has no terminal CLOSE")


def validate_step_journal(path: Path, expected_steps: int) -> None:
    try:
        entries = [json.loads(line) for line in path.read_text().splitlines()]
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(f"invalid step journal: {exc}") from exc
    if len(entries) != expected_steps * 2:
        raise ArtifactValidationError("step journal does not contain prepare/commit pairs")
    for step in range(expected_steps):
        prepared, committed = entries[2 * step : 2 * step + 2]
        transaction = prepared.get("transaction_id")
        if (prepared.get("status") != "prepared" or committed.get("status") != "committed" or
                prepared.get("step") != step or committed.get("step") != step or
                transaction != committed.get("transaction_id") or
                not isinstance(transaction, str) or re.fullmatch(r"[a-f0-9]{64}", transaction) is None or
                committed.get("completed_steps") != step + 1 or
                prepared.get("remote_state_before") != committed.get("remote_state_before") or
                not isinstance(committed.get("remote_state_after"), str)):
            raise ArtifactValidationError("step journal transaction pair is inconsistent")


def validate_attack_banks(path: Path, defense: str) -> dict[str, str]:
    from .r1_attacks import AttackBank, CHANNELS, TARGETS, RidgeModel, attack_bank_digest

    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ArtifactValidationError(f"attack banks are not loadable: {exc}") from exc
    banks = value.get("banks") if isinstance(value, dict) else None
    expected = {f"{target}@{channel}@{defense}" for target in TARGETS for channel in CHANNELS}
    if (not isinstance(value, dict) or value.get("schema_version") != "r1-attack-banks-1.0" or
            not isinstance(banks, dict) or set(banks) != expected):
        raise ArtifactValidationError("attack bank target/channel set is invalid")
    digests = {}
    for key, item in banks.items():
        target, channel, _ = key.split("@", 2)
        if item.get("target") != target or item.get("channel") != channel:
            raise ArtifactValidationError("attack bank identity differs")
        models = []
        for name in ("probe", "null"):
            raw = item.get(name)
            if not isinstance(raw, dict) or set(raw) != {"x_mean", "x_scale", "y_mean", "weights"}:
                raise ArtifactValidationError("attack bank model fields are invalid")
            tensors = {field: _tensor(tensor, f"attack_bank.{name}.{field}")
                       for field, tensor in raw.items()}
            models.append(RidgeModel(*(tensors[field].numpy()
                                      for field in ("x_mean", "x_scale", "y_mean", "weights"))))
        expected_classes = {"punctuation_fraction": None, "next_byte_class": 5, "input_byte": 256}[target]
        if item.get("classes") != expected_classes or type(item.get("fit_seed")) is not int or item["fit_seed"] < 0:
            raise ArtifactValidationError("attack bank class count or fit seed is invalid")
        bank = AttackBank(target, channel, models[0], models[1], item["classes"], item["fit_seed"])
        digest = attack_bank_digest(bank)
        if digest != item.get("sha256"):
            raise ArtifactValidationError("attack bank digest differs")
        digests[key] = digest
    return digests


def validate_attack_report(path: Path) -> None:
    from .r1_attacks import CHANNELS, TARGETS

    value = _json(path)
    try:
        schema = json.loads((SCHEMA_ROOT / "r1_attack.schema.json").read_text())
        validate(instance=value, schema=schema)
    except Exception as exc:
        raise ArtifactValidationError(f"invalid R1 attack report: {exc}") from exc
    expected = {f"{target}@{channel}@{value['defense']}"
                for target in TARGETS for channel in CHANNELS}
    if set(value["results"]) != expected:
        raise ArtifactValidationError("attack report does not contain the required target/channel set")
    for key, result in value["results"].items():
        target, channel, _ = key.split("@", 2)
        qualification = result["qualification"]
        if qualification.get("target") != target or qualification.get("channel") != channel:
            raise ArtifactValidationError("attack qualification identity differs from its result key")
        controls = qualification.get("positive_control")
        if not isinstance(controls, list) or not controls:
            raise ArtifactValidationError("attack qualification has no positive-control curve")
        doses = []
        for control in controls:
            doses.append(_finite(control.get("dose"), "attack.positive_control.dose"))
            for field in ("score", "null_score", "advantage"):
                _finite(control.get(field), f"attack.positive_control.{field}")
        if len(doses) != len(set(doses)) or qualification.get("qualification_dose") != max(doses):
            raise ArtifactValidationError("attack qualification dose rule is inconsistent")
        for field in ("positive_floor", "null_band"):
            _finite(qualification.get(field), f"attack.qualification.{field}")
        null = qualification.get("null_pipeline")
        if not isinstance(null, dict):
            raise ArtifactValidationError("attack qualification has no null pipeline")
        for field in ("score", "null_score", "advantage"):
            _finite(null.get(field), f"attack.null_pipeline.{field}")
        qualified = qualification.get("qualified") is True
        if qualified != (result["status"] != "UNDETERMINED"):
            raise ArtifactValidationError("attack qualification and status disagree")
        if qualified != (result["score"] is not None):
            raise ArtifactValidationError("qualified attack must have exactly one heldout score")
        if result["score"] is not None:
            for field in ("score", "null_score", "advantage"):
                _finite(result["score"].get(field), f"attack.score.{field}")


def validation_record(paths: Sequence[Path], output_dir: Path) -> dict[str, Any]:
    root = output_dir.resolve()
    records = []
    for path in paths:
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            raise ArtifactValidationError("semantic validation input escapes its output directory")
        records.append({"path": str(resolved.relative_to(root)), "bytes": resolved.stat().st_size,
                        "sha256": sha256_file(str(resolved))})
    return {"schema_version": "r1-semantic-validation-1.0", "artifacts": records,
            "status": "validated", "scientific_evidence": False}


def validate_semantic_record(path: Path, output_dir: Path) -> None:
    value = _json(path)
    if (value.get("schema_version") != "r1-semantic-validation-1.0" or
            value.get("status") != "validated" or value.get("scientific_evidence") is not False):
        raise ArtifactValidationError("invalid semantic validation record")
    seen = set()
    for item in value.get("artifacts", []):
        relative = Path(item.get("path", ""))
        if relative.is_absolute() or ".." in relative.parts:
            raise ArtifactValidationError("semantic validation input path is invalid")
        artifact = (output_dir / relative).resolve()
        if (not artifact.is_relative_to(output_dir.resolve()) or artifact in seen or
                not artifact.is_file() or artifact.stat().st_size != item.get("bytes") or
                sha256_file(str(artifact)) != item.get("sha256")):
            raise ArtifactValidationError("semantic validation input differs or escapes")
        seen.add(artifact)
    if not seen:
        raise ArtifactValidationError("semantic validation record is empty")
