"""Fresh target-checkpoint attack and qualification primitives for R1."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import torch

from .privacy_attacks import next_byte_class
from .privacy_attacks_p4b import PUNCTUATION_BYTES
from .utils import derive_seed, sha256_json

CHANNELS = ("forward_values", "returned_values", "inbound_gradients", "outbound_gradients")
TARGETS = ("punctuation_fraction", "next_byte_class", "input_byte")
Status = Literal["PASS", "REJECTED", "UNDETERMINED"]
REQUIRED_INPUT_BINDINGS = frozenset({"panel", "client_checkpoint", "remote_checkpoint", "capture"})


@dataclass(frozen=True)
class AttackSpecification:
    ridge_lambda: float = 1.0
    positive_doses: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)
    positive_floor_correlation: float = 0.20
    positive_floor_accuracy: float = 0.10
    null_band: float = 0.02
    privacy_limit_correlation: float = 0.10
    privacy_limit_accuracy: float = 0.02

    def __post_init__(self) -> None:
        values = (self.ridge_lambda, self.positive_floor_correlation,
                  self.positive_floor_accuracy, self.null_band,
                  self.privacy_limit_correlation, self.privacy_limit_accuracy)
        if any(not math_is_finite_positive(value) for value in values):
            raise ValueError("attack specification values must be finite and positive")
        if (not self.positive_doses or any(not math_is_finite_positive(value)
                                           for value in self.positive_doses)):
            raise ValueError("positive-control doses must be finite and positive")


@dataclass(frozen=True)
class RidgeModel:
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: np.ndarray
    weights: np.ndarray


@dataclass(frozen=True)
class AttackBank:
    target: str
    channel: str
    probe: RidgeModel
    null: RidgeModel
    classes: int | None
    fit_seed: int


def _model_digest(model: RidgeModel) -> str:
    digest = hashlib.sha256()
    for value in (model.x_mean, model.x_scale, model.y_mean, model.weights):
        array = np.asarray(value, dtype="<f8")
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def attack_bank_digest(bank: AttackBank) -> str:
    return sha256_json({"target": bank.target, "channel": bank.channel,
                        "probe": _model_digest(bank.probe), "null": _model_digest(bank.null),
                        "classes": bank.classes, "fit_seed": bank.fit_seed})


def _collection_binding(events: Sequence[Mapping[str, Any]],
                        expected_region: str) -> dict[str, Any]:
    if not events:
        raise ValueError(f"attack panel {expected_region} is empty")
    panel_ids, checkpoint_ids, windows = set(), set(), []
    for event in events:
        tokens = event.get("tokens")
        count = int(tokens.shape[0]) if isinstance(tokens, torch.Tensor) and tokens.ndim == 2 else -1
        indices = event.get("window_indices")
        input_hashes = event.get("window_input_sha256")
        target_hashes = event.get("window_target_sha256")
        if (event.get("panel_region") != expected_region or count <= 0 or
                not isinstance(indices, list) or not isinstance(input_hashes, list) or
                not isinstance(target_hashes, list) or
                len(indices) != count or len(input_hashes) != count or len(target_hashes) != count):
            raise ValueError(f"attack events are not bound to panel region {expected_region}")
        panel_ids.add(event.get("panel_sha256"))
        checkpoint_ids.add(event.get("checkpoint_id"))
        for index, input_hash, target_hash in zip(indices, input_hashes, target_hashes):
            if (type(index) is not int or index < 0 or
                    not isinstance(input_hash, str) or re.fullmatch(r"[a-f0-9]{64}", input_hash) is None or
                    not isinstance(target_hash, str) or re.fullmatch(r"[a-f0-9]{64}", target_hash) is None):
                raise ValueError("attack event window identity is invalid")
            windows.append((expected_region, index, input_hash, target_hash))
    if (len(panel_ids) != 1 or None in panel_ids or len(checkpoint_ids) != 1 or
            None in checkpoint_ids or len(windows) != len(set(windows))):
        raise ValueError("attack event panel/checkpoint bindings are inconsistent")
    return {"region": expected_region, "panel_sha256": panel_ids.pop(),
            "checkpoint_id": checkpoint_ids.pop(), "windows": windows,
            "windows_sha256": sha256_json(windows)}


def _validate_attack_inputs(auxiliary: Sequence[Mapping[str, Any]],
                            qualification: Sequence[Mapping[str, Any]],
                            heldout: Sequence[Mapping[str, Any]],
                            input_bindings: Mapping[str, str]) -> dict[str, Any]:
    if set(input_bindings) != set(REQUIRED_INPUT_BINDINGS):
        raise ValueError(f"attack input bindings must name exactly {sorted(REQUIRED_INPUT_BINDINGS)}")
    if any(not isinstance(value, str) or re.fullmatch(r"[a-f0-9]{64}", value) is None
           for value in input_bindings.values()):
        raise ValueError("attack input binding digest is invalid")
    collections = {
        "attacker_auxiliary": _collection_binding(auxiliary, "attacker_auxiliary"),
        "qualification": _collection_binding(qualification, "qualification"),
        "heldout": _collection_binding(heldout, "heldout"),
    }
    panel_ids = {value["panel_sha256"] for value in collections.values()}
    checkpoint_ids = {value["checkpoint_id"] for value in collections.values()}
    windows = [window for value in collections.values() for window in value["windows"]]
    if len(panel_ids) != 1 or input_bindings["panel"] not in panel_ids:
        raise ValueError("attack collections do not match the bound panel")
    if len(checkpoint_ids) != 1 or len(windows) != len(set(windows)):
        raise ValueError("attack collections reuse windows or checkpoints differ")
    return {"panel_sha256": panel_ids.pop(), "checkpoint_id": checkpoint_ids.pop(),
            "regions": {name: value["windows_sha256"] for name, value in collections.items()}}


def _arrays(events: Sequence[Mapping[str, Any]], channel: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if channel not in CHANNELS or not events:
        raise ValueError("unsupported channel or empty event collection")
    features, tokens, targets = [], [], []
    for event in events:
        tensor = event.get(channel)
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3 or not torch.isfinite(tensor).all():
            raise ValueError(f"event has no finite rank-3 channel {channel}")
        source, target = event.get("tokens"), event.get("targets")
        if not isinstance(source, torch.Tensor) or not isinstance(target, torch.Tensor):
            raise ValueError("attack events require token and target tensors")
        if tuple(source.shape) != tuple(target.shape) or tuple(tensor.shape[:2]) != tuple(source.shape):
            raise ValueError("channel and token panel shapes differ")
        features.append(tensor.detach().cpu().float().numpy())
        tokens.append(source.detach().cpu().numpy().astype(np.int64))
        targets.append(target.detach().cpu().numpy().astype(np.int64))
    return np.concatenate(features), np.concatenate(tokens), np.concatenate(targets)


def _ridge_fit(features: np.ndarray, labels: np.ndarray, ridge_lambda: float) -> RidgeModel:
    if features.ndim != 2 or labels.ndim not in (1, 2) or len(features) != len(labels):
        raise ValueError("invalid ridge training shapes")
    x_mean = features.mean(axis=0)
    x_scale = features.std(axis=0)
    x_scale = np.where(x_scale > 1e-12, x_scale, 1.0)
    y = labels[:, None] if labels.ndim == 1 else labels
    y_mean = y.mean(axis=0)
    centered = (features - x_mean) / x_scale
    weights = np.linalg.solve(centered.T @ centered + ridge_lambda * np.eye(features.shape[1]),
                              centered.T @ (y - y_mean))
    return RidgeModel(x_mean, x_scale, y_mean, weights)


def _ridge_predict(model: RidgeModel, features: np.ndarray) -> np.ndarray:
    return ((features - model.x_mean) / model.x_scale) @ model.weights + model.y_mean


def _sequence_permutation(count: int, seed: int, label: str) -> np.ndarray:
    if count < 2:
        raise ValueError("at least two panel sequences are required")
    return np.random.default_rng(derive_seed("r1-attack-permutation-v1", seed, label)).permutation(count)


def _punctuation_labels(tokens: np.ndarray) -> np.ndarray:
    return np.isin(tokens, list(PUNCTUATION_BYTES)).mean(axis=1).astype(np.float64)


def _target_data(features: np.ndarray, tokens: np.ndarray, targets: np.ndarray,
                 target: str) -> tuple[np.ndarray, np.ndarray, int | None]:
    if target == "punctuation_fraction":
        return features.mean(axis=1), _punctuation_labels(tokens), None
    if target == "next_byte_class":
        return features.reshape(-1, features.shape[-1]), next_byte_class(targets).reshape(-1), 5
    if target == "input_byte":
        return features.reshape(-1, features.shape[-1]), tokens.reshape(-1), 256
    raise ValueError(f"unsupported attack target: {target}")


def _permute_labels(labels: np.ndarray, sequences: int, seq_len: int,
                    seed: int, label: str) -> np.ndarray:
    order = _sequence_permutation(sequences, seed, label)
    if labels.ndim == 1 and len(labels) == sequences:
        return labels[order]
    return labels.reshape(sequences, seq_len)[order].reshape(-1)


def fit_attack(events: Sequence[Mapping[str, Any]], target: str, channel: str,
               specification: AttackSpecification, seed: int) -> AttackBank:
    features, tokens, next_tokens = _arrays(events, channel)
    x, labels, classes = _target_data(features, tokens, next_tokens, target)
    sequence_count, seq_len = tokens.shape
    null_labels = _permute_labels(labels, sequence_count, seq_len, seed, f"{target}:{channel}:fit")
    fit_labels = labels if classes is None else np.eye(classes, dtype=np.float64)[labels]
    null_fit = null_labels if classes is None else np.eye(classes, dtype=np.float64)[null_labels]
    return AttackBank(target, channel, _ridge_fit(x, fit_labels, specification.ridge_lambda),
                      _ridge_fit(x, null_fit, specification.ridge_lambda), classes, seed)


def _score_model(model: RidgeModel, features: np.ndarray, labels: np.ndarray,
                 classes: int | None) -> float:
    prediction = _ridge_predict(model, features)
    if classes is None:
        if np.std(prediction) == 0 or np.std(labels) == 0:
            return 0.0
        return float(np.corrcoef(prediction.reshape(-1), labels)[0, 1])
    return float(np.mean(prediction.argmax(axis=1) == labels))


def score_attack(bank: AttackBank, events: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    features, tokens, next_tokens = _arrays(events, bank.channel)
    x, labels, classes = _target_data(features, tokens, next_tokens, bank.target)
    if classes != bank.classes:
        raise ValueError("attack target class count differs from its fitted bank")
    score = _score_model(bank.probe, x, labels, classes)
    null_score = _score_model(bank.null, x, labels, classes)
    return {"score": score, "null_score": null_score, "advantage": score - null_score}


def _feature_scale(features: np.ndarray) -> float:
    scale = float(np.std(features))
    return scale if math_is_finite_positive(scale) else 1.0


def math_is_finite_positive(value: float) -> bool:
    return bool(np.isfinite(value) and value > 0)


def _codebook(classes: int, width: int, seed: int, label: str) -> np.ndarray:
    generator = np.random.default_rng(derive_seed("r1-positive-codebook-v1", seed, label))
    values = generator.normal(size=(classes, width))
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def inject_positive_control(events: Sequence[Mapping[str, Any]], target: str,
                            channel: str, dose: float, seed: int) -> list[dict[str, Any]]:
    features, tokens, next_tokens = _arrays(events, channel)
    x, labels, classes = _target_data(features, tokens, next_tokens, target)
    scale = _feature_scale(x)
    if classes is None:
        direction = _codebook(1, x.shape[-1], seed, target)[0]
        normalized = (labels - labels.mean()) / max(float(np.std(labels)), 1e-12)
        reshaped = features + dose * scale * normalized[:, None, None] * direction
    else:
        codebook = _codebook(classes, x.shape[-1], seed, target)
        injected = x + dose * scale * codebook[labels]
        reshaped = injected.reshape(features.shape)
    output: list[dict[str, Any]] = []
    offset = 0
    for event in events:
        count = event[channel].shape[0]
        copied = dict(event)
        copied[channel] = torch.from_numpy(reshaped[offset : offset + count]).float()
        output.append(copied)
        offset += count
    return output


def _broken_events(events: Sequence[Mapping[str, Any]], channel: str,
                   seed: int, label: str) -> list[dict[str, Any]]:
    order = _sequence_permutation(sum(event[channel].shape[0] for event in events), seed, label)
    values = torch.cat([event[channel] for event in events], dim=0)[torch.from_numpy(order)]
    output, offset = [], 0
    for event in events:
        count = event[channel].shape[0]
        copied = dict(event)
        copied[channel] = values[offset : offset + count]
        output.append(copied)
        offset += count
    return output


def qualify_attack(auxiliary: Sequence[Mapping[str, Any]], qualification: Sequence[Mapping[str, Any]],
                   target: str, channel: str, specification: AttackSpecification,
                   seed: int) -> dict[str, Any]:
    doses = []
    for dose in specification.positive_doses:
        fit_events = inject_positive_control(auxiliary, target, channel, dose, seed)
        eval_events = inject_positive_control(qualification, target, channel, dose, seed)
        result = score_attack(fit_attack(fit_events, target, channel, specification, seed), eval_events)
        doses.append({"dose": dose, **result})
    ordinary = fit_attack(auxiliary, target, channel, specification, seed)
    broken = score_attack(ordinary, _broken_events(qualification, channel, seed,
                                                   f"{target}:{channel}:pipeline"))
    floor = (specification.positive_floor_correlation if target == "punctuation_fraction"
             else specification.positive_floor_accuracy)
    selected_dose = max(specification.positive_doses)
    selected = next(value for value in doses if value["dose"] == selected_dose)
    qualified = (selected["advantage"] >= floor and
                 abs(broken["advantage"]) <= specification.null_band)
    return {"target": target, "channel": channel, "positive_control": doses,
            "qualification_dose": selected_dose, "positive_floor": floor,
            "null_pipeline": broken, "null_band": specification.null_band,
            "qualified": bool(qualified)}


def classify(score: Mapping[str, float], target: str,
             specification: AttackSpecification) -> Status:
    limit = (specification.privacy_limit_correlation if target == "punctuation_fraction"
             else specification.privacy_limit_accuracy)
    return "PASS" if score["advantage"] <= limit else "REJECTED"


def run_attack_suite(auxiliary: Sequence[Mapping[str, Any]], qualification: Sequence[Mapping[str, Any]],
                     heldout: Sequence[Mapping[str, Any]], specification: AttackSpecification,
                     seed: int, defense: str, input_bindings: Mapping[str, str],
                     attack_banks_sha256: str, *, scientific_evidence: bool = False) -> dict[str, Any]:
    panel_bindings = _validate_attack_inputs(auxiliary, qualification, heldout, input_bindings)
    if re.fullmatch(r"[a-f0-9]{64}", attack_banks_sha256) is None:
        raise ValueError("attack bank artifact digest is invalid")
    results: dict[str, Any] = {}
    for target in TARGETS:
        for channel in CHANNELS:
            key = f"{target}@{channel}@{defense}"
            qualification_result = qualify_attack(auxiliary, qualification, target, channel,
                                                  specification, seed)
            bank = fit_attack(auxiliary, target, channel, specification, seed)
            score = None
            status: Status = "UNDETERMINED"
            if qualification_result["qualified"]:
                score = score_attack(bank, heldout)
                status = classify(score, target, specification)
            results[key] = {"qualification": qualification_result, "score": score,
                            "status": status, "attack_bank_sha256": attack_bank_digest(bank)}
    return {"schema_version": "r1-attack-suite-1.0", "defense": defense, "seed": seed,
            "specification": asdict(specification), "input_bindings": dict(input_bindings),
            "panel_bindings": panel_bindings, "attack_banks_sha256": attack_banks_sha256,
            "results": results, "scientific_evidence": scientific_evidence,
            "status": "scoped_probe_suite_only" if scientific_evidence else "draft_cpu_validation_only"}


def save_attack_banks(auxiliary: Sequence[Mapping[str, Any]],
                      specification: AttackSpecification, seed: int,
                      defense: str, path: Path) -> dict[str, str]:
    payload: dict[str, Any] = {"schema_version": "r1-attack-banks-1.0", "banks": {}}
    digests = {}
    for target in TARGETS:
        for channel in CHANNELS:
            key = f"{target}@{channel}@{defense}"
            bank = fit_attack(auxiliary, target, channel, specification, seed)
            digests[key] = attack_bank_digest(bank)
            payload["banks"][key] = {
                "target": target, "channel": channel, "classes": bank.classes,
                "fit_seed": bank.fit_seed, "sha256": digests[key],
                "probe": _model_tensors(bank.probe), "null": _model_tensors(bank.null),
            }
    torch.save(payload, path)
    return digests


def _model_tensors(model: RidgeModel) -> dict[str, torch.Tensor]:
    return {"x_mean": torch.from_numpy(model.x_mean),
            "x_scale": torch.from_numpy(model.x_scale),
            "y_mean": torch.from_numpy(np.asarray(model.y_mean)),
            "weights": torch.from_numpy(model.weights)}


def attack_report_sha256(report: Mapping[str, Any]) -> str:
    return sha256_json(dict(report))
