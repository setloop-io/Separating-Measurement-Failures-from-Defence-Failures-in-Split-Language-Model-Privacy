"""Deterministic four-region data panels for prospective R1 experiments."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from jsonschema import validate

from .utils import derive_seed, sha256_bytes, sha256_json

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "experiments/schemas/r1_panel.schema.json"
REGIONS = ("model_train", "attacker_auxiliary", "qualification", "heldout")
FRACTIONS = (0.0, 0.80, 0.85, 0.90, 1.0)


@dataclass(frozen=True)
class RegionBounds:
    name: str
    start: int
    end: int


@dataclass(frozen=True)
class PanelWindow:
    region: str
    index: int
    start: int
    end: int
    input_sha256: str
    target_sha256: str


def corpus_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def region_bounds(length: int, seq_len: int) -> dict[str, RegionBounds]:
    if length <= 0 or seq_len <= 0:
        raise ValueError("corpus length and sequence length must be positive")
    cuts = [math.floor(length * value) for value in FRACTIONS]
    result: dict[str, RegionBounds] = {}
    for index, name in enumerate(REGIONS):
        start = cuts[index] + (0 if index == 0 else seq_len + 1)
        end = cuts[index + 1]
        if end - start < seq_len + 1:
            raise ValueError(f"region {name} is too small after its guard gap")
        result[name] = RegionBounds(name, start, end)
    return result


def _candidate_starts(bounds: RegionBounds, seq_len: int) -> list[int]:
    width = seq_len + 1
    return list(range(bounds.start, bounds.end - width + 1, width))


def select_windows(bounds: RegionBounds, seq_len: int, count: int,
                   seed: int) -> list[int]:
    candidates = _candidate_starts(bounds, seq_len)
    if count <= 0 or count > len(candidates):
        raise ValueError(f"requested {count} windows from {len(candidates)} available in {bounds.name}")
    order = np.random.default_rng(derive_seed("r1-panel-v1", seed, bounds.name)).permutation(len(candidates))
    return sorted(candidates[index] for index in order[:count])


def _window_record(data: bytes, region: str, index: int,
                   start: int, seq_len: int) -> PanelWindow:
    source = data[start : start + seq_len]
    target = data[start + 1 : start + seq_len + 1]
    if len(source) != seq_len or len(target) != seq_len:
        raise ValueError("window crosses its declared region")
    return PanelWindow(region, index, start, start + seq_len + 1,
                       sha256_bytes(source), sha256_bytes(target))


def build_panel_manifest(corpus_path: Path, seq_len: int,
                         counts: Mapping[str, int], seed: int,
                         corpus_label: str | None = None) -> dict[str, Any]:
    data = corpus_path.read_bytes()
    bounds = region_bounds(len(data), seq_len)
    if set(counts) != set(REGIONS):
        raise ValueError(f"panel counts must name exactly {REGIONS}")
    windows: list[PanelWindow] = []
    for name in REGIONS:
        starts = select_windows(bounds[name], seq_len, int(counts[name]), seed)
        windows.extend(_window_record(data, name, index, start, seq_len)
                       for index, start in enumerate(starts))
    payload = {"schema_version": "r1-panel-1.0",
               "corpus_path": corpus_label or str(corpus_path),
               "corpus_sha256": corpus_sha256(data), "corpus_bytes": len(data),
               "sequence_length": seq_len, "guard_bytes_per_boundary": seq_len + 1,
               "selection_seed": seed,
               "selection_algorithm": "nonoverlap-grid-permutation-v1",
               "requested_windows": {name: int(counts[name]) for name in REGIONS},
               "regions": {name: asdict(value) for name, value in bounds.items()},
               "windows": [asdict(window) for window in windows]}
    payload["panel_sha256"] = sha256_json(payload)
    validate(instance=payload, schema=json.loads(SCHEMA_PATH.read_text()))
    return payload


def verify_panel_manifest(corpus_path: Path, manifest: Mapping[str, Any]) -> None:
    validate(instance=dict(manifest), schema=json.loads(SCHEMA_PATH.read_text()))
    value = dict(manifest)
    expected_digest = value.pop("panel_sha256", None)
    if value.get("schema_version") != "r1-panel-1.0" or expected_digest != sha256_json(value):
        raise ValueError("panel manifest identity or digest is invalid")
    data = corpus_path.read_bytes()
    if value.get("corpus_sha256") != corpus_sha256(data) or value.get("corpus_bytes") != len(data):
        raise ValueError("panel corpus bytes differ")
    seq_len = int(value["sequence_length"])
    bounds = region_bounds(len(data), seq_len)
    expected_regions = {name: asdict(bound) for name, bound in bounds.items()}
    if value.get("regions") != expected_regions or value.get("guard_bytes_per_boundary") != seq_len + 1:
        raise ValueError("panel region bounds or guard differ")
    if value.get("selection_algorithm") != "nonoverlap-grid-permutation-v1":
        raise ValueError("panel selection algorithm differs")
    counts = value.get("requested_windows")
    if not isinstance(counts, dict) or set(counts) != set(REGIONS):
        raise ValueError("panel requested-window counts differ")
    expected_windows = []
    for name in REGIONS:
        starts = select_windows(bounds[name], seq_len, int(counts[name]), int(value["selection_seed"]))
        expected_windows.extend(asdict(_window_record(data, name, index, start, seq_len))
                                for index, start in enumerate(starts))
    if value.get("windows") != expected_windows:
        raise ValueError("panel windows differ from deterministic selection")


def save_panel_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(dict(manifest), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def panel_batch_records(corpus_path: Path, manifest: Mapping[str, Any],
                        region: str, batch_size: int) -> list[dict[str, Any]]:
    if region not in REGIONS or batch_size <= 0:
        raise ValueError("invalid panel region or batch size")
    verify_panel_manifest(corpus_path, manifest)
    data = corpus_path.read_bytes()
    seq_len = int(manifest["sequence_length"])
    windows = [PanelWindow(**item) for item in manifest["windows"] if item["region"] == region]
    batches = []
    for offset in range(0, len(windows), batch_size):
        selected = windows[offset : offset + batch_size]
        tokens = torch.tensor(
            [list(data[item.start : item.start + seq_len]) for item in selected], dtype=torch.long,
        )
        targets = torch.tensor(
            [list(data[item.start + 1 : item.start + seq_len + 1]) for item in selected],
            dtype=torch.long,
        )
        batches.append({"tokens": tokens, "targets": targets, "panel_region": region,
                        "panel_sha256": manifest["panel_sha256"],
                        "window_indices": [item.index for item in selected],
                        "window_input_sha256": [item.input_sha256 for item in selected],
                        "window_target_sha256": [item.target_sha256 for item in selected]})
    return batches


def panel_batches(corpus_path: Path, manifest: Mapping[str, Any],
                  region: str, batch_size: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [(item["tokens"], item["targets"])
            for item in panel_batch_records(corpus_path, manifest, region, batch_size)]
