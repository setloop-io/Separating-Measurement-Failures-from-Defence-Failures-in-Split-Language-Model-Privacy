"""Real reconstruction example and headline numbers for Paper A Figure 1.

Setting: GPT-2 split after six layers, public copy of the client's layers, candidate budget 32,
undefended training mode. Inputs are the committed campaign shards:
  results/validation-heldout-v3-r3-shard*/validation.json   cell index, metrics, prediction_sha256
  results/validation-heldout-v3-r3-shard*/{cell}.json        predicted token ids per document
  results/validation-heldout-v3-r3-shard*/input.json         true evaluation token ids
Every prediction file is checked against the sha256 recorded in its shard.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEPTH, KNOWLEDGE, MODE = 6, "public", "training"
ACT_ARM, JOINT_ARM = "forward_k32", "joint_true_k32"
SHOWN_VICTIM, SHOWN_ATTACK = 101, 17
TOKENS_PER_DOCUMENT = 32
RunKey = tuple[str, int, int]


@dataclass(frozen=True)
class PredictionFile:
    path: Path
    recorded_sha256: str
    actual_sha256: str
    predictions: dict[str, list[int]]


@dataclass(frozen=True)
class Example:
    doc_index: int
    doc_id: str
    truth: list[int]
    act: list[int]
    joint: list[int]
    act_file: PredictionFile
    joint_file: PredictionFile


@dataclass(frozen=True)
class Aggregate:
    """Pooled means over every document and run, in percent."""

    act_token: float
    joint_token: float
    act_exact: float
    joint_exact: float
    runs_per_attacker: int


def load_prediction_files(results: Path) -> dict[RunKey, PredictionFile]:
    """All activation-only and joint prediction files of the setting, sha256-checked."""
    out: dict[RunKey, PredictionFile] = {}
    for shard in sorted(results.glob("validation-heldout-v3-r3-shard*")):
        for c in json.loads((shard / "validation.json").read_text())["cells"]:
            if (c["depth"], c["knowledge"], c["mode"]) != (DEPTH, KNOWLEDGE, MODE):
                continue
            if c["arm"] not in (ACT_ARM, JOINT_ARM):
                continue
            path = shard / c["prediction_file"]
            raw = path.read_bytes()
            actual = hashlib.sha256(raw).hexdigest()
            if actual != c["prediction_sha256"]:
                raise ValueError(f"sha256 mismatch for {path}")
            pred = json.loads(raw)
            key = (c["arm"], c["victim_seed"], c["attack_seed"])
            if key in out:
                raise ValueError(f"duplicate prediction file {key}")
            out[key] = PredictionFile(path, c["prediction_sha256"], actual,
                                      dict(zip(pred["document_ids"], pred["predictions"], strict=True)))
    return out


def load_documents(results: Path) -> list[dict]:
    """Evaluation documents; every shard must carry the identical list."""
    shards = sorted(results.glob("validation-heldout-v3-r3-shard*"))
    docs = [json.loads((s / "input.json").read_text())["evaluation"] for s in shards]
    if any(d != docs[0] for d in docs[1:]):
        raise ValueError("evaluation documents differ between shards")
    return docs[0]


def correct_matrix(docs: list[dict], files: dict[RunKey, PredictionFile], arm: str) -> np.ndarray:
    """Runs x documents x positions boolean matrix of correct tokens."""
    keys = sorted(k for k in files if k[0] == arm)
    rows = []
    for key in keys:
        pred = files[key].predictions
        rows.append([[p == t for p, t in zip(pred[d["id"]], d["tokens"], strict=True)] for d in docs])
    out = np.asarray(rows, dtype=bool)
    if out.shape[2] != TOKENS_PER_DOCUMENT:
        raise ValueError(f"expected {TOKENS_PER_DOCUMENT} tokens per document, got {out.shape[2]}")
    return out


def aggregate(act: np.ndarray, joint: np.ndarray) -> Aggregate:
    runs = act.shape[0] * act.shape[1]
    if joint.shape != act.shape:
        raise ValueError("attackers were run on different grids")
    return Aggregate(100 * act.mean(), 100 * joint.mean(), 100 * act.all(axis=2).mean(),
                     100 * joint.all(axis=2).mean(), runs)


def closest_document(act: np.ndarray, joint: np.ndarray) -> int:
    """Document whose per-document mean token accuracies are closest to the pooled means.

    Distance is the sum of absolute differences for the two attackers; ties keep
    document order.
    """
    a, j = act.mean(axis=(0, 2)), joint.mean(axis=(0, 2))
    return int(np.argmin(np.abs(a - act.mean()) + np.abs(j - joint.mean())))


def load_example(results: Path) -> tuple[Example, Aggregate]:
    docs = load_documents(results)
    files = load_prediction_files(results)
    act, joint = correct_matrix(docs, files, ACT_ARM), correct_matrix(docs, files, JOINT_ARM)
    index = closest_document(act, joint)
    doc = docs[index]
    act_file = files[ACT_ARM, SHOWN_VICTIM, SHOWN_ATTACK]
    joint_file = files[JOINT_ARM, SHOWN_VICTIM, SHOWN_ATTACK]
    example = Example(index, doc["id"], doc["tokens"], act_file.predictions[doc["id"]],
                      joint_file.predictions[doc["id"]], act_file, joint_file)
    return example, aggregate(act, joint)
