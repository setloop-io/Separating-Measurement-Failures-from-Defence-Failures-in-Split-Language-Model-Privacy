"""Load the committed v3 held-out campaign shards and compute figure statistics.

Reads only results/validation-heldout-v3-r3-shard*/validation.json (the JSON
evidence tier committed to the repository). Levels are means over victim
seeds, attack seeds and documents; intervals reuse the paper's crossed
bootstrap (experiments.analyze_validation.crossed_interval) with the frozen
protocol's alpha, replicate count and seed.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from experiments.analyze_validation import crossed_interval  # noqa: E402

SHARD_GLOB = "validation-heldout-v3-r3-shard*"
SHARD_COUNT = 16
CellKey = tuple[int, int, str, int, str, str]


@dataclass(frozen=True)
class Estimate:
    """A level or paired difference in percent, with its 95% interval."""

    value: float
    lower: float
    upper: float


@dataclass(frozen=True)
class Campaign:
    cells: dict[CellKey, list[dict]]
    protocol: dict
    reports: list[dict]


def load_campaign(results: Path = ROOT / "results") -> Campaign:
    """Union all shard cells; fail if a shard is missing or a cell repeats."""
    dirs = sorted(results.glob(SHARD_GLOB))
    if len(dirs) != SHARD_COUNT:
        raise FileNotFoundError(f"expected {SHARD_COUNT} shards under {results}, found {len(dirs)}")
    cells: dict[CellKey, list[dict]] = {}
    reports = [json.loads((d / "validation.json").read_text()) for d in dirs]
    for report in reports:
        if report["status"] != "COMPLETE":
            raise ValueError("incomplete shard report")
        for c in report["cells"]:
            key = tuple(c[f] for f in ("victim_seed", "depth", "knowledge", "attack_seed", "mode", "arm"))
            if key in cells:
                raise ValueError(f"duplicate cell {key}")
            cells[key] = c["metrics"]
    return Campaign(cells, reports[0]["protocol"], reports)


def seed_by_document(camp: Campaign, arms: list[str], depth: int, knowledges: list[str],
                     field: str) -> np.ndarray:
    """Victim-seed x document matrix, averaged over attack seeds, arms and head levels."""
    p = camp.protocol
    rows = []
    for s in p["victim_seeds"]:
        per = [[m[field] for m in camp.cells[s, depth, k, a, "training", arm]]
               for arm in arms for k in knowledges for a in p["attack_seeds"]]
        rows.append(np.mean(np.asarray(per, dtype=float), axis=0))
    return np.asarray(rows)


def bootstrap(camp: Campaign, matrix: np.ndarray) -> Estimate:
    st = camp.protocol["statistics"]
    ci = crossed_interval(matrix, alpha=st["alpha"], replicates=st["bootstrap_replicates"],
                          seed=st["bootstrap_seed"])
    return Estimate(100 * ci["difference"], 100 * ci["lower"], 100 * ci["upper"])


def level(camp: Campaign, arms: list[str], depth: int, knowledges: list[str], field: str) -> Estimate:
    return bootstrap(camp, seed_by_document(camp, arms, depth, knowledges, field))


def paired_difference(camp: Campaign, left: str, right: str, depth: int, knowledge: str,
                      field: str) -> Estimate:
    a = seed_by_document(camp, [left], depth, [knowledge], field)
    b = seed_by_document(camp, [right], depth, [knowledge], field)
    return bootstrap(camp, a - b)


def point(camp: Campaign, arm: str, depth: int, knowledge: str, field: str = "token_accuracy") -> float:
    """Descriptive mean in percent, no interval (used by the heatmap)."""
    return float(100 * seed_by_document(camp, [arm], depth, [knowledge], field).mean())


def perplexity_increase(camp: Campaign) -> dict[tuple[str, int], float]:
    """Geometric-mean held-out perplexity of each defence and depth vs the undefended final state, in %."""
    undefended = {v["seed"]: v["evaluation_nll"] for r in camp.reports for v in r["victims"]}
    if any(undefended[v["seed"]] != v["evaluation_nll"] for r in camp.reports for v in r["victims"]):
        raise ValueError("undefended victim differs across shards")
    defended: dict[tuple[str, int, int], float] = {}
    for r in camp.reports:
        for v in r["defended_victims"]:
            ce = v["utility"]["utility_after"]["cross_entropy"]
            key = (v["defense"], v["depth"], v["seed"])
            if key in defended and defended[key] != ce:
                raise ValueError(f"defended victim {key} differs across shards")
            defended[key] = ce
    base = np.mean(list(undefended.values()))
    out: dict[tuple[str, int], float] = {}
    for defence, depth in sorted({(d, L) for d, L, _ in defended}):
        ce = np.mean([defended[defence, depth, s] for s in undefended])
        out[defence, depth] = 100 * (float(np.exp(ce - base)) - 1)
    return out
