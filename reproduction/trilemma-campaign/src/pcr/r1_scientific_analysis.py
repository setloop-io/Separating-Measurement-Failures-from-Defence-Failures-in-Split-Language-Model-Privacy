"""R1 scientific-stage analysis: calibration qualification and exploratory classification.

Verification-first aggregation for the two admitted scientific stages defined in
docs/PREREGISTRATION_R1.md. The calibration analyzer aggregates attack
qualification outcomes across calibration seeds into a frozen attack
specification proposal; the exploratory analyzer classifies every matrix cell
under the protocol's three-valued logic (Rejected / Scoped candidate pass /
Undetermined) against that frozen specification. This module trains nothing and
admits nothing: every consumed run is rebound to its launch record, manifest,
resolved configuration, metrics and attack artifacts before any number is read.

Input contract (produced by the scientific runner; this module depends only on
the contract, never on runner internals). Execution root
``outputs/<study>/<stage>--<execution_id>/`` holds ``launch.json`` (binding
``panel_sha256``, a ``runs`` list of resolved configurations, and for the
exploratory stage the frozen attack specification hash), ``execution_summary.json``
(``completed`` list, ``failure``, ``wall_seconds``), ``execution_records/<run_id>.
{start,end,failure}.json`` and ``runs/<run_id>/`` with ``config.resolved.json``,
``metrics_summary.json`` (``run_id``/``mode``/``seed``/``final_eval_loss``) and,
for trained split runs, ``attack_suite.json`` (the ``run_attack_suite`` report)
plus its hash-bound ``attack_banks.pt``. Completed end records carry ``run_id``,
``status: completed``, a ``manifest`` path and ``manifest_sha256``; the manifest
binds the run identity and (when present) the resolved configuration digest.

Confidence limits use a two-sided 95% Student-t interval over per-seed paired
contrasts (df = n - 1); each gate consumes the relevant side (utility: upper;
gain/removal: lower). The two-sided choice matches the project's seed-level
convention in scripts/stats_utils.py (paired_t_test reports a two-sided 95% CI)
and is the conservative reading of the draft's upper/lower 95% confidence
limits. The t distribution is implemented locally via the regularized
incomplete beta function (Numerical Recipes continued fraction) — the same
algorithm validated against textbook reference values in
tests/test_confirmatory.py — so the scientific package has no scripts/
dependency and no scipy requirement.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping, Sequence

from .artifacts import validate_attack_report
from .r1_evidence_review import verify_run_artifacts, verify_stage_bindings
from .r1 import now, read_json, safe_name, write_new
from .r1_attacks import CHANNELS, TARGETS
from .utils import sha256_file, sha256_json

CALIBRATION_ANALYSIS_SCHEMA = "r1-calibration-analysis-1.0"
EXPLORATORY_ANALYSIS_SCHEMA = "r1-exploratory-analysis-1.0"
FROZEN_SPEC_SCHEMA = "r1-frozen-attack-spec-1.0"

MODES = ("trained", "frozen_random", "removed_remote")
ALL_LOCAL_CELL = "all-local"
ALL_LOCAL_MODE = "all_local"

UTILITY_MARGIN = 0.10
UTILITY_UPPER_LIMIT = 0.12
UTILITY_CI_SLACK = UTILITY_UPPER_LIMIT - UTILITY_MARGIN
GAIN_MARGIN = 0.01
GAIN_LOWER_LIMIT = 0.0
REMOVAL_MARGIN = 0.0
REMOVAL_LOWER_LIMIT = 0.0
UTILITY_SENSITIVITY_MARGINS = (0.05, 0.10, 0.20)
GAIN_SENSITIVITY_MARGINS = (0.005, 0.01, 0.02)

ATTACK_SPEC_LAUNCH_KEYS = (
    "attack_specification_sha256",
    "frozen_attack_specification_sha256",
    "attack_spec_sha256",
)

SCOPE = ("Classification follows the frozen R1 protocol over the declared finite cell set and the "
         "declared qualified attack suite. It establishes no universal trilemma, no hardware "
         "necessity, and no complete-transcript privacy claim.")

_METHOD = {
    "inferential_unit": "independent training seed",
    "interval": ("two-sided 95% Student-t interval over per-seed paired contrasts (df = n-1); "
                 "each gate consumes the relevant side (utility: upper; gain/removal: lower)"),
    "interval_choice": ("two-sided 95% matches the project's seed-level convention "
                        "(scripts/stats_utils.py paired_t_test) and is the conservative reading of "
                        "the draft's upper/lower 95% confidence limits"),
    "t_distribution": ("local implementation via the regularized incomplete beta function (Numerical "
                       "Recipes continued fraction), the algorithm validated against textbook "
                       "reference values in tests/test_confirmatory.py"),
    "privacy_rule": ("per required combination qualified in the frozen attack specification: "
                     "maximum per-seed advantage across the cell's seeds, compared to that "
                     "combination's privacy limit"),
    "incomplete_cells": ("gates are never computed on incomplete seed sets (no outcome-based "
                         "deletion); incomplete cells cannot pass and mean-based gates cannot "
                         "reject them, but a qualified privacy breach on available seeds still "
                         "rejects"),
}


# --------------------------------------------------------------------------- #
# Student-t distribution (ported from the validated scripts/stats_utils.py)
# --------------------------------------------------------------------------- #
_FLOORED = 1e-300


def _betacf(a: float, b: float, x: float, max_iter: int = 500, eps: float = 3e-14) -> float:
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _FLOORED:
        d = _FLOORED
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _FLOORED:
            d = _FLOORED
        c = 1.0 + aa / c
        if abs(c) < _FLOORED:
            c = _FLOORED
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _FLOORED:
            d = _FLOORED
        c = 1.0 + aa / c
        if abs(c) < _FLOORED:
            c = _FLOORED
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    if not 0.0 <= x <= 1.0:
        raise ValueError(f"x must be in [0, 1], got {x}")
    if a <= 0 or b <= 0:
        raise ValueError(f"a, b must be positive, got {a}, {b}")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0
    ln_bt = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log1p(-x))
    bt = math.exp(ln_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _t_cdf(t: float, df: float) -> float:
    if df <= 0:
        raise ValueError(f"df must be positive, got {df}")
    x = df / (df + t * t)
    ib = _regularized_incomplete_beta(x, df / 2.0, 0.5)
    return 1.0 - 0.5 * ib if t >= 0 else 0.5 * ib


def _t_ppf(p: float, df: float) -> float:
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")
    lo, hi = -1.0, 1.0
    while _t_cdf(lo, df) > p:
        lo *= 2.0
    while _t_cdf(hi, df) < p:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _seed_interval(values: Sequence[float], level: float = 0.95) -> dict[str, Any]:
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "sd": None, "se": None,
                "lower": None, "upper": None, "df": None, "level": level}
    mu = mean(values)
    if n < 2:
        return {"n": n, "mean": mu, "sd": None, "se": None,
                "lower": None, "upper": None, "df": n - 1, "level": level}
    sd = stdev(values)
    se = sd / math.sqrt(n)
    half = _t_ppf(1.0 - (1.0 - level) / 2.0, n - 1) * se
    return {"n": n, "mean": mu, "sd": sd, "se": se,
            "lower": mu - half, "upper": mu + half, "df": n - 1, "level": level}


# --------------------------------------------------------------------------- #
# identity helpers
# --------------------------------------------------------------------------- #
def _target_kind(target: str) -> str:
    return "correlation" if target == "punctuation_fraction" else "accuracy"


def _defense_label(split: Mapping[str, Any]) -> str:
    return "none" if split.get("grad_defense") is None else "gradient-defense"


def _normalized(config: Mapping[str, Any]) -> Any:
    stripped = {key: value for key, value in config.items() if key != "output_dir"}
    return json.loads(json.dumps(stripped, sort_keys=True))


def _selected_dose(qualification: Mapping[str, Any]) -> Mapping[str, Any]:
    return next(dose for dose in qualification["positive_control"]
                if dose["dose"] == qualification["qualification_dose"])


def _load_launch(root: Path, stage: str) -> dict[str, Any]:
    path = root / "launch.json"
    if not path.is_file():
        raise ValueError(f"{stage} launch record is missing")
    launch = read_json(path)
    if launch.get("stage") not in (None, stage):
        raise ValueError("launch stage differs from the analysis stage")
    if not isinstance(launch.get("panel_sha256"), str):
        raise ValueError("launch does not bind a panel")
    verify_stage_bindings(root, launch)
    return launch


def _planned_structure(planned: Sequence[Mapping[str, Any]], stage: str) -> dict[str, Any]:
    cells: list[str] = []
    seeds: set[int] = set()
    keys: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    for item in planned:
        r1 = item.get("r1", {})
        if r1.get("stage") != stage:
            raise ValueError(f"planned run {item.get('run_id')} is not a {stage} run")
        cell_id, seed, mode = r1.get("cell_id"), item.get("seed"), item.get("mode")
        if not isinstance(cell_id, str) or type(seed) is not int or not isinstance(mode, str):
            raise ValueError("planned run identity is invalid")
        key = (cell_id, seed, mode)
        if key in keys:
            raise ValueError(f"duplicate planned run for {key}")
        keys[key] = item
        if cell_id != ALL_LOCAL_CELL and cell_id not in cells:
            cells.append(cell_id)
        seeds.add(seed)
    ordered_seeds = sorted(seeds)
    if not cells or not ordered_seeds:
        raise ValueError(f"planned {stage} matrix is empty")
    if stage == "calibration":
        expected = {(cell, seed, "trained") for cell in cells for seed in ordered_seeds}
    else:
        expected = {(cell, seed, mode) for cell in cells for seed in ordered_seeds for mode in MODES}
        expected |= {(ALL_LOCAL_CELL, seed, ALL_LOCAL_MODE) for seed in ordered_seeds}
    if set(keys) != expected:
        raise ValueError(f"planned {stage} runs are not the declared complete matrix")
    if stage == "calibration" and any(item.get("split") is None for item in planned):
        raise ValueError("calibration runs must all be trained split runs")
    if stage == "exploratory":
        for seed in ordered_seeds:
            if keys[(ALL_LOCAL_CELL, seed, ALL_LOCAL_MODE)].get("split") is not None:
                raise ValueError("all-local control must not carry a split")
    return {"cells": cells, "seeds": ordered_seeds, "planned_by_key": keys}


# --------------------------------------------------------------------------- #
# execution accounting (completeness + evidence re-binding)
# --------------------------------------------------------------------------- #
def _attack_evidence(run_dir: Path, run_id: str, resolved: Mapping[str, Any],
                     launch: Mapping[str, Any]) -> dict[str, Any] | None:
    if resolved["mode"] != "trained" or resolved.get("split") is None:
        return None
    path = run_dir / "attack_suite.json"
    banks = run_dir / "attack_banks.pt"
    if not path.is_file() or not banks.is_file():
        raise ValueError(f"trained run {run_id} is missing its attack suite or attack banks")
    validate_attack_report(path)
    report = read_json(path)
    if report["seed"] != resolved["seed"] or report["defense"] != _defense_label(resolved["split"]):
        raise ValueError(f"attack report identity differs from run {run_id}")
    if report["attack_banks_sha256"] != sha256_file(str(banks)):
        raise ValueError(f"attack banks differ from the scored suite for run {run_id}")
    if report["input_bindings"]["panel"] != launch["panel_sha256"]:
        raise ValueError(f"attack report panel differs from the launch binding for run {run_id}")
    for key, result in report["results"].items():
        qualification = result["qualification"]
        selected = _selected_dose(qualification)
        expected = (selected["advantage"] >= qualification["positive_floor"] and
                    abs(qualification["null_pipeline"]["advantage"]) <= qualification["null_band"])
        if expected != qualification["qualified"]:
            raise ValueError(f"attack qualification is inconsistent with its thresholds: {run_id} {key}")
    return report


def _account_runs(root: Path, planned: Sequence[Mapping[str, Any]],
                  launch: Mapping[str, Any]) -> dict[str, Any]:
    launch_runs = launch.get("runs")
    if not isinstance(launch_runs, list) or not launch_runs:
        raise ValueError("launch does not bind a planned run list")
    bound: dict[str, Mapping[str, Any]] = {}
    for item in launch_runs:
        run_id = safe_name(str(item.get("run_id")))
        if run_id in bound:
            raise ValueError("launch binds a duplicate run identity")
        bound[run_id] = item
    planned_ids = [safe_name(str(item.get("run_id"))) for item in planned]
    if len(planned_ids) != len(set(planned_ids)) or set(planned_ids) != set(bound):
        raise ValueError("planned runs differ from the launch binding")
    for item in planned:
        if _normalized(item) != _normalized(bound[item["run_id"]]):
            raise ValueError(f"planned configuration differs from the launch binding: {item['run_id']}")
    summary_path = root / "execution_summary.json"
    if not summary_path.is_file():
        raise ValueError("execution summary is missing")
    summary = read_json(summary_path)
    records = root / "execution_records"
    inputs = [{"path": "launch.json", "sha256": sha256_file(str(root / "launch.json"))},
              {"path": "execution_summary.json", "sha256": sha256_file(str(summary_path))}]
    rows, failures, interrupted, not_started = [], [], [], []
    for item in planned:
        run_id = item["run_id"]
        start_path = records / f"{run_id}.start.json"
        end_path = records / f"{run_id}.end.json"
        failure_path = records / f"{run_id}.failure.json"
        if failure_path.is_file():
            failures.append(read_json(failure_path))
            inputs.append({"path": f"execution_records/{failure_path.name}",
                           "sha256": sha256_file(str(failure_path))})
            continue
        if not start_path.is_file():
            not_started.append(run_id)
            continue
        if not end_path.is_file():
            interrupted.append(run_id)
            inputs.append({"path": f"execution_records/{start_path.name}",
                           "sha256": sha256_file(str(start_path))})
            continue
        end = read_json(end_path)
        if end.get("run_id") != run_id or end.get("status") != "completed":
            raise ValueError(f"end record identity differs: {run_id}")
        if "start_sha256" in end and end["start_sha256"] != sha256_file(str(start_path)):
            raise ValueError(f"start/end binding differs: {run_id}")
        manifest_value, manifest_sha = end.get("manifest"), end.get("manifest_sha256")
        if not isinstance(manifest_value, str) or not isinstance(manifest_sha, str):
            raise ValueError(f"completed run has no manifest binding: {run_id}")
        manifest_path = root / "manifests" / f"{run_id}.json"
        # The execution root may be relocated; the original record is retained and
        # the local run-id manifest must match its recorded digest.
        if Path(manifest_value).name != f"{run_id}.json":
            raise ValueError(f"manifest identity differs: {run_id}")
        if not manifest_path.is_file() or sha256_file(str(manifest_path)) != manifest_sha:
            raise ValueError(f"manifest binding differs: {run_id}")
        manifest_record = read_json(manifest_path)
        run_dir = root / "runs" / run_id
        config_path = run_dir / "config.resolved.json"
        if not config_path.is_file():
            raise ValueError(f"resolved configuration is missing: {run_id}")
        resolved = read_json(config_path)
        if _normalized(resolved) != _normalized(bound[run_id]):
            raise ValueError(f"resolved configuration differs from the launch binding: {run_id}")
        if manifest_record.get("run_id") not in (None, run_id):
            raise ValueError(f"manifest identity differs: {run_id}")
        if manifest_record.get("status") not in (None, "completed"):
            raise ValueError(f"manifest status differs: {run_id}")
        if manifest_record.get("resolved_config_sha256") not in (None, sha256_json(resolved)):
            raise ValueError(f"manifest configuration binding differs: {run_id}")
        verification = verify_run_artifacts(root, run_dir, manifest_record, resolved, launch)
        metrics_path = run_dir / "metrics_summary.json"
        if not metrics_path.is_file():
            raise ValueError(f"metrics are missing: {run_id}")
        metrics = read_json(metrics_path)
        if (metrics.get("run_id") != run_id or metrics.get("mode") != resolved["mode"]
                or metrics.get("seed") != resolved["seed"]):
            raise ValueError(f"metrics identity differs: {run_id}")
        loss = metrics.get("final_eval_loss")
        if not isinstance(loss, (int, float)) or isinstance(loss, bool) or not math.isfinite(loss):
            raise ValueError(f"nonfinite held-out loss: {run_id}")
        attack_report = _attack_evidence(run_dir, run_id, resolved, launch)
        split = resolved.get("split")
        rows.append({"run_id": run_id, "cell_id": resolved.get("r1", {}).get("cell_id"),
                     "seed": resolved["seed"], "mode": resolved["mode"], "split": split,
                     "defense": None if split is None else _defense_label(split),
                     "loss": float(loss), "attack_report": attack_report,
                     "artifact_verification": verification})
        for path in (start_path, end_path, manifest_path, config_path, metrics_path):
            inputs.append({"path": str(path.relative_to(root)), "sha256": sha256_file(str(path))})
        if attack_report is not None:
            for name in ("attack_suite.json", "attack_banks.pt"):
                artifact = run_dir / name
                inputs.append({"path": str(artifact.relative_to(root)),
                               "sha256": sha256_file(str(artifact))})
    completed_ids = {row["run_id"] for row in rows}
    if isinstance(summary.get("completed"), list) and set(summary["completed"]) != completed_ids:
        raise ValueError("execution summary completed list differs from the execution records")
    if (summary.get("failure") is not None) != bool(failures):
        raise ValueError("execution summary failure state differs from the execution records")
    return {"rows": rows, "failures": failures, "interrupted_runs": interrupted,
            "not_started_runs": not_started, "planned_count": len(planned_ids),
            "completed_count": len(rows),
            "complete": not failures and not interrupted and not not_started,
            "verification": {"runs": len(rows),
                             "artifact_count": sum(row["artifact_verification"]["artifact_count"] for row in rows),
                             "all_manifest_hashes_verified": True,
                             "scope": "retained bytes and provenance; no numerical replay"},
            "input_bindings": inputs}


# --------------------------------------------------------------------------- #
# calibration: strict qualification aggregation + frozen specification proposal
# --------------------------------------------------------------------------- #
def _frozen_spec_proposal(root: Path, structure: Mapping[str, Any],
                          accounting: Mapping[str, Any]) -> dict[str, Any]:
    planned_by_key = structure["planned_by_key"]
    seeds = structure["seeds"]
    rows_by_id = {row["run_id"]: row for row in accounting["rows"]}
    specifications = [row["attack_report"]["specification"] for row in accounting["rows"]]
    if specifications and any(spec != specifications[0] for spec in specifications):
        raise ValueError("attack specifications differ across calibration runs")
    thresholds = dict(specifications[0]) if specifications else None
    defenses = sorted({_defense_label(item["split"]) for item in planned_by_key.values()})
    combinations: dict[str, Any] = {}
    for defense in defenses:
        contributing = [item for item in planned_by_key.values()
                        if _defense_label(item["split"]) == defense]
        for target in TARGETS:
            for channel in CHANNELS:
                key = f"{target}@{channel}@{defense}"
                per_seed: dict[str, Any] = {}
                for seed in seeds:
                    seed_runs, missing = [], []
                    for item in contributing:
                        if item["seed"] != seed:
                            continue
                        row = rows_by_id.get(item["run_id"])
                        if row is None:
                            missing.append(item["run_id"])
                            continue
                        qualification = row["attack_report"]["results"][key]["qualification"]
                        selected = _selected_dose(qualification)
                        seed_runs.append({
                            "run_id": item["run_id"], "cell_id": item["r1"]["cell_id"],
                            "qualified": qualification["qualified"],
                            "selected_dose_advantage": selected["advantage"],
                            "null_advantage": qualification["null_pipeline"]["advantage"],
                            "dose_curve": [{"dose": dose["dose"], "advantage": dose["advantage"]}
                                           for dose in qualification["positive_control"]]})
                    per_seed[str(seed)] = {
                        "qualified": (not missing and bool(seed_runs)
                                      and all(run["qualified"] for run in seed_runs)),
                        "missing_run_ids": missing, "runs": seed_runs}
                qualified = all(entry["qualified"] for entry in per_seed.values())
                all_runs = [run for entry in per_seed.values() for run in entry["runs"]]
                floor = (None if thresholds is None else
                         thresholds["positive_floor_correlation" if target == "punctuation_fraction"
                                    else "positive_floor_accuracy"])
                band = None if thresholds is None else thresholds["null_band"]
                min_selected = min((run["selected_dose_advantage"] for run in all_runs), default=None)
                max_null = max((abs(run["null_advantage"]) for run in all_runs), default=None)
                combinations[key] = {
                    "target": target, "channel": channel, "defense": defense,
                    "qualified": qualified, "per_seed": per_seed,
                    "diagnostics": {
                        "positive_floor": floor, "null_band": band,
                        "min_selected_dose_advantage": min_selected,
                        "max_abs_null_advantage": max_null,
                        "positive_floor_margin": (None if min_selected is None or floor is None
                                                  else min_selected - floor),
                        "null_band_margin": (None if max_null is None or band is None
                                             else max_null - band),
                        "evidence_complete": len(all_runs) == len(contributing)}}
    proposal = {
        "schema_version": FROZEN_SPEC_SCHEMA, "created_at_utc": now(), "stage": "calibration",
        "status": ("proposal_ready_for_freeze_review" if accounting["complete"]
                   else "invalid_incomplete_calibration"),
        "qualification_rule": ("a combination is qualified only if every planned run sharing its "
                               "defense label is complete and qualified in every calibration seed "
                               "(strict, fail-closed); one qualified target never qualifies another"),
        "thresholds": thresholds,
        "privacy_limits": (None if thresholds is None else {
            "correlation": thresholds["privacy_limit_correlation"],
            "accuracy": thresholds["privacy_limit_accuracy"]}),
        "combinations": combinations,
        "amendment": _amendment(combinations, thresholds),
        "execution": {"planned_count": accounting["planned_count"],
                      "completed_count": accounting["completed_count"],
                      "seeds": seeds, "defenses": defenses,
                      "failures": accounting["failures"],
                      "interrupted_runs": accounting["interrupted_runs"],
                      "not_started_runs": accounting["not_started_runs"],
                      "launch_sha256": sha256_file(str(root / "launch.json"))},
        "input_bindings": accounting["input_bindings"],
        "scientific_evidence": True, "scope": SCOPE}
    return proposal


def _amendment(combinations: Mapping[str, Any],
               thresholds: Mapping[str, Any] | None) -> dict[str, Any]:
    unqualified = [key for key, combo in combinations.items() if not combo["qualified"]]
    per_combination = {}
    for key, combo in combinations.items():
        diag = combo["diagnostics"]
        per_combination[key] = {
            "qualified": combo["qualified"], "target_kind": _target_kind(combo["target"]),
            "floor_criterion_holds_if_floor_at_or_below": diag["min_selected_dose_advantage"],
            "null_criterion_holds_if_band_at_or_above": diag["max_abs_null_advantage"],
            "evidence_complete": diag["evidence_complete"]}
    knobs: dict[str, Any] = {}
    if thresholds is not None:
        for knob, kind in (("positive_floor_correlation", "correlation"),
                           ("positive_floor_accuracy", "accuracy")):
            current = thresholds[knob]
            failing = [combo["diagnostics"] for combo in combinations.values()
                       if not combo["qualified"] and _target_kind(combo["target"]) == kind
                       and combo["diagnostics"]["evidence_complete"]
                       and combo["diagnostics"]["null_band_margin"] is not None
                       and combo["diagnostics"]["null_band_margin"] <= 0]
            passing = [combo["diagnostics"]["min_selected_dose_advantage"]
                       for combo in combinations.values()
                       if combo["qualified"] and _target_kind(combo["target"]) == kind]
            knobs[knob] = {
                "current": current,
                "qualify_more_if_at_or_below": max(
                    (diag["min_selected_dose_advantage"] for diag in failing
                     if diag["min_selected_dose_advantage"] is not None
                     and diag["min_selected_dose_advantage"] < current), default=None),
                "qualify_fewer_if_above": min(passing, default=None)}
        current = thresholds["null_band"]
        failing = [combo["diagnostics"] for combo in combinations.values()
                   if not combo["qualified"] and combo["diagnostics"]["evidence_complete"]
                   and combo["diagnostics"]["positive_floor_margin"] is not None
                   and combo["diagnostics"]["positive_floor_margin"] >= 0]
        passing = [combo["diagnostics"]["max_abs_null_advantage"]
                   for combo in combinations.values() if combo["qualified"]]
        knobs["null_band"] = {
            "current": current,
            "qualify_more_if_at_or_above": min(
                (diag["max_abs_null_advantage"] for diag in failing
                 if diag["max_abs_null_advantage"] is not None
                 and diag["max_abs_null_advantage"] > current), default=None),
            "qualify_fewer_if_below": max(passing, default=None)}
    return {
        "policy": ("Changing thresholds after exploratory scores are inspected is prohibited. Any "
                   "amendment must be recorded and frozen before the affected exploratory evidence "
                   "is read; the values below are the complete freeze-time statement of what any "
                   "single-knob threshold change would do."),
        "required": any(value is not None for knob in knobs.values()
                        for name, value in knob.items() if name.startswith("qualify_more")),
        "unqualified_combinations": unqualified,
        "unqualifiable_by_thresholds_alone": [
            key for key in unqualified if not combinations[key]["diagnostics"]["evidence_complete"]],
        "knobs": knobs,
        "per_combination": per_combination}


# --------------------------------------------------------------------------- #
# exploratory: three-valued cell classification
# --------------------------------------------------------------------------- #
def _privacy_combination(cell_id: str, target: str, channel: str, defense: str,
                         seeds: Sequence[int],
                         rows_by_key: Mapping[tuple[str, int, str], Mapping[str, Any]],
                         frozen_combinations: Mapping[str, Any],
                         privacy_limits: Mapping[str, float]) -> dict[str, Any]:
    key = f"{target}@{channel}@{defense}"
    limit = privacy_limits[_target_kind(target)]
    spec_combo = frozen_combinations.get(key)
    qualified = isinstance(spec_combo, Mapping) and spec_combo.get("qualified") is True
    entry: dict[str, Any] = {"target": target, "channel": channel, "limit": limit,
                             "qualified_in_frozen_specification": qualified}
    if not qualified:
        entry["verdict"] = "UNDETERMINED"
        entry["reason"] = ("unqualified_in_frozen_specification" if spec_combo is not None
                           else "absent_from_frozen_specification")
        return entry
    per_seed: dict[str, Any] = {}
    for seed in seeds:
        row = rows_by_key.get((cell_id, seed, "trained"))
        seed_entry: dict[str, Any] = {"advantage": None, "note": None, "run_qualified": None}
        if row is None:
            seed_entry["note"] = "run_missing"
        else:
            result = row["attack_report"]["results"][key]
            seed_entry["run_qualified"] = result["qualification"]["qualified"]
            if result["score"] is None:
                seed_entry["note"] = "score_absent_run_qualification_failed"
            else:
                seed_entry["advantage"] = result["score"]["advantage"]
        per_seed[str(seed)] = seed_entry
    available = {seed: value["advantage"] for seed, value in per_seed.items()
                 if value["advantage"] is not None}
    worst = max(available.values(), default=None)
    entry["per_seed"] = per_seed
    entry["worst_advantage"] = worst
    entry["worst_seed"] = None if worst is None else max(available, key=available.get)
    if worst is not None and worst > limit:
        entry["verdict"] = "BREACH"
    elif len(available) == len(seeds):
        entry["verdict"] = "PASS"
    else:
        entry["verdict"] = "UNDETERMINED"
        entry["reason"] = "missing_seed_evidence"
    return entry


def _sensitivity(utility_interval: Mapping[str, Any] | None,
                 gain_interval: Mapping[str, Any] | None, complete: bool) -> dict[str, Any]:
    outcomes = []
    for margin in UTILITY_SENSITIVITY_MARGINS:
        for gain_margin in GAIN_SENSITIVITY_MARGINS:
            outcome: dict[str, Any] = {"utility_margin": margin, "gain_margin": gain_margin}
            if complete:
                utility_pass = (utility_interval["mean"] <= margin and
                                utility_interval["upper"] <= margin + UTILITY_CI_SLACK)
                gain_pass = (gain_interval["mean"] >= gain_margin and
                             gain_interval["lower"] > GAIN_LOWER_LIMIT)
                outcome.update({"utility_pass": bool(utility_pass), "gain_pass": bool(gain_pass),
                                "joint_utility_gain_pass": bool(utility_pass and gain_pass)})
            else:
                outcome.update({"utility_pass": None, "gain_pass": None,
                                "joint_utility_gain_pass": None})
            outcomes.append(outcome)
    return {"utility_margins": list(UTILITY_SENSITIVITY_MARGINS),
            "gain_margins": list(GAIN_SENSITIVITY_MARGINS),
            "ci_slack_nats": UTILITY_CI_SLACK, "outcomes": outcomes,
            "note": ("the draft 0.02-nat utility CI slack is held fixed across margins, so margin "
                     "0.10 reproduces the draft gate; the gain lower-limit condition (> 0) is "
                     "unchanged; privacy is not part of this sensitivity grid")}


def _classify_cell(cell_id: str, defense: str, seeds: Sequence[int],
                   planned_by_key: Mapping[tuple[str, int, str], Mapping[str, Any]],
                   rows_by_key: Mapping[tuple[str, int, str], Mapping[str, Any]],
                   frozen_combinations: Mapping[str, Any],
                   privacy_limits: Mapping[str, float]) -> dict[str, Any]:
    seed_records = []
    for seed in seeds:
        roles = {"all_local": (ALL_LOCAL_CELL, seed, ALL_LOCAL_MODE),
                 "trained": (cell_id, seed, "trained"),
                 "frozen_random": (cell_id, seed, "frozen_random"),
                 "removed_remote": (cell_id, seed, "removed_remote")}
        losses, missing = {}, []
        for role, key in roles.items():
            row = rows_by_key.get(key)
            losses[role] = None if row is None else row["loss"]
            if row is None:
                missing.append(planned_by_key[key]["run_id"])
        complete = all(value is not None for value in losses.values())
        seed_records.append({
            "seed": seed, "complete": complete, "losses": losses, "missing_run_ids": missing,
            "replacement_penalty": None if not complete else losses["trained"] - losses["all_local"],
            "gain_against_frozen": None if not complete else losses["frozen_random"] - losses["trained"],
            "gain_against_removed": None if not complete else losses["removed_remote"] - losses["trained"]})
    complete = all(record["complete"] for record in seed_records)
    utility_interval = gain_interval = None
    if complete:
        utility_interval = _seed_interval([r["replacement_penalty"] for r in seed_records])
        gain_interval = _seed_interval([r["gain_against_frozen"] for r in seed_records])
        removal_interval = _seed_interval([r["gain_against_removed"] for r in seed_records])
        gates: dict[str, Any] = {
            "computable": True,
            "utility": {**utility_interval, "mean_margin": UTILITY_MARGIN,
                        "upper_limit": UTILITY_UPPER_LIMIT,
                        "pass": bool(utility_interval["mean"] <= UTILITY_MARGIN and
                                     utility_interval["upper"] <= UTILITY_UPPER_LIMIT)},
            "gain": {**gain_interval, "mean_margin": GAIN_MARGIN,
                     "lower_limit_exclusive": GAIN_LOWER_LIMIT,
                     "pass": bool(gain_interval["mean"] >= GAIN_MARGIN and
                                  gain_interval["lower"] > GAIN_LOWER_LIMIT)},
            "benefit_over_removal": {
                **removal_interval, "mean_margin": REMOVAL_MARGIN,
                "lower_limit_inclusive": REMOVAL_LOWER_LIMIT,
                "pass": bool(removal_interval["mean"] >= REMOVAL_MARGIN and
                             removal_interval["lower"] >= REMOVAL_LOWER_LIMIT),
                "role": "reported_separately_not_a_cell_verdict_gate"}}
    else:
        gates = {"computable": False, "utility": None, "gain": None, "benefit_over_removal": None,
                 "note": "gates are not computed on incomplete seed sets (no outcome-based deletion)"}
    combos = {f"{target}@{channel}@{defense}": _privacy_combination(
                  cell_id, target, channel, defense, seeds, rows_by_key,
                  frozen_combinations, privacy_limits)
              for target in TARGETS for channel in CHANNELS}
    breaches = [key for key, entry in combos.items() if entry["verdict"] == "BREACH"]
    if breaches:
        privacy_status = "REJECTED"
    elif all(entry["verdict"] == "PASS" for entry in combos.values()):
        privacy_status = "PASS"
    else:
        privacy_status = "UNDETERMINED"
    reject_reasons: list[str] = []
    if complete:
        if not gates["utility"]["pass"]:
            reject_reasons.append("utility_gate")
        if not gates["gain"]["pass"]:
            reject_reasons.append("gain_gate")
    if breaches:
        reject_reasons.append("privacy_qualified_breach")
    undetermined_reasons: list[Any] = []
    if reject_reasons:
        verdict = "REJECTED"
    elif not complete:
        verdict = "UNDETERMINED"
        undetermined_reasons.append({"incomplete_seed_evidence":
                                     [run_id for record in seed_records
                                      for run_id in record["missing_run_ids"]]})
    elif privacy_status == "PASS":
        verdict = "PASS"
    else:
        verdict = "UNDETERMINED"
    if verdict == "UNDETERMINED" and privacy_status == "UNDETERMINED":
        undetermined_reasons.append({"privacy_undetermined":
                                     [key for key, entry in combos.items()
                                      if entry["verdict"] != "PASS"]})
    return {"cell_id": cell_id, "defense": defense, "verdict": verdict,
            "reject_reasons": reject_reasons, "privacy_breaches": breaches,
            "undetermined_reasons": undetermined_reasons,
            "complete": complete, "seeds": seed_records, "gates": gates,
            "privacy": {"status": privacy_status,
                        "rule": "worst-seed (maximum per-seed advantage) per required qualified "
                                "combination, compared to the combination's privacy limit",
                        "combinations": combos},
            "sensitivity": _sensitivity(utility_interval, gain_interval, complete)}


def _summary(cells: Mapping[str, Any], accounting: Mapping[str, Any]) -> dict[str, Any]:
    passing = sorted(cell for cell, entry in cells.items() if entry["verdict"] == "PASS")
    rejected = sorted(cell for cell, entry in cells.items() if entry["verdict"] == "REJECTED")
    undetermined = sorted(cell for cell, entry in cells.items() if entry["verdict"] == "UNDETERMINED")
    reason_counts: dict[str, int] = {}
    for entry in cells.values():
        for reason in entry["reject_reasons"]:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    if passing:
        classification = "scoped_candidate_pass_present"
        note = ("a scoped candidate pass is evidence against the open finite-family hypothesis at "
                "its operating point; it is not a certificate against omitted attacks")
    elif undetermined:
        classification = "undetermined"
        note = "undetermined cells block the empty finite feasible set declaration"
    else:
        classification = "empty"
        note = ("every planned cell has a valid rejection reason; the mixture of utility, gain "
                "and privacy rejections is reported per cell")
    return {"verdict_counts": {"PASS": len(passing), "REJECTED": len(rejected),
                               "UNDETERMINED": len(undetermined)},
            "reject_reason_counts": reason_counts,
            "finite_set": {"classification": classification,
                           "empty_feasible_set_declared": classification == "empty",
                           "passing_cells": passing, "rejected_cells": rejected,
                           "undetermined_cells": undetermined, "note": note},
            "run_accounting": {"planned": accounting["planned_count"],
                               "completed": accounting["completed_count"],
                               "failures": len(accounting["failures"]),
                               "interrupted": len(accounting["interrupted_runs"]),
                               "not_started": len(accounting["not_started_runs"])}}


def _load_frozen_spec(launch: Mapping[str, Any], frozen_spec_path: Path,
                      qualification_path: Path | None = None
                      ) -> tuple[dict[str, Any], dict[str, float], str, str]:
    """Read the admitted document without rewriting any frozen evidence.

    Runner-format documents bind canonical JSON in the launch and file bytes in
    per-run bindings. Their reviewed origin pins the separate qualification
    proposal. Older standalone analysis fixtures use a byte-bound proposal.
    """
    if not frozen_spec_path.is_file():
        raise ValueError("frozen attack specification is missing")
    spec_sha = sha256_file(str(frozen_spec_path))
    document = read_json(frozen_spec_path)
    bound = next((launch[key] for key in ATTACK_SPEC_LAUNCH_KEYS
                  if isinstance(launch.get(key), str)), None)
    wrapped = document.get("schema_version") == "r1-attack-spec-1.0"
    expected = sha256_json(document) if wrapped else spec_sha
    if bound is None or bound != expected:
        raise ValueError("frozen attack specification differs from the launch binding")
    if wrapped:
        from .r1_scientific import _specification_from_document
        _specification_from_document(document)
        if document.get("status") != "frozen":
            raise ValueError("exploratory attack specification is not frozen")
        matches = re.findall(r"proposal_sha256 ([a-f0-9]{64})(?![a-f0-9])",
                             document.get("origin", ""))
        if qualification_path is None or len(matches) != 1:
            raise ValueError("frozen document requires its hash-bound qualification proposal")
        if sha256_file(str(qualification_path)) != matches[0]:
            raise ValueError("qualification proposal differs from the frozen origin binding")
        spec = read_json(qualification_path)
        execution = spec.get("execution", {})
        if (spec.get("status") != "proposal_ready_for_freeze_review"
                or execution.get("planned_count", 0) <= 0
                or execution.get("completed_count") != execution.get("planned_count")
                or any(execution.get(k) != [] for k in
                       ("failures", "interrupted_runs", "not_started_runs"))):
            raise ValueError("qualification proposal is incomplete")
        if spec.get("thresholds") != document["specification"]:
            raise ValueError("qualification thresholds differ from the frozen specification")
        # Keep original proposal bytes and status; review authority is the wrapper.
        spec = {**spec, "status": "frozen", "review_document": document}
    else:
        spec = document
    if spec.get("schema_version") != FROZEN_SPEC_SCHEMA:
        raise ValueError("frozen attack specification schema is unsupported")
    if not isinstance(spec.get("combinations"), dict):
        raise ValueError("frozen attack specification has no combination verdicts")
    limits = spec.get("privacy_limits")
    if limits is None and isinstance(spec.get("thresholds"), dict):
        limits = {"correlation": spec["thresholds"].get("privacy_limit_correlation"),
                  "accuracy": spec["thresholds"].get("privacy_limit_accuracy")}
    if (not isinstance(limits, dict) or
            any(not isinstance(limits.get(name), (int, float))
                or isinstance(limits.get(name), bool)
                or not math.isfinite(limits[name]) or limits[name] <= 0
                for name in ("correlation", "accuracy"))):
        raise ValueError("frozen attack specification privacy limits are invalid")
    if wrapped and limits != {"correlation": document["specification"]["privacy_limit_correlation"],
                              "accuracy": document["specification"]["privacy_limit_accuracy"]}:
        raise ValueError("qualification privacy limits differ from the frozen specification")
    return (spec, {name: float(limits[name]) for name in ("correlation", "accuracy")},
            spec_sha, bound)


def _write_sidecar(path: Path, inputs: Sequence[Mapping[str, Any]]) -> None:
    write_new(path.with_suffix(path.suffix + ".sha256.json"),
              {"path": path.name, "sha256": sha256_file(str(path)),
               "input_bindings_sha256": sha256_json(list(inputs)),
               "scientific_evidence": True})


# --------------------------------------------------------------------------- #
# public writers
# --------------------------------------------------------------------------- #
def analyze_calibration_execution(root: Path, planned: Sequence[Mapping[str, Any]],
                                  out_path: Path) -> dict[str, Any]:
    """Aggregate a calibration execution root into a frozen attack specification proposal."""
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"calibration execution root is missing: {root}")
    launch = _load_launch(root, "calibration")
    structure = _planned_structure(planned, "calibration")
    accounting = _account_runs(root, planned, launch)
    proposal = _frozen_spec_proposal(root, structure, accounting)
    spec_path = out_path.parent / f"{out_path.stem}.frozen-attack-spec.json"
    report = {"schema_version": CALIBRATION_ANALYSIS_SCHEMA, "created_at_utc": now(),
              "stage": "calibration", "seeds": structure["seeds"], "cells": structure["cells"],
              "accounting": {key: accounting[key] for key in
                             ("planned_count", "completed_count", "complete", "failures",
                              "interrupted_runs", "not_started_runs", "verification")},
              "frozen_attack_specification": proposal,
              "frozen_attack_specification_path": spec_path.name,
              "input_bindings": accounting["input_bindings"],
              "scientific_evidence": True, "scope": SCOPE}
    write_new(out_path, report)
    _write_sidecar(out_path, accounting["input_bindings"])
    write_new(spec_path, proposal)
    _write_sidecar(spec_path, proposal["input_bindings"])
    return report


def analyze_exploratory_execution(root: Path, planned: Sequence[Mapping[str, Any]],
                                  frozen_spec_path: Path, out_path: Path, *,
                                  qualification_path: Path | None = None) -> dict[str, Any]:
    """Classify every exploratory cell against a hash-bound frozen attack specification."""
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"exploratory execution root is missing: {root}")
    launch = _load_launch(root, "exploratory")
    spec, privacy_limits, spec_sha, bound = _load_frozen_spec(launch, Path(frozen_spec_path), qualification_path)
    structure = _planned_structure(planned, "exploratory")
    accounting = _account_runs(root, planned, launch)
    rows_by_key = {(row["cell_id"], row["seed"], row["mode"]): row
                   for row in accounting["rows"]}
    cells = {}
    for cell_id in structure["cells"]:
        defense = _defense_label(
            structure["planned_by_key"][(cell_id, structure["seeds"][0], "trained")]["split"])
        cells[cell_id] = _classify_cell(cell_id, defense, structure["seeds"],
                                        structure["planned_by_key"], rows_by_key,
                                        spec["combinations"], privacy_limits)
    inputs = list(accounting["input_bindings"])
    inputs.append({"path": str(frozen_spec_path), "sha256": spec_sha})
    if qualification_path is not None:
        inputs.append({"path": str(qualification_path),
                       "sha256": sha256_file(str(qualification_path))})
    for row in accounting["rows"]:
        if row["attack_report"] is not None and row["attack_report"]["specification"] != spec["thresholds"]:
            raise ValueError("exploratory attack thresholds differ from the frozen specification")
    report = {"schema_version": EXPLORATORY_ANALYSIS_SCHEMA, "created_at_utc": now(),
              "stage": "exploratory", "seeds": structure["seeds"],
              "frozen_attack_specification": {"path": str(frozen_spec_path), "sha256": spec_sha,
                                              "schema_version": spec["schema_version"],
                                              "status": spec.get("status"),
                                              "launch_binding_sha256": bound,
                                              "launch_binding_kind": ("canonical_json" if "review_document" in spec else "file_bytes"),
                                              "qualification_path": None if qualification_path is None else str(qualification_path)},
              "method": _METHOD,
              "gates": {"utility": {"mean_margin": UTILITY_MARGIN,
                                    "upper_95_limit": UTILITY_UPPER_LIMIT},
                        "gain": {"mean_margin": GAIN_MARGIN,
                                 "lower_95_limit_exclusive": GAIN_LOWER_LIMIT},
                        "benefit_over_removal": {
                            "mean_margin": REMOVAL_MARGIN,
                            "lower_95_limit_inclusive": REMOVAL_LOWER_LIMIT,
                            "role": "reported_separately_not_a_cell_verdict_gate"},
                        "privacy": {"rule": "worst-seed maximum advantage per required qualified "
                                            "combination", "limits": privacy_limits}},
              "accounting": {key: accounting[key] for key in
                             ("planned_count", "completed_count", "complete", "failures",
                              "interrupted_runs", "not_started_runs", "verification")},
              "cells": cells,
              "summary": _summary(cells, accounting),
              "input_bindings": inputs,
              "scientific_evidence": True, "scope": SCOPE}
    write_new(out_path, report)
    _write_sidecar(out_path, inputs)
    return report
