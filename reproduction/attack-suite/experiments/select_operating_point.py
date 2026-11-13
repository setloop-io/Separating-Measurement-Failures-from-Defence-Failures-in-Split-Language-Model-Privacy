"""Mechanical operating-point selection for the bottleneck sweep.

Reads a sweep analysis.json (schema
llm_attacker.bottleneck_sweep_analysis.v1, including the per-seed rows)
and applies the PREDEFINED acceptance criteria with no judgment calls:

  * utility gates are the predefined 1.5x (primary) / 2x (secondary)
    ppl_after ratios vs the paired undefended control, RECOMPUTED from
    the analysis's per-seed rows (the summary's PASS/FAIL labels are
    cross-checked against the per-seed rows, never trusted). Thresholds
    are hard-coded named constants; --thresholds is REFUSED
    (no-retuning rule).
  * an analysis containing FAILED/incomplete cells (status
    FAILED_CELLS or a nonempty failures list) fails closed: EVERY
    candidate is INELIGIBLE and every split outcome is negative — a
    partial compared set never yields an operating point.
  * an adversarial candidate (tb-frozen / tb-cotrain) is eligible only
    if it shows LOWER leakage than the pca-compress control at the same
    dim+split, paired over seeds, under BOTH separate conditions:
      (a) bootstrap condition: the point estimate and the deterministic
          percentile-bootstrap CI lower bound of the paired difference
          must both be > 0; AND
      (b) all-seeds-positive condition: the per-seed paired difference
          must be > 0 for EVERY seed.
    (a) does NOT imply (b); both are recorded separately in the
    selection record. Per-seed leakage is the STRONGEST OBSERVED attack
    result across all measured arms for the cell (base-tier
    forward/SIPIT, the defense-aware surrogate, the white-box arms,
    and — when the analyzer exports whitebox_e2e_seed_rows — the
    adaptive end-to-end arm)
    — a measured statistic over the measured arms only: stronger or
    better-optimized attacks may recover more. Every arm value is
    validated (finite float in [0, 1]) BEFORE the max; an invalid value
    makes the cell ineligible, never silently dropped. Cells whose
    white-box tier the analyzer recorded NOT_EVALUABLE (co-trained
    encoders, where the saved checkpoint never matches the capture-time
    encoder) are scored over the remaining measured arms with the
    exclusion recorded. The comparison is recorded either way.
  * compute: candidate train+attack wall seconds across ALL measured
    arms (base tier plus the white-box addendum's attack seconds) must
    be <= 2x the pca-compress cell's (fixed factor; ratio recorded
    either way).
  * attacker access / approximation limits are carried verbatim in
    ACCESS_LABELS: the PCA surrogate arm is never a faithful decoder;
    white-box rows are a strictly stronger access condition; adaptive
    end-to-end rows are strictly stronger AND adaptive, not a bound.
  * outcome labels per split: "primary" / "secondary" / "negative".
    A negative outcome retains the NEGATIVE result and v3 stays
    UNFROZEN — this tool never freezes anything.

Fail closed: any missing gate, comparison, or field makes the candidate
INELIGIBLE with an explicit reason, never silently passed. Output is
deterministic given the input (fixed bootstrap seed/replicates).

Usage:
  python experiments/select_operating_point.py --analysis <analysis.json>
      --output selection.json [--markdown selection.md]
"""
import argparse
import json
import math
from pathlib import Path
import random
import statistics
import sys

SCHEMA = "llm_attacker.operating_point_selection.v1"
ANALYSIS_SCHEMA = "llm_attacker.bottleneck_sweep_analysis.v1"

# Predefined gates — NOT tunable. The no-retuning rule forbids changing
# these per result; --thresholds is refused in main().
PRIMARY_GATE = 1.5
SECONDARY_GATE = 2.0
COMPUTE_FACTOR = 2.0  # max candidate/PCA train+attack wall-second ratio

CANDIDATE_METHODS = ("tb-frozen", "tb-cotrain")
CONTROL_METHOD = "pca-compress"
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 2026
BOOTSTRAP_ALPHA = 0.05

LEAKAGE_STATISTIC_LABEL = (
    "strongest observed across measured arms: per-seed max over the "
    "base-tier forward/SIPIT attacks, the defense-aware surrogate arm, "
    "the white-box-aware arms, and — when present — the adaptive "
    "end-to-end (true transform) arm. A measured statistic over the "
    "measured arms only — stronger or better-optimized attacks may "
    "recover more.")
ACCESS_LABELS = {
    "surrogate_arm": (
        "the defense-aware arm fits a PCA surrogate (stand-in) inverse "
        "on public calibration; it is NOT a faithful decoder of the "
        "true defense transform and its recovery numbers are an "
        "approximation, not a certificate"),
    "whitebox_tier": (
        "white-box-aware rows grant the attacker the TRUE defense "
        "checkpoint (encoder weights / PCA basis + provenance) — a "
        "strictly stronger access condition than the matched matrix. "
        "White-box recovery is a stronger-access measurement; it does "
        "not constrain black-box recovery of the same wire, nor the "
        "reverse"),
    "whitebox_e2e_tier": (
        "adaptive end-to-end rows grant the attacker the TRUE defense "
        "checkpoint and optimize each wire's token sequence straight "
        "through it — a strictly stronger, adaptive access condition. "
        "Adaptive e2e recovery is a stronger-access measurement only: "
        "it does not constrain black-box recovery of the same wire, "
        "nor the surrogate arm, nor any defense-internals-hidden "
        "adversary"),
}
V3_RULE = ("a negative outcome retains the NEGATIVE result; v3 remains "
           "UNFROZEN — this tool never freezes anything")

NO_RETUNING_ERROR = (
    "refused: utility-gate thresholds are PREDEFINED constants "
    f"({PRIMARY_GATE}x primary / {SECONDARY_GATE}x secondary). The "
    "no-retuning rule forbids overriding them after seeing results; "
    "selection must follow the predefined gates mechanically.")


def percentile_bootstrap(diffs, replicates=BOOTSTRAP_REPLICATES,
                         seed=BOOTSTRAP_SEED, alpha=BOOTSTRAP_ALPHA):
    """Deterministic percentile bootstrap CI of the mean paired
    difference over seeds (same construction as the analyzer's)."""
    if not diffs:
        return {"mean": None, "lower": None, "upper": None, "n_seeds": 0,
                "status": "MISSING"}
    mean = statistics.mean(diffs)
    if len(diffs) < 3:
        return {"mean": mean, "lower": None, "upper": None,
                "n_seeds": len(diffs), "status": "INSUFFICIENT_EVIDENCE"}
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(statistics.mean(rng.choices(diffs, k=n))
                   for _ in range(replicates))
    lo = means[int((alpha / 2) * replicates)]
    hi = means[min(replicates - 1, int((1 - alpha / 2) * replicates) - 1)]
    return {"mean": mean, "lower": lo, "upper": hi, "n_seeds": n,
            "status": "ESTIMATED"}


ARM_NAMES = ("forward", "sipit", "da_forward", "da_sipit",
             "wb_inverter_forward", "wb_inverter_sipit",
             "wb_opt_forward", "wb_opt_snap")
BASE_ARM_NAMES = ARM_NAMES[:4]
WHITEBOX_ARM_NAMES = ARM_NAMES[4:]
E2E_ARM_NAME = "adaptive_e2e_snap"  # optional arm: included when present


def _valid_accuracy(value):
    """An arm accuracy is a finite float in [0, 1] (bool excluded)."""
    return isinstance(value, (int, float)) \
        and not isinstance(value, bool) \
        and math.isfinite(value) and 0.0 <= value <= 1.0


def measured_leakage(seed_row):
    """Strongest observed attack result for one cell-seed: the maximum
    across ALL measured arms present in seed_row (base-tier
    forward/SIPIT, the defense-aware surrogate, and — when the white-box
    tier was evaluable — the white-box-aware arms). This is a measured
    statistic over the measured arms only; stronger or better-optimized
    attacks may recover more.

    Every arm value is validated BEFORE the max: a non-finite,
    out-of-[0,1], or missing arm value raises ValueError (callers fail
    closed); invalid values never silently disappear inside the max.
    The adaptive end-to-end arm (adaptive_e2e_snap) is optional: when
    the key is present it is validated and included in the max; when
    absent (tier not run, or the cell's e2e tier NOT_EVALUABLE) the
    max runs over the remaining measured arms."""
    missing = [name for name in ARM_NAMES if name not in seed_row]
    if missing == list(WHITEBOX_ARM_NAMES):
        # white-box tier legitimately not evaluable for this cell
        # (recorded by the caller); the max runs over the base arms
        names = BASE_ARM_NAMES
        missing = [name for name in names if name not in seed_row]
    else:
        names = list(ARM_NAMES)
    if E2E_ARM_NAME in seed_row:
        names = list(names) + [E2E_ARM_NAME]
    bad = [name for name in names
           if name not in seed_row or not _valid_accuracy(seed_row[name])]
    if bad:
        raise ValueError(f"invalid attack value(s) {bad}: every measured "
                         "arm must be a finite float in [0, 1] (fail "
                         "closed; never dropped from the max)")
    return max(seed_row[name] for name in names)


def _index_summary(analysis):
    rows = {}
    for row in analysis.get("summary") or []:
        rows[(row["split"], row["dim"], row["method"])] = row
    return rows


def _whitebox_not_evaluable(analysis):
    """Cells whose white-box tier is recorded NOT_EVALUABLE by the
    analyzer (e.g. tb-cotrain: the saved checkpoint is the FINAL
    co-trained encoder, mismatched with the capture-time encoder)."""
    out = {}
    for entry in analysis.get("whitebox_not_evaluable") or []:
        out[(entry["split"], entry["dim"], entry["method"])] = \
            entry.get("reason") or "white-box tier not evaluable"
    return out


def _whitebox_e2e_not_evaluable(analysis):
    """Cells whose adaptive end-to-end tier is recorded NOT_EVALUABLE
    by the analyzer (tb-cotrain: training-phase captures predate the
    saved final co-trained encoder)."""
    out = {}
    for entry in analysis.get("whitebox_e2e_not_evaluable") or []:
        out[(entry["split"], entry["dim"], entry["method"])] = \
            entry.get("reason") or "whitebox-e2e tier not evaluable"
    return out


def _e2e_seed_values(analysis, split, dim, method, e2e_na=None):
    """Adaptive e2e arm values for one cell from the analyzer's
    whitebox_e2e_seed_rows. Returns ({seed: value}, status): status is
    None when the e2e arm is measured for this cell, the recorded
    NOT_EVALUABLE reason when legitimately excluded, or "absent" when
    the tier was not run. Raises ValueError on any invalid arm value
    (fail closed; invalid values never silently disappear)."""
    vals = {}
    for row in analysis.get("whitebox_e2e_seed_rows") or []:
        if (row["split"], row["dim"], row["method"]) == (split, dim,
                                                         method):
            v = row.get(E2E_ARM_NAME)
            if not _valid_accuracy(v):
                raise ValueError(
                    f"invalid adaptive e2e value {v!r} for {method} "
                    f"dim={dim} split={split} seed={row.get('seed')}: "
                    "every measured arm must be a finite float in "
                    "[0, 1] (fail closed; never dropped from the max)")
            vals[row["seed"]] = v
    if vals:
        return vals, None
    na_reason = (e2e_na or {}).get((split, dim, method))
    if na_reason is not None:
        return {}, f"NOT_EVALUABLE: {na_reason}"
    return {}, "absent"


def _seed_leakage(analysis, split, dim, method, wb_na=None, e2e_na=None):
    """Per-seed strongest observed attack result for one cell, merged
    from the analyzer's base-tier seed_rows, whitebox_seed_rows and —
    when present — whitebox_e2e_seed_rows. Returns (per_seed,
    wb_status, e2e_status): wb_status/e2e_status are None when the
    tier's arms are included, or the recorded NOT_EVALUABLE reason /
    "absent" when the tier is legitimately missing for this cell (the
    max then runs over the remaining measured arms). per_seed is {}
    when rows are missing/unpairable without a NOT_EVALUABLE record
    (callers fail closed). Raises ValueError on any invalid arm
    value."""
    base, wb = {}, {}
    for row in analysis.get("seed_rows") or []:
        if (row["split"], row["dim"], row["method"]) == (split, dim,
                                                         method):
            base[row["seed"]] = row
    for row in analysis.get("whitebox_seed_rows") or []:
        if (row["split"], row["dim"], row["method"]) == (split, dim,
                                                         method):
            wb[row["seed"]] = row
    e2e, e2e_status = _e2e_seed_values(analysis, split, dim, method,
                                       e2e_na)
    if e2e and set(e2e) != set(base):
        # a partially measured e2e tier is not a legitimate exclusion
        return {}, None, e2e_status
    if not base:
        return {}, None, e2e_status
    na_reason = (wb_na or {}).get((split, dim, method))
    if not wb or set(base) != set(wb):
        if na_reason is None:
            return {}, None, e2e_status
        out = {}
        for seed, b in base.items():
            row = {
                "forward": b.get("forward_accuracy"),
                "sipit": b.get("sipit_accuracy"),
                "da_forward": b.get("da_forward_accuracy"),
                "da_sipit": b.get("da_sipit_accuracy")}
            if seed in e2e:
                row[E2E_ARM_NAME] = e2e[seed]
            out[seed] = measured_leakage(row)
        return out, f"NOT_EVALUABLE: {na_reason}", e2e_status
    out = {}
    for seed in base:
        b, w = base[seed], wb[seed]
        row = {
            "forward": b.get("forward_accuracy"),
            "sipit": b.get("sipit_accuracy"),
            "da_forward": b.get("da_forward_accuracy"),
            "da_sipit": b.get("da_sipit_accuracy"),
            "wb_inverter_forward": w.get("wb_inverter_forward"),
            "wb_inverter_sipit": w.get("wb_inverter_sipit"),
            "wb_opt_forward": w.get("wb_opt_forward"),
            "wb_opt_snap": w.get("wb_opt_snap")}
        if seed in e2e:
            row[E2E_ARM_NAME] = e2e[seed]
        out[seed] = measured_leakage(row)
    return out, None, e2e_status


def _seed_gate(analysis, split, dim, method):
    """Utility-gate verdict RECOMPUTED from the per-seed rows (the
    summary's PASS/FAIL labels are never trusted). Returns None when
    the per-seed rows are missing or malformed (callers fail closed)."""
    rows = [r for r in analysis.get("seed_rows") or []
            if (r["split"], r["dim"], r["method"]) == (split, dim,
                                                       method)]
    if not rows:
        return None
    per_seed = {}
    for row in rows:
        p15, p20 = row.get("pass_1_5x"), row.get("pass_2x")
        ratio = row.get("ppl_ratio")
        if not isinstance(p15, bool) or not isinstance(p20, bool) \
                or not isinstance(ratio, (int, float)) \
                or isinstance(ratio, bool) or not math.isfinite(ratio) \
                or ratio <= 0:
            return None
        per_seed[row["seed"]] = {"pass_1_5x": p15, "pass_2x": p20,
                                 "ppl_ratio": ratio}
    n = len(per_seed)
    n15 = sum(1 for v in per_seed.values() if v["pass_1_5x"])
    n20 = sum(1 for v in per_seed.values() if v["pass_2x"])
    return {"seeds": sorted(per_seed), "per_seed": per_seed,
            "pass_1_5x_seeds": n15, "pass_2x_seeds": n20,
            "gate_1_5x": "PASS" if n15 == n else "FAIL",
            "gate_2x": "PASS" if n20 == n else "FAIL",
            "source": "recomputed from the per-seed rows; the summary's "
                      "labels are cross-checked, never trusted"}


def _whitebox_attack_seconds(analysis, split, dim, method):
    """Mean white-box-tier attack seconds for one cell (None when the
    tier is absent). Raises ValueError on invalid values."""
    vals = []
    for row in analysis.get("whitebox_seed_rows") or []:
        if (row["split"], row["dim"], row["method"]) == (split, dim,
                                                         method):
            v = row.get("attack_seconds")
            if not isinstance(v, (int, float)) or isinstance(v, bool) \
                    or not math.isfinite(v) or v < 0:
                raise ValueError(f"invalid white-box attack_seconds "
                                 f"{v!r} for {method} dim={dim} "
                                 f"split={split}")
            vals.append(v)
    return statistics.mean(vals) if vals else None


def evaluate_candidate(analysis, summary, split, dim, method):
    """Mechanical eligibility check for one candidate cell. Returns a
    record with every intermediate comparison; eligible=False whenever
    any predefined check fails or any required field is missing."""
    record = {"split": split, "dim": dim, "method": method,
              "eligible": True, "reasons": [], "gate": None}
    # 0. fail closed on a tainted analysis: any FAILED/incomplete cell
    # in the compared set makes every candidate ineligible
    failures = analysis.get("failures") or []
    if analysis.get("status") != "COMPLETE" or failures:
        record["eligible"] = False
        record["reasons"].append(
            f"analysis contains FAILED/incomplete cells (status "
            f"{analysis.get('status')!r}, {len(failures)} failure(s)); "
            "failing closed: no candidate is eligible from a sweep "
            "whose compared set is incomplete")
        return record
    row = summary.get((split, dim, method))
    if row is None:
        record["eligible"] = False
        record["reasons"].append("summary row missing")
        return record
    for key in ("gate_1_5x", "gate_2x", "wall_seconds_mean",
                "attack_seconds_mean"):
        if key not in row:
            record["eligible"] = False
            record["reasons"].append(f"summary row missing {key!r}")
            return record
    # 1. predefined utility gates, RECOMPUTED from the per-seed rows
    # (the summary's PASS/FAIL labels are cross-checked, never trusted)
    sg = _seed_gate(analysis, split, dim, method)
    if sg is None:
        record["eligible"] = False
        record["reasons"].append(
            "per-seed utility rows missing or malformed; the gate "
            "verdict cannot be recomputed (fail closed)")
        return record
    record["utility_gate_recomputed"] = sg
    if sg["gate_1_5x"] != row["gate_1_5x"] \
            or sg["gate_2x"] != row["gate_2x"]:
        record["eligible"] = False
        record["reasons"].append(
            f"summary gate labels ({row['gate_1_5x']}/{row['gate_2x']}) "
            f"contradict the per-seed rows ({sg['gate_1_5x']}/"
            f"{sg['gate_2x']}); failing closed")
    if sg["gate_1_5x"] == "PASS":
        record["gate"] = "primary"
    elif sg["gate_2x"] == "PASS":
        record["gate"] = "secondary"
    else:
        record["gate"] = "fail"
        record["eligible"] = False
        record["reasons"].append(
            f"utility gate FAIL at both predefined thresholds "
            f"({PRIMARY_GATE}x / {SECONDARY_GATE}x) on the per-seed "
            "rows")
    control = summary.get((split, dim, CONTROL_METHOD))
    if control is None:
        record["eligible"] = False
        record["reasons"].append(f"{CONTROL_METHOD} control row missing "
                                 "at same dim+split")
        return record
    # 2. strongest observed leakage must beat the PCA control, paired
    wb_na = _whitebox_not_evaluable(analysis)
    e2e_na = _whitebox_e2e_not_evaluable(analysis)
    try:
        cand_wb, cand_wb_status, cand_e2e_status = _seed_leakage(
            analysis, split, dim, method, wb_na, e2e_na)
        ctrl_wb, ctrl_wb_status, ctrl_e2e_status = _seed_leakage(
            analysis, split, dim, CONTROL_METHOD, wb_na, e2e_na)
    except ValueError as exc:
        record["eligible"] = False
        record["reasons"].append(str(exc))
        cand_wb, cand_wb_status, cand_e2e_status = {}, None, None
        ctrl_wb, ctrl_wb_status, ctrl_e2e_status = {}, None, None
    comparison = {"metric": "strongest observed attack result across "
                            "measured arms (max per seed)",
                  "statistic_label": LEAKAGE_STATISTIC_LABEL,
                  "candidate_per_seed": cand_wb,
                  "control_per_seed": ctrl_wb,
                  "whitebox_tier": cand_wb_status or "measured",
                  "control_whitebox_tier": ctrl_wb_status or "measured",
                  "e2e_tier": cand_e2e_status or "measured",
                  "control_e2e_tier": ctrl_e2e_status or "measured"}
    if not cand_wb or not ctrl_wb:
        record["eligible"] = False
        record["reasons"].append(
            "per-seed measured-arm rows (base tier and/or white-box "
            "tier) missing or unpairable for candidate or "
            f"{CONTROL_METHOD} control (fail closed)")
    elif set(cand_wb) != set(ctrl_wb):
        record["eligible"] = False
        record["reasons"].append(
            f"per-seed seed sets differ (candidate {sorted(cand_wb)} vs "
            f"control {sorted(ctrl_wb)}); cannot pair")
    else:
        seeds = sorted(cand_wb)
        diffs = [ctrl_wb[s] - cand_wb[s] for s in seeds]
        ci = percentile_bootstrap(diffs)
        bootstrap_ok = (ci["mean"] is not None and ci["mean"] > 0
                        and ci["lower"] is not None and ci["lower"] > 0)
        all_seeds_ok = all(d > 0 for d in diffs)
        comparison.update({"paired_seeds": seeds,
                           "control_minus_candidate_diffs": diffs,
                           "point_estimate": ci["mean"],
                           "ci_low": ci["lower"], "ci_high": ci["upper"],
                           "ci_status": ci["status"],
                           "bootstrap_condition": {
                               "point_estimate_positive":
                                   ci["mean"] is not None
                                   and ci["mean"] > 0,
                               "ci_low_positive": ci["lower"] is not None
                                   and ci["lower"] > 0,
                               "passed": bootstrap_ok},
                           "all_seeds_positive_condition": {
                               "per_seed_diffs_positive":
                                   {str(s): d > 0
                                    for s, d in zip(seeds, diffs)},
                               "passed": all_seeds_ok}})
        if not bootstrap_ok:
            record["eligible"] = False
            if ci["mean"] is None or ci["mean"] <= 0:
                record["reasons"].append(
                    f"strongest observed leakage NOT below "
                    f"{CONTROL_METHOD} control (paired point estimate "
                    f"{ci['mean']})")
            else:
                record["reasons"].append(
                    f"strongest observed leakage improvement over "
                    f"{CONTROL_METHOD} not established: bootstrap CI "
                    f"[{ci['lower']}, {ci['upper']}] includes 0 "
                    f"({ci['status']})")
        if not all_seeds_ok:
            record["eligible"] = False
            bad = [s for s, d in zip(seeds, diffs) if d <= 0]
            record["reasons"].append(
                f"per-seed paired difference not positive for every "
                f"seed (seeds {bad}); the bootstrap condition alone "
                f"does not establish improvement on every paired seed")
    record["pca_leakage_comparison"] = comparison
    # 3. compute: candidate train + attack wall seconds (base tier PLUS
    # the white-box-tier attack seconds — attack compute across all
    # measured arms) must be <= 2x the pca-compress cell's total
    try:
        cand_wb_attack = _whitebox_attack_seconds(analysis, split, dim,
                                                  method)
        ctrl_wb_attack = _whitebox_attack_seconds(analysis, split, dim,
                                                  CONTROL_METHOD)
    except ValueError as exc:
        record["eligible"] = False
        record["reasons"].append(str(exc))
        return record
    cand_compute = row["wall_seconds_mean"] + row["attack_seconds_mean"] \
        + (cand_wb_attack or 0.0)
    try:
        ctrl_compute = control["wall_seconds_mean"] + \
            control["attack_seconds_mean"] + (ctrl_wb_attack or 0.0)
    except KeyError as exc:
        record["eligible"] = False
        record["reasons"].append(f"control row missing {exc}")
        return record
    ratio = cand_compute / ctrl_compute if ctrl_compute > 0 else None
    record["compute_comparison"] = {
        "candidate_train_plus_attack_seconds": cand_compute,
        "control_train_plus_attack_seconds": ctrl_compute,
        "whitebox_attack_seconds_mean": cand_wb_attack,
        "control_whitebox_attack_seconds_mean": ctrl_wb_attack,
        "scope": "train + attack wall seconds across ALL measured arms "
                 "(base tier plus the white-box addendum tier)",
        "ratio": ratio, "max_factor": COMPUTE_FACTOR,
        "within_factor": ratio is not None and ratio <= COMPUTE_FACTOR}
    if ratio is None or ratio > COMPUTE_FACTOR:
        record["eligible"] = False
        record["reasons"].append(
            f"compute ratio {ratio} exceeds the predefined "
            f"{COMPUTE_FACTOR}x train+attack wall-second factor")
    return record


def select(analysis):
    """Apply the predefined criteria to every candidate cell. Per split,
    the selected operating point is the eligible candidate with the
    lowest mean strongest-observed leakage (ties: lower dim, then method
    name); outcome is primary/secondary/negative."""
    if analysis.get("schema") != ANALYSIS_SCHEMA:
        raise ValueError(f"unexpected analysis schema "
                         f"{analysis.get('schema')!r}")
    summary = _index_summary(analysis)
    splits = sorted({k[0] for k in summary})
    per_split = {}
    for split in splits:
        evals = []
        for (s, dim, method) in sorted(summary):
            if s == split and method in CANDIDATE_METHODS:
                evals.append(evaluate_candidate(analysis, summary,
                                                split, dim, method))
        eligible = [e for e in evals if e["eligible"]]
        primary = [e for e in eligible if e["gate"] == "primary"]
        pool = primary or [e for e in eligible if e["gate"] == "secondary"]

        def leak_mean(e):
            vals = list(e["pca_leakage_comparison"]
                        ["candidate_per_seed"].values())
            return (statistics.mean(vals) if vals else float("inf"),
                    e["dim"], e["method"])
        chosen = min(pool, key=leak_mean) if pool else None
        if primary:
            outcome = "primary"
        elif pool:
            outcome = "secondary"
        else:
            outcome = "negative"
        per_split[str(split)] = {
            "outcome": outcome,
            "selected": ({"method": chosen["method"], "dim": chosen["dim"],
                          "gate": chosen["gate"]} if chosen else None),
            "candidates": evals}
    negative = [s for s, r in per_split.items() if r["outcome"]
                == "negative"]
    record = {
        "schema": SCHEMA,
        "analysis_status": analysis.get("status"),
        "analysis_failures": len(analysis.get("failures") or []),
        "gates": {"primary_ppl_ratio": PRIMARY_GATE,
                  "secondary_ppl_ratio": SECONDARY_GATE,
                  "predefined": True,
                  "retuning": "refused (no-retuning rule)"},
        "compute_factor": COMPUTE_FACTOR,
        "candidate_methods": list(CANDIDATE_METHODS),
        "control_method": CONTROL_METHOD,
        "bootstrap": {"replicates": BOOTSTRAP_REPLICATES,
                      "seed": BOOTSTRAP_SEED,
                      "method": "percentile, paired over seeds"},
        "leakage_statistic": LEAKAGE_STATISTIC_LABEL,
        "eligibility_conditions": (
            "PCA-superiority requires BOTH the bootstrap condition "
            "(point estimate > 0 AND bootstrap CI lower > 0 on the "
            "paired per-seed differences) AND the all-seeds-positive "
            "condition (per-seed paired difference > 0 for EVERY "
            "seed). They are separate mechanical checks, recorded "
            "separately per candidate; the bootstrap condition alone "
            "does not imply improvement on every paired seed."),
        "meta": {
            "notes": [
                "field wording corrected post-review: the per-seed "
                "leakage statistic is the strongest observed attack "
                "result across measured arms (earlier drafts mislabeled "
                "the white-box-only max as a conservative ceiling); "
                "schema name unchanged "
                "(llm_attacker.operating_point_selection.v1), the "
                "comparison block now carries statistic_label, "
                "bootstrap_condition and all_seeds_positive_condition "
                "sub-records.",
                "adaptive end-to-end (whitebox-e2e) tier ingested: "
                "when the analyzer's whitebox_e2e_seed_rows are "
                "present, the adaptive_e2e snap value joins the "
                "per-seed max (validated fail-closed like every other "
                "arm; NOT_EVALUABLE cells are scored over the remaining "
                "measured arms with the exclusion recorded under "
                "e2e_tier); schema name unchanged."]},
        "access_labels": dict(ACCESS_LABELS),
        "v3_status": "UNFROZEN",
        "v3_rule": V3_RULE,
        "splits": per_split,
        "negative_splits": negative,
        "negative_statement": (
            f"split(s) {negative} selected NO operating point; the "
            "NEGATIVE result is retained and v3 remains UNFROZEN"
            if negative else None)}
    return record


def render_markdown(record):
    lines = ["# Mechanical operating-point selection (bottleneck sweep)",
             "",
             f"Schema `{record['schema']}`. Gates are PREDEFINED: "
             f"ppl_after <= {record['gates']['primary_ppl_ratio']}x "
             f"(primary) / <= {record['gates']['secondary_ppl_ratio']}x "
             f"(secondary) the paired undefended control; retuning is "
             f"refused. Compute factor <= {record['compute_factor']}x "
             f"the {record['control_method']} control's train+attack "
             "wall seconds.", "",
             "Leakage statistic: " + record["leakage_statistic"], "",
             "Eligibility: " + record["eligibility_conditions"], "",
             "## Access and approximation limits", ""]
    lines += ["- " + v for v in record["access_labels"].values()]
    lines += ["- " + record["v3_rule"], "", "## Outcome per split", "",
              "| Split | Outcome | Selected |", "|---:|:---:|---|"]
    for split, r in sorted(record["splits"].items(), key=lambda kv:
                           int(kv[0])):
        sel = r["selected"]
        lines.append(f"| {split} | {r['outcome']} | "
                     + (f"{sel['method']} dim={sel['dim']} "
                        f"({sel['gate']} gate)" if sel else "NONE")
                     + " |")
    lines += ["", "## Candidate comparisons", "",
              "| Split | Dim | Method | Gate | Strongest-observed "
              "leakage vs PCA (ctrl-cand [CI]) | Bootstrap | "
              "All seeds + | Compute ratio | Eligible |",
              "|---:|---:|---|:---:|---|:---:|:---:|---:|:---:|"]
    for split, r in sorted(record["splits"].items(),
                           key=lambda kv: int(kv[0])):
        for e in r["candidates"]:
            comp = e.get("pca_leakage_comparison") or {}
            pe, lo, hi = comp.get("point_estimate"), comp.get("ci_low"), \
                comp.get("ci_high")
            leak = (f"{pe:.4f} [{lo:.4f}, {hi:.4f}]"
                    if pe is not None and lo is not None else
                    (f"{pe:.4f} [n/a]" if pe is not None else "n/a"))
            boot = (comp.get("bootstrap_condition") or {}).get("passed")
            asp = (comp.get("all_seeds_positive_condition") or {}) \
                .get("passed")
            ratio = (e.get("compute_comparison") or {}).get("ratio")
            lines.append(
                f"| {split} | {e['dim']} | {e['method']} | {e['gate']} "
                f"| {leak} | "
                + ("n/a" if boot is None else "PASS" if boot else "FAIL")
                + " | "
                + ("n/a" if asp is None else "PASS" if asp else "FAIL")
                + " | "
                + (f"{ratio:.3f}" if ratio is not None else "n/a")
                + f" | {'YES' if e['eligible'] else 'NO: '
                     + '; '.join(e['reasons'])} |")
    if record.get("negative_statement"):
        lines += ["", "**NEGATIVE**: " + record["negative_statement"], ""]
    lines.append("")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--analysis", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--markdown", type=Path, default=None)
    ap.add_argument("--thresholds", type=float, nargs=2, default=None,
                    metavar=("PRIMARY", "SECONDARY"),
                    help="REFUSED: thresholds are predefined "
                         "(no-retuning rule)")
    args = ap.parse_args(argv)
    if args.thresholds is not None:
        print(NO_RETUNING_ERROR, file=sys.stderr)
        return 2
    analysis = json.loads(args.analysis.read_text())
    record = select(analysis)
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False)
                           + "\n")
    if args.markdown is not None:
        args.markdown.write_text(render_markdown(record))
    print(json.dumps({"schema": SCHEMA,
                      "outcomes": {s: r["outcome"] for s, r in
                                   record["splits"].items()},
                      "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
