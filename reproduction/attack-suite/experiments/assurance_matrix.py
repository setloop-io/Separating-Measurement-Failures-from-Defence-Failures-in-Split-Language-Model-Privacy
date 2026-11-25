#!/usr/bin/env python3
"""Build an adversary-conditioned assurance matrix from measured study analyses.

The matrix is a data artifact (schema ``llm_attacker.assurance_matrix.v1``)
with one row per (defense, attacker access condition):

  defense x attacker access x measured leakage x utility cost x evidence strength

Evidence strength is drawn from a fixed enum ladder (see ``EVIDENCE_LADDER``);
there is deliberately NO aggregate scalar risk score and NO maturity model.
The current evidence base does not validate either, and this tool refuses to
emit one: no row or matrix-level field may carry a combined/weighted score.

Fail-closed policy: a source cell that is missing its leakage or utility
fields is emitted as a row with status ``INCOMPLETE`` — never silently dropped,
never imputed.

Usage:
    python experiments/assurance_matrix.py \
        --pilot results/pilot-defended-v1/analysis/analysis.json \
        --output assurance_matrix.json

    # v2 combined artifact (pilot rows + bottleneck-family sweep rows):
    python experiments/assurance_matrix.py \
        --pilot results/pilot-defended-v1/analysis/analysis.json \
        --sweep-analysis results/bottleneck-sweep-v1/analysis/analysis.json \
        --sweep-selection results/bottleneck-sweep-v1/analysis/selection.json \
        --output experiments/assurance_matrix_v2.json
"""

import argparse
import json
import sys

SCHEMA = "llm_attacker.assurance_matrix.v1"

# Fixed evidence-strength ladder, weakest to strongest. Rows may only carry
# one of these exact values; extending the ladder is a schema change.
# Enum extension (documented, schema name unchanged): the level
# "strengthened_attacker_development_sweep" was added when the
# bottleneck-family sweep ingestion path landed (v2 artifacts). It sits
# between "defended_pilot_paired" and "frozen_crossed_bootstrap": stronger
# attacker access than the pilot (defense-aware surrogate arm plus a
# separately labeled white-box tier), still a development-tier, non-
# confirmatory measurement.
EVIDENCE_LADDER = [
    "fixture",
    "live_channel_engineering",
    "defended_pilot_paired",
    "strengthened_attacker_development_sweep",
    "frozen_crossed_bootstrap",
]

EVIDENCE_LADDER_META = {
    "fixture": (
        "Fixture/synthetic batteries (e.g. replica-integrity fixtures, "
        "view-privacy synthetic de-mixing). Engineering checks on synthetic "
        "data; positives do not falsify a deployed defense."
    ),
    "live_channel_engineering": (
        "Live in-repo channels exercised end to end (capture/transport/"
        "recovery plumbing). Demonstrates mechanism, not measured leakage "
        "against a defended target."
    ),
    "defended_pilot_paired": (
        "Defended development pilot over trained defenders: paired "
        "defense-minus-none seed differences with percentile bootstrap CIs, "
        "utility deltas, failures-and-nulls accounting. Development tier; "
        "conditional on a fixed article sample; not a confirmatory claim."
    ),
    "frozen_crossed_bootstrap": (
        "Frozen crossed-bootstrap study under the v2 protocol "
        "(crossed document/run bootstrap with multiplicity correction). "
        "Protocol frozen; evidence counts at this level only once executed."
    ),
    "strengthened_attacker_development_sweep": (
        "Development sweep over the bottleneck defense family "
        "(pca-compress / tb-frozen / tb-cotrain x dims x splits) with a "
        "strengthened attacker: base-tier forward/SIPIT attacks plus a "
        "defense-aware surrogate arm, and a separately labeled white-box "
        "tier that grants the true defense checkpoint. Predefined dual "
        "utility gates (no retuning); the sweep selection outcome was "
        "NEGATIVE. Development tier — stronger attacker access than the "
        "defended pilot, still not a confirmatory claim."
    ),
}

ACCESS_CONDITIONS = {
    "split-training-wire": (
        "Captured training-phase forward/backward wires at the split point; "
        "public calibration partition; pretrained public head only (final "
        "victim head not granted); query budgets matched across defenses."
    ),
    "split-training-wire-whitebox": (
        "As split-training-wire, plus the attacker is granted the TRUE "
        "defense checkpoint (encoder weights / PCA basis + provenance). "
        "Strictly stronger access than the matched wire tier: white-box "
        "recovery is a stronger-access measurement — it neither bounds "
        "black-box recovery of the same wire nor is bounded by it."
    ),
}

# Entries introduced by the v2 sweep-ingestion enum extension. They are
# documented in emitted artifacts only when rows actually use them, so the
# pilot-only v1 artifact remains byte-reproducible (see _finalize_meta).
LADDER_EXTENSION_LEVELS = {"strengthened_attacker_development_sweep"}
ACCESS_EXTENSION_IDS = {"split-training-wire-whitebox"}

# Fields that must never appear anywhere in the emitted artifact. The matrix
# reports per-row measurements only; any aggregate scalar risk score or
# maturity rating is out of scope by design.
FORBIDDEN_FIELDS = (
    "risk_score",
    "aggregate_score",
    "overall_score",
    "composite_score",
    "security_score",
    "maturity",
    "maturity_level",
    "rating",
)

PILOT_LEAKAGE_FIELDS = (
    "forward_minus_none_mean",
    "forward_minus_none_ci_low",
    "forward_minus_none_ci_high",
)
PILOT_UTILITY_FIELDS = (
    "ppl_delta_vs_none_mean",
    "ppl_delta_vs_none_ci_low",
    "ppl_delta_vs_none_ci_high",
)


def _assert_no_forbidden_fields(node):
    """Refuse to emit any aggregate scalar risk score, at any depth."""
    if isinstance(node, dict):
        for key, value in node.items():
            lowered = str(key).lower()
            for forbidden in FORBIDDEN_FIELDS:
                if forbidden in lowered:
                    raise ValueError(
                        "assurance matrix must not contain aggregate score "
                        "field %r" % key
                    )
            _assert_no_forbidden_fields(value)
    elif isinstance(node, list):
        for item in node:
            _assert_no_forbidden_fields(item)


def _pilot_row(cell):
    split = cell.get("split")
    defense = cell.get("defense")
    if split is None or defense is None:
        raise ValueError("pilot summary cell missing split/defense: %r" % cell)

    access_id = "split-training-wire"
    row = {
        "defense": defense,
        "access": {
            "id": access_id,
            "condition": ACCESS_CONDITIONS[access_id],
            "split": split,
        },
        "evidence_strength": "defended_pilot_paired",
        "seeds": cell.get("seeds"),
    }

    missing = [
        f
        for f in PILOT_LEAKAGE_FIELDS + PILOT_UTILITY_FIELDS
        if cell.get(f) is None
    ]
    if missing:
        row["status"] = "INCOMPLETE"
        row["missing_fields"] = sorted(missing)
        row["leakage"] = None
        row["utility_cost"] = None
        return row

    row["status"] = "ESTIMATED"
    row["leakage"] = {
        "metric": "forward_recovery_minus_none",
        "unit": "fraction",
        "paired_effect": cell["forward_minus_none_mean"],
        "ci_low": cell["forward_minus_none_ci_low"],
        "ci_high": cell["forward_minus_none_ci_high"],
        "estimation_status": cell.get("forward_minus_none_status"),
        "note": (
            "Paired defense-minus-none position-aligned token recovery; "
            "negative means the defense reduced recovery."
        ),
    }
    row["utility_cost"] = {
        "metric": "heldout_ppl_delta_vs_none",
        "paired_effect": cell["ppl_delta_vs_none_mean"],
        "ci_low": cell["ppl_delta_vs_none_ci_low"],
        "ci_high": cell["ppl_delta_vs_none_ci_high"],
        "note": "Positive means the defense cost held-out perplexity.",
    }
    return row


def _finalize_meta(matrix):
    """Resolve which ladder levels and access conditions the artifact emits.

    Base levels/conditions are always documented (they define the schema).
    Enum-extension entries (the strengthened-attacker sweep level and the
    white-box access condition, both introduced with the v2 sweep ingestion)
    are documented only when the matrix's rows actually use them — so a
    pilot-only build stays byte-identical to the committed v1 artifact.
    """
    used_levels = {row["evidence_strength"] for row in matrix["rows"]}
    unknown = used_levels - set(EVIDENCE_LADDER)
    if unknown:
        raise ValueError("rows carry non-enum evidence levels: %r" % unknown)
    matrix["meta"]["evidence_ladder"] = [
        {"level": level, "meaning": EVIDENCE_LADDER_META[level]}
        for level in EVIDENCE_LADDER
        if level not in LADDER_EXTENSION_LEVELS or level in used_levels
    ]
    used_access = {row["access"]["id"] for row in matrix["rows"]}
    matrix["meta"]["access_conditions"] = {
        key: ACCESS_CONDITIONS[key]
        for key in ACCESS_CONDITIONS
        if key not in ACCESS_EXTENSION_IDS or key in used_access
    }
    return matrix


def build_from_pilot(pilot):
    schema = pilot.get("schema")
    if schema != "llm_attacker.defended_pilot_analysis.v1":
        raise ValueError("unsupported pilot analysis schema: %r" % schema)
    summary = pilot.get("summary")
    if not isinstance(summary, list):
        raise ValueError("pilot analysis has no summary list")

    rows = [_pilot_row(cell) for cell in summary]
    rows.sort(key=lambda r: (r["access"]["split"], r["defense"]))

    matrix = {
        "schema": SCHEMA,
        "meta": {
            "description": (
                "Adversary-conditioned assurance matrix: defense x attacker "
                "access x measured leakage x utility cost x evidence strength. "
                "Per-row measurements only — no universal risk score and no "
                "maturity model; the current evidence does not validate either."
            ),
            "evidence_ladder": [
                {"level": level, "meaning": EVIDENCE_LADDER_META[level]}
                for level in EVIDENCE_LADDER
            ],
            "access_conditions": ACCESS_CONDITIONS,
            "fail_closed_policy": (
                "Rows with missing leakage or utility fields are emitted as "
                "INCOMPLETE, never dropped or imputed."
            ),
            "sources": [
                {
                    "kind": "defended_pilot",
                    "tier": pilot.get("tier"),
                    "status": pilot.get("status"),
                    "cells_ok": pilot.get("cells_ok"),
                    "cells_failed": pilot.get("cells_failed"),
                    "failures": pilot.get("failures", []),
                    "bootstrap": pilot.get("bootstrap"),
                    "analysis_script_sha256": pilot.get("analysis_script_sha256"),
                }
            ],
        },
        "rows": rows,
    }
    _finalize_meta(matrix)
    _assert_no_forbidden_fields(matrix)
    return matrix


SWEEP_BASE_ARMS = (
    ("forward_accuracy_mean", "base forward"),
    ("sipit_accuracy_mean", "base SIPIT"),
    ("defense_aware_forward_mean", "defense-aware surrogate forward"),
    ("defense_aware_sipit_mean", "defense-aware surrogate SIPIT"),
)
SWEEP_WHITEBOX_ARMS = (
    ("wb_inverter_forward_mean", "white-box inverter forward"),
    ("wb_inverter_sipit_mean", "white-box inverter SIPIT"),
    ("wb_opt_forward_mean", "white-box optimized forward"),
    ("wb_opt_snap_mean", "white-box optimized snapshot"),
)
SWEEP_UTILITY_FIELDS = (
    "ppl_delta_vs_none_mean",
    "ppl_delta_vs_none_ci_low",
    "ppl_delta_vs_none_ci_high",
)

_STRONGEST_NOTE = (
    "Strongest observed attack result across the measured arms only — "
    "stronger or better-optimized attacks may recover more."
)


def _strongest_arm(cell, arms):
    """Return (value, arm_label, arms_considered); (None, ...) if any missing."""
    considered = {}
    for field, label in arms:
        value = cell.get(field)
        if value is None:
            return None, None, None
        considered[label] = value
    label = max(considered, key=considered.get)
    return considered[label], label, considered


def _sweep_utility(cell):
    missing = [f for f in SWEEP_UTILITY_FIELDS if cell.get(f) is None]
    if missing:
        return None, missing
    return {
        "metric": "heldout_ppl_delta_vs_none",
        "paired_effect": cell["ppl_delta_vs_none_mean"],
        "ci_low": cell["ppl_delta_vs_none_ci_low"],
        "ci_high": cell["ppl_delta_vs_none_ci_high"],
        "estimation_status": cell.get("ppl_delta_vs_none_status"),
        "ppl_ratio_mean": cell.get("ppl_ratio_mean"),
        "gates": {
            "thresholds_predefined": True,
            "retuning": "refused (no-retuning rule)",
            "1.5x": cell.get("gate_1_5x"),
            "2.0x": cell.get("gate_2x"),
        },
        "note": "Positive means the defense cost held-out perplexity.",
    }, []


def _sweep_selection_context(cell, selection):
    """Per-row selection context for a sweep cell (control or candidate)."""
    split_block = (selection.get("splits") or {}).get(str(cell["split"])) or {}
    context = {
        "outcome": split_block.get("outcome"),
        "selected": split_block.get("selected"),
        "v3_status": selection.get("v3_status"),
    }
    if cell["method"] == selection.get("control_method"):
        context["role"] = "control"
        return context
    for candidate in split_block.get("candidates", []):
        if (
            candidate.get("method") == cell["method"]
            and candidate.get("dim") == cell["dim"]
        ):
            context["role"] = "candidate"
            context["eligible"] = candidate.get("eligible")
            context["reasons"] = candidate.get("reasons")
            return context
    context["role"] = "candidate"
    context["eligible"] = None
    context["reasons"] = ["candidate block missing from selection artifact"]
    return context


def _sweep_base_row(cell, selection):
    row = {
        "defense": cell["method"],
        "dim": cell["dim"],
        "access": {
            "id": "split-training-wire",
            "condition": ACCESS_CONDITIONS["split-training-wire"],
            "split": cell["split"],
        },
        "evidence_strength": "strengthened_attacker_development_sweep",
        "seeds": cell.get("seeds"),
        "selection": _sweep_selection_context(cell, selection),
    }
    utility, missing_util = _sweep_utility(cell)
    strongest, arm, considered = _strongest_arm(cell, SWEEP_BASE_ARMS)
    missing = list(missing_util)
    if strongest is None:
        missing += [f for f, _ in SWEEP_BASE_ARMS if cell.get(f) is None]
    if missing:
        row["status"] = "INCOMPLETE"
        row["missing_fields"] = sorted(missing)
        row["leakage"] = None
        row["utility_cost"] = utility
        return row
    row["status"] = "ESTIMATED"
    row["leakage"] = {
        "metric": "strongest_observed_recovery_across_measured_arms",
        "unit": "fraction",
        "observed": strongest,
        "strongest_arm": arm,
        "arms_considered": considered,
        "note": _STRONGEST_NOTE,
    }
    row["utility_cost"] = utility
    return row


def _sweep_whitebox_row(cell, selection, not_evaluable):
    row = {
        "defense": cell["method"],
        "dim": cell["dim"],
        "access": {
            "id": "split-training-wire-whitebox",
            "condition": ACCESS_CONDITIONS["split-training-wire-whitebox"],
            "split": cell["split"],
        },
        "evidence_strength": "strengthened_attacker_development_sweep",
        "seeds": cell.get("seeds"),
        "selection": _sweep_selection_context(cell, selection),
    }
    ne_key = (cell["split"], cell["dim"], cell["method"])
    if ne_key in not_evaluable:
        # Declared non-evaluable (e.g. tb-cotrain's final checkpoint does not
        # match any captured wire) — fail-closed: emit the row, never drop it.
        row["status"] = "NOT_EVALUABLE"
        row["reason"] = not_evaluable[ne_key]
        row["leakage"] = None
        row["utility_cost"] = None
        return row
    utility, missing_util = _sweep_utility(cell)
    strongest, arm, considered = _strongest_arm(cell, SWEEP_WHITEBOX_ARMS)
    missing = list(missing_util)
    if strongest is None:
        missing += [f for f, _ in SWEEP_WHITEBOX_ARMS if cell.get(f) is None]
    if missing:
        row["status"] = "INCOMPLETE"
        row["missing_fields"] = sorted(missing)
        row["leakage"] = None
        row["utility_cost"] = utility
        return row
    row["status"] = "ESTIMATED"
    row["leakage"] = {
        "metric": "strongest_observed_recovery_whitebox_arms",
        "unit": "fraction",
        "observed": strongest,
        "strongest_arm": arm,
        "arms_considered": considered,
        "note": (
            "White-box tier: attacker granted the true defense checkpoint. "
            + _STRONGEST_NOTE
        ),
    }
    row["utility_cost"] = utility
    return row


def build_from_sweep(analysis, selection):
    """Bottleneck-family sweep rows from analysis.json + selection.json."""
    schema = analysis.get("schema")
    if schema != "llm_attacker.bottleneck_sweep_analysis.v1":
        raise ValueError("unsupported sweep analysis schema: %r" % schema)
    sel_schema = selection.get("schema")
    if sel_schema != "llm_attacker.operating_point_selection.v1":
        raise ValueError("unsupported sweep selection schema: %r" % sel_schema)
    summary = analysis.get("summary")
    if not isinstance(summary, list):
        raise ValueError("sweep analysis has no summary list")

    not_evaluable = {}
    for entry in analysis.get("whitebox_not_evaluable", []):
        key = (entry.get("split"), entry.get("dim"), entry.get("method"))
        not_evaluable[key] = entry.get("reason")

    rows = [_sweep_base_row(cell, selection) for cell in summary]
    wb_summary = analysis.get("whitebox_summary") or []
    wb_keys = {(c["split"], c["dim"], c["method"]) for c in wb_summary}
    for cell in wb_summary:
        rows.append(_sweep_whitebox_row(cell, selection, not_evaluable))
    # White-box cells declared NOT_EVALUABLE with no summary row still get a
    # fail-closed row built from the matching base cell's identity fields.
    base_by_key = {(c["split"], c["dim"], c["method"]): c for c in summary}
    for key, reason in sorted(not_evaluable.items()):
        if key in wb_keys:
            continue
        base = base_by_key.get(key)
        if base is None:
            raise ValueError("NOT_EVALUABLE whitebox cell has no base cell: %r" % (key,))
        rows.append(_sweep_whitebox_row(base, selection, not_evaluable))

    source = {
        "kind": "bottleneck_sweep",
        "tier": analysis.get("tier"),
        "status": analysis.get("status"),
        "cells_ok": analysis.get("cells_ok"),
        "cells_failed": analysis.get("cells_failed"),
        "expected_cells": analysis.get("expected_cells"),
        "whitebox_cells_ok": analysis.get("whitebox_cells_ok"),
        "failures": analysis.get("failures", []),
        "bootstrap": analysis.get("bootstrap"),
        "disjointness": analysis.get("disjointness"),
        "analysis_script_sha256": analysis.get("analysis_script_sha256"),
    }
    selection_meta = {
        "schema": sel_schema,
        "analysis_status": selection.get("analysis_status"),
        "outcome": "NEGATIVE",
        "negative_splits": selection.get("negative_splits"),
        "negative_statement": selection.get("negative_statement"),
        "v3_status": selection.get("v3_status"),
        "v3_rule": selection.get("v3_rule"),
        "gates": selection.get("gates"),
        "compute_factor": selection.get("compute_factor"),
        "candidate_methods": selection.get("candidate_methods"),
        "control_method": selection.get("control_method"),
        "leakage_statistic": selection.get("leakage_statistic"),
        "eligibility_conditions": selection.get("eligibility_conditions"),
        "access_labels": selection.get("access_labels"),
    }
    return rows, source, selection_meta


def _row_sort_key(row):
    return (
        row["access"]["split"],
        row.get("dim") or 0,
        row["access"]["id"],
        row["defense"],
    )


def extend_with_sweep(matrix, analysis, selection):
    """Extend a pilot-built matrix in place with bottleneck-sweep rows."""
    rows, source, selection_meta = build_from_sweep(analysis, selection)
    matrix["rows"].extend(rows)
    matrix["rows"].sort(key=_row_sort_key)
    matrix["meta"]["sources"].append(source)
    matrix["meta"]["sweep_selection"] = selection_meta
    matrix["meta"]["fail_closed_policy"] += (
        " White-box cells declared non-evaluable by the source analysis are "
        "emitted as NOT_EVALUABLE rows with the source reason, never dropped."
    )
    _finalize_meta(matrix)
    _assert_no_forbidden_fields(matrix)
    return matrix


def _fmt(value, nd=4):
    if value is None:
        return "—"
    return ("%.*f" % (nd, value)).rstrip("0").rstrip(".") if isinstance(value, float) else str(value)


def _fmt_pp(value):
    if value is None:
        return "—"
    return _fmt(100.0 * value, 2)


def _fmt_ppl(value):
    if value is None:
        return "—"
    if abs(value) >= 1000:
        return "%.3g" % value
    return _fmt(value, 3)


def _render_row_extended(row):
    split = row["access"]["split"]
    dim = row.get("dim")
    dim_cell = str(dim) if dim is not None else "—"
    if row["access"]["id"] == "split-training-wire-whitebox":
        access = "split-%s wire + white-box (true checkpoint)" % split
    else:
        access = "split-%s training-wire" % split
    if row["status"] == "INCOMPLETE":
        leak = "INCOMPLETE (missing: %s)" % ", ".join(row["missing_fields"])
        util = "INCOMPLETE" if row["utility_cost"] is None else _render_sweep_utility(
            row["utility_cost"])
    elif row["status"] == "NOT_EVALUABLE":
        leak = "NOT_EVALUABLE (reason in JSON)"
        util = "—"
    elif row["leakage"]["metric"] == "forward_recovery_minus_none":
        lk = row["leakage"]
        leak = "fwd−none %s pp [%s, %s]" % (
            _fmt_pp(lk["paired_effect"]),
            _fmt_pp(lk["ci_low"]),
            _fmt_pp(lk["ci_high"]),
        )
        util = _render_sweep_utility(row["utility_cost"])
    else:
        lk = row["leakage"]
        leak = "strongest observed %s (%s)" % (
            _fmt_pp(lk["observed"]), lk["strongest_arm"])
        util = _render_sweep_utility(row["utility_cost"])
    return "| %s | %s | %s | %s | %s | `%s` | %s |" % (
        row["defense"], dim_cell, access, leak, util,
        row["evidence_strength"], row["status"])


def _render_sweep_utility(ut):
    text = "%s [%s, %s]" % (
        _fmt_ppl(ut["paired_effect"]),
        _fmt_ppl(ut["ci_low"]),
        _fmt_ppl(ut["ci_high"]),
    )
    gates = ut.get("gates")
    if gates:
        text += " (gates 1.5x/2.0x: %s/%s)" % (gates.get("1.5x"), gates.get("2.0x"))
    return text


def _render_sweep_selection_section(selection_meta):
    lines = [
        "",
        "## Bottleneck-family sweep selection outcome",
        "",
        "- Outcome: **%s** — %s" % (
            selection_meta["outcome"], selection_meta["negative_statement"]),
        "- Negative splits: %s" % ", ".join(selection_meta["negative_splits"]),
        "- Utility gates (predefined, no retuning): primary %sx, secondary %sx" % (
            selection_meta["gates"]["primary_ppl_ratio"],
            selection_meta["gates"]["secondary_ppl_ratio"]),
        "- Candidates: %s; control: %s" % (
            ", ".join(selection_meta["candidate_methods"]),
            selection_meta["control_method"]),
        "- v3 status: **%s** — %s" % (
            selection_meta["v3_status"], selection_meta["v3_rule"]),
        "",
    ]
    return lines


def render_markdown(matrix):
    extended = any("dim" in row for row in matrix["rows"])
    lines = [
        "# Assurance matrix (adversary-conditioned)",
        "",
        "Defense x attacker access x measured leakage x utility cost x evidence",
        "strength. Per-row measurements only: **no universal risk score, no",
        "maturity model** — the current evidence does not validate either.",
        "Rows with missing leakage or utility fields are INCOMPLETE, never",
        "dropped.",
        "",
        "## Evidence-strength ladder",
        "",
        "| Level | Meaning |",
        "|---|---|",
    ]
    for entry in matrix["meta"]["evidence_ladder"]:
        lines.append("| `%s` | %s |" % (entry["level"], entry["meaning"]))
    if extended:
        lines += [
            "",
            "## Matrix",
            "",
            "| Defense | Dim | Access (split) | Leakage | "
            "Utility: ΔPPL vs none [95% CI] | Evidence strength | Status |",
            "|---|---|---|---|---:|---|---|",
        ]
        for row in matrix["rows"]:
            lines.append(_render_row_extended(row))
        if "sweep_selection" in matrix["meta"]:
            lines += _render_sweep_selection_section(matrix["meta"]["sweep_selection"])
        lines.append("")
        return "\n".join(lines)
    lines += [
        "",
        "## Matrix",
        "",
        "| Defense | Access (split) | Leakage: fwd recovery − none pp [95% CI] | "
        "Utility: ΔPPL vs none [95% CI] | Evidence strength | Status |",
        "|---|---|---:|---:|---|---|",
    ]
    for row in matrix["rows"]:
        split = row["access"]["split"]
        if row["status"] == "INCOMPLETE":
            leak = "INCOMPLETE (missing: %s)" % ", ".join(row["missing_fields"])
            util = "INCOMPLETE"
        else:
            lk = row["leakage"]
            leak = "%s [%s, %s]" % (
                _fmt_pp(lk["paired_effect"]),
                _fmt_pp(lk["ci_low"]),
                _fmt_pp(lk["ci_high"]),
            )
            ut = row["utility_cost"]
            util = "%s [%s, %s]" % (
                _fmt_ppl(ut["paired_effect"]),
                _fmt_ppl(ut["ci_low"]),
                _fmt_ppl(ut["ci_high"]),
            )
        lines.append(
            "| %s | split-%s training-wire | %s | %s | `%s` | %s |"
            % (row["defense"], split, leak, util, row["evidence_strength"], row["status"])
        )
    lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", required=True, help="defended-pilot analysis.json")
    parser.add_argument(
        "--sweep-analysis",
        default=None,
        help="bottleneck-sweep analysis.json (requires --sweep-selection)",
    )
    parser.add_argument(
        "--sweep-selection",
        default=None,
        help="bottleneck-sweep selection.json (requires --sweep-analysis)",
    )
    parser.add_argument("--output", required=True, help="matrix JSON output path")
    parser.add_argument(
        "--markdown",
        default=None,
        help="markdown rendering path (default: --output with .md suffix)",
    )
    args = parser.parse_args(argv)

    if (args.sweep_analysis is None) != (args.sweep_selection is None):
        parser.error("--sweep-analysis and --sweep-selection must be given together")

    with open(args.pilot, "r", encoding="utf-8") as fh:
        pilot = json.load(fh)
    matrix = build_from_pilot(pilot)

    if args.sweep_analysis is not None:
        with open(args.sweep_analysis, "r", encoding="utf-8") as fh:
            sweep_analysis = json.load(fh)
        with open(args.sweep_selection, "r", encoding="utf-8") as fh:
            sweep_selection = json.load(fh)
        extend_with_sweep(matrix, sweep_analysis, sweep_selection)

    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(matrix, fh, indent=2, sort_keys=True)
        fh.write("\n")

    md_path = args.markdown
    if md_path is None:
        md_path = args.output[:-5] + ".md" if args.output.endswith(".json") else args.output + ".md"
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(render_markdown(matrix))

    estimated = sum(1 for r in matrix["rows"] if r["status"] == "ESTIMATED")
    incomplete = sum(1 for r in matrix["rows"] if r["status"] == "INCOMPLETE")
    not_evaluable = sum(1 for r in matrix["rows"] if r["status"] == "NOT_EVALUABLE")
    print(
        "assurance matrix: %d rows (%d ESTIMATED, %d INCOMPLETE, %d NOT_EVALUABLE)"
        " -> %s, %s"
        % (len(matrix["rows"]), estimated, incomplete, not_evaluable,
           args.output, md_path)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
