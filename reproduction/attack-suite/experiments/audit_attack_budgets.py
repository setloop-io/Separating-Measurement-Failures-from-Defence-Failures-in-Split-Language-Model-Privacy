#!/usr/bin/env python3
"""Audit that recorded attack work counters match the declared budgets.

Walks the defended-pilot and bottleneck-sweep results trees (including the
whitebox tier) and, for every attack cell, compares RECORDED counters in the
cell.json attack blocks against the DECLARED budgets in the tree's access
spec / plan. Fail-closed: any shortfall or mismatch flags the cell and the
tool exits nonzero. Cells that cannot be checked (e.g. NOT_EVALUABLE
whitebox cells) are accounted for explicitly as UNCHECKABLE with a reason.

Pure stdlib. Works from JSON metadata only (no *.pt artifacts needed).

Report schema: llm_attacker.budget_audit.v1
"""

import argparse
import json
import sys
from pathlib import Path

SCHEMA = "llm_attacker.budget_audit.v1"

# Counters that are exact integers by construction: zero tolerance.
# head_forward_token_positions / vocabulary_dot_products depend on prefix
# reuse and candidate-scoring internals, so they are only checked for being
# positive (documented per check below); the binding checks are on calls,
# sequences and the declared knobs.

DEFENDED_PILOT = {
    "plan_schema": "llm_attacker.defended_pilot.v1",
    "cell_key": "defense",
    "budget_keys": {
        "candidate_count": "forward_candidates",
        "beam_width": "beam_width",
        "steps_per_token": "sipit_steps",
        "n_sequences": "attack_sequences",
    },
}

BOTTLENECK_SWEEP = {
    "plan_schema": "llm_attacker.bottleneck_sweep.v1",
    "cell_key": "method",
    "budget_keys": {
        "candidate_count": "forward_candidates",
        "beam_width": "beam_width",
        "steps_per_token": "sipit_steps",
        "n_sequences": "attack_sequences",
    },
}


def _load(path):
    with open(path) as fh:
        return json.load(fh)


def _check(eq_checks, pos_checks):
    """Return list of failure strings. eq_checks: (name, actual, expected)."""
    failures = []
    for name, actual, expected in eq_checks:
        if actual is None:
            failures.append(f"{name}: MISSING (expected {expected!r})")
        elif actual != expected:
            failures.append(f"{name}: recorded {actual!r} != declared {expected!r}")
    for name, actual in pos_checks:
        if not isinstance(actual, (int, float)) or actual <= 0:
            failures.append(f"{name}: recorded {actual!r} is not positive")
    return failures


def _audit_forward(block, declared, failures):
    work = block.get("work") or {}
    failures.extend(_check(
        [
            ("forward.candidate_count", block.get("candidate_count"),
             declared["candidate_count"]),
            ("forward.beam_width", block.get("beam_width"),
             declared["beam_width"]),
            ("forward.n_sequences", block.get("n_sequences"),
             declared["n_sequences"]),
            # beam search scores each sequence with one batched head call
            ("forward.work.head_forward_calls",
             work.get("head_forward_calls"), declared["n_sequences"]),
            # forward/beam never backprops through the head
            ("forward.work.head_backward_calls",
             work.get("head_backward_calls"), 0),
        ],
        [
            # approximate: depends on prefix-reuse during beam expansion
            ("forward.work.head_forward_token_positions",
             work.get("head_forward_token_positions")),
            ("forward.work.vocabulary_dot_products",
             work.get("vocabulary_dot_products")),
        ],
    ))


def _audit_sipit(block, declared, failures):
    work = block.get("work") or {}
    steps = block.get("steps_per_token")
    n_seq = block.get("n_sequences")
    failures.extend(_check(
        [
            ("sipit.steps_per_token", steps, declared["steps_per_token"]),
            ("sipit.n_sequences", n_seq, declared["n_sequences"]),
        ],
        [
            ("sipit.work.head_forward_token_positions",
             work.get("head_forward_token_positions")),
        ],
    ))
    if isinstance(steps, int) and isinstance(n_seq, int):
        # one backward per (sequence, step); one gradient + one candidate
        # scoring forward per backward
        failures.extend(_check(
            [
                ("sipit.work.head_backward_calls",
                 work.get("head_backward_calls"), n_seq * steps),
                ("sipit.work.head_forward_calls",
                 work.get("head_forward_calls"), 2 * n_seq * steps),
            ],
            [],
        ))


def audit_cell(cell_path, declared, tree_label, tree_root):
    """Audit one cell.json. Returns a cell record dict."""
    cell = _load(cell_path)
    record = {
        "cell": str(cell_path),
        "cell_id": f"{tree_label}:{cell_path.parent.relative_to(tree_root)}",
        "status": cell.get("status"),
        "verdict": "PASS",
        "failures": [],
        "notes": [],
    }
    if cell.get("status") != "COMPLETE":
        reason = cell.get("not_evaluable_reason") or "no attack blocks recorded"
        record["verdict"] = "UNCHECKABLE"
        record["notes"].append(
            f"status={cell.get('status')}: {reason[:200]}")
        return record

    access = cell.get("access") or {}
    qb = access.get("query_budgets") or {}
    cell_declared = {
        "candidate_count": (qb.get("forward_beam") or {}).get(
            "candidate_count", declared["candidate_count"]),
        "beam_width": (qb.get("forward_beam") or {}).get(
            "beam_width", declared["beam_width"]),
        "steps_per_token": (qb.get("sipit") or {}).get(
            "steps_per_token", declared["steps_per_token"]),
        "n_sequences": qb.get("scored_sequences", declared["n_sequences"]),
    }
    # The cell-embedded access spec must not drift from the tree declaration.
    for key, val in cell_declared.items():
        if val != declared[key]:
            record["failures"].append(
                f"access.query_budgets.{key}: cell declares {val!r} "
                f"but tree access spec declares {declared[key]!r}")

    attacks = cell.get("attacks") or {}
    whitebox = "whitebox_inverter" in attacks or \
        "whitebox_optimization" in attacks
    if not whitebox:
        if "forward" not in attacks:
            record["failures"].append("attacks.forward: MISSING")
        else:
            _audit_forward(attacks["forward"], cell_declared,
                           record["failures"])
        if "sipit" not in attacks:
            record["failures"].append("attacks.sipit: MISSING")
        else:
            _audit_sipit(attacks["sipit"], cell_declared,
                         record["failures"])

    # Defense-aware arm (sweep cells only): the cell records only the
    # declared knobs and accuracy for this arm, not work counters, so the
    # knobs are audited exactly and the missing counters are noted as
    # UNCHECKABLE-detail rather than a shortfall.
    da = attacks.get("defense_aware")
    if da is not None:
        if not da.get("defense_aware"):
            record["failures"].append("defense_aware.defense_aware is not true")
        if da.get("surrogate") != "pca_encoder_inverse_on_public_calibration":
            record["failures"].append(
                f"defense_aware.surrogate: {da.get('surrogate')!r}")
        da_f = da.get("forward") or {}
        da_s = da.get("sipit") or {}
        record["failures"].extend(_check(
            [
                ("defense_aware.forward.candidate_count",
                 da_f.get("candidate_count"), cell_declared["candidate_count"]),
                ("defense_aware.forward.beam_width",
                 da_f.get("beam_width"), cell_declared["beam_width"]),
                ("defense_aware.forward.n_sequences",
                 da_f.get("n_sequences"), cell_declared["n_sequences"]),
                ("defense_aware.sipit.steps_per_token",
                 da_s.get("steps_per_token"), cell_declared["steps_per_token"]),
                ("defense_aware.sipit.n_sequences",
                 da_s.get("n_sequences"), cell_declared["n_sequences"]),
            ],
            [],
        ))
        record["notes"].append(
            "defense-aware arm records no work counters in the cell; "
            "declared knobs audited, execution counters UNCHECKABLE for "
            "this arm")

    # Whitebox arms: structural budget matching against the SIPIT step count.
    wb_opt = attacks.get("whitebox_optimization")
    if wb_opt is not None:
        if wb_opt.get("steps_per_wire") != cell_declared["steps_per_token"]:
            record["failures"].append(
                "whitebox_optimization.steps_per_wire: recorded "
                f"{wb_opt.get('steps_per_wire')!r} != declared structural "
                f"match {cell_declared['steps_per_token']!r}")
        if not wb_opt.get("true_encoder_granted"):
            record["failures"].append(
                "whitebox_optimization.true_encoder_granted is not true")
        _audit_forward(wb_opt.get("forward") or {}, cell_declared,
                       record["failures"])
    wb_inv = attacks.get("whitebox_inverter")
    if wb_inv is not None:
        if not wb_inv.get("true_encoder_granted"):
            record["failures"].append(
                "whitebox_inverter.true_encoder_granted is not true")
        if not isinstance((wb_inv.get("inverter") or {}).get("steps"), int) \
                or wb_inv["inverter"]["steps"] <= 0:
            record["failures"].append(
                "whitebox_inverter.inverter.steps: "
                f"{(wb_inv.get('inverter') or {}).get('steps')!r}")
        _audit_forward(wb_inv.get("forward") or {}, cell_declared,
                       record["failures"])
        _audit_sipit(wb_inv.get("sipit") or {}, cell_declared,
                     record["failures"])

    # Defense-specific provenance knobs (pilot tree).
    prov = cell.get("provenance") or {}
    prov_path = cell_path.parent / (prov.get("path") or "")
    defense = cell.get("defense")
    if prov.get("path") and prov_path.exists():
        sidecar = _load(prov_path)
        if defense == "secret-mixup":
            if not sidecar.get("blocks"):
                record["failures"].append(
                    "secret-mixup provenance: no mixup blocks recorded")
        elif defense == "replica-median":
            for knob in ("replica_count", "tolerance"):
                if sidecar.get(knob) is None and \
                        (sidecar.get("config") or {}).get(knob) is None:
                    record["failures"].append(
                        f"replica-median provenance: knob {knob!r} missing")
        elif defense == "trained-bottleneck":
            for knob in ("bottleneck_dim", "config"):
                if sidecar.get(knob) is None:
                    record["failures"].append(
                        f"trained-bottleneck provenance: knob {knob!r} missing")
            cfg = sidecar.get("config") or {}
            if cfg.get("minimax_steps") is None:
                record["failures"].append(
                    "trained-bottleneck provenance: config.minimax_steps missing")
        elif defense == "gaussian":
            # provenance points at run.json; sigma must be recorded there
            results = sidecar.get("results") or sidecar.get("summary") or []
            sigma = results[0].get("sigma") if results else None
            if sigma is None:
                record["failures"].append(
                    "gaussian provenance: run.json records no sigma")
    elif defense in ("secret-mixup", "replica-median", "trained-bottleneck"):
        record["failures"].append(
            f"{defense} provenance sidecar missing at {prov.get('path')!r}")

    if record["failures"]:
        record["verdict"] = "FAIL"
    return record


def audit_tree(root, spec, label):
    root = Path(root)
    cells = []
    plan_path = root / "plan.json"
    access_path = root / "access.json"
    if not plan_path.exists():
        return None
    plan = _load(plan_path)
    budgets = plan.get("budgets") or {}
    keys = spec["budget_keys"]
    declared = {k: budgets.get(v) for k, v in keys.items()}
    if access_path.exists():
        qb = (_load(access_path).get("query_budgets") or {})
        fb = qb.get("forward_beam") or {}
        for k, v in [("candidate_count", fb.get("candidate_count")),
                     ("beam_width", fb.get("beam_width")),
                     ("steps_per_token", (qb.get("sipit") or {})
                      .get("steps_per_token")),
                     ("n_sequences", qb.get("scored_sequences"))]:
            if v is not None and declared[k] is not None and v != declared[k]:
                cells.append({
                    "cell": str(access_path), "cell_id": f"{label}:access",
                    "status": "COMPLETE", "verdict": "FAIL",
                    "failures": [
                        f"access spec {k}={v!r} != plan budget {declared[k]!r}"],
                    "notes": []})
    for cell_info in plan.get("cells") or []:
        cell_path = root / cell_info["path"] / "cell.json"
        if not cell_path.exists():
            cells.append({
                "cell": str(cell_path),
                "cell_id": f"{label}:{cell_info['path']}",
                "status": "MISSING", "verdict": "FAIL",
                "failures": ["cell.json listed in plan but absent on disk"],
                "notes": []})
            continue
        cells.append(audit_cell(cell_path, declared, label, root))
    return cells, declared


def audit_sweep(root):
    """Bottleneck sweep: base matrix plus the whitebox tier."""
    root = Path(root)
    out = audit_tree(root, BOTTLENECK_SWEEP, "bottleneck-sweep-v1")
    if out is None:
        return None
    cells, declared = out
    wb_plan_path = root / "whitebox" / "plan.json"
    if wb_plan_path.exists():
        wb_plan = _load(wb_plan_path)
        wb_budgets = wb_plan.get("budgets") or {}
        wb_declared = {
            "candidate_count": wb_budgets.get("forward_candidates"),
            "beam_width": wb_budgets.get("beam_width"),
            "steps_per_token": wb_budgets.get("sipit_steps"),
            "n_sequences": wb_budgets.get("attack_sequences"),
        }
        # Whitebox budgets are declared structurally matched to the base.
        for k in wb_declared:
            if wb_declared[k] != declared[k]:
                cells.append({
                    "cell": str(wb_plan_path),
                    "cell_id": "bottleneck-sweep-v1:whitebox/plan",
                    "status": "COMPLETE", "verdict": "FAIL",
                    "failures": [
                        f"whitebox plan budget {k}={wb_declared[k]!r} != "
                        f"base sweep {declared[k]!r} (structural match "
                        "declared)"],
                    "notes": []})
        for cell_info in wb_plan.get("cells") or []:
            cell_path = root / cell_info["path"] / "cell.json"
            if not cell_path.exists():
                cells.append({
                    "cell": str(cell_path),
                    "cell_id": f"bottleneck-sweep-v1:{cell_info['path']}",
                    "status": "MISSING", "verdict": "FAIL",
                    "failures": ["whitebox cell.json listed in plan but absent"],
                    "notes": []})
                continue
            cells.append(audit_cell(cell_path, wb_declared,
                                    "bottleneck-sweep-v1", root))
    return cells, declared


def render_markdown(report):
    lines = [
        "# Attacker budget-execution audit",
        "",
        f"schema: `{report['schema']}`",
        f"verdict: **{report['verdict']}**",
        "",
        "| tree | cells | PASS | FAIL | UNCHECKABLE |",
        "|---|---|---|---|---|",
    ]
    for tree in report["trees"]:
        lines.append(
            f"| {tree['tree']} | {tree['cells']} | {tree['pass']} "
            f"| {tree['fail']} | {tree['uncheckable']} |")
    lines.append("")
    failed = [c for t in report["trees"] for c in t["records"]
              if c["verdict"] == "FAIL"]
    if failed:
        lines.append("## Shortfalls / mismatches")
        lines.append("")
        for c in failed:
            lines.append(f"### `{c['cell_id']}`")
            for f in c["failures"]:
                lines.append(f"- {f}")
            lines.append("")
    else:
        lines.append("No budget shortfalls or counter mismatches found.")
        lines.append("")
    unch = [c for t in report["trees"] for c in t["records"]
            if c["verdict"] == "UNCHECKABLE"]
    if unch:
        lines.append("## UNCHECKABLE cells")
        lines.append("")
        for c in unch:
            lines.append(f"- `{c['cell_id']}` — {c['notes'][0]}")
        lines.append("")
    lines.append("## Tolerances")
    lines.append("")
    lines.append(
        "- `head_forward_calls`, `head_backward_calls`, `n_sequences`, "
        "`steps_per_token`, `candidate_count`, `beam_width`, "
        "`steps_per_wire`: exact integer match, zero tolerance.")
    lines.append(
        "- `head_forward_token_positions` and `vocabulary_dot_products` "
        "depend on prefix reuse and candidate-scoring internals; checked "
        "only for presence and positivity.")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", default="results/pilot-defended-v1")
    parser.add_argument("--sweep", default="results/bottleneck-sweep-v1")
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--md-out", default=None)
    args = parser.parse_args(argv)

    report = {"schema": SCHEMA, "verdict": "PASS", "trees": []}
    for label, path, fn in [
        ("pilot-defended-v1", args.pilot,
         lambda r: audit_tree(r, DEFENDED_PILOT, "pilot-defended-v1")),
        ("bottleneck-sweep-v1", args.sweep, audit_sweep),
    ]:
        root = Path(path)
        if not root.exists():
            report["trees"].append({
                "tree": label, "path": str(root), "present": False,
                "cells": 0, "pass": 0, "fail": 0, "uncheckable": 0,
                "records": [],
            })
            continue
        result = fn(root)
        cells, declared = result
        n_pass = sum(1 for c in cells if c["verdict"] == "PASS")
        n_fail = sum(1 for c in cells if c["verdict"] == "FAIL")
        n_un = sum(1 for c in cells if c["verdict"] == "UNCHECKABLE")
        report["trees"].append({
            "tree": label, "path": str(root), "present": True,
            "declared_budgets": declared,
            "cells": len(cells), "pass": n_pass, "fail": n_fail,
            "uncheckable": n_un,
            "records": cells,
        })
        if n_fail:
            report["verdict"] = "FAIL"

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=1) + "\n")
    if args.md_out:
        Path(args.md_out).write_text(render_markdown(report))
    print(json.dumps({t["tree"]: {k: t[k] for k in
                                  ("present", "cells", "pass", "fail",
                                   "uncheckable")}
                      for t in report["trees"]}, indent=1))
    print(f"verdict: {report['verdict']}")
    return 1 if report["verdict"] == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
