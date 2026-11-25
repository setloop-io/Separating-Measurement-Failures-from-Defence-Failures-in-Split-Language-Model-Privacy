#!/usr/bin/env python3
"""Domain-shift-v1 gate statistics (Track A rebuttal).

Reads the sealed attacker.experiment runs under results/domain-shift-v1/
and scores each corpus x mode cell against the frozen gate
(attacker.gate as a library: Bonferroni-Wilson upper-95 excess over the
label-free majority control, z=2.5392, threshold +1.0 pp, floor
reported). Torch-free.

Writes results/domain-shift-v1/domain-shift-v1-summary.{json,md}.
"""
import argparse
import json
from pathlib import Path

from attacker.gate import DEFAULT_GATE, bonferroni_z, verdict


def cell_stats(recs, gate=DEFAULT_GATE):
    """One corpus x mode cell: per-seed means plus the token-pooled gate.

    recs: experiment.json result records for a fixed (corpus, mode,
    defense, arm). Token counts are summed exactly from per_sequence."""
    n_tok = sum(r["n_tokens"] for r in recs)
    n_corr = sum(q["correct"] for r in recs for q in r["per_sequence"])
    n_ctl = sum(q["control_correct"] for r in recs
                for q in r["per_sequence"])
    recovery = n_corr / n_tok
    control = n_ctl / n_tok
    g = verdict(recovery, control, n_tok, config=gate,
                n_seeds=len({r["seed"] for r in recs}))
    return {
        "n_seeds": len({r["seed"] for r in recs}),
        "seeds": sorted({r["seed"] for r in recs}),
        "n_tokens": n_tok,
        "n_sequences": sum(r["n_sequences"] for r in recs),
        "recovery_pooled": round(recovery, 6),
        "control_pooled": round(control, 6),
        "recovery_mean_over_seeds": round(
            sum(r["accuracy"] for r in recs) / len(recs), 6),
        "control_mean_over_seeds": round(
            sum(r["control_accuracy"] for r in recs) / len(recs), 6),
        "upper95_pct": g["upper95_pct"],
        "excess_pp": g["excess_pp"],
        "floor_pp": g["floor_pp"],
        "at_floor": g["at_floor"],
        "gate_verdict": g["verdict"],
        "attack_seconds_total": round(
            sum(r["attack_seconds"] for r in recs), 2)}


def analyze(run_dirs, gate=DEFAULT_GATE):
    out = {"schema": "llm_attacker.domain_shift_summary.v1",
           "gate": {"version": gate.version, "z": bonferroni_z(gate),
                    "gate_pp": gate.gate_pp,
                    "n_seeds_required": gate.n_seeds_required,
                    "control": gate.control},
           "corpora": {}}
    for name, root in run_dirs.items():
        report = json.loads((Path(root) / "experiment.json").read_text())
        if report.get("status") != "COMPLETE":
            raise ValueError(f"{root}: status {report.get('status')}")
        recs = [r for r in report["results"] if r["arm"] == "vmi"]
        defenses = sorted({r["defense"] for r in recs})
        cells = {}
        for mode in ("training", "inference"):
            for defense in defenses:
                sel = [r for r in recs
                       if r["mode"] == mode and r["defense"] == defense]
                if sel:
                    cells[f"{mode}/{defense}"] = cell_stats(sel, gate)
        paired = {s["mode"]: s for s in report["summary"]
                  if s["arm"] == "vmi" and s["defense"] != "none"
                  and s["defense_minus_undefended_pp"]["mean"] is not None}
        out["corpora"][name] = {
            "run_dir": str(root),
            "corpus": report["config"]["corpus"],
            "corpus_sha256": report["corpus_sha256"],
            "split": report["config"]["split"],
            "seq_len": report["config"]["seq_len"],
            "train_steps": report["config"]["train_steps"],
            "attack_steps": report["config"]["attack_steps"],
            "wall_seconds": round(report["wall_seconds"], 1),
            "n_docs": sum(len(v) for v in report["partitions"].values()),
            "defense_paired_pp": {
                m: {"mean": s["defense_minus_undefended_pp"]["mean"],
                    "lower": s["defense_minus_undefended_pp"]["lower"],
                    "upper": s["defense_minus_undefended_pp"]["upper"],
                    "defense": s["defense"]}
                for m, s in paired.items()},
            "cells": cells}
    return out


def render_md(summary):
    lines = ["# domain-shift-v1: seam inversion across domains (GPT-2 "
             "split 6/12, vmi arm)",
             "",
             f"Gate: {summary['gate']['version']} (z="
             f"{summary['gate']['z']}, threshold +"
             f"{summary['gate']['gate_pp']} pp, >="
             f"{summary['gate']['n_seeds_required']} seeds; control = "
             f"{summary['gate']['control']}).",
             "",
             "| corpus | mode | docs | n tok | control % | recovery % "
             "(mean seeds) | upper-95 excess pp | floor pp | verdict | "
             "wall s |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, c in summary["corpora"].items():
        for mode_def, s in c["cells"].items():
            lines.append(
                f"| {name} | {mode_def} | {c['n_docs']} | {s['n_tokens']} "
                f"| {100 * s['control_pooled']:.2f} "
                f"| {100 * s['recovery_mean_over_seeds']:.2f} "
                f"| {s['excess_pp']:.3f} | {s['floor_pp']:.3f} "
                f"| {s['gate_verdict']} | {c['wall_seconds']} |")
        for m, d in c.get("defense_paired_pp", {}).items():
            lines.append(
                f"| {name} | {m}: {d['defense']} minus none (paired) | "
                f"{c['n_docs']} | - | - | {d['mean']:.2f} pp "
                f"[{d['lower']:.2f}, {d['upper']:.2f}] | - | - | - "
                f"| {c['wall_seconds']} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", required=True,
                    help="name=run_dir pairs")
    ap.add_argument("--output",
                    default="results/domain-shift-v1/"
                            "domain-shift-v1-summary.json")
    args = ap.parse_args(argv)
    run_dirs = dict(pair.split("=", 1) for pair in args.runs)
    summary = analyze(run_dirs)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    out.with_suffix(".md").write_text(render_md(summary))
    print(render_md(summary))
    print(f"[domain-shift] wrote {out} and {out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
