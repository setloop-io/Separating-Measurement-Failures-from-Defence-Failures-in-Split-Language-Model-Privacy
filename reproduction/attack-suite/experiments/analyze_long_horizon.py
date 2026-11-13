"""Aggregate the long-horizon matrix into report.json + report.md.

Reads <workdir>/run-index.jsonl + artifacts/*.json(+ .jsonl journals)
and produces the compact rebuttal tables: per-arm token recovery vs the
majority control at 20 vs 100 steps, Bonferroni-Wilson excess at the
current frozen-gate.v2 (from the artifacts) and recomputed at the
paper's frozen-gate.v1 z=2.62 for comparability with the 20-step
reference numbers, within-horizon recovery windows (drift), loss-trajectory
drift proxies, and the cross-epoch accumulation probe results.

Pure stdlib + attacker.gate (torch-free).
"""
import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from attacker.gate import wilson_upper, bonferroni_z, DEFAULT_GATE  # noqa: E402

#: paper/paper_b/main.tex "Defence matrix" (lines 846-850): 20-step cells,
#: frozen-gate.v1 (z=2.62), WikiText-2 first-steps corpus, battery attack
#: budget (samples=8), GPU fleet. External reference, NOT a paired run.
PAPER_REFERENCE_20STEP = {
    "none": {"excess_pp": 86.0},
    "gaussian-s15": {"excess_pp": 9.49, "loss": 8.6},
    "sparsify-10": {"excess_pp": 55.32},
    "mid-lite-k64": {"excess_pp": 2.08, "loss": 7.6},
    "santext-30": {"excess_pp": 66.08},
    "at-lite-e15": {"excess_pp": 86.17},
}
V1_Z = 2.62  # the paper's gate z (PR-101 later corrected it to 2.5392 = v2)
WINDOWS = ((0, 7), (8, 19), (20, 49), (50, 99))


def load_json(p):
    with open(p) as f:
        return json.load(f)


def excess_at_z(recovery, control, n, z):
    correct = int(round(recovery * n))
    p = correct / n
    den = 1.0 + z * z / n
    center = p + z * z / (2.0 * n)
    radius = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return 100.0 * ((center + radius) / den - control)


def parse_tag(path, prefix):
    name = os.path.basename(path)[len(prefix):-len(".json")]
    arm, rest = name.rsplit("_s", 1)
    seed, horizon = rest.split("_h")
    return arm, int(seed), int(horizon) if horizon.isdigit() else horizon


def seam_rows(workdir):
    rows = []
    for p in sorted(glob.glob(os.path.join(workdir, "artifacts",
                                           "seam-inversion_vmi_*.json"))):
        arm, seed, steps = parse_tag(p, "seam-inversion_vmi_")
        art = load_json(p)
        if not art.get("summary"):
            continue
        s = art["summary"][0]
        rec, ctl, n = s["seq_accuracy"], s["control_accuracy"], s["n_tokens"]
        rows.append({"arm": arm, "seed": seed, "steps": steps,
                     "recovery": rec, "control": ctl, "n": n,
                     "raw_gap_pp": 100.0 * (rec - ctl),
                     "excess_v2_pp": s["gate"]["excess_pp"],
                     "excess_v1z_pp": excess_at_z(rec, ctl, n, V1_Z),
                     "verdict": s["gate"]["verdict"],
                     "cells": [{"step": c["step"], "n_correct": c["n_correct"],
                                "n_tokens": c["n_tokens"],
                                "n_control_correct": c["n_control_correct"]}
                               for c in art["results"] if "step" in c]})
    return rows


def pool(cells):
    nc = sum(c["n_correct"] for c in cells)
    nt = sum(c["n_tokens"] for c in cells)
    cc = sum(c["n_control_correct"] for c in cells)
    return (nc / max(1, nt), cc / max(1, nt), nt)


def window_profile(rows):
    """Per arm: h100 recovery pooled over seeds within step windows."""
    out = {}
    for r in rows:
        if r["steps"] != 100:
            continue
        a = out.setdefault(r["arm"], {w: [] for w in WINDOWS})
        for w in WINDOWS:
            a[w] += [c for c in r["cells"] if w[0] <= c["step"] <= w[1]]
    prof = {}
    for arm, ws in out.items():
        prof[arm] = {}
        for w, cells in ws.items():
            rec, ctl, n = pool(cells)
            prof[arm][f"{w[0]}-{w[1]}"] = {
                "recovery": round(rec, 4), "control": round(ctl, 4),
                "n": n, "excess_v2_pp": round(
                    100.0 * (wilson_upper(int(round(rec * n)), n) - ctl), 2)
                if n else None}
    return prof


def drift_rows(workdir):
    rows = []
    for p in sorted(glob.glob(os.path.join(workdir, "artifacts",
                                           "splitbench_*.json"))):
        arm, seed, steps = parse_tag(p, "splitbench_")
        art = load_json(p)
        if not art.get("results") or "final_loss" not in art["results"][0]:
            continue
        r0 = art["results"][0]
        losses = []
        jl = p + ".jsonl"
        if os.path.isfile(jl):
            for line in open(jl):
                try:
                    losses.append(json.loads(line))
                except ValueError:
                    pass
        by_epoch = {}
        for rec in losses:
            by_epoch.setdefault(rec.get("epoch"), []).append(rec["loss"])
        rows.append({"arm": arm, "seed": seed, "steps": steps,
                     "loss_step0": losses[0]["loss"] if losses else None,
                     "loss_final": r0["final_loss"],
                     "mean_loss": r0["mean_loss"],
                     "epoch_mean_loss": {str(k): round(sum(v) / len(v), 4)
                                         for k, v in sorted(by_epoch.items())},
                     "captures_written": r0["captures_written"]})
    return rows


def label_rows(workdir):
    rows = []
    for p in sorted(glob.glob(os.path.join(workdir, "artifacts",
                                           "label-scoring_norm_*.json"))):
        arm, seed, steps = parse_tag(p, "label-scoring_norm_")
        art = load_json(p)
        for s in art.get("summary", []):
            if s.get("kind") == "norm_stats":
                rows.append({"arm": arm, "seed": seed, "steps": steps,
                             "norm_mean": s["norm_mean"],
                             "norm_std": s["norm_std"]})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workdir", default="results/long-horizon-v1/matrix")
    ap.add_argument("--out-prefix", default="results/long-horizon-v1/report")
    args = ap.parse_args()

    cells = []
    jl = os.path.join(args.workdir, "run-index.jsonl")
    if os.path.isfile(jl):
        cells = [json.loads(x) for x in open(jl) if x.strip()]
    timing = {"n_cells": len(cells),
              "wall_total_s": round(sum(c.get("wall_s", 0) for c in cells), 1),
              "by_kind": {},
              "errors": [c for c in cells if c.get("rc") not in (0, None)]}
    for c in cells:
        k = timing["by_kind"].setdefault(c.get("kind", "?"),
                                         {"n": 0, "wall_s": 0.0})
        k["n"] += 1
        k["wall_s"] = round(k["wall_s"] + c.get("wall_s", 0), 1)

    rows = seam_rows(args.workdir)
    arms = sorted({r["arm"] for r in rows})
    paired = {}
    for arm in arms:
        ent = {}
        for steps in (20, 100):
            sel = [r for r in rows if r["arm"] == arm and r["steps"] == steps]
            if not sel:
                continue
            recs = [r["recovery"] for r in sel]
            ent[str(steps)] = {
                "seeds": [r["seed"] for r in sel],
                "recovery_mean": round(sum(recs) / len(recs), 4),
                "recovery_sd": round(
                    (sum((x - sum(recs) / len(recs)) ** 2 for x in recs)
                     / max(1, len(recs) - 1)) ** 0.5, 4),
                "control_mean": round(sum(r["control"] for r in sel)
                                      / len(sel), 4),
                "excess_v2_pp_mean": round(sum(r["excess_v2_pp"] for r in sel)
                                           / len(sel), 2),
                "excess_v1z_pp_mean": round(sum(r["excess_v1z_pp"] for r in sel)
                                            / len(sel), 2),
                "raw_gap_pp_mean": round(sum(r["raw_gap_pp"] for r in sel)
                                         / len(sel), 2),
                "n_tokens_per_run": sel[0]["n"]}
        if "20" in ent and "100" in ent:
            ent["delta_recovery_pp"] = round(
                100.0 * (ent["100"]["recovery_mean"]
                         - ent["20"]["recovery_mean"]), 2)
        ent["paper_20step_v1"] = PAPER_REFERENCE_20STEP.get(arm)
        paired[arm] = ent

    probes = {}
    for p in sorted(glob.glob(os.path.join(args.workdir, "artifacts",
                                           "accumulation_*.json"))) + \
            sorted(glob.glob(os.path.join(args.workdir, "artifacts",
                                          "aligned_*.json"))):
        probes[os.path.basename(p)] = load_json(p)

    report = {"timing": timing, "paired": paired,
              "windows_h100": window_profile(rows),
              "drift": drift_rows(args.workdir),
              "label_scoring": label_rows(args.workdir),
              "probes": probes,
              "gate": {"v2_z": bonferroni_z(DEFAULT_GATE), "v1_z": V1_Z,
                       "note": "v1 excess recomputed from artifact "
                               "recovery/control/n at the paper's z=2.62"},
              "paper_reference": PAPER_REFERENCE_20STEP,
              "per_run": [{k: v for k, v in r.items() if k != "cells"}
                          for r in rows]}
    out_json = args.out_prefix + ".json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=1)

    lines = ["# Long-horizon defense matrix (GPT-2 @6/12, 100 steps)",
             "",
             f"cells: {timing['n_cells']}, wall {timing['wall_total_s']}s, "
             f"errors: {len(timing['errors'])}",
             "",
             "| arm | steps | seeds | recovery | control | raw gap pp | "
             "excess v2 pp | excess v1z pp | paper 20-step (v1) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for arm in arms:
        ent = paired[arm]
        ref = ent.get("paper_20step_v1") or {}
        for steps in ("20", "100"):
            if steps not in ent:
                continue
            e = ent[steps]
            lines.append(
                f"| {arm} | {steps} | {len(e['seeds'])} | "
                f"{100*e['recovery_mean']:.2f}% | {100*e['control_mean']:.2f}% | "
                f"{e['raw_gap_pp_mean']:+.2f} | {e['excess_v2_pp_mean']:+.2f} | "
                f"{e['excess_v1z_pp_mean']:+.2f} | "
                f"{ref.get('excess_pp', '—') if steps == '20' else ''} |")
        if "delta_recovery_pp" in ent:
            lines.append(f"| {arm} | Δ100−20 | | | | "
                         f"{ent['delta_recovery_pp']:+.2f} rec | | | |")
    lines += ["", "## Within-horizon windows (h100, pooled over seeds)", "",
              "| arm | window | recovery | control | n | excess v2 pp |",
              "|---|---|---|---|---|---|"]
    for arm, ws in report["windows_h100"].items():
        for w, d in ws.items():
            lines.append(f"| {arm} | {w} | {100*d['recovery']:.2f}% | "
                         f"{100*d['control']:.2f}% | {d['n']} | "
                         f"{d['excess_v2_pp']} |")
    lines += ["", "## Drift proxies (loss trajectory)", "",
              "| arm | steps | seed | loss step0 | final | mean |",
              "|---|---|---|---|---|---|"]
    for d in report["drift"]:
        lines.append(f"| {d['arm']} | {d['steps']} | {d['seed']} | "
                     f"{d['loss_step0']:.3f} | {d['loss_final']:.3f} | "
                     f"{d['mean_loss']:.3f} |")
    with open(args.out_prefix + ".md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[analyze] wrote {out_json} and {args.out_prefix}.md "
          f"({len(rows)} seam runs, {len(timing['errors'])} errors)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
