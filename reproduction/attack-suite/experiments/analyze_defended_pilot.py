"""Analyze the defended development pilot (results/pilot-defended-v1).

Fail-closed contracts, mirroring analyze_pilot.py conventions:
  * article-hash disjointness of the development corpus against the
    frozen corpus manifest is re-asserted before any table is written
    (strict pilot-corpus identity check for development_pilot tier);
  * the attacker-access spec must be identical across all defense
    conditions (matched access) — drift aborts the analysis;
  * every planned cell is accounted for: a cell that is missing,
    incomplete, or lacks provenance / utility / access-spec / common
    baselines is reported FAILED in the failures-and-nulls section, never
    silently dropped — and any failure makes the analysis raise after the
    report is written.

Statistics: paired defense-minus-none differences over victim seeds with
percentile bootstrap confidence intervals (fixed seed, conditional on
this fixed article sample; not population guarantees).
"""
import argparse
import csv
import json
from pathlib import Path
import random
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from attacker.evidence import file_sha256  # noqa: E402
from run_defended_pilot import (  # noqa: E402
    DEFENSES, assert_development_corpus_disjoint)

SCHEMA = "llm_attacker.defended_pilot_analysis.v1"
REQUIRED_CELL_KEYS = ("wall_seconds", "attack_seconds",
                      "cuda_peak_allocated_bytes", "utility_before",
                      "utility_after", "access", "provenance", "attacks")


def percentile_bootstrap(diffs, replicates=10000, seed=2026, alpha=0.05):
    """Percentile bootstrap CI of the mean paired difference over seeds."""
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


def check_matched_access(cells):
    """Every cell's attacker-access spec must be byte-identical."""
    seen = {}
    for cell in cells:
        access = cell.get("access")
        key = json.dumps(access, sort_keys=True)
        seen.setdefault(key, []).append(
            f"split{cell.get('split')}/seed-{cell.get('seed')}/{cell.get('defense')}")
    if len(seen) > 1:
        raise ValueError("attacker-access spec is not matched across "
                         f"defense conditions: {len(seen)} distinct specs")
    return True


def load_cells(root, plan):
    cells, failures = [], []
    for entry in plan["cells"]:
        path = root / entry["path"] / "cell.json"
        label = entry["path"]
        if not path.is_file():
            failures.append({"cell": label, "reason": "cell.json missing"})
            continue
        try:
            cell = json.loads(path.read_text())
        except ValueError as exc:
            failures.append({"cell": label, "reason": f"cell.json invalid: {exc}"})
            continue
        if cell.get("status") != "COMPLETE":
            failures.append({"cell": label,
                             "reason": f"status {cell.get('status')!r}"})
            continue
        if cell.get('measurement_version') != 2 or cell.get('observation_location') != 'body_input_seam':
            failures.append({'cell': label, 'reason': 'legacy or mismatched observation/utility measurement; corrected rerun required'})
            continue
        missing = [k for k in REQUIRED_CELL_KEYS if k not in cell]
        if missing:
            failures.append({"cell": label,
                             "reason": f"missing keys: {missing}"})
            continue
        prov = cell.get("provenance") or {}
        prov_path = path.parent / str(prov.get("path", ""))
        if not prov.get("path") or not prov_path.is_file() or \
                file_sha256(prov_path) != prov.get("sha256"):
            failures.append({"cell": label,
                             "reason": "provenance sidecar missing or hash "
                                       "mismatch"})
            continue
        for split_name in ("utility_before", "utility_after"):
            util = cell.get(split_name) or {}
            expected_stage = ('before' if split_name == 'utility_before' else 'after') + '_victim_training'
            if util.get('evaluation_contract') != 'matched-final-state-v2' or util.get('partition') != 'utility_eval' or util.get('stage') != expected_stage:
                failures.append({'cell': label, 'reason': 'unmatched utility stage/partition contract'})
                break
            if not isinstance(util.get("cross_entropy"), (int, float)) or \
                    not isinstance(util.get("perplexity"), (int, float)):
                failures.append({"cell": label,
                                 "reason": f"{split_name} incomplete"})
                break
        else:
            attacks = cell.get("attacks") or {}
            if not all(isinstance(attacks.get(a), dict)
                       and isinstance(attacks[a].get("accuracy"), (int, float))
                       for a in ("forward", "sipit")):
                failures.append({"cell": label,
                                 "reason": "common forward/SIPIT baselines "
                                           "missing"})
            elif cell["defense"] == "trained-bottleneck" and \
                    "representation-privacy" not in attacks:
                failures.append({"cell": label, "reason": "representation-privacy consumer missing"})
            elif cell["defense"] == "secret-mixup" and \
                    "view-privacy" not in attacks:
                failures.append({"cell": label, "reason": "view-privacy consumer missing"})
            elif cell["defense"] == "replica-median" and \
                    "replica-integrity" not in attacks:
                failures.append({"cell": label, "reason": "replica-integrity consumer missing"})
            else:
                cells.append(cell)
                continue
        continue
    return cells, failures


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def analyze(root, manifest_path, replicates=10000, seed=2026):
    root = Path(root)
    plan = json.loads((root / "plan.json").read_text())
    if plan.get("schema") != "llm_attacker.defended_pilot.v1":
        raise ValueError("unexpected plan schema")
    corpus = plan["corpus"]["path"]
    corpus_path = Path(corpus)
    if not corpus_path.is_absolute():
        candidate = root / corpus
        corpus_path = candidate if candidate.is_file() else Path(corpus)
    tier = plan.get("tier")
    disjointness = assert_development_corpus_disjoint(
        corpus_path, manifest_path,
        require_excluded_pilot=(tier == "development_pilot"))
    cells, failures = load_cells(root, plan)
    if cells:
        check_matched_access(cells)
    baseline = {(c["split"], c["seed"]): c for c in cells
                if c["defense"] == "none"}
    summary = []
    seed_rows = []
    utility_rows = []
    for split in plan["splits"]:
        for defense in plan["defenses"]:
            group = sorted([c for c in cells
                            if c["split"] == split and c["defense"] == defense],
                           key=lambda c: c["seed"])
            if not group:
                continue
            row = {"split": split, "defense": defense,
                   "seeds": len(group),
                   "forward_accuracy_mean": statistics.mean(
                       c["attacks"]["forward"]["accuracy"] for c in group),
                   "sipit_accuracy_mean": statistics.mean(
                       c["attacks"]["sipit"]["accuracy"] for c in group),
                   "attack_seconds_mean": statistics.mean(
                       c["attack_seconds"] for c in group),
                   "wall_seconds_mean": statistics.mean(
                       c["wall_seconds"] for c in group),
                   "ppl_after_mean": statistics.mean(
                       c["utility_after"]["perplexity"] for c in group)}
            for metric in ("forward", "sipit"):
                diffs = [c["attacks"][metric]["accuracy"]
                         - baseline[(split, c["seed"])]["attacks"][metric]["accuracy"]
                         for c in group if (split, c["seed"]) in baseline]
                ci = percentile_bootstrap(diffs, replicates, seed)
                row[f"{metric}_minus_none_mean"] = ci["mean"]
                row[f"{metric}_minus_none_ci_low"] = ci["lower"]
                row[f"{metric}_minus_none_ci_high"] = ci["upper"]
                row[f"{metric}_minus_none_status"] = ci["status"]
            ppl_diffs = [c["utility_after"]["perplexity"]
                         - baseline[(split, c["seed"])]["utility_after"]["perplexity"]
                         for c in group if (split, c["seed"]) in baseline]
            ci = percentile_bootstrap(ppl_diffs, replicates, seed)
            row["ppl_delta_vs_none_mean"] = ci["mean"]
            row["ppl_delta_vs_none_ci_low"] = ci["lower"]
            row["ppl_delta_vs_none_ci_high"] = ci["upper"]
            summary.append(row)
            for c in group:
                seed_rows.append({
                    "split": split, "defense": defense, "seed": c["seed"],
                    "forward_accuracy": c["attacks"]["forward"]["accuracy"],
                    "sipit_accuracy": c["attacks"]["sipit"]["accuracy"],
                    "wall_seconds": c["wall_seconds"],
                    "attack_seconds": c["attack_seconds"],
                    "cuda_peak_allocated_bytes": c["cuda_peak_allocated_bytes"],
                    "final_loss": (c.get("training") or {}).get("final_loss")})
                utility_rows.append({
                    "split": split, "defense": defense, "seed": c["seed"],
                    "ppl_before": c["utility_before"]["perplexity"],
                    "ppl_after": c["utility_after"]["perplexity"],
                    "ce_before": c["utility_before"]["cross_entropy"],
                    "ce_after": c["utility_after"]["cross_entropy"]})
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    if summary:
        write_csv(out / "summary.csv", summary)
    if seed_rows:
        write_csv(out / "seed_metrics.csv", seed_rows)
    if utility_rows:
        write_csv(out / "utility.csv", utility_rows)
    write_csv(out / "failures.csv",
              failures or [{"cell": "", "reason": "none"}])
    lines = ["# Defended development pilot (trained splitbench defenders)", "",
             f"Tier: {tier}. Matrix: {' x '.join(plan['defenses'])} defenses "
             f"x splits {plan['splits']} x seeds {plan['seeds']} on "
             + ("the 96 development wikitext-2 articles."
                if tier == "development_pilot" else
                "a SYNTHETIC smoke corpus (engineering tier; not security "
                "evidence)."), "",
             f"Frozen-corpus separation: development corpus sha256 "
             f"`{disjointness['corpus_sha256']}`; article ids disjoint from "
             "every frozen-corpus partition (asserted fail-closed against "
             f"`{disjointness['frozen_manifest']}`); identity checked "
             f"against excluded_pilot_sha256: {disjointness['identity_checked']}.",
             "",
             "Attacker access is matched across defenses: public calibration "
             "partition, pretrained public head only (final victim head not "
             "granted), identical query budgets. Per-cell access specs were "
             "verified byte-identical.", "",
             "## Recovery (common seam-inversion baselines)", "",
             "Accuracy is position-aligned token recovery on captured "
             "training-phase wires; intervals are percentile bootstrap over "
             f"paired defense-minus-none seed differences (B={replicates}), "
             "conditional on this fixed article sample. Negative means the "
             "defense reduced recovery. Fewer than three seeds gives no "
             "interval (INSUFFICIENT_EVIDENCE), not equivalence.", "",
             "| Split | Defense | Forward % | SIPIT % | Forward − none pp [CI] | SIPIT − none pp [CI] | Attack s | Victim + evaluation s |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in summary:
        def fmt(ci_key):
            lo, hi = r[f"{ci_key}_ci_low"], r[f"{ci_key}_ci_high"]
            if lo is None:
                return f"{100 * r[f'{ci_key}_mean']:.2f} [n/a: {r[f'{ci_key}_status']}]" \
                    if r[f"{ci_key}_mean"] is not None else "n/a"
            return (f"{100 * r[f'{ci_key}_mean']:.2f} "
                    f"[{100 * lo:.2f}, {100 * hi:.2f}]")
        lines.append(
            f"| {r['split']} | {r['defense']} | "
            f"{100 * r['forward_accuracy_mean']:.2f} | "
            f"{100 * r['sipit_accuracy_mean']:.2f} | "
            f"{fmt('forward_minus_none')} | {fmt('sipit_minus_none')} | "
            f"{r['attack_seconds_mean']:.2f} | {r['wall_seconds_mean']:.2f} |")
    lines += ["", "## Matched held-out utility", "",
              "Perplexity deltas are defended-minus-undefended, paired over "
              "seeds (positive = the defense cost utility).", "",
              "| Split | Defense | PPL before → after | ΔPPL vs none [CI] |",
              "|---|---|---:|---:|"]
    for r in summary:
        util = [u for u in utility_rows
                if u["split"] == r["split"] and u["defense"] == r["defense"]]
        before = statistics.mean(u["ppl_before"] for u in util)
        after = statistics.mean(u["ppl_after"] for u in util)
        lo, hi = r["ppl_delta_vs_none_ci_low"], r["ppl_delta_vs_none_ci_high"]
        delta = (f"{r['ppl_delta_vs_none_mean']:.3f} [{lo:.3f}, {hi:.3f}]"
                 if lo is not None else
                 (f"{r['ppl_delta_vs_none_mean']:.3f} [n/a]"
                  if r["ppl_delta_vs_none_mean"] is not None else "n/a"))
        lines.append(f"| {r['split']} | {r['defense']} | "
                     f"{before:.2f} → {after:.2f} | {delta} |")
    consumers = [c for c in cells if c["defense"] in
                 ("trained-bottleneck", "secret-mixup", "replica-median")]
    lines += ["", "## Defense-specific consumers", "",
              "representation-privacy (bottleneck arm; no separate Gram-flattening defense claim) on "
              "public auxiliary/fresh scoring bottleneck releases; view-privacy de-mixing on "
              "converted mixup releases; replica-integrity clean / "
              "one-malicious / two-colluding arms with perturbation "
              "injected at the aggregation point.", ""]
    if consumers:
        lines += ["| Split | Defense | Seed | Consumer summary |",
                  "|---|---|---:|---|"]
        for c in consumers:
            attacks = c["attacks"]
            if c["defense"] == "trained-bottleneck":
                detail = "; ".join(
                    f"{arm}/{probe['family']}/{probe['control']}: accuracy={probe['accuracy']:.4f}"
                    for arm, v in attacks["representation-privacy"]["arms"].items()
                    for probe in v['probes'])
            elif c["defense"] == "secret-mixup":
                rows = attacks["view-privacy"]["rows"]
                detail = "; ".join(f"{r['arm']}: mse={r['mse']:.4g}"
                                   for r in rows[:4] if r.get("mse") is not None) or "n/a"
            else:
                detail = "; ".join(
                    f"{s['arm']}: accepted_error_rate={s.get('accepted_error_rate')}"
                    for s in attacks["replica-integrity"]["arms"])
            lines.append(f"| {c['split']} | {c['defense']} | {c['seed']} | {detail} |")
    else:
        lines.append("No trained-defender cells completed.")
    lines += ["", "## Failures and nulls", ""]
    if failures:
        lines.append("Every planned cell below is counted FAILED, not "
                     "dropped; the analysis exits nonzero.")
        lines += ["", "| Cell | Reason |", "|---|---|"]
        for f in failures:
            lines.append(f"| {f['cell']} | {f['reason']} |")
    else:
        lines.append(f"All {plan['expected_cells']} planned cells "
                     "completed with provenance, utility, matched access "
                     "spec and both common baselines. None dropped.")
    lines += ["", "## Scope and limitations", ""]
    lines += ["- " + x for x in plan.get("limits", [])]
    lines += ["", "Regenerate: `python experiments/analyze_defended_pilot.py "
              "--root results/pilot-defended-v1`", ""]
    (out / "report.md").write_text("\n".join(lines))
    result = {"schema": SCHEMA,
              "status": "COMPLETE" if not failures else "FAILED_CELLS",
              "cells_ok": len(cells), "cells_failed": len(failures),
              "expected_cells": plan["expected_cells"], "tier": tier,
              "disjointness": disjointness,
              "bootstrap": {"replicates": replicates, "seed": seed,
                            "method": "percentile, paired over seeds"},
              "summary": summary, "failures": failures,
              "analysis_script_sha256": file_sha256(__file__),
              "files": {p.name: file_sha256(p) for p in out.iterdir()
                        if p.suffix in (".csv", ".md")}}
    (out / "analysis.json").write_text(json.dumps(result, indent=2,
                                                  allow_nan=False))
    print(json.dumps({"status": result["status"], "cells_ok": len(cells),
                      "cells_failed": len(failures),
                      "report": str(out / "report.md")}, indent=2))
    if failures:
        raise ValueError(f"{len(failures)} planned cells FAILED; see "
                         f"{out / 'report.md'} failures-and-nulls section")
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--manifest", type=Path,
                    default=Path("experiments/validation_corpus_manifest.json"))
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args(argv)
    analyze(args.root, args.manifest, args.bootstrap, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
