#!/usr/bin/env python3
"""Cluster-aware paired re-analysis of the frozen gate (reviewer response).

The frozen gate (attacker.gate, frozen-gate.v2) scores a cell as the
Bonferroni-Wilson upper-95 of token-pooled recovery minus the observed
label-free majority control. A reviewer objected that (a) tokens within
a document are not independent binomial trials and (b) the majority
control is itself an estimate, so "Wilson upper bound minus observed
control" is not a valid population bound on the excess.

This analyzer answers with data — no new attack runs. It re-reads the
sealed per-sequence attack records of the two newest result tables
(results/domain-shift-v1 and results/long-horizon-v1/matrix), forms
document-level clusters (one cluster = one evaluation document; tokens
within a document are dependent), and recomputes the excess as a PAIRED
per-document difference (attack correct - control correct on the same
tokens). Per-document control correctness is recorded by the attack
itself (per_sequence[].control_correct: the calibration-corpus mode
token decoded at every position, attacker/attacks/seam_inversion.py),
so no control reconstruction is needed; the analyzer asserts the pooled
per-sequence counts reproduce every artifact's published recovery and
control exactly.

Inference: cluster bootstrap. Documents are resampled with replacement
WITHIN each seed (pinned bootstrap seed, default 93281); the statistic
is the token-pooled paired excess over the pooled resample (equal
per-seed token counts make this identical to the mean of the per-seed
excesses). A two-sided 95% percentile interval (type-7 quantile) per
cell is compared against the published Wilson excess and the +1.0 pp
gate threshold; per-seed excess breakdowns are reported across the 3
seeds of every cell, and every cell is flagged if its interval contains
+1.0 pp or if the clustered verdict (PASS iff interval upper <= +1.0 pp)
disagrees with the Wilson verdict.

Writes results/clustered-gate-v1/clustered-gate-v1-summary.{json,md}.
Pure stdlib + attacker.gate (torch-free), mirroring
experiments/analyze_domain_shift.py conventions.
"""
import argparse
import glob
import json
import math
import os
import platform
import random
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from attacker.gate import DEFAULT_GATE, bonferroni_z, verdict  # noqa: E402

SCHEMA = "llm_attacker.clustered_gate_summary.v1"
DS_ROOT = Path("results/domain-shift-v1")
LH_MATRIX = Path("results/long-horizon-v1/matrix")
LH_REPORT = Path("results/long-horizon-v1/report.json")
DS_CORPORA = ("code", "clinical", "wikitext-reference")
DS_MODES = ("training", "inference")
LH_ARMS = ("none", "gaussian-s15", "sparsify-10", "mid-lite-k64",
           "santext-30", "at-lite-e15")
LH_STEPS = (20, 100)
BOOTSTRAP_SEED = 93281
RECONCILE_TOL_PP = 0.051  # long-horizon report.json rounds to 2 decimals


def load_json(p):
    return json.loads(Path(p).read_text())


def docs_from_sequences(per_sequence, fallback_prefix):
    """One cluster per evaluation document. Rows carry the attack's own
    per-document control_correct (mode-token decode at every position).
    Duplicate document_ids within a result are merged into one cluster
    (none occur in either campaign; each row is its own document)."""
    by_doc = {}
    for i, q in enumerate(per_sequence):
        doc = q.get("document_id") or f"{fallback_prefix}-{q.get('index', i)}"
        d = by_doc.setdefault(doc, {"doc_id": doc, "n_tokens": 0,
                                    "correct": 0, "control_correct": 0})
        d["n_tokens"] += q["n_tokens"]
        d["correct"] += q["correct"]
        d["control_correct"] += q["control_correct"]
    return list(by_doc.values())


def pooled(docs):
    nc = sum(d["correct"] for d in docs)
    cc = sum(d["control_correct"] for d in docs)
    nt = sum(d["n_tokens"] for d in docs)
    return nc, cc, nt


def excess_pp(nc, cc, nt):
    return 100.0 * (nc - cc) / nt


def cluster_bootstrap(seeds_docs, replicates, rng):
    """Resample documents with replacement within each seed; statistic =
    token-pooled paired excess over the pooled resample. Returns sorted
    bootstrap replicates (pp)."""
    per_seed = [([d["correct"] - d["control_correct"] for d in docs],
                 [d["n_tokens"] for d in docs]) for docs in seeds_docs]
    ntok_values = {n for _, nts in per_seed for n in nts}
    choices = rng.choices
    stats = []
    if len(ntok_values) == 1:
        # both campaigns: every document carries the same token count,
        # so the pooled excess is a plain sum over resampled differences
        ntok = next(iter(ntok_values))
        denom = ntok * sum(len(d) for d, _ in per_seed)
        for _ in range(replicates):
            td = 0
            for diffs, _ in per_seed:
                td += sum(choices(diffs, k=len(diffs)))
            stats.append(100.0 * td / denom)
    else:  # general token-weighted path (not exercised by either table)
        for _ in range(replicates):
            td = tn = 0
            for diffs, nts in per_seed:
                n = len(diffs)
                for i in choices(range(n), k=n):
                    td += diffs[i]
                    tn += nts[i]
            stats.append(100.0 * td / tn)
    stats.sort()
    return stats


def quantile(sorted_vals, q):
    """Type-7 (linear interpolation) quantile of sorted values."""
    if not sorted_vals:
        return float("nan")
    pos = (len(sorted_vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] * (hi - pos) + sorted_vals[hi] * (pos - lo)


def finalize_cell(cell, replicates, bootstrap_seed, gate=DEFAULT_GATE):
    seeds_docs = cell["seeds_docs"]
    seeds = sorted(seeds_docs)
    all_docs = [d for s in seeds for d in seeds_docs[s]]
    nc, cc, nt = pooled(all_docs)
    per_seed = {}
    for s in seeds:
        snc, scc, snt = pooled(seeds_docs[s])
        g = verdict(snc / snt, scc / snt, snt, config=gate)
        per_seed[s] = {
            "n_docs": len(seeds_docs[s]), "n_tokens": snt,
            "recovery": round(snc / snt, 6),
            "control": round(scc / snt, 6),
            "excess_pp": round(excess_pp(snc, scc, snt), 4),
            "wilson_excess_pp": round(g["excess_pp"], 4)}
    rng = random.Random(f"{bootstrap_seed}:{cell['cell']}")
    stats = cluster_bootstrap([seeds_docs[s] for s in seeds],
                              replicates, rng)
    lo, hi = quantile(stats, 0.025), quantile(stats, 0.975)
    wilson_excess = cell["wilson_excess_pp"]
    wilson_verdict = "PASS" if wilson_excess <= gate.gate_pp else "FAIL"
    clustered_verdict = ("PASS" if hi <= gate.gate_pp + 1e-9 else "FAIL")
    contains = lo <= gate.gate_pp <= hi
    reasons = []
    if contains:
        reasons.append("clustered 95% interval contains the +1.0 pp "
                       "threshold")
    if wilson_verdict != clustered_verdict:
        reasons.append(f"clustered verdict {clustered_verdict} != Wilson "
                       f"verdict {wilson_verdict}")
    out = dict(cell)
    out.update({
        "n_seeds": len(seeds), "seeds": seeds,
        "n_docs_total": len(all_docs), "n_tokens_total": nt,
        "tokens_per_doc_constant": len({d["n_tokens"]
                                        for d in all_docs}) == 1,
        "recovery_pooled": round(nc / nt, 6),
        "control_pooled": round(cc / nt, 6),
        "paired_excess_pp": round(excess_pp(nc, cc, nt), 4),
        "per_seed": per_seed,
        "per_seed_excess_range_pp": [
            round(min(v["excess_pp"] for v in per_seed.values()), 4),
            round(max(v["excess_pp"] for v in per_seed.values()), 4)],
        "clustered_ci95_pp": [round(lo, 4), round(hi, 4)],
        "bootstrap_replicates": replicates,
        "wilson_verdict": wilson_verdict,
        "clustered_verdict": clustered_verdict,
        "interval_contains_threshold": contains,
        "verdict_sensitive": bool(reasons),
        "sensitivity_reason": "; ".join(reasons)})
    return out


def ds_cells(gate=DEFAULT_GATE):
    """Domain-shift cells: corpus x mode x defense, 3 seeds each.
    none from the base bundles; gaussian from the *-defended bundles."""
    pub_base = load_json(DS_ROOT / "domain-shift-v1-summary.json")
    pub_def = load_json(DS_ROOT / "domain-shift-v1-defended-summary.json")
    pub_base["_source"] = str(DS_ROOT / "domain-shift-v1-summary.json")
    pub_def["_source"] = str(DS_ROOT
                             / "domain-shift-v1-defended-summary.json")
    cells = []
    for corpus in DS_CORPORA:
        for defense, bundle, pub in (("none", corpus, pub_base),
                                     ("gaussian", f"{corpus}-defended",
                                      pub_def)):
            report = load_json(DS_ROOT / bundle / "experiment.json")
            if report.get("status") != "COMPLETE":
                raise ValueError(f"{bundle}: status {report.get('status')}")
            for mode in DS_MODES:
                recs = [r for r in report["results"]
                        if r["arm"] == "vmi" and r["mode"] == mode
                        and r["defense"] == defense]
                seeds_docs = {}
                for r in recs:
                    docs = docs_from_sequences(
                        r["per_sequence"], f"{bundle}-{mode}-s{r['seed']}")
                    snc, scc, snt = pooled(docs)
                    assert abs(snc / snt - r["accuracy"]) < 1e-6, (
                        bundle, mode, r["seed"], "recovery mismatch")
                    assert abs(scc / snt - r["control_accuracy"]) < 1e-9, (
                        bundle, mode, r["seed"], "control mismatch")
                    seeds_docs[str(r["seed"])] = docs
                all_docs = [d for docs in seeds_docs.values() for d in docs]
                nc, cc, nt = pooled(all_docs)
                g = verdict(nc / nt, cc / nt, nt, config=gate,
                            n_seeds=len(seeds_docs))
                published = pub["corpora"][corpus]["cells"][
                    f"{mode}/{defense}"]
                assert abs(nc / nt - published["recovery_pooled"]) < 1e-6
                assert abs(cc / nt - published["control_pooled"]) < 1e-6
                cells.append({
                    "campaign": "domain-shift-v1",
                    "cell": f"domain-shift/{corpus}/{mode}/{defense}",
                    "corpus": corpus, "mode": mode, "defense": defense,
                    "source": str(DS_ROOT / bundle / "experiment.json"),
                    "seeds_docs": seeds_docs,
                    "wilson_excess_pp": round(g["excess_pp"], 4),
                    "wilson_convention": "pooled across seeds "
                                         "(attacker.gate.verdict)",
                    "published_excess_pp": published["excess_pp"],
                    "published_source": str(pub["_source"])})
    return cells


def lh_cells(gate=DEFAULT_GATE):
    """Long-horizon cells: arm x steps, 3 seeds each. Each step result
    attacks exactly one sequence with a distinct truth, so one cluster
    per step-document."""
    pub = load_json(LH_REPORT)["paired"]
    grouped = {}
    pat = str(LH_MATRIX / "artifacts" / "seam-inversion_vmi_*.json")
    for p in sorted(glob.glob(pat)):
        name = os.path.basename(p)
        if ".evidence." in name or not name.endswith(".json"):
            continue
        tag = name[len("seam-inversion_vmi_"):-len(".json")]
        arm, rest = tag.rsplit("_s", 1)
        seed, horizon = rest.split("_h")
        if arm not in LH_ARMS or not horizon.isdigit():
            continue
        steps = int(horizon)
        if steps not in LH_STEPS:
            continue
        art = load_json(p)
        if not art.get("summary"):
            raise ValueError(f"{p}: missing summary")
        docs = []
        for r in art["results"]:
            if len(r["per_sequence"]) != 1:
                raise ValueError(f"{p} step {r.get('step')}: expected one "
                                 "sequence per step cell")
            docs += docs_from_sequences(r["per_sequence"],
                                        f"step-{r['step']}")
        nc, cc, nt = pooled(docs)
        s0 = art["summary"][0]
        assert abs(nc / nt - s0["seq_accuracy"]) < 1e-6, (p, "recovery")
        assert abs(cc / nt - s0["control_accuracy"]) < 1e-9, (p, "control")
        grouped.setdefault((arm, steps), {})[seed] = docs
    cells = []
    for arm in LH_ARMS:
        for steps in LH_STEPS:
            seeds_docs = grouped.get((arm, steps))
            if not seeds_docs:
                raise ValueError(f"long-horizon: no artifacts for "
                                 f"{arm} h{steps}")
            per_seed_wilson = []
            for docs in seeds_docs.values():
                nc, cc, nt = pooled(docs)
                per_seed_wilson.append(
                    verdict(nc / nt, cc / nt, nt, config=gate)["excess_pp"])
            wilson = sum(per_seed_wilson) / len(per_seed_wilson)
            cells.append({
                "campaign": "long-horizon-v1",
                "cell": f"long-horizon/{arm}/h{steps}",
                "arm": arm, "steps": steps,
                "source": str(LH_MATRIX / "artifacts"
                              f"/seam-inversion_vmi_{arm}_s*_h{steps}.json"),
                "seeds_docs": seeds_docs,
                "wilson_excess_pp": round(wilson, 4),
                "wilson_convention": "mean over seeds of per-seed "
                                     "attacker.gate.verdict excess",
                "published_excess_pp": pub[arm][str(steps)][
                    "excess_v2_pp_mean"],
                "published_source": str(LH_REPORT)})
    return cells


def provenance(argv):
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], timeout=10,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        head = None
    return {"argv": argv, "cwd": os.getcwd(),
            "generated_utc": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "git_head": head or None,
            "packages": "stdlib-only (attacker.gate is pure python)"}


def analyze(replicates, bootstrap_seed, argv, gate=DEFAULT_GATE):
    cells = ds_cells(gate) + lh_cells(gate)
    results = [finalize_cell(c, replicates, bootstrap_seed, gate)
               for c in cells]
    recon = [abs(c["wilson_excess_pp"] - c["published_excess_pp"])
             for c in results]
    for c in results:
        c.pop("seeds_docs")
    return {
        "schema": SCHEMA,
        "provenance": provenance(argv),
        "gate": {"version": gate.version, "z": bonferroni_z(gate),
                 "gate_pp": gate.gate_pp,
                 "n_seeds_required": gate.n_seeds_required,
                 "control": gate.control,
                 "control_source": "attack-recorded per_sequence."
                 "control_correct (calibration-corpus mode token decoded "
                 "at every position, attacker/attacks/seam_inversion.py); "
                 "pooled counts asserted to reproduce every artifact's "
                 "published recovery/control exactly",
                 "confidence_note": "the frozen gate's z=2.5392 is a "
                 "one-sided 99.44% bound (Bonferroni over 9 trials); the "
                 "clustered interval is two-sided 95%. The clustered "
                 "check drops the binomial-independence assumption and "
                 "the fixed-control assumption; it is not "
                 "multiplicity-adjusted."},
        "bootstrap": {"seed": bootstrap_seed, "replicates": replicates,
                      "resampling": "documents with replacement within "
                      "each seed; seeds never resampled (3 fixed "
                      "replicates)",
                      "statistic": "token-pooled paired excess pp = "
                      "100*(sum correct - sum control_correct)/sum "
                      "n_tokens over the pooled resample; equal per-seed "
                      "token counts make this the mean of the per-seed "
                      "excesses",
                      "interval": "two-sided 95% percentile (type-7)",
                      "rng": "random.Random(f'{seed}:{cell_id}')"},
        "clustering": {"unit": "evaluation document",
                       "domain-shift": "per_sequence.document_id "
                       "(unique per row in every result)",
                       "long-horizon": "one attacked sequence per "
                       "training-step cell; all truths distinct per run, "
                       "so one cluster per step-document"},
        "reconciliation": {
            "n_cells": len(results),
            "max_abs_diff_pp": round(max(recon), 6),
            "tolerance_pp": RECONCILE_TOL_PP,
            "all_within_tolerance": max(recon) <= RECONCILE_TOL_PP},
        "cells": results}


def render_md(summary):
    gate = summary["gate"]
    b = summary["bootstrap"]
    lines = ["# clustered-gate-v1: cluster-aware paired re-analysis of "
             "the frozen gate",
             "",
             "Reviewer points answered with data (no new attack runs): "
             "(a) tokens within a document are not independent binomial "
             "trials -> one cluster per evaluation document, documents "
             "resampled with replacement within each seed; (b) the "
             "majority control is itself an estimate -> the excess is "
             "recomputed as a PAIRED per-document difference, so the "
             "control's own sampling error is resampled jointly with the "
             "attack's. Per-document control correctness is the attack's "
             "own recorded `control_correct` (no reconstruction "
             "assumption; pooled counts reproduce every artifact's "
             "published recovery/control exactly, asserted at load).",
             "",
             f"Gate: {gate['version']} (z={gate['z']:.4f}, threshold +"
             f"{gate['gate_pp']} pp; control = {gate['control']}). "
             f"Cluster bootstrap: {b['replicates']} replicates, pinned "
             f"seed {b['seed']}, two-sided 95% percentile interval. "
             "Clustered verdict analog: PASS iff interval upper <= "
             f"+{gate['gate_pp']} pp.",
             ""]
    ds = [c for c in summary["cells"] if c["campaign"] == "domain-shift-v1"]
    lh = [c for c in summary["cells"] if c["campaign"] == "long-horizon-v1"]
    hdr = ("| cell | docs (seed x seeds) | Wilson excess pp (published) "
           "| Wilson verdict | paired excess pp | clustered 95% CI pp | "
           "per-seed excess range pp | contains +1.0? | verdict flips? |")
    sep = "|---|---|---|---|---|---|---|---|---|"
    lines += ["## domain-shift-v1 (GPT-2 @6/12, vmi arm)", "", hdr, sep]
    for c in ds:
        lines.append(_row(c, f"{c['corpus']} {c['mode']}/{c['defense']}"))
    lines += ["", "## long-horizon-v1 (defense matrix, vmi arm)", "",
              hdr, sep]
    for c in lh:
        lines.append(_row(c, f"{c['arm']} h{c['steps']}"))
    n_sens = sum(c["verdict_sensitive"] for c in summary["cells"])
    lines += ["",
              "## Verdict sensitivity",
              "",
              f"Cells whose clustered 95% interval contains the "
              f"+{gate['gate_pp']} pp threshold or whose verdict moves: "
              f"{n_sens} of {len(summary['cells'])}."]
    for c in summary["cells"]:
        if c["verdict_sensitive"]:
            lines.append(f"- **{c['cell']}**: {c['sensitivity_reason']} "
                         f"(Wilson {c['published_excess_pp']:+.2f} pp "
                         f"{c['wilson_verdict']}; clustered CI "
                         f"{c['clustered_ci95_pp']} pp).")
    deg = [c for c in summary["cells"]
           if c["clustered_ci95_pp"][0] == c["clustered_ci95_pp"][1]]
    if deg:
        lines += ["",
                  "## Degenerate intervals (no cross-document variance)",
                  ""]
        for c in deg:
            if c["paired_excess_pp"] == 0.0:
                lines.append(
                    f"- {c['cell']}: zero correct and zero control "
                    "tokens in every document -> degenerate interval "
                    f"{c['clustered_ci95_pp']} pp. A percentile "
                    "bootstrap cannot manufacture events that never "
                    "occurred; the frozen gate's PASS here is its "
                    "statistical floor (Wilson upper at 0 events), and "
                    "clustering does not move it.")
            else:
                lines.append(
                    f"- {c['cell']}: every document shows the identical "
                    f"paired outcome ({c['paired_excess_pp']:+.2f} pp) "
                    f"-> degenerate interval {c['clustered_ci95_pp']} "
                    "pp; there is no cross-document variance to "
                    "resample, and the verdict is unaffected.")
    r = summary["reconciliation"]
    lines += ["",
              f"Reconciliation: recomputed Wilson excesses match the "
              f"published tables within {r['max_abs_diff_pp']} pp on all "
              f"{r['n_cells']} cells (tolerance {r['tolerance_pp']} pp).",
              "",
              f"Provenance: `{summary['provenance']['argv']}` @ "
              f"{summary['provenance']['generated_utc']}, "
              f"git {summary['provenance']['git_head']}."]
    return "\n".join(lines) + "\n"


def _row(c, label):
    docs = f"{c['n_docs_total'] // c['n_seeds']}x{c['n_seeds']}"
    return (f"| {label} | {docs} | {c['published_excess_pp']:+.2f} "
            f"| {c['wilson_verdict']} | {c['paired_excess_pp']:+.2f} "
            f"| [{c['clustered_ci95_pp'][0]:+.2f}, "
            f"{c['clustered_ci95_pp'][1]:+.2f}] "
            f"| [{c['per_seed_excess_range_pp'][0]:+.2f}, "
            f"{c['per_seed_excess_range_pp'][1]:+.2f}] "
            f"| {'yes' if c['interval_contains_threshold'] else 'no'} "
            f"| {'SENSITIVE' if c['verdict_sensitive'] else 'no'} |")


def self_test(replicates=2000):
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    print("cluster bootstrap self-test:")

    def mk(diff, ctl=0):
        return {"doc_id": "d", "n_tokens": 32, "correct": diff + ctl,
                "control_correct": ctl}

    const = [[mk(1) for _ in range(10)] for _ in range(3)]
    stats = cluster_bootstrap(const, replicates, random.Random(93281))
    check("constant per-document excess -> degenerate interval",
          min(stats) == max(stats) == 100.0 / 32)
    zero = [[mk(0) for _ in range(10)] for _ in range(3)]
    stats0 = cluster_bootstrap(zero, replicates, random.Random(93281))
    check("zero outcomes -> [0, 0]",
          min(stats0) == max(stats0) == 0.0)
    mixed = [[{"doc_id": "d", "n_tokens": 32, "correct": c,
               "control_correct": 1} for c in (3, 7, 9, 2, 8)]
             for _ in range(3)]
    a = cluster_bootstrap(mixed, replicates, random.Random("s"))
    b = cluster_bootstrap(mixed, replicates, random.Random("s"))
    check("same bootstrap seed reproduces the interval exactly", a == b)
    nc, cc, nt = pooled([d for docs in mixed for d in docs])
    point = excess_pp(nc, cc, nt)
    lo, hi = quantile(a, 0.025), quantile(a, 0.975)
    check("95% interval brackets the point excess with nonzero width",
          lo < point < hi)
    wide = [[{"doc_id": "d", "n_tokens": 32, "correct": c,
              "control_correct": 0} for c in (0, 32)]]
    statsw = cluster_bootstrap(wide, replicates, random.Random(93281))
    check("two-outcome population spans its support",
          min(statsw) == 0.0 and max(statsw) == 100.0)
    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replicates", type=int, default=100_000,
                    help="cluster bootstrap replicates per cell")
    ap.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    ap.add_argument("--output", default="results/clustered-gate-v1/"
                                        "clustered-gate-v1-summary.json")
    ap.add_argument("--self-test", action="store_true",
                    help="synthetic bootstrap checks; no campaign I/O")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    summary = analyze(args.replicates, args.bootstrap_seed,
                      sys.argv if argv is None else argv)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1) + "\n")
    out.with_suffix(".md").write_text(render_md(summary))
    print(render_md(summary))
    print(f"[clustered-gate] wrote {out} and {out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
