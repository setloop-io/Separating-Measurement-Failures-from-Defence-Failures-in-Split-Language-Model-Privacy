"""P6 Mode-2 crossed-span grid driver (protocol p6-crossed-span-1.0, frozen
2026-09-16; experiments/crossed_span_protocol.p6.json).

One span per invocation (the runbook section C fleet split: soula=s2,
voula=s4, sasa=s6). Grid per span: starts {0,1,2,3} x defenses {D0,D1} x
victim seeds {11..88} = 64 cells. Each cell is a Mode-2 capture
(attacker.splitbench finetune of backbone blocks [start, start+span) in
place, everything else frozen) followed by the seam-inversion VMI attack
on the cell's capture dir, mirroring experiments/run_long_horizon.py's
proven structure (SUBENV thread caps, timeouts, run-index conventions).

Per the freeze record's start=0 mapping decision: splitbench requires
--split >= 1, so start=0 cells execute as --split 1 --trainable-range
0,SPAN (seam after block 0, trainable from block 0); start>=1 cells use
--split START --trainable-range START,START+SPAN. The attack's --split
always matches the capture's.

Execution order is a deterministic interleave (no RNG): 8 seed rounds;
within round si the starts are swept in a Latin rotation (si mod 4) and
the defense sweep alternates direction, so no (start, defense) pair is
time-clustered. --cells k truncates to the first k cells of that order.

Resume/discipline (runbook ground rules): fresh output dir required
unless --resume; --resume skips cells already journaled complete and any
phase whose artifact already exists and parses; a failed cell is
journaled and the driver CONTINUES (no in-run retries -- the protocol's
missing-data rule is handled at analysis). Per-cell wall-clock + rc are
appended crash-safe to <workdir>/run-index.jsonl (one record per phase
plus one "cell" summary record); <workdir>/run-index.json is rewritten
from the journal at exit.

Per-cell summary record fields: span/start/defense/seed (+ split,
trainable_range, node), capture_wall_s/attack_wall_s and per-phase rc,
final_loss/steps_run from the capture artifact, vmi_recovery_pct /
vmi_control_pct / vmi_excess_pp (+ verdict, n_cells) from the attack
artifact's frozen-gate.v2 summary, and the analytic delegation share
span/12 (GPT-2 n_layer; start-independent by the Mode-2 convention).

Torch-free stdlib driver: it only shells out. Run with the node venv:
  .venv/bin/python experiments/run_p6_grid.py --span 2 --node soula \
      --workdir results/p6-grid-s2-v1
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

PROTOCOL_ID = "p6-crossed-span-1.0"
MODEL = "gpt2"
N_LAYER = 12                      # GPT-2 124M block count (B_mode2)
STARTS = (0, 1, 2, 3)
DEFENSES = ("D0", "D1")
SEEDS = (11, 22, 33, 44, 55, 66, 77, 88)
STEPS = 100                       # per-cell finetune budget (canary-anchored)
SEQ_LEN = 32
BATCH_SIZE = 1
LR = "5e-5"
CORPUS = "results/long-horizon-v1/inputs/pilot-corpus.txt"

#: defenses.mode_2_executable of the frozen protocol
DEFENSE_ARGV = {
    "D0": ("--defense", "quant", "--quant-bits", "8", "--quant-grad"),
    "D1": ("--defense", "quant", "--quant-bits", "4", "--quant-grad",
           "--grad-clip", "1.0", "--grad-noise", "0.1"),
}

TIMEOUTS = {"capture": 1800, "attack": 900}

SUBENV = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
              HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false")


def utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def interleaved_order():
    """Deterministic interleave of starts x defenses x seeds (see docstring):
    one seed per round, starts Latin-rotated by round, defense sweep
    alternating. 64 cells."""
    cells = []
    for si, seed in enumerate(SEEDS):
        starts = STARTS[si % len(STARTS):] + STARTS[:si % len(STARTS)]
        defenses = DEFENSES if si % 2 == 0 else DEFENSES[::-1]
        for start in starts:
            for defense in defenses:
                cells.append((start, defense, seed))
    return cells


def start_mapping(start, span):
    """Freeze-record Mode-2 mapping -> (split, trainable_range) strings."""
    if start >= 1:
        return str(start), f"{start},{start + span}"
    return "1", f"0,{span}"


def cell_tag(span, start, defense, seed):
    return f"span{span}_b{start}_{defense}_s{seed}"


def journal(workdir, rec):
    with open(os.path.join(workdir, "run-index.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")


def load_journal(workdir):
    path = os.path.join(workdir, "run-index.jsonl")
    recs = []
    if os.path.isfile(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        recs.append(json.loads(line))
                    except ValueError:
                        pass
    return recs


def artifact_ok(path):
    try:
        with open(path) as f:
            art = json.load(f)
        return bool(art.get("results") or art.get("summary"))
    except (OSError, ValueError):
        return False


def drive(workdir, cell, cmd, timeout, artifact=None):
    """Run one cell subprocess; never raises; journals the outcome."""
    if artifact and artifact_ok(artifact):
        rec = {**cell, "skipped": "artifact exists", "rc": 0, "wall_s": 0.0}
        journal(workdir, rec)
        print(f"[skip] {cell['cell_id']}")
        return rec
    t0 = time.monotonic()
    try:
        proc = subprocess.run(cmd, cwd=REPO, env=SUBENV, capture_output=True,
                              text=True, timeout=timeout)
        rc, tail = proc.returncode, (proc.stdout or proc.stderr or "")[-2000:]
    except subprocess.TimeoutExpired:
        rc, tail = 124, f"error: cell timed out after {timeout}s"
    wall = time.monotonic() - t0
    rec = {**cell, "rc": rc, "wall_s": round(wall, 2),
           "argv": [os.path.basename(str(x)) for x in cmd]}
    if rc != 0:
        rec["error"] = (tail.strip().splitlines()[-1] if tail.strip()
                        else f"rc={rc}")
    journal(workdir, rec)
    print(f"[cell] {cell['cell_id']}: rc={rc} {wall:.1f}s", flush=True)
    return rec


def capture_cmd(span, start, defense, seed, capdir, out):
    split, tr = start_mapping(start, span)
    return [sys.executable, "-m", "attacker.splitbench",
            "--model", MODEL, "--split", split,
            "--trainable-range", tr, "--mode", "finetune",
            "--steps", str(STEPS), "--seq-len", str(SEQ_LEN),
            "--batch-size", str(BATCH_SIZE), "--lr", LR,
            "--device", "cpu", "--dtype", "fp32", "--seed", str(seed),
            *DEFENSE_ARGV[defense],
            "--corpus-file", CORPUS,
            "--capture-dir", capdir, "--save-canonical",
            "--output", out, "--evidence"]


def attack_cmd(span, start, capdir, out):
    split, _ = start_mapping(start, span)
    return [sys.executable, "-m", "attacker", "--mode", "training",
            "--attack", "seam-inversion", "--arm", "vmi",
            "--model", MODEL, "--split", split,
            "--capture-dir", capdir,
            "--canonical-pt", os.path.join(capdir, "canonical.pt"),
            "--samples", str(STEPS), "--device", "cpu",
            "--output", out, "--evidence"]


def capture_metrics(path):
    with open(path) as f:
        art = json.load(f)
    res = (art.get("results") or [{}])[0]
    return {"final_loss": res.get("final_loss"),
            "steps_run": res.get("steps_run")}


def attack_metrics(path):
    with open(path) as f:
        art = json.load(f)
    summ = (art.get("summary") or [{}])[0]
    gate = summ.get("gate") or {}
    return {"vmi_recovery_pct": gate.get("recovery_pct"),
            "vmi_control_pct": gate.get("control_pct"),
            "vmi_excess_pp": gate.get("excess_pp"),
            "vmi_verdict": gate.get("verdict"),
            "vmi_n_cells": summ.get("n_cells")}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--span", type=int, required=True, choices=(2, 4, 6))
    ap.add_argument("--node", required=True,
                    help="node name recorded for provenance (e.g. soula)")
    ap.add_argument("--workdir", required=True,
                    help="fresh output dir (relative to CWD); must not "
                         "already exist non-empty unless --resume")
    ap.add_argument("--cells", type=int, default=None,
                    help="run only the first k cells of the interleaved "
                         "order (default: all 64)")
    ap.add_argument("--resume", action="store_true",
                    help="skip cells already journaled complete / phases "
                         "whose artifacts exist and parse")
    args = ap.parse_args()

    workdir = os.path.abspath(args.workdir)
    if (os.path.isdir(workdir) and os.listdir(workdir)
            and not args.resume):
        print(f"error: workdir {workdir} exists and is not empty; use a "
              f"fresh directory or --resume", file=sys.stderr)
        return 2
    artdir = os.path.join(workdir, "artifacts")
    caproot = os.path.join(workdir, "captures")
    os.makedirs(artdir, exist_ok=True)
    os.makedirs(caproot, exist_ok=True)

    order = interleaved_order()
    if args.cells is not None:
        order = order[:max(0, args.cells)]

    done = set()
    if args.resume:
        done = {r["tag"] for r in load_journal(workdir)
                if r.get("kind") == "cell" and r.get("rc") == 0}

    manifest = {"driver": Path(__file__).name, "protocol": PROTOCOL_ID,
                "node": args.node, "span": args.span, "model": MODEL,
                "n_layer": N_LAYER, "starts": list(STARTS),
                "defenses": list(DEFENSES),
                "defense_argv": {k: list(v) for k, v in DEFENSE_ARGV.items()},
                "seeds": list(SEEDS), "steps": STEPS, "seq_len": SEQ_LEN,
                "batch_size": BATCH_SIZE, "lr": LR, "corpus": CORPUS,
                "attack": "seam-inversion --arm vmi --samples "
                          f"{STEPS} (full horizon)",
                "start0_mapping": "--split 1 --trainable-range 0,SPAN "
                                  "(freeze-record Mode-2 approximation)",
                "delegation_share": "span/12 (analytic, Mode-2 convention)",
                "timeouts": TIMEOUTS, "threads": 4, "resume": args.resume,
                "cell_limit": args.cells,
                "order": [cell_tag(args.span, s, d, sd)
                          for s, d, sd in order],
                "started_utc": utcnow(), "argv": sys.argv}
    with open(os.path.join(workdir, "run-manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)

    for start, defense, seed in order:
        tag = cell_tag(args.span, start, defense, seed)
        if tag in done:
            print(f"[skip] {tag} (journaled complete)")
            continue
        split, tr = start_mapping(start, args.span)
        base = {"tag": tag, "node": args.node, "span": args.span,
                "start": start, "defense": defense, "seed": seed,
                "split": split, "trainable_range": tr}
        capdir = os.path.join(caproot, tag)
        cap_out = os.path.join(artdir, f"splitbench_{tag}.json")
        atk_out = os.path.join(artdir, f"seam-inversion_vmi_{tag}.json")

        cap = drive(workdir,
                    {**base, "cell_id": f"capture:{tag}", "kind": "capture",
                     "artifact": cap_out},
                    capture_cmd(args.span, start, defense, seed, capdir,
                                cap_out),
                    TIMEOUTS["capture"], artifact=cap_out)

        if artifact_ok(cap_out):
            atk = drive(workdir,
                        {**base, "cell_id": f"attack:{tag}", "kind": "attack",
                         "artifact": atk_out},
                        attack_cmd(args.span, start, capdir, atk_out),
                        TIMEOUTS["attack"], artifact=atk_out)
        else:
            atk = {**base, "cell_id": f"attack:{tag}", "kind": "attack",
                   "rc": None, "wall_s": 0.0,
                   "error": "skipped: capture failed or missing"}
            journal(workdir, atk)
            print(f"[cell] attack:{tag}: skipped (capture failed)")

        rec = {**base, "cell_id": f"cell:{tag}", "kind": "cell",
               "capture_rc": cap.get("rc"), "attack_rc": atk.get("rc"),
               "capture_wall_s": cap.get("wall_s"),
               "attack_wall_s": atk.get("wall_s"),
               "delegation_share": args.span / N_LAYER}
        cap_ok = cap.get("rc") == 0 and artifact_ok(cap_out)
        atk_ok = atk.get("rc") == 0 and artifact_ok(atk_out)
        if cap_ok:
            rec.update(capture_metrics(cap_out))
        if atk_ok:
            rec.update(attack_metrics(atk_out))
        rec["rc"] = 0 if (cap_ok and atk_ok) else 1
        rec["status"] = "ok" if rec["rc"] == 0 else "failed"
        if rec["rc"] != 0:
            rec["error"] = (atk.get("error") or cap.get("error")
                            or "artifact invalid")
        journal(workdir, rec)
        print(f"[cell] {tag}: {rec['status']} "
              f"(capture {rec['capture_wall_s']}s, "
              f"attack {rec['attack_wall_s']}s)", flush=True)

    cells = load_journal(workdir)
    n_cell = sum(1 for c in cells if c.get("kind") == "cell")
    n_err = sum(1 for c in cells
                if c.get("kind") == "cell" and c.get("rc") != 0)
    with open(os.path.join(workdir, "run-index.json"), "w") as f:
        json.dump({"manifest": manifest, "cells": cells,
                   "n_cell_records": n_cell, "n_failed": n_err,
                   "finished_utc": utcnow()}, f, indent=1)
    print(f"[run_p6_grid] {n_cell} cell records, {n_err} failed "
          f"-> {workdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
