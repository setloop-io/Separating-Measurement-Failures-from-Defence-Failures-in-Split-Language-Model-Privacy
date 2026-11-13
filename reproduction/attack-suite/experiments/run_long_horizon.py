"""Long-horizon (100-step) defense-matrix driver for the GPT-2 @6/12 rebuttal.

Scripted splitbench loop in the spirit of attacker/benchmark.py's
_capture_argv (benchmark.v1), but pinned to the paper's reported defense
operating points (paper/paper_b/main.tex "Defence matrix"): gaussian
sigma=15, sparsify-10% (splitbench default), mid-lite k=64, santext-30%,
at-lite eps=15. The benchmark.v1 battery pins only gaussian; the other
battery arms run splitbench defaults (mid-lite k=8, santext p=0.1,
at-lite eps=0.1), which are NOT the paper's cells — hence this driver.

Cells: captures (attacker.splitbench --save-canonical --evidence) then
per-capture seam-inversion VMI and label-scoring attacks (both
--evidence), each journaled with wall-clock seconds into
<workdir>/run-index.jsonl (crash-safe) + run-index.json (final).

Phases:
  capture    arms x horizons x seeds capture cells
  attack     per-capture VMI (--samples = full horizon) + label-scoring
  epochwrap  4-epoch wrap-arm captures (800-token corpus prefix, 25
             windows/epoch) for the cross-epoch accumulation probes
  probe      canonical replay + attacks/accumulation on real captures
  synthetic  leak-accumulation / subspace-joint --toy at --epochs 50
             (clearly synthetic tier; these modules take no --evidence)

Resume-safe: a cell whose artifact already exists and parses is skipped
unless --force. Threads are capped at 4 via the subprocess env.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: (arm label, splitbench --defense, extra splitbench knobs)
ARMS = (
    ("none", "none", ()),
    ("gaussian-s15", "gaussian", ("--sigma", "15.0")),
    ("sparsify-10", "sparsify", ()),
    ("mid-lite-k64", "mid-lite", ("--bottleneck-dim", "64")),
    ("santext-30", "santext", ("--santext-p", "0.3")),
    ("at-lite-e15", "at-lite", ("--at-eps", "15.0")),
)
WRAP_ARMS = ("none", "mid-lite-k64")   # epoch-wrap probe defenses

TIMEOUTS = {"capture": 1800, "attack": 900, "synthetic": 900, "replay": 900,
            "probe": 900}

SUBENV = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
              HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false")


def journal(workdir, rec):
    with open(os.path.join(workdir, "run-index.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")


def artifact_ok(path):
    try:
        with open(path) as f:
            art = json.load(f)
        return bool(art.get("results") or art.get("summary"))
    except (OSError, ValueError):
        return False


def drive(workdir, cell, cmd, timeout, artifact=None, force=False):
    """Run one cell subprocess; never raises; journals the outcome."""
    if artifact and not force and artifact_ok(artifact):
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
    print(f"[cell] {cell['cell_id']}: rc={rc} {wall:.1f}s")
    return rec


def capture_cmd(cfg, defense, knobs, capdir, out, steps, seed):
    return [sys.executable, "-m", "attacker.splitbench",
            "--model", cfg.model, "--split", cfg.split, "--mode", "finetune",
            "--steps", str(steps), "--seq-len", str(cfg.seq_len),
            "--batch-size", str(cfg.batch_size), "--lr", str(cfg.lr),
            "--device", "cpu", "--dtype", "fp32", "--seed", str(seed),
            "--defense", defense, *map(str, knobs),
            "--corpus-file", cfg.corpus_file,
            "--capture-dir", capdir, "--save-canonical",
            "--output", out, "--evidence"]


def attack_cmds(cfg, capdir, steps):
    common = [sys.executable, "-m", "attacker", "--mode", "training"]
    vmi = common + ["--attack", "seam-inversion", "--arm", "vmi",
                    "--model", cfg.model, "--split", cfg.split,
                    "--capture-dir", capdir,
                    "--canonical-pt", os.path.join(capdir, "canonical.pt"),
                    "--samples", str(steps), "--device", "cpu"]
    lab = common + ["--attack", "label-scoring",
                    "--capture-dir", capdir,
                    "--canonical-pt", os.path.join(capdir, "canonical.pt"),
                    "--property", f"contains-token:{cfg.property_token}"]
    return vmi, lab


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", default="results/long-horizon-v1/matrix")
    ap.add_argument("--target-config", default="results/long-horizon-v1/target.json")
    ap.add_argument("--wrap-corpus",
                    default="results/long-horizon-v1/inputs/pilot-corpus-epochwrap.txt")
    ap.add_argument("--phases", nargs="+",
                    default=["capture", "attack"],
                    choices=["capture", "attack", "epochwrap", "probe",
                             "synthetic"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1234, 2026, 7])
    ap.add_argument("--horizons", type=int, nargs="+", default=[20, 100])
    ap.add_argument("--arms", nargs="+", default=[a[0] for a in ARMS])
    ap.add_argument("--synthetic-epochs", type=int, default=50)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, str(REPO))
    from attacker.benchmark import TargetConfig
    cfg = TargetConfig.from_json(args.target_config)
    workdir = os.path.abspath(args.workdir)
    artdir = os.path.join(workdir, "artifacts")
    os.makedirs(artdir, exist_ok=True)
    arms = [a for a in ARMS if a[0] in args.arms]
    manifest = {"driver": Path(__file__).name, "target": vars(cfg),
                "arms": {a[0]: {"defense": a[1], "knobs": list(a[2])}
                         for a in arms},
                "seeds": args.seeds, "horizons": args.horizons,
                "wrap_corpus": args.wrap_corpus,
                "wrap_arms": list(WRAP_ARMS),
                "timeouts": TIMEOUTS, "threads": 4,
                "note": "benchmark.v1-style cells pinned to the paper's "
                        "defense operating points (mid-lite k=64, santext "
                        "p=0.3, at-lite eps=15); not the battery defaults."}
    with open(os.path.join(workdir, "run-manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)

    def capdir_of(tag):
        return os.path.join(workdir, "captures", tag)

    if "capture" in args.phases:
        for arm, defense, knobs in arms:
            for steps in args.horizons:
                for seed in args.seeds:
                    tag = f"{arm}_s{seed}_h{steps}"
                    out = os.path.join(artdir, f"splitbench_{tag}.json")
                    drive(workdir, {"cell_id": f"capture:{tag}", "kind": "capture",
                                    "arm": arm, "steps": steps, "seed": seed,
                                    "artifact": out},
                          capture_cmd(cfg, defense, knobs, capdir_of(tag),
                                      out, steps, seed),
                          TIMEOUTS["capture"], artifact=out, force=args.force)

    if "attack" in args.phases:
        for arm, _, _ in arms:
            for steps in args.horizons:
                for seed in args.seeds:
                    tag = f"{arm}_s{seed}_h{steps}"
                    capdir = capdir_of(tag)
                    if not os.path.isfile(os.path.join(capdir, "canonical.pt")):
                        journal(workdir, {"cell_id": f"attack:{tag}",
                                          "kind": "attack",
                                          "error": "skipped: capture missing"})
                        continue
                    vmi, lab = attack_cmds(cfg, capdir, steps)
                    out_v = os.path.join(artdir, f"seam-inversion_vmi_{tag}.json")
                    out_l = os.path.join(artdir, f"label-scoring_norm_{tag}.json")
                    drive(workdir, {"cell_id": f"S1:{tag}:seam-inversion",
                                    "kind": "attack", "artifact": out_v},
                          vmi + ["--output", out_v, "--evidence"],
                          TIMEOUTS["attack"], artifact=out_v, force=args.force)
                    drive(workdir, {"cell_id": f"S9:{tag}:label-scoring",
                                    "kind": "attack", "artifact": out_l},
                          lab + ["--output", out_l, "--evidence"],
                          TIMEOUTS["attack"], artifact=out_l, force=args.force)

    if "epochwrap" in args.phases:
        wcfg = TargetConfig(**{**vars(cfg), "corpus_file":
                               os.path.abspath(args.wrap_corpus)})
        for arm, defense, knobs in arms:
            if arm not in WRAP_ARMS:
                continue
            for seed in args.seeds[:1]:
                tag = f"{arm}_s{seed}_h100_epochwrap"
                out = os.path.join(artdir, f"splitbench_{tag}.json")
                drive(workdir, {"cell_id": f"capture:{tag}", "kind": "capture",
                                "arm": arm, "steps": 100, "seed": seed,
                                "artifact": out},
                      capture_cmd(wcfg, defense, knobs, capdir_of(tag), out,
                                  100, seed),
                      TIMEOUTS["capture"], artifact=out, force=args.force)

    if "probe" in args.phases:
        replay = str(REPO / "experiments" / "canonical_replay.py")
        targets = [f"{a}_s{args.seeds[0]}_h100_epochwrap" for a in WRAP_ARMS]
        targets.append(f"none_s{args.seeds[0]}_h100")  # single-epoch control
        for tag in targets:
            capdir = capdir_of(tag)
            canon = os.path.join(capdir, "canonical.pt")
            if not os.path.isfile(canon):
                journal(workdir, {"cell_id": f"probe:{tag}", "kind": "probe",
                                  "error": "skipped: capture missing"})
                continue
            rep = os.path.join(capdir, "canonical-replay.pt")
            if args.force or not os.path.isfile(rep):
                drive(workdir, {"cell_id": f"replay:{tag}", "kind": "replay"},
                      [sys.executable, replay, "--model", cfg.model,
                       "--split", cfg.split, "--canonical-pt", canon,
                       "--out", rep, "--device", "cpu"],
                      TIMEOUTS["replay"])
            out = os.path.join(artdir, f"accumulation_fwd_{tag}.json")
            drive(workdir, {"cell_id": f"probe:{tag}:accumulation",
                            "kind": "probe", "artifact": out},
                  [sys.executable, "-m", "attacker", "--mode", "training",
                   "--attack", "accumulation", "--capture-dir", capdir,
                   "--canonical-pt", rep, "--phase", "fwd",
                   "--epoch-sizes", "64", "256", "768",
                   "--output", out],
                  TIMEOUTS["probe"], artifact=out, force=args.force)

    if "synthetic" in args.phases:
        for name, extra in (("leak-accumulation", ("--epochs",)),
                            ("subspace-joint", ("--epochs",))):
            out = os.path.join(artdir, f"{name}_toy_e{args.synthetic_epochs}.json")
            drive(workdir, {"cell_id": f"synthetic:{name}", "kind": "synthetic",
                            "artifact": out},
                  [sys.executable, "-m", "attacker", "--mode", "training",
                   "--attack", name, "--toy", "--epochs",
                   str(args.synthetic_epochs), "--output", out],
                  TIMEOUTS["synthetic"], artifact=out, force=args.force)

    # final machine-readable index (the jsonl is the crash-safe source)
    cells = []
    jl = os.path.join(workdir, "run-index.jsonl")
    if os.path.isfile(jl):
        with open(jl) as f:
            cells = [json.loads(x) for x in f if x.strip()]
    with open(os.path.join(workdir, "run-index.json"), "w") as f:
        json.dump({"manifest": manifest, "cells": cells}, f, indent=1)
    n_err = sum(1 for c in cells if c.get("rc") not in (0, None))
    print(f"[run_long_horizon] {len(cells)} cells journaled, {n_err} "
          f"non-zero rc -> {workdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
