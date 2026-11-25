"""Resolve configs and run one experiment cell.

Usage:
  python -m pcr.run_experiment --config experiments/configs/pilot_matrix.yaml --run trained-short-low-s101
  python -m pcr.run_experiment --config experiments/configs/pilot_matrix.yaml \
      --mode trained --span short --width low --seed 101 --steps 200 --run-id smoke-trained-s101

Resolution turns named factor levels (span short/medium/long, width low/high)
into a fully explicit resolved config; every executed run records the resolved
config and its sha256 in the run manifest.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from .train import run_cell
from .utils import sha256_file

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MATRIX = REPO_ROOT / "experiments" / "configs" / "pilot_matrix.yaml"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "pilot"


def load_matrix(path: str | Path = DEFAULT_MATRIX) -> dict:
    with Path(path).open() as handle:
        return yaml.safe_load(handle)


def cell_run_id(mode: str, span: str | None, width: str | None, seed: int) -> str:
    if mode == "all_local":
        return f"all_local-s{seed}"
    if mode == "removed_remote":
        return f"removed_remote-{span}-s{seed}"
    return f"{mode}-{span}-{width}-s{seed}"


def resolve_cell(
    matrix: dict,
    mode: str,
    seed: int,
    span: str | None = None,
    width: str | None = None,
    steps: int | None = None,
    run_id: str | None = None,
    cohort: str | None = None,
    output_root: str | Path | None = None,
    device: str | None = None,
    pairing_block_id: str | None = None,
    data_path: str | None = None,
    save_final_state: bool = False,
) -> dict:
    fixed = matrix["fixed"]
    factors = matrix["factors"]
    backbone = dict(fixed["backbone"])

    split = None
    if mode != "all_local":
        span_level = factors["segment_span"][span]
        if mode == "oracle":
            width_dim, q_bits = backbone["d_model"], 32
        elif mode == "removed_remote":
            controls = matrix.get("controls", {}).get("removed_remote", {})
            width_dim = factors["latent_width"][controls.get("width", "low")]
            q_bits = controls.get("q_bits", fixed["boundary_defense"]["q_bits"])
        else:
            width_dim = factors["latent_width"][width]
            q_bits = fixed["boundary_defense"]["q_bits"]
        split = {
            "start": span_level["start"],
            "span": span_level["span"],
            "width": width_dim,
            "q_bits": q_bits,
        }

    if mode in ("trained", "frozen_random"):
        block_id = pairing_block_id or matrix["pairing"]["block_id"].format(seed=seed)
    else:
        block_id = None

    corpus_path = data_path or fixed["corpus"]
    optim = dict(fixed["optimizer"])
    resolved = {
        "run_id": run_id or cell_run_id(mode, span, width, seed),
        "cohort": cohort or matrix.get("cohort", "exploratory"),
        "protocol_version": matrix["protocol_version"],
        "mode": mode,
        "seed": seed,
        "pairing_block_id": block_id,
        "device": device or fixed.get("device", "auto"),
        "backbone": backbone,
        "split": split,
        "data": {
            "path": str(corpus_path),
            "corpus_sha256": sha256_file(str(corpus_path)),
            "train_fraction": fixed["corpus_split"]["train_fraction"],
            "batch_size": fixed["batch_size"],
            "seq_len": fixed["sequence_length"],
            "eval_batches": fixed["eval_batches"],
        },
        "optim": {
            "lr": optim["lr"],
            "min_lr": optim["min_lr"],
            "betas": list(optim["betas"]),
            "weight_decay": optim["weight_decay"],
            "warmup_steps": fixed["scheduler"]["warmup_steps"],
            "grad_clip": optim["grad_clip"],
            "steps": steps or fixed["steps"],
        },
        "logging": {
            "eval_every": fixed["eval_every"],
            "eval_batches": fixed["eval_batches"],
            "transcript_sample_every": fixed["transcript_sample_every"],
            "probe_batch_size": fixed["probe_batch_size"],
            "ema_beta": fixed.get("ema_beta", 0.95),
        },
        "transcript_schema_version": fixed["transcript_schema_version"],
    }
    out_root = Path(output_root) if output_root else DEFAULT_OUTPUT_ROOT
    resolved["output_dir"] = str(out_root / resolved["run_id"])
    if save_final_state:
        # additive opt-in (P3 downstream probes); absent from the resolved
        # config otherwise, so existing resolved-config hashes are unaffected
        resolved["save_final_state"] = True
    return resolved


def resolve_calibration_cell(matrix: dict, output_root: str | Path | None = None) -> dict:
    cal = matrix["calibration"]
    resolved = resolve_cell(
        matrix,
        mode=cal["mode"],
        seed=cal["seed"],
        span=cal["span"],
        width=cal["width"],
        steps=cal["steps"],
        run_id=f"calibration-s{cal['seed']}",
        cohort="calibration",
        output_root=output_root,
        pairing_block_id=f"pb-{cal['seed']}",
    )
    resolved["calibration"] = {
        "capture_every": cal["capture_every"],
        "capture_max_pairs": cal["capture_max_pairs"],
    }
    return resolved


def enumerate_pilot_runs(matrix: dict, output_root: str | Path | None = None) -> list[dict]:
    """All pilot cells: 3 all_local + 9 removed_remote + 36 trained/frozen + 1 calibration."""
    seeds = matrix["factors"]["seed"]
    spans = list(matrix["factors"]["segment_span"].keys())
    widths = list(matrix["factors"]["latent_width"].keys())
    runs: list[dict] = []
    for seed in seeds:
        runs.append(resolve_cell(matrix, "all_local", seed, output_root=output_root))
    for seed in seeds:
        for span in spans:
            runs.append(resolve_cell(matrix, "removed_remote", seed, span=span, output_root=output_root))
    for seed in seeds:
        for span in spans:
            for width in widths:
                for mode in ("trained", "frozen_random"):
                    runs.append(
                        resolve_cell(matrix, mode, seed, span=span, width=width, output_root=output_root)
                    )
    runs.append(resolve_calibration_cell(matrix, output_root=output_root))
    return runs


def main(argv: list[str] | None = None) -> int:
    from . import manifest as manifest_state

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_MATRIX))
    parser.add_argument("--run", help="canonical run id from the pilot enumeration")
    parser.add_argument("--mode", choices=["all_local", "trained", "frozen_random", "removed_remote", "oracle"])
    parser.add_argument("--span", choices=["short", "medium", "long"])
    parser.add_argument("--width", choices=["low", "high"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--run-id")
    parser.add_argument("--cohort")
    parser.add_argument("--pairing-block-id")
    parser.add_argument("--output-root")
    parser.add_argument("--manifest-dir", help="separate manifest namespace for a new execution; requires fresh outputs")
    parser.add_argument("--device")
    parser.add_argument("--data-path")
    args = parser.parse_args(argv)
    if args.manifest_dir:
        manifest_state.MANIFEST_DIR = Path(args.manifest_dir).resolve()

    matrix = load_matrix(args.config)
    if args.run:
        runs = {r["run_id"]: r for r in enumerate_pilot_runs(matrix, args.output_root)}
        if args.run not in runs:
            parser.error(f"unknown run '{args.run}'; known: {sorted(runs)}")
        resolved = runs[args.run]
        if args.steps:
            resolved["optim"]["steps"] = args.steps
        if args.device:
            resolved["device"] = args.device
    else:
        missing = [f"--{k}" for k in ("mode", "seed") if getattr(args, k) is None]
        if args.mode not in (None, "all_local"):
            if args.span is None:
                missing.append("--span")
            if args.mode in ("trained", "frozen_random") and args.width is None:
                missing.append("--width")
        if missing:
            parser.error(f"missing required flags: {missing}")
        resolved = resolve_cell(
            matrix,
            mode=args.mode,
            seed=args.seed,
            span=args.span,
            width=args.width,
            steps=args.steps,
            run_id=args.run_id,
            cohort=args.cohort,
            output_root=args.output_root,
            device=args.device,
            pairing_block_id=args.pairing_block_id,
            data_path=args.data_path,
        )

    summary = run_cell(resolved)
    print(
        f"[{summary['run_id']}] mode={summary['mode']} seed={summary['seed']} "
        f"steps={summary['steps']} final_eval_loss={summary['final_eval_loss']:.4f} "
        f"wall_time={summary['wall_time_seconds']:.1f}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
