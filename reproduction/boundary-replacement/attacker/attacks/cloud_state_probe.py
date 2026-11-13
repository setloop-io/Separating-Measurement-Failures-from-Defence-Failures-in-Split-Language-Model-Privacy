#!/usr/bin/env python3
"""Cloud-state and protocol-metadata probe for captured compromised-node sessions.

Measures whether the captured cloud-side state (final surrogate weights) and the
protocol metadata (per-frame sizes, step indices, inter-arrival timing) leak
token information BEYOND the wire frames themselves.  Inputs join two captured
views of the SAME execution:

  --capture-dir   a koula capture session directory (wire_XXXXXX.pt/json
                  sidecars + cloud_state.pt + manifest.json; schemas
                  dtraining.koula_compromised_view.v1 / dtraining.koula_cloud_state.v1)
  --bundle        the trusted exact-capture label join written by the runner
                  (dtraining.exact_forward_wire_label_join.v1)

Arms (all scored as top-1 token recovery vs the label-free majority baseline,
Wilson upper-95 excess in pp, deep_inversion_probe scoring idiom):

  frames_alone        wire rows only (baseline arm for the delta)
  state_only          flattened final cloud_state model weights, constant
                      across rows by construction — the state alone can carry
                      no per-row token signal; this arm is the sanity floor
  frames_plus_state   wire rows concatenated with the state vector; the excess
                      delta over frames_alone is the state's marginal leak
  metadata_only       per-frame sidecar features (frame bytes, step, microbatch
                      id, inter-arrival timing) broadcast to the frame's rows —
                      the traffic-analysis arm

The module fails closed with clear errors when inputs are missing or the
capture/bundle join does not hold (shape or value mismatch on any joined frame).
"""

import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None

from .. import artifacts
from .common import add_common_args, require_torch

EXPERIMENT_ID = "cloud_state_probe"
MODES = ("training",)
REQUIRES_LABELS = True
DESCRIPTION = ("Cloud-state and protocol-metadata probe: leaks beyond the wire "
               "frames of a captured compromised-node session")

THREAT_MODEL = (
    "Fully compromised cloud node: the attacker holds every captured wire "
    "frame, the final cloud weights and optimizer state, and the protocol "
    "metadata (frame sizes, step indices, timing, session digests). Labels "
    "stay on the trusted node and enter only through the exact-capture label "
    "join. The question is whether state/metadata leak token information "
    "BEYOND the wire frames; frames_alone is the reference arm and "
    "frames_plus_state measures the marginal state leak."
)


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--capture-dir", help="koula capture session directory")
    ap.add_argument("--bundle", help="trusted exact-capture label bundle (.pt)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--probe-width", type=int, default=128,
                    help="hidden width of the per-row MLP probes")
    return ap


def wilson_upper(correct, total, alpha, trials=1):
    if total <= 0:
        return float("nan")
    z = NormalDist().inv_cdf(1.0 - alpha / max(1, trials))
    p = correct / total
    den = 1.0 + z * z / total
    center = p + z * z / (2.0 * total)
    radius = z * math.sqrt(p * (1.0 - p) / total
                           + z * z / (4.0 * total * total))
    return (center + radius) / den


def load_capture(capture_dir):
    """Load and validate the compromised-node capture session."""
    root = Path(capture_dir)
    if not root.is_dir():
        raise SystemExit(f"capture directory not found: {root}")
    manifest_path = root / "manifest.json"
    state_path = root / "cloud_state.pt"
    if not manifest_path.exists():
        raise SystemExit(f"capture manifest missing: {manifest_path}")
    if not state_path.exists():
        raise SystemExit(f"cloud state missing: {state_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "dtraining.koula_compromised_view.v1":
        raise SystemExit(f"{manifest_path}: not a koula_compromised_view "
                         "artifact")
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if state.get("schema") != "dtraining.koula_cloud_state.v1":
        raise SystemExit(f"{state_path}: not a koula_cloud_state artifact")
    sidecars = {}
    for meta_path in sorted(root.glob("wire_*.json")):
        meta = json.loads(meta_path.read_text())
        tensor_path = meta_path.with_suffix(".pt")
        if not tensor_path.exists():
            raise SystemExit(f"sidecar without tensor: {tensor_path}")
        tensor = torch.load(tensor_path, map_location="cpu",
                            weights_only=True)
        sidecars.setdefault(meta["phase"], []).append(
            {"tensor": tensor, "mb_id": meta["mb_id"], "step": meta["step"],
             "mtime": meta_path.stat().st_mtime,
             "nbytes": tensor_path.stat().st_size})
    if not sidecars.get("fwd"):
        raise SystemExit("capture has no forward (fwd) wire sidecars")
    return manifest, state, sidecars


def flatten_state(model_state):
    parts = []
    for key in sorted(model_state):
        parts.append(model_state[key].detach().float().reshape(-1).cpu())
    if not parts:
        raise SystemExit("cloud_state.pt model_state is empty")
    return torch.cat(parts)


def join_frames(bundle, sidecars):
    """Join bundle frames to forward sidecars by microbatch id; fail closed."""
    if bundle.get("schema") != "dtraining.exact_forward_wire_label_join.v1":
        raise SystemExit("--bundle is not an exact_forward_wire_label_join "
                         "artifact")
    fwd_by_mb = {}
    for frame in sidecars["fwd"]:
        fwd_by_mb.setdefault(frame["mb_id"], []).append(frame)
    joined = {"train": [], "eval": []}
    for partition in ("train", "eval"):
        wires = bundle[f"{partition}_wire"]
        tokens = bundle[f"{partition}_tokens"]
        mb_ids = bundle[f"{partition}_mb_ids"]
        if not (len(wires) == len(tokens)):
            raise SystemExit(f"bundle {partition} partition is ragged")
        # Eval packets coalesce up to 64 blocks under one microbatch id:
        # expand such ids to one per frame when the bundle is grouped.
        if len(mb_ids) != len(wires):
            if len(mb_ids) and len(wires) % len(mb_ids) == 0:
                group = len(wires) // len(mb_ids)
                mb_ids = [mb for mb in mb_ids for _ in range(group)]
            else:
                raise SystemExit(
                    f"bundle {partition} partition is ragged: "
                    f"{len(wires)} frames vs {len(mb_ids)} mb_ids")
        # Group consecutive frames sharing one mb_id so coalesced packets
        # compare as single tensors against their sidecar.
        groups = []
        for wire, labels, mb_id in zip(wires, tokens, mb_ids):
            if groups and groups[-1][0] == mb_id:
                groups[-1][1].append((wire, labels))
            else:
                groups.append((mb_id, [(wire, labels)]))
        for mb_id, frame_group in groups:
            matches = fwd_by_mb.get(int(mb_id))
            if not matches:
                raise SystemExit(
                    f"no forward sidecar joins bundle mb_id={mb_id} "
                    f"({partition}); the capture session does not cover the "
                    "label join")
            group_wire = torch.stack([w for w, _ in frame_group])
            exact = []
            for f in matches:
                t = f["tensor"].reshape(-1, *group_wire.shape[-2:])
                if t.shape == group_wire.shape and torch.equal(
                        t, group_wire.float().cpu()):
                    exact.append(f)
            if not exact:
                raise SystemExit(
                    f"capture/bundle value mismatch at mb_id={mb_id}: the "
                    "sidecar is not the transmitted frame the bundle labels")
            for wire, labels in frame_group:
                joined[partition].append((wire.float(), labels.long(),
                                          exact[0]))
    return joined


def metadata_features(joined_frames):
    """Per-frame traffic features, z-scored: bytes, step, mb index, timing."""
    frames = joined_frames["train"] + joined_frames["eval"]
    first_mtime = min(frame["mtime"] for _, _, frame in frames)
    feats = torch.tensor(
        [[frame["nbytes"], float(frame["step"]), float(frame["mb_id"]),
          frame["mtime"] - first_mtime] for _, _, frame in frames],
        dtype=torch.float32)
    mean = feats.mean(dim=0, keepdim=True)
    std = feats.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (feats - mean) / std


def row_probe_class(width):
    """Small per-row MLP probe (built here so torch stays inside run())."""
    class RowProbe(nn.Module):
        def __init__(self, in_dim, num_classes):
            super().__init__()
            self.net = nn.Sequential(
                nn.LayerNorm(in_dim), nn.Linear(in_dim, width), nn.GELU(),
                nn.Linear(width, num_classes))

        def forward(self, x):
            return self.net(x)

    return RowProbe


def train_row_probe(train_x, train_y, eval_x, eval_y, classes, args):
    """One arm: small per-row MLP, best-of-restarts top-1 vs majority."""
    probe_class = row_probe_class(args.probe_width)
    lookup = torch.full((int(classes.max()) + 1,), -1, dtype=torch.long)
    lookup[classes] = torch.arange(classes.numel())
    train_classes = lookup[train_y]
    eval_classes = lookup[eval_y]
    known_eval = eval_classes >= 0
    flat_train = train_classes.reshape(-1)
    majority = flat_train.mode().values.item()
    majority_correct = (eval_classes == majority).sum().item()
    total = eval_y.numel()
    majority_acc = majority_correct / total if total > 0 else 0.0

    scores = []
    for restart in range(args.restarts):
        torch.manual_seed(args.seed + 1009 * restart)
        model = probe_class(train_x.shape[-1], classes.numel())
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                      weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        n_rows = train_x.shape[0]
        for _epoch in range(args.epochs):
            model.train()
            order = torch.randperm(n_rows)
            for start in range(0, n_rows, args.batch_size):
                idx = order[start:start + args.batch_size]
                optimizer.zero_grad()
                loss = criterion(model(train_x[idx]),
                                 train_classes.reshape(-1)[idx])
                loss.backward()
                optimizer.step()
        model.eval()
        with torch.no_grad():
            preds = model(eval_x).argmax(dim=-1).reshape(-1)
            target = eval_classes.reshape(-1)
            correct = (preds[known_eval.reshape(-1)]
                       == target[known_eval.reshape(-1)]).sum().item()
        scores.append(correct / total if total > 0 else 0.0)
    best = max(scores)
    upper = wilson_upper(int(best * total), total, 0.05, args.restarts)
    return {
        "majority_acc": majority_acc,
        "best_eval_acc": best,
        "wilson_upper": upper,
        "excess_pp": (upper - majority_acc) * 100,
        "restart_scores": scores,
    }


def run(args):
    require_torch(EXPERIMENT_ID)
    if args.toy:
        raise SystemExit("cloud_state_probe requires a real capture session")
    if not args.capture_dir or not args.bundle:
        raise SystemExit("--capture-dir and --bundle are required")

    manifest, state, sidecars = load_capture(args.capture_dir)
    bundle = torch.load(args.bundle, map_location="cpu", weights_only=False)
    joined = join_frames(bundle, sidecars)
    meta = metadata_features(joined)

    state_vec = flatten_state(state["model_state"])
    frames = joined["train"] + joined["eval"]
    n_train_frames = len(joined["train"])
    wire = torch.stack([wire for wire, _, _ in frames], dim=0)
    labels = torch.stack([lab for _, lab, _ in frames], dim=0)
    rows, rows_per_frame, width = wire.shape
    flat_wire = wire.reshape(rows * rows_per_frame, width)
    flat_labels = labels.reshape(-1)
    # Row i of frame f inherits frame f's state/metadata features.
    state_rows = state_vec.repeat(rows, rows_per_frame, 1).reshape(
        rows * rows_per_frame, -1)
    meta_rows = meta.repeat_interleave(rows_per_frame, dim=0)
    split_row = n_train_frames * rows_per_frame

    classes = torch.unique(flat_labels[:split_row]).sort().values
    split = {
        "train": slice(0, split_row),
        "eval": slice(split_row, rows * rows_per_frame),
    }

    def row_set(features):
        return (features[split["train"]], flat_labels[split["train"]],
                features[split["eval"]], flat_labels[split["eval"]])

    arm_features = {
        "frames_alone": flat_wire,
        "state_only": state_rows,
        "frames_plus_state": torch.cat([flat_wire, state_rows], dim=-1),
        "metadata_only": meta_rows,
    }

    out = artifacts.make_artifact(
        "dtraining.attacker.cloud_state_probe.v1",
        {"attack": EXPERIMENT_ID, "mode": args.mode,
         "capture_dir": str(args.capture_dir),
         "bundle": str(args.bundle),
         "session_id": manifest.get("session_id"),
         "privacy_cell_id": manifest.get("privacy_cell_id"),
         "frames_joined": len(frames),
         "train_frames": n_train_frames,
         "rows_per_frame": rows_per_frame, "latent_dim": width,
         "state_params": int(state_vec.numel()),
         "epochs": args.epochs, "batch_size": args.batch_size,
         "restarts": args.restarts, "probe_width": args.probe_width},
        THREAT_MODEL,
        interpretation=(
            "state_only is constant across rows by construction and serves as "
            "the sanity floor; the state's marginal leak is frames_plus_state "
            "minus frames_alone; metadata_only is the traffic-analysis arm."),
    )

    results = {}
    for arm, features in arm_features.items():
        train_x, train_y, eval_x, eval_y = row_set(features)
        result = train_row_probe(train_x, train_y, eval_x, eval_y, classes,
                                 args)
        result["arm"] = arm
        results[arm] = result
        out["results"].append(result)
        print(f"[{arm}] best {result['best_eval_acc']*100:.2f}% vs majority "
              f"{result['majority_acc']*100:.2f}% -> excess "
              f"{result['excess_pp']:+.2f} pp")

    out["summary"].append({
        "state_marginal_leak_pp": (
            results["frames_plus_state"]["excess_pp"]
            - results["frames_alone"]["excess_pp"]),
        "metadata_excess_pp": results["metadata_only"]["excess_pp"],
        "state_only_excess_pp": results["state_only"]["excess_pp"],
        "frames_alone_excess_pp": results["frames_alone"]["excess_pp"],
    })
    print("[summary] state marginal leak "
          f"{out['summary'][0]['state_marginal_leak_pp']:+.2f} pp; metadata "
          f"{out['summary'][0]['metadata_excess_pp']:+.2f} pp")

    artifacts.write_artifact(
        args.output or f"cloud_state_probe_{manifest.get('session_id')}.json",
        out)
    return 0


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
