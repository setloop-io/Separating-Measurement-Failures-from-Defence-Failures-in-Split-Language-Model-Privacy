#!/usr/bin/env python3
"""CE-probe MI cross-check (v14/P3): an independent, second-family mutual
information lower bound on a released-latent bundle.

Fano-style bound from a trained token probe:
    I(wire rows ; token identity) >= H(tokens) - CE_probe   (nats),
reported in bits.  Complements the MINE Donsker-Varadhan estimator used in
training: different estimator family, same released view, same labels.  If
MINE reads ~0 and this probe also reads ~0 on the defended boundary, the
"certified at the estimator floor" claim no longer rests on one instrument.

Bundle schema: dtraining.latent_release_bundle.v1 (train/eval wire + tokens).
The bundle is trusted-side capture: delete it after scoring.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path


def main() -> int:
    import torch
    from torch import nn

    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    started = time.perf_counter()

    bundle = torch.load(args.bundle, map_location="cpu", weights_only=False)
    train_x = bundle["train_wire"].reshape(-1, bundle["train_wire"].shape[-1])
    train_y = bundle["train_tokens"].reshape(-1)
    eval_x = bundle["eval_wire"].reshape(-1, bundle["eval_wire"].shape[-1])
    eval_y = bundle["eval_tokens"].reshape(-1)
    keep_tr = train_y >= 0
    keep_ev = eval_y >= 0
    train_x, train_y = train_x[keep_tr], train_y[keep_tr]
    eval_x, eval_y = eval_x[keep_ev], eval_y[keep_ev]
    latent_dim = int(train_x.shape[-1])
    vocab = int(max(train_y.max().item(), eval_y.max().item())) + 1

    counts = torch.bincount(train_y, minlength=vocab).float()
    probs = counts / counts.sum()
    h_tokens_nats = float(-(probs[probs > 0] * probs[probs > 0].log()).sum())

    def make_probe():
        return nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, 512), nn.GELU(),
            nn.Linear(512, vocab)).to(device)

    tx, ty = train_x.float(), train_y
    ex, ey = eval_x.float().to(device), eval_y.to(device)
    best = None
    for restart in range(args.restarts):
        torch.manual_seed(args.seed + restart)
        probe = make_probe()
        opt = torch.optim.AdamW(probe.parameters(), lr=args.lr)
        for _ in range(args.epochs):
            perm = torch.randperm(tx.shape[0])
            for start in range(0, tx.shape[0], args.batch_size):
                idx = perm[start:start + args.batch_size]
                opt.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(
                    probe(tx[idx].to(device)), ty[idx].to(device))
                loss.backward()
                opt.step()
        with torch.no_grad():
            logits = probe(ex)
            ce = float(nn.functional.cross_entropy(logits, ey).item())
            top1 = float((logits.argmax(-1) == ey).float().mean().item())
        if best is None or ce < best["ce_nats"]:
            best = {"ce_nats": ce, "top1_pct": 100.0 * top1,
                    "restart": restart}

    mi_nats = h_tokens_nats - best["ce_nats"]
    result = {
        "schema": "dtraining.ce_probe_mi.v1",
        "bundle": str(args.bundle),
        "latent_dim": latent_dim,
        "train_rows": int(tx.shape[0]), "eval_rows": int(ex.shape[0]),
        "epochs": args.epochs, "restarts": args.restarts, "seed": args.seed,
        "prior_entropy_nats": h_tokens_nats,
        "probe_ce_nats": best["ce_nats"],
        "probe_top1_pct": best["top1_pct"],
        "mi_lower_bound_nats": mi_nats,
        "mi_lower_bound_bits": mi_nats / math.log(2),
        "seconds": time.perf_counter() - started,
        "note": ("Fano-style lower bound H(tokens) - CE(probe); independent "
                 "second-family cross-check of the MINE DV reading on the "
                 "same released view."),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
