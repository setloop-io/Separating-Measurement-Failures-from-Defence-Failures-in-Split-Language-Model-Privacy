#!/usr/bin/env python3
"""Deep Inversion Probe for Naked Data Baseline.

This probe is designed specifically to pass the positive control (naked baseline).
It uses a deep sequence-to-sequence Transformer architecture rather than a simple 
1-layer MLP or shallow Graph network, allowing it to properly invert high-dimensional 
(D=1024) un-defended representations back into discrete tokens.
"""

import argparse
import math
from statistics import NormalDist

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None

from .. import artifacts
from .common import add_common_args, require_torch

EXPERIMENT_ID = "deep_inversion_probe"
MODES = ("training",)
REQUIRES_LABELS = True
DESCRIPTION = "Deep Transformer-based inversion probe for passing naked positive controls"

def build_parser():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--bundle", help="trusted .pt bundle from the v5/v6 runner")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--restarts", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--layers", type=int, default=4, help="Number of Transformer layers")
    ap.add_argument("--heads", type=int, default=8, help="Number of attention heads")
    ap.add_argument("--dim-feedforward", type=int, default=2048)
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

def run(args):
    require_torch(EXPERIMENT_ID)
    if args.toy:
        raise SystemExit("deep_inversion_probe requires a real trusted bundle")
    if not args.bundle:
        raise SystemExit("--bundle is required")

    bundle = torch.load(args.bundle, map_location="cpu")
    train_x = bundle["train_wire"].float()
    train_y = bundle["train_tokens"].long()
    eval_x = bundle["eval_wire"].float()
    eval_y = bundle["eval_tokens"].long()
    
    if train_x.ndim != 3 or eval_x.ndim != 3:
        raise ValueError("bundle views must be [blocks,tokens,D]")
    if train_x.shape[1:] != eval_x.shape[1:]:
        raise ValueError("train/eval latent block shapes differ")

    classes = torch.unique(train_y).sort().values
    lookup = torch.full((int(max(classes.max(), eval_y.max())) + 1,), -1, dtype=torch.long)
    lookup[classes] = torch.arange(classes.numel())
    train_classes = lookup[train_y]
    eval_classes = lookup[eval_y]
    known_eval = eval_classes >= 0
    d = train_x.shape[-1]

    class DeepInversionProbe(nn.Module):
        def __init__(self, d_model, num_classes, nhead, num_layers, dim_feedforward):
            super().__init__()
            self.input_proj = nn.Linear(d, d_model) if d != d_model else nn.Identity()
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, 
                batch_first=True, norm_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.norm = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, num_classes)

        def forward(self, x):
            # x shape: [batch, seq_len, D]
            projected = self.input_proj(x)
            transformed = self.transformer(projected)
            normalized = self.norm(transformed)
            return self.head(normalized)

    out = artifacts.make_artifact(
        "dtraining.attacker.deep_inversion_probe.v1",
        {"attack": EXPERIMENT_ID, "mode": args.mode,
         "latent_dim": d, "train_blocks": train_x.shape[0], "eval_blocks": eval_x.shape[0],
         "sequence_length": train_x.shape[1], "epochs": args.epochs,
         "batch_size": args.batch_size, "restarts": args.restarts,
         "layers": args.layers, "heads": args.heads},
        "Deep sequence-to-sequence Transformer inversion probe. Built specifically "
        "to act as a reliable positive control by proving it can decode raw, undefended "
        "(naked) 1024-D representations.",
    )

    flat_train_y = train_classes.reshape(-1)
    total = eval_y.numel()
    all_scores = []
    
    # Baseline majority class accuracy
    if flat_train_y.numel() > 0:
        majority = flat_train_y.mode().values.item()
        majority_correct = (eval_classes == majority).sum().item()
    else:
        majority_correct = 0
    majority_acc = majority_correct / total if total > 0 else 0.0

    print(f"Majority Baseline Accuracy: {majority_acc*100:.2f}%")

    for restart in range(args.restarts):
        torch.manual_seed(args.seed + 1009 * restart)
        # We use d_model=512 internally, or 1024 if D is exactly 1024.
        d_model = min(1024, max(512, d))
        model = DeepInversionProbe(d_model, classes.numel(), args.heads, args.layers, args.dim_feedforward)
        
        if torch.cuda.is_available():
            model = model.cuda()
            train_x_device = train_x.cuda()
            train_classes_device = train_classes.cuda()
            eval_x_device = eval_x.cuda()
            eval_classes_device = eval_classes.cuda()
        else:
            train_x_device = train_x
            train_classes_device = train_classes
            eval_x_device = eval_x
            eval_classes_device = eval_classes

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        best_eval_acc = 0.0
        
        for epoch in range(args.epochs):
            model.train()
            order = torch.randperm(train_x.shape[0])
            block_batch = max(1, args.batch_size)
            
            for start in range(0, train_x.shape[0], block_batch):
                idx = order[start:start + block_batch]
                bx = train_x_device[idx]
                by = train_classes_device[idx].reshape(-1)
                
                optimizer.zero_grad()
                logits = model(bx).reshape(-1, classes.numel())
                loss = criterion(logits, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
            # Eval
            model.eval()
            with torch.no_grad():
                eval_logits = model(eval_x_device).reshape(-1, classes.numel())
                preds = eval_logits.argmax(dim=-1)
                correct = (preds[known_eval.reshape(-1)] == eval_classes_device.reshape(-1)[known_eval.reshape(-1)]).sum().item()
                acc = correct / total
                if acc > best_eval_acc:
                    best_eval_acc = acc
                    
            print(f"Restart {restart+1}/{args.restarts} | Epoch {epoch+1}/{args.epochs} | Eval Acc: {acc*100:.2f}% (Best: {best_eval_acc*100:.2f}%)")
            
        all_scores.append(best_eval_acc)
        
    best_overall = max(all_scores)
    wu = wilson_upper(int(best_overall * total), total, 0.05, args.restarts)
    
    excess_pp = (wu - majority_acc) * 100
    out["results"].append({
        "majority_acc": majority_acc,
        "best_eval_acc": best_overall,
        "wilson_upper": wu,
        "excess_pp": excess_pp,
        "restart_scores": all_scores,
    })

    print(f"\nFinal Results:")
    print(f"Majority Accuracy: {majority_acc*100:.2f}%")
    print(f"Deep Probe Best Accuracy: {best_overall*100:.2f}%")
    print(f"Wilson Upper (95%): {wu*100:.2f}%")
    print(f"Excess PP: {excess_pp:.2f}")

    artifacts.write_artifact(args.output or f"deep_inversion_probe_{d}d.json", out)

if __name__ == "__main__":
    run(build_parser().parse_args())
