#!/usr/bin/env python3
"""v11.2: per-layer MI-budget estimation.

For each probed layer, trains a small token probe on that layer's
activations and converts probe cross-entropy into a mutual-information
lower bound:  I(activations; tokens) >= H(tokens) - CE_probe  (nats),
reported in bits.  Run twice — on the undefended backbone (prefix layers)
and on the defended boundary latent (encoder output) — to show the
information decay the bottleneck enforces.

This is the empirical bridge to the MI-budget formalism (UCLA
arXiv:2606.11592 / Fraunhofer WACV'26): it does not claim a bound, it
reports the lower-bound profile.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "split-training"))

from split_trainer import make_layer_kwargs, run_layer_stack


def main() -> int:
    import math
    import torch
    from torch import nn
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--layers", default="2,8,14,21",
                    help="comma-separated prefix layer indices to probe")
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--train-blocks", type=int, default=128)
    ap.add_argument("--eval-blocks", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=dtype).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    core = model.model
    layers = core.layers

    text = Path(args.corpus).read_text(errors="replace")
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    width = args.seq_len + 1
    blocks = [ids[i:i + width]
            for i in range(0, len(ids) - width + 1, width)]
    blocks = blocks[: args.train_blocks + args.eval_blocks]
    train_blocks = blocks[: args.train_blocks]
    eval_blocks = blocks[args.train_blocks:]

    probe_layers = sorted(int(v) for v in args.layers.split(","))

    def tensors(block):
        # Privacy target: the CURRENT token at each position (identity
        # recovery), not the next-token shift used for LM loss.
        ids_t = torch.tensor(block, dtype=torch.long, device=device)[None]
        return ids_t[:, :-1], ids_t[:, :-1]

    def hidden_at(input_ids, upto):
        positions = torch.arange(input_ids.shape[1], device=device)[None]
        with torch.no_grad():
            hidden = core.embed_tokens(input_ids)
            kwargs = make_layer_kwargs(getattr(core, "rotary_emb", None),
                                       hidden, positions,
                                       type("Args", (),
                                            {"attn_impl": "sdpa"})())
            return run_layer_stack(layers[: upto + 1], hidden, kwargs).float()

    # Token prior entropy of the train partition (nats)
    all_labels = torch.tensor(
        [t for block in train_blocks for t in block[1:]], device=device)
    counts = torch.bincount(all_labels)
    probs = counts.float() / counts.sum()
    h_tokens = float(-(probs * (probs + 1e-12).log()).sum())

    results = []
    for layer_index in probe_layers:
        # Collect activations
        train_x, train_y = [], []
        for block in train_blocks:
            input_ids, labels = tensors(block)
            train_x.append(hidden_at(input_ids, layer_index).cpu())
            train_y.append(labels.cpu())
        eval_x, eval_y = [], []
        for block in eval_blocks:
            input_ids, labels = tensors(block)
            eval_x.append(hidden_at(input_ids, layer_index).cpu())
            eval_y.append(labels.cpu())
        train_x = torch.cat(train_x); train_y = torch.cat(train_y)
        eval_x = torch.cat(eval_x); eval_y = torch.cat(eval_y)

        vocab = int(model.config.vocab_size)
        probe = nn.Sequential(
            nn.LayerNorm(train_x.shape[-1]),
            nn.Linear(train_x.shape[-1], 512), nn.GELU(),
            nn.Linear(512, vocab)).to(device)
        opt = torch.optim.AdamW(probe.parameters(), lr=1e-3)
        for _ in range(args.epochs):
            perm = torch.randperm(train_x.shape[0])
            for start in range(0, train_x.shape[0], 8):
                idx = perm[start:start + 8]
                bx = train_x[idx].to(device)
                by = train_y[idx].to(device)
                opt.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(
                    probe(bx).flatten(0, 1), by.flatten())
                loss.backward()
                opt.step()
        # Temperature calibration on the first eval quarter (an uncalibrated
        # probe's CE overestimates H(Y|X), making the MI bound vacuous)
        cal_n = max(1, eval_x.shape[0] // 4)
        with torch.no_grad():
            cal_logits = probe(eval_x[:cal_n].to(device))
            cal_labels = eval_y[:cal_n].to(device)
        temperature = torch.ones((), device=device, requires_grad=True)
        cal_opt = torch.optim.LBFGS([temperature], lr=0.05, max_iter=64)
        def closure():
            cal_opt.zero_grad()
            loss = nn.functional.cross_entropy(
                (cal_logits / temperature.clamp_min(1e-3)).flatten(0, 1),
                cal_labels.flatten())
            loss.backward()
            return loss
        cal_opt.step(closure)
        # Eval CE and accuracy
        total_ce, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for start in range(cal_n, eval_x.shape[0], 16):
                bx = eval_x[start:start + 16].to(device)
                by = eval_y[start:start + 16].to(device)
                logits = probe(bx) / temperature.clamp_min(1e-3)
                total_ce += float(nn.functional.cross_entropy(
                    logits.flatten(0, 1), by.flatten(),
                    reduction="sum").item())
                correct += int((logits.argmax(-1) == by).sum())
                total += by.numel()
        ce = total_ce / max(1, total)
        mi_lower_bits = max(0.0, (h_tokens - ce) / math.log(2))
        results.append({
            "layer": layer_index,
            "probe_ce_nats": ce,
            "token_prior_entropy_nats": h_tokens,
            "mi_lower_bound_bits_per_token": mi_lower_bits,
            "top1_pct": 100.0 * correct / max(1, total),
        })
        print(f"layer {layer_index}: CE={ce:.3f} MI>={mi_lower_bits:.3f} bits "
              f"top1={100.0*correct/max(1,total):.2f}%")

    out = {
        "schema": "dtraining.mi_budget.v1",
        "model": str(args.model), "seed": args.seed,
        "layers_probed": probe_layers,
        "target": "current-token identity (privacy variable)",
        "convention": "I(activations;tokens) >= H(tokens) - CE_probe, bits; "
                      "probe is a small MLP, so this is a LOWER bound",
        "results": results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2)[:600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
