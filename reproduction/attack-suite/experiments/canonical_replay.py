"""Canonical replay for the accumulation probe on real splitbench captures.

splitbench --save-canonical writes <capture_dir>/canonical.pt as a
LongTensor [n_captured*batch, seq] of the victim's original input_ids,
row-aligned to the fwd wire captures (step-major). attacks/accumulation's
real-capture path instead expects a torch-saved dict {"h": [N, H],
"tok": [N]}: the ATTACKER's canonical replay — the same ids pushed
through the PUBLIC base head (split at --split), one row per (sequence,
position), the position-aligned convention attacks/seam_inversion uses
for scoring. This helper produces exactly that file.

Rows are emitted in capture order (sequence-major, position-minor), which
is the row order captures.load_epoch_rows produces for the wire side, so
the accumulation module's within-epoch row alignment holds.

    .venv/bin/python experiments/canonical_replay.py \
        --model gpt2 --split 6 \
        --canonical-pt <capture_dir>/canonical.pt \
        --out <capture_dir>/canonical-replay.pt
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="6")
    ap.add_argument("--canonical-pt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--batch", type=int, default=16,
                    help="sequences per forward pass")
    args = ap.parse_args()

    import torch
    from transformers import GPT2LMHeadModel
    from attacker.splitbench import SplitPlan, build_split

    ids = torch.load(args.canonical_pt, map_location="cpu").long()
    if ids.dim() != 2:
        raise SystemExit(f"[canonical-replay] canonical.pt must be "
                         f"[n_seq, seq], got {tuple(ids.shape)}")
    model = GPT2LMHeadModel.from_pretrained(args.model).to(args.device)
    model.eval()
    plan = SplitPlan.parse(args.split, n_layers=model.config.n_layer)
    split = build_split(model, plan)
    rows = []
    with torch.no_grad():
        for i in range(0, ids.shape[0], args.batch):
            h = split.head_forward(ids[i:i + args.batch].to(args.device))
            rows.append(h.reshape(-1, h.shape[-1]).cpu())
    h_all = torch.cat(rows).float()
    tok_all = ids.reshape(-1)
    torch.save({"h": h_all, "tok": tok_all}, args.out)
    print(f"[canonical-replay] {ids.shape[0]} seqs x {ids.shape[1]} tok -> "
          f"h {tuple(h_all.shape)}, tok {tuple(tok_all.shape)} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
