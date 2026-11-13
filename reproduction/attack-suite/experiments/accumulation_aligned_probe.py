"""Aligned cross-epoch accumulation probe on real splitbench captures.

attacks/accumulation.py's real-capture path misaligns the victim labels
whenever the capture spans more than one epoch: the victim wire rows are
the tail of the FIRST epoch (`wire_by_ep[ep0][-nv:]`) while the victim
token labels are taken as `canon_tok[off-nv:off]` — the tail of the LAST
epoch (off accumulates over every epoch's rows). Single-epoch captures
are unaffected. This probe re-runs the identical solve with the
framework's own primitives (attacker.solve_primitives.solve_w,
attacks/common.recovery_with_what_nn, label_free_band) but with the
victim labels taken from epoch 0's canonical tail, so the cross-epoch
pooling control is measured against the correct labels. The framework
module's own output is left untouched and reported alongside.

    .venv/bin/python experiments/accumulation_aligned_probe.py \
        --capture-dir CAPS --canonical-pt CAPS/canonical-replay.pt \
        --phase fwd --epoch-sizes 64 256 768 --output out.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture-dir", required=True)
    ap.add_argument("--canonical-pt", required=True,
                    help="canonical-replay.pt from experiments/canonical_replay.py")
    ap.add_argument("--phase", default="fwd")
    ap.add_argument("--epoch-sizes", type=int, nargs="+",
                    default=[64, 256, 768])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--solve-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    import torch
    from attacker import captures
    from attacker.solve_primitives import partition_pairs, solve_w
    from attacker.attacks.common import (label_free_band,
                                         recovery_with_what_nn)

    records = captures.scan_captures(args.capture_dir, "training")
    by_ep = captures.load_epoch_rows(records, phase=args.phase)
    can = torch.load(args.canonical_pt, map_location="cpu")
    canon_rows, canon_tok = can["h"].float(), can["tok"]
    canon_by_ep, tok_by_ep, off = {}, {}, 0
    epochs = sorted(by_ep, key=lambda e: (e is None, e))
    for ep in epochs:
        n = by_ep[ep].shape[0]
        canon_by_ep[ep] = canon_rows[off:off + n]
        tok_by_ep[ep] = canon_tok[off:off + n]
        off += n
    ep0 = epochs[0]
    nv = min(512, by_ep[ep0].shape[0] // 4)
    victim = {"h_wire": by_ep[ep0][-nv:].double(),
              "ref_h": canon_rows, "ref_tok": canon_tok,
              # the fix: labels from epoch 0's canonical-token tail,
              # aligned to h_wire (framework uses canon_tok[off-nv:off] =
              # the LAST epoch's tail when epochs > 1)
              "victim_tok": tok_by_ep[ep0][-nv:]}

    def solve_and_score(h_in, hw_in, seed):
        recs = []
        for ss in args.solve_seeds:
            g = torch.Generator().manual_seed(seed + 500000 + ss)
            order = torch.randperm(h_in.shape[0], generator=g)
            w_hat, tag = solve_w(h_in[order], hw_in[order])
            if w_hat is None:
                continue
            recs.append(recovery_with_what_nn(
                victim["h_wire"], w_hat, victim["ref_h"], victim["ref_tok"],
                victim["victim_tok"]))
        return (sum(recs) / len(recs)) if recs else None

    results = []
    for seed in args.seeds:
        band = label_free_band(canon_by_ep[ep0][-nv:], victim["ref_h"],
                               victim["ref_tok"], victim["victim_tok"],
                               args.seeds[0] + seed)
        for E in args.epoch_sizes:
            if E > canon_by_ep[ep0].shape[0]:
                continue
            within = solve_and_score(canon_by_ep[ep0][:E],
                                     by_ep[ep0][:E].double(), seed)
            parts = partition_pairs(E, min(4, len(epochs)))
            hp, hwp, off2 = [], [], E
            for t, n_t in zip(epochs, parts):
                hp.append(canon_by_ep[t][off2:off2 + n_t])
                hwp.append(by_ep[t][off2:off2 + n_t].double())
                off2 += n_t
            pooled = solve_and_score(torch.cat(hp), torch.cat(hwp),
                                     seed + 77)
            results.append({"seed": seed, "epoch_size": E,
                            "n_epochs": len(epochs),
                            "within_epoch_top1": within,
                            "pooled_cross_epoch_top1": pooled,
                            "label_free_band_top1": band})
    out = {"probe": "accumulation_aligned", "capture_dir": args.capture_dir,
           "phase": args.phase, "epochs_present": len(epochs),
           "rows_per_epoch": {str(e): int(by_ep[e].shape[0]) for e in epochs},
           "victim_rows": nv,
           "note": "victim labels aligned to epoch 0 (framework "
                   "attacks/accumulation.py misaligns them for "
                   "multi-epoch real captures); solves use the "
                   "framework's solve_w/recovery primitives verbatim.",
           "results": results}
    with open(args.output, "w") as f:
        json.dump(out, f, indent=1)
    for r in results:
        print(f"[aligned] E={r['epoch_size']} seed={r['seed']}: "
              f"within={r['within_epoch_top1']} pooled="
              f"{r['pooled_cross_epoch_top1']} band="
              f"{r['label_free_band_top1']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
