#!/usr/bin/env python3
"""E9 eval: does the deployed wire obfuscation defeat the trained attacker?
Trains the MLP decoder on public 7B boundary activations (plain h), then evaluates
on the activations captured from the LIVE wire (h·W), using the known generated
tokens as ground truth. Expect ~random vs ~60% on plain capture.

Capture alignment: the cloud's capture hook (E9_CAPTURE_DIR) writes each
wire_NNNN.pt tensor with a wire_NNNN.json sidecar carrying
{"session_id", "request_seq", "phase", "position", "timestamp"}. We align
decode-phase captures to generated tokens via the sidecars (sorted by
session_id, then committed position), NOT by filename order.

Utility: held-out CE loss of the full model on a few fixed eval docs, once
plain and once with the W round-trip (h·W·W^T) inserted at the boundary —
the E9 utility co-metric (W loaded from the --obfuscate-key file the local
node used)."""
import argparse, glob, json, os, sys
import torch

sys.path.insert(0, "/workspace/experiments/split-training")
sys.path.insert(0, "/workspace/dtraining/split-training")
sys.path.insert(0, os.path.expanduser("~/dtraining/split-training"))  # si mount convention

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/workspace/experiments/models/mistral-7b-instruct")
ap.add_argument("--capture-dir", required=True)
ap.add_argument("--tokens-json", required=True, help="generated token ids from the run")
ap.add_argument("--corpus-file", default="/workspace/experiments/models/wikitext2_corpus.txt")
ap.add_argument("--w-key", default=None,
                help="local node's --obfuscate-key file (torch-saved fp32 W); "
                     "enables the with-W utility measurement")
ap.add_argument("--eval-docs", type=int, default=3,
                help="held-out docs (from AFTER the attack train/val pools) "
                     "for the utility loss")
ap.add_argument("--output", required=True)
ap.add_argument("--arm", default="unknown", help="which wire arm these captures came from (recorded in the artifact)")
ap.add_argument("--per-epoch", action="store_true",
                help="[ER] group sidecar-aligned captures by sidecar 'epoch' and "
                     "score each epoch with >=8 aligned tokens separately "
                     "(within-epoch budget), plus a pooled cross-epoch control")
args = ap.parse_args()

from trained_inversion import (collect_base_pairs, train_decoder, evaluate_decoder,
                               split_at, seed_all)
from split_trainer import build_modules, make_layer_kwargs, run_layer_stack

class A:  # minimal args namespace for build_modules/collect
    model = args.model; dtype = "bf16"; device = "cuda"; toy = False
    seq_len = 64; max_pairs = 20000; epochs = 30; batch_size = 256
    lora_rank = 16; lora_alpha = 32; attn_impl = "sdpa"; lr = 1e-4

embed, layers, norm, lm_head, rotary, encode = build_modules(A)
n_layers = len(layers)
docs = [l.strip() for l in open(args.corpus_file) if len(l.strip()) > 300]
train_docs, val_docs = docs[:400], docs[400:500]

# boundary after layer 1 (the deployed 7B split point: local layers 0-1)
head, cloud_l, tail, sa, ra = split_at(layers, 1, n_layers)
tr_h, _, tr_tok = collect_base_pairs(embed, head, cloud_l, tail, norm, lm_head,
                                     rotary, encode, train_docs, A, with_grad=False)
va_h, _, va_tok = collect_base_pairs(embed, head, cloud_l, tail, norm, lm_head,
                                     rotary, encode, val_docs, A, with_grad=False)
seed_all(0)
vocab_size = lm_head.weight.shape[0]  # model-derived (NeMo-12B 131072, 7B 32768)
dec = train_decoder(tr_h, tr_tok, va_h, va_tok, tr_h.shape[1], vocab_size, A, "e9_wire")

# --- utility: held-out loss of the model without vs with the W round-trip ---
def heldout_loss(eval_docs, W=None):
    """Mean CE on eval blocks through the full model; with W, the boundary
    activation makes the fp32 round-trip h -> h·W -> h·W·W^T (cast back to
    model dtype), as the deployed local node does."""
    losses = []
    with torch.no_grad():
        for doc in eval_docs:
            b = encode([doc], A.seq_len)
            if not b:
                continue
            ids = b[0].to(A.device)
            input_ids, labels = ids[:-1].unsqueeze(0), ids[1:].unsqueeze(0)
            position_ids = torch.arange(input_ids.shape[1],
                                        device=A.device).unsqueeze(0)
            hidden = embed(input_ids)
            lk = make_layer_kwargs(rotary, hidden, position_ids, A)
            out = run_layer_stack(head, hidden, lk)
            if W is not None:
                out = (out.float() @ W @ W.T).to(out.dtype)
            out = run_layer_stack(cloud_l, out, lk)
            out = run_layer_stack(tail, out, lk)
            logits = lm_head(norm(out))
            losses.append(torch.nn.functional.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]),
                labels.reshape(-1)).item())
    if not losses:
        raise ValueError("no held-out eval blocks: eval docs too short for seq_len")
    return sum(losses) / len(losses)

# eval docs come from AFTER the attack train/val pools: never seen by the
# decoder, never fine-tuned on — generalization, not memorization
eval_docs = docs[500:500 + args.eval_docs]
utility = {"eval_doc_indices": list(range(500, 500 + len(eval_docs))),
           "heldout_loss_no_W": round(heldout_loss(eval_docs), 4)}
if args.w_key and os.path.exists(args.w_key):
    W = torch.load(args.w_key, map_location="cpu").float().to(A.device)
    utility["heldout_loss_with_W"] = round(heldout_loss(eval_docs, W=W), 4)
    utility["w_key"] = args.w_key
else:
    utility["heldout_loss_with_W"] = None
    utility["note"] = "no --w-key found; with-W utility not measured"
print(f"[utility] held-out loss: no_W={utility['heldout_loss_no_W']:.4f} "
      f"with_W={utility['heldout_loss_with_W']}")

# --- captured wire tensors: align decode-step captures to generated tokens
# via the wire_*.json sidecars (NOT filename order) ---
gen_ids = json.load(open(args.tokens_json))
records = []
for jf in sorted(glob.glob(os.path.join(args.capture_dir, "wire_*.json"))):
    pt = jf[:-len(".json")] + ".pt"
    if os.path.exists(pt):
        records.append((json.load(open(jf)), pt))
decode = [(m, p) for m, p in records if m.get("phase") == "decode"]
# deterministic token order: session, then committed sequence position
decode.sort(key=lambda r: (str(r[0].get("session_id")), r[0].get("position", 0)))
xs = []
epochs = []  # [ER] sidecar "epoch" per capture (null when ratchet off)
for meta, pt in decode:
    t = torch.load(pt)
    if t.dim() == 3 and t.shape[1] == 1:  # decode step
        xs.append(t[0, 0])
        epochs.append(meta.get("epoch"))
ys = []
for i, x in enumerate(xs):
    if i < len(gen_ids):
        ys.append(gen_ids[i])
if not xs:
    raise SystemExit(f"no decode-phase captures in {args.capture_dir} — wrong arm or capture dir")
xs = torch.stack(xs)[:len(ys)]; ys = torch.tensor(ys)  # truncate to label count
epochs = epochs[:len(ys)]
print(f"captured decode-step tensors: {xs.shape[0]} (sidecar-aligned, "
      f"{len(decode)} decode captures, {len(gen_ids)} generated tokens)")
top1, top5 = evaluate_decoder(dec, xs, ys, "cuda")
out = {"arm": args.arm, "captured": int(xs.shape[0]), "top1_wire_pct": top1, "top5_wire_pct": top5,
       "alignment": "sidecar (session_id, position); phase=decode only",
       "n_sidecars": len(records),
       "n_generated_tokens": len(gen_ids),
       "alignment_warning": (None if len(xs) == len(gen_ids)
                             else f"capture/token count mismatch: {len(xs)} vs {len(gen_ids)}"),
       "utility": utility,
       "note": "interpretation depends on --arm: base arm expect ~60%%, obf arm expect ~random"}
# [ER] Per-epoch attack: score each epoch's captures on their own
# (within-epoch budget — expect ~0% while epochs are fresh and short), plus a
# pooled control over ALL epochs' captures (each wrapped in a different W_t,
# so a single decoder aligned to plain h / one stale W should sit in the
# label-free band). The same public-pair decoder `dec` scores every epoch:
# retraining per epoch would be bit-identical (same train data, same seed).
if args.per_epoch:
    groups = {}
    for i in range(len(ys)):
        groups.setdefault(epochs[i], []).append(i)
    per_epoch = []
    for ep in sorted(groups, key=lambda e: (e is None, e)):
        idx = groups[ep]
        if len(idx) < 8:  # below the within-epoch attack floor
            continue
        t1, t5 = evaluate_decoder(dec, xs[idx], ys[idx], "cuda")
        per_epoch.append({"epoch": ep, "n_tokens": len(idx),
                          "top1_pct": t1, "top5_pct": t5})
    p1, p5 = evaluate_decoder(dec, xs, ys, "cuda")
    out["per_epoch"] = per_epoch
    out["pooled_cross_epoch"] = {"top1_pct": p1, "top5_pct": p5}
json.dump(out, open(args.output, "w"), indent=2)
print(json.dumps(out))
