#!/usr/bin/env python3
"""E-R7 train eval: does per-epoch rotation defeat the wire attacker on
TRAINING captures? Clone of probes/e9_eval.py's structure, with the E-R1a
attack machinery (solve_w / recovery_with_what from
split-training/rotation_lifetime.py) applied per epoch.

Attacker model (known-plaintext / self-labeled, the strongest standard one):
the attacker knows the public base weights, the training corpus, and the run
config (data order is seed-deterministic), so it can REPLAY the canonical
fine-tune locally and pair every captured wire row with its canonical
counterpart. Per epoch t it gathers the within-epoch (h, h @ W_t) pairs,
solves polar(lstsq) for W_hat_t, and scores victim-token recovery. Reported:

  - per-epoch top-1 (within-epoch budget; expect ~= label-free band while
    epochs are fresh and short);
  - the label-free band itself (decoder on raw rotated rows, no solve);
  - the cross-epoch pooling control (pairs pooled across epochs, solved as
    if one W — MUST sit in the band, proving accumulation does not cross
    rotation boundaries);
  - fwd-only vs fwd+bwd solves (does the backward direction — wire grads,
    which are dL/d(rotated cloud output) = grad_canonical @ W_t — give the
    accumulation attack extra leverage?);
  - an undefended reference (decoder on canonical replay rows) so a broken
    eval cannot masquerade as a defense.

Wire-capture alignment: ER_CAPTURE_DIR sidecars carry
{session_id, mb_id, phase, step, epoch} (er_ratchet.SIDECAR_KEYS); rows are
aligned to the replay by (step, mb_id), NOT filename order.

Usage:
    python er_train_eval.py --capture-dir <dir> --run-json <trainer output> \
        --model <path> --output <json> [--corpus-file <txt>] [--toy]
"""
import argparse, glob, json, os, random, sys
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "split-training"))
sys.path.insert(0, "/workspace/experiments/split-training")  # remote convention
sys.path.insert(0, os.path.expanduser("~/dtraining/split-training"))  # si mount

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=None,
                help="HF model path (default: from --run-json config)")
ap.add_argument("--toy", action="store_true",
                help="the run used split_trainer.py --toy (CPU machinery check)")
ap.add_argument("--capture-dir", required=True)
ap.add_argument("--run-json", required=True,
                help="split_trainer.py output JSON of the captured run "
                     "(config block drives the canonical replay)")
ap.add_argument("--corpus-file", default=None,
                help="MUST be the same corpus the run trained on")
ap.add_argument("--output", required=True)
ap.add_argument("--arm", default="unknown",
                help="which arm these captures came from (recorded)")
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--max-pairs", type=int, default=20000,
                help="public decoder-training pairs (collect_base_pairs cap)")
ap.add_argument("--decode-epochs", type=int, default=30)
ap.add_argument("--batch-size", type=int, default=256)
ap.add_argument("--solve-rows", type=int, default=4096,
                help="max within-epoch pairs fed to the lstsq solve")
ap.add_argument("--pooled-solve-rows", type=int, default=None,
                help="total pairs used by the cross-epoch pooled control "
                     "(default: --solve-rows)")
ap.add_argument("--victim-rows", type=int, default=2048,
                help="max per-epoch victim rows scored for top-1")
ap.add_argument("--min-epoch-rows", type=int, default=16,
                help="epochs with fewer captured rows are skipped (attack floor)")
args = ap.parse_args()

from split_trainer import (TEXT_SAMPLES, build_modules, make_layer_kwargs,
                           run_layer_stack)
from trained_inversion import (collect_base_pairs, evaluate_decoder, seed_all,
                               train_decoder)
from rotation_lifetime import solve_w, recovery_with_what

cfg = json.load(open(args.run_json))["config"]
seed = cfg["seed"]
seq_len = cfg["seq_len"]
mbs = cfg["micro_batch_size"]
grad_accum = cfg["grad_accum"]
steps = cfg["steps"]
sa, ra = cfg["split_after"], cfg["resume_after"]
lr = cfg["lr"]
model_path = args.model or cfg["model"]

# Seed BEFORE build_modules so --toy builds the same weights the trainer did
# (trainer: random.seed; torch.manual_seed; then build_modules).
random.seed(seed)
torch.manual_seed(seed)


class A:  # minimal args namespace for build_modules/collect/train_decoder
    model = model_path; dtype = cfg["dtype"]; device = args.device
    toy = args.toy; seq_len = seq_len; max_pairs = args.max_pairs
    epochs = args.decode_epochs; batch_size = args.batch_size
    attn_impl = "sdpa"


embed, layers, norm, lm_head, rotary, encode = build_modules(A)
n_layers = len(layers)
head = torch.nn.ModuleList(list(layers[: sa + 1]))
middle = torch.nn.ModuleList(list(layers[sa + 1: ra]))
tail = torch.nn.ModuleList(list(layers[ra:]))
vocab_size = lm_head.weight.shape[0]

# --- corpus: regenerate the trainer's block stream EXACTLY ---
texts = list(TEXT_SAMPLES)
if args.corpus_file:
    with open(args.corpus_file) as f:
        texts.extend(l.strip() for l in f if l.strip())
# Mirror split_trainer.py's hard document partition exactly.  Encoding the
# concatenated train+eval corpus changes token-block boundaries and therefore
# the seeded minibatch stream, silently invalidating known-plaintext replay.
n_eval_docs = max(1, len(texts) // 10)
train_texts = texts[:-n_eval_docs]
eval_texts = texts[-n_eval_docs:]
blocks = encode(train_texts, seq_len)
eval_blocks = encode(eval_texts, seq_len)
print(f"[replay] {len(texts)} docs -> {len(blocks) + len(eval_blocks)} blocks "
      f"(train={len(blocks)}, eval={len(eval_blocks)}); "
      f"sa={sa} ra={ra} steps={steps} ga={grad_accum} seq={seq_len}")

# --- decoder on PUBLIC pairs from the base model (pre-finetune weights),
# the same convention as e9_eval / rotation_lifetime ---
# collect_base_pairs encodes PER DOC, so docs shorter than seq_len+1 tokens
# contribute zero blocks (the trainer's own encode concatenates the corpus
# first and is unaffected). Real-tokenizer corpora of short samples (e.g.
# TEXT_SAMPLES) therefore yield ZERO pairs unless we chunk the joined
# corpus; fail loud if there is still nothing to train on.
pub_docs = [t for t in texts if len(t) > 300]
if not pub_docs or not any(encode([d], seq_len) for d in pub_docs[:20]):
    joined = "\n".join(texts)
    pub_docs = [joined[i:i + 4000] for i in range(0, len(joined), 4000)
                if len(joined[i:i + 4000]) > 100]
n_tr = max(1, int(0.8 * len(pub_docs)))
tr_docs, va_docs = pub_docs[:n_tr], pub_docs[n_tr:] or pub_docs[:1]
if not any(encode([d], seq_len) for d in tr_docs):
    raise SystemExit(
        "[er_train_eval] zero public pairs: every attack-training doc is "
        f"shorter than seq_len+1={seq_len + 1} tokens. Provide a longer "
        "--corpus-file (the decoder needs public text with blocks of at "
        "least seq_len tokens).")
tr_h, _, tr_tok = collect_base_pairs(embed, head, middle, tail, norm,
                                     lm_head, rotary, encode, tr_docs, A,
                                     with_grad=False)
try:
    va_h, _, va_tok = collect_base_pairs(embed, head, middle, tail, norm,
                                         lm_head, rotary, encode, va_docs, A,
                                         with_grad=False)
except ValueError:  # torch.cat of zero blocks: val docs too short
    va_h, va_tok = tr_h, tr_tok
    print("[decoder] val docs yielded no blocks; scoring val on train pairs")
seed_all(0)
dec = train_decoder(tr_h, tr_tok, va_h, va_tok, tr_h.shape[1], vocab_size,
                    A, "er_train_wire")
print(f"[decoder] public-pair decoder best val top1 "
      f"{dec.best_val_top1:.2f}% ({tr_h.shape[0]} train pairs)")

# --- canonical replay of the captured run: full-model fine-tune with the
# same data order, capturing the boundary tensors the wire carried ---
# fwd wire row (step, mb)[i]  = leaf_in[i] @ W_t     (boundary activation)
# bwd wire row (step, mb)[i]  = leaf_out.grad[i] @ W_t (dL/d rotated output)
replay = {}  # (step, mb) -> dict(h_in, g_out, tok)
opt = torch.optim.AdamW([p for m in (embed, head, middle, tail, norm, lm_head)
                         for p in m.parameters() if p.requires_grad], lr=lr)
rng = random.Random(seed)
for step in range(steps):
    opt.zero_grad(set_to_none=True)
    for mb in range(grad_accum):
        batch = torch.stack([blocks[rng.randrange(len(blocks))]
                             for _ in range(mbs)]).to(args.device)
        input_ids, labels = batch[:, :-1], batch[:, 1:]
        position_ids = torch.arange(input_ids.shape[1],
                                    device=args.device).unsqueeze(0)
        hidden = embed(input_ids)
        lk = make_layer_kwargs(rotary, hidden, position_ids, A)
        head_out = run_layer_stack(head, hidden, lk)
        leaf_in = head_out.detach().requires_grad_(True)
        mid_out = run_layer_stack(middle, leaf_in, lk)
        leaf_out = mid_out.detach().requires_grad_(True)
        out = run_layer_stack(tail, leaf_out, lk)
        logits = lm_head(norm(out))
        loss = torch.nn.functional.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            labels.reshape(-1)) / grad_accum
        loss.backward()
        replay[(step, mb)] = {
            "h_in": leaf_in.detach().float().cpu(),
            "g_out": leaf_out.grad.float().cpu(),
            "tok": input_ids.cpu(),  # input-side label convention
        }
    opt.step()
    print(f"[replay] step {step}/{steps} loss={loss.item() * grad_accum:.4f}")

# --- captures, aligned by (step, mb_id) via the sidecars ---
records = []
for jf in sorted(glob.glob(os.path.join(args.capture_dir, "cloud_*.json"))):
    pt = jf[:-len(".json")] + ".pt"
    if os.path.exists(pt):
        records.append((json.load(open(jf)), pt))
if not records:
    raise SystemExit(f"no cloud captures in {args.capture_dir}")


def rows_for(phase):
    """{epoch: (wire_rows, can_rows, tok_rows)} aligned to the replay.
    mb_id conventions differ (global counter in-process, per-step ids on
    the wire), so alignment is by RANK within the step: every step yields
    exactly grad_accum captures in mb order under both conventions."""
    out = {}
    phase_records = [(m, p) for m, p in records if m.get("phase") == phase]
    # Held-out evaluation forwards are captured for provenance but have no
    # training step. They must not enter the known-plaintext training replay.
    sel = [(m, p) for m, p in phase_records
           if isinstance(m.get("step"), int) and isinstance(m.get("mb_id"), int)]
    excluded = len(phase_records) - len(sel)
    if excluded:
        print(f"[capture] excluded {excluded} non-training {phase} records "
              "without integer step/mb_id metadata")
    sel.sort(key=lambda r: (r[0]["step"], r[0]["mb_id"]))
    key = "h_in" if phase == "fwd" else "g_out"
    rank = {}
    for meta, pt in sel:
        step = meta["step"]
        mb = rank.get(step, 0)
        rank[step] = mb + 1
        k = (step, mb)
        if k not in replay:
            raise SystemExit(f"capture {pt} has (step, mb_rank)={k} with no "
                             "replay counterpart — wrong --run-json?")
        wire = torch.load(pt).float().cpu()
        can = replay[k][key]
        if wire.shape != can.shape:
            raise SystemExit(f"shape mismatch at {k}: wire {wire.shape} vs "
                             f"replay {can.shape}")
        ep = meta.get("epoch")
        w_rows = wire.reshape(-1, wire.shape[-1])
        c_rows = can.reshape(-1, can.shape[-1])
        t_rows = replay[k]["tok"].reshape(-1)
        w_, c_, t_ = out.get(ep, ([], [], []))
        w_.append(w_rows); c_.append(c_rows); t_.append(t_rows)
        out[ep] = (w_, c_, t_)
    return {ep: (torch.cat(w_), torch.cat(c_), torch.cat(t_))
            for ep, (w_, c_, t_) in out.items()}


fwd = rows_for("fwd")
bwd = rows_for("bwd")
print(f"[capture] {len(records)} wire records; fwd epochs: "
      + ", ".join(f"{e}:{fwd[e][0].shape[0]} rows" for e in sorted(fwd)))

# --- per-epoch attack: within-epoch solve, victim rows held out from it ---
gen0 = torch.Generator().manual_seed(0)
per_epoch = []
victim_pool_w, victim_pool_c, victim_pool_t = [], [], []
for ep in sorted(fwd):
    w, c, t = fwd[ep]
    n = w.shape[0]
    if n < args.min_epoch_rows:
        print(f"[attack] epoch {ep}: only {n} rows, below the floor — skipped")
        continue
    order = torch.randperm(n, generator=gen0)
    n_solve = min(args.solve_rows, n // 2)
    si, vi = order[:n_solve], order[n_solve:n_solve + args.victim_rows]
    w_hat, solver_tag = solve_w(c[si], w[si])
    if w_hat is None:
        raise SystemExit(f"epoch {ep} fwd solve failed: {solver_tag}")
    top1 = recovery_with_what(dec, w[vi], w_hat, t[vi], args.device)
    # fwd+bwd: same solve rows plus the backward-direction pairs
    top1_fb = None
    if ep in bwd:
        bw, bc, _bt = bwd[ep]
        w_hat_fb, solver_tag_fb = solve_w(
            torch.cat([c[si], bc]), torch.cat([w[si], bw]))
        if w_hat_fb is None:
            raise SystemExit(f"epoch {ep} fwd+bwd solve failed: {solver_tag_fb}")
        top1_fb = recovery_with_what(dec, w[vi], w_hat_fb, t[vi], args.device)
    base = evaluate_decoder(dec, c[vi], t[vi], args.device)[0]
    per_epoch.append({"epoch": ep, "n_rows": n, "n_solve": n_solve,
                      "n_victim": int(vi.shape[0]),
                      "solver_tag": solver_tag,
                      "top1_fwd_only_pct": top1,
                      "top1_fwd_bwd_pct": top1_fb,
                      "undefended_top1_pct": base})
    victim_pool_w.append(w[vi]); victim_pool_c.append(c[vi])
    victim_pool_t.append(t[vi])
    print(f"[attack] epoch {ep}: top1 fwd-only={top1:.2f}% "
          f"fwd+bwd={top1_fb if top1_fb is None else f'{top1_fb:.2f}%'} "
          f"(undefended {base:.2f}%, {n_solve} solve rows)")

vw = torch.cat(victim_pool_w); vc = torch.cat(victim_pool_c)
vt = torch.cat(victim_pool_t)

# label-free band: decoder on the raw rotated rows, NO solve (the ceiling
# any failed accumulation attack sits in — e9's obf-arm convention)
band = evaluate_decoder(dec, vw, vt, args.device)[0]
undefended = evaluate_decoder(dec, vc, vt, args.device)[0]

# Cross-epoch pooling control: equal, explicitly disjoint solve/victim shares
# from every epoch, solved as if one W.  Earlier code sampled pooled solve rows
# independently from the victim rows, allowing direct overlap and inflating
# recovery.  Keep the first epoch's victims disjoint from every solve row.
pooled = None
scored = [e for e in sorted(fwd) if fwd[e][0].shape[0] >= args.min_epoch_rows]
if len(scored) >= 2:
    pooled_solve_rows = (args.solve_rows if args.pooled_solve_rows is None
                         else args.pooled_solve_rows)
    if pooled_solve_rows < len(scored):
        raise SystemExit("--pooled-solve-rows must provide at least one row "
                         "per scored epoch")
    shares_w, shares_c = [], []
    pooled_victim = None
    pooled_victim_tokens = None
    for ep in scored:
        w, c, t = fwd[ep]
        n = w.shape[0]
        g = torch.Generator().manual_seed(1000 + ep)
        order = torch.randperm(n, generator=g)
        n_share = min(pooled_solve_rows // len(scored), n // 2)
        solve_idx = order[:n_share]
        shares_w.append(w[solve_idx]); shares_c.append(c[solve_idx])
        if pooled_victim is None:
            victim_idx = order[n_share:n_share + args.victim_rows]
            pooled_victim = w[victim_idx]
            pooled_victim_tokens = t[victim_idx]
    w_pool, pool_solver_tag = solve_w(torch.cat(shares_c), torch.cat(shares_w))
    if w_pool is None:
        raise SystemExit(f"cross-epoch pooled solve failed: {pool_solver_tag}")
    pooled = {"n_epochs": len(scored),
              "rows_per_epoch": int(shares_w[0].shape[0]),
              "total_solve_rows": int(sum(x.shape[0] for x in shares_w)),
              "solve_victim_disjoint": True,
              "solver_tag": pool_solver_tag,
              "top1_pct": recovery_with_what(
                  dec, pooled_victim, w_pool, pooled_victim_tokens,
                  args.device)}
    print(f"[control] pooled {pooled['n_epochs']}-epoch solve: "
          f"{pooled['top1_pct']:.2f}% (band {band:.2f}%)")

out = {"arm": args.arm,
       "run_json": args.run_json, "capture_dir": args.capture_dir,
       "config_echo": {k: cfg[k] for k in
                       ("model", "split_after", "resume_after", "seq_len",
                        "micro_batch_size", "grad_accum", "steps", "seed",
                        "lr", "dtype", "obf")},
       "attacker": "known-plaintext replay (public weights + corpus + "
                   "data order); solve_w=polar(lstsq) per epoch",
       "decoder_public_pairs": int(tr_h.shape[0]),
       "decoder_val_top1": dec.best_val_top1,
       "per_epoch": per_epoch,
       "labelfree_band_top1_pct": band,
       "undefended_top1_pct": undefended,
       "pooled_cross_epoch": pooled,
       "note": "interpretation depends on --arm: rotated arm expect "
               "per-epoch ~= band (<= ~3.7%) and pooled in band; baseline "
               "arm (no rotation) expect per-epoch ~= undefended"}
json.dump(out, open(args.output, "w"), indent=2)
print(json.dumps(out, indent=2))
