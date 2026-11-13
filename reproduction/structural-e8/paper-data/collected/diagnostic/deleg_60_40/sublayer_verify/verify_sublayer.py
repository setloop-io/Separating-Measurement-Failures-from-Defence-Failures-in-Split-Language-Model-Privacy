#!/usr/bin/env python3
"""Correctness proof for run_sublayer_stack against the real Qwen3-0.6B.

Not a repo file: a one-off verification harness for PR #104.  Every number it
prints is measured on the campaign's own model, corpus and eval window, so the
undefended baseline it recomputes must read 4.977673412300646 exactly.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "split-training"))
sys.path.insert(0, str(ROOT / "bin"))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from split_trainer import make_layer_kwargs, run_layer_stack, run_sublayer_stack
from run_latent_native_v5_06b import TRUSTED_SUBLAYERS, load_blocks

MODEL = "/workspace/experiments/models/qwen3-0.6b"
CORPUS = "/workspace/experiments/models/wikitext2_corpus.txt"
SEQ_LEN, TRAIN_BLOCKS, EVAL_BLOCKS = 32, 256, 256
SPLIT_AFTER, RESUME_AFTER = 14, 26
EXPECTED_BASELINE = 4.977673412300646

ok = True


def check(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' :: ' + detail) if detail else ''}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, local_files_only=True, dtype=dtype).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    for p in model.parameters():
        p.requires_grad_(False)
    core, layers = model.model, model.model.layers
    blocks = load_blocks(tokenizer, Path(CORPUS), SEQ_LEN,
                         TRAIN_BLOCKS + EVAL_BLOCKS)
    eval_blocks = blocks[TRAIN_BLOCKS:]
    print(f"device={device} dtype={dtype} layers={len(layers)} "
          f"eval_blocks={len(eval_blocks)}")
    print(f"TRUSTED_SUBLAYERS={TRUSTED_SUBLAYERS}")

    args = type("Args", (), {"attn_impl": "sdpa"})()
    reference, stacked, decomposed, split_decomposed = [], [], [], []
    worst_logit_delta = 0.0
    worst_whole_delta = 0.0
    worst_middle_delta = 0.0
    skeleton_mlp_gap = skeleton_attn_gap = 0.0
    delegated = layers[SPLIT_AFTER + 1:RESUME_AFTER]

    with torch.no_grad():
        for index, block in enumerate(eval_blocks):
            ids = torch.tensor(block, dtype=torch.long, device=device)[None]
            input_ids, labels = ids[:, :-1], ids[:, 1:]
            positions = torch.arange(input_ids.shape[1], device=device)[None]
            hidden = core.embed_tokens(input_ids)
            kwargs = make_layer_kwargs(getattr(core, "rotary_emb", None),
                                       hidden, positions, args)

            ref_logits = model(input_ids).logits.float()
            reference.append(float(F.cross_entropy(
                ref_logits.flatten(0, 1), labels.flatten()).item()))

            # A: the whole 28-layer stack driven sublayer by sublayer, against
            # both the model's own forward and run_layer_stack on identical
            # kwargs (the like-for-like reference).
            whole_ref = run_layer_stack(layers, hidden, kwargs)
            stacked.append(float(F.cross_entropy(
                model.lm_head(core.norm(whole_ref)).float().flatten(0, 1),
                labels.flatten()).item()))
            whole = run_sublayer_stack(layers, hidden, kwargs, ("attn", "mlp"))
            worst_whole_delta = max(
                worst_whole_delta,
                float((whole.float() - whole_ref.float()).abs().max().item()))
            dec_logits = model.lm_head(core.norm(whole)).float()
            worst_logit_delta = max(
                worst_logit_delta,
                float((dec_logits - ref_logits).abs().max().item()))
            decomposed.append(float(F.cross_entropy(
                dec_logits.flatten(0, 1), labels.flatten()).item()))

            # B: the runner's own prefix / delegated-span / tail composition,
            # with the delegated span run sublayer by sublayer.
            prefix = run_layer_stack(layers[:SPLIT_AFTER + 1], hidden, kwargs)
            middle_ref = run_layer_stack(delegated, prefix, kwargs)
            middle_dec = run_sublayer_stack(delegated, prefix, kwargs,
                                            ("attn", "mlp"))
            worst_middle_delta = max(
                worst_middle_delta,
                float((middle_dec.float() - middle_ref.float()).abs().max().item()))
            tail = run_layer_stack(layers[RESUME_AFTER:], middle_dec, kwargs)
            split_logits = model.lm_head(core.norm(tail)).float()
            split_decomposed.append(float(F.cross_entropy(
                split_logits.flatten(0, 1), labels.flatten()).item()))

            # C: the two trusted skeletons the new modes actually run.  They
            # must be finite and materially different from the true span, or
            # the flag would be a silent no-op.
            if index < 8:
                for mode in ("mlp", "attn"):
                    skel = run_sublayer_stack(delegated, prefix, kwargs,
                                              TRUSTED_SUBLAYERS[mode])
                    if not torch.isfinite(skel).all():
                        raise RuntimeError(f"skeleton {mode} is non-finite")
                    gap = float((skel.float() - middle_ref.float())
                                .norm().item() / middle_ref.float().norm().item())
                    if mode == "mlp":
                        skeleton_mlp_gap = max(skeleton_mlp_gap, gap)
                    else:
                        skeleton_attn_gap = max(skeleton_attn_gap, gap)

    ref_mean = sum(reference) / len(reference)
    stack_mean = sum(stacked) / len(stacked)
    dec_mean = sum(decomposed) / len(decomposed)
    split_mean = sum(split_decomposed) / len(split_decomposed)
    print(f"\nreference model() baseline_eval_loss = {ref_mean!r}")
    print(f"run_layer_stack over 28 layers       = {stack_mean!r}")
    print(f"whole-stack sublayer decomposition   = {dec_mean!r}")
    print(f"prefix/span/tail decomposition       = {split_mean!r}")
    print(f"max |logit| delta vs model()         = {worst_logit_delta}")
    print(f"max |hidden| delta vs run_layer_stack= {worst_whole_delta}")
    print(f"max |hidden| delta (delegated span)  = {worst_middle_delta}")
    print(f"relative gap, attn-only skeleton     = {skeleton_mlp_gap}")
    print(f"relative gap, mlp-only  skeleton     = {skeleton_attn_gap}\n")

    check("reference reproduces the campaign baseline",
          ref_mean == EXPECTED_BASELINE, repr(ref_mean))
    check("whole-stack decomposition is bit-identical to run_layer_stack",
          worst_whole_delta == 0.0)
    check("whole-stack decomposition reproduces the campaign baseline",
          dec_mean == EXPECTED_BASELINE, repr(dec_mean))
    check("delegated-span decomposition is bit-identical to run_layer_stack",
          worst_middle_delta == 0.0)
    check("prefix/span/tail composition reproduces the baseline",
          split_mean == EXPECTED_BASELINE, repr(split_mean))
    check("attn-only skeleton is not the full span",
          skeleton_mlp_gap > 1e-3, f"rel gap {skeleton_mlp_gap:.4f}")
    check("mlp-only skeleton is not the full span",
          skeleton_attn_gap > 1e-3, f"rel gap {skeleton_attn_gap:.4f}")
    check("delegating mlp keeps attention trusted",
          TRUSTED_SUBLAYERS["mlp"] == ("attn",))
    check("delegating attn keeps the mlp trusted",
          TRUSTED_SUBLAYERS["attn"] == ("mlp",))
    print("VERIFY PASSED" if ok else "VERIFY FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
