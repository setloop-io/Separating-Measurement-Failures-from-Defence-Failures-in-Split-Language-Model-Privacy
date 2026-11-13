#!/usr/bin/env python3
"""Utility sweep for private-FT defense (review #4 follow-up):
held-out loss of the 7B after local-stage FT at lr x steps grid."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("/workspace/experiments/models/mistral-7b-instruct")
docs = [l.strip() for l in open("/workspace/experiments/models/wikitext2_corpus.txt")
        if len(l.strip()) > 500]

def utility_after(lr, steps):
    model = AutoModelForCausalLM.from_pretrained(
        "/workspace/experiments/models/mistral-7b-instruct",
        dtype=torch.bfloat16, device_map="cuda")
    model.train()
    def eval_loss():
        model.eval()
        losses = []
        with torch.no_grad():
            for t in docs[-10:]:
                ids = tok(t[:1500], return_tensors="pt", truncation=True).input_ids.cuda()
                losses.append(model(ids, labels=ids).loss.item())
        model.train()
        return sum(losses) / len(losses)
    base = eval_loss()
    mods = ([model.model.embed_tokens, model.model.norm, model.lm_head]
            + [model.model.layers[i] for i in (0, 1, 30, 31)])
    for p in model.parameters():
        p.requires_grad_(False)
    params = [p for m in mods for p in m.parameters()]
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=lr)
    for step in range(steps):
        ids = tok(docs[step % 40][:1500], return_tensors="pt", truncation=True).input_ids.cuda()
        loss = model(ids, labels=ids).loss
        opt.zero_grad(); loss.backward(); opt.step()
    after = eval_loss()
    out = model.generate(tok("The capital of France is", return_tensors="pt").input_ids.cuda(),
                         max_new_tokens=10, do_sample=False)
    print(f"lr={lr} steps={steps}: loss {base:.3f} -> {after:.3f} | sample: {tok.decode(out[0][-10:])[:60]}")
    del model
    torch.cuda.empty_cache()

for lr, steps in [(1e-5, 10), (1e-5, 100), (1e-6, 100), (1e-6, 1000)]:
    utility_after(lr, steps)
