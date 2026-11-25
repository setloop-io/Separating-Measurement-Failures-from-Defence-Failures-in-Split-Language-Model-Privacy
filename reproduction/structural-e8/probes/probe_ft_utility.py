#!/usr/bin/env python3
"""Utility check for the 'private FT defends in 10 steps' finding (review blocker #4):
does 10-step lr=1e-4 local-stage FT damage the 7B model?"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("/workspace/experiments/models/mistral-7b-instruct")
model = AutoModelForCausalLM.from_pretrained(
    "/workspace/experiments/models/mistral-7b-instruct", dtype=torch.bfloat16, device_map="cuda")
model.train()
docs = [l.strip() for l in open("/workspace/experiments/models/wikitext2_corpus.txt")
        if len(l.strip()) > 500]

def eval_loss():
    model.eval()
    losses = []
    with torch.no_grad():
        for t in docs[-10:]:
            ids = tok(t[:1500], return_tensors="pt", truncation=True).input_ids.cuda()
            losses.append(model(ids, labels=ids).loss.item())
    model.train()
    return sum(losses) / len(losses)

print("loss before FT:", round(eval_loss(), 4))
mods = ([model.model.embed_tokens, model.model.norm, model.lm_head]
        + [model.model.layers[i] for i in (0, 1, 30, 31)])
params = [p for m in mods for p in m.parameters()]
for p in model.parameters():
    p.requires_grad_(False)
for p in params:
    p.requires_grad_(True)
opt = torch.optim.AdamW(params, lr=1e-4)
for step in range(10):
    ids = tok(docs[step][:1500], return_tensors="pt", truncation=True).input_ids.cuda()
    loss = model(ids, labels=ids).loss
    opt.zero_grad(); loss.backward(); opt.step()
print("loss after 10-step FT:", round(eval_loss(), 4))
out = model.generate(tok("The capital of France is", return_tensors="pt").input_ids.cuda(),
                     max_new_tokens=12, do_sample=False)
print("sample:", tok.decode(out[0][-12:]))
