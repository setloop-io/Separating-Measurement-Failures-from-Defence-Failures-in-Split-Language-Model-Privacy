#!/usr/bin/env python3
"""Diagnose the utility-eval anomaly: per-doc held-out loss before/after short FT."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("/workspace/experiments/models/mistral-7b-instruct")
model = AutoModelForCausalLM.from_pretrained("/workspace/experiments/models/mistral-7b-instruct",
                                             dtype=torch.bfloat16, device_map="cuda")
model.eval()
docs = [l.strip() for l in open("/workspace/experiments/models/wikitext2_corpus.txt") if len(l.strip()) > 500]
ft_docs = docs[-4:]        # "victim" docs used for FT
eval_docs = docs[100:104]  # disjoint held-out docs

def per_doc_loss(dset):
    out = []
    with torch.no_grad():
        for t in dset:
            ids = tok(t[:1500], return_tensors="pt", truncation=True).input_ids.cuda()
            out.append(model(ids, labels=ids).loss.item())
    return out

print("BEFORE FT: eval-docs losses:", [round(x, 3) for x in per_doc_loss(eval_docs)])
print("BEFORE FT: ft-docs losses:  ", [round(x, 3) for x in per_doc_loss(ft_docs)])

model.train()
mods = ([model.model.embed_tokens, model.model.norm, model.lm_head]
        + [model.model.layers[i] for i in (0, 1, 30, 31)])
for p in model.parameters():
    p.requires_grad_(False)
params = [p for m in mods for p in m.parameters()]
for p in params:
    p.requires_grad_(True)
opt = torch.optim.AdamW(params, lr=1e-5)
for step in range(100):
    ids = tok(ft_docs[step % 4][:1500], return_tensors="pt", truncation=True).input_ids.cuda()
    loss = model(ids, labels=ids).loss
    opt.zero_grad(); loss.backward(); opt.step()
model.eval()
print("AFTER 100 steps @1e-5 on ft-docs: eval-docs:", [round(x, 3) for x in per_doc_loss(eval_docs)])
print("AFTER 100 steps @1e-5 on ft-docs: ft-docs:  ", [round(x, 3) for x in per_doc_loss(ft_docs)])
# block-style eval (mimics the run's heldout_loss: first 64 tokens of each doc)
def block_loss(dset):
    out = []
    with torch.no_grad():
        for t in dset:
            ids = tok(t, return_tensors="pt", truncation=True, max_length=65).input_ids.cuda()
            out.append(model(ids[:, :-1], labels=ids[:, 1:]).loss.item())
    return out
print("BLOCK-STYLE eval-docs:", [round(x, 3) for x in block_loss(eval_docs)])
