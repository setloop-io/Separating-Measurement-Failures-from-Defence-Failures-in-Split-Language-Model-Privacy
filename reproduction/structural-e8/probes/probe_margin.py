#!/usr/bin/env python3
"""Probe top-2 logit margin at a sequential-vs-lookahead divergence.

Usage: probe_margin.py <model-dir> <split-after> <resume-at> <prompt-key>
prompt-key: code | structured | creative | conversational
"""
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoModelForCausalLM, AutoTokenizer
from jacobi_server import JacobiDecoder, BLOCK_SIZE, MAX_JACOBI_ITERS
import jacobi_server

MODEL = sys.argv[1]
SA = int(sys.argv[2]); RA = int(sys.argv[3])
KEY = sys.argv[4] if len(sys.argv) > 4 else "structured"
CLOUD = "http://10.10.10.2:5000"
PROMPTS = {
    "code": "Write a Python function to implement binary search on a sorted array. Include docstring and type hints.",
    "structured": "List the top 10 largest countries by area with their capitals and populations in a formatted table.",
    "creative": "Write a short story about a robot that discovers it can dream. Make it emotional and surprising.",
    "conversational": "What are the main differences between Python and Rust? When should I choose one over the other?",
}
PROMPT = PROMPTS[KEY]
MAX_TOK = 120
DEVICE = "cuda"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
# [QWEN36] streamed load: local modules to GPU, middle to meta
from transformers import AutoConfig
_cfg = AutoConfig.from_pretrained(MODEL)
_tc = getattr(_cfg, "text_config", None)
_n = (getattr(_cfg, "num_hidden_layers", None)
      or getattr(_tc, "num_hidden_layers", None))
_map = {"model.embed_tokens": 0, "model.rotary_emb": 0,
        "model.norm": 0, "lm_head": 0}
for _i in range(_n):
    _map[f"model.layers.{_i}"] = 0 if (_i <= SA or _i >= RA) else "meta"
try:
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map=_map)
except Exception:
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map="cpu")
model.eval()
num_layers = len(model.model.layers)
model.model.embed_tokens = model.model.embed_tokens.to(DEVICE)
model.model.rotary_emb = model.model.rotary_emb.to(DEVICE)
for i in range(0, SA + 1):
    model.model.layers[i] = model.model.layers[i].to(DEVICE)
for i in range(RA, num_layers):
    model.model.layers[i] = model.model.layers[i].to(DEVICE)
model.model.norm = model.model.norm.to(DEVICE)
model.lm_head = model.lm_head.to(DEVICE)
import gc
for i in range(SA + 1, RA):
    model.model.layers[i] = torch.nn.Identity()
gc.collect(); torch.cuda.empty_cache()
jacobi_server.NUM_LAYERS = num_layers
jacobi_server.SPLIT_AFTER = SA
jacobi_server.RESUME_AT = RA

decoder = JacobiDecoder(model, tokenizer, DEVICE, cloud_url=CLOUD,
                        block_size=BLOCK_SIZE, max_iters=MAX_JACOBI_ITERS)

seq_tokens, seq_logits = [], []
for ev in decoder.generate_stream(PROMPT, MAX_TOK, block_size=1, return_logits=True):
    if 'token_id' in ev:
        seq_tokens.append(ev['token_id'])
        seq_logits.append(ev['logits'])

la_tokens = []
for ev in decoder.generate_stream_lookahead(PROMPT, MAX_TOK, return_logits=True):
    if 'token_id' in ev:
        la_tokens.append(ev['token_id'])

div = next((i for i, (a, b) in enumerate(zip(seq_tokens, la_tokens)) if a != b), None)
print(f"model={os.path.basename(MODEL)} prompt={KEY} divergence at:", div)
if div is not None:
    lg = seq_logits[div].float()
    top = torch.topk(lg, 5)
    for v, idx in zip(top.values.tolist(), top.indices.tolist()):
        mark = ""
        if idx == seq_tokens[div]:
            mark += "  <- SEQ argmax"
        if idx == la_tokens[div]:
            mark += "  <- LA pick"
        print(f"  token {idx:6d}  logit {v:10.6f}{mark}")
    s = torch.sort(lg, descending=True).values
    print(f"top-2 margin (sequential path): {(s[0]-s[1]).item():.6f}")
    print("SEQ token:", seq_tokens[div], repr(tokenizer.decode([seq_tokens[div]])))
    print("LA  token:", la_tokens[div], repr(tokenizer.decode([la_tokens[div]])))
    print("context:", repr(tokenizer.decode(seq_tokens[max(0, div-12):div])))
