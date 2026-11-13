#!/usr/bin/env python3
"""
Speculation-Defense Sweep Experiment

Sweeps boundary activation defense settings (quantization and differential privacy noise)
and measures their impact on speculative decoding acceptance rate (throughput) and token identity.
"""

import argparse
import json
import os
import sys
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "split-training"))

from transformers import AutoModelForCausalLM, AutoTokenizer
from jacobi_server import JacobiDecoder
from trained_inversion import make_provenance

MODEL_PATH = "models/mistral-7b-instruct"
DEVICE = "cuda"
MAX_NEW_TOKENS = 100

PROMPTS = {
    "code": "Write a Python function to implement binary search on a sorted array. Include docstring and type hints.",
    "structured": "List the top 10 largest countries by area with their capitals and populations in a formatted table.",
    "creative": "Write a short story about a robot that discovers it can dream. Make it emotional and surprising.",
    "conversational": "What are the main differences between Python and Rust? When should I choose one over the other?",
}

def run_experiment(decoder, prompt, quant, sigma, max_tokens=100):
    decoder.quantize_boundary = quant
    decoder.boundary_noise_sigma = sigma

    # Collect lookahead tokens and count steps
    decoder._step_timings = []
    tokens = []
    steps = 0
    accept_lens = []

    # Intercept lookahead metrics
    orig_generate = decoder.generate_stream_lookahead
    
    # We run lookahead decoding
    events = list(decoder.generate_stream_lookahead(prompt, max_tokens))
    
    # Extract stats from done event
    done_event = next((e for e in events if 'done' in e), {})
    jacobi_stats = done_event.get("jacobi_stats", {})
    tokens_per_block = jacobi_stats.get("tokens_per_block", [])
    avg_acceptance = sum(tokens_per_block) / len(tokens_per_block) if tokens_per_block else 1.0

    # Also run sequential arm as a baseline to verify token identity
    decoder.quantize_boundary = 'none'
    decoder.boundary_noise_sigma = 0.0
    seq_tokens = []
    for event in decoder.generate_stream(prompt, max_tokens, block_size=1):
        if 'token_id' in event:
            seq_tokens.append(event['token_id'])

    la_tokens = [e['token_id'] for e in events if 'token_id' in e]
    identity = seq_tokens == la_tokens

    return {
        "acceptance_rate": round(avg_acceptance, 3),
        "tokens_identical": identity,
        "num_steps": len(tokens_per_block),
        "num_tokens": len(la_tokens),
        "tokens_per_second": done_event.get("tokens_per_second", 0),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--cloud", default=None, help="Cloud server URL")
    parser.add_argument("--split-after", type=int, default=1)
    parser.add_argument("--resume-at", type=int, default=None)
    parser.add_argument("--output", default="experiment_data/speculation_defense_sweep.json")
    args = parser.parse_args()

    print(f"Loading tokenizer and model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Load model configuration to find layers count
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(args.model)
    num_layers = getattr(cfg, "num_hidden_layers", 32)
    resume_at = args.resume_at if args.resume_at is not None else num_layers - 2

    # Streaming load map
    _map = {"model.embed_tokens": 0, "model.rotary_emb": 0, "model.norm": 0, "lm_head": 0}
    for i in range(num_layers):
        _map[f"model.layers.{i}"] = 0 if (i <= args.split_after or i >= resume_at) else "meta"

    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16, device_map=_map)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16, device_map="cpu")
    model.eval()

    # Stub middle layers for client side execution
    for i in range(args.split_after + 1, resume_at):
        model.model.layers[i] = torch.nn.Identity()

    import jacobi_server
    jacobi_server.NUM_LAYERS = num_layers
    jacobi_server.SPLIT_AFTER = args.split_after
    jacobi_server.RESUME_AT = resume_at

    decoder = JacobiDecoder(model, tokenizer, DEVICE, cloud_url=args.cloud)
    
    if decoder.cloud_url:
        decoder._cloud_new_session()

    quant_options = ["none", "fp8", "int8"]
    noise_options = [0.0, 0.01, 0.05, 0.1, 0.2]

    results = []

    for quant in quant_options:
        for sigma in noise_options:
            print(f"--- Sweeping Quantization={quant}, Noise Sigma={sigma} ---")
            run_results = {}
            for cat, prompt in PROMPTS.items():
                print(f"  Category: {cat}...")
                run_results[cat] = run_experiment(decoder, prompt, quant, sigma, MAX_NEW_TOKENS)
                print(f"    Acceptance Rate: {run_results[cat]['acceptance_rate']}, Identity: {run_results[cat]['tokens_identical']}")
            
            results.append({
                "quantization": quant,
                "noise_sigma": sigma,
                "metrics": run_results
            })

    if decoder.cloud_url:
        decoder._cloud_end_session()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out = {
        "schema": "dtraining.speculation_defense.v2",
        "experiment": "speculation defense sweep",
        "config": {**vars(args), "measurement_kind": "measured"},
        "evidence_status": "supporting",
        "known_limitations": [
            "The sweep changes boundary perturbation and may change model semantics; identity is therefore a required endpoint.",
            "Prompt categories are fixed and are not independent corpus samples."
        ],
        "provenance": make_provenance(None, "embedded_prompts", len(PROMPTS), [],
                                      model_path=args.model),
        "results": results,
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved results to {args.output}")

if __name__ == "__main__":
    main()
