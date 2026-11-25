#!/usr/bin/env python3
"""
Dynamic K-Adaptation Evaluation Script

Simulates network RTT jitter/spikes and compares static K vs. dynamic K-adaptation
in terms of throughput stability and average latency.
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
from jacobi_server import JacobiDecoder, SPLIT_AFTER, RESUME_AT, NUM_LAYERS
import jacobi_server
from trained_inversion import make_provenance

PROMPT = "List the top 10 largest countries by area with their capitals and populations in a formatted table."
MAX_NEW_TOKENS = 60

class SimulatedDecoder(JacobiDecoder):
    def __init__(self, model, tokenizer, device, rtt_profile, **kwargs):
        super().__init__(model, tokenizer, device, cloud_url=None, **kwargs)
        self.rtt_profile = rtt_profile  # List of RTT values in ms per step
        self.current_step = 0
        self.split_after = 1
        self.resume_at = NUM_LAYERS - 2

    def _forward_split_layers(self, hidden, position_embeddings, cache,
                              cache_position, position_ids, attn_mask=None):
        # 1. Run local layers (0 to split_after)
        for idx in range(self.split_after + 1):
            out = self.model.model.layers[idx](
                hidden,
                attention_mask=attn_mask,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            hidden = out[0]
            if hidden.dim() == 2:
                hidden = hidden.unsqueeze(0)

        # 2. Simulate RTT delay
        if self.current_step < len(self.rtt_profile):
            rtt = self.rtt_profile[self.current_step]
        else:
            rtt = self.rtt_profile[-1]
        
        self.current_step += 1
        
        # Simulate network sleep
        time.sleep(rtt / 1000.0)

        # 3. Run cloud layers (split_after+1 to resume_at-1)
        for idx in range(self.split_after + 1, self.resume_at):
            out = self.model.model.layers[idx](
                hidden,
                attention_mask=attn_mask,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            hidden = out[0]
            if hidden.dim() == 2:
                hidden = hidden.unsqueeze(0)

        # Record simulated timing
        self._step_timings.append({
            "network_rtt_ms": rtt,
            "total_ms": rtt + 15.0,  # 15ms simulated cloud GPU time
        })

        # 4. Run late layers (resume_at to end)
        for idx in range(self.resume_at, NUM_LAYERS):
            out = self.model.model.layers[idx](
                hidden,
                attention_mask=attn_mask,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            hidden = out[0]
            if hidden.dim() == 2:
                hidden = hidden.unsqueeze(0)

        return hidden

def run_simulation(model, tokenizer, rtt_profile, dynamic_k):
    # Initialize simulated decoder
    decoder = SimulatedDecoder(model, tokenizer, "cuda", rtt_profile)
    decoder.dynamic_k = dynamic_k
    
    start_time = time.time()
    events = list(decoder.generate_stream_lookahead(PROMPT, MAX_NEW_TOKENS))
    total_time = time.time() - start_time
    
    # Extract metrics
    done_event = next((e for e in events if 'done' in e), {})
    tokens = [e['token_id'] for e in events if 'token_id' in e]
    
    # Calculate throughput variance
    steps_timings = [t["total_ms"] for t in decoder._step_timings]
    avg_step_time = sum(steps_timings) / len(steps_timings) if steps_timings else 0
    step_variance = sum((t - avg_step_time)**2 for t in steps_timings) / len(steps_timings) if steps_timings else 0
    
    return {
        "tokens_generated": len(tokens),
        "total_time_s": total_time,
        "throughput_tok_s": len(tokens) / total_time,
        "step_time_mean_ms": avg_step_time,
        "step_time_std_ms": step_variance**0.5,
        "timings": steps_timings
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/qwen3-0.6b")
    parser.add_argument("--output", default="experiment_data/dynamic_k_eval.json")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16, device_map="cuda")
    model.eval()

    num_layers = len(model.model.layers)
    jacobi_server.NUM_LAYERS = num_layers
    jacobi_server.SPLIT_AFTER = 1
    jacobi_server.RESUME_AT = num_layers - 2

    # Define RTT profile simulating network jitter & sudden spikes
    # Base latency ~80ms, spike up to 300ms at steps 15-20, then back down
    rtt_profile = [80] * 15 + [300] * 8 + [80] * 20

    print("--- Running Static K=5 Baseline ---")
    static_results = run_simulation(model, tokenizer, rtt_profile, dynamic_k=False)
    print(f"  Throughput: {static_results['throughput_tok_s']:.2f} tok/s, Std Dev of step time: {static_results['step_time_std_ms']:.1f}ms")

    print("--- Running Dynamic K-Adaptation ---")
    dynamic_results = run_simulation(model, tokenizer, rtt_profile, dynamic_k=True)
    print(f"  Throughput: {dynamic_results['throughput_tok_s']:.2f} tok/s, Std Dev of step time: {dynamic_results['step_time_std_ms']:.1f}ms")

    output_data = {
        "schema": "dtraining.dynamic_k.simulation.v2",
        "experiment": "dynamic K simulated RTT response",
        "config": {**vars(args), "measurement_kind": "simulated",
                   "rtt_profile": rtt_profile},
        "evidence_status": "supporting",
        "known_limitations": [
            "RTT and cloud compute are simulated with sleep and a fixed 15 ms term; this is not a netem or real-WAN measurement.",
            "Use only for controller design and validate selected policies on the distributed harness."
        ],
        "provenance": make_provenance(None, "embedded_prompt", 1, [],
                                      model_path=args.model),
        "static": static_results,
        "dynamic": dynamic_results
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Saved results to {args.output}")

if __name__ == "__main__":
    main()
