#!/usr/bin/env python3
"""E-D2 distributed draft-model speculation over the split target path."""
import argparse, hashlib, json, os, platform, socket, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "split-training"))
from draft_spec import PROMPTS, accept_prefix_len, check_tokenizer_compat, round_trip_metrics


def input_ids_tensor(encoded):
    """Normalize tokenizer outputs across Transformers return conventions."""
    if hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    elif isinstance(encoded, dict) and "input_ids" in encoded:
        encoded = encoded["input_ids"]
    shape = getattr(encoded, "shape", None)
    if shape is None or len(shape) != 2:
        raise TypeError("tokenizer output must contain a rank-2 input_ids tensor")
    return encoded


def cache_seq_len(cache):
    try:
        return int(cache.get_seq_length())
    except Exception:
        lengths = []
        for layer in getattr(cache, "layers", []):
            keys = getattr(layer, "keys", None)
            if keys is not None:
                lengths.append(int(keys.shape[-2]))
        return max(lengths, default=0)


def sampled_cache_state_hash(cache, torch, samples_per_tensor=32):
    """Opaque sampled fingerprint; never serializes raw cache state."""
    digest = hashlib.sha256()
    for layer_idx, layer in enumerate(getattr(cache, "layers", [])):
        for attr in ("keys", "values", "conv_states", "recurrent_states"):
            value = getattr(layer, attr, None)
            if value is None or not torch.is_tensor(value):
                continue
            flat = value.detach().reshape(-1)
            digest.update(f"{layer_idx}:{attr}:{tuple(value.shape)}:{value.dtype}".encode())
            if flat.numel():
                count = min(samples_per_tensor, flat.numel())
                indices = torch.linspace(
                    0, flat.numel() - 1, steps=count, device=flat.device
                ).long()
                digest.update(
                    flat.index_select(0, indices).float().cpu().numpy().tobytes()
                )
    return digest.hexdigest()


def logit_comparison(block_row, reference_row, torch):
    block = block_row.detach().float()
    reference = reference_row.detach().float().to(block.device)
    block_top = torch.topk(block, 2)
    reference_top = torch.topk(reference, 2)
    delta = block - reference
    return {
        "block_token_id": int(block_top.indices[0]),
        "reference_token_id": int(reference_top.indices[0]),
        "block_top2_margin": float(block_top.values[0] - block_top.values[1]),
        "reference_top2_margin": float(reference_top.values[0] - reference_top.values[1]),
        "block_vs_single_max_abs_logit_delta": float(delta.abs().max()),
        "block_vs_single_rms_logit_delta": float(delta.square().mean().sqrt()),
    }


def margin_requires_fallback(margins, threshold):
    """Pure trigger used by the conservative greedy-exactness guard."""
    return threshold is not None and bool(margins) and min(margins) <= threshold


def top2_margin(logit_row, torch):
    values = torch.topk(logit_row.detach().float(), 2).values
    return float(values[0] - values[1])


def rebuild_sequential_history(dec, prompt_ids, generated, torch):
    """Recreate target state from the prompt and emitted tokens one step at a time.

    A current-block replay is insufficient after an earlier batched verification
    has taken a numerically different low-precision path.  Start a fresh cloud
    session and local cache so the fallback follows the same trajectory as the
    independent greedy baseline.  The returned forward count includes the fresh
    prompt prefill because it is an additional target round trip.
    """
    if not generated:
        raise ValueError("full-history rebuild requires at least one emitted token")
    dec._cloud_end_session()
    dec._cloud_new_session()
    dec._cloud_is_prompt = True
    cache = dec._new_cache()
    logits = dec._prefill(prompt_ids, cache)[:, -1, :]
    dec._cloud_is_prompt = False
    committed = prompt_ids.shape[1]
    forwards = 1
    for token in generated:
        dec._cloud_crop_to = committed
        logits = dec._forward_block_logits(
            torch.tensor([token], device=dec.device), committed, cache
        )[:, -1, :]
        committed += 1
        forwards += 1
    return cache, logits[0], committed, forwards


def target_generate(dec, prompt_ids, max_tokens, capture_logits=False):
    import torch
    dec._cloud_new_session(); dec._cloud_is_prompt = True
    cache = dec._new_cache(); logits = dec._prefill(prompt_ids, cache)[:, -1, :]
    dec._cloud_is_prompt = False; committed = prompt_ids.shape[1]; out = []; trace = []
    t0 = time.time()
    try:
        for _ in range(max_tokens):
            if capture_logits:
                trace.append(logits[0].detach().float().cpu())
            tok = int(logits.argmax())
            out.append(tok)
            if tok == dec.tokenizer.eos_token_id: break
            dec._cloud_crop_to = committed
            logits = dec._forward_block_logits(torch.tensor([tok], device=dec.device), committed, cache)[:, -1, :]
            committed += 1
    finally: dec._cloud_end_session()
    return out, time.time() - t0, trace


def draft_tokens(model, prefix, k, torch):
    ids = prefix
    proposed = []
    with torch.no_grad():
        result = model(input_ids=ids, use_cache=True)
        cache = result.past_key_values; logits = result.logits[:, -1, :]
        for _ in range(k):
            tok = int(logits.argmax()); proposed.append(tok)
            result = model(input_ids=torch.tensor([[tok]], device=ids.device),
                           past_key_values=cache, use_cache=True)
            cache = result.past_key_values; logits = result.logits[:, -1, :]
    return proposed


def speculate(dec, draft, prompt_ids, k, max_tokens, reference_logits=None,
              reference_tokens=None, diagnostic=False,
              exactness_margin_threshold=None):
    import torch
    dec._diagnostic_state = diagnostic
    dec._cloud_new_session(); dec._cloud_is_prompt = True
    cache = dec._new_cache(); pre = dec._prefill(prompt_ids, cache)[:, -1, :]
    dec._cloud_is_prompt = False
    committed = prompt_ids.shape[1]; cur = int(pre.argmax()); generated = []
    rounds = 0; replays = 0; partial_rounds = 0; accepted_total = 0
    exactness_fallback_rounds = 0; exactness_fallback_forwards = 0
    wire_start = len(dec._step_timings)
    diagnostic_rounds = []
    t0 = time.time()
    try:
        while len(generated) < max_tokens:
            generated.append(cur)
            if cur == dec.tokenizer.eos_token_id or len(generated) >= max_tokens: break
            prefix = torch.cat([prompt_ids, torch.tensor([generated], device=dec.device)], dim=1)
            proposals = draft_tokens(draft, prefix, k, torch)
            lin = dec._checkpoint_linear_states(cache) if dec.hybrid else None
            before = {
                "local_cache_len": cache_seq_len(cache),
                "local_sampled_state_sha256": sampled_cache_state_hash(cache, torch),
                "cloud": dict(dec._last_cloud_diagnostic or {}),
            } if diagnostic else None
            block = torch.tensor([cur] + proposals, device=dec.device)
            dec._cloud_crop_to = committed
            logits = dec._forward_block_logits(block, committed, cache)[0]
            cloud_after_block = dict(dec._last_cloud_diagnostic or {})
            preds = [int(x) for x in logits.argmax(dim=-1)]
            nacc = accept_prefix_len(preds, proposals)
            corr = preds[min(nacc, len(preds)-1)]
            accepted = proposals[:nacc]
            block_nacc = nacc
            block_corr = corr
            decision_margins = [
                top2_margin(row, torch) for row in logits[:nacc + 1]
            ]
            exactness_triggered = margin_requires_fallback(
                decision_margins, exactness_margin_threshold
            )
            round_trace = None
            if diagnostic:
                output_start = len(generated)
                comparisons = []
                for offset, block_row in enumerate(logits):
                    output_index = output_start + offset
                    if reference_logits is None or output_index >= len(reference_logits):
                        break
                    same_context = (
                        reference_tokens is not None
                        and generated == reference_tokens[:len(generated)]
                        and proposals[:offset]
                        == reference_tokens[output_start:output_start + offset]
                    )
                    comparison = {
                        "output_index": output_index,
                        "same_context_as_sequential": same_context,
                        "block_token_id": int(block_row.argmax()),
                        "reference_token_id": int(reference_tokens[output_index]),
                    }
                    if same_context:
                        comparison.update(logit_comparison(
                            block_row, reference_logits[output_index], torch
                        ))
                    comparisons.append(comparison)
                round_trace = {
                    "round": rounds,
                    "committed_before": int(committed),
                    "current_token_id": int(cur),
                    "proposal_token_ids": [int(x) for x in proposals],
                    "target_prediction_token_ids": preds,
                    "block_accepted_prefix_len": int(block_nacc),
                    "block_correction_token_id": int(block_corr),
                    "decision_top2_margins": decision_margins,
                    "exactness_margin_threshold": exactness_margin_threshold,
                    "exactness_fallback_triggered": exactness_triggered,
                    "replay_required": nacc < k,
                    "before_block": before,
                    "after_block": {
                        "local_cache_len": cache_seq_len(cache),
                        "local_sampled_state_sha256": sampled_cache_state_hash(cache, torch),
                        "cloud": cloud_after_block,
                    },
                    "position_comparisons": comparisons,
                }
            fallback_predictions = []
            if exactness_triggered:
                cache, sequential_logits, rebuilt_committed, rebuild_forwards = (
                    rebuild_sequential_history(dec, prompt_ids, generated, torch)
                )
                expected_committed = committed + 1
                if rebuilt_committed != expected_committed:
                    raise AssertionError(
                        "full-history rebuild frontier mismatch: "
                        f"{rebuilt_committed} != {expected_committed}"
                    )
                replays += rebuild_forwards
                exactness_fallback_forwards += rebuild_forwards
                sequential_nacc = 0
                for offset in range(k + 1):
                    sequential_pred = int(sequential_logits.argmax())
                    fallback_predictions.append(sequential_pred)
                    if offset < k and sequential_pred == proposals[offset]:
                        sequential_nacc += 1
                        dec._cloud_crop_to = rebuilt_committed
                        sequential_logits = dec._forward_block_logits(
                            torch.tensor([proposals[offset]], device=dec.device),
                            rebuilt_committed, cache
                        )[0, -1]
                        rebuilt_committed += 1
                        replays += 1
                        exactness_fallback_forwards += 1
                        continue
                    corr = sequential_pred
                    break
                nacc = sequential_nacc
                accepted = proposals[:nacc]
                exactness_fallback_rounds += 1
                if diagnostic:
                    round_trace["exactness_fallback"] = {
                        "sequential_prediction_token_ids": fallback_predictions,
                        "history_rebuild_target_forwards": rebuild_forwards,
                        "additional_target_forwards": (
                            rebuild_forwards + max(0, len(fallback_predictions) - 1)
                        ),
                        "final_accepted_prefix_len": int(nacc),
                        "final_correction_token_id": int(corr),
                        "local_cache_len": cache_seq_len(cache),
                        "local_sampled_state_sha256": sampled_cache_state_hash(cache, torch),
                        "cloud": dict(dec._last_cloud_diagnostic or {}),
                    }
            elif nacc < k:
                dec._crop_cache(cache, committed, lin)
                replay = torch.tensor([cur] + accepted, device=dec.device)
                dec._cloud_crop_to = committed
                dec._forward_block_logits(replay, committed, cache)
                replays += 1
                if diagnostic:
                    round_trace["after_replay"] = {
                        "local_cache_len": cache_seq_len(cache),
                        "local_sampled_state_sha256": sampled_cache_state_hash(cache, torch),
                        "cloud": dict(dec._last_cloud_diagnostic or {}),
                    }
            if nacc < k:
                partial_rounds += 1
            generated.extend(accepted)
            rounds += 1; accepted_total += nacc
            if diagnostic:
                round_trace["accepted_prefix_len"] = int(nacc)
                round_trace["correction_token_id"] = int(corr)
                diagnostic_rounds.append(round_trace)
            committed += 1 + nacc
            cur = corr
    finally:
        dec._cloud_end_session()
        dec._diagnostic_state = False
    generated = generated[:max_tokens]
    elapsed = time.time() - t0
    wire = dec._step_timings[wire_start:]
    metrics = {"k": k, "rounds": rounds, "accepted": accepted_total,
                       **round_trip_metrics(len(generated), rounds, replays),
                       "partial_accept_rate": round(partial_rounds / rounds, 4) if rounds else 0,
                       "exactness_margin_threshold": exactness_margin_threshold,
                       "exactness_fallback_rounds": exactness_fallback_rounds,
                       "exactness_fallback_forwards": exactness_fallback_forwards,
                       "time_s": elapsed, "tokens_per_second": len(generated)/max(elapsed, 1e-9),
                       "wire_calls": len(wire),
                       "mean_wire_total_ms": (sum(x.get("total_ms", 0) for x in wire)/len(wire) if wire else None)}
    if diagnostic:
        metrics["diagnostic_rounds"] = diagnostic_rounds
    return generated, metrics


def run(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import jacobi_server as js
    js.CLOUD_URL = args.cloud
    js.load_model(args.target, "cuda", args.split_after, args.resume_at)
    dec = js.decoder
    draft_tok = AutoTokenizer.from_pretrained(args.draft)
    check_tokenizer_compat(dec.tokenizer, draft_tok)
    draft = AutoModelForCausalLM.from_pretrained(args.draft, dtype=torch.bfloat16,
                                                  device_map="cuda").eval()
    prompts = [(c, p) for c, ps in PROMPTS.items() for p in ps]
    if args.prompt_category:
        prompts = [(c, p) for c, p in prompts if c == args.prompt_category]
    if args.prompt_index is not None:
        if args.prompt_index >= len(prompts):
            raise ValueError("prompt-index is outside the selected prompt set")
        prompts = [prompts[args.prompt_index]]
    if args.diagnostic:
        args.k = [4]
        args.repetitions = 1
        result_max_tokens = min(args.max_tokens, 32)
    else:
        result_max_tokens = args.max_tokens
    if args.quick:
        prompts, args.k, args.max_tokens, args.repetitions = (
            prompts[:2], [4], 16, 1)
        result_max_tokens = args.max_tokens
    def token_hash(tokens):
        return hashlib.sha256(
            ",".join(str(x) for x in tokens).encode("ascii")).hexdigest()
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        commit = "unknown"
    result = {"schema": "dtraining.ed2.distributed_draft.v2",
              "experiment": "E-D2 distributed draft speculation",
              "config": {**vars(args), "measurement_kind": "emulated"},
              "evidence_status": ("diagnostic_only" if args.diagnostic else "supporting"),
              "timings_valid": not args.diagnostic,
              "known_limitations": [
                  "The WAN profile is externally applied; measured wire RTT is retained per run.",
                  "Fallback and optimized kernel paths require separate artifacts and must not be pooled."
              ],
              "provenance": {"dtraining_commit": commit,
                             "hostname": socket.gethostname(),
                             "platform": platform.platform(),
                             "container_image": os.getenv("CONTAINER_IMAGE", "unknown")},
              "runs": []}
    for repetition in range(args.repetitions):
        for category, prompt in prompts:
            ids = input_ids_tensor(dec.tokenizer.apply_chat_template(
                [{"role":"user", "content":prompt}], return_tensors="pt",
                enable_thinking=False, add_generation_prompt=True)).to(dec.device)
            baseline, base_s, reference_logits = target_generate(
                dec, ids, result_max_tokens, capture_logits=args.diagnostic
            )
            for k in args.k:
                tokens, metrics = speculate(
                    dec, draft, ids, k, result_max_tokens,
                    reference_logits=reference_logits,
                    reference_tokens=baseline,
                    diagnostic=args.diagnostic,
                    exactness_margin_threshold=args.exactness_margin_threshold,
                )
                prefix = next((i for i,(a,b) in enumerate(zip(tokens,baseline)) if a != b),
                              min(len(tokens),len(baseline)))
                mismatch = None
                if prefix < min(len(tokens), len(baseline)):
                    mismatch = {
                        "output_index": prefix,
                        "baseline_token_id": int(baseline[prefix]),
                        "draft_path_token_id": int(tokens[prefix]),
                    }
                result["runs"].append({"repetition": repetition,
                                       "category":category, "prompt":prompt, "k":k,
                                       "baseline_time_s":base_s,
                                       "baseline_tokens_per_second": len(baseline)/max(base_s, 1e-9),
                                       "baseline_token_ids_sha256": token_hash(baseline),
                                       "draft_token_ids_sha256": token_hash(tokens),
                                       **metrics,
                                       "identity": tokens == baseline,
                                       "identical_prefix_len": prefix,
                                       "first_mismatch": mismatch})
    result["all_identity"] = all(r["identity"] for r in result["runs"])
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {args.output}")


def main():
    ap=argparse.ArgumentParser(description="E-D2 distributed draft speculation")
    ap.add_argument("--target"); ap.add_argument("--draft")
    ap.add_argument("--cloud"); ap.add_argument("--split-after",type=int,default=1)
    ap.add_argument("--resume-at",type=int); ap.add_argument("--k",type=int,nargs="+",default=[2,4,8])
    ap.add_argument("--max-tokens",type=int,default=80); ap.add_argument("--output",default="ed2_distributed.json")
    ap.add_argument("--repetitions", type=int, default=3)
    ap.add_argument("--wan-profile", default="externally_measured",
                    help="profile label applied outside this process")
    ap.add_argument("--quick",action="store_true"); ap.add_argument("--self-test",action="store_true")
    ap.add_argument("--diagnostic", action="store_true",
                    help="focused correctness trace; emitted timings are invalid")
    ap.add_argument("--prompt-category", choices=sorted(PROMPTS))
    ap.add_argument("--prompt-index", type=int,
                    help="zero-based index after category filtering")
    ap.add_argument("--exactness-margin-threshold", type=float,
                    help="sequentially re-verify decisions at or below this top-two margin")
    args=ap.parse_args()
    if args.self_test:
        class TensorLike:
            shape = (1, 2)
        class BatchLike:
            input_ids = TensorLike()
        assert input_ids_tensor(BatchLike()).shape == (1, 2)
        assert input_ids_tensor({"input_ids": TensorLike()}).shape == (1, 2)
        assert accept_prefix_len([1,2,9], [1,2]) == 2
        m = round_trip_metrics(10, 4, 2)
        assert m["target_forward_calls"] + m["target_replay_forwards"] == 7
        assert m["target_replay_forwards"] == 2
        assert margin_requires_fallback([0.5, 0.03125], 0.0625)
        assert not margin_requires_fallback([0.5, 0.125], 0.0625)
        assert not margin_requires_fallback([], 0.0625)
        assert not margin_requires_fallback([0.0], None)
        try:
            import torch
        except ModuleNotFoundError:
            torch = None
        if torch is not None:
            cmp = logit_comparison(
                torch.tensor([1.0, 3.0, 2.0]),
                torch.tensor([1.0, 2.5, 2.0]), torch
            )
            assert cmp["block_token_id"] == cmp["reference_token_id"] == 1
            assert cmp["block_vs_single_max_abs_logit_delta"] == 0.5
            class Layer:
                keys = torch.tensor([[[[1.0], [2.0]]]])
                values = conv_states = recurrent_states = None
            class Cache:
                layers = [Layer()]
                def get_seq_length(self): return 2
            cache = Cache()
            before = sampled_cache_state_hash(cache, torch)
            cache.layers[0].keys[0, 0, 1, 0] = 3.0
            assert sampled_cache_state_hash(cache, torch) != before
            assert cache_seq_len(cache) == 2
            class RebuildDecoder:
                device = "cpu"
                def __init__(self):
                    self.ended = 0; self.started = 0; self.calls = []
                    self._cloud_is_prompt = False; self._cloud_crop_to = None
                def _cloud_end_session(self): self.ended += 1
                def _cloud_new_session(self): self.started += 1
                def _new_cache(self): return []
                def _prefill(self, prompt, cache):
                    self.calls.append(("prefill", int(prompt.shape[1])))
                    return torch.tensor([[[0.0, 1.0, 0.0]]])
                def _forward_block_logits(self, token, committed, cache):
                    value = int(token.reshape(-1)[0]); cache.append(value)
                    self.calls.append(("token", value, committed))
                    return torch.tensor([[[0.0, float(value), float(value + 1)]]])
            rebuild_dec = RebuildDecoder()
            rebuilt_cache, rebuilt_logits, rebuilt_at, extra = (
                rebuild_sequential_history(
                    rebuild_dec, torch.tensor([[10, 11]]), [1, 2], torch
                )
            )
            assert rebuild_dec.ended == rebuild_dec.started == 1
            assert rebuild_dec.calls == [
                ("prefill", 2), ("token", 1, 2), ("token", 2, 3)
            ]
            assert rebuilt_cache == [1, 2]
            assert rebuilt_at == 4 and extra == 3
            assert int(rebuilt_logits.argmax()) == 2
        print("SELF-TEST PASSED"); return
    missing=[n for n in ("target","draft","cloud","resume_at") if getattr(args,n) is None]
    if missing: ap.error("required arguments: " + ", ".join("--"+x.replace("_","-") for x in missing))
    run(args)
if __name__ == "__main__": main()
