#!/usr/bin/env python3
"""
Debug harness: run lookahead decoding in lockstep with a fresh-cache
sequential reference and find the FIRST step where lookahead state
(frontier logits or KV cache) diverges from ground truth.

Method: after every lookahead step, we know committed_len and the emitted
token ids. We recompute reference logits at the committed frontier with an
independent sequential forward over (prompt + emitted ids) using a FRESH
cache, and compare against the lookahead engine's internal last_logits.

Local-only mode (no cloud) so the bug can be isolated from the wire
protocol; pass --cloud to run the same check in split mode.

Usage:
    python3 debug_lookahead_lockstep.py --model <path> [--cloud URL]
"""

import torch
import argparse
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PROMPT = ("List the top 10 largest countries by area with their capitals "
          "and populations in a formatted table.")


def reference_logits(decoder, ids):
    """Greedy-reference: full forward over ids with a fresh cache.
    Returns logits [1, len, vocab] for all positions."""
    input_ids = torch.tensor(ids, device=decoder.device).unsqueeze(0)
    cache = decoder._new_cache()
    seq_len = input_ids.shape[1]
    cache_position = torch.arange(seq_len, device=decoder.device)
    position_ids = cache_position.unsqueeze(0)
    with torch.no_grad():
        hidden = decoder.model.model.embed_tokens(input_ids)
        cos, sin = decoder.model.model.rotary_emb(hidden, position_ids)
        hidden = decoder._forward_layers(
            hidden, (cos, sin), cache, cache_position, position_ids
        )
        hidden = decoder.model.model.norm(hidden)
        return decoder.model.lm_head(hidden)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cloud", default=None)
    ap.add_argument("--max-tokens", type=int, default=120)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from jacobi_server import JacobiDecoder
    import jacobi_server

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map="cuda")
    model.eval()
    jacobi_server.NUM_LAYERS = len(model.model.layers)
    jacobi_server.SPLIT_AFTER = 1
    jacobi_server.RESUME_AT = jacobi_server.NUM_LAYERS - 2
    if args.cloud is None:
        # local-only: _forward_all_layers uses NUM_LAYERS
        pass
    decoder = JacobiDecoder(model, tokenizer, "cuda", cloud_url=args.cloud)

    # Monkeypatch generate_stream_lookahead? No - instead re-implement the
    # driving loop minimally by wrapping the generator and, after each
    # yielded token batch, checking state. But last_logits is internal.
    # Simplest: patch the method's internals via source instrumentation is
    # invasive; instead we re-run the exact algorithm here with checks.

    from collections import defaultdict
    N, G = 5, 5
    messages = [{"role": "user", "content": PROMPT}]
    input_ids = tokenizer.apply_chat_template(
        messages, return_tensors="pt", return_dict=False).to(decoder.device)
    prompt_ids = input_ids[0].tolist()

    pool = defaultdict(list)
    for i in range(len(prompt_ids) - N + 1):
        key = prompt_ids[i]
        ngram = tuple(prompt_ids[i + 1:i + N])
        if ngram not in pool[key]:
            pool[key].append(ngram)

    if decoder.cloud_url:
        decoder._cloud_new_session()
    cache = decoder._new_cache()
    decoder._cloud_is_prompt = True
    decoder._cloud_crop_to = None
    decoder._cloud_attn_mask = None
    decoder._cloud_relocate = None
    prefill_logits = decoder._prefill(input_ids, cache)
    decoder._cloud_is_prompt = False
    committed_len = input_ids.shape[1]

    last_logits = prefill_logits[:, -1, :]
    tokens_remaining = args.max_tokens
    all_generated_ids = []
    step = 0

    def check_state(tag):
        """Compare engine frontier logits + full KV cache against reference."""
        ids = prompt_ids + all_generated_ids
        ref = reference_logits(decoder, ids)
        # frontier logits: ref at last position predicts next token
        ref_frontier = ref[:, -1, :].float()
        eng = last_logits.float().reshape(ref_frontier.shape)
        diff = (eng - ref_frontier).abs().max().item()
        agree = torch.argmax(eng, -1).item() == torch.argmax(ref_frontier, -1).item()
        # cache length check
        eng_len = cache.get_seq_length()
        print(f"    [check {tag}] step={step} committed={committed_len} "
              f"cache_len={eng_len} (expect {len(ids)}) "
              f"frontier_logit_maxdiff={diff:.4f} argmax_agree={agree}")
        return diff, agree, eng_len == len(ids)

    while tokens_remaining > 0:
        first_token_id = torch.argmax(last_logits, dim=-1).item()
        if first_token_id == tokenizer.eos_token_id:
            break
        candidates = pool.get(first_token_id, [])[:G]
        step += 1

        if candidates and tokens_remaining > 1:
            combined_ids = [first_token_id]
            for cand in candidates:
                combined_ids.extend(cand)
            total_tokens = len(combined_ids)
            pos_ids = [committed_len]
            for cand in candidates:
                pos_ids.extend(range(committed_len + 1, committed_len + 1 + len(cand)))
            cache_pos = list(range(committed_len, committed_len + total_tokens))
            kv_total = committed_len + total_tokens
            mask = torch.full((1, 1, total_tokens, kv_total), float('-inf'),
                              device=decoder.device, dtype=torch.float16)
            mask[:, :, :, :committed_len] = 0.0
            mask[:, :, 0, committed_len] = 0.0
            offset = 1
            for cand in candidates:
                cand_len = len(cand)
                for j in range(cand_len):
                    row = offset + j
                    mask[:, :, row, committed_len] = 0.0
                    for prev_j in range(j + 1):
                        mask[:, :, row, committed_len + offset + prev_j] = 0.0
                offset += cand_len

            combined_tensor = torch.tensor(combined_ids, device=decoder.device).unsqueeze(0)
            cache_position = torch.tensor(cache_pos, device=decoder.device)
            position_ids = torch.tensor(pos_ids, device=decoder.device).unsqueeze(0)
            decoder._cloud_crop_to = committed_len
            decoder._cloud_attn_mask = mask
            with torch.no_grad():
                hidden = decoder.model.model.embed_tokens(combined_tensor)
                cos, sin = decoder.model.model.rotary_emb(hidden, position_ids)
                hidden = decoder._forward_layers(
                    hidden, (cos, sin), cache, cache_position, position_ids, mask)
                hidden = decoder.model.model.norm(hidden)
                logits = decoder.model.lm_head(hidden)

            # SANITY: verify the speculative pass itself against reference.
            # Row 0 logits should equal reference logits at position
            # committed_len-1 over ids = prompt+gen+[first_token].
            ids_so_far = prompt_ids + all_generated_ids + [first_token_id]
            ref = reference_logits(decoder, ids_so_far)
            r0 = (logits[0, 0].float() - ref[0, -1].float()).abs().max().item()
            print(f"  [step {step}] C={committed_len} T={total_tokens} "
                  f"spec-pass row0 vs ref maxdiff={r0:.4f}")

            first_guess = torch.argmax(logits[0, 0]).item()
            best_hit, best_cand_idx, best_last_logits_row = 0, -1, 0
            offset = 1
            for ci, cand in enumerate(candidates):
                cand_len = len(cand)
                hit = 0
                if cand[0] == first_guess:
                    hit = 1
                    for j in range(1, cand_len):
                        pred = torch.argmax(logits[0, offset + j - 1]).item()
                        if cand[j] == pred:
                            hit = j + 1
                        else:
                            break
                if hit > best_hit:
                    best_hit = hit
                    best_cand_idx = ci
                    best_last_logits_row = offset + hit - 1
                offset += cand_len

            if best_hit > 0:
                accepted_count = best_hit + 1
                src_cache = committed_len + 1
                for ci in range(best_cand_idx):
                    src_cache += len(candidates[ci])
                dst_cache = committed_len + 1
                if src_cache != dst_cache:
                    decoder._relocate_local_cache(cache, src_cache, dst_cache, best_hit)
                    decoder._cloud_relocate = {"src": src_cache, "dst": dst_cache,
                                               "len": best_hit}
                cache.crop(committed_len + accepted_count)
                last_logits = logits[:, best_last_logits_row:best_last_logits_row + 1, :]
                emit_ids = [first_token_id] + list(candidates[best_cand_idx][:best_hit])
                print(f"    ACCEPT cand={best_cand_idx} hit={best_hit} "
                      f"src={src_cache} dst={dst_cache} emit={emit_ids}")
                for tid in emit_ids:
                    if tid == tokenizer.eos_token_id:
                        tokens_remaining = 0
                        break
                    all_generated_ids.append(tid)
                committed_len += accepted_count
                tokens_remaining -= accepted_count
            else:
                cache.crop(committed_len + 1)
                last_logits = logits[:, 0:1, :]
                all_generated_ids.append(first_token_id)
                committed_len += 1
                tokens_remaining -= 1
        else:
            decoder._cloud_crop_to = committed_len
            decoder._cloud_attn_mask = None
            block_id = torch.tensor([first_token_id], device=decoder.device)
            cache_position = torch.tensor([committed_len], device=decoder.device)
            position_ids = cache_position.unsqueeze(0)
            with torch.no_grad():
                hidden = decoder.model.model.embed_tokens(block_id.unsqueeze(0))
                cos, sin = decoder.model.model.rotary_emb(hidden, position_ids)
                hidden = decoder._forward_layers(
                    hidden, (cos, sin), cache, cache_position, position_ids, None)
                hidden = decoder.model.model.norm(hidden)
                logits = decoder.model.lm_head(hidden)
            last_logits = logits[:, -1:, :]
            all_generated_ids.append(first_token_id)
            committed_len += 1
            tokens_remaining -= 1

        # pool growth (same as engine)
        all_ids = prompt_ids + all_generated_ids
        if len(all_ids) >= N:
            key = all_ids[-N]
            ngram = tuple(all_ids[-N + 1:])
            if ngram not in pool[key]:
                pool[key].append(ngram)

        diff, agree, len_ok = check_state("post-step")
        if not agree or not len_ok or diff > 1.0:
            print(f"  !!! DIVERGENCE DETECTED at step {step}")
            ids = prompt_ids + all_generated_ids
            print("  text so far:", tokenizer.decode(all_generated_ids)[-200:])
            break

    if decoder.cloud_url:
        decoder._cloud_end_session()

    # Phase 2: engine sequential arm (generate_stream, block_size=1) vs the
    # reference-greedy ids collected by the lockstep above.
    ref_ids = list(all_generated_ids)
    seq_ids = []
    for event in decoder.generate_stream(PROMPT, args.max_tokens, block_size=1):
        if 'token_id' in event:
            seq_ids.append(event['token_id'])
        elif 'done' in event:
            break
    n = min(len(ref_ids), len(seq_ids))
    first_diff = next((i for i in range(n) if ref_ids[i] != seq_ids[i]), None)
    print(f"\n[phase2] reference_greedy={len(ref_ids)} tokens, "
          f"engine_sequential={len(seq_ids)} tokens")
    if first_diff is None:
        print(f"[phase2] identical over common prefix ({n} tokens); "
              f"len diff = {len(seq_ids) - len(ref_ids)}")
    else:
        print(f"[phase2] FIRST DIFF at position {first_diff}: "
              f"ref={ref_ids[first_diff]} ({tokenizer.decode([ref_ids[first_diff]])!r}) "
              f"seq={seq_ids[first_diff]} ({tokenizer.decode([seq_ids[first_diff]])!r})")
        lo = max(0, first_diff - 5)
        print("  ref ctx:", tokenizer.decode(ref_ids[lo:first_diff + 3]))
        print("  seq ctx:", tokenizer.decode(seq_ids[lo:first_diff + 3]))

        # Phase 3: logit margin at the flip position (tie or corruption?)
        ids = prompt_ids + ref_ids[:first_diff]
        ref_row = reference_logits(decoder, ids)[0, -1].float()
        top = torch.topk(ref_row, 4)
        print("[phase3] reference top-4 at flip position:")
        for v, i in zip(top.values.tolist(), top.indices.tolist()):
            print(f"    {i:7d} {v:10.4f}  {tokenizer.decode([i])!r}")
        print(f"[phase3] top1-top2 margin: "
              f"{(top.values[0] - top.values[1]).item():.4f}; "
              f"ref pick={ref_ids[first_diff]}, seq pick={seq_ids[first_diff]}")


if __name__ == "__main__":
    main()
