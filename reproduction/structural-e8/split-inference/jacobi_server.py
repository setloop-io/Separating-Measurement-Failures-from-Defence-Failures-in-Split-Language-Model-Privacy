#!/usr/bin/env python3
"""
Split Inference - Local Decoding Server

Supports sequential, Jacobi parallel, and lookahead (n-gram speculation)
decoding modes. Model-agnostic: works with any HuggingFace causal LM
(tested with Mistral 7B and LLaMA 2 13B).

Modes:
  --cloud None  → All layers run locally (for testing/baseline)
  --cloud URL   → Early/late layers local, middle layers on cloud GPU
                  Layer split configured via --split-after / --resume-at
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from flask import Flask, request, jsonify, Response
from dataclasses import dataclass, field
import json
import struct
import time
import base64
import hashlib
import requests as http_requests
from websockets.sync.client import connect as ws_connect
import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from privacy_runtime.ratchet_v2 import derive_orthogonal as derive_orthogonal_v2
from privacy_runtime.pair_budget import PairBudget

# === Configuration ===
MODEL_PATH = "models/mistral-7b-instruct"
CLOUD_URL = None  # Set to "http://<cloud-ip>:5000" when A100 is available
DEVICE = "cuda"
BLOCK_SIZE = 16
MAX_JACOBI_ITERS = 32  # Must be >= BLOCK_SIZE for vanilla Jacobi (1 token/iter worst case)
NUM_LAYERS = 32
SPLIT_AFTER = 1   # Local: layers 0-1
RESUME_AT = 30    # Local: layers 30-31


@dataclass
class JacobiMetrics:
    """Tracks performance metrics for a single generation."""
    block_size: int = 16
    prompt_tokens: int = 0
    generated_tokens: int = 0
    total_time_ms: float = 0
    prefill_time_ms: float = 0
    blocks_completed: int = 0
    total_iterations: int = 0
    iterations_per_block: list = field(default_factory=list)
    tokens_per_block: list = field(default_factory=list)
    forward_times_ms: list = field(default_factory=list)

    @property
    def avg_iters_per_block(self):
        return self.total_iterations / max(1, self.blocks_completed)

    @property
    def tokens_per_second(self):
        return self.generated_tokens / max(0.001, self.total_time_ms / 1000)

    @property
    def speedup_vs_sequential(self):
        """generated_tokens / total_iterations = tokens gained per forward pass."""
        return self.generated_tokens / max(1, self.total_iterations)

    def summary(self) -> dict:
        return {
            "tokens_generated": self.generated_tokens,
            "total_time_ms": round(self.total_time_ms, 1),
            "tokens_per_second": round(self.tokens_per_second, 1),
            "prefill_time_ms": round(self.prefill_time_ms, 1),
            "jacobi_stats": {
                "block_size": self.block_size,
                "blocks": self.blocks_completed,
                "total_iterations": self.total_iterations,
                "avg_iters_per_block": round(self.avg_iters_per_block, 2),
                "iterations_per_block": self.iterations_per_block,
                "tokens_per_block": self.tokens_per_block,
                "avg_forward_ms": round(
                    sum(self.forward_times_ms) / max(1, len(self.forward_times_ms)), 1
                ),
                "speedup_vs_sequential": round(self.speedup_vs_sequential, 2),
            }
        }


class JacobiDecoder:
    """Jacobi parallel decoding engine for split inference."""

    def __init__(self, model, tokenizer, device, cloud_url=None,
                 block_size=16, max_iters=32):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.cloud_url = cloud_url
        self.block_size = block_size
        self.max_iters = max_iters
        self.cloud_session_id = None
        self._ws = None  # WebSocket connection
        # State passed to _send_to_cloud during forward passes
        self._cloud_is_prompt = False
        self._cloud_crop_to = None
        self._cloud_attn_mask = None   # Custom mask tensor (for speculation)
        self._cloud_relocate = None    # {"src": int, "dst": int, "len": int}
        self._step_timings = []        # Per-cloud-call timing decomposition
        self._diagnostic_state = False # Request opaque cloud cache fingerprints
        self._last_cloud_diagnostic = None
        # [QWEN36] Hybrid linear-attention models (Qwen3.5/Qwen3.6, gated
        # delta-net + full attention) keep per-layer recurrent + conv state
        # that DynamicCache.crop() cannot roll back. Detect once, gate all
        # hybrid-only behavior on this flag so Mistral/LLaMA are untouched.
        self.hybrid = any(hasattr(l, "linear_attn") for l in model.model.layers)

        # [E8] Secret obfuscation state (None = off). Set via load_obfuscation().
        self.obf_W = None
        self.obf_Wt = None

        # [ER] Ratcheted per-epoch rotation (E-R1b/E-R2/E-R4). Off by default;
        # set via enable_ratchet(). Production v2 uses the portable,
        # domain-separated SHA-256 counter stream in privacy_runtime. Legacy
        # v1 remains available only to reproduce historical evidence. The epoch rides in each cloud
        # request ("obf_epoch") — the two sides do NOT keep independent
        # counters (drift risk); the cloud derives the same W_t on demand.
        self.obf_ratchet_tokens = 0      # fixed mode: served tokens per epoch (0=off)
        self.obf_seed_base = None        # int S for the seed chain (None=off)
        self.obf_budget_events = 0       # budget mode: evidence budget B (0=off)
        self._pair_budget = PairBudget()
        self._obf_epoch_cache = {}       # {epoch: (W, Wt)} — current epoch only
        self._obf_epoch_whash = {}       # {epoch: w_hash} — D2 (issue #88)
        self._obf_seen_prefixes = set()  # first-64-token hashes (prefix channel)
        self._obf_current_epoch = None   # one-shot state for _send_to_cloud
        self.obf_ratchet_version = "v2"
        self.obf_strict_budget = True
        self.obf_count_return_rows = False
        self._privacy_exchange_next = 0

        # [Speculation-Defense Experiment] Quantization and noise fields
        self.quantize_boundary = 'none'       # 'none' | 'fp8' | 'int8'
        self.boundary_noise_sigma = 0.0        # float (noise standard deviation factor)
        self.dynamic_k = False                 # boolean (enable dynamic K speculation window adaptation)

    def load_obfuscation(self, key_path=None, seed=42):
        """[E8] Load or generate the secret orthogonal W for boundary obfuscation.
        If key_path exists, load it; otherwise generate from the seed and save.
        W is fp32 on the model device; Wt = W^T (exact inverse)."""
        import os
        H = self.model.model.embed_tokens.weight.shape[1]
        if key_path and os.path.exists(key_path):
            W = torch.load(key_path, map_location="cpu")
            print(f"[E8] loaded obfuscation key from {key_path}")
        else:
            g = torch.Generator().manual_seed(seed)
            A = torch.randn(H, H, generator=g, dtype=torch.float32)
            Q, R = torch.linalg.qr(A)
            W = Q * torch.sign(torch.diagonal(R))  # orthonormal, det-stable
            if key_path:
                torch.save(W, key_path)
                print(f"[E8] generated and saved obfuscation key to {key_path}")
        err = (W @ W.T - torch.eye(H)).abs().max().item()
        self.obf_W = W.to(self.device)
        self.obf_Wt = W.T.contiguous().to(self.device)
        print(f"[E8] obfuscation active (H={H}, orthogonality err {err:.2e})")

    # --- [ER] ratcheted per-epoch rotation ---
    # NOTE: token accounting is per boundary forward pass (prefill adds the
    # prompt length, each decode pass adds 1), exact for the deployed
    # sequential mode; Jacobi/lookahead retries simply rotate faster.
    # Cross-side consistency never relies on these counters — the epoch is
    # transmitted with every request.

    def enable_ratchet(self, ratchet_tokens=0, seed_base=None, budget_events=0,
                       version="v2", strict_budget=True,
                       count_return_rows=False):
        """[ER] Configure per-epoch rotation. Inert unless seed_base is set
        AND (ratchet_tokens > 0 or budget_events > 0)."""
        self.obf_ratchet_tokens = ratchet_tokens
        self.obf_seed_base = seed_base
        self.obf_budget_events = budget_events
        if version not in ("v1", "v2"):
            raise ValueError("ratchet version must be v1 or v2")
        self.obf_ratchet_version = version
        self.obf_strict_budget = bool(strict_budget)
        self.obf_count_return_rows = bool(count_return_rows)
        self._pair_budget = PairBudget(ratchet_tokens=ratchet_tokens,
                                       budget_events=budget_events,
                                       strict=strict_budget)

    def _ratchet_enabled(self):
        return (self.obf_seed_base is not None
                and (self.obf_ratchet_tokens > 0 or self.obf_budget_events > 0))

    @staticmethod
    def _epoch_seed(seed_base, epoch):
        """[ER] Seed chain (identical on both sides, no key file, no exchange):
        int.from_bytes(sha256(f"{S}:{t}")[:8], "little")."""
        return int.from_bytes(
            hashlib.sha256(f"{seed_base}:{epoch}".encode()).digest()[:8],
            "little")

    def _derive_epoch_W(self, epoch):
        """[ER] Derive W_t with the same QR-sign construction as
        load_obfuscation; cache ONLY the current epoch (evict older ones)."""
        if epoch in self._obf_epoch_cache:
            return self._obf_epoch_cache[epoch]
        H = self.model.model.embed_tokens.weight.shape[1]
        if self.obf_ratchet_version != "v2":
            # D4 (issue #88): v1 derivation removed; see er_ratchet._derive_W.
            raise RuntimeError(
                f"[ER] ratchet version {self.obf_ratchet_version!r} is no "
                "longer supported; only the shared ratchet_v2 derivation "
                "exists")
        W = derive_orthogonal_v2(self.obf_seed_base, epoch, H)
        w_hash = hashlib.sha256(W.numpy().tobytes()).hexdigest()[:16]
        self._obf_epoch_cache = {epoch: (W.to(self.device),
                                         W.T.contiguous().to(self.device))}
        self._obf_epoch_whash.clear()
        self._obf_epoch_whash[epoch] = w_hash
        print(f"[ER] ratchet epoch {epoch} active "
              f"(W sha256={w_hash}, seed=redacted, version={self.obf_ratchet_version})")
        return self._obf_epoch_cache[epoch]

    def _ratchet_advance(self, n_tokens):
        """[ER] Count n served tokens and return the epoch for THIS forward
        pass. Fixed mode: epoch = served // N. Budget mode: every served token
        adds 1 to the evidence counter; the counter advances evidence // B
        epochs and CARRIES the remainder.

        Must stay behaviourally identical to er_ratchet.TokenRatchet.advance
        (the training twin), which carries the same reasoning: zeroing the
        counter on overshoot discards evidence and under-rotates (#89)."""
        previous = self._pair_budget.epoch
        if self.obf_count_return_rows:
            exchange_id = ("inference", self._privacy_exchange_next)
            self._privacy_exchange_next += 1
            epoch = self._pair_budget.reserve_exchange(
                n_tokens, exchange_id, phase="inference",
                directions=("activation_request", "activation_response"))
            self._pair_budget.complete_exchange(exchange_id)
        else:
            epoch = self._pair_budget.advance(n_tokens)
        if epoch != previous and self.obf_budget_events:
            print(f"[ER] budget {self.obf_budget_events} reached; "
                  f"advanced {epoch - previous} to epoch {epoch}")
        return epoch

    OBF_PREFIX_TOKENS = 64  # labelable pairs contributed by one repeated prefix

    def note_request_prefix(self, input_ids):
        """[ER] Budget-mode template/prefix channel: an incoming request whose
        first-64-token hash equals a previously seen request's adds +64
        labeled pairs to the evidence counter (the prefix is known plaintext)."""
        if not (self._ratchet_enabled() and self.obf_budget_events > 0):
            return
        h = hashlib.sha256(str(input_ids[0, :self.OBF_PREFIX_TOKENS].tolist()).encode()).hexdigest()
        if h in self._obf_seen_prefixes:
            self._pair_budget.add_labelable_evidence(self.OBF_PREFIX_TOKENS)
        else:
            self._obf_seen_prefixes.add(h)

    def _apply_wire_defenses(self, hidden):
        # Apply quantization
        if getattr(self, "quantize_boundary", "none") == "int8":
            hidden = self._defense_int8(hidden)
        elif getattr(self, "quantize_boundary", "none") == "fp8":
            hidden = self._defense_fp8(hidden)

        # Apply noise
        if getattr(self, "boundary_noise_sigma", 0.0) > 0.0:
            sigma_frac = self.boundary_noise_sigma
            rms = hidden.float().pow(2).mean(dim=-1, keepdim=True).sqrt()
            noise = torch.randn_like(hidden.float()) * (sigma_frac * rms)
            hidden = (hidden.float() + noise).to(hidden.dtype)

        return hidden

    def _defense_int8(self, x):
        scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
        q = torch.round(x / scale).clamp(-127, 127)
        return (q * scale).to(x.dtype)

    def _defense_fp8(self, x):
        if hasattr(torch, "float8_e4m3fn"):
            return x.float().to(torch.float8_e4m3fn).to(x.dtype)
        xf = x.float()
        max_val = 448.0
        xf = xf.clamp(-max_val, max_val)
        mantissa, exponent = torch.frexp(xf)
        m_q = torch.round(mantissa * 8.0) / 8.0
        overflow = m_q >= 1.0
        m_q = torch.where(overflow, m_q * 0.5, m_q)
        exponent = torch.where(overflow, exponent + 1, exponent)
        out = torch.ldexp(m_q, exponent)
        return out.clamp(-max_val, max_val).to(x.dtype)

    def _adapt_speculation_window(self, current_G, max_G):
        if not getattr(self, "dynamic_k", False) or not self._step_timings:
            return current_G

        # Look at recent network RTTs
        recent_rtts = [t["network_rtt_ms"] for t in self._step_timings[-5:]]
        if len(recent_rtts) < 2:
            return current_G

        # Compute standard deviation or sudden RTT spike
        mean_rtt = sum(recent_rtts) / len(recent_rtts)
        last_rtt = recent_rtts[-1]
        prev_rtt = recent_rtts[-2]

        # Spike detection: if last RTT is > 1.8x the previous RTT, shrink K to minimize packet sizes/count
        if last_rtt > 1.8 * prev_rtt and last_rtt > 20.0:
            new_G = max(1, current_G - 2)
            print(f"    [Dynamic K] RTT spike detected ({prev_rtt:.1f}ms -> {last_rtt:.1f}ms). Shrinking speculation G: {current_G} -> {new_G}")
            return new_G
        
        # If RTT is stable and low, we can gradually grow G back to max_G
        if last_rtt < 1.2 * prev_rtt or last_rtt < 15.0:
            if current_G < max_G:
                new_G = min(max_G, current_G + 1)
                print(f"    [Dynamic K] Network stable ({last_rtt:.1f}ms). Growing speculation G: {current_G} -> {new_G}")
                return new_G

        return current_G

    # === [QWEN36] Hybrid linear-attention cache helpers ===

    def _new_cache(self):
        """Fresh per-prompt cache. Hybrid models need config-aware
        DynamicCache so linear layers get LinearAttentionLayer entries
        (plain DynamicCache() creates DynamicLayer for every layer and
        crashes with IndexError in update_conv_state at the first
        linear-attention layer)."""
        if self.hybrid:
            return DynamicCache(config=self.model.config)
        return DynamicCache()

    def _checkpoint_linear_states(self, cache):
        """Clone conv/recurrent state of every initialized linear-attention
        layer. This is the only rollback mechanism for those layers:
        LinearAttentionCacheLayerMixin.crop() is a deliberate no-op."""
        ckpt = []
        for layer in cache.layers:
            if getattr(layer, "recurrent_states", None) is not None:
                ckpt.append((layer,
                             layer.conv_states.clone(),
                             layer.recurrent_states.clone()))
        return ckpt

    def _restore_linear_states(self, ckpt):
        for layer, conv, rec in ckpt:
            layer.conv_states.copy_(conv)
            layer.recurrent_states.copy_(rec)

    def _crop_cache(self, cache, length, lin_ckpt=None):
        """Roll the whole cache back to `length`: KV crop for full-attention
        layers plus state restore for linear-attention layers (hybrid)."""
        cache.crop(length)
        if lin_ckpt is not None:
            self._restore_linear_states(lin_ckpt)

    def _relocate_local_cache(self, cache, src, dst, length):
        """Copy KV cache entries from src to dst positions, then crop."""
        for layer in cache.layers:
            # [QWEN36] LinearAttentionLayer entries have no .keys/.values
            if getattr(layer, "keys", None) is None:
                continue
            layer.keys[:, :, dst:dst+length, :] = \
                layer.keys[:, :, src:src+length, :].clone()
            layer.values[:, :, dst:dst+length, :] = \
                layer.values[:, :, src:src+length, :].clone()

    def _make_causal_mask(self, committed_len, k):
        """Create causal attention mask for a Jacobi block of k tokens.

        SDPA's is_causal=True creates a RELATIVE mask that breaks when
        there's an existing KV cache. We need an ABSOLUTE mask where
        query i (at position committed_len + i) attends to positions
        0..committed_len+i.
        """
        kv_len = committed_len + k
        mask = torch.full(
            (1, 1, k, kv_len), float('-inf'),
            device=self.device, dtype=torch.float16
        )
        for i in range(k):
            mask[0, 0, i, :committed_len + i + 1] = 0.0
        return mask

    def generate_stream(self, prompt: str, max_new_tokens: int = 500,
                        block_size: int = None, return_logits: bool = False):
        """Generator that yields token dicts as blocks converge.

        Key insight: In a transformer, output at position i predicts token i+1.
        So for a Jacobi block at positions [n, n+1, ..., n+k-1]:
          - Token at position n comes from the PREVIOUS step's logits (already known)
          - Token at position n+j (j>0) comes from logits at position n+j-1
          - Logits at position n+k-1 predict the NEXT block's first token

        Yields:
            {"token": "text"}          for each generated token
            {"done": True, ...stats}   when generation is complete
        """
        if block_size is None:
            block_size = self.block_size
        metrics = JacobiMetrics(block_size=block_size)
        gen_start = time.time()

        # Tokenize
        messages = [{"role": "user", "content": prompt}]
        # [QWEN36] Qwen3.6's chat template opens a <think> block by default;
        # enable_thinking=False makes it emit an empty closed think block, and
        # add_generation_prompt=True appends the assistant header (Mistral's
        # template has [/INST] built in and does not need it).
        template_kwargs = ({"enable_thinking": False, "add_generation_prompt": True}
                           if self.hybrid else {})
        input_ids = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", return_dict=False, **template_kwargs
        ).to(self.device)
        metrics.prompt_tokens = input_ids.shape[1]
        self.note_request_prefix(input_ids)  # [ER] budget-mode prefix channel

        # Create cloud session if in split mode
        if self.cloud_url:
            self._cloud_new_session()

        # Prefill
        cache = self._new_cache()  # [QWEN36] was: DynamicCache()
        prefill_start = time.time()
        self._cloud_is_prompt = True
        self._cloud_crop_to = None
        prefill_logits = self._prefill(input_ids, cache)
        self._cloud_is_prompt = False
        metrics.prefill_time_ms = (time.time() - prefill_start) * 1000
        committed_len = input_ids.shape[1]

        # last_logits: logits from the last committed position, used to
        # determine the first token of the next block
        last_logits = prefill_logits[:, -1, :]  # [1, vocab]
        tokens_remaining = max_new_tokens

        # For correct decoding with SentencePiece, accumulate all token IDs
        # and decode incrementally (single-token decode strips spaces)
        all_generated_ids = []
        prev_text = ""

        # Jacobi block loop
        while tokens_remaining > 0:
            k = min(block_size, tokens_remaining)

            # Position 0 of block is determined by last_logits (already known)
            first_token_id = torch.argmax(last_logits, dim=-1)  # [1]

            # Check EOS at position 0
            if first_token_id.item() == self.tokenizer.eos_token_id:
                tokens_remaining = 0
                break

            # Initialize block: position 0 is known, rest are random guesses
            # Random init converges faster than repeat-last (2-4 tok/iter vs 1)
            block_ids = torch.randint(
                0, self.model.config.vocab_size, (k,), device=self.device
            )
            block_ids[0] = first_token_id  # Position 0 is always known

            block_converged = False
            iters_used = 0

            # In split mode, tell cloud to crop to committed_len before each
            # forward pass. First iteration: no-op. Subsequent: rolls back.
            self._cloud_crop_to = committed_len

            # [QWEN36] Snapshot linear-attention state at the committed
            # frontier; non-converged iterations must restore it (crop()
            # only rolls back full-attention KV). Skipped for k==1
            # (sequential): a single-token block always "converges" on the
            # first iteration, so no rollback is ever needed.
            lin_ckpt = self._checkpoint_linear_states(cache) \
                if (self.hybrid and k > 1) else None

            for iteration in range(self.max_iters):
                iters_used += 1
                fwd_start = time.time()

                # Forward pass for the block
                block_logits = self._forward_block_logits(
                    block_ids, committed_len, cache
                )
                metrics.forward_times_ms.append((time.time() - fwd_start) * 1000)

                # Extract predictions using the SHIFTED mapping:
                #   new[0] = argmax(last_logits)           (from previous step)
                #   new[j] = argmax(block_logits[j-1])     (output at j-1 predicts j)
                new_block_ids = torch.empty_like(block_ids)
                new_block_ids[0] = first_token_id  # Always the same
                if k > 1:
                    new_block_ids[1:] = torch.argmax(block_logits[0, :-1, :], dim=-1)

                # Save logits from last position (predicts next block's first token)
                next_last_logits = block_logits[:, -1, :]

                # Check convergence (fixed point)
                if torch.equal(block_ids, new_block_ids):
                    # Converged! KV cache entries from this forward pass are correct.
                    block_converged = True
                    break

                # Not converged: crop cache back and try again
                self._crop_cache(cache, committed_len, lin_ckpt)  # [QWEN36] was: cache.crop(committed_len)
                block_ids = new_block_ids

            # --- Block finished (converged or max iterations) ---

            # If max iters hit without convergence, commit with a final forward pass
            if not block_converged:
                self._crop_cache(cache, committed_len, lin_ckpt)  # [QWEN36] was: cache.crop(committed_len)
                final_logits = self._forward_block_logits(
                    block_ids, committed_len, cache
                )
                next_last_logits = final_logits[:, -1, :]

            # Save logits for return_logits mode (before last_logits gets updated)
            if return_logits:
                _emit_first_logits = last_logits[0].cpu()
                _emit_source = block_logits if block_converged else final_logits

            # Check for EOS in the final block (only after convergence/commit)
            eos_mask = (block_ids == self.tokenizer.eos_token_id) if not block_converged else \
                       (new_block_ids == self.tokenizer.eos_token_id)
            final_block = new_block_ids if block_converged else block_ids

            if eos_mask.any():
                eos_pos = eos_mask.nonzero(as_tuple=True)[0][0].item()
                # Truncate to EOS (exclusive - don't emit EOS itself)
                k = eos_pos
                if k > 0:
                    # Crop cache to only committed + truncated block
                    if lin_ckpt is not None:
                        # [QWEN36] Linear state absorbed tokens past EOS;
                        # restore the frontier and re-commit the kept prefix.
                        self._crop_cache(cache, committed_len, lin_ckpt)
                        self._cloud_crop_to = committed_len
                        self._commit_block(final_block[:k], committed_len, cache)
                    else:
                        cache.crop(committed_len + k)
                else:
                    self._crop_cache(cache, committed_len, lin_ckpt)  # [QWEN36] was: cache.crop(committed_len)
                committed_len += k
                tokens_remaining = 0
            else:
                committed_len += k
                last_logits = next_last_logits
                tokens_remaining -= k

            # Emit tokens (decode incrementally for correct spacing)
            for j in range(k):
                tid = final_block[j].item()
                all_generated_ids.append(tid)
                full_text = self.tokenizer.decode(all_generated_ids)
                new_text = full_text[len(prev_text):]
                prev_text = full_text
                event = {"token": new_text, "token_id": tid}
                if return_logits:
                    if j == 0:
                        event["logits"] = _emit_first_logits
                    else:
                        event["logits"] = _emit_source[0, j - 1].cpu()
                yield event
                metrics.generated_tokens += 1

            metrics.blocks_completed += 1
            metrics.total_iterations += iters_used
            metrics.iterations_per_block.append(iters_used)
            metrics.tokens_per_block.append(k)

            print(f"  [Jacobi] Block {metrics.blocks_completed}: "
                  f"{'converged' if block_converged else 'max iters'} in {iters_used} iters "
                  f"({k} tokens, "
                  f"{metrics.forward_times_ms[-1]:.0f}ms/fwd)")

        metrics.total_time_ms = (time.time() - gen_start) * 1000
        print(f"  [Jacobi] Done: {metrics.generated_tokens} tokens in "
              f"{metrics.total_time_ms:.0f}ms ({metrics.tokens_per_second:.1f} tok/s), "
              f"speedup vs sequential: {metrics.speedup_vs_sequential:.2f}x")

        # Clean up cloud session
        if self.cloud_url:
            self._cloud_end_session()

        yield {"done": True, **metrics.summary()}

    def generate_stream_lookahead(self, prompt: str, max_new_tokens: int = 500,
                                   ngram_size: int = 5, max_candidates: int = 5,
                                   return_logits: bool = False):
        """Generate with n-gram speculation for multi-token acceptance.

        Maintains an n-gram pool (seeded from prompt, grown from output).
        Each step: look up candidate continuations, verify in a single
        forward pass with isolated attention masks, accept longest match.

        Falls back to sequential (1 token/step) when no candidates match.
        """
        from collections import defaultdict

        N = ngram_size       # n-gram size (key + N-1 continuation tokens)
        G = max_candidates   # max candidates to verify per step
        match_len = N - 1    # length of each candidate continuation

        metrics = JacobiMetrics(block_size=1)
        gen_start = time.time()

        # Tokenize
        messages = [{"role": "user", "content": prompt}]
        # [QWEN36] see generate_stream: suppress think blocks and add the
        # assistant generation header for Qwen3.6
        template_kwargs = ({"enable_thinking": False, "add_generation_prompt": True}
                           if self.hybrid else {})
        input_ids = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", return_dict=False, **template_kwargs
        ).to(self.device)
        prompt_ids = input_ids[0].tolist()
        metrics.prompt_tokens = len(prompt_ids)
        self.note_request_prefix(input_ids)  # [ER] budget-mode prefix channel

        # Build n-gram pool from prompt
        pool = defaultdict(list)
        for i in range(len(prompt_ids) - N + 1):
            key = prompt_ids[i]
            ngram = tuple(prompt_ids[i + 1:i + N])
            if ngram not in pool[key]:
                pool[key].append(ngram)
        print(f"  [Lookahead] Pool seeded with {sum(len(v) for v in pool.values())} "
              f"n-grams from {len(prompt_ids)} prompt tokens")

        # Create cloud session if in split mode
        if self.cloud_url:
            self._cloud_new_session()

        # Prefill
        cache = self._new_cache()  # [QWEN36] was: DynamicCache()
        prefill_start = time.time()
        self._cloud_is_prompt = True
        self._cloud_crop_to = None
        self._cloud_attn_mask = None
        self._cloud_relocate = None
        prefill_logits = self._prefill(input_ids, cache)
        self._cloud_is_prompt = False
        metrics.prefill_time_ms = (time.time() - prefill_start) * 1000
        committed_len = input_ids.shape[1]

        last_logits = prefill_logits[:, -1, :]
        tokens_remaining = max_new_tokens

        all_generated_ids = []
        prev_text = ""
        total_accepted = 0
        total_steps = 0

        while tokens_remaining > 0:
            first_token_id = torch.argmax(last_logits, dim=-1).item()

            if first_token_id == self.tokenizer.eos_token_id:
                break

            # Adapt speculation window G dynamically
            G = self._adapt_speculation_window(G, max_candidates)

            # Look up candidates in n-gram pool
            candidates = pool.get(first_token_id, [])[:G]
            if self.hybrid:
                # [QWEN36] Linear-attention layers process the combined
                # candidate sequence strictly in order: candidate i's
                # tokens see a recurrent state polluted by candidates
                # 0..i-1, so only candidate 0's branch is verifiable.
                # (Full-attention KV of candidate 0 is also the only one
                # computed from unpolluted hidden states.)
                candidates = candidates[:1]

            fwd_start = time.time()
            total_steps += 1

            if candidates and tokens_remaining > 1:
                # === SPECULATIVE VERIFICATION ===
                # Build combined input: [first_token, cand0_tokens..., cand1_tokens...]
                combined_ids = [first_token_id]
                for cand in candidates:
                    combined_ids.extend(cand)

                total_tokens = len(combined_ids)

                # Position IDs: first_token at committed_len,
                # each candidate at committed_len+1..+match_len
                pos_ids = [committed_len]
                for cand in candidates:
                    pos_ids.extend(range(committed_len + 1, committed_len + 1 + len(cand)))

                # Cache positions: unique sequential
                cache_pos = list(range(committed_len, committed_len + total_tokens))

                # Build attention mask
                kv_total = committed_len + total_tokens
                mask = torch.full(
                    (1, 1, total_tokens, kv_total), float('-inf'),
                    device=self.device, dtype=torch.float16
                )
                # All tokens attend to committed cache
                mask[:, :, :, :committed_len] = 0.0
                # first_token attends to self
                mask[:, :, 0, committed_len] = 0.0
                # Each candidate: causal within itself + see first_token
                offset = 1
                for cand in candidates:
                    cand_len = len(cand)
                    for j in range(cand_len):
                        row = offset + j
                        mask[:, :, row, committed_len] = 0.0  # see first_token
                        for prev_j in range(j + 1):
                            mask[:, :, row, committed_len + offset + prev_j] = 0.0
                    offset += cand_len

                # Forward pass
                combined_tensor = torch.tensor(
                    combined_ids, device=self.device
                ).unsqueeze(0)
                cache_position = torch.tensor(cache_pos, device=self.device)
                position_ids = torch.tensor(pos_ids, device=self.device).unsqueeze(0)

                self._cloud_crop_to = committed_len
                self._cloud_attn_mask = mask

                # [QWEN36] Snapshot linear state at the committed frontier
                # before the speculative pass pollutes it.
                lin_ckpt = self._checkpoint_linear_states(cache) \
                    if self.hybrid else None

                with torch.no_grad():
                    hidden = self.model.model.embed_tokens(combined_tensor)
                    cos, sin = self.model.model.rotary_emb(hidden, position_ids)
                    position_embeddings = (cos, sin)

                    hidden = self._forward_layers(
                        hidden, position_embeddings, cache, cache_position,
                        position_ids, mask
                    )

                    hidden = self.model.model.norm(hidden)
                    logits = self.model.lm_head(hidden)

                metrics.forward_times_ms.append((time.time() - fwd_start) * 1000)

                # Verify candidates
                first_guess = torch.argmax(logits[0, 0]).item()

                best_hit = 0
                best_cand_idx = -1
                best_last_logits_row = 0  # row in logits for last accepted token

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

                # Handle results
                if best_hit > 0:
                    accepted_count = best_hit + 1  # first_token + matched tokens

                    # Relocate winning candidate's KV entries
                    src_cache = committed_len + 1
                    for ci in range(best_cand_idx):
                        src_cache += len(candidates[ci])
                    dst_cache = committed_len + 1

                    if src_cache != dst_cache:
                        self._relocate_local_cache(
                            cache, src_cache, dst_cache, best_hit
                        )
                        # Tell cloud to do the same on next request
                        self._cloud_relocate = {
                            "src": src_cache, "dst": dst_cache, "len": best_hit
                        }

                    if lin_ckpt is not None and accepted_count < total_tokens:
                        # [QWEN36] Partial accept: the linear-attention state
                        # absorbed the rejected suffix tokens and cannot be
                        # cropped. Restore the frontier and re-commit exactly
                        # the accepted tokens (KV + linear state consistent).
                        self._crop_cache(cache, committed_len, lin_ckpt)
                        self._cloud_crop_to = committed_len
                        commit_ids = torch.tensor(
                            [first_token_id] + list(candidates[best_cand_idx][:best_hit]),
                            device=self.device
                        )
                        self._commit_block(commit_ids, committed_len, cache)
                    else:
                        # Full accept (or non-hybrid): states/KV are correct
                        # up to committed_len + accepted_count.
                        cache.crop(committed_len + accepted_count)
                    # Build per-token logits for return_logits mode.
                    # NOTE: must capture the PREVIOUS step's last_logits
                    # (the distribution that predicted first_token_id)
                    # BEFORE last_logits is reassigned below — otherwise
                    # _emit_logits[0] holds the logits predicting the token
                    # AFTER the accepted run (off-by-one; inflates the
                    # reported lookahead perplexity).
                    if return_logits:
                        best_offset = 1 + sum(len(candidates[ci]) for ci in range(best_cand_idx))
                        _emit_logits = [last_logits.squeeze().cpu()]  # first_token_id [vocab]
                        _emit_logits.append(logits[0, 0].cpu())  # cand[0]
                        for ej in range(1, best_hit):
                            _emit_logits.append(logits[0, best_offset + ej - 1].cpu())
                    last_logits = logits[:, best_last_logits_row:best_last_logits_row + 1, :]

                    # Emit tokens (never past the requested budget: an
                    # accepted run may overshoot tokens_remaining; the
                    # surplus KV is already committed, which is harmless
                    # because the loop exits right after).
                    emit_ids = [first_token_id] + list(candidates[best_cand_idx][:best_hit])
                    for ei, tid in enumerate(emit_ids):
                        if ei >= tokens_remaining:
                            break
                        if tid == self.tokenizer.eos_token_id:
                            tokens_remaining = 0
                            break
                        all_generated_ids.append(tid)
                        full_text = self.tokenizer.decode(all_generated_ids)
                        new_text = full_text[len(prev_text):]
                        prev_text = full_text
                        event = {"token": new_text, "token_id": tid}
                        if return_logits:
                            event["logits"] = _emit_logits[ei]
                        yield event
                        metrics.generated_tokens += 1

                    committed_len += accepted_count
                    tokens_remaining -= accepted_count
                    total_accepted += accepted_count

                    metrics.blocks_completed += 1
                    metrics.total_iterations += 1
                    metrics.iterations_per_block.append(1)
                    metrics.tokens_per_block.append(accepted_count)

                    print(f"  [Lookahead] Step {total_steps}: "
                          f"accepted {accepted_count} tokens "
                          f"({best_hit} from n-gram match), "
                          f"{metrics.forward_times_ms[-1]:.0f}ms")
                else:
                    # No match, accept just first_token
                    if lin_ckpt is not None:
                        # [QWEN36] Linear state absorbed the rejected
                        # candidate; restore + re-commit first_token only.
                        self._crop_cache(cache, committed_len, lin_ckpt)
                        self._cloud_crop_to = committed_len
                        self._commit_block(
                            torch.tensor([first_token_id], device=self.device),
                            committed_len, cache
                        )
                    else:
                        cache.crop(committed_len + 1)
                    _no_match_logits = last_logits.squeeze().cpu() if return_logits else None
                    last_logits = logits[:, 0:1, :]

                    all_generated_ids.append(first_token_id)
                    full_text = self.tokenizer.decode(all_generated_ids)
                    new_text = full_text[len(prev_text):]
                    prev_text = full_text
                    event = {"token": new_text, "token_id": first_token_id}
                    if return_logits:
                        event["logits"] = _no_match_logits
                    yield event
                    metrics.generated_tokens += 1

                    committed_len += 1
                    tokens_remaining -= 1
                    total_accepted += 1

                    metrics.blocks_completed += 1
                    metrics.total_iterations += 1
                    metrics.iterations_per_block.append(1)
                    metrics.tokens_per_block.append(1)

                    print(f"  [Lookahead] Step {total_steps}: "
                          f"accepted 1 token (no match from {len(candidates)} candidates), "
                          f"{metrics.forward_times_ms[-1]:.0f}ms")
            else:
                # No candidates available, standard sequential forward
                self._cloud_crop_to = committed_len
                self._cloud_attn_mask = None

                block_id = torch.tensor([first_token_id], device=self.device)
                cache_position = torch.tensor([committed_len], device=self.device)
                position_ids = cache_position.unsqueeze(0)

                with torch.no_grad():
                    hidden = self.model.model.embed_tokens(block_id.unsqueeze(0))
                    cos, sin = self.model.model.rotary_emb(hidden, position_ids)
                    position_embeddings = (cos, sin)

                    hidden = self._forward_layers(
                        hidden, position_embeddings, cache, cache_position,
                        position_ids, None
                    )

                    hidden = self.model.model.norm(hidden)
                    logits = self.model.lm_head(hidden)

                metrics.forward_times_ms.append((time.time() - fwd_start) * 1000)
                _no_cand_logits = last_logits.squeeze().cpu() if return_logits else None
                last_logits = logits[:, -1:, :]

                all_generated_ids.append(first_token_id)
                full_text = self.tokenizer.decode(all_generated_ids)
                new_text = full_text[len(prev_text):]
                prev_text = full_text
                event = {"token": new_text, "token_id": first_token_id}
                if return_logits:
                    event["logits"] = _no_cand_logits
                yield event
                metrics.generated_tokens += 1

                committed_len += 1
                tokens_remaining -= 1
                total_accepted += 1

                metrics.blocks_completed += 1
                metrics.total_iterations += 1
                metrics.iterations_per_block.append(1)
                metrics.tokens_per_block.append(1)

            # Add new n-grams from generated output to pool
            all_ids = prompt_ids + all_generated_ids
            if len(all_ids) >= N:
                key = all_ids[-N]
                ngram = tuple(all_ids[-N + 1:])
                if ngram not in pool[key]:
                    pool[key].append(ngram)

        metrics.total_time_ms = (time.time() - gen_start) * 1000
        avg_accepted = total_accepted / max(1, total_steps)
        print(f"  [Lookahead] Done: {metrics.generated_tokens} tokens in "
              f"{metrics.total_time_ms:.0f}ms ({metrics.tokens_per_second:.1f} tok/s), "
              f"avg accepted/step: {avg_accepted:.2f}, "
              f"pool size: {sum(len(v) for v in pool.values())}")

        if self.cloud_url:
            self._cloud_end_session()

        yield {"done": True, **metrics.summary()}

    def _prefill(self, input_ids, cache):
        """Process full prompt through all layers, building KV cache.
        Returns logits for the entire prompt."""
        seq_len = input_ids.shape[1]
        cache_position = torch.arange(seq_len, device=self.device)
        position_ids = cache_position.unsqueeze(0)

        with torch.no_grad():
            hidden = self.model.model.embed_tokens(input_ids)
            cos, sin = self.model.model.rotary_emb(hidden, position_ids)
            position_embeddings = (cos, sin)

            hidden = self._forward_layers(
                hidden, position_embeddings, cache, cache_position, position_ids
            )

            hidden = self.model.model.norm(hidden)
            logits = self.model.lm_head(hidden)

        return logits

    def _forward_block_logits(self, block_ids, committed_len, cache):
        """Forward pass for a Jacobi block of k tokens.
        Appends k KV entries to cache. Caller must crop if not converged.

        Returns: logits [1, k, vocab_size]
        """
        k = block_ids.shape[0]
        cache_position = torch.arange(
            committed_len, committed_len + k, device=self.device
        )
        position_ids = cache_position.unsqueeze(0)
        # k == 1 (sequential): a 1-row mask is all zeros = attend everything,
        # identical to attn_mask=None. Passing None keeps sdpa on the same
        # kernel path as an unmasked reference forward; the explicit fp16
        # zero mask perturbs logits by ~tie-flip magnitude (seen as a
        # deterministic '--' vs '---' flip at a 0.0156-margin position).
        attn_mask = None if k == 1 else self._make_causal_mask(committed_len, k)

        with torch.no_grad():
            hidden = self.model.model.embed_tokens(block_ids.unsqueeze(0))
            cos, sin = self.model.model.rotary_emb(hidden, position_ids)
            position_embeddings = (cos, sin)

            hidden = self._forward_layers(
                hidden, position_embeddings, cache, cache_position,
                position_ids, attn_mask
            )

            hidden = self.model.model.norm(hidden)
            logits = self.model.lm_head(hidden)

        return logits

    def _commit_block(self, block_ids, committed_len, cache):
        """Forward a block through all layers with use_cache=True
        to populate the KV cache with correct entries."""
        k = block_ids.shape[0]
        cache_position = torch.arange(
            committed_len, committed_len + k, device=self.device
        )
        position_ids = cache_position.unsqueeze(0)
        attn_mask = self._make_causal_mask(committed_len, k)

        with torch.no_grad():
            hidden = self.model.model.embed_tokens(block_ids.unsqueeze(0))
            cos, sin = self.model.model.rotary_emb(hidden, position_ids)
            position_embeddings = (cos, sin)

            self._forward_layers(
                hidden, position_embeddings, cache, cache_position,
                position_ids, attn_mask
            )

    def _forward_layers(self, hidden, position_embeddings, cache,
                        cache_position, position_ids, attn_mask=None):
        """Route through layers: all local or split with cloud."""
        if self.cloud_url is None:
            return self._forward_all_layers(
                hidden, position_embeddings, cache, cache_position,
                position_ids, attn_mask
            )
        else:
            return self._forward_split_layers(
                hidden, position_embeddings, cache, cache_position,
                position_ids, attn_mask
            )

    def _forward_all_layers(self, hidden, position_embeddings, cache,
                            cache_position, position_ids, attn_mask=None):
        """All layers on local GPU (local-only mode)."""
        for idx in range(NUM_LAYERS):
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

    def _forward_split_layers(self, hidden, position_embeddings, cache,
                              cache_position, position_ids, attn_mask=None):
        """Split mode: early layers local, middle cloud, late layers local."""
        # Early layers (0-1)
        for idx in range(SPLIT_AFTER + 1):
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

        # [E8] Secret obfuscation at the wire boundary (defense experiment):
        # h -> h @ W before transmission; inverse (W^T) on the returned tensor.
        # [ER] With the ratchet enabled, W is the per-epoch W_t and the epoch
        # is forwarded to the cloud via _send_to_cloud ("obf_epoch" header).
        obf_W, obf_Wt = self.obf_W, self.obf_Wt
        if self._ratchet_enabled():
            epoch = self._ratchet_advance(
                hidden.shape[1] if self._cloud_is_prompt else 1)
            obf_W, obf_Wt = self._derive_epoch_W(epoch)
            self._obf_current_epoch = epoch
        if obf_W is not None:
            hidden = torch.einsum('bsh,hd->bsd', hidden.float(), obf_W).to(hidden.dtype)
        # Apply wire defenses (simulated quantization/noise)
        hidden = self._apply_wire_defenses(hidden)
        # Cloud layers (2-29) - pass custom mask if set
        if self._cloud_attn_mask is None and attn_mask is not None:
            self._cloud_attn_mask = attn_mask
        hidden = self._send_to_cloud(
            hidden, position_embeddings, cache_position,
            is_prompt=self._cloud_is_prompt,
            crop_to=self._cloud_crop_to
        )
        if obf_Wt is not None:
            hidden = torch.einsum('bsh,hd->bsd', hidden.float(), obf_Wt).to(hidden.dtype)

        # Late layers (30-31)
        for idx in range(RESUME_AT, NUM_LAYERS):
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

    def _cloud_new_session(self):
        """Open WebSocket connection to cloud server."""
        # Derive WebSocket URL from HTTP URL (port 5001)
        ws_url = self.cloud_url.replace('http://', 'ws://').replace(':5000', ':5001')
        self._ws = ws_connect(ws_url, max_size=20 * 1024 * 1024)
        # Server sends session ID on connect
        resp = json.loads(self._ws.recv())
        self.cloud_session_id = resp['session_id']
        print(f"    [cloud-ws] Connected, session={self.cloud_session_id[:8]}")

    def _cloud_end_session(self):
        """Close WebSocket connection."""
        if self._ws:
            try:
                self._ws.send(json.dumps({"type": "end"}))
                self._ws.close()
            except:
                pass
            self._ws = None
            self.cloud_session_id = None

    def _send_to_cloud(self, hidden, position_embeddings, cache_position,
                       is_prompt=False, crop_to=None):
        """Send intermediate activations to cloud via WebSocket.

        Binary protocol: [4B header_len][JSON header][tensor bytes]
        Tensor bytes: hidden + cos + sin + [optional mask] (raw float16, no base64)
        """
        t0 = time.time()

        # Serialize to raw bytes (no base64!)
        hs_np = hidden.cpu().numpy()
        cos_np = position_embeddings[0].cpu().numpy()
        sin_np = position_embeddings[1].cpu().numpy()

        t1 = time.time()

        # Build header with optional mask and relocate
        header_dict = {
            "hidden_shape": list(hs_np.shape),
            "pe_shape": list(cos_np.shape),
            "is_prompt": is_prompt,
            "crop_to": crop_to,
            "diagnostic_state": self._diagnostic_state,
        }

        # Optional relocate command (for n-gram speculation KV management)
        if self._cloud_relocate is not None:
            header_dict["relocate"] = self._cloud_relocate
            self._cloud_relocate = None  # One-shot

        # [ER] Ratchet: tell the cloud which epoch's W_t this tensor is
        # wrapped with (absent = no ratchet; backward compatible).
        # D2 (issue #88): also send the W_t digest for cross-node comparison.
        if self._obf_current_epoch is not None:
            header_dict["obf_epoch"] = self._obf_current_epoch
            header_dict["obf_ratchet_version"] = self.obf_ratchet_version
            if self._obf_current_epoch in self._obf_epoch_whash:
                header_dict["w_hash"] = self._obf_epoch_whash[
                    self._obf_current_epoch]
            self._obf_current_epoch = None  # One-shot

        # Optional custom attention mask
        mask_bytes = b''
        if self._cloud_attn_mask is not None:
            mask_np = self._cloud_attn_mask.cpu().numpy()
            mask_bytes = mask_np.tobytes()
            header_dict["has_mask"] = True
            header_dict["mask_shape"] = list(mask_np.shape)
            self._cloud_attn_mask = None  # One-shot

        header = json.dumps(header_dict).encode()

        msg = (struct.pack('>I', len(header))
               + header
               + hs_np.tobytes()
               + cos_np.tobytes()
               + sin_np.tobytes()
               + mask_bytes)

        self._ws.send(msg)

        t2 = time.time()

        # Receive binary response
        resp = self._ws.recv()

        t3 = time.time()

        # Unpack: [4B header_len][JSON header][tensor bytes]
        resp_header_len = struct.unpack('>I', resp[:4])[0]
        result = json.loads(resp[4:4 + resp_header_len])
        self._last_cloud_diagnostic = {
            "cached_seq_len": result.get("cached_seq_len"),
            "sampled_state_sha256": result.get("sampled_state_sha256"),
        }
        tensor_data = resp[4 + resp_header_len:]

        out = torch.frombuffer(
            bytearray(tensor_data), dtype=torch.float16
        ).reshape(result['hidden_shape']).clone().to(self.device)

        if out.dim() == 2:
            out = out.unsqueeze(0)

        t4 = time.time()
        cloud_ms = result.get('process_time_ms', 0)
        print(f"    [cloud-ws] serialize={1000*(t1-t0):.0f}ms "
              f"send={1000*(t2-t1):.0f}ms "
              f"recv={1000*(t3-t2):.0f}ms "
              f"(cloud_gpu={cloud_ms:.0f}ms) "
              f"deserialize={1000*(t4-t3):.0f}ms "
              f"total={1000*(t4-t0):.0f}ms "
              f"seq_len={hidden.shape[1]}")

        self._step_timings.append({
            "serialize_ms": round((t1 - t0) * 1000, 2),
            "send_ms": round((t2 - t1) * 1000, 2),
            "recv_ms": round((t3 - t2) * 1000, 2),
            "deserialize_ms": round((t4 - t3) * 1000, 2),
            "total_ms": round((t4 - t0) * 1000, 2),
            "cloud_gpu_ms": cloud_ms,
            "network_rtt_ms": round((t3 - t1) * 1000 - cloud_ms, 2),
            "seq_len": hidden.shape[1],
        })

        return out


# === Flask App ===

app = Flask(__name__)
decoder = None
DECODE_MODE = "sequential"  # "sequential", "jacobi", or "lookahead"


def load_model(model_path, device, split_after=None, resume_at=None):
    global decoder, NUM_LAYERS, SPLIT_AFTER, RESUME_AT
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if CLOUD_URL:
        # Split mode: only local layers + embeddings + head are needed locally.
        # [QWEN36] Determine split points from the config first, then stream the
        # load so middle layers land on "meta" (never materialized). On
        # unified-memory systems (DGX Spark) a full CPU copy of a 27B/35B model
        # (~54–73GB) OOMs the node before any trim can run. Falls back to the
        # original full-CPU load if the custom map is rejected.
        from transformers import AutoConfig
        _cfg = AutoConfig.from_pretrained(model_path)
        _tc = getattr(_cfg, "text_config", None)
        _n = (getattr(_cfg, "num_hidden_layers", None)
              or getattr(_tc, "num_hidden_layers", None))
        SPLIT_AFTER = split_after if split_after is not None else 1
        RESUME_AT = resume_at if resume_at is not None else (_n - 2 if _n else None)

        model = None
        if _n is not None:
            # Local modules straight to GPU (accelerate materializes only
            # GPU-resident modules eagerly; "cpu" would mean lazy offload).
            _map = {"model.embed_tokens": 0, "model.rotary_emb": 0,
                    "model.norm": 0, "lm_head": 0}
            for _i in range(_n):
                _map[f"model.layers.{_i}"] = (
                    0 if (_i <= SPLIT_AFTER or _i >= RESUME_AT) else "meta")
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path, dtype=torch.float16, device_map=_map)
                print(f"Streamed load: {_n} layers, middle on meta (not materialized)")
            except Exception as e:
                print(f"Custom device_map rejected ({type(e).__name__}: {e}); "
                      f"falling back to full CPU load")
                model = None
        if model is None:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, dtype=torch.float16, device_map="cpu",
            )
        model.eval()

        NUM_LAYERS = len(model.model.layers)
        RESUME_AT = RESUME_AT if RESUME_AT is not None else NUM_LAYERS - 2

        # Move only local layers + embeddings + head to GPU
        model.model.embed_tokens = model.model.embed_tokens.to(device)
        model.model.rotary_emb = model.model.rotary_emb.to(device)
        for i in range(SPLIT_AFTER + 1):
            model.model.layers[i] = model.model.layers[i].to(device)
        for i in range(RESUME_AT, NUM_LAYERS):
            model.model.layers[i] = model.model.layers[i].to(device)
        model.model.norm = model.model.norm.to(device)
        model.lm_head = model.lm_head.to(device)

        # [QWEN36] Free the middle-layer weights from the local CPU copy — in
        # split mode they are never executed locally. On unified-memory systems
        # (DGX Spark) the CPU copy of a 27B/35B model is ~54–73GB and OOMs the
        # node; Identity() keeps layer indexing intact.
        import gc
        for i in range(SPLIT_AFTER + 1, RESUME_AT):
            model.model.layers[i] = torch.nn.Identity()
        gc.collect()
        torch.cuda.empty_cache()

        mem_gb = torch.cuda.memory_allocated() / 1e9
        print(f"Split mode: {NUM_LAYERS} layers total, "
              f"local 0-{SPLIT_AFTER} + {RESUME_AT}-{NUM_LAYERS-1} on {torch.cuda.get_device_name(0)}, "
              f"cloud {SPLIT_AFTER+1}-{RESUME_AT-1}, "
              f"VRAM: {mem_gb:.1f}GB")
    else:
        # Local-only mode: load everything to GPU
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.float16, device_map=device,
        )
        model.eval()

        NUM_LAYERS = len(model.model.layers)
        SPLIT_AFTER = split_after if split_after is not None else 1
        RESUME_AT = resume_at if resume_at is not None else NUM_LAYERS - 2

        mem_gb = torch.cuda.memory_allocated() / 1e9
        print(f"Local-only: {NUM_LAYERS} layers on {torch.cuda.get_device_name(0)}, "
              f"VRAM: {mem_gb:.1f}GB")

    decoder = JacobiDecoder(model, tokenizer, device, cloud_url=CLOUD_URL,
                            block_size=BLOCK_SIZE, max_iters=MAX_JACOBI_ITERS)


@app.route('/health', methods=['GET'])
def health():
    mode = f"local-only (all {NUM_LAYERS} layers)" if CLOUD_URL is None else f"split (cloud: {CLOUD_URL})"
    return jsonify({
        "status": "ok",
        "gpu": torch.cuda.get_device_name(0),
        "memory_used_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
        "num_layers": NUM_LAYERS,
        "local_layers": f"0-{NUM_LAYERS-1}" if CLOUD_URL is None else f"0-{SPLIT_AFTER}, {RESUME_AT}-{NUM_LAYERS-1}",
        "cloud_url": CLOUD_URL or "disabled",
        "streaming": True,
        "jacobi": True,
        "block_size": decoder.block_size,
        "max_jacobi_iters": decoder.max_iters,
        "mode": mode,
    })


def _get_generator(prompt, max_new_tokens):
    """Route to the appropriate generation method based on DECODE_MODE."""
    if DECODE_MODE == "lookahead":
        return decoder.generate_stream_lookahead(prompt, max_new_tokens)
    else:
        return decoder.generate_stream(prompt, max_new_tokens)


@app.route('/generate_stream', methods=['POST'])
def generate_stream():
    data = request.json
    prompt = data.get('prompt', '')
    max_new_tokens = data.get('max_new_tokens', 500)

    print(f"\n[Request] mode={DECODE_MODE} prompt={prompt[:80]}... max_tokens={max_new_tokens}")

    def event_stream():
        for event in _get_generator(prompt, max_new_tokens):
            yield f"data: {json.dumps(event)}\n\n"

    return Response(event_stream(), mimetype='text/event-stream')


@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt', '')
    max_new_tokens = data.get('max_new_tokens', 500)

    print(f"\n[Request] mode={DECODE_MODE} prompt={prompt[:80]}... max_tokens={max_new_tokens}")

    token_ids = []
    stats = None
    for event in _get_generator(prompt, max_new_tokens):
        if 'token_id' in event:
            token_ids.append(event['token_id'])
        elif 'done' in event:
            stats = event

    response_text = decoder.tokenizer.decode(token_ids, skip_special_tokens=True)
    return jsonify({
        "response": response_text,
        **stats,
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Jacobi Parallel Decoding Server')
    parser.add_argument('--model', default=MODEL_PATH, help='Path to model (e.g., Mistral 7B, LLaMA 2 13B)')
    parser.add_argument('--port', type=int, default=5001, help='Server port')
    parser.add_argument('--cloud', default=None, help='Cloud server URL for split mode')
    parser.add_argument('--block-size', type=int, default=BLOCK_SIZE, help='Jacobi block size k')
    parser.add_argument('--max-iters', type=int, default=MAX_JACOBI_ITERS, help='Max Jacobi iterations per block')
    parser.add_argument('--mode', default='sequential', choices=['sequential', 'jacobi', 'lookahead'],
                        help='Decoding mode: sequential (block_size=1), jacobi, or lookahead (n-gram speculation)')
    parser.add_argument('--split-after', type=int, default=None,
                        help='Last local layer before cloud (default: 1)')
    parser.add_argument('--resume-at', type=int, default=None,
                        help='First local layer after cloud (default: num_layers - 2)')
    parser.add_argument('--obfuscate-key', default=None,
                        help='[E8] path to secret orthogonal key file (generated if missing); enables boundary obfuscation')
    parser.add_argument('--obfuscate-seed', type=int, default=42,
                        help='[E8] deployment/session seed used only when generating a new --obfuscate-key; rotate by using a new key path and seed')
    parser.add_argument('--obfuscate-ratchet-tokens', type=int, default=0,
                        help='[ER] served tokens per epoch for ratcheted rotation (0=off); W_t derived from the seed chain, no key file')
    parser.add_argument('--obfuscate-seed-base', type=int, default=None,
                        help='[ER] seed base S for the ratchet seed chain sha256("S:t"); required to enable ratcheting')
    parser.add_argument('--obfuscate-budget-events', type=int, default=0,
                        help='[ER] evidence budget B for budget-triggered rotation (0=off); served tokens +1, repeated 64-token prefixes +S. '
                             'When --obfuscate-seed-base is set without N or B, defaults to B=128 (half the E-R1a max_safe_epoch=256, depths 1/4/8, 12B+27B)')
    parser.add_argument('--obfuscate-ratchet-version', choices=['v1', 'v2'],
                        default='v2',
                        help='portable v2 is mandatory for production; v1 is historical replay only')
    parser.add_argument('--obfuscate-count-return-rows', action='store_true',
                        help='production-v3: reserve request and response rows '
                             'before every boundary exchange')
    parser.add_argument('--allow-unsafe-oversized-ratchet-forward', action='store_true',
                        help='UNSAFE historical replay only: allow a prefill larger than the per-key cadence')
    args = parser.parse_args()

    if args.obfuscate_ratchet_tokens and args.obfuscate_budget_events:
        parser.error("choose only one of --obfuscate-ratchet-tokens and --obfuscate-budget-events")

    # [ER] Evidence-based default cadence: E-R1a measured max_safe_epoch=256
    # (50% recovery bracket (256, 1024]) on Mistral-NeMo-12B and Qwen3.6-27B,
    # depths 1/4/8 — rotation_lifetime_{12b,27b}_bf16_20260811T175827.json.
    # Budget-triggered rotation at B=128 applies a 2x safety margin below the
    # measured safe limit (12B pooled cross-epoch accumulation drifts higher
    # than 27B) without a fixed token cadence.
    if (args.obfuscate_seed_base is not None
            and args.obfuscate_ratchet_tokens == 0
            and args.obfuscate_budget_events == 0):
        args.obfuscate_budget_events = 128
        print("[ER] no cadence specified; defaulting to budget-triggered "
              "rotation B=128 (E-R1a max_safe_epoch=256, 2x margin)")

    MODEL_PATH = args.model
    CLOUD_URL = args.cloud
    BLOCK_SIZE = args.block_size if args.mode == 'jacobi' else 1
    MAX_JACOBI_ITERS = args.max_iters
    DECODE_MODE = args.mode

    load_model(MODEL_PATH, DEVICE, split_after=args.split_after, resume_at=args.resume_at)
    if args.obfuscate_key:
        decoder.load_obfuscation(key_path=args.obfuscate_key,
                                 seed=args.obfuscate_seed)
    decoder.enable_ratchet(ratchet_tokens=args.obfuscate_ratchet_tokens,
                           seed_base=args.obfuscate_seed_base,
                           budget_events=args.obfuscate_budget_events,
                           version=args.obfuscate_ratchet_version,
                           strict_budget=not args.allow_unsafe_oversized_ratchet_forward,
                           count_return_rows=args.obfuscate_count_return_rows)
    if decoder._ratchet_enabled():
        print(f"[ER] ratchet enabled: N={args.obfuscate_ratchet_tokens} "
              f"B={args.obfuscate_budget_events} "
              f"version={args.obfuscate_ratchet_version} seed=redacted")
    # [ER] Seed-base hygiene: whoever guesses S derives EVERY epoch key.
    # Warn (not refuse — historical evidence runs used small S by design and
    # must stay reproducible). Generate a real one with:
    #   python split-training/er_ratchet.py --generate-seed
    if (args.obfuscate_seed_base is not None
            and args.obfuscate_seed_base < (1 << 32)):
        print(f"[ER] WARNING: seed base S={args.obfuscate_seed_base} is "
              f"below 2^32 — trivially brute-forceable. Fine for "
              f"reproducing evidence runs; for real deployments use a "
              f"128-bit CSPRNG seed (er_ratchet.py --generate-seed).")
    print(f"\nStarting server on port {args.port}...")
    print(f"  Mode: {DECODE_MODE}, Block size: {BLOCK_SIZE}")
    app.run(host='0.0.0.0', port=args.port, threaded=True)
