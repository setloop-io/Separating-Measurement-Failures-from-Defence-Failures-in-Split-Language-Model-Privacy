"""
Split Inference - Cloud Server with KV Cache
Maintains session-based KV cache for fast token generation.
Model-agnostic: works with any HuggingFace causal LM.

Transport:
  - WebSocket (primary, low-latency binary protocol)
  - HTTP/Flask (health checks, backward compat)

Usage:
  python cloud_server_kv.py --model ~/models/mistral-7b-instruct
  python cloud_server_kv.py --model ~/models/llama-2-13b-chat --cloud-start 2 --cloud-end 37
"""

import torch
import numpy as np
import os
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
import asyncio
import websockets
import struct
import json
import hashlib
import base64
import time
import uuid
import socket
import threading
import traceback
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from privacy_runtime.ratchet_v2 import derive_orthogonal as derive_orthogonal_v2

app = Flask(__name__)
model = None

CLOUD_START = 2
CLOUD_END = 29
HYBRID = False  # [QWEN36] True for hybrid linear-attention models (Qwen3.5/3.6)

# [E9] Optional shared-key deobfuscation (transport-encryption threat model:
# W protects the wire from eavesdroppers, NOT the cloud from itself — the
# cloud holds the key here). None = off.
OBF_W = None
OBF_Wt = None

# [ER] Ratcheted per-epoch rotation: production v2 uses the portable,
# domain-separated SHA-256 counter stream in privacy_runtime. The epoch and
# derivation version ride in each request header. NOTE: in this shared-key mode
# the cloud derives W_t
# itself, so per-epoch freshness protects the WIRE and logged captures, not
# cloud RAM. Off by default.
OBF_SEED_BASE = None
OBF_RATCHET = False
OBF_RATCHET_VERSION = "v2"
OBF_EPOCH_CACHE = {}  # {epoch: (W, Wt)} — current epoch only
OBF_EPOCH_WHASH = {}  # {epoch: w_hash} — D2 (issue #88): digest for compare


def _epoch_seed(seed_base, epoch):
    """[ER] Seed chain, identical to the local node's derivation."""
    return int.from_bytes(
        hashlib.sha256(f"{seed_base}:{epoch}".encode()).digest()[:8],
        "little")


def derive_epoch_W(epoch):
    """[ER] Derive (W_t, W_t^T) with the same QR-sign construction the local
    node uses; cache ONLY the requested epoch (evict older ones)."""
    global OBF_EPOCH_CACHE
    if epoch in OBF_EPOCH_CACHE:
        return OBF_EPOCH_CACHE[epoch]
    H = model.model.embed_tokens.weight.shape[1]
    if OBF_RATCHET_VERSION != "v2":
        # D4 (issue #88): v1 derivation removed; see er_ratchet._derive_W.
        raise RuntimeError(
            f"[ER] ratchet version {OBF_RATCHET_VERSION!r} is no longer "
            "supported; only the shared ratchet_v2 derivation exists")
    W = derive_orthogonal_v2(OBF_SEED_BASE, epoch, H)
    w_hash = hashlib.sha256(W.numpy().tobytes()).hexdigest()[:16]
    device = next(model.parameters()).device
    OBF_EPOCH_CACHE = {epoch: (W.to(device), W.T.contiguous().to(device))}
    OBF_EPOCH_WHASH.clear()
    OBF_EPOCH_WHASH[epoch] = w_hash
    print(f"[ER] ratchet epoch {epoch} active "
          f"(W sha256={w_hash}, seed=redacted, version={OBF_RATCHET_VERSION})")
    return OBF_EPOCH_CACHE[epoch]


def enable_ratchet(seed_base, ratchet_tokens=0, budget_events=0,
                   version="v2"):
    """[ER] Enable per-epoch unwrapping. The token/budget knobs live on the
    local node (it counts and transmits the epoch); they are accepted here
    for flag parity only."""
    global OBF_SEED_BASE, OBF_RATCHET, OBF_RATCHET_VERSION
    if version not in ("v1", "v2"):
        raise ValueError("ratchet version must be v1 or v2")
    OBF_SEED_BASE = seed_base
    OBF_RATCHET_VERSION = version
    OBF_RATCHET = (seed_base is not None
                   and (ratchet_tokens > 0 or budget_events > 0))
    if OBF_RATCHET:
        print(f"[ER] ratchet enabled: version={version}, seed=redacted "
              "(local node counts tokens; epochs arrive per request)")


def validate_ratchet_header(obf_epoch, version, w_hash=None):
    """Fail closed on missing, extra, or mismatched rotation metadata."""
    if OBF_RATCHET:
        if obf_epoch is None:
            raise RuntimeError("[ER] rotated cloud refused frame without obf_epoch")
        if (not isinstance(obf_epoch, int) or isinstance(obf_epoch, bool)
                or obf_epoch < 0):
            raise RuntimeError("[ER] obf_epoch must be a non-negative integer")
        if version != OBF_RATCHET_VERSION:
            raise RuntimeError("[ER] ratchet-version mismatch: "
                               f"cloud={OBF_RATCHET_VERSION!r}, frame={version!r}")
        if w_hash is not None:
            # D2 (issue #88): compare the peer's W_t digest against ours.
            # A mismatch means key divergence (wrong master, version drift,
            # or corruption) — continuing would decode garbage, so abort.
            derive_epoch_W(obf_epoch)  # ensures OBF_EPOCH_WHASH is populated
            local = OBF_EPOCH_WHASH.get(obf_epoch)
            if local is not None and w_hash != local:
                raise RuntimeError(
                    f"[ER] w_hash mismatch at epoch {obf_epoch}: "
                    f"cloud={local}, frame={w_hash} — key divergence, aborting")
    elif obf_epoch is not None or version is not None:
        raise RuntimeError("[ER] unconfigured cloud refused rotated frame")


def load_deobfuscation(key_path):
    """[E9] Load the boundary key W: unwrap hW -> h on receipt, re-wrap on
    return. Set via --deobfuscate-key."""
    global OBF_W, OBF_Wt
    W = torch.load(key_path, map_location="cpu")
    OBF_W = W.cuda()
    OBF_Wt = W.T.contiguous().cuda()
    err = (W @ W.T - torch.eye(W.shape[0])).abs().max().item()
    print(f"[E9] deobfuscation key loaded from {key_path} "
          f"(H={W.shape[0]}, orthogonality err {err:.2e})")

# Session-based KV caches: {session_id: {'cache': DynamicCache, 'last_access': time}}
sessions = {}
SESSION_TIMEOUT = 300  # 5 minutes


def sampled_cache_state_hash(cache, samples_per_tensor=32):
    """Return an opaque, low-overhead fingerprint of cache state.

    Diagnostic runs need to establish whether rollback restores the same
    state without exporting cache tensors.  Hash tensor metadata plus fixed,
    evenly-spaced samples; no activations or cache values leave the server.
    The name in the wire response explicitly says ``sampled`` so this is not
    mistaken for a byte-for-byte hash of every cache element.
    """
    digest = hashlib.sha256()
    if cache is None:
        digest.update(b"empty-cache")
        return digest.hexdigest()
    for layer_idx, layer in enumerate(cache.layers):
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
                sample = flat.index_select(0, indices).float().cpu().numpy()
                digest.update(sample.tobytes())
    return digest.hexdigest()


def load_model(model_path, cloud_start=None, cloud_end=None):
    global model, CLOUD_START, CLOUD_END, HYBRID
    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        device_map="cuda"
    )
    model.eval()
    num_layers = len(model.model.layers)
    # [QWEN36] Hybrid models interleave gated-delta-net (linear attention)
    # layers with full-attention layers; linear layers keep recurrent + conv
    # state in the cache that crop() cannot roll back.
    HYBRID = any(hasattr(l, "linear_attn") for l in model.model.layers)
    CLOUD_START = cloud_start if cloud_start is not None else 2
    CLOUD_END = cloud_end if cloud_end is not None else num_layers - 3
    print(f"Loaded. {num_layers} layers total, serving {CLOUD_START}-{CLOUD_END}, "
          f"hybrid={HYBRID}, "
          f"GPU mem: {torch.cuda.memory_allocated()/1e9:.1f}GB")


# [QWEN36] Linear-attention rollback helpers (hybrid models only).
# LinearAttentionCacheLayerMixin.crop() is a no-op by design, so the only
# way to roll those layers back to the committed frontier is to clone and
# restore their conv/recurrent state tensors.

def checkpoint_linear_states(cache):
    return [(layer, layer.conv_states.clone(), layer.recurrent_states.clone())
            for layer in cache.layers
            if getattr(layer, "recurrent_states", None) is not None]


def restore_linear_states(ckpt):
    for layer, conv, rec in ckpt:
        layer.conv_states.copy_(conv)
        layer.recurrent_states.copy_(rec)


def cleanup_old_sessions():
    """Remove sessions older than timeout."""
    now = time.time()
    expired = [sid for sid, data in sessions.items() if now - data['last_access'] > SESSION_TIMEOUT]
    for sid in expired:
        del sessions[sid]
    if expired:
        print(f"Cleaned up {len(expired)} expired sessions")


def process_tensors(session, hidden, position_embeddings, seq_len, is_prompt,
                    crop_to, relocate=None, custom_mask=None,
                    session_id=None, request_seq=None, obf_epoch=None):
    """Core processing logic shared by HTTP and WebSocket handlers."""
    # [ER] Per-epoch key for this request (ratchet mode; the local node
    # counts tokens and transmits the epoch — no independent counter here).
    epoch_W = epoch_Wt = None
    if OBF_RATCHET and obf_epoch is not None:
        epoch_W, epoch_Wt = derive_epoch_W(obf_epoch)
    # [E9] Optional wire capture: dump received activations for the attack eval.
    # Each wire_NNNN.pt is paired with a wire_NNNN.json sidecar carrying the
    # session/phase/position metadata — the eval aligns tensors to tokens via
    # the sidecars, NOT filename order. "epoch" is the ratchet epoch (null
    # when the ratchet is off).
    cap_dir = os.environ.get("E9_CAPTURE_DIR")
    if cap_dir:
        os.makedirs(cap_dir, exist_ok=True)
        n = len([f for f in os.listdir(cap_dir)
                 if f.startswith("wire_") and f.endswith(".pt")])
        if n < 64:
            torch.save(hidden.detach().cpu(), os.path.join(cap_dir, f"wire_{n:04d}.pt"))
            meta = {"session_id": session_id,
                    "request_seq": request_seq,
                    "phase": "prefill" if is_prompt else "decode",
                    "position": 0 if is_prompt else session.get('seq_len', 0),
                    "epoch": obf_epoch,  # transmitted epoch (None when ratchet off)
                    "timestamp": time.time()}
            with open(os.path.join(cap_dir, f"wire_{n:04d}.json"), "w") as mf:
                json.dump(meta, mf)
    # [E9] shared-key mode: unwrap the obfuscated wire tensor before the
    # layer stack (the capture above deliberately records the RAW wire
    # tensor hW — that is what a wire eavesdropper sees). [ER] the per-epoch
    # key takes precedence over the static --deobfuscate-key.
    if epoch_Wt is not None:
        hidden = torch.einsum('bsh,hd->bsd', hidden.float(), epoch_Wt).to(hidden.dtype)
    elif OBF_Wt is not None:
        hidden = torch.einsum('bsh,hd->bsd', hidden.float(), OBF_Wt).to(hidden.dtype)
    # Handle cache management
    if is_prompt:
        # [QWEN36] config-aware cache: linear layers need
        # LinearAttentionLayer entries (plain DynamicCache() crashes with
        # IndexError in update_conv_state on hybrid models)
        session['cache'] = DynamicCache(config=model.config) if HYBRID else DynamicCache()
        session['linear_ckpt'] = None
        cache_position = torch.arange(seq_len, device='cuda')
        session['seq_len'] = seq_len
    else:
        # Relocate KV entries before cropping (for n-gram speculation)
        if relocate is not None and session['cache'] is not None:
            src, dst, length = relocate['src'], relocate['dst'], relocate['len']
            for layer in session['cache'].layers:
                # [QWEN36] LinearAttentionLayer entries have no .keys/.values
                if getattr(layer, "keys", None) is None:
                    continue
                layer.keys[:, :, dst:dst+length, :] = \
                    layer.keys[:, :, src:src+length, :].clone()
                layer.values[:, :, dst:dst+length, :] = \
                    layer.values[:, :, src:src+length, :].clone()

        if crop_to is not None and session['cache'] is not None:
            if HYBRID and crop_to >= session['seq_len']:
                # [QWEN36] No-op crop at the committed frontier: the client
                # is about to speculate from here — snapshot linear state.
                session['linear_ckpt'] = checkpoint_linear_states(session['cache'])
            else:
                session['cache'].crop(crop_to)
                if HYBRID and session.get('linear_ckpt'):
                    # [QWEN36] Real rollback: crop KV and restore linear state.
                    restore_linear_states(session['linear_ckpt'])
            session['seq_len'] = crop_to
        cache_position = torch.arange(
            session['seq_len'], session['seq_len'] + seq_len, device='cuda'
        )
        session['seq_len'] += seq_len

    position_ids = cache_position.unsqueeze(0)

    # Use custom mask if provided, otherwise build default causal mask
    if custom_mask is not None:
        attn_mask = custom_mask
    else:
        attn_mask = None
        if not is_prompt and seq_len > 1 and session['seq_len'] - seq_len > 0:
            committed_len = session['seq_len'] - seq_len
            kv_len = session['seq_len']
            attn_mask = torch.full(
                (1, 1, seq_len, kv_len), float('-inf'),
                device='cuda', dtype=torch.float16
            )
            for i in range(seq_len):
                attn_mask[0, 0, i, :committed_len + i + 1] = 0.0

    # Process middle layers with KV cache
    with torch.no_grad():
        for idx in range(CLOUD_START, CLOUD_END + 1):
            layer = model.model.layers[idx]
            out = layer(
                hidden,
                attention_mask=attn_mask,
                position_ids=position_ids,
                past_key_values=session['cache'],
                use_cache=True,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            hidden = out[0]
            if hidden.dim() == 2:
                hidden = hidden.unsqueeze(0)

    # [E9] shared-key mode: re-wrap before returning to the local node
    if epoch_W is not None:
        hidden = torch.einsum('bsh,hd->bsd', hidden.float(), epoch_W).to(hidden.dtype)
    elif OBF_W is not None:
        hidden = torch.einsum('bsh,hd->bsd', hidden.float(), OBF_W).to(hidden.dtype)

    return hidden


# === WebSocket Handler (primary, low-latency) ===

async def ws_handler(websocket):
    """Handle a WebSocket connection for one generation session.

    Protocol:
      1. Server sends text JSON: {"session_id": "..."}
      2. Client sends binary frames: [4B header_len][JSON header][tensor bytes]
         Header: {"hidden_shape", "pe_shape", "is_prompt", "crop_to"}
         Tensor bytes: hidden_bytes + cos_bytes + sin_bytes
      3. Server responds binary: [4B header_len][JSON header][tensor bytes]
         Header: {"hidden_shape", "process_time_ms", "cached_seq_len"}
      4. Client sends text JSON: {"type": "end"} to close
    """
    cleanup_old_sessions()
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        'cache': None,
        'last_access': time.time(),
        'seq_len': 0
    }

    # Send session ID
    await websocket.send(json.dumps({"session_id": session_id}))
    print(f"[WS] New session {session_id[:8]}...")

    try:
        async for message in websocket:
            # Text message = control command
            if isinstance(message, str):
                data = json.loads(message)
                if data.get('type') == 'end':
                    break
                continue

            # Binary message = process request
            start = time.time()

            session = sessions[session_id]
            session['last_access'] = time.time()

            # Unpack: [4B header_len][JSON header][tensor data]
            header_len = struct.unpack('>I', message[:4])[0]
            header = json.loads(message[4:4 + header_len])
            tensor_data = message[4 + header_len:]

            hs_shape = header['hidden_shape']
            pe_shape = header['pe_shape']
            is_prompt = header.get('is_prompt', True)
            crop_to = header.get('crop_to')
            relocate = header.get('relocate')
            has_mask = header.get('has_mask', False)
            mask_shape = header.get('mask_shape')
            diagnostic_state = header.get('diagnostic_state', False)
            obf_epoch = header.get('obf_epoch')  # [ER] absent = no ratchet
            obf_version = header.get('obf_ratchet_version')
            validate_ratchet_header(obf_epoch, obf_version,
                                    header.get('w_hash'))

            # Reconstruct hidden states
            hs_size = 1
            for d in hs_shape:
                hs_size *= d
            hs_bytes = hs_size * 2  # float16

            hidden = torch.frombuffer(
                bytearray(tensor_data[:hs_bytes]), dtype=torch.float16
            ).reshape(hs_shape).clone().cuda()
            if hidden.dim() == 2:
                hidden = hidden.unsqueeze(0)

            # Reconstruct position embeddings
            pe_size = 1
            for d in pe_shape:
                pe_size *= d
            pe_byte_count = pe_size * 2

            pe_end = hs_bytes + pe_byte_count * 2  # cos + sin
            cos = torch.frombuffer(
                bytearray(tensor_data[hs_bytes:hs_bytes + pe_byte_count]),
                dtype=torch.float16
            ).reshape(pe_shape).clone().cuda()
            sin = torch.frombuffer(
                bytearray(tensor_data[hs_bytes + pe_byte_count:hs_bytes + pe_byte_count * 2]),
                dtype=torch.float16
            ).reshape(pe_shape).clone().cuda()
            position_embeddings = (cos, sin)

            # Reconstruct custom attention mask if provided
            custom_mask = None
            if has_mask and mask_shape:
                mask_data = tensor_data[pe_end:]
                custom_mask = torch.frombuffer(
                    bytearray(mask_data), dtype=torch.float16
                ).reshape(mask_shape).clone().cuda()

            seq_len = hidden.shape[1]

            # Per-session request counter (0 = prefill) for capture metadata
            req_seq = session.get('req_seq', 0)
            session['req_seq'] = req_seq + 1

            # Process through layers
            hidden = process_tensors(
                session, hidden, position_embeddings, seq_len, is_prompt,
                crop_to, relocate=relocate, custom_mask=custom_mask,
                session_id=session_id, request_seq=req_seq, obf_epoch=obf_epoch
            )

            # Pack response: [4B header_len][JSON header][tensor bytes]
            out_np = hidden.cpu().numpy()
            out_bytes = out_np.tobytes()

            resp_meta = {
                "hidden_shape": list(out_np.shape),
                "process_time_ms": round((time.time() - start) * 1000, 2),
                "cached_seq_len": session['seq_len']
            }
            if diagnostic_state:
                resp_meta["sampled_state_sha256"] = sampled_cache_state_hash(
                    session['cache']
                )
            resp_header = json.dumps(resp_meta).encode()

            resp = struct.pack('>I', len(resp_header)) + resp_header + out_bytes
            await websocket.send(resp)

    except websockets.exceptions.ConnectionClosed:
        print(f"[WS] Connection closed for session {session_id[:8]}")
    except Exception as e:
        print(f"[WS] Error in session {session_id[:8]}: {e}")
        traceback.print_exc()
    finally:
        if session_id in sessions:
            del sessions[session_id]
        print(f"[WS] Session {session_id[:8]} ended")


async def start_ws_server(port=5001):
    async with websockets.serve(
        ws_handler, "0.0.0.0", port,
        max_size=20 * 1024 * 1024,  # 20MB max message
        ping_interval=30,
        ping_timeout=60,
    ):
        print(f"WebSocket server running on port {port}")
        await asyncio.Future()  # run forever


# === HTTP/Flask Handlers (health checks, backward compat) ===

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "server_instance_id": os.getenv("DTRAINING_SERVER_INSTANCE_ID",
                                        socket.gethostname()),
        "gpu": torch.cuda.get_device_name(0),
        "memory_used_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
        "layers": f"{CLOUD_START}-{CLOUD_END}",
        "active_sessions": len(sessions),
        "kv_cache": True,
        "serialization": "websocket+binary",
        "jacobi_support": True
    })


@app.route('/new_session', methods=['POST'])
def new_session():
    """Create a new session with fresh KV cache (HTTP fallback)."""
    cleanup_old_sessions()
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        'cache': None,
        'last_access': time.time(),
        'seq_len': 0
    }
    return jsonify({"session_id": session_id})


@app.route('/process', methods=['POST'])
def process():
    """Process hidden states through cloud layers (HTTP fallback)."""
    try:
        start = time.time()
        data = request.json

        session_id = data.get('session_id')
        is_prompt = data.get('is_prompt', True)
        crop_to = data.get('crop_to')
        obf_epoch = data.get('obf_epoch')  # [ER] absent = no ratchet
        obf_version = data.get('obf_ratchet_version')
        validate_ratchet_header(obf_epoch, obf_version,
                                header.get('w_hash'))

        if session_id not in sessions:
            return jsonify({"error": "Invalid session_id. Call /new_session first."}), 400

        session = sessions[session_id]
        session['last_access'] = time.time()

        # Decode hidden states
        hs_shape = data['hidden_shape']
        hidden = torch.frombuffer(
            base64.b64decode(data['hidden_states']),
            dtype=torch.float16
        ).reshape(hs_shape).clone().cuda()
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(0)

        # Decode position embeddings
        pe_shape = data['pe_shape']
        pe_raw = base64.b64decode(data['position_embeddings'])
        pe_size = 1
        for d in pe_shape:
            pe_size *= d
        pe_bytes = pe_size * 2
        cos = torch.frombuffer(
            bytearray(pe_raw[:pe_bytes]), dtype=torch.float16
        ).reshape(pe_shape).clone().cuda()
        sin = torch.frombuffer(
            bytearray(pe_raw[pe_bytes:]), dtype=torch.float16
        ).reshape(pe_shape).clone().cuda()
        position_embeddings = (cos, sin)

        seq_len = hidden.shape[1]

        # Per-session request counter (0 = prefill) for capture metadata
        req_seq = session.get('req_seq', 0)
        session['req_seq'] = req_seq + 1

        hidden = process_tensors(
            session, hidden, position_embeddings, seq_len, is_prompt, crop_to,
            session_id=session_id, request_seq=req_seq, obf_epoch=obf_epoch
        )

        out_tensor = hidden.cpu()
        out_bytes = base64.b64encode(out_tensor.numpy().tobytes()).decode()

        return jsonify({
            "hidden_states": out_bytes,
            "hidden_shape": list(out_tensor.shape),
            "process_time_ms": round((time.time() - start) * 1000, 2),
            "cached_seq_len": session['seq_len']
        })

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route('/end_session', methods=['POST'])
def end_session():
    """Clean up a session (HTTP fallback)."""
    data = request.json
    session_id = data.get('session_id')
    if session_id in sessions:
        del sessions[session_id]
        return jsonify({"status": "session ended"})
    return jsonify({"status": "session not found"})


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Split Inference Cloud Server')
    parser.add_argument('--model', default="models/mistral-7b-instruct",
                        help='Path to model')
    parser.add_argument('--cloud-start', type=int, default=None,
                        help='First cloud layer (default: 2)')
    parser.add_argument('--cloud-end', type=int, default=None,
                        help='Last cloud layer (default: num_layers - 3)')
    parser.add_argument('--http-port', type=int, default=5000,
                        help='HTTP health check port')
    parser.add_argument('--ws-port', type=int, default=5001,
                        help='WebSocket port')
    parser.add_argument('--deobfuscate-key', default=None,
                        help='[E9] path to the boundary key W: unwrap wire '
                             'tensors on receipt and re-wrap on return '
                             '(shared-key / transport-encryption threat model)')
    parser.add_argument('--obfuscate-ratchet-tokens', type=int, default=0,
                        help='[ER] served tokens per epoch (0=off); the local '
                             'node counts and transmits the epoch — accepted '
                             'here for flag parity')
    parser.add_argument('--obfuscate-seed-base', type=int, default=None,
                        help='[ER] 128-bit master S for the portable v2 ratchet; '
                             'W_t derived on demand per request '
                             'epoch, no key file needed')
    parser.add_argument('--obfuscate-budget-events', type=int, default=0,
                        help='[ER] evidence budget for budget-triggered '
                             'rotation (0=off); counted on the local node')
    parser.add_argument('--obfuscate-ratchet-version', choices=['v1', 'v2'],
                        default='v2',
                        help='must match the local node; v2 is production')
    args = parser.parse_args()

    if args.obfuscate_ratchet_tokens and args.obfuscate_budget_events:
        parser.error("choose only one of --obfuscate-ratchet-tokens and --obfuscate-budget-events")

    load_model(args.model, cloud_start=args.cloud_start, cloud_end=args.cloud_end)
    if args.deobfuscate_key:
        load_deobfuscation(args.deobfuscate_key)
    enable_ratchet(args.obfuscate_seed_base,
                   ratchet_tokens=args.obfuscate_ratchet_tokens,
                   budget_events=args.obfuscate_budget_events,
                   version=args.obfuscate_ratchet_version)

    # Start WebSocket server in background thread
    def run_ws():
        asyncio.run(start_ws_server(port=args.ws_port))

    ws_thread = threading.Thread(target=run_ws, daemon=True)
    ws_thread.start()

    # Start Flask HTTP server (main thread)
    print(f"Starting HTTP health server on port {args.http_port}...")
    app.run(host='0.0.0.0', port=args.http_port, threaded=True)
