#!/usr/bin/env python3
"""
Defense Experiment for Split Inference Paper (§5.3)

Measures the two defenses proposed-but-not-measured in the paper —
activation quantization (int8 / fp8) and calibrated additive noise —
against a stronger MLP inversion attacker, plus their utility cost on
greedy generation.

Experiment:
1. Build a larger, sequence-disjoint corpus: the 15 TEXT_SAMPLES from
   inversion_experiment.py + optional --corpus-file documents + synthetic
   text greedily generated from the model itself (~40 seed prompts).
   Target >= 10K (activation, token) pairs. Train/val/test split 70/15/15
   BY DOCUMENT (no document spans splits).
2. For split depths 2, 4, 8 local layers, collect hidden states at the
   split point, apply each defense to the fp16 activation tensor, and
   train the repo's InversionDecoder MLP on the defended activations
   (adaptive attacker, 3 seeds, mean +/- std top-1/top-5).
3. Output fidelity: greedy generation (64 tokens, one prompt each of
   code/structured/creative/conversational) with the defense applied at
   the split point, vs undefended greedy: token-identity rate,
   first-divergence position, mean KL over matched positions.

Runs on ONE GPU node with a local model copy (no cloud, no network).

Expected runtime on a GB10-class GPU (Mistral-7B fp16, defaults):
    corpus build (generation + activation collection)   ~5 min
    inversion grid: 3 depths x 7 defenses x 3 seeds     ~25-35 min
    fidelity grid: 3 depths x 7 defenses x 4 prompts    ~5-8 min
    total                                               ~35-45 min
--quick (depth 2, none + int8 + noise-1%, 1 seed, 1 prompt, 16 tokens):
    ~3-5 min.

Usage:
    python defense_experiment.py --model /path/to/mistral-7b --quick
    python defense_experiment.py --model /path/to/mistral-7b
    python defense_experiment.py --model /path/to/mistral-7b --corpus-file extra_corpus.txt
"""

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime

# torch / transformers are imported lazily-guarded so that `--help` works on
# hosts without them (e.g. a torch-less Mac used only for editing).
try:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover - torch-less host
    np = None
    torch = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # pragma: no cover
    AutoModelForCausalLM = None
    AutoTokenizer = None

MODEL_PATH = "models/mistral-7b-instruct"
DEVICE = "cuda"
DATA_DIR = "experiment_data"

# Training status reporting for the spark dashboard (best-effort, atomic).
STATUS_FILE = os.environ.get(
    "TRAINING_STATUS_FILE", "/home/geo/experiments/results/training_status.json")
_status_state = {}
_status_last_write = 0.0


def _write_training_status(**fields):
    """Merge fields into the dashboard training status JSON.

    Atomic (tmp + os.replace), throttled to one write per 5s unless the new
    state is done/failed, and never fatal to the experiment."""
    global _status_last_write
    now = time.time()
    force = fields.get("state") in ("done", "failed")
    if not force and now - _status_last_write < 5:
        _status_state.update(fields)
        return
    _status_last_write = now
    _status_state.update(fields)
    _status_state["updated"] = datetime.now().astimezone().isoformat()
    try:
        os.makedirs(os.path.dirname(STATUS_FILE), exist_ok=True)
        tmp_path = STATUS_FILE + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(_status_state, f)
        os.replace(tmp_path, STATUS_FILE)
    except Exception:
        pass

SPLIT_DEPTHS = [2, 4, 8]
NOISE_SIGMAS = [0.005, 0.01, 0.02, 0.05]  # fraction of per-token activation RMS
MAX_NEW_TOKENS = 64

# Fidelity prompts: one per category, reused from lookahead_ablation.py /
# perplexity_comparison.py (first prompt of each category).
FIDELITY_PROMPTS = {
    "code": "Write a Python function to implement binary search on a sorted array. Include docstring and type hints.",
    "structured": "List the top 10 largest countries by area with their capitals and populations in a formatted table.",
    "creative": "Write a short story about a robot that discovers it can dream. Make it emotional and surprising.",
    "conversational": "What are the main differences between Python and Rust? When should I choose one over the other?",
}

# ~40 diverse seed prompts for synthetic corpus generation (greedy, temp 0).
SEED_PROMPTS = [
    "The history of the Roman Empire began",
    "In organic chemistry, a functional group is",
    "The recipe for a classic margherita pizza requires",
    "To debug a segmentation fault in C, first",
    "The weather patterns in the Pacific Northwest are",
    "A balanced investment portfolio typically includes",
    "The plot of the novel opens with",
    "Photosynthesis in desert plants differs because",
    "The rules of chess state that a pawn",
    "Modern microprocessors improve performance by",
    "The committee meeting agenda for Tuesday covers",
    "To train for a marathon in sixteen weeks",
    "The theory of plate tectonics explains",
    "A REST API endpoint should return",
    "The symptoms of seasonal allergies include",
    "In macroeconomics, quantitative easing refers to",
    "The migration route of monarch butterflies",
    "To replace a bathroom faucet, you will need",
    "The SQL query optimizer chooses an execution plan",
    "Ancient Egyptian hieroglyphs were deciphered",
    "The guitar chord progression for this song",
    "A load balancer distributes incoming traffic",
    "The greenhouse effect warms the planet because",
    "When writing a cover letter, always",
    "The neural network architecture consists of",
    "Sourdough bread fermentation depends on",
    "The Supreme Court ruling established that",
    "To improve your credit score, you should",
    "The film's cinematography uses long takes to",
    "Compilers translate source code into",
    "The ecosystem of a coral reef supports",
    "A 401(k) retirement account allows",
    "The painting technique known as chiaroscuro",
    "To winterize your home before the freeze",
    "The supply chain disruption caused",
    "Machine learning models can overfit when",
    "The itinerary for our trip to Japan includes",
    "Antibiotic resistance develops when bacteria",
    "The contract clause specifies that either party",
    "To set up a home Wi-Fi mesh network",
    "The philosophy of stoicism teaches that",
]


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def _model_dims(cfg):
    """(hidden_size, vocab_size) from a model config, tolerating nested
    multimodal configs (e.g. qwen3_5 wraps text params in text_config)."""
    tc = getattr(cfg, "text_config", None)
    hidden = getattr(cfg, "hidden_size", None) or getattr(tc, "hidden_size", None)
    vocab = getattr(cfg, "vocab_size", None) or getattr(tc, "vocab_size", None)
    return hidden, vocab


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Defenses (pure torch functions on the fp16 activation tensor)
# ---------------------------------------------------------------------------

def defense_none(x):
    return x


def defense_int8(x):
    """Per-token symmetric int8 quantization: scale = absmax/127 per token
    vector, then dequantize. Returns same dtype as input."""
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
    q = torch.round(x / scale).clamp(-127, 127)
    return (q * scale).to(x.dtype)


def defense_fp8(x):
    """Simulated e4m3 (fn) quantization + dequantize."""
    if hasattr(torch, "float8_e4m3fn"):
        return x.float().to(torch.float8_e4m3fn).to(x.dtype)
    # Manual fallback: round mantissa to 3 bits, clamp exponent to e4m3 range.
    xf = x.float()
    max_val = 448.0  # e4m3fn max normal
    xf = xf.clamp(-max_val, max_val)
    mantissa, exponent = torch.frexp(xf)  # xf = mantissa * 2^exponent, |m| in [0.5, 1)
    # 3 mantissa bits -> quantize mantissa to steps of 2^-3
    m_q = torch.round(mantissa * 8.0) / 8.0
    # handle mantissa rounding overflow (1.0 -> 0.5 * 2^(e+1))
    overflow = m_q >= 1.0
    m_q = torch.where(overflow, m_q * 0.5, m_q)
    exponent = torch.where(overflow, exponent + 1, exponent)
    out = torch.ldexp(m_q, exponent)
    return out.clamp(-max_val, max_val).to(x.dtype)


def make_noise_defense(sigma_frac):
    """Additive Gaussian noise with sigma = sigma_frac * per-token activation RMS."""
    def fn(x):
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).sqrt()
        noise = torch.randn_like(x.float()) * (sigma_frac * rms)
        return (x.float() + noise).to(x.dtype)
    fn.sigma_frac = sigma_frac
    return fn


def build_defenses(quick=False):
    """Ordered dict {name: fn}."""
    defenses = {"none": defense_none}
    if quick:
        defenses["int8"] = defense_int8
        defenses["noise_1.0%"] = make_noise_defense(0.01)
        return defenses
    defenses["int8"] = defense_int8
    defenses["fp8"] = defense_fp8
    for s in NOISE_SIGMAS:
        defenses[f"noise_{s*100:.1f}%"] = make_noise_defense(s)
    return defenses


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

def generate_synthetic_docs(model, tokenizer, seed_prompts, device,
                            gen_tokens=64, rounds=3):
    """Greedy (temperature 0) continuation of each seed prompt, chained for
    `rounds` iterations to reach the >=10K (activation, token) pair target
    deterministically. Each final document = prompt + rounds*gen_tokens."""
    docs = []
    for prompt in seed_prompts:
        text = prompt
        for _ in range(rounds):
            input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(
                    input_ids, max_new_tokens=gen_tokens, do_sample=False,
                    num_beams=1, pad_token_id=tokenizer.eos_token_id,
                )
            text = tokenizer.decode(out[0], skip_special_tokens=True)
        docs.append(text)
    return docs


def split_documents(docs, seed, ratios=(0.70, 0.15, 0.15)):
    """Split by document (not token position) into train/val/test."""
    idx = list(range(len(docs)))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n = len(docs)
    n_train = int(round(ratios[0] * n))
    n_val = int(round(ratios[1] * n))
    splits = {
        "train": [docs[i] for i in idx[:n_train]],
        "val": [docs[i] for i in idx[n_train:n_train + n_val]],
        "test": [docs[i] for i in idx[n_train + n_val:]],
    }
    return splits


def split_hashes(splits):
    """SHA-256 of each split's document list, for reproducibility records."""
    hashes = {}
    for name, docs in splits.items():
        h = hashlib.sha256()
        for d in docs:
            h.update(d.encode("utf-8"))
            h.update(b"\x00")
        hashes[name] = {"sha256": h.hexdigest(), "num_docs": len(docs)}
    return hashes


# ---------------------------------------------------------------------------
# Activation collection
# ---------------------------------------------------------------------------

def collect_activations(model, tokenizer, texts, device, split_layer):
    """Collect (activation, token ID) pairs from the output of layer
    `split_layer` (0-indexed), matching inversion_experiment.py's hook
    methodology."""
    model.eval()
    all_activations = []
    all_token_ids = []

    captured = {}

    def hook_fn(module, input, output):
        hidden = output[0]
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(0)
        captured['hidden'] = hidden.detach()

    handle = model.model.layers[split_layer].register_forward_hook(hook_fn)

    with torch.no_grad():
        for text in texts:
            input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
            tokens = input_ids[0]
            _ = model(input_ids)
            hidden = captured['hidden'][0]  # [seq_len, hidden_dim]
            for i in range(len(tokens)):
                all_activations.append(hidden[i].cpu())
                all_token_ids.append(tokens[i].item())

    handle.remove()
    return torch.stack(all_activations), torch.tensor(all_token_ids, dtype=torch.long)


# ---------------------------------------------------------------------------
# Inversion attack (repo's InversionDecoder architecture)
# ---------------------------------------------------------------------------

if torch is not None:
    class InversionDecoder(nn.Module):
        """Small MLP that predicts token IDs from split-point activations.
        Architecture identical to inversion_experiment.InversionDecoder."""

        def __init__(self, hidden_dim, vocab_size, intermediate_dim=2048):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(hidden_dim, intermediate_dim),
                nn.GELU(),
                nn.LayerNorm(intermediate_dim),
                nn.Linear(intermediate_dim, intermediate_dim),
                nn.GELU(),
                nn.LayerNorm(intermediate_dim),
                nn.Linear(intermediate_dim, vocab_size),
            )

        def forward(self, x):
            return self.net(x)


def evaluate_decoder(decoder, activations, token_ids, device):
    decoder.eval()
    act = activations.float().to(device)
    tok = token_ids.to(device)
    with torch.no_grad():
        logits = decoder(act)
        preds = logits.argmax(dim=-1)
        top1 = (preds == tok).float().mean().item() * 100
        top5_preds = logits.topk(5, dim=-1).indices
        top5 = (top5_preds == tok.unsqueeze(1)).any(dim=1).float().mean().item() * 100
    return round(top1, 2), round(top5, 2)


def train_decoder(train_act, train_tok, val_act, val_tok, test_act, test_tok,
                  hidden_dim, vocab_size, epochs=50, batch_size=256, lr=1e-3):
    """Train inversion decoder; select best epoch by val top-1, report test."""
    device = DEVICE if torch.cuda.is_available() else "cpu"

    train_ds = TensorDataset(train_act.float().to(device), train_tok.to(device))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    decoder = InversionDecoder(hidden_dim, vocab_size).to(device)
    optimizer = optim.AdamW(decoder.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val, best_state = -1.0, None
    for epoch in range(epochs):
        decoder.train()
        total_loss = 0.0
        for batch_act, batch_tok in train_dl:
            loss = criterion(decoder(batch_act), batch_tok)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        val_top1, _ = evaluate_decoder(decoder, val_act, val_tok, device)
        _write_training_status(
            phase="train", state="running", epoch=epoch + 1, epochs=epochs,
            loss=round(total_loss / max(len(train_dl), 1), 4),
            top1=val_top1, metric_name="val_top1", metric_value=val_top1)
        if val_top1 > best_val:
            best_val = val_top1
            best_state = {k: v.detach().clone() for k, v in decoder.state_dict().items()}

    if best_state is not None:
        decoder.load_state_dict(best_state)
    top1, top5 = evaluate_decoder(decoder, test_act, test_tok, device)
    return {"top1_accuracy": top1, "top5_accuracy": top5, "best_val_top1": best_val}


# ---------------------------------------------------------------------------
# Output fidelity
# ---------------------------------------------------------------------------

def greedy_generate_with_defense(model, tokenizer, prompt, device, split_layer,
                                 defense_fn, max_new_tokens):
    """Greedy generation with `defense_fn` applied to the hidden states at the
    output of layer `split_layer` (simulating the defended WAN hop). Returns
    (token_ids, per-step next-token log-distributions)."""
    model.eval()

    def hook_fn(module, input, output):
        hidden = output[0]
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(0)
        defended = defense_fn(hidden)
        if isinstance(output, tuple):
            return (defended,) + output[1:]
        return defended

    handle = None
    if defense_fn is not None:
        handle = model.model.layers[split_layer].register_forward_hook(hook_fn)

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    logprobs = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(input_ids)
            logits = out.logits[0, -1].float()
            logprobs.append(torch.log_softmax(logits, dim=-1).cpu())
            next_tok = logits.argmax().view(1, 1)
            input_ids = torch.cat([input_ids, next_tok], dim=1)
            if next_tok.item() == tokenizer.eos_token_id:
                break

    if handle is not None:
        handle.remove()
    gen_ids = input_ids[0, input_ids.shape[1] - len(logprobs):]
    return gen_ids.cpu(), logprobs


def fidelity_metrics(base_ids, base_lps, def_ids, def_lps):
    """Token-identity rate, first-divergence position, mean KL over matched
    (pre-divergence) positions."""
    n = min(len(base_ids), len(def_ids))
    if n == 0:
        return {"token_identity_rate": 0.0, "first_divergence": 0, "mean_kl": None}
    matches = (base_ids[:n] == def_ids[:n])
    identity = matches.float().mean().item()
    div = next((i for i in range(n) if not matches[i]), None)
    first_div = div if div is not None else -1  # -1 = never diverged
    matched = n if div is None else div
    mean_kl = None
    if matched > 0:
        kls = []
        for i in range(matched):
            p = base_lps[i].exp()
            kl = (p * (base_lps[i] - def_lps[i])).sum().item()
            kls.append(kl)
        mean_kl = round(float(np.mean(kls)), 6)
    return {
        "token_identity_rate": round(identity, 4),
        "first_divergence": first_div,
        "mean_kl": mean_kl,
        "tokens_compared": n,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Defense experiment: quantization/noise vs inversion attack (paper §5.3)")
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--corpus-file", default=None,
                        help="Plain text, one document per line; added to the corpus")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--output", default=os.path.join(DATA_DIR, "defense_results.json"))
    parser.add_argument("--quick", action="store_true",
                        help="Smoke test: depth 2 only, none+int8+noise-1%%, 1 seed, "
                             "1 prompt, 16 tokens")
    args = parser.parse_args()

    if torch is None or AutoModelForCausalLM is None:
        sys.exit("torch/transformers are required to run the experiment "
                 "(this host can only show --help).")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    model_name = os.path.basename(args.model.rstrip("/")) or args.model
    _write_training_status(
        run_id=f"defense_{int(time.time())}", script="defense_experiment.py",
        model=model_name, phase="collect", state="running", epoch=0,
        epochs=args.epochs, extra={})

    depths = [2] if args.quick else SPLIT_DEPTHS
    defenses = build_defenses(quick=args.quick)
    n_seeds = 1 if args.quick else args.seeds
    max_new = 16 if args.quick else MAX_NEW_TOKENS
    fidelity_prompts = ({"code": FIDELITY_PROMPTS["code"]} if args.quick
                        else FIDELITY_PROMPTS)
    base_seed = 1234

    seed_all(base_seed)

    print("=" * 60)
    print("Loading model:", args.model)
    print("=" * 60)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map=DEVICE)
    model.eval()
    hidden_dim, vocab_size = _model_dims(model.config)
    print(f"Hidden dim: {hidden_dim}, Vocab size: {vocab_size}")

    # ---------------- Corpus ----------------
    from inversion_experiment import TEXT_SAMPLES

    docs = list(TEXT_SAMPLES)
    if args.corpus_file:
        with open(args.corpus_file) as f:
            docs += [ln.strip() for ln in f if ln.strip()]
        print(f"Loaded {len(docs) - len(TEXT_SAMPLES)} documents from {args.corpus_file}")

    print(f"Generating synthetic corpus from {len(SEED_PROMPTS)} seed prompts "
          f"(greedy, {MAX_NEW_TOKENS} tokens x 3 rounds)...")
    synth = generate_synthetic_docs(model, tokenizer, SEED_PROMPTS, DEVICE)
    docs += synth
    print(f"Corpus: {len(TEXT_SAMPLES)} repo samples + {len(synth)} synthetic "
          f"= {len(docs)} documents")

    splits = split_documents(docs, seed=base_seed)
    hashes = split_hashes(splits)
    print({k: v["num_docs"] for k, v in hashes.items()})

    results = {
        "config": {
            "model": args.model,
            "split_depths": depths,
            "defenses": list(defenses.keys()),
            "noise_sigmas": NOISE_SIGMAS if not args.quick else [0.01],
            "epochs": args.epochs,
            "seeds": n_seeds,
            "max_new_tokens": max_new,
            "quick": args.quick,
            "split_ratios": "70/15/15 by document",
            "split_hashes": hashes,
        },
        "git_commit": git_commit(),
        "inversion": {},
        "fidelity": {},
    }

    # ---------------- Inversion grid ----------------
    for depth in depths:
        split_layer = depth - 1  # activations after `depth` local layers
        print("\n" + "=" * 60)
        print(f"Depth {depth}: collecting activations after layer {split_layer}")
        print("=" * 60)
        raw = {}
        for name in ("train", "val", "test"):
            act, tok = collect_activations(model, tokenizer, splits[name], DEVICE, split_layer)
            raw[name] = (act, tok)
            print(f"  {name}: {act.shape[0]} pairs")
        total_pairs = sum(raw[k][0].shape[0] for k in raw)
        print(f"  total pairs: {total_pairs} (target >= 10K)")
        results["config"].setdefault("pairs_per_depth", {})[str(depth)] = total_pairs

        for def_name, def_fn in defenses.items():
            defended = {k: (def_fn(a.float()).float(), t) for k, (a, t) in raw.items()}
            seed_metrics = []
            for s in range(n_seeds):
                seed_all(base_seed + s)
                _write_training_status(
                    run_id=f"defense_d{depth}_{def_name}_seed{s}",
                    phase="train", state="running", epoch=0, epochs=args.epochs)
                m = train_decoder(
                    *defended["train"], *defended["val"], *defended["test"],
                    hidden_dim, vocab_size, epochs=args.epochs)
                _write_training_status(
                    state="done", epoch=args.epochs,
                    top1=m["top1_accuracy"], top5=m["top5_accuracy"],
                    metric_name="top1_accuracy", metric_value=m["top1_accuracy"])
                seed_metrics.append(m)
                print(f"  depth={depth} defense={def_name} seed={s}: "
                      f"top-1={m['top1_accuracy']:.2f}% top-5={m['top5_accuracy']:.2f}%")
            t1 = [m["top1_accuracy"] for m in seed_metrics]
            t5 = [m["top5_accuracy"] for m in seed_metrics]
            results["inversion"].setdefault(str(depth), {})[def_name] = {
                "top1_mean": round(float(np.mean(t1)), 2),
                "top1_std": round(float(np.std(t1)), 2),
                "top5_mean": round(float(np.mean(t5)), 2),
                "top5_std": round(float(np.std(t5)), 2),
                "per_seed": seed_metrics,
                "random_baseline": round(100 / vocab_size, 4),
            }

    # ---------------- Fidelity grid ----------------
    _write_training_status(phase="fidelity", state="running")
    for depth in depths:
        split_layer = depth - 1
        for category, prompt in fidelity_prompts.items():
            seed_all(base_seed)
            base_ids, base_lps = greedy_generate_with_defense(
                model, tokenizer, prompt, DEVICE, split_layer, None, max_new)
            for def_name, def_fn in defenses.items():
                if def_name == "none":
                    continue
                seed_all(base_seed)
                def_ids, def_lps = greedy_generate_with_defense(
                    model, tokenizer, prompt, DEVICE, split_layer, def_fn, max_new)
                fm = fidelity_metrics(base_ids, base_lps, def_ids, def_lps)
                results["fidelity"].setdefault(str(depth), {}).setdefault(def_name, {})[category] = fm
                print(f"fidelity depth={depth} defense={def_name} {category}: "
                      f"identity={fm['token_identity_rate']:.2f} "
                      f"div@{fm['first_divergence']} kl={fm['mean_kl']}")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")
    _write_training_status(phase="done", state="done", epoch=args.epochs)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        _write_training_status(state="failed", extra={"error": str(exc)})
        raise
