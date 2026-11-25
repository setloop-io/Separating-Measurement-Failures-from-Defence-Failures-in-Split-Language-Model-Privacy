#!/usr/bin/env python3
"""E-K: fallback-versus-FLA model and linear-attention timing benchmark.

Run once in the historical fallback image and once in the optimized image.
The artifacts remain separate and are joined only by the analysis script; this
prevents accidental pooling of different numerical/kernel paths.
"""

import argparse
import json
import os
import platform
import socket
import subprocess
import time
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None


def availability():
    found = {}
    for name in ("fla", "causal_conv1d"):
        try:
            mod = __import__(name)
            found[name] = {"available": True,
                           "version": getattr(mod, "__version__", "unknown")}
        except Exception as exc:
            found[name] = {"available": False, "error": str(exc)}
    return found


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    idx = (len(values) - 1) * q
    lo, hi = int(idx), min(int(idx) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (idx - lo)


def git_head():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def run(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    avail = availability()
    if args.implementation == "optimized":
        missing = [k for k, v in avail.items() if not v["available"]]
        if missing:
            raise RuntimeError("optimized arm requested but unavailable: "
                               + ", ".join(missing))
    dtype = getattr(torch, args.dtype)
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, device_map=args.device,
        local_files_only=True).eval()
    ids = tok(args.prompt, return_tensors="pt").input_ids.to(args.device)
    ids = ids[:, :args.seq_len]
    if ids.shape[1] < args.seq_len:
        ids = ids.repeat(1, (args.seq_len + ids.shape[1] - 1) // ids.shape[1])
        ids = ids[:, :args.seq_len]
    linear_names = []
    needles = ("deltanet", "linearattention", "gateddelta", "causalconv")
    for name, module in model.named_modules():
        cname = module.__class__.__name__.lower()
        if any(x in cname for x in needles):
            linear_names.append({"name": name, "class": module.__class__.__name__})

    def once():
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(input_ids=ids, use_cache=False).logits
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        return (time.perf_counter() - t0) * 1000, out.float().cpu()

    for _ in range(args.warmup):
        once()
    timings, reference = [], None
    determinism = []
    for _ in range(args.repetitions):
        ms, logits = once()
        timings.append(ms)
        if reference is None:
            reference = logits
        else:
            determinism.append(float((logits - reference).abs().max()))
    result = {
        "schema": "dtraining.fla_kernel_benchmark.v1",
        "experiment": "E-K FLA/causal-conv benchmark",
        "config": {"measurement_kind": "measured",
                   "model": args.model, "implementation": args.implementation,
                   "dtype": args.dtype, "device": args.device,
                   "seq_len": args.seq_len, "warmup": args.warmup,
                   "repetitions": args.repetitions},
        "evidence_status": ("fallback_baseline" if args.implementation == "fallback"
                            else "supporting"),
        "known_limitations": [
            "Each implementation is measured in a separate environment; compare only artifacts with identical model, dtype, sequence length, and hardware.",
            "Full-model timing identifies end-to-end kernel impact; named linear-attention modules document path coverage but are not independently timed."
        ],
        "backend_availability": avail,
        "linear_attention_modules": linear_names,
        "timing_ms": {"raw": timings, "median": percentile(timings, 0.5),
                      "p05": percentile(timings, 0.05),
                      "p95": percentile(timings, 0.95)},
        "determinism_max_abs": max(determinism, default=0.0),
        "provenance": {"dtraining_commit": git_head(),
                       "hostname": socket.gethostname(),
                       "platform": platform.platform(),
                       "torch": torch.__version__,
                       "cuda": torch.version.cuda,
                       "gpu": (torch.cuda.get_device_name(0)
                               if torch.cuda.is_available() else "cpu"),
                       "container_image": os.getenv("CONTAINER_IMAGE", "unknown")},
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model")
    ap.add_argument("--implementation", choices=["fallback", "optimized"])
    ap.add_argument("--dtype", choices=["float16", "bfloat16", "float32"],
                    default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--repetitions", type=int, default=20)
    ap.add_argument("--prompt", default="Explain distributed inference over a WAN.")
    ap.add_argument("--output", default="fla_kernel_benchmark.json")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        assert percentile([3, 1, 2], .5) == 2
        assert set(availability()) == {"fla", "causal_conv1d"}
        print("SELF-TEST PASSED"); return
    if torch is None:
        ap.error("torch is required; --help and --self-test work without it")
    if not args.model or not args.implementation:
        ap.error("--model and --implementation are required")
    run(args)


if __name__ == "__main__":
    main()
