"""Same-checkpoint operator-placement scenarios on explicitly plaintext CPU loopback."""
from copy import deepcopy
from pathlib import Path
import json
import math
import time

import torch
import yaml

from pcr.model import BackboneConfig
from pcr.train import build_initialized_model
from pcr.utils import derive_seed, parameters_checksum
from .experiment import write_json
from .routing import POLICIES, Router, install_router, generate
from .scenario_io import Campaign, require_engineering, validate_seed_namespace


def validate_config(cfg):
    require_engineering(cfg, "routing")
    validate_seed_namespace(cfg["seeds"], "routing_engineering")
    if cfg.get("backend") != "plaintext_loopback" or cfg.get("hardware_attested") is not False:
        raise ValueError("Real encryption/attestation is not implemented; no silent emulation")
    if cfg.get("ciphertext_only") is not False or cfg.get("kv_cache") != "none_full_prefix_recompute":
        raise ValueError("This backend exposes plaintext and has no KV cache")
    if cfg["policies"] != list(POLICIES):
        raise ValueError("Need the three declared placement policies including the local control")
    if (not cfg["prompt_lengths"] or len(set(cfg["prompt_lengths"])) != len(cfg["prompt_lengths"])
            or any(type(n) is not int or n < 1 for n in cfg["prompt_lengths"])
            or type(cfg["new_tokens"]) is not int or cfg["new_tokens"] < 2
            or max(cfg["prompt_lengths"]) + cfg["new_tokens"] - 1 > cfg["backbone"]["max_seq_len"]):
        raise ValueError("Prompt/generation budget must exercise both phases within model context")
    if not math.isfinite(cfg["logit_atol"]) or cfg["logit_atol"] < 0:
        raise ValueError("Invalid correctness tolerance")
    return len(cfg["seeds"]) * len(cfg["prompt_lengths"]) * len(cfg["policies"])


def run(config_path, label=None, validate_only=False):
    cfg = yaml.safe_load(Path(config_path).read_text())
    count = validate_config(cfg)
    if validate_only:
        return {"scenario_count": count, "hardware_attested": False,
                "ciphertext_only": False, "scientific_execution_enabled": False}
    campaign = Campaign(cfg, config_path, label)
    for seed in cfg["seeds"]:
        with torch.random.fork_rng(devices=[]):
            base = build_initialized_model(BackboneConfig(**cfg["backbone"]), "all_local", None,
                                           seed, f"p4-routing-{seed}").eval()
        checksum = parameters_checksum(base.parameters())
        for length in cfg["prompt_lengths"]:
            g = torch.Generator().manual_seed(derive_seed("p4-routing-input", seed, length))
            prompt = torch.randint(0, 256, (1, length), generator=g)
            expected_tokens, expected_logits = generate(base, prompt, cfg["new_tokens"])
            for policy in cfg["policies"]:
                ident = f"{policy}-p{length}-s{seed}"
                inputs = {"model_sha256": checksum, "seed": seed, "prompt_bytes": prompt.tolist(),
                          "input_source": "seeded synthetic bytes; no trained research checkpoint"}
                out, manifest = campaign.start(ident, inputs)
                error = None
                try:
                    router = Router(policy)
                    model = install_router(deepcopy(base), router)
                    start = time.perf_counter()
                    tokens, logits = generate(model, prompt, cfg["new_tokens"], router)
                    elapsed = time.perf_counter() - start
                    delta = float((logits - expected_logits).abs().max())
                    equal = torch.equal(tokens, expected_tokens) and delta <= cfg["logit_atol"]
                    n_layers = cfg["backbone"]["n_layers"]
                    expected_calls = 4 * n_layers * (0 if policy == "local_reference" else
                        cfg["new_tokens"] if policy == "delegate_all_phases" else cfg["new_tokens"] - 1)
                    actual_calls = sum(e["delegated"] for e in router.events)
                    result = {"policy": policy, "seed": seed, "prompt_length": length,
                        "new_tokens": cfg["new_tokens"], "model_sha256": checksum,
                        "backend": "plaintext_loopback", "hardware_attested": False,
                        "ciphertext_only": False, "kv_cache": cfg["kv_cache"],
                        "scientific_evidence": False, "privacy_outcome": "undetermined",
                        "maximum_logit_error": delta, "correctness_passed": equal,
                        "generated_bytes": tokens.tolist(), "delegated_calls": actual_calls,
                        "expected_delegated_calls": expected_calls,
                        "weight_setup_serialized_bytes": router.backend.setup_bytes,
                        "online_serialized_bytes": sum(e["request_bytes"] + e["response_bytes"] for e in router.events),
                        "local_diagnostic_seconds": elapsed,
                        "timing_scope": "one CPU loopback run including lazy weight provisioning; not network/TEE/FHE latency",
                        "active_attack_qualified": False,
                        "limitations": ["plaintext exposed", "no hardware isolation", "no cryptography",
                            "no authenticated transport", "no KV cache", "no backward path",
                            "no confidential parameter updates", "no active/security qualification"]}
                    write_json(out / "metrics.json", result)
                    write_json(out / "trace.json", router.events)
                    if not equal or actual_calls != expected_calls:
                        raise RuntimeError("Placement correctness or call-count check failed")
                    if parameters_checksum(base.parameters()) != checksum:
                        raise RuntimeError("Reference checkpoint mutated")
                except Exception as exc:
                    error = exc
                campaign.finish(out, manifest, error)
    return campaign.summarize({"expected_scenarios": count, "hardware_attested": False,
                               "ciphertext_only": False, "kv_cache": cfg["kv_cache"]})
