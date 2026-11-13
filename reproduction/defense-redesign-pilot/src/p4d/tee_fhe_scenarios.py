"""Fleet-side TEE/FHE engineering stages; plaintext emulation and one real CKKS projection.

Protocol `p4-tee-fhe-eng-0.1` (docs/TEE_FHE_EXEC_PROTOCOL.md). Stage A restores the
hash-bound qualification d0-q8 checkpoint, runs the ordinary forward on the frozen
evaluation window prefix, then re-runs it with every linear operator of the split
model except the LM head executing through a serialized in-process fp32 loopback
(the head stays on the trusted control path; the remote surrogate's own linears are
inside the remote module and are wrapped like every other attention/FFN linear),
recording the observation contract per operator call: operator, direction, shape,
dtype, serialized bytes, and global step. The loopback roundtrip is bitwise, so the
emulated forward must match the reference exactly; the frozen tolerances
(max |Δlogit| ≤ 1e-4, |Δloss| ≤ 1e-5, 100% argmax agreement) are checked and any
failure is recorded as a failed scenario, never patched over.

Stage B measures one real CKKS encrypted projection of the boundary tensor through
the remote block's FFN up-projection weight using TenSEAL in the separate track
environment (.venv-tee); TenSEAL importability is checked before anything runs and
its absence is a hard error — no mock is ever substituted. Layout: per-token
encrypted 256-vectors (ckks_vector) times the plaintext weight matrix via
TenSEAL's vector-times-plain-matrix multiply; timings are recorded per phase per
repetition, all five repetitions retained. Every record carries
`hardware_attested: false`, `scientific_evidence: false`, privacy `undetermined`,
and timing records are explicitly not secure-serving latencies.

`full_budget: false` exists only so tests can exercise the pipeline with tiny
models, panels, and CKKS parameters; the protocol config sets it true and
validation then pins every constant.
"""
from functools import partial
from pathlib import Path
import json
import time

import jsonschema
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from pcr.model import BackboneConfig
from pcr.modes import remote_parameters
from pcr.split import SplitConfig
from pcr.train import build_initialized_model
from pcr.utils import derive_seed, parameters_checksum

from .devices import resolve_device
from .experiment import ROOT, digest, write_json
from .model import attach_defense
from .null_precision import prequant_capture
from .routing import pack_tensor, unpack_tensor
from .run_verification import verify_run
from .scenario_io import Campaign, require_engineering, validate_seed_namespace
from .scenario_verification import verify_scenario
from .secure_compute_record import (SecureComputeRecorder, artifact_record,
                                    write_secure_compute_bundle)
from .structure_precision import quantized_release
from .warmup_scenarios import FLAGS, tensor_hash, trusted_checksum

PROTOCOL_ID = "p4-tee-fhe-eng-0.1"
SEEDS = [410801]
BUDGET = {"stage_a_windows": 512, "stage_a_batch_size": 32, "logit_atol": 1e-4,
          "loss_atol": 1e-5, "stage_b_n_tokens": [1, 32, 128], "stage_b_repetitions": 5,
          "ckks_poly_modulus_degree": 8192, "ckks_coeff_mod_bit_sizes": [60, 40, 40, 60],
          "ckks_global_scale_exp": 40, "cpu_threads": 2}
TIMING_SCOPE = ("in-process engineering timing on a research node; not a secure-serving "
                "latency (no TEE, no attestation, no network)")
DEPLOYMENT_SCOPE = ("Plaintext-emulated boundary (stage A) and one real CKKS projection "
                    "(stage B); no confidentiality, integrity, or attestation claim")


def validate(cfg):
    require_engineering(cfg, "tee_fhe")
    if cfg.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Protocol ID differs from the TEE/FHE engineering protocol")
    validate_seed_namespace(cfg.get("seeds"), "tee_fhe_engineering")
    if cfg.get("seeds") != SEEDS:
        raise ValueError("The track engineering campaign declares exactly the one reserved seed")
    for key in ("checkpoint_manifest", "panels_manifest"):
        if not isinstance(cfg.get(key), str) or not cfg[key]:
            raise ValueError("The checkpoint and panels manifest bindings are required")
    full = bool(cfg.get("full_budget", False))
    if full and any(cfg.get(k) != v for k, v in BUDGET.items()):
        raise ValueError("Changed tolerances, budgets, or CKKS constants require a new protocol")
    return {"expected_scenarios": 2 * len(cfg["seeds"]), "full_budget": full,
            **FLAGS, "hardware_attested": False}


def load_checkpoint(manifest_path):
    """Verify the bound qualification run and restore its d0-q8 checkpoint."""
    record, artifacts, cfg = verify_run(manifest_path)
    if (record["status"] != "completed" or record["protocol_id"] != "p4-qualification-0.1"
            or record["cell"] != "d0-q8" or record["mode"] != "trained"):
        raise ValueError("Need the completed qualification d0-q8 trained checkpoint")
    metrics = json.loads(artifacts["metrics.json"].read_text())
    state = torch.load(artifacts["checkpoint.pt"], weights_only=True)
    cell = record["inputs"]["cell"]
    model = build_initialized_model(BackboneConfig(**cfg["backbone"]), "frozen_random",
                                    SplitConfig(**cfg["split"], q_bits=cell["q_bits"]),
                                    record["seed"], "checkpoint-restore-only")
    attach_defense(model, {"name": "none"})
    model.load_state_dict(state, strict=True)
    if (trusted_checksum(model) != metrics["trusted_final_sha256"]
            or parameters_checksum(remote_parameters(model)) != metrics["remote_final_sha256"]):
        raise ValueError("Restored model differs from the bound qualification checkpoint")
    return model, record


def bind_panels(manifest_path):
    """Bind the pilot campaign's shared panels through a pilot manifest's recorded digest."""
    record = json.loads((ROOT / manifest_path).read_text())
    schema = json.loads((ROOT / "experiments/schemas/run_manifest.schema.json").read_text())
    jsonschema.validate(record, schema)
    if (record["status"] != "completed" or record["protocol_id"] != "p4-defense-pilot-1.0"
            or record.get("cohort") != "exploratory"):
        raise ValueError("Need a completed exploratory pilot manifest for the panels binding")
    panels_sha = record["inputs"]["panels_sha256"]
    campaign = (ROOT / record["source_record"]).parent
    if digest(campaign / "panels.pt") != panels_sha:
        raise ValueError("Pilot campaign panels differ from the manifest binding")
    panels = torch.load(campaign / "panels.pt", weights_only=True)
    declarations = json.loads((campaign / "panels.json").read_text())
    for name, panel in panels.items():
        for key, expected in declarations["panels"][name]["tensor_sha256"].items():
            if tensor_hash(panel[key]) != expected:
                raise ValueError("Panel tensor mismatch")
    return panels, panels_sha


class ObservationContract:
    """Per-operator-call record of the emulated serialized boundary."""

    def __init__(self, secure_recorder=None):
        self.events = []
        self.secure_recorder = secure_recorder

    def record(self, operator, direction, tensor, packet, step):
        serialized_bytes = len(packet)
        self.events.append({"operator": operator, "direction": direction,
                            "shape": list(tensor.shape), "dtype": str(tensor.dtype),
                            "serialized_bytes": serialized_bytes, "step": step})
        if self.secure_recorder is not None:
            direction_map = {"in": "trusted_to_external", "out": "external_to_trusted"}
            self.secure_recorder.record_pair(
                request_id=f"request-{step:06d}", step=step,
                phases=["forward", "staging"], direction=direction_map[direction],
                operation=f"serialized_linear_{direction}", operator=operator,
                tensor=tensor.detach().cpu().float().contiguous(), packet=packet)

    def totals(self):
        per_operator = {}
        for event in self.events:
            key = (event["operator"], event["direction"])
            aggregate = per_operator.setdefault(key, {"calls": 0, "serialized_bytes": 0})
            aggregate["calls"] += 1
            aggregate["serialized_bytes"] += event["serialized_bytes"]
        return {"per_operator": [{"operator": key[0], "direction": key[1], **value}
                                 for key, value in sorted(per_operator.items())],
                "total_calls": len(self.events),
                "total_serialized_bytes": sum(e["serialized_bytes"] for e in self.events),
                "steps": max((e["step"] for e in self.events), default=-1) + 1}


def _loopback_pre(name, recorder, step, module, args):
    (x,) = args
    serialized = x.detach().cpu().float().contiguous()
    packet = pack_tensor(serialized)
    recorder.record(name, "in", serialized, packet, step[0])
    return (unpack_tensor(packet).to(x.device),)


def _loopback_post(name, recorder, step, module, args, output):
    serialized = output.detach().cpu().float().contiguous()
    packet = pack_tensor(serialized)
    recorder.record(name, "out", serialized, packet, step[0])


def install_loopback(model, recorder, step):
    """Wrap every nn.Linear of the split model except the LM head in the loopback."""
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name != "head":
            handles.append(module.register_forward_pre_hook(
                partial(_loopback_pre, name, recorder, step)))
            handles.append(module.register_forward_hook(
                partial(_loopback_post, name, recorder, step)))
    return handles


def stage_a_evaluate(model, tokens, next_tokens, batch_size, cfg, device,
                     secure_recorder=None):
    """Reference forward, then the serialized-loopback forward, on the frozen window prefix."""
    if not 1 <= batch_size <= len(tokens):
        raise ValueError("Invalid evaluation batch")
    was_training = model.training
    model.eval()
    recorder = ObservationContract(secure_recorder=secure_recorder)
    step = [0]
    reference, emulated = [], []
    with torch.no_grad():
        for i in range(0, len(tokens), batch_size):
            x = tokens[i:i + batch_size].to(device)
            reference.append(model(x).cpu())
        handles = install_loopback(model, recorder, step)
        try:
            for i in range(0, len(tokens), batch_size):
                step[0] = i // batch_size
                x = tokens[i:i + batch_size].to(device)
                emulated.append(model(x).cpu())
        finally:
            for handle in handles:
                handle.remove()
    model.train(was_training)
    ref, emu = torch.cat(reference), torch.cat(emulated)
    delta = float((emu - ref).abs().max())
    targets = next_tokens.reshape(-1)
    loss_ref = float(F.cross_entropy(ref.reshape(-1, 256), targets))
    loss_emu = float(F.cross_entropy(emu.reshape(-1, 256), targets))
    loss_delta = abs(loss_emu - loss_ref)
    agreement = float((emu.argmax(-1) == ref.argmax(-1)).float().mean())
    passed = (delta <= cfg["logit_atol"] and loss_delta <= cfg["loss_atol"]
              and agreement == 1.0)
    contract = {"events": recorder.events, "totals": recorder.totals()}
    correctness = {"max_abs_logit_deviation": delta, "reference_loss": loss_ref,
                   "emulated_loss": loss_emu, "abs_loss_delta": loss_delta,
                   "token_agreement": agreement,
                   "tolerances": {"logit_atol": cfg["logit_atol"], "loss_atol": cfg["loss_atol"],
                                  "token_agreement": 1.0},
                   "passed": passed}
    outputs = {"reference_logits": ref, "emulated_logits": emu}
    return correctness, contract, outputs


def extract_remote_up_weight(model):
    """The remote block's FFN up-projection weight (width -> 4*width) from the bound checkpoint."""
    remote = model.remote_path.remote
    blocks = getattr(remote, "blocks", None)
    if not blocks or not isinstance(blocks[0].mlp_fc, nn.Linear):
        raise ValueError("Remote block FFN up-projection not found")
    weight = blocks[0].mlp_fc.weight.detach().cpu().float().clone()
    return weight


def boundary_window(model, panels, seed, device):
    """The released boundary tensor of evaluation-panel window 0 (deterministic, hashed)."""
    window = {"tokens": panels["evaluation"]["tokens"][:1],
              "next_tokens": panels["evaluation"]["next_tokens"][:1]}
    pre = prequant_capture(model, window, 1, derive_seed(PROTOCOL_ID, seed, "stage-b-window"), device)
    released, support = quantized_release(pre, model.remote_path.quantizer, 1)
    if not support["support_valid"]:
        raise RuntimeError("Boundary window release failed support checks")
    return released[0].cpu(), tensor_hash(released[0])


def stage_b(weight, x_window, cfg, tenseal=None):
    """Real CKKS encrypted per-token projection; all five repetitions retained."""
    if tenseal is None:
        try:
            import tenseal as ts
        except ImportError as exc:
            raise RuntimeError("Stage B requires TenSEAL in the track environment "
                               "(.venv-tee); no mock is ever substituted") from exc
    else:
        ts = tenseal
    x_window = x_window.cpu().float()
    w_plain = weight.t().contiguous().numpy().tolist()  # (width, 4*width) plaintext matrix
    results = []
    for n_tokens in cfg["stage_b_n_tokens"]:
        x = x_window[:n_tokens]
        start = time.perf_counter()
        reference = (x @ weight.t().contiguous()).numpy()
        reference_seconds = time.perf_counter() - start
        repetitions = []
        for repetition in range(cfg["stage_b_repetitions"]):
            start = time.perf_counter()
            context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=cfg["ckks_poly_modulus_degree"],
                                 coeff_mod_bit_sizes=cfg["ckks_coeff_mod_bit_sizes"])
            context.global_scale = 2 ** cfg["ckks_global_scale_exp"]
            context.generate_galois_keys()
            keygen_seconds = time.perf_counter() - start
            start = time.perf_counter()
            encrypted = [ts.ckks_vector(context, row.numpy()) for row in x]
            encryption_seconds = time.perf_counter() - start
            ciphertext_bytes = sum(len(e.serialize()) for e in encrypted)
            start = time.perf_counter()
            outputs = [e.mm(w_plain) for e in encrypted]
            compute_seconds = time.perf_counter() - start
            start = time.perf_counter()
            decrypted = np.array([o.decrypt() for o in outputs])
            decryption_seconds = time.perf_counter() - start
            start = time.perf_counter()
            for value in decrypted:
                ts.ckks_vector(context, value)
            refresh_seconds = time.perf_counter() - start
            error = np.abs(decrypted - reference)
            repetitions.append({"repetition": repetition,
                                "context_keygen_seconds": keygen_seconds,
                                "encryption_seconds": encryption_seconds,
                                "serialized_ciphertext_bytes": ciphertext_bytes,
                                "homomorphic_matmul_seconds": compute_seconds,
                                "decryption_seconds": decryption_seconds,
                                "decrypt_reencrypt_refresh_seconds": refresh_seconds,
                                "end_to_end_seconds": keygen_seconds + encryption_seconds
                                + compute_seconds + decryption_seconds + refresh_seconds,
                                "max_abs_error": float(error.max()),
                                "mean_abs_error": float(error.mean()),
                                "relative_error": float(error.max() / np.abs(reference).max())})
        results.append({"n_tokens": n_tokens, "reference_seconds_plaintext_cpu": reference_seconds,
                        "repetitions": repetitions})
    return {"backend": "tenseal", "backend_version": ts.__version__,
            "layout": "per-token ckks_vector (width) times plaintext (width, 4*width) matrix "
                      "via CKKSVector.mm; fresh ciphertexts and context per repetition",
            "ckks": {"poly_modulus_degree": cfg["ckks_poly_modulus_degree"],
                     "coeff_mod_bit_sizes": cfg["ckks_coeff_mod_bit_sizes"],
                     "global_scale": f"2^{cfg['ckks_global_scale_exp']}"},
            "weight_sha256": tensor_hash(weight), "results": results,
            "timing_scope": TIMING_SCOPE}


def run(config_path, label=None, validate_only=False, device=None):
    cfg = yaml.safe_load(Path(config_path).read_text())
    if device is not None:
        cfg["device"] = device
    validation = validate(cfg)
    if validate_only:
        return validation
    try:
        import tenseal  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("The TEE/FHE track requires TenSEAL in the separate track "
                           "environment (.venv-tee); no mock is ever substituted") from exc
    device = resolve_device(cfg.get("device", "cpu"))
    campaign = Campaign(cfg, config_path, label)
    for seed in cfg["seeds"]:
        model, checkpoint_record = load_checkpoint(cfg["checkpoint_manifest"])
        panels, panels_sha = bind_panels(cfg["panels_manifest"])
        model.to(device)
        binding = {"checkpoint_manifest": cfg["checkpoint_manifest"],
                   "checkpoint_manifest_sha256": digest(ROOT / cfg["checkpoint_manifest"]),
                   "checkpoint_run_id": checkpoint_record["run_id"],
                   "panels_manifest": cfg["panels_manifest"],
                   "panels_manifest_sha256": digest(ROOT / cfg["panels_manifest"]),
                   "panels_sha256": panels_sha, "device": str(device), "seed": seed}
        out, record = campaign.start(f"stage-a-s{seed}",
                                     {**binding, "stage": "A_same_checkpoint_correctness"})
        error = None
        try:
            windows = cfg["stage_a_windows"]
            tokens = panels["evaluation"]["tokens"][:windows]
            next_tokens = panels["evaluation"]["next_tokens"][:windows]
            secure_recorder = SecureComputeRecorder(
                bundle_id=f"{record['run_id']}:bundle",
                protocol_id=cfg["protocol_id"], run_id=record["run_id"], mode="trained",
                session_id=f"{record['run_id']}:session-0",
                bindings={"source_tree_sha256": campaign.source["source_tree_sha256"],
                          "config_sha256": campaign.config_hash,
                          "checkpoint_manifest_sha256": binding["checkpoint_manifest_sha256"],
                          "panels_manifest_sha256": binding["panels_manifest_sha256"],
                          "panels_sha256": binding["panels_sha256"]})
            start = time.perf_counter()
            correctness, contract, outputs = stage_a_evaluate(model, tokens, next_tokens,
                                                              cfg["stage_a_batch_size"], cfg, device,
                                                              secure_recorder=secure_recorder)
            torch.save(outputs, out / "stage_a_outputs.pt")
            write_json(out / "contract.json", contract)
            for phase, reason in {
                    "backward": "Stage A is inference-only and performs no backward pass.",
                    "update": "Stage A performs no parameter update.",
                    "optimizer": "Stage A creates no optimizer or optimizer state.",
                    "log": "Artifact emission occurs after the emulated execution boundary.",
                    "checkpoint": "The checkpoint is restored before recording; no protected checkpoint path runs.",
                    "crash": "No crash-path injection is part of Stage A.",
            }.items():
                secure_recorder.declare_absent_phase(phase, reason)
            bundle = secure_recorder.build_bundle(
                artifacts=[artifact_record(out / "contract.json", "stage-a-contract",
                                           "trusted_diagnostic", "application/json"),
                           artifact_record(out / "stage_a_outputs.pt", "stage-a-outputs",
                                           "trusted_diagnostic", "application/x-pytorch")],
                omissions=["No confidential hardware or attestation.",
                           "No encryption or network transport.",
                           "No backward, update, optimizer, checkpoint, log, or crash-path execution.",
                           "No active attack or privacy-instrument qualification."])
            bundle_path = write_secure_compute_bundle(out / "secure_compute_bundle.json", bundle)
            write_json(out / "metrics.json",
                       {**FLAGS, "hardware_attested": False, "seed": seed,
                        "stage": "A_same_checkpoint_correctness", "binding": binding,
                        "windows": windows, "batch_size": cfg["stage_a_batch_size"],
                        "backend": "plaintext_serialized_loopback", "ciphertext_only": False,
                        "encryption": "none", "attestation": "none",
                        "correctness": correctness,
                        "outputs_sha256": {k: tensor_hash(v) for k, v in outputs.items()},
                        "contract_totals": contract["totals"],
                        "secure_compute_bundle_schema_version": bundle["schema_version"],
                        "secure_compute_bundle_sha256": digest(bundle_path),
                        "deployment_scope": DEPLOYMENT_SCOPE, "timing_scope": TIMING_SCOPE,
                        "wall_seconds": time.perf_counter() - start})
            if not correctness["passed"]:
                raise RuntimeError("Stage A frozen correctness check failed")
        except Exception as exc:
            error = exc
        campaign.finish(out, record, error)
        out, record = campaign.start(f"stage-b-s{seed}",
                                     {**binding, "stage": "B_ckks_encrypted_projection"})
        error = None
        try:
            start = time.perf_counter()
            weight = extract_remote_up_weight(model)
            x_window, window_sha = boundary_window(model, panels, seed, device)
            stage_b_records = stage_b(weight, x_window, cfg)
            write_json(out / "stage_b_records.json", stage_b_records)
            write_json(out / "metrics.json",
                       {**FLAGS, "hardware_attested": False, "seed": seed,
                        "stage": "B_ckks_encrypted_projection", "binding": binding,
                        "input_window_sha256": window_sha,
                        "ckks": stage_b_records["ckks"], "backend": stage_b_records["backend"],
                        "backend_version": stage_b_records["backend_version"],
                        "results": stage_b_records["results"],
                        "deployment_scope": DEPLOYMENT_SCOPE, "timing_scope": TIMING_SCOPE,
                        "wall_seconds": time.perf_counter() - start})
        except Exception as exc:
            error = exc
        campaign.finish(out, record, error)
    return campaign.summarize({"expected_scenarios": validation["expected_scenarios"],
                               "hardware_attested": False, "qualified": False})


BATCHED_PROTOCOL_ID = "p4-tee-fhe-batched-0.1"
BATCHED_STAGES = ("stage-c", "stage-d")
BATCHED_BUDGET = {"stage_c_block_counts": [1, 2, 4], "stage_c_amortized_mms": 32,
                  "stage_c_n_tokens": 128, "stage_d_n_tokens": 128, "stage_repetitions": 5,
                  "ckks_poly_modulus_degree": 8192, "ckks_coeff_mod_bit_sizes": [60, 40, 40, 60],
                  "ckks_global_scale_exp": 40, "cpu_threads": 2}
REMOTE_SHAPES = {"qkv": [256, 768], "proj": [256, 256], "mlp_fc": [256, 1024],
                 "mlp_proj": [1024, 256]}
POLICIES = {"none": [], "ffn-up-only": ["mlp_fc"],
            "all-remote-linears": ["qkv", "proj", "mlp_fc", "mlp_proj"]}
INTERACTIVE_BUDGET_SECONDS = 1.0
POLICY_SCOPE = ("Policy costs are measured per-operator and composed arithmetically; trusted-side "
                "plaintext work and network time are outside this table (transfer is recorded as "
                "bytes). Conceivability uses a declared 1.0 s interactive reference budget for one "
                "sequence; a reporting convention from these measurements only, not a gate")


def validate_batched(cfg, only_stages=None):
    require_engineering(cfg, "tee_fhe")
    if cfg.get("protocol_id") != BATCHED_PROTOCOL_ID:
        raise ValueError("Protocol ID differs from the batched-matmul engineering protocol")
    validate_seed_namespace(cfg.get("seeds"), "tee_fhe_engineering")
    if cfg.get("seeds") != SEEDS:
        raise ValueError("The track engineering campaign declares exactly the one reserved seed")
    for key in ("checkpoint_manifest", "panels_manifest", "stage_a_scenario_manifest"):
        if not isinstance(cfg.get(key), str) or not cfg[key]:
            raise ValueError("The checkpoint, panels, and stage-A contract bindings are required")
    selected = list(BATCHED_STAGES) if only_stages is None else list(only_stages)
    if not selected or any(s not in BATCHED_STAGES for s in selected) or len(set(selected)) != len(selected):
        raise ValueError("Selected stages must be a nonempty unique subset of stage-c/stage-d")
    full = bool(cfg.get("full_budget", False))
    if full:
        if any(cfg.get(k) != v for k, v in BATCHED_BUDGET.items()):
            raise ValueError("Changed budgets or CKKS constants require a new protocol")
        if cfg.get("stage_d_shapes") != REMOTE_SHAPES:
            raise ValueError("Remote-linear shape declaration differs from the protocol")
    return {"expected_scenarios": len(cfg["seeds"]) * len(selected), "full_budget": full,
            "stages": selected, **FLAGS, "hardware_attested": False}


def _require_tenseal(tenseal):
    if tenseal is not None:
        return tenseal
    try:
        import tenseal as ts
    except ImportError as exc:
        raise RuntimeError("Stages C/D require TenSEAL in the track environment "
                           "(.venv-tee); no mock is ever substituted") from exc
    return ts


def _ckks_context(ts, cfg):
    start = time.perf_counter()
    context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=cfg["ckks_poly_modulus_degree"],
                         coeff_mod_bit_sizes=cfg["ckks_coeff_mod_bit_sizes"])
    context.global_scale = 2 ** cfg["ckks_global_scale_exp"]
    context.generate_galois_keys()
    return context, time.perf_counter() - start


def _timed_projection(ts, context, vectors, w_plain):
    """Phase-timed encrypt/mm/decrypt/refresh of (packed) vectors against a plaintext matrix."""
    start = time.perf_counter()
    encrypted = [ts.ckks_vector(context, vector) for vector in vectors]
    encryption_seconds = time.perf_counter() - start
    ciphertext_bytes = sum(len(e.serialize()) for e in encrypted)
    start = time.perf_counter()
    outputs = [e.mm(w_plain) for e in encrypted]
    mm_seconds = time.perf_counter() - start
    start = time.perf_counter()
    decrypted = [np.asarray(o.decrypt()) for o in outputs]
    decryption_seconds = time.perf_counter() - start
    start = time.perf_counter()
    for value in decrypted:
        ts.ckks_vector(context, value)
    refresh_seconds = time.perf_counter() - start
    return {"encryption_seconds": encryption_seconds, "serialized_ciphertext_bytes": ciphertext_bytes,
            "homomorphic_matmul_seconds": mm_seconds, "decryption_seconds": decryption_seconds,
            "decrypt_reencrypt_refresh_seconds": refresh_seconds, "decrypted": decrypted}


def _timed_packed_projection(ts, context, blocks, w_plain):
    """Phase-timed encrypt+pack/mm/decrypt/refresh of token blocks (pack_vectors layout)."""
    start = time.perf_counter()
    encrypted = [ts.CKKSVector.pack_vectors([ts.ckks_vector(context, row) for row in block])
                 for block in blocks]
    encryption_seconds = time.perf_counter() - start
    ciphertext_bytes = sum(len(e.serialize()) for e in encrypted)
    start = time.perf_counter()
    outputs = [e.mm(w_plain) for e in encrypted]
    mm_seconds = time.perf_counter() - start
    start = time.perf_counter()
    decrypted = [np.asarray(o.decrypt()) for o in outputs]
    decryption_seconds = time.perf_counter() - start
    start = time.perf_counter()
    for value in decrypted:
        ts.ckks_vector(context, value)
    refresh_seconds = time.perf_counter() - start
    return {"encryption_seconds": encryption_seconds, "serialized_ciphertext_bytes": ciphertext_bytes,
            "homomorphic_matmul_seconds": mm_seconds, "decryption_seconds": decryption_seconds,
            "decrypt_reencrypt_refresh_seconds": refresh_seconds, "decrypted": decrypted}


def _block_errors(decrypted, references):
    errors = [np.abs(np.asarray(dec).reshape(ref.shape) - ref)
              for dec, ref in zip(decrypted, references)]
    return float(max(e.max() for e in errors)), float(np.mean([e.mean() for e in errors]))


def stage_c(weight, x_window, cfg, tenseal=None):
    """Batched-matmul corrective: per-token vs slot-packed vs keygen-amortized timing."""
    ts = _require_tenseal(tenseal)
    x = x_window[:cfg["stage_c_n_tokens"]].cpu().float()
    n_tokens, width = len(x), weight.shape[1]
    w_plain = weight.t().contiguous().numpy().tolist()
    reference = (x @ weight.t().contiguous()).numpy()
    per_token_reps = []
    for repetition in range(cfg["stage_repetitions"]):
        context, keygen = _ckks_context(ts, cfg)
        measured = _timed_projection(ts, context, [row.numpy() for row in x], w_plain)
        max_error, mean_error = _block_errors(measured["decrypted"], reference)
        per_token_reps.append({"repetition": repetition, "context_keygen_seconds": keygen,
                               **{k: v for k, v in measured.items() if k != "decrypted"},
                               "end_to_end_seconds": keygen + measured["encryption_seconds"]
                               + measured["homomorphic_matmul_seconds"] + measured["decryption_seconds"]
                               + measured["decrypt_reencrypt_refresh_seconds"],
                               "max_abs_error": max_error, "mean_abs_error": mean_error})
    baseline_mm = float(np.mean([r["homomorphic_matmul_seconds"] for r in per_token_reps]))
    packed = []
    for b in cfg["stage_c_block_counts"]:
        if b < 1 or n_tokens % b or b * width * 2 > cfg["ckks_poly_modulus_degree"]:
            raise ValueError("Block count must divide the token count and fit the CKKS slots")
        w_bd = np.zeros((b * width, b * 4 * width))
        wt = weight.t().numpy()
        for i in range(b):
            w_bd[i * width:(i + 1) * width, i * 4 * width:(i + 1) * 4 * width] = wt
        blocks = [[row.numpy() for row in x[i:i + b]] for i in range(0, n_tokens, b)]
        references = [reference[i:i + b] for i in range(0, n_tokens, b)]
        reps = []
        for repetition in range(cfg["stage_repetitions"]):
            context, keygen = _ckks_context(ts, cfg)
            measured = _timed_packed_projection(ts, context, blocks, w_bd.tolist())
            max_error, mean_error = _block_errors(measured["decrypted"], references)
            reps.append({"repetition": repetition, "context_keygen_seconds": keygen,
                         **{k: v for k, v in measured.items() if k != "decrypted"},
                         "per_token_matmul_seconds": measured["homomorphic_matmul_seconds"] / n_tokens,
                         "max_abs_error": max_error, "mean_abs_error": mean_error})
        packed_mm = float(np.mean([r["homomorphic_matmul_seconds"] for r in reps]))
        packed.append({"block_count": b, "blocks_per_sequence": len(blocks),
                       "packed_per_token_ratio": packed_mm / baseline_mm, "repetitions": reps})
    context, keygen = _ckks_context(ts, cfg)
    mm_times = []
    for i in range(cfg["stage_c_amortized_mms"]):
        encrypted = ts.ckks_vector(context, x[i % n_tokens].numpy())
        start = time.perf_counter()
        encrypted.mm(w_plain)
        mm_times.append(time.perf_counter() - start)
    ratios = [p["packed_per_token_ratio"] for p in packed]
    return {"baseline_per_token": {"n_tokens": n_tokens, "repetitions": per_token_reps,
                                   "mean_total_matmul_seconds": baseline_mm},
            "packed": packed,
            "layout": ("per-token ckks_vector.mm baseline; packed variant encrypts per token, "
                       "packs via CKKSVector.pack_vectors into b-token blocks, and runs one "
                       "block-diagonal mm (b*width -> b*4*width); fresh context per repetition"),
            "keygen_amortized": {"sequential_mms": len(mm_times), "keygen_seconds": keygen,
                                 "mean_per_token_matmul_seconds": float(np.mean(mm_times)),
                                 "keygen_share_seconds_per_token": keygen / len(mm_times)},
            "op_count_explanation": (
                f"measured packed/per-token homomorphic-matmul ratios {ratios} at block counts "
                f"{cfg['stage_c_block_counts']}: rotations and diagonal multiplies scale with the "
                "output dimension regardless of packing, so slot-packing does not reduce the "
                "homomorphic matmul cost at this shape; the saving is ciphertext count, not "
                "computation (measured, not asserted)"),
            "weight_sha256": tensor_hash(weight),
            "reference_sha256": tensor_hash(torch.tensor(reference)),
            "ckks": {"poly_modulus_degree": cfg["ckks_poly_modulus_degree"],
                     "coeff_mod_bit_sizes": cfg["ckks_coeff_mod_bit_sizes"],
                     "global_scale": f"2^{cfg['ckks_global_scale_exp']}"},
            "backend": "tenseal", "backend_version": ts.__version__, "timing_scope": TIMING_SCOPE}


def extract_remote_weights(model, cfg):
    """The remote block's four linear weights, shapes verified against the declaration."""
    block = model.remote_path.remote.blocks[0]
    weights = {}
    for name, (in_features, out_features) in cfg["stage_d_shapes"].items():
        linear = getattr(block, name, None)
        if not isinstance(linear, nn.Linear) or tuple(linear.weight.shape) != (out_features, in_features):
            raise ValueError(f"Remote linear {name} shape differs from the protocol declaration")
        weights[name] = linear.weight.detach().cpu().float().clone()
    return weights


def load_contract(manifest_path):
    """The stage-A observation contract, verified through its scenario manifest."""
    record, artifacts, cfg = verify_scenario(manifest_path)
    if (record["status"] != "completed" or record["kind"] != "tee_fhe"
            or record["inputs"].get("stage") != "A_same_checkpoint_correctness"):
        raise ValueError("Need the completed stage-A observation-contract scenario")
    metrics = json.loads(artifacts["metrics.json"].read_text())
    contract = json.loads(artifacts["contract.json"].read_text())
    return contract["totals"]["per_operator"], metrics["windows"], record


def compose_policy_table(shape_records, contract_rows, operator_prefix, contract_windows,
                         n_tokens, budget_seconds=INTERACTIVE_BUDGET_SECONDS):
    """Compose the frozen policies arithmetically from measured per-shape costs and contract bytes."""
    traffic = {}
    for row in contract_rows:
        traffic[row["operator"]] = traffic.get(row["operator"], 0) + row["serialized_bytes"]
    table = []
    for policy, operators in POLICIES.items():
        names = [f"{operator_prefix}.{name}" for name in operators]
        missing = [name for name in names if name not in traffic]
        if missing:
            raise ValueError(f"Observation contract lacks policy operators: {missing}")
        phases = ["encryption_seconds", "homomorphic_matmul_seconds", "decryption_seconds",
                  "decrypt_reencrypt_refresh_seconds"]
        totals = {phase: n_tokens * sum(shape_records[name]["mean_per_token"][phase] for name in operators)
                  for phase in phases}
        plaintext_seconds = sum(shape_records[name]["reference_batched_seconds"] for name in operators)
        ciphertext_bytes = n_tokens * sum(shape_records[name]["mean_per_token"]["serialized_ciphertext_bytes"]
                                          for name in operators)
        end_to_end = sum(totals.values())
        ratio = end_to_end / plaintext_seconds if operators else 1.0
        conceivable = end_to_end <= budget_seconds
        table.append({"policy": policy, "operators": names,
                      "homomorphic_seconds": totals["homomorphic_matmul_seconds"],
                      "encryption_seconds": totals["encryption_seconds"],
                      "decryption_seconds": totals["decryption_seconds"],
                      "refresh_seconds": totals["decrypt_reencrypt_refresh_seconds"],
                      "end_to_end_seconds": end_to_end,
                      "ciphertext_bytes": ciphertext_bytes,
                      "traffic_bytes": sum(traffic[name] for name in names) / contract_windows,
                      "reference_plaintext_batched_seconds": plaintext_seconds,
                      "ratio_to_plaintext_batched": ratio,
                      "interactive_latency_conceivable": conceivable,
                      "conceivability_statement": (
                          f"end-to-end {end_to_end:.4g}s for one {n_tokens}-token sequence at this "
                          f"scale against a declared {budget_seconds}s interactive reference budget; "
                          f"interactive-latency conceivable: {conceivable} — from these measurements "
                          "only, no extrapolation to other models, hardware, backends, or layouts")})
    return table


def stage_d(model, x_window, cfg, tenseal=None):
    """Selective split policy costs: per-shape measurements composed into the frozen policies."""
    ts = _require_tenseal(tenseal)
    weights = extract_remote_weights(model, cfg)
    n_tokens = cfg["stage_d_n_tokens"]
    x = x_window[:n_tokens].cpu().float()
    hidden = (x @ weights["mlp_fc"].t().contiguous()).float()
    inputs = {"qkv": x, "proj": x, "mlp_fc": x, "mlp_proj": hidden}
    operator_prefix = next(name for name, module in model.named_modules()
                           if module is model.remote_path.remote.blocks[0])
    contract_rows, contract_windows, contract_record = load_contract(cfg["stage_a_scenario_manifest"])
    shape_records = {}
    for name, weight in weights.items():
        x_in = inputs[name]
        w_plain = weight.t().contiguous().numpy().tolist()
        reference = (x_in @ weight.t().contiguous()).numpy()
        start = time.perf_counter()
        (x_in @ weight.t().contiguous()).numpy()
        reference_batched = time.perf_counter() - start
        reps = []
        for repetition in range(cfg["stage_repetitions"]):
            context, keygen = _ckks_context(ts, cfg)
            measured = _timed_projection(ts, context, [x_in[0].numpy()], w_plain)
            max_error, mean_error = _block_errors(measured["decrypted"], reference[:1])
            reps.append({"repetition": repetition, "context_keygen_seconds": keygen,
                         **{k: v for k, v in measured.items() if k != "decrypted"},
                         "max_abs_error": max_error, "mean_abs_error": mean_error})
        phases = ["context_keygen_seconds", "encryption_seconds", "homomorphic_matmul_seconds",
                  "decryption_seconds", "decrypt_reencrypt_refresh_seconds",
                  "serialized_ciphertext_bytes"]
        shape_records[name] = {"repetitions": reps,
                               "mean_per_token": {phase: float(np.mean([r[phase] for r in reps]))
                                                  for phase in phases},
                               "reference_batched_seconds": reference_batched,
                               "input_sha256": tensor_hash(inputs[name]),
                               "reference_sha256": tensor_hash(torch.tensor(reference))}
    table = compose_policy_table(shape_records, contract_rows, operator_prefix,
                                 contract_windows, n_tokens)
    return {"weights": {name: {"in_features": cfg["stage_d_shapes"][name][0],
                               "out_features": cfg["stage_d_shapes"][name][1],
                               "weight_sha256": tensor_hash(weight)}
                        for name, weight in weights.items()},
            "shapes": shape_records, "policy_table": table,
            "contract": {"stage_a_scenario_manifest": cfg["stage_a_scenario_manifest"],
                         "stage_a_scenario_manifest_sha256": digest(ROOT / cfg["stage_a_scenario_manifest"]),
                         "stage_a_run_id": contract_record["run_id"],
                         "contract_windows": contract_windows, "operator_prefix": operator_prefix},
            "interactive_budget_seconds": INTERACTIVE_BUDGET_SECONDS,
            "ckks": {"poly_modulus_degree": cfg["ckks_poly_modulus_degree"],
                     "coeff_mod_bit_sizes": cfg["ckks_coeff_mod_bit_sizes"],
                     "global_scale": f"2^{cfg['ckks_global_scale_exp']}"},
            "backend": "tenseal", "backend_version": ts.__version__,
            "policy_scope": POLICY_SCOPE, "timing_scope": TIMING_SCOPE}


def run_batched(config_path, label=None, validate_only=False, device=None, only_stages=None):
    cfg = yaml.safe_load(Path(config_path).read_text())
    if device is not None:
        cfg["device"] = device
    validation = validate_batched(cfg, only_stages)
    if validate_only:
        return validation
    ts = _require_tenseal(None)
    device = resolve_device(cfg.get("device", "cpu"))
    campaign = Campaign(cfg, config_path, label)
    selected = validation["stages"]
    for seed in cfg["seeds"]:
        model, checkpoint_record = load_checkpoint(cfg["checkpoint_manifest"])
        panels, panels_sha = bind_panels(cfg["panels_manifest"])
        model.to(device)
        x_window, window_sha = boundary_window(model, panels, seed, device)
        binding = {"checkpoint_manifest": cfg["checkpoint_manifest"],
                   "checkpoint_manifest_sha256": digest(ROOT / cfg["checkpoint_manifest"]),
                   "checkpoint_run_id": checkpoint_record["run_id"],
                   "panels_manifest": cfg["panels_manifest"],
                   "panels_manifest_sha256": digest(ROOT / cfg["panels_manifest"]),
                   "panels_sha256": panels_sha, "input_window_sha256": window_sha,
                   "device": str(device), "seed": seed, "selected_stages": selected}
        if "stage-c" in selected:
            out, record = campaign.start(f"stage-c-s{seed}",
                                         {**binding, "stage": "C_batched_matmul_corrective"})
            error = None
            try:
                start = time.perf_counter()
                weight = extract_remote_up_weight(model)
                records = stage_c(weight, x_window, cfg, ts)
                write_json(out / "stage_c_records.json", records)
                write_json(out / "metrics.json",
                           {**FLAGS, "hardware_attested": False, "seed": seed,
                            "stage": "C_batched_matmul_corrective", "binding": binding,
                            "packed_per_token_ratio": {p["block_count"]: p["packed_per_token_ratio"]
                                                       for p in records["packed"]},
                            "baseline_mean_total_matmul_seconds":
                                records["baseline_per_token"]["mean_total_matmul_seconds"],
                            "keygen_amortized": records["keygen_amortized"],
                            "op_count_explanation": records["op_count_explanation"],
                            "ckks": records["ckks"], "backend": records["backend"],
                            "backend_version": records["backend_version"],
                            "deployment_scope": DEPLOYMENT_SCOPE, "timing_scope": TIMING_SCOPE,
                            "wall_seconds": time.perf_counter() - start})
            except Exception as exc:
                error = exc
            campaign.finish(out, record, error)
        if "stage-d" in selected:
            out, record = campaign.start(f"stage-d-s{seed}",
                                         {**binding, "stage": "D_selective_split_policy"})
            error = None
            try:
                start = time.perf_counter()
                records = stage_d(model, x_window, cfg, ts)
                write_json(out / "stage_d_records.json", records)
                write_json(out / "metrics.json",
                           {**FLAGS, "hardware_attested": False, "seed": seed,
                            "stage": "D_selective_split_policy", "binding": binding,
                            "policy_table": records["policy_table"],
                            "contract": records["contract"],
                            "interactive_budget_seconds": records["interactive_budget_seconds"],
                            "ckks": records["ckks"], "backend": records["backend"],
                            "backend_version": records["backend_version"],
                            "deployment_scope": DEPLOYMENT_SCOPE, "timing_scope": TIMING_SCOPE,
                            "policy_scope": POLICY_SCOPE,
                            "wall_seconds": time.perf_counter() - start})
            except Exception as exc:
                error = exc
            campaign.finish(out, record, error)
    return campaign.summarize({"expected_scenarios": validation["expected_scenarios"],
                               "stages": selected, "hardware_attested": False, "qualified": False})
