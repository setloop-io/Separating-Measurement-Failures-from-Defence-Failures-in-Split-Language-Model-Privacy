"""Verification and descriptive aggregation for the TEE/FHE engineering stages."""
import json
import re
import numpy as np
import torch

from .experiment import ROOT, digest, write_json
from .scenario_verification import verify_scenario
from .secure_compute_record import load_secure_compute_bundle
from .tee_fhe_scenarios import (DEPLOYMENT_SCOPE, TIMING_SCOPE,
                                compose_policy_table, load_contract)
from .warmup_scenarios import FLAGS, tensor_hash

STAGE_SCENARIOS = {"A_same_checkpoint_correctness": "stage-a",
                   "B_ckks_encrypted_projection": "stage-b",
                   "C_batched_matmul_corrective": "stage-c",
                   "D_selective_split_policy": "stage-d"}


def check_stage_a(artifacts, cfg, manifest=None):
    metrics = json.loads(artifacts["metrics.json"].read_text())
    if (any(metrics.get(k) != v for k, v in FLAGS.items())
            or metrics["hardware_attested"] is not False
            or metrics["encryption"] != "none" or metrics["attestation"] != "none"):
        raise ValueError("Stage A claim-scope mismatch")
    outputs = torch.load(artifacts["stage_a_outputs.pt"], weights_only=True)
    for name, tensor in outputs.items():
        if tensor_hash(tensor) != metrics["outputs_sha256"][name]:
            raise ValueError("Stage A outputs differ from their recorded hashes")
    ref, emu = outputs["reference_logits"], outputs["emulated_logits"]
    delta = float((emu - ref).abs().max())
    correctness = metrics["correctness"]
    if (delta != correctness["max_abs_logit_deviation"]
            or float((emu.argmax(-1) == ref.argmax(-1)).float().mean()) != correctness["token_agreement"]):
        raise ValueError("Stage A correctness record differs from the saved outputs")
    contract = json.loads(artifacts["contract.json"].read_text())
    events = contract["events"]
    totals = {"total_calls": len(events),
              "total_serialized_bytes": sum(e["serialized_bytes"] for e in events),
              "steps": max((e["step"] for e in events), default=-1) + 1}
    if (totals != {k: contract["totals"][k] for k in totals}
            or totals != {k: metrics["contract_totals"][k] for k in totals}):
        raise ValueError("Observation-contract byte totals differ from the events")
    if any(not e["operator"] or e["direction"] not in ("in", "out")
           or e["serialized_bytes"] <= 0 or not e["shape"] or e["dtype"] != "torch.float32"
           for e in events):
        raise ValueError("Invalid observation-contract event")
    if not correctness["passed"]:
        raise ValueError("Stage A recorded a failed frozen correctness check")
    bundle_summary = None
    bundle_declared = ("secure_compute_bundle_sha256" in metrics
                       or "secure_compute_bundle_schema_version" in metrics)
    bundle_present = "secure_compute_bundle.json" in artifacts
    if bundle_declared != bundle_present:
        raise ValueError("Stage A secure-compute bundle declaration/artifact mismatch")
    if bundle_present:
        bundle_path = artifacts["secure_compute_bundle.json"]
        if (digest(bundle_path) != metrics["secure_compute_bundle_sha256"]
                or metrics["secure_compute_bundle_schema_version"]
                != "p4-secure-compute-bundle-0.1"):
            raise ValueError("Stage A secure-compute bundle identity mismatch")
        bundle = load_secure_compute_bundle(bundle_path, root=ROOT)
        if manifest is not None and bundle["run_id"] != manifest["run_id"]:
            raise ValueError("Stage A secure-compute bundle/run mismatch")
        external = bundle["views"]["external_attacker"]["events"]
        trusted = bundle["views"]["trusted_diagnostic"]["events"]
        if (len(external) != len(events) or len(trusted) != len(events)
                or sum(e["external_observation"]["serialized_bytes"] for e in external)
                != totals["total_serialized_bytes"]):
            raise ValueError("Stage A secure-compute events differ from observation contract")
        bundle_summary = {"schema_version": bundle["schema_version"],
                          "paired_events": len(external),
                          "phase_coverage": bundle["phase_coverage"],
                          "event_chain_sha256": bundle["event_chain_sha256"]}
    return {"max_abs_logit_deviation": correctness["max_abs_logit_deviation"],
            "abs_loss_delta": correctness["abs_loss_delta"],
            "token_agreement": correctness["token_agreement"],
            "contract_totals": contract["totals"],
            "secure_compute_bundle": bundle_summary}


def check_stage_b(artifacts, cfg):
    metrics = json.loads(artifacts["metrics.json"].read_text())
    records = json.loads(artifacts["stage_b_records.json"].read_text())
    if (any(metrics.get(k) != v for k, v in FLAGS.items())
            or metrics["hardware_attested"] is not False
            or records["backend"] != "tenseal" or records["timing_scope"] != TIMING_SCOPE):
        raise ValueError("Stage B claim-scope mismatch")
    if (records["ckks"] != {"poly_modulus_degree": cfg["ckks_poly_modulus_degree"],
                            "coeff_mod_bit_sizes": cfg["ckks_coeff_mod_bit_sizes"],
                            "global_scale": f"2^{cfg['ckks_global_scale_exp']}"}
            or [r["n_tokens"] for r in records["results"]] != cfg["stage_b_n_tokens"]):
        raise ValueError("Stage B parameters differ from the protocol configuration")
    aggregated = []
    for result in records["results"]:
        reps = result["repetitions"]
        if len(reps) != cfg["stage_b_repetitions"]:
            raise ValueError("Missing or duplicated stage B repetition")
        phases = ["context_keygen_seconds", "encryption_seconds", "homomorphic_matmul_seconds",
                  "decryption_seconds", "decrypt_reencrypt_refresh_seconds", "end_to_end_seconds"]
        row = {"n_tokens": result["n_tokens"],
               "reference_seconds_plaintext_cpu": result["reference_seconds_plaintext_cpu"],
               "repetitions": len(reps)}
        for phase in phases:
            values = np.array([r[phase] for r in reps], dtype=np.float64)
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError("Invalid stage B timing record")
            row[phase] = {"mean": float(values.mean()),
                          "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0}
        errors = np.array([[r["max_abs_error"], r["mean_abs_error"], r["relative_error"]]
                           for r in reps])
        if not np.isfinite(errors).all():
            raise ValueError("Nonfinite stage B numerical error")
        if any(r["serialized_ciphertext_bytes"] <= 0 for r in reps):
            raise ValueError("Stage B recorded no ciphertext transfer bytes")
        row["serialized_ciphertext_bytes"] = [r["serialized_ciphertext_bytes"] for r in reps]
        row["max_abs_error"] = {"mean": float(errors[:, 0].mean()), "max": float(errors[:, 0].max())}
        row["mean_abs_error"] = float(errors[:, 1].mean())
        row["relative_error"] = {"mean": float(errors[:, 2].mean()), "max": float(errors[:, 2].max())}
        aggregated.append(row)
    return aggregated


def check_stage_c(artifacts, cfg):
    metrics = json.loads(artifacts["metrics.json"].read_text())
    records = json.loads(artifacts["stage_c_records.json"].read_text())
    if (any(metrics.get(k) != v for k, v in FLAGS.items())
            or metrics["hardware_attested"] is not False
            or records["backend"] != "tenseal" or records["timing_scope"] != TIMING_SCOPE
            or not records["op_count_explanation"]):
        raise ValueError("Stage C claim-scope mismatch")
    if (records["ckks"] != {"poly_modulus_degree": cfg["ckks_poly_modulus_degree"],
                            "coeff_mod_bit_sizes": cfg["ckks_coeff_mod_bit_sizes"],
                            "global_scale": f"2^{cfg['ckks_global_scale_exp']}"}
            or records["baseline_per_token"]["n_tokens"] != cfg["stage_c_n_tokens"]):
        raise ValueError("Stage C parameters differ from the protocol configuration")
    baseline = records["baseline_per_token"]["repetitions"]
    if len(baseline) != cfg["stage_repetitions"]:
        raise ValueError("Missing or duplicated baseline repetition")
    baseline_mm = float(np.mean([r["homomorphic_matmul_seconds"] for r in baseline]))
    if baseline_mm != records["baseline_per_token"]["mean_total_matmul_seconds"]:
        raise ValueError("Baseline mean differs from its repetitions")
    for entry in records["packed"]:
        reps = entry["repetitions"]
        if len(reps) != cfg["stage_repetitions"]:
            raise ValueError("Missing or duplicated packed repetition")
        for rep in reps:
            if (rep["per_token_matmul_seconds"] != rep["homomorphic_matmul_seconds"] / cfg["stage_c_n_tokens"]
                    or rep["serialized_ciphertext_bytes"] <= 0
                    or not np.isfinite(rep["max_abs_error"]) or not np.isfinite(rep["mean_abs_error"])):
                raise ValueError("Invalid packed repetition record")
        packed_mm = float(np.mean([r["homomorphic_matmul_seconds"] for r in reps]))
        if entry["packed_per_token_ratio"] != packed_mm / baseline_mm:
            raise ValueError("Packed ratio differs from the measured totals")
    amortized = records["keygen_amortized"]
    if (amortized["sequential_mms"] != cfg["stage_c_amortized_mms"]
            or not np.isfinite(amortized["mean_per_token_matmul_seconds"])
            or not np.isfinite(amortized["keygen_share_seconds_per_token"])):
        raise ValueError("Invalid keygen-amortized record")
    ratios = {entry["block_count"]: entry["packed_per_token_ratio"] for entry in records["packed"]}
    if {int(k): v for k, v in metrics["packed_per_token_ratio"].items()} != ratios:
        raise ValueError("Metrics/records ratio mismatch")
    return {"packed_per_token_ratio": ratios, "baseline_mean_total_matmul_seconds": baseline_mm,
            "keygen_amortized": amortized,
            "repetitions_retained": len(baseline) + sum(len(p["repetitions"]) for p in records["packed"])}


def check_stage_d(artifacts, cfg):
    metrics = json.loads(artifacts["metrics.json"].read_text())
    records = json.loads(artifacts["stage_d_records.json"].read_text())
    if (any(metrics.get(k) != v for k, v in FLAGS.items())
            or metrics["hardware_attested"] is not False
            or records["backend"] != "tenseal" or records["timing_scope"] != TIMING_SCOPE):
        raise ValueError("Stage D claim-scope mismatch")
    contract_rows, contract_windows, contract_record = load_contract(cfg["stage_a_scenario_manifest"])
    binding = records["contract"]
    if (binding["contract_windows"] != contract_windows
            or binding["stage_a_run_id"] != contract_record["run_id"]
            or digest(ROOT / binding["stage_a_scenario_manifest"]) != binding["stage_a_scenario_manifest_sha256"]):
        raise ValueError("Stage D contract binding mismatch")
    for name, shape in records["shapes"].items():
        reps = shape["repetitions"]
        if len(reps) != cfg["stage_repetitions"]:
            raise ValueError("Missing or duplicated stage D repetition")
        for phase, value in shape["mean_per_token"].items():
            if value != float(np.mean([r[phase] for r in reps])):
                raise ValueError("Shape mean differs from its repetitions")
            if phase == "serialized_ciphertext_bytes":
                if value <= 0:
                    raise ValueError("Stage D recorded no ciphertext bytes")
            elif not np.isfinite(value) or value < 0:
                raise ValueError("Invalid stage D timing record")
        if any(not np.isfinite(r["max_abs_error"]) for r in reps):
            raise ValueError("Nonfinite stage D numerical error")
    expected_table = compose_policy_table(records["shapes"], contract_rows,
                                          binding["operator_prefix"], contract_windows,
                                          cfg["stage_d_n_tokens"],
                                          records["interactive_budget_seconds"])
    if expected_table != records["policy_table"] or records["policy_table"] != metrics["policy_table"]:
        raise ValueError("Policy table differs from measured shapes and the verified contract")
    return {"policy_table": records["policy_table"],
            "per_shape_mean_matmul_seconds": {name: s["mean_per_token"]["homomorphic_matmul_seconds"]
                                              for name, s in records["shapes"].items()}}


def analyze(label):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", label):
        raise ValueError("Invalid campaign label")
    path = ROOT / "outputs/scenarios" / label
    summary = json.loads((path / "summary.json").read_text())
    if (summary["kind"] != "tee_fhe" or any(summary.get(k) != v for k, v in FLAGS.items())
            or summary.get("hardware_attested") is not False):
        raise ValueError("Invalid campaign kind/claim flags")
    config_snapshot = json.loads((path / "config.json").read_text())
    source = json.loads((path / "source.json").read_text())
    statuses, identities = [], set()
    stage_a, stage_b, stage_c, stage_d = None, None, None, None
    for run_id in summary["run_ids"]:
        manifest, artifacts, cfg = verify_scenario(f"experiments/manifests/{run_id}.json")
        # The committed config pins device: cpu; execution may record a CLI device
        # override, which must match the per-manifest recorded device instead.
        cfg_base = {k: v for k, v in cfg.items() if k != "device"}
        snap_base = {k: v for k, v in config_snapshot.items() if k != "device"}
        if (cfg_base != snap_base or manifest["kind"] != "tee_fhe"
                or manifest["source_record"] != str((path / "source.json").relative_to(ROOT))
                or manifest["inputs"].get("device", "cpu") != config_snapshot.get("device", "cpu")):
            raise ValueError("Manifest/campaign config mismatch")
        statuses.append(manifest["status"])
        if manifest["status"] != "completed":
            continue
        stage = manifest["inputs"]["stage"]
        identities.add((manifest["inputs"]["seed"], stage))
        family = stage.split("_", 1)[0]
        if family == "A":
            stage_a = check_stage_a(artifacts, cfg, manifest)
        elif family == "B":
            stage_b = check_stage_b(artifacts, cfg)
        elif family == "C":
            stage_c = check_stage_c(artifacts, cfg)
        elif family == "D":
            stage_d = check_stage_d(artifacts, cfg)
        else:
            raise ValueError("Unknown stage")
    scenario_stages = summary.get("stages") or ["stage-a", "stage-b"]
    expected = {(s, name) for s in config_snapshot["seeds"]
                for name, sid in STAGE_SCENARIOS.items() if sid in scenario_stages}
    if (len(statuses) != summary["expected_scenarios"] or len(statuses) != len(expected)
            or statuses.count("completed") != summary["completed"] or statuses.count("failed") != summary["failed"]
            or (not summary["failed"] and identities != expected)):
        raise ValueError("Incomplete campaign or mismatched scenario accounting")
    verified = {"label": label, **FLAGS, "hardware_attested": False,
                "completed": summary["completed"], "failed": summary["failed"],
                "source_archive_and_artifacts_verified": True, "stored_estimates_verified": True,
                "stage_a": stage_a, "stage_b": stage_b, "stage_c": stage_c, "stage_d": stage_d,
                "deployment_scope": DEPLOYMENT_SCOPE, "timing_scope": TIMING_SCOPE,
                "git_commit": source["git_commit"], "source_tree_sha256": source["source_tree_sha256"],
                "analysis_module_sha256": digest(__file__)}
    lines = [f"# TEE/FHE fleet-side engineering report: {label}", "",
             f"Completed {summary['completed']}/{summary['expected_scenarios']} scenarios; failures: {summary['failed']}.",
             f"Execution revision: `{source['git_commit']}`. Manifests, source archives, artifact bindings,",
             "saved outputs, and the observation-contract byte totals verified.", "",
             "Engineering diagnostics only: `scientific_evidence: false`, `hardware_attested: false`,",
             "privacy **undetermined**. No confidentiality, integrity, or attestation claim.", ""]
    if stage_a is not None:
        totals = stage_a["contract_totals"]
        lines += ["## Stage A — same-checkpoint correctness (plaintext-emulated boundary)", "",
                  f"Max |Δlogit| {stage_a['max_abs_logit_deviation']:.3g} (tolerance 1e-4); "
                  f"|Δloss| {stage_a['abs_loss_delta']:.3g} (tolerance 1e-5); "
                  f"argmax agreement {stage_a['token_agreement']:.4f} (required 1.0).",
                  f"Observation contract: {totals['total_calls']} operator calls over "
                  f"{totals['steps']} steps, {totals['total_serialized_bytes']} serialized bytes total.",
                  "This contract is the traffic inventory later attested stages must match.", ""]
    if stage_b is not None:
        lines += ["## Stage B — real CKKS encrypted projection (TenSEAL)", "",
                  "| Tokens | Reps | Encrypt (s, mean±sd) | Homomorphic mm (s, mean±sd) | Decrypt (s) | Refresh (s) | End-to-end (s) | Plaintext CPU reference (s) | Ciphertext bytes | Max abs error |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for row in stage_b:
            lines.append(f"| {row['n_tokens']} | {row['repetitions']} | "
                         f"{row['encryption_seconds']['mean']:.4f}±{row['encryption_seconds']['sd']:.4f} | "
                         f"{row['homomorphic_matmul_seconds']['mean']:.4f}±{row['homomorphic_matmul_seconds']['sd']:.4f} | "
                         f"{row['decryption_seconds']['mean']:.4f} | "
                         f"{row['decrypt_reencrypt_refresh_seconds']['mean']:.4f} | "
                         f"{row['end_to_end_seconds']['mean']:.4f} | "
                         f"{row['reference_seconds_plaintext_cpu']:.6f} | "
                         f"{row['serialized_ciphertext_bytes'][0]} | "
                         f"{row['max_abs_error']['max']:.3g} |")
        lines += ["", "All repetitions retained; no best-of selection. These timings are real FHE",
                  "measurements of one encrypted projection at the boundary scale — they are not",
                  "secure-serving latencies (no TEE, no attestation), not end-to-end inference times,",
                  "and not a defense evaluation.", ""]
    if stage_c is not None:
        lines += ["## Stage C — batched-matmul corrective", "",
                  "| Block count b | Packed/per-token ratio (measured) |",
                  "|---|---|"]
        for b, ratio in sorted(stage_c["packed_per_token_ratio"].items()):
            lines.append(f"| {b} | {ratio:.4f} |")
        amortized = stage_c["keygen_amortized"]
        lines += ["", f"{stage_c['repetitions_retained']} repetitions retained across baseline and packed variants; no best-of selection.",
                  f"Keygen-amortized serving constant: mean per-token mm "
                  f"{amortized['mean_per_token_matmul_seconds']:.4f}s over "
                  f"{amortized['sequential_mms']} sequential mms on one context; keygen share "
                  f"{amortized['keygen_share_seconds_per_token']:.4f}s/token. The op-count",
                  "explanation (rotations and diagonal multiplies scale with the output dimension",
                  "regardless of packing) is recorded from these measurements, not asserted.", ""]
    if stage_d is not None:
        lines += ["## Stage D — selective split policy costs", "",
                  "Per-token homomorphic mm cost per remote-linear shape: "
                  + "; ".join(f"{name} {value:.4f}s" for name, value in stage_d["per_shape_mean_matmul_seconds"].items())
                  + ".", "",
                  "| Policy | Homomorphic (s) | End-to-end estimate (s) | Ciphertext bytes | Traffic bytes | Ratio to plaintext batched | Interactive conceivable |",
                  "|---|---|---|---|---|---|---|"]
        for row in stage_d["policy_table"]:
            lines.append(f"| {row['policy']} | {row['homomorphic_seconds']:.3f} | "
                         f"{row['end_to_end_seconds']:.3f} | {row['ciphertext_bytes']} | "
                         f"{row['traffic_bytes']:.0f} | {row['ratio_to_plaintext_batched']:.4g} | "
                         f"{row['interactive_latency_conceivable']} |")
        lines += ["", "Conceivability statements are recorded per policy from these measurements only",
                  "(declared interactive reference budget; no extrapolation to other models, hardware,",
                  "backends, or layouts, and no security claim).", ""]
    lines += ["See [verification](verification.json), [summary](summary.json), and [source record](source.json).", ""]
    write_json(path / "verification.json", verified)
    (path / "report.md").write_text("\n".join(lines))
    return verified
