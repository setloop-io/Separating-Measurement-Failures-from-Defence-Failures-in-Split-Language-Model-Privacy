"""Next-byte structure precision at the pilot operating point; engineering only.

Protocol `p4-structure-precision-0.1` (docs/STRUCTURE_PRECISION_PROTOCOL.md): the
sister control campaign of `p4-null-precision-128-0.1` for the five-class next-byte
structure target. Its sources are the precision-128 model-preparation manifests,
which bind the verified full-budget d0 control checkpoints and, through them, the
warmup-budget setups, panels, corpus, and source archives. No normal candidate is
scored and no instrument is qualified here. `full_budget: false` exists only so
tests can run the pipeline against tiny synthetic source campaigns; the protocol
config sets it true and validation then pins every budget constant.

Labels: per token position t of window i, the label is the next-byte class of the
panel's own `next_tokens[i, t]` (`pcr.privacy_attacks.next_byte_class`;
newline/space/letter/digit/punctuation, aligned including the final position;
never np.roll).

Positive-control feature scale: the scalar population standard deviation over every
token feature value of the full fit panel's post-defense/pre-quantization tensor
(all 16,384 x 128 x width entries at the protocol operating point), multiplied by
sqrt(width). It is computed once per scenario and held fixed across panel sizes;
the five class directions and doses are likewise fixed across panel sizes.
"""
from pathlib import Path
import json
import math
import time

import numpy as np
import torch
from torch import nn
import yaml
from pcr.privacy_attacks_p4b import next_byte_class
from pcr.utils import derive_seed

from .devices import resolve_device
from .experiment import ROOT, digest, write_json
from .null_precision import prequant_capture
from .null_precision_128 import RELEASE_CONDITIONS, restore_control_model
from .scenario_io import Campaign, require_engineering, validate_seed_namespace
from .scenario_verification import verify_scenario
from .warmup_scenarios import FLAGS, OMITTED, derangement, tensor_hash

PROTOCOL_ID = "p4-structure-precision-0.1"
ADAPTER = "p4-structure-control-128-0.1"
TARGET = "next_byte_class"
BUDGET = {"panel_sizes": [8192, 16384], "matched_null_counts": [1, 4], "outer_null_repeats": 16,
          "positive_doses": [.5, 2.], "null_band": .02, "structure_positive_floor": .02,
          "capture_batch_size": 64, "mlp_hidden": 64, "mlp_lr": 3e-3, "mlp_row_batch": 8192,
          "cpu_threads": 2}


def validate(cfg, only_seeds=None):
    require_engineering(cfg, "structure_precision")
    if cfg.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Protocol ID differs from the structure precision engineering protocol")
    sources = cfg.get("sources")
    if not isinstance(sources, list) or len(sources) != 2:
        raise ValueError("Exactly two verified precision-128 preparation sources are required")
    for source in sources:
        if not isinstance(source, dict) or set(source) != {"control_seed", "prepare_manifest"}:
            raise ValueError("Each source declares control_seed and prepare_manifest")
    seeds = [s["control_seed"] for s in sources]
    validate_seed_namespace(seeds, "structure_precision_engineering")
    if len({s["prepare_manifest"] for s in sources}) != 2:
        raise ValueError("Two distinct precision-128 preparation sources are required")
    executed = seeds if only_seeds is None else list(only_seeds)
    if not executed or any(s not in seeds for s in executed):
        raise ValueError("Executed seed subset must be nonempty and declared by the protocol")
    full = bool(cfg.get("full_budget", False))
    if full:
        if cfg.get("release_conditions") != RELEASE_CONDITIONS:
            raise ValueError("Release conditions differ from the engineering protocol")
        if any(cfg.get(k) != v for k, v in BUDGET.items()):
            raise ValueError("Changed scoring budget/threshold requires a new protocol")
    expected = len(executed) * len(cfg.get("release_conditions") or [])
    return {"expected_scenarios": expected, "full_budget": full, **FLAGS}


def load_source(prepare_manifest_path):
    """Verify the prepare manifest and its chain, then load panels, config, and bases."""
    record, artifacts, cfg = verify_scenario(prepare_manifest_path)
    inputs = record["inputs"]
    if (record["status"] != "completed" or record["kind"] != "null_precision_128"
            or inputs.get("stage") != "control_model_preparation"):
        raise ValueError("Need a completed precision-128 control-model preparation")
    prepare = json.loads(artifacts["prepare_metrics.json"].read_text())
    replay = prepare["replay_verification"]
    if (replay["verified"] is not True or replay["actual"] != replay["expected"]
            or prepare["binding"] != inputs
            or prepare["control_checkpoint_sha256"] != digest(artifacts["control_checkpoint.pt"])):
        raise ValueError("Control-model preparation is not a verified replay")
    if digest(ROOT / inputs["setup_manifest"]) != inputs["setup_manifest_sha256"]:
        raise ValueError("Bound warmup-budget setup manifest changed")
    setup_record, setup_artifacts, setup_cfg = verify_scenario(inputs["setup_manifest"])
    if setup_record["run_id"] != inputs["setup_run_id"]:
        raise ValueError("Setup/branch binding mismatch")
    panels = torch.load(setup_artifacts["panels.pt"], weights_only=True)
    declarations = json.loads(setup_artifacts["panels.json"].read_text())
    if digest(setup_artifacts["panels.pt"]) != inputs["panels_sha256"]:
        raise ValueError("Setup panels differ from the preparation binding")
    for name, panel in panels.items():
        for key, expected in declarations["panels"][name]["tensor_sha256"].items():
            if tensor_hash(panel[key]) != expected:
                raise ValueError("Panel tensor mismatch")
    saved = torch.load(artifacts["control_checkpoint.pt"], weights_only=True)
    for name, basis in saved["bases"].items():
        if tensor_hash(basis) != prepare["bases"][name]:
            raise ValueError("Control checkpoint basis mismatch")
    binding = {"prepare_manifest": prepare_manifest_path,
               "prepare_manifest_sha256": digest(ROOT / prepare_manifest_path),
               "prepare_run_id": record["run_id"],
               "setup_manifest": inputs["setup_manifest"],
               "setup_manifest_sha256": inputs["setup_manifest_sha256"],
               "setup_run_id": inputs["setup_run_id"],
               "source_tree_sha256": record["source_tree_sha256"],
               "control_checkpoint_path": str(artifacts["control_checkpoint.pt"].relative_to(ROOT)),
               "control_checkpoint_sha256": prepare["control_checkpoint_sha256"],
               "panels_sha256": inputs["panels_sha256"],
               "corpus_sha256": inputs["corpus_sha256"],
               "training_seed": inputs["training_seed"],
               "training_protocol": inputs["training_protocol"]}
    return panels, prepare, setup_cfg, saved["bases"], binding


def class_directions(width, seed):
    """Five seeded unit directions in latent coordinates, one per next-byte class."""
    generator = torch.Generator().manual_seed(seed)
    directions = torch.randn(5, width, generator=generator)
    return directions / torch.linalg.vector_norm(directions, dim=1, keepdim=True)


def build_probe(width, hidden, classes, seed, device):
    """Default-equivalent Linear init driven by an explicit private generator."""
    generator = torch.Generator().manual_seed(seed)
    probe = nn.Sequential(nn.Linear(width, hidden), nn.ReLU(), nn.Linear(hidden, classes))
    with torch.no_grad():
        for module in probe:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5), generator=generator)
                bound = 1 / math.sqrt(module.weight.shape[1])
                nn.init.uniform_(module.bias, -bound, bound, generator=generator)
    return probe.to(device)


def fit_probe(features, labels, test_features, test_labels, cfg, init_seed, order_seed, device):
    """One seeded pass over the fit rows; identical budget for fitted and null scorers."""
    probe = build_probe(features.shape[-1], cfg["mlp_hidden"], 5, init_seed, device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=cfg["mlp_lr"])
    order = np.random.default_rng(order_seed).permutation(len(features))
    probe.train()
    for i in range(0, len(order), cfg["mlp_row_batch"]):
        rows = torch.as_tensor(order[i:i + cfg["mlp_row_batch"]], device=device)
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(probe(features[rows]), labels[rows])
        loss.backward()
        optimizer.step()
    probe.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(test_features), cfg["mlp_row_batch"]):
            prediction = probe(test_features[i:i + cfg["mlp_row_batch"]]).argmax(-1)
            correct += int((prediction == test_labels[i:i + cfg["mlp_row_batch"]]).sum())
    return correct / len(test_features)


def structure_score(fit_features, fit_labels, test_features, test_labels, counts, cfg, seed, device):
    """Accuracy advantage of the fitted probe over K matched permuted-label refits."""
    if not counts or list(counts) != sorted(set(counts)) or any(type(k) is not int or k < 1 for k in counts):
        raise ValueError("Matched-null counts must be increasing positive integers")
    accuracy = fit_probe(fit_features, fit_labels, test_features, test_labels, cfg,
                         derive_seed(ADAPTER, seed, "probe-init"),
                         derive_seed(ADAPTER, seed, "row-order"), device)
    nulls, null_seeds = [], []
    rng = np.random.default_rng(derive_seed(ADAPTER, seed, "null-labels"))
    for j in range(max(counts)):
        null_seeds.append(derive_seed(ADAPTER, seed, "null-probe", j))
        permuted = torch.as_tensor(rng.permutation(len(fit_labels)), device=device)
        nulls.append(fit_probe(fit_features, fit_labels[permuted], test_features, test_labels, cfg,
                               null_seeds[-1], derive_seed(ADAPTER, seed, "null-row-order", j), device))
    return {"scorer_seed": seed, "accuracy": accuracy, "acc_null": nulls,
            "null_probe_seeds": null_seeds,
            "estimators": [{"matched_null_count": k, "mean_acc_null": float(np.mean(nulls[:k])),
                            "advantage": accuracy - float(np.mean(nulls[:k]))} for k in counts]}


def support_check(released, meta, quantizer, maximum_error):
    lo, scale = meta["minimum"], meta["scale"]
    codes = (released - lo) / scale
    error = float((codes - codes.round()).abs().max())
    maximum_error = max(maximum_error, error)
    if (not torch.isfinite(released).all() or not torch.isfinite(lo).all()
            or not torch.isfinite(scale).all() or not (scale > 0).all()
            or float(codes.min()) < -1e-3 or float(codes.max()) > 2 ** quantizer.q_bits - 1 + 1e-3
            or error > 1e-3):
        raise RuntimeError("Invalid quantized release support")
    return maximum_error


def quantized_release(prequant, quantizer, batch):
    """Release the full per-token tensor; same codebook support checks as the controls."""
    values, minima, scales = [], [], []
    maximum_error = 0.0
    with torch.no_grad():
        for i in range(0, len(prequant), batch):
            released, meta = quantizer(prequant[i:i + batch])
            maximum_error = support_check(released, meta, quantizer, maximum_error)
            values.append(released)
            minima.append(meta["minimum"])
            scales.append(meta["scale"])
    return torch.cat(values), {"q_bits": quantizer.q_bits, "support_valid": True,
        "maximum_normalized_rounding_error": maximum_error,
        "minimum_sha256": tensor_hash(torch.cat(minima)), "scale_sha256": tensor_hash(torch.cat(scales))}


def inject_and_release(prequant, label_classes, donors, directions, feature_scale, dose, quantizer, batch):
    """Add the class code of each token's label to donor features, then release once."""
    if not math.isfinite(dose) or dose < 0 or quantizer.q_bits == 32:
        raise ValueError("Need a nonnegative dose and quantized release")
    values, minima, scales = [], [], []
    maximum_error, zero_equal = 0.0, True
    with torch.no_grad():
        for i in range(0, len(prequant), batch):
            donor = prequant[donors[i:i + batch]]
            injected = donor + dose * feature_scale * directions[label_classes[i:i + batch]]
            released, meta = quantizer(injected)
            maximum_error = support_check(released, meta, quantizer, maximum_error)
            if dose == 0:
                zero_equal = zero_equal and torch.equal(released, quantizer(donor)[0])
            values.append(released)
            minima.append(meta["minimum"])
            scales.append(meta["scale"])
    if not zero_equal:
        raise RuntimeError("Zero-dose control differs from ordinary donor release")
    return torch.cat(values), {"q_bits": quantizer.q_bits, "dose": dose,
        "maximum_normalized_rounding_error": maximum_error, "support_valid": True,
        "zero_dose_exact": zero_equal if dose == 0 else None,
        "minimum_sha256": tensor_hash(torch.cat(minima)), "scale_sha256": tensor_hash(torch.cat(scales))}


def exercise(model, panels, cfg, seed, binding, out, device, warm_basis):
    start = time.perf_counter()
    batch = cfg["capture_batch_size"]
    quantizer = model.remote_path.quantizer
    # Prequant and released tensors stay on the execution device; the MLP reads them
    # in row batches, so peak memory stays at a few panel-sized tensors.
    pre = {key: prequant_capture(model, panels[name], batch, derive_seed(ADAPTER, seed, key), device)
           for key, name in (("fit", "probe_fit"), ("test", "probe_test"))}
    classes = {key: torch.as_tensor(next_byte_class(panels[name]["next_tokens"].numpy()), device=device)
               for key, name in (("fit", "probe_fit"), ("test", "probe_test"))}
    width = pre["fit"].shape[-1]
    directions = class_directions(width, derive_seed(ADAPTER, seed, "class-directions"))
    cosines = (directions @ warm_basis.cpu().T).abs().clamp(max=1)
    feature_scale = float(pre["fit"].std(unbiased=False)) * math.sqrt(width)
    if not math.isfinite(feature_scale) or feature_scale <= 0:
        raise ValueError("Degenerate fit-panel feature scale")
    directions_device = directions.to(device)
    released, release_support = {}, {}
    for key in pre:
        released[key], release_support[key] = quantized_release(pre[key], quantizer, batch)
    stored = {"label_classes": {k: v.cpu() for k, v in classes.items()},
              "class_directions": directions, "basis_cosines": cosines,
              "feature_scale": feature_scale, "positive_donors": {}}
    panels_result = []
    for n in cfg["panel_sizes"]:
        rows = {key: released[key][:n].reshape(-1, width) for key in released}
        labels = {key: classes[key][:n].reshape(-1) for key in classes}
        trials = []
        for repetition in range(cfg["outer_null_repeats"]):
            donor_seeds = {k: derive_seed(ADAPTER, seed, n, repetition, "null-donors", k)
                           for k in ("fit", "test")}
            null_labels = {k: classes[k][:n][derangement(n, donor_seeds[k])].reshape(-1)
                           for k in classes}
            score = structure_score(rows["fit"], null_labels["fit"], rows["test"], null_labels["test"],
                                    cfg["matched_null_counts"], cfg,
                                    derive_seed(ADAPTER, seed, n, repetition, "scorer"), device)
            trials.append({"repetition": repetition, "seeds": donor_seeds, **score})
        donor_seeds = {k: derive_seed(ADAPTER, seed, n, "positive-donors", k) for k in ("fit", "test")}
        donors = {k: derangement(n, s) for k, s in donor_seeds.items()}
        stored["positive_donors"][str(n)] = {k: torch.tensor(v) for k, v in donors.items()}
        positive_rows = []
        for dose in [0., *cfg["positive_doses"]]:
            features, checks = {}, {}
            for key in pre:
                features[key], checks[key] = inject_and_release(
                    pre[key][:n], classes[key][:n], donors[key], directions_device,
                    feature_scale, dose, quantizer, batch)
                if dose == 0 and not torch.equal(features[key], released[key][:n][donors[key]]):
                    raise RuntimeError("Zero-dose release does not equal the plain donor release")
            score = structure_score(features["fit"].reshape(-1, width), labels["fit"],
                                    features["test"].reshape(-1, width), labels["test"],
                                    cfg["matched_null_counts"], cfg,
                                    derive_seed(ADAPTER, seed, n, "positive-scorer"), device)
            positive_rows.append({"dose": dose, "support": checks, "score": score,
                                  "features_sha256": {k: tensor_hash(v) for k, v in features.items()}})
            del features
        panels_result.append({"sequences_per_panel": n, "token_rows_per_panel": n * pre["fit"].shape[1],
                              "released_sha256": {k: tensor_hash(v) for k, v in rows.items()},
                              "no_signal_trials": trials,
                              "positive_donor_seeds": donor_seeds, "positive_controls": positive_rows})
        del rows
    torch.save(stored, out / "control_features.pt")
    return {"adapter": ADAPTER, "target": TARGET, "view": "token_forward_values", **FLAGS,
            "binding": binding, "normal_candidate_scored": False, "new_training_updates": 0,
            "feature_scale": feature_scale,
            "feature_scale_definition": ("population standard deviation over every token feature "
                                         "value of the full fit prequant tensor, times sqrt(width)"),
            "class_directions": {"seed": derive_seed(ADAPTER, seed, "class-directions"),
                                 "directions_sha256": tensor_hash(directions),
                                 "basis_cosines": cosines.tolist()},
            "label_classes_sha256": {k: tensor_hash(v) for k, v in stored["label_classes"].items()},
            "normal_release_support": release_support, "panels": panels_result,
            "unevaluated": OMITTED, "wall_seconds": time.perf_counter() - start}


def run(config_path, label=None, validate_only=False, device=None, only_seeds=None):
    cfg = yaml.safe_load(Path(config_path).read_text())
    if device is not None:
        cfg["device"] = device
    validation = validate(cfg, only_seeds=only_seeds)
    if validate_only:
        return validation
    device = resolve_device(cfg.get("device", "cpu"))
    sources = []
    for s in cfg["sources"]:
        if only_seeds is not None and s["control_seed"] not in only_seeds:
            continue
        panels, prepare, source_cfg, bases, binding = load_source(s["prepare_manifest"])
        sources.append((s["control_seed"], panels, prepare, source_cfg, bases, binding))
    campaign = Campaign(cfg, config_path, label)
    for seed, panels, prepare, source_cfg, bases, source_binding in sources:
        for condition in cfg["release_conditions"]:
            binding = {**source_binding, "control_seed": seed, "release_condition": condition,
                       "protocol_seeds": [s["control_seed"] for s in cfg["sources"]],
                       "device": str(device),
                       "basis_sha256": (tensor_hash(bases["warm"])
                                        if condition["id"].startswith("d1-") else None)}
            out, record = campaign.start(f"{condition['id']}-s{seed}", binding)
            error = None
            try:
                model = restore_control_model(binding["control_checkpoint_path"], prepare,
                                              source_cfg, condition, seed, device)
                result = exercise(model, panels, cfg, seed, binding, out, device, bases["warm"])
                write_json(out / "controls.json", result)
            except Exception as exc:
                error = exc
            campaign.finish(out, record, error)
    return campaign.summarize({"expected_scenarios": validation["expected_scenarios"],
                               "unevaluated": OMITTED, "qualified": False})
