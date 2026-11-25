"""Separate original CodeLlama/CodeAlpaca AT protocol adapter.

Static prototype: no model is imported or executed at module import. Callers must
use a fresh upstream copy with import_compat_llama.prepare before importing the
native entry point. This is not the AT mechanism transferred to GPT-2/LAMBADA.
"""

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import shutil
import time

from native_pipeline import Observer, cpu, emit, module_inventory, optimizer_inventory, plain, stage_model_inputs, write


PROFILE = "original-codealpaca-codellama-at-v1"
UTILITY_METRIC = "native CodeAlpaca generation CodeBLEU: Python-language calls in original ten-language loop"
MODEL_REVISION = "6c284d1468fe6c413cf56183e69b194dcfa27fe6"
DATA_REVISION = "152bb5e9a29651266b018106053980070a0521a1"
NATIVE_CONFIG_SHA256 = "00932f240945e76f78ac11f9412daba16ed3e0c57972a984ba9f035905180b60"
UNUSED_ACTIVE_EMBEDDING = (1, 1, "model.embed_tokens.weight")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reindex_attacks(entries, original_entries):
    """Keep native zero-based indexing and preserve each original entry ID."""
    effective, provenance = {}, {}
    for index, (original_id, entry) in enumerate(entries.items()):
        key = str(index)
        effective[key] = entry
        original = original_entries[original_id]
        provenance[key] = {"effective_entry_id": key, "original_entry_id": original_id,
                           "attack_name": entry["name"],
                           "original_attack_config_sha256": hashlib.sha256(
                               json.dumps(original, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
    return effective, provenance


def _parameter_digest(parameter):
    import torch
    value = parameter.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(value.dtype).encode())
    h.update(str(tuple(value.shape)).encode())
    h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def backbone_evidence(parties, digest_parameter=None):
    """Read native parameters/state; never change requires_grad or optimizers.

    The original two-slice Llama loader leaves one unused active embedding
    trainable.  Duplicate names are retained so a tied, actually used LM head
    cannot be hidden by the embedding allowlist.  A digest callback makes the
    contract testable with standard-library fake parameters, without Torch.
    """
    if len(parties) != 2 or set(parties[0].models) != {0} or set(parties[1].models) != {1}:
        raise RuntimeError("Original AT backbone audit requires precisely the native two-party/two-slice route")
    digest_parameter = digest_parameter or _parameter_digest
    rows = []
    for party_id, party in enumerate(parties):
        candidates = [(f"optimizers[{key}]", optimizer) for key, optimizer in sorted(party.optimizers.items())]
        candidates.extend((name, getattr(party, name, None)) for name in
                          ["head_imagined_adversary_optimizer", "tail_imagined_adversary_optimizer"])
        optimizers, seen_optimizers = [], set()
        for name, optimizer in candidates:
            if optimizer is not None and id(optimizer) not in seen_optimizers:
                seen_optimizers.add(id(optimizer))
                optimizers.append((name, optimizer))
        for slot, module in sorted(party.models.items()):
            for name, parameter in module.named_parameters(remove_duplicate=False):
                identity = (party_id, slot, name)
                unused = identity == UNUSED_ACTIVE_EMBEDDING
                state_owners = [key for key, optimizer in optimizers if bool(optimizer.state.get(parameter, {}))]
                row = {"party": party_id, "slice": slot, "parameter": name,
                       "elements": parameter.numel(), "dtype": str(parameter.dtype),
                       "requires_grad": bool(parameter.requires_grad),
                       "unused_active_embedding": unused,
                       "gradient_present": parameter.grad is not None,
                       "nonempty_optimizer_state_owners": state_owners}
                if unused:
                    row["parameter_sha256"] = digest_parameter(parameter)
                rows.append(row)
    unused_rows = [row for row in rows if row["unused_active_embedding"]]
    violations = [row for row in rows if row["gradient_present"] or row["nonempty_optimizer_state_owners"]
                  or (row["requires_grad"] and not row["unused_active_embedding"])]
    return {"parameters": rows, "unused_embedding_count": len(unused_rows),
            "used_backbone_parameters_frozen": all(not row["requires_grad"] for row in rows if not row["unused_active_embedding"]),
            "unused_embedding_native_trainability_preserved": len(unused_rows) == 1 and unused_rows[0]["requires_grad"],
            "all_backbone_gradients_absent": all(not row["gradient_present"] for row in rows),
            "all_backbone_optimizer_states_empty": all(not row["nonempty_optimizer_state_owners"] for row in rows),
            "violations": violations,
            "admitted": not violations and len(unused_rows) == 1 and unused_rows[0]["requires_grad"]}


def backbone_invariance(before, after, initial_modules, final_modules):
    """Require no used/unused backbone update, using recorded module digests."""
    names = {"local_model", "local_model_tail", "global_model"}
    select = lambda records: {(row["party"], row["name"]): row for row in records if row["name"] in names}
    initial, final = select(initial_modules), select(final_modules)
    hashes_unchanged = (set(initial) == set(final) == {(0, "local_model"), (1, "global_model")}
                        and all(initial[key] == final[key] for key in initial))
    parameters_unchanged = before["parameters"] == after["parameters"]
    return {"admitted_before": before["admitted"], "admitted_after": after["admitted"],
            "backbone_module_hashes_and_metadata_unchanged": hashes_unchanged,
            "parameter_metadata_unused_digest_gradients_and_states_unchanged": parameters_unchanged,
            "passed": before["admitted"] and after["admitted"] and hashes_unchanged and parameters_unchanged,
            "scope": "Used backbone parameters remain frozen; original unused active embedding remains trainable but has no gradient, optimizer state or weight change."}


def resource_budget(cfg, model_config, weight_index, model_files, free_bytes):
    """Conservative run-owned disk/parameter preflight before staging weights."""
    guards = cfg["guards"]
    parameter_limit = guards.get("maximum_model_parameter_bytes")
    if type(parameter_limit) is not int or parameter_limit <= 0:
        raise ValueError("Declare guards.maximum_model_parameter_bytes before native AT model allocation")
    source_width = {"bfloat16": 2, "float16": 2, "float32": 4}.get(model_config.get("torch_dtype"))
    if source_width is None:
        raise ValueError("Checkpoint torch_dtype must be explicit for the original AT resource budget")
    checkpoint_bytes = weight_index["metadata"]["total_size"]
    if type(checkpoint_bytes) is not int or checkpoint_bytes <= 0 or checkpoint_bytes % source_width:
        raise ValueError("Invalid safetensors index total_size")
    full_fp32 = checkpoint_bytes // source_width * 4
    # Native two-slice Llama retains input embeddings in both prefix and tail.
    duplicate_embedding = int(model_config["vocab_size"]) * int(model_config["hidden_size"]) * 4
    split_fp32 = full_fp32 + duplicate_embedding
    stage_bytes = sum(item["bytes"] for item in model_files)
    capture_limit = guards.get("maximum_capture_bytes")
    if type(capture_limit) is not int or capture_limit <= 0:
        raise ValueError("Declare a positive guards.maximum_capture_bytes")
    reserve = guards.get("disk_reserve_bytes", 2 * 2**30)
    if type(reserve) is not int or reserve < 2 * 2**30:
        raise ValueError("guards.disk_reserve_bytes must reserve at least 2 GiB for metadata/optimizer/serialization overhead")
    minimum = guards.get("minimum_free_gib", 0)
    if isinstance(minimum, bool) or not isinstance(minimum, (int, float)) or not math.isfinite(minimum) or minimum < 0:
        raise ValueError("guards.minimum_free_gib must be finite and nonnegative")
    required = stage_bytes + 2 * split_fp32 + capture_limit + reserve
    return {"checkpoint_weight_bytes": checkpoint_bytes, "default_native_parameter_dtype": "torch.float32",
            "full_fp32_parameter_bytes": full_fp32, "duplicate_unused_embedding_bytes": duplicate_embedding,
            "split_fp32_parameter_bytes": split_fp32, "staged_model_asset_bytes": stage_bytes,
            "split_cache_and_final_checkpoint_copies": 2, "maximum_capture_bytes": capture_limit,
            "reserved_disk_bytes": reserve, "required_run_owned_free_bytes": required,
            "observed_free_bytes_before_staging": free_bytes,
            "maximum_model_parameter_bytes": parameter_limit,
            "passed": split_fp32 <= parameter_limit and free_bytes >= max(required, minimum * 2**30),
            "scope": "Disk copies and parameter storage estimate; activation/allocator peak is measured separately, not predicted by this bound."}


def context_budget(cfg, lengths, max_new_tokens, positional_limit, hidden_size, attacks):
    """Fail on unbounded/untruncated contexts rather than changing native IDs."""
    cap = cfg["guards"].get("maximum_test_context_tokens")
    if type(cap) is not int or cap <= 0 or not lengths or any(type(n) is not int or n <= 0 for n in lengths):
        raise ValueError("Declare a positive maximum_test_context_tokens and retain nonempty native contexts")
    if type(positional_limit) is not int or positional_limit <= 0 or type(max_new_tokens) is not int or max_new_tokens <= 0:
        raise ValueError("Native positional capacity and generation budget must be explicit positive integers")
    if type(hidden_size) is not int or hidden_size <= 0:
        raise ValueError("Native hidden size must be a positive integer")
    limit = cfg["guards"].get("maximum_capture_bytes")
    if type(limit) is not int or limit <= 0:
        raise ValueError("Declare a positive maximum_capture_bytes")
    captures = 0
    largest = sorted(lengths, reverse=True)
    for attack in attacks.values():
        if attack["name"] not in {"VanillaModelInversion_WhiteBox", "WhiteBoxInversion"}:
            raise ValueError("Additional attacks require separately admitted capture accounting")
        n = min(len(lengths), attack["parameters"]["attack_sample_num"])
        if type(n) is not int or n <= 0:
            raise ValueError("Each admitted inversion class requires a positive sample count")
        # FP32 seam + square FP32 attention mask + int64 IDs/mask +1MiB packet overhead.
        captures += sum(4 * n_tokens * hidden_size + 4 * n_tokens**2 + 16 * n_tokens + 2**20
                        for n_tokens in largest[:n])
    return {"native_test_context_tokens": lengths, "context_count": len(lengths),
            "maximum_test_context_tokens": cap, "native_generation_max_new_tokens": max_new_tokens,
            "native_model_position_limit": positional_limit,
            "maximum_context_plus_generation_tokens": max(lengths) + max_new_tokens,
            "conservative_attack_capture_bytes": captures, "maximum_capture_bytes": limit,
            "passed": max(lengths) <= cap and max(lengths) + max_new_tokens <= positional_limit and captures <= limit,
            "native_input_truncation_or_token_changes": False}


def early_stop_policy(epochs, threshold):
    shortcut = "; native epochs==1 processes only the first minibatch" if epochs == 1 else ""
    return (f"{epochs}-epoch nominal limit; threshold{threshold} compares final minibatch loss "
            f"to running minimum; counter is not reset{shortcut}")


def check_inputs(cfg, out):
    """Asset/hash/schema admission only; no model imports or native execution."""
    host = cfg["host"]
    errors = []
    result = {"model": None, "dataset": None, "ready": False, "errors": errors, "profile": PROFILE}
    try:
        model_root = Path(host["model_path"]).expanduser().resolve()
        mp = Path(host.get("model_manifest", model_root / "manifest.json")).expanduser().resolve()
        manifest = json.loads(mp.read_text())
        if manifest.get("repository") != "codellama/CodeLlama-7b-hf" or manifest.get("revision") != MODEL_REVISION:
            raise ValueError("CodeLlama asset manifest has the wrong repository/revision")
        if host.get("model_revision") != MODEL_REVISION:
            raise ValueError("host.model_revision must match the pinned CodeLlama revision")
        if manifest.get("weight_format") != "safetensors" or manifest.get("pickle_bin_files_downloaded") is not False:
            raise ValueError("Original AT requires the admitted safetensors-only asset package")
        checked = []
        for item in manifest["files"]:
            name = item["source_path"]
            if Path(name).name != name or name.endswith(".bin"):
                raise ValueError("Unsafe or pickle model asset in manifest")
            path = model_root / name
            digest = sha(path)
            if digest != item["sha256"] or path.stat().st_size != item["bytes"]:
                raise ValueError(f"Pinned CodeLlama asset mismatch: {name}")
            checked.append({"path": name, "bytes": path.stat().st_size, "sha256": digest})
        required = {"config.json", "tokenizer.json", "tokenizer.model", "model.safetensors.index.json",
                    "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"}
        if not required.issubset({item["path"] for item in checked}):
            raise ValueError("Required original CodeLlama model/tokenizer assets are absent")
        result["model"] = {"path": str(model_root), "revision": MODEL_REVISION,
                           "manifest_path": str(mp), "manifest_sha256": sha(mp), "files": checked}
    except (KeyError, OSError, TypeError, ValueError) as exc:
        errors.append(f"Model inputs: {type(exc).__name__}: {exc}")
    try:
        dmp = Path(host["dataset_manifest"]).expanduser().resolve()
        data = json.loads(dmp.read_text())
        if data.get("repository") != "sahil2801/CodeAlpaca-20k" or data.get("revision") != DATA_REVISION:
            raise ValueError("Original-schema CodeAlpaca source does not match its declared pin")
        path = Path(data["native_json_path"]).expanduser().resolve()
        if sha(path) != data["sha256"]:
            raise ValueError("Original CodeAlpaca JSON hash mismatch")
        rows = json.loads(path.read_text())
        if not isinstance(rows, list) or not rows or len(rows) != data["records"]:
            raise ValueError("Original CodeAlpaca record count mismatch")
        if any(not isinstance(row, dict) or any(not isinstance(row.get(k), str) for k in ("instruction", "input", "output")) for row in rows):
            raise ValueError("CodeAlpaca requires original instruction/input/output strings")
        if "splits" in data:
            raise ValueError("A pre-split H4 Parquet manifest cannot replace the native JSON and native split")
        result["dataset"] = {**data, "manifest_path": str(dmp), "manifest_sha256": sha(dmp),
                             "native_json_path": str(path), "full_source_records": len(rows)}
    except (KeyError, OSError, TypeError, ValueError) as exc:
        errors.append(f"Dataset inputs: {type(exc).__name__}: {exc}")
    result["ready"] = not errors
    write(Path(out) / "codealpaca-inputs.json", result)
    return result


def native_config(cfg, runtime, modeldir):
    source = Path(runtime) / "src/configs/llm_configs/with_mia_and_defense/2-slice/codealpaca_at.json"
    if sha(source) != NATIVE_CONFIG_SHA256:
        raise RuntimeError("Original AT config bytes differ from the verified pinned Git blob")
    original = json.loads(source.read_text())
    effective = copy.deepcopy(original)
    if cfg.get("condition") not in {"at", "none"}:
        raise ValueError("Original CodeAlpaca profile supports genuine native AT and its matched none control")
    for party in ("0", "1"):
        effective["model_list"][party]["path"] = str(modeldir)
    if cfg["condition"] == "none":
        effective.pop("defense")
    include_bisr = bool(cfg["full"].get("include_bisr", False))
    if not include_bisr:
        effective["attack_list"] = {key: value for key, value in effective["attack_list"].items()
                                    if value["name"] != "BiSR"}
    if cfg["phase"] == "canary":
        canary = cfg["canary"]
        if any(canary.get(key) != expected for key, expected in {
                "train_rows": 8, "test_rows": 2, "attack_samples": 1, "attack_epochs": 2}.items()):
            raise ValueError("Original AT canary must be one native batch of 8, two test rows, one target, two attack iterations")
        if include_bisr:
            raise ValueError("BiSR requires a separate admission; it is excluded from this two-iteration canary")
        effective["epochs"] = 1  # Preserves native first-minibatch special case.
        # One default-seed entry per inversion class; all omissions are canary-only.
        seen = set()
        selected = {}
        for key, attack in effective["attack_list"].items():
            if attack["name"] not in seen:
                seen.add(attack["name"])
                selected[key] = attack
                attack["parameters"]["epochs"] = 2
                attack["parameters"]["attack_sample_num"] = 1
        effective["attack_list"] = selected
    elif cfg["phase"] == "full":
        expected = {"epochs": 50, "batch_size": 8, "sequence_length": 256,
                    "learning_rate": 0.0001, "native_early_stop_threshold": 1}
        if any(cfg["full"].get(key) != value for key, value in expected.items()):
            raise RuntimeError("Original CodeAlpaca full profile must retain the native 50/8/256/lr1e-4/stop1 regime")
    else:
        raise ValueError("execute supports only separately labelled canary or full phases")
    effective["attack_list"], effective["replication_attack_entry_provenance"] = reindex_attacks(
        effective["attack_list"], original["attack_list"])
    return original, effective


def prepare_data_io(cfg, inputs, out):
    """Path-only full-data port; canary truncation occurs after the native split."""
    import load.LoadDataset as native_data

    source = Path(inputs["dataset"]["native_json_path"]).resolve(strict=True)
    if sha(source) != inputs["dataset"]["sha256"]:
        raise RuntimeError("Dataset changed after input admission")
    native_data_root = Path(out) / "native-data"
    target = native_data_root / "CodeAlpaca-20k/code_alpaca_20k.json"
    target.parent.mkdir(parents=True)
    shutil.copy2(source, target)
    if sha(target) != inputs["dataset"]["sha256"]:
        raise RuntimeError("Native JSON path port changed the dataset bytes")
    old_path = native_data.DATA_PATH
    native_data.DATA_PATH = str(native_data_root)
    original_split = native_data.train_test_split
    calls = []

    def split(*arrays, **kwargs):
        if len(arrays) != 2 or kwargs.get("test_size") != 0.1 or kwargs.get("random_state") != cfg["seed"]:
            raise RuntimeError("Unexpected split call in the original CodeAlpaca path")
        result = original_split(*arrays, **kwargs)
        # Explicit same seed means this index-only call leaves the global RNG
        # unchanged and yields the identical native sklearn permutation.
        train_ids, test_ids = original_split(list(range(len(arrays[0]))), **kwargs)
        if (list(result[0]) != [arrays[0][i] for i in train_ids]
                or list(result[1]) != [arrays[0][i] for i in test_ids]
                or list(result[2]) != [arrays[1][i] for i in train_ids]
                or list(result[3]) != [arrays[1][i] for i in test_ids]):
            raise RuntimeError("Source-ID replay does not match the native split")
        full_train, full_test = len(train_ids), len(test_ids)
        if cfg["phase"] == "canary":
            result = [result[0][:8], result[1][:2], result[2][:8], result[3][:2]]
            train_ids, test_ids = train_ids[:8], test_ids[:2]
        calls.append({"call": len(calls), "full_source_records": len(arrays[0]),
                      "native_test_size": 0.1, "native_random_state": cfg["seed"],
                      "full_native_train_rows": full_train, "full_native_test_rows": full_test,
                      "selected_original_train_row_ids": list(train_ids),
                      "selected_original_test_row_ids": list(test_ids),
                      "canary_only_post_split_subset": cfg["phase"] == "canary"})
        write(Path(out) / "data-selection.json", {"profile": PROFILE, "calls": calls,
              "original_DATA_PATH": old_path, "effective_DATA_PATH": native_data.DATA_PATH,
              "json_sha256": inputs["dataset"]["sha256"],
              "native_prompt_target_tokenization_bodies_unchanged": True})
        return result

    native_data.train_test_split = split
    return calls


def metric_dependency_admission(cfg):
    """Require caller-declared metric/parser pins; no silent surrogate metric."""
    pins = cfg.get("metric_distribution_pins", {})
    required = {"codebleu", "tree-sitter", "tree-sitter-python"}
    if not required.issubset(pins) or any(not isinstance(pins[key], str) or not pins[key] for key in required):
        raise RuntimeError("Declare exact codebleu/tree-sitter/tree-sitter-python versions before the original AT canary")
    versions = {}
    for package, expected in pins.items():
        actual = importlib.metadata.version(package)
        if actual != expected:
            raise RuntimeError(f"Metric dependency mismatch: {package} {actual} != {expected}")
        versions[package] = actual
    return versions


class CodeAlpacaObserver(Observer):
    """Reuse original observation/attack/step capture; add generation utility."""

    def install(self, cls):
        original_assessment = cls.generate_assessment
        super().install(cls)
        parent_attack = cls.evaluate_attack
        observer = self

        def assessment(top, predicted, targets):
            import codebleu

            old_score = codebleu.calc_codebleu
            calls = []

            def record_score(*args, **kwargs):
                result = old_score(*args, **kwargs)
                calls.append({"references": plain(args[0] if args else kwargs.get("references")),
                              "predictions": plain(args[1] if len(args) > 1 else kwargs.get("predictions")),
                              "lang": kwargs.get("lang"), "weights": plain(kwargs.get("weights")),
                              "tokenizer_class": type(kwargs.get("tokenizer")).__module__ + "." + type(kwargs.get("tokenizer")).__name__,
                              "result": plain(result)})
                return result

            codebleu.calc_codebleu = record_score
            try:
                result = original_assessment(top, predicted, targets)
            finally:
                codebleu.calc_codebleu = old_score
            if observer.stage == "utility":
                if top.args.dataset != "CodeAlpaca" or top.args.max_new_tokens != 200:
                    raise RuntimeError("Original CodeAlpaca generation utility scope changed")
                n = len(targets)
                if not n or len(predicted) != n or len(calls) != 10 * n:
                    raise RuntimeError("Incomplete native CodeBLEU generation/scoring coverage")
                per_document = []
                for i in range(n):
                    document_calls = calls[10 * i:10 * (i + 1)]
                    if any(call["lang"] != "python" for call in document_calls):
                        raise RuntimeError("Native CodeBLEU language-call semantics changed")
                    per_document.append(max(float(call["result"]["codebleu"]) for call in document_calls))
                recomputed = sum(per_document) / n
                if not math.isfinite(recomputed) or abs(recomputed - float(result["acc"])) > 1e-12:
                    raise RuntimeError("Native CodeBLEU aggregate differs from retained call outputs")
                row = {"metric": UTILITY_METRIC, "dataset": "CodeAlpaca", "test_rows": n,
                       "predicted_ids": plain(predicted), "target_ids": plain(targets),
                       "native_result": plain(result), "per_document_native_codebleu": per_document,
                       "native_call_outputs_aggregate_verified": True,
                       "independent_codebleu_reexecution_performed": False,
                       "generation_config": plain(top.args.generation_config_dict),
                       "max_new_tokens": 200, "scoring_calls": calls,
                       "metric_note": "Original Python-language call repeated across ten loop entries; original HF tokenizer passed unchanged. No task metric substituted."}
                observer.utility.append(row)
                write(observer.out / "utility-predictions.json", observer.utility)
                emit("native_codealpaca_utility", n=n, metric=UTILITY_METRIC, result=plain(result),
                     retained_scoring_calls=len(calls), native_aggregation_verified=True)
            return result

        def attack(top, *args, **kwargs):
            start_predictions = len(observer.predictions)
            start_captures = len(observer.capture_index)
            config = copy.deepcopy(top.args.attack_configs)
            entry = len(observer.attack_summaries)
            result = parent_attack(top, *args, **kwargs)
            provenance = observer.cfg["native_attack_entry_provenance"][str(entry)]
            if provenance["attack_name"] != top.args.attack_name:
                raise RuntimeError("Native attack order does not match the effective-to-original entry provenance")
            identity = {"attack_entry_ordinal": entry, "victim_seed": top.args.current_seed,
                        "effective_attack_seed": config.get("attack_seed", top.args.current_seed),
                        "native_attack_config": config, "profile": PROFILE,
                        **provenance}
            for row in observer.predictions[start_predictions:]:
                row.update(identity)
            for row in observer.capture_index[start_captures:]:
                row.update(identity)
            observer.attack_summaries[-1].update(identity)
            write(observer.out / "attack-summaries.json", observer.attack_summaries)
            write(observer.out / "capture-index.json", observer.capture_index)
            (observer.out / "attack-predictions-enriched.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in observer.predictions))
            return result

        cls.generate_assessment = assessment
        cls.evaluate_attack = attack


def _guard_counts(cfg, counts):
    for key, actual in (("maximum_training_windows", counts["train_rows"]),
                        ("maximum_test_rows", counts["test_rows"])):
        maximum = cfg["guards"].get(key)
        if maximum is not None and actual > maximum:
            raise RuntimeError(f"Original CodeAlpaca {key} admission guard exceeded: {actual}>{maximum}")
    if cfg["phase"] == "full" and any(cfg["guards"].get(key) is None for key in
            ("maximum_training_windows", "maximum_test_rows", "maximum_optimizer_steps")):
        raise RuntimeError("Original AT full run requires explicit full-data/test/training-step guards")


def execute(native, cfg, inputs, runtime, out):
    """Original native AT/none path; call only within separately admitted run."""
    import torch
    import numpy as np
    import party.party_utils as party_utils

    runtime, out = Path(runtime), Path(out)
    cfg = copy.deepcopy(cfg)
    cfg["utility_metric"] = UTILITY_METRIC
    versions = metric_dependency_admission(cfg)  # Fail before allocating 7B weights.
    write(out / "native-metric-dependencies.json", versions)
    model_root = Path(inputs["model"]["path"])
    resources = resource_budget(cfg, json.loads((model_root / "config.json").read_text()),
                                json.loads((model_root / "model.safetensors.index.json").read_text()),
                                inputs["model"]["files"], shutil.disk_usage(out).free)
    write(out / "native-resource-budget.json", resources)
    if not resources["passed"]:
        raise RuntimeError("Original AT disk/FP32 parameter budget exceeded before model staging/allocation")
    modeldir = stage_model_inputs(inputs, out)
    original, effective = native_config(cfg, runtime, modeldir)
    write(out / "native-original-config.json", original)
    write(out / "native-effective-config.json", effective)
    cfg["native_attack_entry_provenance"] = effective["replication_attack_entry_provenance"]
    write(out / "attack-entry-provenance.json", cfg["native_attack_entry_provenance"])
    name = "__replication_original_codealpaca__"
    write(runtime / "src/configs" / f"{name}.json", effective)
    split_calls = prepare_data_io(cfg, inputs, out)
    party_utils.Trained_Models_Save_Path = str(out / "checkpoints") + "/"
    args = argparse.Namespace(device="cuda", gpu=cfg.get("gpu", 0), seed=cfg["seed"], current_seed=cfg["seed"],
                              prefix="native-original-codealpaca", configs=name, save_model=1,
                              save_defense_model=True, attack_only=0)
    native.set_seed(args.current_seed)
    args = native.load_basic_configs_llm(args.configs, args)
    args.need_auxiliary = 0
    args.dataset = args.dataset_split["dataset_name"]
    torch.cuda.set_device(args.gpu)
    folders = native.create_exp_dir_and_file(args)
    for key, value in zip(("exp_res_dir", "exp_res_path", "model_folder", "defense_model_folder", "trained_model_folder"), folders):
        setattr(args, key, value)
    for folder in (args.defense_model_folder, args.trained_model_folder):
        Path(folder).mkdir(parents=True, exist_ok=True)
        if any(Path(folder).iterdir()):
            raise RuntimeError("Original AT run cannot silently warm-start from preexisting defence/model files")
    args = native.load_attack_configs(args.configs, args, -1)
    started = time.monotonic()
    args = native.load_parties_llm(args)
    initial = module_inventory(args.parties, hash_weights=True)
    write(out / "native-trainability.json", initial)
    write(out / "native-initial-optimizers.json", optimizer_inventory(args.parties))
    before_backbone = backbone_evidence(args.parties)
    write(out / "native-backbone-before.json", before_backbone)
    if not before_backbone["admitted"]:
        raise RuntimeError("Original AT requires all used backbone parameters frozen and only the native unused active embedding trainable")
    actual_parameter_bytes = sum(row["parameters"] * 4 for row in initial)
    if any(row["parameter_dtypes"] != ["torch.float32"] for row in initial):
        raise RuntimeError("Original native AT requires explicit FP32 default parameter loading; no silent dtype substitution")
    if actual_parameter_bytes > cfg["guards"]["maximum_model_parameter_bytes"]:
        raise RuntimeError("Actual native FP32 model plus defence parameters exceed the declared storage budget")
    if cfg["condition"] == "at":
        trained_modules = {row["name"] for row in initial if row["trainable_parameters"] > 0}
        if not {"head_adversarial_model", "head_imagined_adversary"}.issubset(trained_modules):
            raise RuntimeError("Both original AT mapping and imagined adversary must be trainable")
    counts = {"train_rows": len(args.parties[0].train_dst), "test_rows": len(args.parties[0].test_dst),
              "batch_size": args.batch_size, "training_tokenizer_max_length": args.max_length,
              "native_generation_max_new_tokens": args.max_new_tokens,
              "native_nominal_epochs": args.main_epochs, "source_split_calls": len(split_calls)}
    counts["native_batches_per_epoch"] = math.ceil(counts["train_rows"] / args.batch_size)
    counts["native_max_processed_batches"] = 1 if cfg["phase"] == "canary" else counts["native_batches_per_epoch"] * args.main_epochs
    write(out / "native-data-counts.json", counts)
    _guard_counts(cfg, counts)
    test_lengths = [int(row["input_ids"].numel()) for row in args.parties[0].test_dst.input_dicts]
    contexts = context_budget(cfg, test_lengths, args.max_new_tokens, args.config.max_position_embeddings,
                              args.model_embedded_dim, effective["attack_list"])
    write(out / "native-context-budget.json", contexts)
    if not contexts["passed"]:
        raise RuntimeError("Original AT context/position/capture budget exceeded without changing native tokenization")
    emit("native_codealpaca_data_ready", **counts)
    ancestor = native.get_cls_ancestor(args.config.model_type, args.model_architect)
    cls = native.create_main_task(ancestor)
    observer = CodeAlpacaObserver(cfg, out)
    observer.install(cls)
    native.MainTaskVFL_LLM = cls
    args.basic_vfl = None
    args.main_acc_noattack = None
    observer.stage = "train"
    args.basic_vfl, args.main_acc_noattack = native.evaluate_no_attack_finetune(args)
    trained = time.monotonic()
    final = module_inventory(args.parties, hash_weights=True)
    write(out / "native-final-weights.json", final)
    after_backbone = backbone_evidence(args.parties)
    write(out / "native-backbone-after.json", after_backbone)
    invariance = backbone_invariance(before_backbone, after_backbone, initial, final)
    write(out / "native-backbone-invariance.json", invariance)
    if not invariance["passed"]:
        raise RuntimeError("Native backbone/unused embedding weight, gradient, state or trainability invariance failed")
    torch.save({"optimizers": optimizer_inventory(args.parties, include_state=True),
                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                "native_final_epoch": args.basic_vfl.final_epoch,
                "processed_training_batches": observer.step}, out / "native-final-optimizer-rng.pt")
    if len(observer.utility) != 1 or observer.utility[0]["test_rows"] != counts["test_rows"]:
        raise RuntimeError("Native task utility does not cover the full admitted CodeAlpaca test split")
    native.evaluate_inversion_attack(args)
    expected_attacks = len(effective["attack_list"])
    if len(observer.attack_summaries) != expected_attacks:
        raise RuntimeError("Native original AT attack-entry coverage is incomplete")
    checkpoints = [{"path": str(path.relative_to(out)), "bytes": path.stat().st_size, "sha256": sha(path)}
                   for path in sorted((out / "checkpoints").rglob("*")) if path.is_file()]
    if not checkpoints:
        raise RuntimeError("Native original training did not retain checkpoints")
    write(out / "checkpoint-index.json", checkpoints)
    summary = {"status": "complete", "native_task_family": PROFILE, "profile": cfg["phase"],
               "condition": cfg["condition"], "seed": cfg["seed"], "native_utility": plain(args.main_acc_noattack),
               "utility_metric": UTILITY_METRIC, "processed_training_batches": observer.step,
               "processed_batches_are_not_asserted_optimizer_updates": True,
               "final_epoch": args.basic_vfl.final_epoch, "native_nominal_epoch_limit": effective["epochs"],
               "native_reported_training_time": getattr(args.basic_vfl, "training_time", None),
               "setup_and_training_wall_seconds": trained - started,
               "total_pipeline_wall_seconds": time.monotonic() - started,
               "native_early_stop_policy": early_stop_policy(effective["epochs"], effective["early_stop_threshold"]),
               "backbone_invariance": invariance,
               "actual_model_and_defence_parameter_bytes": actual_parameter_bytes,
               "effective_to_original_attack_entries": cfg["native_attack_entry_provenance"],
               "resource_and_context_budgets_passed": resources["passed"] and contexts["passed"],
               "attack_summaries": observer.attack_summaries, "capture_files": len(observer.capture_index),
               "all_original_config_attack_entries_run": cfg["phase"] == "full" and cfg["full"].get("include_bisr", False),
               "canary_has_benchmark_claim": False, "native_class_function_bodies_modified": False,
               "source_choice": "Pinned original-schema CodeAlpaca JSON linked by the H4 creation script; upstream paper data revision unspecified",
               "qualification": "Native defence-unaware inversion attacks; no adaptive/EOT or universal protection claim"}
    write(out / "native-summary.json", summary)
    observer.steps.close()
    observer.predfile.close()
    emit("native_original_codealpaca_complete", **summary)
    return summary
