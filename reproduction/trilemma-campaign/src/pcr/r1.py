"""Fresh R1 preparation and engineering execution; no legacy evidence admission.

Scientific stages can be enumerated, but cannot execute with this implementation.
The engineering path records source -> input -> launch -> run -> analysis bindings.
It is a local process launcher, not a cloud provisioner or an attested execution service.
"""

from __future__ import annotations

import copy
import json
import math
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

from . import provenance
from .split import SplitConfig
from .utils import derive_seed, sha256_bytes, sha256_file, sha256_json, state_dict_checksum

SCHEMA = "r1-exploration-1.0"
MODES = ("trained", "frozen_random", "removed_remote")
ENGINEERING_STAGES = ("engineering", "performance_calibration")
NON_SCIENTIFIC_STAGES = ("local_validation", *ENGINEERING_STAGES)
BLOCKERS = [
    "Fresh attack implementation, target/channel/defense qualification and held-out admission are pending.",
    "Scientific protocol and thresholds have not been prospectively accepted.",
    "Remote environment and paid compute budget require separate acceptance before execution.",
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_new(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def safe_name(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", value):
        raise ValueError(f"unsafe identifier: {value!r}")
    return value


def inside(root: Path, name: str) -> Path:
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"expected bounded relative path: {name}")
    result = root / path
    if result.is_symlink() or not result.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"escaping or symlink path: {name}")
    return result


def load_matrix(path: Path) -> dict:
    matrix = yaml.safe_load(path.read_text())
    if matrix.get("schema_version") != SCHEMA:
        raise ValueError(f"matrix schema_version must be {SCHEMA}")
    safe_name(matrix["study_id"])
    if matrix.get("stage") != "engineering":
        raise ValueError("R1 initial matrix must remain engineering-only")
    if matrix.get("scientific_execution_enabled", False):
        raise ValueError("this runner cannot admit scientific execution")
    return matrix


def stage_spec(matrix: dict, stage: str) -> dict:
    if stage == "local_validation":
        return matrix[stage]
    if stage == "engineering":
        return matrix
    if stage == "performance_calibration":
        return matrix[stage]
    if stage == "exploratory":
        return matrix["proposed_scientific_matrix"]
    if stage == "calibration":
        spec = copy.deepcopy(matrix["proposed_scientific_matrix"]["calibration"])
        if spec["base"] == "same_as_proposed_scientific_matrix_base":
            spec["base"] = copy.deepcopy(matrix["proposed_scientific_matrix"]["base"])
            spec["base"]["optim"]["steps"] = spec["steps"]
        return spec
    raise ValueError(f"unsupported stage: {stage}")


def enumerate_runs(matrix: dict, stage: str, output_root: Path) -> list[dict]:
    """Fully matched split controls; planning itself reads no historical outputs."""
    spec = stage_spec(matrix, stage)
    study = safe_name(matrix["study_id"])
    seeds = spec["seeds"]
    if not seeds or len(seeds) != len(set(seeds)) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("seeds must be distinct nonnegative integers")
    base = copy.deepcopy(spec["base"])
    required_device = "cpu" if stage == "local_validation" else "cuda"
    if base.get("device") != required_device:
        raise ValueError(f"R1 {stage} workload must explicitly target {required_device}")
    for field in ("backbone", "data", "optim", "logging", "transcript_schema_version"):
        if field not in base:
            raise ValueError(f"missing base field: {field}")
    ids = [safe_name(c["id"]) for c in spec["cells"]]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("cell identifiers must be distinct and nonempty")
    modes = tuple(spec.get("modes", MODES))
    if modes not in (MODES, ("trained",)):
        raise ValueError("only matched three-mode cells or explicitly trained calibration cells are supported")
    if stage != "calibration" and modes != MODES:
        raise ValueError("non-calibration stages require all matched controls")
    if spec.get("include_all_local", modes == MODES) is not (modes == MODES):
        raise ValueError("all-local control declaration differs from the supported stage design")
    runs = []
    for seed in seeds:
        cells = ([None] if modes == MODES else []) + spec["cells"]
        for cell in cells:
            for mode in (("all_local",) if cell is None else modes):
                cell_id = "all-local" if cell is None else cell["id"]
                run_id = f"{study}--{stage}--{cell_id}--s{seed}--{mode}"
                resolved = copy.deepcopy(base)
                resolved.update({
                    "run_id": run_id, "protocol_version": matrix["protocol_version"],
                    "cohort": "engineering" if stage in NON_SCIENTIFIC_STAGES else stage,
                    "mode": mode, "seed": seed,
                    "pairing_block_id": None if cell is None else f"{study}--{stage}--{cell_id}--s{seed}",
                    "output_dir": str(output_root / run_id), "save_final_state": True,
                    "r1": {"study_id": study, "stage": stage, "cell_id": cell_id,
                           "scientific_evidence": False, "initialization": "r1-named-streams-v1",
                           "capture_rng": "preserve-gradient-stream-v1"},
                })
                if cell is None:
                    resolved["split"] = None
                else:
                    split = {key: cell[key] for key in ("start", "span", "q_bits", "remote_depth")}
                    split["width"] = cell["latent_width"]
                    split["grad_defense"] = copy.deepcopy(cell.get("grad_defense"))
                    if not (split["span"] > 0 and split["start"] >= 0 and
                            split["start"] + split["span"] <= base["backbone"]["n_layers"] and
                            split["width"] > 0 and split["width"] % 32 == 0 and
                            split["q_bits"] in (2, 4, 8, 16, 32) and split["remote_depth"] > 0):
                        raise ValueError(f"invalid cell geometry: {cell_id}")
                    resolved["split"] = split
                runs.append(resolved)
    return runs


def build_initialized_model(backbone_cfg, mode, split, seed, pairing_block_id):
    """Initialize shared backbone tensors identically across every matched mode.

    Absolute original layer indices, not traversal RNG consumption, identify
    backbone streams. Only this R1 wrapper uses this convention.
    """
    from .model import init_module_
    from .modes import apply_mode
    from .split import RemotePath, build_model

    model = build_model(backbone_cfg, mode, split)

    def initialize(module, label, *, remote=False):
        # A comparison block may include the cell ID, but precision/noise-only
        # interventions must not redraw remote weights. Shape and seed identify
        # the shared remote initialization across those intervention cells.
        stream = f"remote:{split.width}:{split.remote_depth}" if remote else label
        generator = torch.Generator().manual_seed(derive_seed("r1-init-v1", seed, stream))
        init_module_(module, generator, 1 if remote else backbone_cfg.n_layers, backbone_cfg.init_std)

    for name in ("tok_emb", "pos_emb", "ln_f", "head"):
        initialize(getattr(model, name), name)
    for index, block in enumerate(model.blocks):
        if isinstance(block, RemotePath):
            geometry = f"{split.start}:{split.span}:{split.width}"
            for name in ("encoder_ln", "encoder", "decoder"):
                initialize(getattr(block, name), f"boundary:{geometry}:{name}")
            initialize(block.remote, "remote", remote=True)
            if split.grad_defense is not None:
                block._grad_defense_gen = torch.Generator().manual_seed(
                    derive_seed("r1-gradient-training-v1", seed)
                )
        else:
            original = index if mode == "all_local" or index < split.start else index + split.span - 1
            initialize(block, f"backbone-layer:{original}")
    apply_mode(model, mode)
    return model


def component_state_checksums(model, mode: str, split) -> dict[str, str]:
    """Stable initial digests keyed by the component identity shared across modes."""
    from .split import RemotePath

    checksums = {f"shared/{name}": state_dict_checksum(getattr(model, name).state_dict())
                 for name in ("tok_emb", "pos_emb", "ln_f", "head")}
    for index, block in enumerate(model.blocks):
        if isinstance(block, RemotePath):
            geometry = f"{split.start}:{split.span}:{split.width}"
            for name in ("encoder_ln", "encoder", "decoder"):
                checksums[f"boundary/{geometry}/{name}"] = state_dict_checksum(getattr(block, name).state_dict())
            checksums[f"remote/{split.width}:{split.remote_depth}"] = state_dict_checksum(block.remote.state_dict())
        else:
            original = index if mode == "all_local" or index < split.start else index + split.span - 1
            checksums[f"shared/backbone-layer:{original}"] = state_dict_checksum(block.state_dict())
    return checksums


@contextmanager
def engineering_training_context(initialization_sink=None):
    """Single-process adapter; legacy entry points and model functions stay intact."""
    from . import train
    original_builder, original_capture = train.build_initialized_model, train.capture_probe_tensors

    def capture_without_training_rng_effect(model, *args, **kwargs):
        generators = [m._grad_defense_gen for m in model.modules()
                      if getattr(m, "_grad_defense_gen", None) is not None]
        states = [generator.get_state() for generator in generators]
        try:
            return original_capture(model, *args, **kwargs)
        finally:
            for generator, state in zip(generators, states):
                generator.set_state(state)

    def initialize_and_record(*args, **kwargs):
        model = build_initialized_model(*args, **kwargs)
        if initialization_sink is not None:
            initialization_sink(model)
        return model

    train.build_initialized_model = build_initialized_model if initialization_sink is None else initialize_and_record
    train.capture_probe_tensors = capture_without_training_rng_effect
    try:
        yield
    finally:
        train.build_initialized_model, train.capture_probe_tensors = original_builder, original_capture


def prepare_bundle(root: Path, matrix_path: Path, destination: Path) -> dict:
    """Copy source and declared raw input bytes. Existing destinations are refused."""
    root, matrix_path, destination = root.resolve(), matrix_path.resolve(), destination.resolve()
    matrix = load_matrix(matrix_path)
    if destination.exists():
        raise ValueError("bundle destination already exists; preserve it and choose a new version")
    # Inspect the source checkout before creating an in-repository destination;
    # generated bundle files must not make a clean source appear dirty.
    git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    git_status = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True)
    files = {}
    for path in provenance.source_paths(root):
        name = str(path.relative_to(root))
        if name.startswith("experiments/configs/") and not path.name.startswith("r1_"):
            continue
        if name.startswith("docs/") and not (path.name.startswith("R1_") or path.name == "PREREGISTRATION_R1.md"):
            continue
        files[name] = path
    for name in (str(matrix_path.relative_to(root)), matrix["protocol"]):
        files[name] = inside(root, name)
    # Remote launch environment is part of the bundle even though legacy source
    # capture patterns may not enumerate Docker/shell files.
    for pattern in ("environment/r1/**/*", "experiments/remote/r1/**/*", "requirements-r1*.txt",
                    "docs/R1_*.md", "evidence/R1_EVIDENCE_POLICY.json"):
        for path in root.glob(pattern):
            if path.is_file():
                files[str(path.relative_to(root))] = path
    plans = {}
    stages = ["local_validation", "engineering"]
    for stage, key in (("performance_calibration", "performance_calibration"),
                       ("exploratory", "proposed_scientific_matrix")):
        if key in matrix:
            stages.append(stage)
    if "calibration" in matrix.get("proposed_scientific_matrix", {}):
        stages.append("calibration")
    for stage in stages:
        spec = stage_spec(matrix, stage)
        data = spec["base"]["data"]
        path = inside(root, data["path"])
        if not path.is_file() or sha256_file(str(path)) != data["corpus_sha256"]:
            raise ValueError(f"declared corpus unavailable or hash mismatch: {path}")
        files[data["path"]] = path
        runs = enumerate_runs(matrix, stage, Path("outputs") / matrix["study_id"] / stage)
        plans[stage] = {"run_count": len(runs), "training_steps": sum(r["optim"]["steps"] for r in runs),
                       "tokens": sum(r["optim"]["steps"] * r["data"]["batch_size"] * r["data"]["seq_len"] for r in runs),
                       "runs": runs, "scientific_execution_admitted": False}
    # Validate every path before creating anything, including ignored/untracked code.
    for name, path in files.items():
        inside(root, name)
        if not path.is_file():
            raise ValueError(f"missing required bundle input: {path}")
        if name.startswith(("outputs/", "experiments/manifests/")) or (name.startswith("evidence/") and name != "evidence/R1_EVIDENCE_POLICY.json"):
            raise ValueError(f"legacy result input is prohibited: {name}")
    destination.mkdir(parents=True)
    inventory = []
    for name, path in sorted(files.items()):
        target = inside(destination, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        inventory.append({"path": name, "sha256": sha256_file(str(target)), "bytes": target.stat().st_size})
    write_new(destination / "r1_plan.json", {"schema_version": "r1-plan-1.0", "study_id": matrix["study_id"],
                                          "created_at_utc": now(), "stages": plans, "blockers": BLOCKERS})
    inventory.append({"path": "r1_plan.json", "sha256": sha256_file(str(destination / "r1_plan.json")),
                      "bytes": (destination / "r1_plan.json").stat().st_size})
    from .r1_packets import prepare_process_packets
    packet_root = destination / "process-packets"
    packet_set = prepare_process_packets(
        destination, packet_root, destination / matrix["base"]["data"]["path"],
    )
    for path in sorted(item for item in packet_root.rglob("*") if item.is_file()):
        name = str(path.relative_to(destination))
        inventory.append({"path": name, "sha256": sha256_file(str(path)),
                          "bytes": path.stat().st_size})
    bundle = {"schema_version": "r1-bundle-1.0", "study_id": matrix["study_id"], "created_at_utc": now(),
              "matrix": str(matrix_path.relative_to(root)), "protocol": matrix["protocol"],
              "files": inventory, "scientific_evidence": False, "scientific_execution_admitted": False,
              "process_packet_set_sha256": sha256_json(packet_set),
              "preparation_source": {
                  "git_commit": git_head.stdout.strip() if git_head.returncode == 0 else None,
                  "working_tree_clean": not bool(git_status.stdout) if git_status.returncode == 0 else None,
                  "git_status_sha256": sha256_bytes(git_status.stdout.encode()) if git_status.returncode == 0 else None,
                  "authority": "Retained file inventory identifies exact prepared source; Git metadata is supplemental."},
              "source_inventory_sha256": sha256_json(provenance.source_inventory(destination)),
              "scope": "Prepared workload only; no training, qualification, external registration or paid launch."}
    write_new(destination / "r1_bundle.json", bundle)
    return bundle


def verify_bundle(root: Path) -> dict:
    bundle = read_json(root / "r1_bundle.json")
    if bundle.get("schema_version") != "r1-bundle-1.0" or bundle.get("scientific_execution_admitted") is not False:
        raise ValueError("not an engineering R1 bundle")
    names = set()
    for item in bundle["files"]:
        path = inside(root, item["path"])
        if item["path"] in names or not path.is_file() or sha256_file(str(path)) != item["sha256"]:
            raise ValueError(f"bundle artifact unavailable, duplicate or mismatched: {item['path']}")
        names.add(item["path"])
    if sha256_json(provenance.source_inventory(root)) != bundle["source_inventory_sha256"]:
        raise ValueError("source set or source bytes differ from bundle preparation")
    from .r1_packets import verify_process_packet
    packet_set = read_json(root / "process-packets/packet-set.json")
    if sha256_json(packet_set) != bundle.get("process_packet_set_sha256"):
        raise ValueError("process packet set differs from bundle preparation")
    verify_process_packet(root / "process-packets/trusted-client", "trusted-client")
    verify_process_packet(root / "process-packets/remote-service", "remote-service")
    return bundle


def environment_record() -> dict:
    result = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
    available = torch.cuda.is_available()
    return {"python": sys.version, "platform": platform.platform(), "pip_freeze": result.stdout,
            "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
            "cuda_available": available,
            "gpu_name": torch.cuda.get_device_name(0) if available else None,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}


def validate_budget(budget: dict, *, bundle_sha256: str, study: str, stage: str, count: int) -> None:
    for key, value in (("schema_version", "r1-budget-1.0"), ("status", "approved_for_engineering"),
                       ("study_id", study), ("stage", stage), ("bundle_sha256", bundle_sha256)):
        if budget.get(key) != value:
            raise ValueError(f"budget review {key} is absent or does not match")
    if stage not in ENGINEERING_STAGES:
        raise ValueError("scientific execution remains blocked")
    if not budget.get("reviewed_by") or not budget.get("approval_reference") or not budget.get("reviewed_at_utc"):
        raise ValueError("an explicit engineering budget review is required")
    if type(budget.get("max_runs")) is not int or not (count <= budget["max_runs"] and count > 0):
        raise ValueError("requested run count exceeds the reviewed budget")
    for key in ("max_wall_seconds", "max_total_usd", "hourly_usd"):
        if not isinstance(budget.get(key), (int, float)) or not math.isfinite(budget[key]) or budget[key] <= 0:
            raise ValueError(f"budget {key} must be a finite positive number")
    if budget["max_wall_seconds"] * budget["hourly_usd"] / 3600 > budget["max_total_usd"]:
        raise ValueError("wall-time allowance exceeds reviewed compute dollars")


def engineering_limits(matrix: dict, stage: str) -> dict:
    value = (matrix["engineering_limits"] if stage == "engineering"
             else matrix["performance_calibration"]["limits"])
    for key in ("wall_clock_seconds", "output_bytes"):
        if type(value.get(key)) is not int or value[key] <= 0:
            raise ValueError(f"invalid engineering resource cap: {key}")
    return value


def output_size(directory: Path) -> int:
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


@contextmanager
def wall_clock_guard(seconds: float):
    """Linux/POSIX engineering watchdog; provider termination remains external.

    Python signals may be delivered only after a native kernel returns, so this
    guard cannot promise instant cancellation of a stalled CUDA call.
    """
    if not hasattr(signal, "setitimer"):
        raise ValueError("engineering execution requires a POSIX wall-time watchdog")
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    if previous_timer[0] or previous_timer[1]:
        raise ValueError("refusing to replace an existing process watchdog")
    def expired(signum, frame):
        raise TimeoutError("R1 engineering wall-clock cap reached")
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def validate_environment_review(review: dict, environment: dict, bundle_sha256: str) -> None:
    for key, value in (("schema_version", "r1-environment-review-1.0"), ("status", "approved_for_engineering"),
                       ("bundle_sha256", bundle_sha256), ("environment_sha256", sha256_json(environment)),
                       ("pip_freeze_sha256", sha256_bytes(environment["pip_freeze"].encode()))):
        if review.get(key) != value:
            raise ValueError(f"remote environment review {key} is absent or mismatched")
    digest = review.get("container_image_digest")
    if not isinstance(digest, str) or not re.fullmatch(r"sha256:[a-f0-9]{64}", digest):
        raise ValueError("review must identify the immutable container image digest")
    if not all(review.get(key) for key in ("reviewed_by", "reviewed_at_utc", "approval_reference")):
        raise ValueError("explicit remote environment acceptance is required")


def freeze_engineering(root: Path, stage: str, budget_path: Path, admission_path: Path,
                       environment_review_path: Path) -> dict:
    """Run on the selected remote environment after explicit budget approval."""
    bundle = verify_bundle(root)
    if stage not in ENGINEERING_STAGES:
        raise ValueError("fresh scientific admission is not implemented")
    plan = read_json(root / "r1_plan.json")["stages"][stage]
    limits = engineering_limits(load_matrix(root / bundle["matrix"]), stage)
    budget = read_json(budget_path)
    digest = sha256_file(str(root / "r1_bundle.json"))
    validate_budget(budget, bundle_sha256=digest, study=bundle["study_id"], stage=stage, count=plan["run_count"])
    if budget["max_wall_seconds"] > limits["wall_clock_seconds"]:
        raise ValueError("reviewed duration exceeds the stage engineering cap")
    environment = environment_record()
    if not environment["cuda_available"]:
        raise ValueError("acceptance requires the selected CUDA environment; CPU/MPS fallback is prohibited")
    environment_review = read_json(environment_review_path)
    validate_environment_review(environment_review, environment, digest)
    admission = {"schema_version": "r1-engineering-admission-1.0", "created_at_utc": now(),
                 "study_id": bundle["study_id"], "stage": stage, "bundle_sha256": digest,
                 "budget": budget, "budget_sha256": sha256_json(budget), "environment": environment,
                 "environment_review": environment_review, "environment_review_sha256": sha256_json(environment_review),
                 "environment_sha256": sha256_json(environment), "scientific_evidence": False,
                 "scope": "Local pre-execution binding; no independently witnessed preregistration asserted."}
    write_new(admission_path, admission)
    return admission


def claim_admission(root: Path, admission: dict, execution_id: str,
                    scientific_evidence: bool = False) -> dict:
    """Consume one local admission and its underlying budget approval exactly once."""
    safe_name(execution_id)
    claims = root / "outputs" / admission["study_id"] / "admission_claims"
    claims.mkdir(parents=True, exist_ok=True)
    admission_digest = sha256_json(admission)
    budget_approval_digest = sha256_json({
        "study_id": admission["study_id"],
        "approval_reference": admission["budget"]["approval_reference"],
        "reviewed_by": admission["budget"]["reviewed_by"],
    })
    claim = {"schema_version": "r1-admission-claim-1.0", "claimed_at_utc": now(),
             "admission_sha256": admission_digest, "budget_approval_sha256": budget_approval_digest,
             "execution_id": execution_id, "pid": os.getpid(),
             "scientific_evidence": scientific_evidence,
             "note": "Single-use local reservation; failures consume the approval. A new explicit review is required for another attempt."}
    write_new(claims / f"budget-{budget_approval_digest}.json", claim)
    write_new(claims / f"admission-{admission_digest}.json", claim)
    return claim


def _engineering_panel(root: Path, matrix: dict, stage: str) -> tuple[Path, dict]:
    from .r1_panels import REGIONS, build_panel_manifest

    spec = stage_spec(matrix, stage)
    data = spec["base"]["data"]
    corpus_path = inside(root, data["path"])
    counts = {name: 4 for name in REGIONS}
    seed = derive_seed("r1-engineering-panel-v1", matrix["study_id"], stage)
    return corpus_path, build_panel_manifest(
        corpus_path, int(data["seq_len"]), counts, seed, corpus_label=data["path"],
    )


def _bind_distributed_runs(root: Path, runs: list[dict], execution_root: Path,
                           panel: dict, execution_id: str) -> None:
    for resolved in runs:
        resolved["run_id"] += f"--e{execution_id}"
        resolved["output_dir"] = str(execution_root / "runs" / resolved["run_id"])
        data_path = inside(root, resolved["data"]["path"])
        resolved["data"]["path"] = str(data_path.relative_to(root))
        resolved["r1"].update({"transport": "loopback-tcp-r1-wire-1.0",
                               "semantic_validation": True,
                               "attack_validation": False,
                               "panel_sha256": panel["panel_sha256"]})


def execute_engineering(root: Path, admission_path: Path, execution_id: str) -> dict:
    """Execute only approved engineering stages, with fresh exclusive destinations."""
    from . import manifest
    from .r1_local_validation import run_distributed_engineering_cell
    root = root.resolve()
    if Path(__file__).resolve().parents[2] != root:
        raise ValueError("execute the copied bundle's scripts using its own source tree")
    bundle = verify_bundle(root)
    admission = read_json(admission_path)
    stage = admission.get("stage")
    if admission.get("schema_version") != "r1-engineering-admission-1.0" or stage not in ENGINEERING_STAGES:
        raise ValueError("no scientific execution admitted")
    if (admission.get("study_id") != bundle["study_id"] or admission.get("scientific_evidence") is not False or
            admission["bundle_sha256"] != sha256_file(str(root / "r1_bundle.json"))):
        raise ValueError("admission identity mismatch")
    if (sha256_json(admission["budget"]) != admission["budget_sha256"] or
            sha256_json(admission["environment"]) != admission["environment_sha256"] or
            sha256_json(admission["environment_review"]) != admission["environment_review_sha256"]):
        raise ValueError("admission contents differ from their digests")
    validate_environment_review(admission["environment_review"], admission["environment"], admission["bundle_sha256"])
    if sha256_json(environment_record()) != admission["environment_sha256"]:
        raise ValueError("remote environment changed after acceptance")
    matrix = load_matrix(root / bundle["matrix"])
    safe_name(execution_id)
    execution_root = root / "outputs" / bundle["study_id"] / f"{stage}--{execution_id}"
    runs = enumerate_runs(matrix, stage, execution_root / "runs")
    _, panel = _engineering_panel(root, matrix, stage)
    _bind_distributed_runs(root, runs, execution_root, panel, execution_id)
    validate_budget(admission["budget"], bundle_sha256=admission["bundle_sha256"],
                    study=bundle["study_id"], stage=stage, count=len(runs))
    limits = engineering_limits(matrix, stage)
    if admission["budget"]["max_wall_seconds"] > limits["wall_clock_seconds"]:
        raise ValueError("reviewed duration exceeds engineering cap")
    # A copied admission or a changed execution ID cannot multiply approval.
    claim = claim_admission(root, admission, execution_id)
    execution_root.mkdir(parents=True, exist_ok=False)
    from .r1_panels import save_panel_manifest
    panel_path = execution_root / "panel_manifest.json"
    save_panel_manifest(panel_path, panel)
    write_new(execution_root / "admission.json", admission)
    write_new(execution_root / "admission_claim.json", claim)
    launch = {"schema_version": "r1-launch-1.0", "study_id": bundle["study_id"], "stage": stage,
              "execution_id": execution_id, "launch_uuid": str(uuid.uuid4()), "started_at_utc": now(),
              "pid": os.getpid(), "hostname": platform.node(), "argv": list(sys.argv), "cwd": str(Path.cwd()),
              "bundle_root_at_execution": str(root),
              "bundle_sha256": admission["bundle_sha256"],
              "admission_sha256": sha256_file(str(execution_root / "admission.json")),
              "admission_claim_sha256": sha256_file(str(execution_root / "admission_claim.json")),
              "panel_manifest": "panel_manifest.json",
              "panel_sha256": panel["panel_sha256"],
              "runs": runs, "scientific_evidence": False}
    write_new(execution_root / "launch.json", launch)
    previous_manifest_dir = manifest.MANIFEST_DIR
    manifest.MANIFEST_DIR = execution_root / "manifests"
    start = time.monotonic()
    completed = []
    failure = None
    try:
        with wall_clock_guard(admission["budget"]["max_wall_seconds"]):
            for resolved in runs:
                if time.monotonic() - start >= admission["budget"]["max_wall_seconds"]:
                    raise RuntimeError("engineering wall-time allowance reached before next run")
                verify_bundle(root)
                if output_size(execution_root) >= limits["output_bytes"]:
                    raise RuntimeError("engineering output cap reached before next run")
                record_root = execution_root / "execution_records"
                run_id = resolved["run_id"]
                begin = {"schema_version": "r1-process-start-1.0", "launch_uuid": launch["launch_uuid"],
                         "launch_sha256": sha256_file(str(execution_root / "launch.json")),
                         "pid": os.getpid(), "run_id": run_id, "started_at_utc": now(),
                         "config_sha256": sha256_json(resolved), "scientific_evidence": False}
                write_new(record_root / f"{run_id}.start.json", begin)
                try:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                    run_start_monotonic = time.monotonic()
                    manifest_path = run_distributed_engineering_cell(root, resolved, panel, panel_path)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    manifest.completed_manifest_exists(run_id, resolved=resolved)
                    metrics = read_json(Path(resolved["output_dir"]) / "metrics_summary.json")
                    if output_size(execution_root) > limits["output_bytes"]:
                        raise RuntimeError("engineering output cap exceeded; preserve artifacts and stop")
                    end = {"schema_version": "r1-process-end-1.0", "run_id": run_id,
                           "launch_uuid": launch["launch_uuid"], "pid": os.getpid(),
                           "finished_at_utc": now(), "status": "completed",
                           "start_sha256": sha256_file(str(record_root / f"{run_id}.start.json")),
                           "manifest": str(manifest_path.relative_to(root)),
                           "manifest_sha256": sha256_file(str(manifest_path)), "scientific_evidence": False,
                           "resource_measurements": {
                               "complete_run_wall_seconds": time.monotonic() - run_start_monotonic,
                               "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
                               "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None,
                               "remote_cuda_peak_allocated_bytes": metrics.get("remote_cuda_peak_allocated_bytes"),
                               "remote_cuda_peak_reserved_bytes": metrics.get("remote_cuda_peak_reserved_bytes"),
                               "actual_wire_bytes": metrics.get("actual_wire_bytes", 0),
                               "output_bytes_including_source": output_size(Path(resolved["output_dir"])),
                               "scope": "Two-process loopback split execution; not a two-host network latency measurement."}}
                    write_new(record_root / f"{run_id}.end.json", end)
                    completed.append(run_id)
                except BaseException as exc:
                    write_new(record_root / f"{run_id}.failure.json", {"run_id": run_id,
                              "launch_uuid": launch["launch_uuid"], "failed_at_utc": now(),
                              "error_type": type(exc).__name__, "error": str(exc),
                              "start_sha256": sha256_file(str(record_root / f"{run_id}.start.json")),
                              "scientific_evidence": False})
                    raise
    except BaseException as exc:
        failure = {"type": type(exc).__name__, "detail": str(exc)}
        raise
    finally:
        manifest.MANIFEST_DIR = previous_manifest_dir
        write_new(execution_root / "execution_summary.json", {"finished_at_utc": now(),
                  "launch_sha256": sha256_file(str(execution_root / "launch.json")),
                  "completed": completed, "failure": failure, "wall_seconds": time.monotonic() - start,
                  "scientific_evidence": False,
                  "budget_note": "Python wall-time watchdog is supplemented by before-run checks; external provider watchdog must enforce billing cap.",
                  "storage_note": "Output bytes are checked before and after each run. This is a stop threshold and may overshoot during a run; it is not a filesystem quota."})
    return {"execution_root": str(execution_root), "completed": len(completed), "scientific_evidence": False}


def analyze_execution(root: Path, execution_root: Path, destination: Path) -> dict:
    """Verify fresh source/input/process lineage before descriptive aggregation.

    No external legacy scores or qualification records can enter this analyzer.
    Missing privacy measurements are explicitly UNDETERMINED.
    """
    from . import manifest
    root, execution_root = root.resolve(), execution_root.resolve()
    if not execution_root.is_relative_to(root / "outputs"):
        raise ValueError("execution must belong to this bundle's fresh output namespace")
    bundle = verify_bundle(root)
    launch_path = execution_root / "launch.json"
    launch = read_json(launch_path)
    if (launch.get("schema_version") != "r1-launch-1.0" or launch.get("study_id") != bundle["study_id"] or
            launch.get("scientific_evidence") is not False or launch.get("stage") not in ENGINEERING_STAGES or
            launch.get("bundle_sha256") != sha256_file(str(root / "r1_bundle.json"))):
        raise ValueError("legacy, wrong-study or unsupported launch evidence")
    admission_path = execution_root / "admission.json"
    if sha256_file(str(admission_path)) != launch["admission_sha256"]:
        raise ValueError("admission binding mismatch")
    admission = read_json(admission_path)
    if (admission["bundle_sha256"] != launch["bundle_sha256"] or admission["stage"] != launch["stage"] or
            admission["study_id"] != launch["study_id"] or admission.get("scientific_evidence") is not False or
            admission.get("schema_version") != "r1-engineering-admission-1.0" or
            sha256_json(admission["budget"]) != admission["budget_sha256"] or
            sha256_json(admission["environment"]) != admission["environment_sha256"] or
            sha256_json(admission["environment_review"]) != admission["environment_review_sha256"]):
        raise ValueError("admission/launch identity mismatch")
    validate_environment_review(admission["environment_review"], admission["environment"], admission["bundle_sha256"])
    claim_path = execution_root / "admission_claim.json"
    claim = read_json(claim_path)
    if (sha256_file(str(claim_path)) != launch["admission_claim_sha256"] or
            claim["admission_sha256"] != sha256_json(admission) or
            claim["execution_id"] != launch["execution_id"] or claim["pid"] != launch["pid"]):
        raise ValueError("single-use admission claim binding mismatch")
    matrix = load_matrix(root / bundle["matrix"])
    expected_runs = enumerate_runs(matrix, launch["stage"], execution_root / "runs")
    _, expected_panel = _engineering_panel(root, matrix, launch["stage"])
    _bind_distributed_runs(root, expected_runs, execution_root, expected_panel,
                           safe_name(launch["execution_id"]))
    recorded_root = Path(launch["bundle_root_at_execution"])
    for resolved in expected_runs:
        relative = Path(resolved["output_dir"]).relative_to(root)
        resolved["output_dir"] = str(recorded_root / relative)
    panel_path = execution_root / launch["panel_manifest"]
    if (read_json(panel_path) != expected_panel or
            launch.get("panel_sha256") != expected_panel["panel_sha256"]):
        raise ValueError("launch panel differs from the prepared engineering panel")
    if expected_runs != launch["runs"]:
        raise ValueError("launch configurations differ from the prepared matrix")
    validate_budget(admission["budget"], bundle_sha256=launch["bundle_sha256"], study=launch["study_id"],
                    stage=launch["stage"], count=len(expected_runs))
    inputs = [{"path": str(launch_path.relative_to(root)), "sha256": sha256_file(str(launch_path))},
              {"path": str(admission_path.relative_to(root)), "sha256": sha256_file(str(admission_path))},
              {"path": str(claim_path.relative_to(root)), "sha256": sha256_file(str(claim_path))}]
    rows, failures, interrupted, not_started = [], [], [], []
    previous_dir, previous_root = manifest.MANIFEST_DIR, manifest.REPO_ROOT
    manifest.MANIFEST_DIR, manifest.REPO_ROOT = execution_root / "manifests", root
    try:
        seen = set()
        for resolved in launch["runs"]:
            run_id = safe_name(resolved["run_id"])
            if run_id in seen or resolved.get("r1", {}).get("study_id") != bundle["study_id"]:
                raise ValueError("duplicate or wrong-study resolved run")
            seen.add(run_id)
            original_bundle_root = Path(launch["bundle_root_at_execution"])
            output_dir = execution_root / "runs" / run_id
            original_execution_relative = execution_root.relative_to(root)
            if Path(resolved["output_dir"]) != original_bundle_root / original_execution_relative / "runs" / run_id:
                raise ValueError("run output root does not match launch location")
            original_data = resolved["data"]
            data_path = inside(root, original_data["path"])
            if sha256_file(str(data_path)) != original_data["corpus_sha256"]:
                raise ValueError("raw input unavailable or hash mismatch")
            start_path = execution_root / "execution_records" / f"{run_id}.start.json"
            end_path = execution_root / "execution_records" / f"{run_id}.end.json"
            failure_path = execution_root / "execution_records" / f"{run_id}.failure.json"
            if failure_path.is_file():
                failures.append(read_json(failure_path))
                continue
            if not start_path.is_file():
                not_started.append(run_id)
                continue
            if not end_path.is_file():
                interrupted.append(run_id)
                continue
            begin, end = read_json(start_path), read_json(end_path)
            if (begin["launch_uuid"] != launch["launch_uuid"] or begin["launch_sha256"] != sha256_file(str(launch_path)) or
                    begin["run_id"] != run_id or begin["config_sha256"] != sha256_json(resolved) or
                    end["run_id"] != run_id or end["launch_uuid"] != begin["launch_uuid"] or
                    end["pid"] != begin["pid"] or begin["pid"] != launch["pid"] or
                    end["start_sha256"] != sha256_file(str(start_path)) or end["status"] != "completed"):
                raise ValueError("execution start/end binding mismatch")
            manifest_path = inside(root, end["manifest"])
            if (manifest_path != manifest.MANIFEST_DIR / f"{run_id}.json" or
                    sha256_file(str(manifest_path)) != end["manifest_sha256"]):
                raise ValueError("execution manifest binding mismatch")
            recorded = read_json(manifest_path)
            if recorded.get("schema_version") != "0.2.0" or "execution_provenance" not in recorded:
                raise ValueError("legacy evidence without retained execution source is inadmissible")
            # Verification does not rewrite execution-time absolute paths. Artifact
            # paths are bundle-relative, allowing a reviewer to move the bundle.
            manifest.validate_manifest(recorded)
            for key in ("run_id", "protocol_version", "seed", "cohort", "pairing_block_id"):
                if recorded.get(key) != resolved.get(key):
                    raise ValueError(f"manifest identity differs: {key}")
            execution_provenance = recorded["execution_provenance"]
            if (recorded["status"] != "completed" or recorded["resolved_config_sha256"] != sha256_json(resolved) or
                    execution_provenance["run_id"] != run_id or execution_provenance["output_dir"] != resolved["output_dir"] or
                    execution_provenance["resolved_config_sha256"] != recorded["resolved_config_sha256"]):
                raise ValueError("manifest/config/source identity mismatch")
            for key in ("git_commit", "environment_digest"):
                if recorded[key] != execution_provenance[key]:
                    raise ValueError(f"manifest differs from retained provenance: {key}")
            bound_paths = set()
            for item in recorded["artifacts"]:
                artifact = inside(root, item["path"])
                if (not artifact.is_relative_to(output_dir) or artifact in bound_paths or
                        not artifact.is_file() or sha256_file(str(artifact)) != item["sha256"]):
                    raise ValueError("required artifact unavailable, duplicate, escaping or corrupted")
                bound_paths.add(artifact)
            required = manifest._required_artifacts(resolved, prospective=True)
            if not all(output_dir / name in bound_paths for name in required):
                raise ValueError("required run artifacts are not all bound")
            if (read_json(output_dir / "config.resolved.json") != resolved or
                    read_json(output_dir / "provenance/run_start.json") != execution_provenance):
                raise ValueError("retained source/config differs from manifest")
            provenance.verify_snapshot(root, output_dir, execution_provenance)
            manifest._verify_semantics(output_dir, resolved)
            initialization_path = output_dir / "provenance/r1_initialization.json"
            initialization = read_json(initialization_path)
            initial_state = initialization_path.parent / "r1_initial_state.pt"
            bound = {item["path"]: item["sha256"] for item in recorded["artifacts"]}
            if (initialization.get("run_id") != run_id or initialization.get("initialization") != "r1-named-streams-v1" or
                    sha256_file(str(initial_state)) != initialization["retained_state_file_sha256"] or
                    str(initialization_path.relative_to(root)) not in bound or str(initial_state.relative_to(root)) not in bound):
                raise ValueError("initialization is not bound to the actual run manifest")
            metrics_path = output_dir / "metrics_summary.json"
            metrics = read_json(metrics_path)
            if metrics.get("run_id") != run_id or metrics.get("mode") != resolved["mode"] or metrics.get("seed") != resolved["seed"]:
                raise ValueError("metrics identity mismatch")
            loss = metrics["final_eval_loss"]
            if not isinstance(loss, (int, float)) or not math.isfinite(loss):
                raise ValueError("nonfinite held-out loss")
            rows.append({"run_id": run_id, "cell_id": resolved["r1"]["cell_id"], "seed": resolved["seed"],
                         "mode": resolved["mode"], "loss": loss, "wall_seconds": metrics["wall_time_seconds"],
                         "privacy": "UNDETERMINED", "scientific_evidence": False,
                         "initial_model_sha256": initialization["model_state_sha256"],
                         "initial_trusted_sha256": initialization["trusted_state_sha256"],
                         "initial_components": initialization.get("components", {}),
                         "resource_measurements": end.get("resource_measurements", {})})
            for path in (start_path, end_path, manifest_path, metrics_path):
                inputs.append({"path": str(path.relative_to(root)), "sha256": sha256_file(str(path))})
    finally:
        manifest.MANIFEST_DIR, manifest.REPO_ROOT = previous_dir, previous_root
    contrasts = []
    by_key = {(r["cell_id"], r["seed"], r["mode"]): r for r in rows}
    for row in rows:
        if row["mode"] != "trained":
            continue
        baseline = by_key.get(("all-local", row["seed"], "all_local"))
        frozen = by_key.get((row["cell_id"], row["seed"], "frozen_random"))
        removed = by_key.get((row["cell_id"], row["seed"], "removed_remote"))
        if frozen is not None and frozen["initial_model_sha256"] != row["initial_model_sha256"]:
            raise ValueError("trained/frozen initialization pairing did not match")
        if removed is not None and removed["initial_trusted_sha256"] != row["initial_trusted_sha256"]:
            raise ValueError("trained/removed trusted initialization pairing did not match")
        if baseline is not None:
            for key, digest in baseline["initial_components"].items():
                if key in row["initial_components"] and row["initial_components"][key] != digest:
                    raise ValueError("split/all-local common initialization pairing did not match")
        contrasts.append({"cell_id": row["cell_id"], "seed": row["seed"],
                          "replacement_penalty": None if baseline is None else row["loss"] - baseline["loss"],
                          "gain_against_frozen": None if frozen is None else frozen["loss"] - row["loss"],
                          "gain_against_removed": None if removed is None else removed["loss"] - row["loss"],
                          "privacy": "UNDETERMINED", "joint_feasibility": "UNDETERMINED"})
    report = {"schema_version": "r1-analysis-1.0", "created_at_utc": now(), "study_id": bundle["study_id"],
              "stage": launch["stage"], "scientific_evidence": False, "input_bindings": inputs,
              "analyzer_source_inventory_sha256": sha256_json(provenance.source_inventory(root)),
              "completed_count": len(rows), "planned_count": len(launch["runs"]),
              "failures": failures, "interrupted_runs": interrupted,
              "not_started_runs": not_started,
              "rows": rows, "descriptive_contrasts": contrasts, "privacy": "UNDETERMINED",
              "joint_feasibility": "UNDETERMINED", "blockers": BLOCKERS,
              "scope": "Engineering behavior only. No attacks are implemented/admitted by this analyzer."}
    write_new(destination, report)
    write_new(destination.with_suffix(destination.suffix + ".sha256.json"),
              {"path": destination.name, "sha256": sha256_file(str(destination)),
               "input_bindings_sha256": sha256_json(inputs), "scientific_evidence": False})
    return report
