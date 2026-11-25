"""CPU software/asset admission for a separate, immutable original-AT baseline.

No model is constructed or executed. Asset stages require an earlier matching
dependency receipt. All runtime work belongs to an ORX run; --validate-only is a
stdlib-only local source/configuration check.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import selectors
import shutil
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
_HEADER_SPEC = importlib.util.spec_from_file_location("original_at_private_headers", HERE / "private_headers.py")
private_headers = importlib.util.module_from_spec(_HEADER_SPEC)
_HEADER_SPEC.loader.exec_module(private_headers)
REPO = HERE.parents[2]
STAGES = {"dependencies", "dataset-assets", "model-assets", "assets-verify"}
FORBIDDEN_INSTALLS = {"torch", "torchvision", "torchaudio", "triton"}
ACTIVE_CHILD = None
REPAIR_OWNER = "6e2732f206cfe1a158439fd226cb553810b027b96b0d975f17ec0ae6c2f583d5"
REPAIR_RUN = "2a8327a1-dd19-4999-9ca7-28f73f029a85"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def emit(event, **values):
    print(json.dumps({"event": event, "unix_time": time.time(), **values}), flush=True)


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate(cfg):
    if cfg.get("schema") != "vflair-original-at-setup-v1" or cfg.get("stage") not in STAGES:
        raise ValueError("Unrecognised original AT setup schema/stage")
    host, pins = cfg["host"], cfg["pins"]
    private = Path(host["private_environment"]).expanduser().resolve()
    forbidden = [Path(path).expanduser().resolve() for path in host["forbidden_environment_paths"]]
    if private in forbidden or private == Path(host["base_python"]).expanduser().resolve().parent.parent:
        raise ValueError("AT bootstrap cannot modify the MID or base environment")
    if private.name != "vflair-native-at-20261022":
        raise ValueError("Use the explicitly separate AT private-environment path")
    repair = cfg.get("provisional_software_repair")
    if repair != {"prior_contract_sha256": REPAIR_OWNER, "failed_orx_run_id": REPAIR_RUN,
                  "reason": "missing-system-Python.h-before-any-model-or-asset-stage"}:
        raise ValueError("Provisional recovery must name the exact failed software-only contract/run")
    if cfg.get("native_metric_policy") != "preserve-upstream-mapping-callback-v1":
        raise ValueError("Setup does not silently replace the native metric convention")
    if cfg.get("optional_corrected_metric") != "codebleu-default-whitespace.v1-from-saved-generations":
        raise ValueError("Optional rescoring must keep its distinct declared metric version")
    if cfg.get("corrected_metric_execution_in_setup") or cfg.get("model_import_training_generation_attack_in_setup"):
        raise ValueError("Setup cannot execute a model or a corrected utility analysis")
    expected = {"python_major_minor": "3.12", "setuptools": "80.9.0", "wheel": "0.45.1",
                "upstream_revision": "ff18565d89dce6367b6f13881a415c053b60cb0f",
                "native_at_config_sha256": "00932f240945e76f78ac11f9412daba16ed3e0c57972a984ba9f035905180b60",
                "model_repository": "codellama/CodeLlama-7b-hf", "model_revision": "6c284d1468fe6c413cf56183e69b194dcfa27fe6",
                "dataset_repository": "sahil2801/CodeAlpaca-20k", "dataset_revision": "152bb5e9a29651266b018106053980070a0521a1",
                "codebleu": "0.7.0", "tree-sitter": "0.22.3", "tree-sitter-python": "0.21.0",
                "ubuntu_python_dev_sha256": "945ad3f651f683c11e72d19d804b1cff097308dcffa01057635cba038f875164",
                "ubuntu_python_dev_version": "3.12.3-1ubuntu0.17",
                "cffi": "1.17.1", "pycparser": "2.22", "nvidia-ml-py": "12.535.161",
                "grammar_source_commit": "0f9047c857ed0990931b1f899c7d3bf403703147",
                "grammar_source_sha256": "62cf59b7013e8d1c7e60aa181fe3216f54df18145d218699061f84376b17170c"}
    if any(pins.get(key) != value for key, value in expected.items()):
        raise ValueError("Original AT setup pin differs from its reviewed contract")
    header_spec = json.loads((HERE / "inputs/ubuntu-python-dev-package.json").read_text())
    if header_spec["sha256"] != pins["ubuntu_python_dev_sha256"] or header_spec["version"] != pins["ubuntu_python_dev_version"]:
        raise ValueError("Python header metadata differs from its exact reviewed pin")
    manifest = json.loads((HERE / "vendor/upstream-manifest.json").read_text())
    if manifest["revision"] != pins["upstream_revision"]:
        raise ValueError("Vendored source manifest revision mismatch")
    for item in manifest["files"]:
        path = HERE / "vendor/upstream" / item["path"]
        if sha(path) != item["sha256"] or path.stat().st_size != item["bytes"]:
            raise ValueError(f"Vendored pinned source mismatch: {item['path']}")
    proof = json.loads((HERE / "vendor/config-provenance.json").read_text())
    for item in proof["files"]:
        raw = (HERE / "vendor/upstream" / item["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != item["sha256"] or hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() != item["git_blob_sha1"]:
            raise ValueError("Pinned config Git blob mismatch")
    native_config = HERE / "vendor/upstream/src/configs/llm_configs/with_mia_and_defense/2-slice/codealpaca_at.json"
    if sha(native_config) != pins["native_at_config_sha256"]:
        raise ValueError("Native original AT config differs")
    for name, repository, revision in (("codellama-metadata.json", pins["model_repository"], pins["model_revision"]),
                                        ("codealpaca-origin-metadata.json", pins["dataset_repository"], pins["dataset_revision"])):
        metadata = json.loads((HERE / "inputs" / name).read_text())
        if metadata.get("id") != repository or metadata.get("sha") != revision:
            raise ValueError(f"Pinned metadata mismatch: {name}")
    spec = importlib.util.spec_from_file_location("at_native_input_preparation", HERE / "vendor/native_input_preparation.py")
    preparer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preparer)
    plan = preparer.codellama_download_plan(HERE / "inputs/codellama-tree.json")
    model_bytes = sum(item["size"] for item in plan)
    if len(plan) != 12 or model_bytes != cfg["guards"]["maximum_model_asset_bytes"]:
        raise ValueError("Model asset allowlist/byte budget changed")
    if any(item["path"].endswith(".bin") for item in plan):
        raise ValueError("Pickle model assets are not admitted")
    files = {}
    for path in sorted(HERE.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.name not in {"run.json", "SOURCE-FILES.json"}:
            files[str(path.relative_to(HERE))] = sha(path)
    contract = {"host": host, "pins": pins, "native_metric_policy": cfg["native_metric_policy"],
                "optional_corrected_metric": cfg["optional_corrected_metric"], "files": files,
                "provisional_software_repair": repair}
    return {"contract_sha256": canonical_sha(contract), "source_files_verified": len(manifest["files"]),
            "config_git_blobs_verified": len(proof["files"]), "model_files": len(plan),
            "model_bytes": model_bytes, "model_bin_files": 0,
            "private_environment": host["private_environment"], "local_path_resolution": str(private)}


def reject_torch_plan(report):
    planned = json.loads(Path(report).read_text()).get("install", [])
    names = {item["metadata"]["name"].lower().replace("_", "-") for item in planned}
    blocked = sorted(names.intersection(FORBIDDEN_INSTALLS))
    if blocked:
        raise RuntimeError(f"Pip plan would replace/install inherited CUDA packages: {blocked}")


def run_logged(command, path, env, seconds):
    global ACTIVE_CHILD
    emit("at_setup_command", command=command, timeout_seconds=seconds)
    started = time.monotonic()
    with Path(path).open("w") as log:
        child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 env=env, start_new_session=True)
        ACTIVE_CHILD = child
        selector = selectors.DefaultSelector()
        selector.register(child.stdout, selectors.EVENT_READ)
        try:
            while selector.get_map():
                if time.monotonic() - started > seconds:
                    os.killpg(child.pid, signal.SIGKILL)
                    raise TimeoutError(f"AT setup command exceeded {seconds}s")
                for key, _ in selector.select(timeout=1):
                    data = os.read(key.fd, 65536)
                    if not data:
                        selector.unregister(key.fileobj)
                        continue
                    text = data.decode("utf-8", errors="replace")
                    print(text, end="", flush=True)
                    log.write(text)
                    log.flush()
            result = child.wait()
            if result:
                raise subprocess.CalledProcessError(result, command)
        finally:
            selector.close()
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            ACTIVE_CHILD = None
    return {"command": command, "wall_seconds": time.monotonic() - started, "exit_code": result}


def software_env(cfg):
    asset_root = Path(cfg["host"]["asset_root"]).expanduser()
    private = Path(cfg["host"]["private_environment"]).expanduser()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", HF_HOME=str(asset_root / "software-cache/hf"),
               HF_HUB_CACHE=str(asset_root / "software-cache/hf/hub"),
               PIP_CACHE_DIR=str(private / "pip-cache"), PIP_INDEX_URL="https://pypi.org/simple",
               PIP_EXTRA_INDEX_URL="", TOKENIZERS_PARALLELISM="false", MPLBACKEND="Agg",
               PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python", PYTHONHASHSEED=str(cfg["seed"]))
    env.pop("PIP_TARGET", None)
    env.pop("PIP_PREFIX", None)
    env.pop("PYTHONPATH", None)
    return env


def json_probe(python, code, env):
    result = subprocess.run([str(python), "-c", code], env=env, check=True, capture_output=True, text=True, timeout=60)
    return json.loads(result.stdout)


TORCH_PROBE = ("import hashlib,json,sys,sysconfig,platform,torch; from pathlib import Path; "
               "p=Path(torch.__file__); print(json.dumps({'python':'.'.join(map(str,sys.version_info[:2])),"
               "'python_full':sys.version,'platform':sys.platform,'machine':platform.machine(),"
               "'site':sysconfig.get_path('purelib'),'torch_file':str(p.resolve()),'torch_version':torch.__version__,"
               "'torch_init_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))")
VERSIONS_PROBE = ("import importlib.metadata as m,json; "
                  "print(json.dumps(sorted([{'name':d.metadata['Name'].lower().replace('_','-'),'version':d.version} "
                  "for d in m.distributions()],key=lambda x:(x['name'],x['version']))))")


def torch_identity(value):
    return {key: value[key] for key in ("torch_file", "torch_version", "torch_init_sha256")}


def recoverable_owner(cfg, owner, contract):
    """An unadmitted first-attempt directory may undergo this one named repair."""
    old = owner.get("contract_sha256")
    if old == contract["contract_sha256"]:
        return None
    if old != cfg["provisional_software_repair"]["prior_contract_sha256"]:
        raise RuntimeError("Existing provisional AT environment belongs to another source/config contract")
    return {"schema": "original-at-provisional-software-repair-v1", **cfg["provisional_software_repair"],
            "replacement_contract_sha256": contract["contract_sha256"], "prior_owner": owner,
            "successful_prior_admission_present": False, "environment_deleted": False,
            "MID_or_base_environment_modified": False}


def pip_check(python, path, env):
    result = subprocess.run([str(python), "-m", "pip", "check"], env=env, capture_output=True, text=True, timeout=120)
    record = {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    write(path, record)
    return record


def pip_install(python, requirement, label, out, env, constraints, seconds, no_build=False):
    prefix = [str(python), "-m", "pip", "install", "--disable-pip-version-check", "--no-cache-dir",
              "--index-url", "https://pypi.org/simple", "-c", str(constraints), "-r", str(requirement)]
    if no_build:
        prefix.append("--no-build-isolation")
    plan = out / f"pip-{label}-plan.json"
    run_logged(prefix + ["--dry-run", "--report", str(plan)], out / f"pip-{label}-plan.log", env, seconds)
    reject_torch_plan(plan)
    run_logged(prefix + ["--report", str(out / f"pip-{label}-installed.json")], out / f"pip-{label}-install.log", env, seconds)


def check_receipt(cfg, contract, env):
    private = Path(cfg["host"]["private_environment"]).expanduser()
    path = private / ".llm-attack-suite-original-at-admission.json"
    receipt = json.loads(path.read_text())
    if receipt.get("status") != "DEPENDENCIES_ADMITTED_NATIVE_METRIC_QUALIFIED" or receipt.get("contract_sha256") != contract["contract_sha256"]:
        raise RuntimeError("Asset preparation requires the matching successful AT dependency gate")
    python = private / "bin/python"
    current = json_probe(python, TORCH_PROBE, env)
    if torch_identity(current) != receipt["inherited_torch"]:
        raise RuntimeError("Inherited Torch identity changed after the AT dependency gate")
    if {key: current[key] for key in ("python_full", "platform", "machine")} != receipt["compiled_environment_platform"]:
        raise RuntimeError("AT compiled-package platform differs from dependency admission")
    versions = json_probe(python, VERSIONS_PROBE, env)
    if canonical_sha(versions) != receipt["installed_versions_sha256"]:
        raise RuntimeError("AT environment versions changed after admission; repeat software admission in a new frozen node")
    return receipt


def dependencies(cfg, contract, out, env):
    base = Path(cfg["host"]["base_python"]).expanduser()
    private = Path(cfg["host"]["private_environment"]).expanduser()
    python = private / "bin/python"
    marker = private / ".llm-attack-suite-original-at-owner.json"
    admitted = private / ".llm-attack-suite-original-at-admission.json"
    if admitted.is_file():
        receipt = check_receipt(cfg, contract, env)
        write(out / "dependency-admission.json", receipt)
        emit("at_dependencies_reused", contract_sha256=contract["contract_sha256"])
        return receipt
    if private.exists() and not marker.is_file():
        raise RuntimeError("Refusing to mutate an existing unowned AT environment")
    recovery = recoverable_owner(cfg, json.loads(marker.read_text()), contract) if marker.is_file() else None
    private.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(private.parent).free < cfg["guards"]["minimum_free_gib_dependencies"] * 2**30:
        raise RuntimeError("Insufficient disk for the cheap AT software admission")
    before = json_probe(base, TORCH_PROBE, env)
    if before["python"] != cfg["pins"]["python_major_minor"]:
        raise RuntimeError("Original AT native package stack requires the declared Python 3.12 interpreter")
    if before["platform"] != "linux" or before["machine"] != "aarch64":
        raise RuntimeError("Pinned Ubuntu Python headers/wheels require Linux aarch64")
    base_versions_before = json_probe(base, VERSIONS_PROBE, env)
    base_check_before = pip_check(base, out / "base-pip-check-before.json", env)
    if not python.exists():
        run_logged([str(base), "-m", "venv", "--system-site-packages", str(private)], out / "venv.log", env, 120)
    if recovery:
        write(out / "provisional-environment-repair.json", recovery)
    write(marker, {"schema": "original-at-private-environment-owner-v1", "contract_sha256": contract["contract_sha256"],
                   "provisional_repair": recovery})
    site = json_probe(python, "import json,sysconfig; print(json.dumps(sysconfig.get_path('purelib')))", env)
    # A venv created from a venv does not automatically inherit the outer venv.
    Path(site, "original_at_inherited_cuda.pth").write_text(before["site"] + "\n")
    inherited = json_probe(python, TORCH_PROBE, env)
    if torch_identity(inherited) != torch_identity(before):
        raise RuntimeError("Separate AT environment does not inherit the exact base CUDA Torch")
    pip_check(python, out / "private-pip-check-before.json", env)
    header_spec = json.loads((HERE / "inputs/ubuntu-python-dev-package.json").read_text())
    env, header_receipt = private_headers.prepare(header_spec, private, out, env, run_logged,
        lambda command: subprocess.run(command, env=env, check=True, capture_output=True, text=True, timeout=30).stdout)
    constraints = out / "inherited-torch-constraint.txt"
    constraints.write_text("torch===" + before["torch_version"] + "\n")
    seconds = cfg["timeouts"]["dependency_seconds"]
    pip_install(python, HERE / "requirements-build.txt", "build", out, env, constraints, seconds)
    pip_install(python, HERE / "requirements-metric.txt", "metric", out, env, constraints, seconds, no_build=True)
    pip_install(python, HERE / "requirements-native.txt", "native", out, env, constraints, seconds)
    pip_install(python, HERE / "requirements-private-compat.txt", "private-compat", out, env, constraints, seconds)
    run_logged([str(python), "-m", "pip", "--no-cache-dir", "check"], out / "pip-check.log", env, 120)
    private_check_after = pip_check(python, out / "private-pip-check-after.json", env)
    if private_check_after["returncode"] != 0:
        raise RuntimeError("Private pip check did not pass; inherited conflicts are not suppressed")
    run_logged([str(python), str(HERE / "metric_gate.py"), "--output", str(out / "metric-gate.json"),
                "--native-source", str(HERE / "vendor/upstream/src/evaluates/MainTaskVFL_LLM.py")],
               out / "metric-gate.log", env, 120)
    gate = json.loads((out / "metric-gate.json").read_text())
    if gate.get("status") != "PASS_NATIVE_METRIC_QUALIFIED" or gate.get("checks_passed") != 19:
        raise RuntimeError("AT CPU software metric gate did not pass all 19 checks")
    after_base, after_private = json_probe(base, TORCH_PROBE, env), json_probe(python, TORCH_PROBE, env)
    if torch_identity(before) != torch_identity(after_base) or torch_identity(before) != torch_identity(after_private):
        raise RuntimeError("Private bootstrap changed/replaced inherited CUDA Torch")
    base_check_after = pip_check(base, out / "base-pip-check-after.json", env)
    if base_check_before != base_check_after or base_versions_before != json_probe(base, VERSIONS_PROBE, env):
        raise RuntimeError("Read-only base environment changed during private AT setup")
    versions = json_probe(python, VERSIONS_PROBE, env)
    write(out / "installed-versions.json", versions)
    receipt = {"schema": "original-at-dependency-admission-v1", "status": "DEPENDENCIES_ADMITTED_NATIVE_METRIC_QUALIFIED",
               "contract_sha256": contract["contract_sha256"], "inherited_torch": torch_identity(before),
               "compiled_environment_platform": {key: after_private[key] for key in ("python_full", "platform", "machine")},
               "originating_orx_run_id": os.environ["ORX_RUN_ID"], "originating_output_directory": str(out),
               "private_environment": str(private), "installed_versions_sha256": canonical_sha(versions),
               "metric_gate_sha256": sha(out / "metric-gate.json"), "metric_gate_checks": 19,
               "installation_report_hashes": {path.name: sha(path) for path in sorted(out.glob("pip-*-installed.json"))},
               "python_header_package_sha256": header_receipt["sha256"], "python_header_admission_sha256": sha(out / "python-header-admission.json"),
               "private_pip_check_passed": True, "base_pip_check_unchanged": True,
               "base_pip_check_preexisting_returncode": base_check_before["returncode"],
               "provisional_software_repair_recorded": bool(recovery),
               "native_metric_policy": cfg["native_metric_policy"], "native_metric_api_issue_confirmed": True,
               "native_metric_corrected_or_replaced": False, "model_assets_downloaded_by_dependency_stage": False,
               "model_constructed_loaded_or_executed": False}
    write(admitted, receipt)
    write(out / "dependency-admission.json", receipt)
    emit("at_dependencies_admitted", **receipt)
    return receipt


def assets(cfg, contract, out, env):
    receipt = check_receipt(cfg, contract, env)  # Always precedes network/model assets.
    write(out / "dependency-admission.json", receipt)
    private = Path(cfg["host"]["private_environment"]).expanduser()
    root = Path(cfg["host"]["asset_root"]).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    stage = cfg["stage"]
    minimum = cfg["guards"]["minimum_free_gib_model_assets" if stage == "model-assets" else "minimum_free_gib_dataset_assets"]
    if shutil.disk_usage(root).free < minimum * 2**30:
        raise RuntimeError("Insufficient asset disk; no download started")
    helper = HERE / "vendor/native_input_preparation.py"
    common = [str(private / "bin/python"), "-u", str(helper)]
    if stage in {"dataset-assets", "model-assets", "assets-verify"}:
        command = common + ["codealpaca-native", "--metadata", str(HERE / "inputs/codealpaca-origin-metadata.json"),
                            "--tree", str(HERE / "inputs/codealpaca-origin-tree.json"), "--h4-script", str(HERE / "inputs/codealpaca-create_dataset.py"),
                            "--h4-tree", str(HERE / "inputs/codealpaca-tree.json"), "--output-dir", str(root / "CodeAlpaca-20k")]
        if stage in {"model-assets", "assets-verify"}:
            command.append("--offline")
        run_logged(command, out / "dataset-preparation.log", env, cfg["timeouts"]["dataset_asset_seconds"])
    if stage in {"model-assets", "assets-verify"}:
        command = common + ["codellama", "--tree", str(HERE / "inputs/codellama-tree.json"),
                            "--output-dir", str(root / "CodeLlama-7b-hf")]
        if stage == "assets-verify":
            command.append("--offline")
        run_logged(command, out / "model-preparation.log", env, cfg["timeouts"]["model_asset_seconds"])
    model_manifest, data_manifest = root / "CodeLlama-7b-hf/manifest.json", root / "CodeAlpaca-20k/manifest.json"
    result = {"schema": "original-at-asset-admission-v1", "stage": stage,
              "contract_sha256": contract["contract_sha256"], "dependency_gate_passed": True,
              "model_manifest_present": model_manifest.is_file(), "dataset_manifest_present": data_manifest.is_file(),
              "model_manifest_path": str(model_manifest), "dataset_manifest_path": str(data_manifest),
              "model_manifest_sha256": sha(model_manifest) if model_manifest.is_file() else None,
              "dataset_manifest_sha256": sha(data_manifest) if data_manifest.is_file() else None,
              "full_offline_reverification_completed": stage == "assets-verify",
              "next_executor_host": {"base_python": cfg["host"]["base_python"], "private_environment": str(private),
                  "model_path": str(root / "CodeLlama-7b-hf"), "model_revision": cfg["pins"]["model_revision"],
                  "model_manifest": str(model_manifest), "dataset_manifest": str(data_manifest)},
              "native_runtime_canary_or_full_execution_performed": False}
    write(out / "asset-admission.json", result)
    emit("at_asset_stage_complete", **result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    contract = validate(cfg)
    if args.validate_only:
        emit("at_static_validation_passed", **contract, execution_performed=False)
        return 0
    run_id = os.environ.get("ORX_RUN_ID")
    if not run_id:
        raise RuntimeError("Runtime setup/asset work must be launched through a recorded ORX run")
    out = REPO / cfg["output_root"] / run_id
    out.mkdir(parents=True, exist_ok=False)
    write(out / "effective-run.json", cfg)
    write(out / "source-contract.json", contract)
    started = time.monotonic()
    status, error = "FAILED", None
    stage_seconds = cfg["timeouts"]["dependency_seconds" if cfg["stage"] == "dependencies" else
                                    "model_asset_seconds" if cfg["stage"] in {"model-assets", "assets-verify"} else "dataset_asset_seconds"]
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(int(stage_seconds))
    try:
        env = software_env(cfg)
        if cfg["stage"] == "dependencies":
            dependencies(cfg, contract, out, env)
        else:
            assets(cfg, contract, out, env)
        status = "COMPLETE_SOFTWARE_OR_ASSETS_ONLY"
        return 0
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        signal.alarm(0)
        terminal = {"status": status, "error": error, "stage": cfg["stage"], "run_id": run_id,
                    "wall_seconds": time.monotonic() - started, "model_training_generation_attack_executed": False}
        write(out / "terminal.json", terminal)
        emit("at_setup_terminal", **terminal)


def stop(signum, _frame):
    if ACTIVE_CHILD is not None and ACTIVE_CHILD.poll() is None:
        os.killpg(ACTIVE_CHILD.pid, signal.SIGTERM)
    raise SystemExit(128 + signum)


def deadline(_signum, _frame):
    if ACTIVE_CHILD is not None and ACTIVE_CHILD.poll() is None:
        os.killpg(ACTIVE_CHILD.pid, signal.SIGTERM)
    raise TimeoutError("Overall AT setup-stage wall budget exceeded")


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    sys.exit(main())
