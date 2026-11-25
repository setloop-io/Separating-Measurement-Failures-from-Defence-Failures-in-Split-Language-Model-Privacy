"""Read-only integrity review of relocated R1 execution evidence.

Original absolute paths remain identities, never local read targets. This verifies
retained bytes and provenance links, not a fresh scientific execution or tensor replay.
"""
from __future__ import annotations

import json
import tarfile
from pathlib import Path

from .utils import sha256_bytes, sha256_file, sha256_json


def _read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"unreadable evidence: {path}") from exc


def _file(path, root):
    path, root = Path(path), Path(root).resolve()
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"artifact escapes retained root: {path}")
    relative = path.absolute().relative_to(root)
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"symlink evidence is forbidden: {path}")
    if not path.is_file():
        raise ValueError(f"artifact missing: {path}")
    return path


def _digest(path, digest, root):
    if sha256_file(str(_file(path, root))) != digest:
        raise ValueError(f"artifact hash differs: {path}")


def _relative(value):
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or str(path) in ("", "."):
        raise ValueError(f"unsafe relative artifact path: {value}")
    return path


def _source_review(run_dir, manifest, resolved, launch):
    record = manifest.get("execution_provenance")
    if not isinstance(record, dict) or _read(run_dir / "provenance/run_start.json") != record:
        raise ValueError("manifest source provenance is absent or differs")
    for key, expected in (("run_id", resolved["run_id"]),
                          ("output_dir", resolved["output_dir"]),
                          ("resolved_config_sha256", sha256_json(resolved)),
                          ("git_commit", manifest["git_commit"]),
                          ("environment_digest", manifest["environment_digest"])):
        if record.get(key) != expected:
            raise ValueError(f"source provenance {key} differs")
    directory = run_dir / "provenance"
    for name, key in (("environment.txt", "environment_digest"),
                      ("working_tree.patch", "working_tree_patch_sha256")):
        _digest(directory / name, record[key], run_dir)
    empty = (directory / "working_tree.patch").stat().st_size == 0
    if empty != record["working_tree_patch_empty"]:
        raise ValueError("source patch emptiness differs")
    if manifest["dirty_patch_sha256"] != (None if empty else record["working_tree_patch_sha256"]):
        raise ValueError("manifest source patch digest differs")
    inventory = _read(directory / "source_inventory.json")
    if not isinstance(inventory, list) or not inventory or sha256_json(inventory) != record["source_inventory_sha256"]:
        raise ValueError("source inventory digest differs")
    expected = {}
    for item in inventory:
        name = str(_relative(item["path"]))
        if name in expected:
            raise ValueError("duplicate source inventory path")
        expected[name] = item["sha256"]
    observed = {}
    with tarfile.open(directory / "source.tar.gz", "r:gz") as archive:
        for member in archive:
            if not member.isfile() or member.name not in expected or member.name in observed:
                raise ValueError(f"unexpected source archive member: {member.name}")
            payload = archive.extractfile(member)
            if payload is None:
                raise ValueError("unreadable source archive member")
            observed[member.name] = sha256_bytes(payload.read())
    if observed != expected:
        raise ValueError("source archive differs from inventory")
    if launch.get("protocol") and expected.get(launch["protocol"]) != launch["protocol_sha256"]:
        raise ValueError("archived protocol differs from launch")


def verify_stage_bindings(root, launch):
    """Check actual scientific-stage envelope; synthetic fixtures have no envelope."""
    root = Path(root).resolve()
    if launch.get("schema_version") != "r1-scientific-launch-1.0":
        return {"scientific_launch": False}
    if _read(root / "launch.json") != launch or launch.get("scientific_evidence") is not True:
        raise ValueError("scientific launch identity differs")
    for name, key in (("admission.json", "admission_sha256"),
                      ("admission_claim.json", "admission_claim_sha256")):
        _digest(root / name, launch[key], root)
    admission = _read(root / "admission.json")
    claim = _read(root / "admission_claim.json")
    if (claim.get("admission_sha256") != sha256_json(admission)
            or claim.get("execution_id") != launch.get("execution_id")
            or claim.get("scientific_evidence") is not True):
        raise ValueError("admission claim identity differs")
    for key in ("stage", "study_id", "bundle_sha256", "protocol_sha256", "attack_spec_sha256", "attack_spec_status"):
        if admission.get(key) != launch.get(key):
            raise ValueError(f"admission {key} differs from launch")
    if admission.get("run_count") != len(launch["runs"]):
        raise ValueError("admission run count differs")
    for field in ("budget", "environment", "environment_review", "attack_spec_document"):
        key = "attack_spec_sha256" if field == "attack_spec_document" else field + "_sha256"
        if sha256_json(admission[field]) != admission[key]:
            raise ValueError(f"admission {field} digest differs")
    spec = _read(_file(root / _relative(launch["attack_spec"]), root))
    if spec != admission["attack_spec_document"] or sha256_json(spec) != launch["attack_spec_sha256"]:
        raise ValueError("stage attack specification differs")
    panel = _read(_file(root / _relative(launch["panel_manifest"]), root))
    payload = dict(panel)
    digest = payload.pop("panel_sha256", None)
    if digest != launch["panel_sha256"] or sha256_json(payload) != digest:
        raise ValueError("stage panel canonical digest differs")
    summary = _read(root / "execution_summary.json")
    if summary.get("launch_sha256") != sha256_file(str(root / "launch.json")):
        raise ValueError("execution summary launch digest differs")
    return {"scientific_launch": True, "admission_verified": True, "panel_verified": True}


def verify_run_artifacts(root, run_dir, manifest_record, resolved, launch):
    """Verify all artifact bytes, required inventory and scientific cross-bindings."""
    root, run_dir = Path(root).resolve(), Path(run_dir).resolve()
    if run_dir != root / "runs" / resolved["run_id"]:
        raise ValueError("run directory differs from stage identity")
    scientific = bool(resolved.get("r1", {}).get("semantic_validation"))
    manifest = manifest_record
    if scientific:
        from .manifest import validate_manifest, _required_artifacts
        try:
            validate_manifest(manifest)
        except Exception as exc:
            raise ValueError(f"invalid scientific manifest: {exc}") from exc
        required = _required_artifacts(resolved, prospective=True)
        required |= {"provenance/r1_initialization.json", "final_state.pt"}
    else:
        required = {"config.resolved.json", "metrics_summary.json"}
    if resolved.get("mode") == "trained" and resolved.get("split") is not None:
        required |= {"attack_suite.json", "attack_banks.pt"}
        if scientific:
            required.add("attack_spec_binding.json")
    original_output = Path(resolved["output_dir"])
    bundle = Path(launch.get("bundle_root_at_execution", "/"))
    if not original_output.is_absolute():
        original_output = bundle / original_output
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("manifest has no artifact inventory")
    verified = set()
    for item in artifacts:
        recorded = Path(item["path"])
        if ".." in recorded.parts:
            raise ValueError("artifact path contains parent traversal")
        if not recorded.is_absolute():
            recorded = bundle / recorded
        try:
            name = str(recorded.relative_to(original_output))
        except ValueError as exc:
            raise ValueError(f"artifact escapes original run directory: {recorded}") from exc
        if name in verified:
            raise ValueError(f"duplicate artifact: {name}")
        _digest(run_dir / name, item["sha256"], run_dir)
        verified.add(name)
    if missing := required - verified:
        raise ValueError(f"required artifacts not bound: {sorted(missing)}")
    if _read(run_dir / "config.resolved.json") != resolved:
        raise ValueError("resolved configuration bytes differ from supplied configuration")
    if scientific:
        if manifest.get("scientific_evidence") is not True:
            raise ValueError("manifest does not admit scientific evidence")
        for key in ("run_id", "seed", "cohort", "protocol_version", "pairing_block_id"):
            if manifest.get(key) != resolved.get(key):
                raise ValueError(f"manifest {key} differs")
        if manifest.get("status") != "completed" or manifest.get("resolved_config_sha256") != sha256_json(resolved):
            raise ValueError("manifest completion/configuration differs")
        _source_review(run_dir, manifest, resolved, launch)
        begin = _read(root / "execution_records" / f"{resolved['run_id']}.start.json")
        for key, expected in (("run_id", resolved["run_id"]), ("config_sha256", sha256_json(resolved)),
                              ("launch_uuid", launch["launch_uuid"]),
                              ("launch_sha256", sha256_file(str(root / "launch.json")))):
            if begin.get(key) != expected:
                raise ValueError(f"start record {key} differs")
        end = _read(root / "execution_records" / f"{resolved['run_id']}.end.json")
        if (end.get("run_id") != resolved["run_id"] or end.get("launch_uuid") != launch["launch_uuid"]
                or end.get("status") != "completed"
                or end.get("start_sha256") != sha256_file(str(root / "execution_records" / f"{resolved['run_id']}.start.json"))):
            raise ValueError("end record identity/start binding differs")
        panel_binding = _read(run_dir / "panel_binding.json")
        panel_path = root / launch["panel_manifest"]
        if (run_dir / panel_binding["path"]).resolve() != panel_path.resolve():
            raise ValueError("run panel path differs from stage panel")
        _digest(panel_path, panel_binding["sha256"], root)
        if resolved.get("split") is not None:
            _split_review(root, run_dir, resolved, launch, verified)
    return {"artifact_count": len(verified), "all_manifest_hashes_verified": True,
            "scientific_provenance_verified": scientific,
            "scope": "retained byte and provenance verification; no numerical replay"}


def _split_review(root, run_dir, resolved, launch, verified):
    required = {"training_captures.pt", "training_captures.json",
                "remote_snapshots/remote-initial.pt", "remote_snapshots/remote-final.pt",
                "remote_snapshots/remote-snapshots.json"}
    if required - verified:
        raise ValueError(f"scientific split artifacts missing: {sorted(required - verified)}")
    frames = _read(run_dir / "training_captures.json")
    for item in ({"path": frames["capture_file"], "sha256": frames["capture_sha256"]}, frames["remote_captures"]):
        name = str(_relative(item["path"]))
        if name not in verified:
            raise ValueError("training capture is not manifest-bound")
        _digest(run_dir / name, item["sha256"], run_dir)
    snapshots = _read(run_dir / "remote_snapshots/remote-snapshots.json")
    for kind in ("initial", "final"):
        item = snapshots[kind]
        _digest(run_dir / "remote_snapshots" / _relative(item["path"]), item["sha256"], run_dir)
    remote = run_dir / f"remote-process/remote/remote-checkpoint-step-{resolved['optim']['steps']}.pt"
    _digest(remote, snapshots["final"]["remote_checkpoint_sha256"], run_dir)
    if resolved["mode"] != "trained":
        return
    report = _read(run_dir / "attack_suite.json")
    document = _read(root / launch["attack_spec"])
    if report["specification"] != document["specification"]:
        raise ValueError("attack report specification differs from launch")
    expected = {"panel": launch["panel_sha256"],
                "client_checkpoint": sha256_file(str(run_dir / "training_checkpoint.pt")),
                "remote_checkpoint": sha256_file(str(remote)),
                "capture": sha256_file(str(run_dir / "transcript_tensors.pt"))}
    if report["input_bindings"] != expected:
        raise ValueError("attack checkpoint/capture/panel bindings differ")
    binding = _read(run_dir / "attack_spec_binding.json")
    if binding.get("run_id") != resolved["run_id"] or binding.get("specification_sha256") != sha256_json(document["specification"]):
        raise ValueError("run attack specification binding differs")
    if binding.get("status") != document["status"]:
        raise ValueError("run attack specification status differs")
    if document["status"] == "frozen":
        frozen = binding.get("frozen_document") or {}
        _digest(root / launch["attack_spec"], frozen.get("sha256"), root)
    for key, result in report["results"].items():
        qualification = result["qualification"]
        spec = document["specification"]
        floor = spec["positive_floor_correlation" if key.startswith("punctuation_fraction@") else "positive_floor_accuracy"]
        if (qualification["positive_floor"] != floor or qualification["null_band"] != spec["null_band"]
                or sorted(row["dose"] for row in qualification["positive_control"]) != sorted(spec["positive_doses"])):
            raise ValueError("attack qualification parameters differ from specification")
