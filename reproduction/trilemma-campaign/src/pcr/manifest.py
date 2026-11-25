"""Run manifests: schema-validated provenance records for every executed cell.

Legacy records remain readable. Prospective executions additionally retain a
source archive before training. A status flag alone never authorizes a skip or
reuse of a historical output directory.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from jsonschema import validate

from . import provenance
from .transcript import TRANSCRIPT_SCHEMA_VERSION
from .utils import sha256_bytes, sha256_file, sha256_json

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_SCHEMA_PATH = REPO_ROOT / "experiments" / "schemas" / "run_manifest.schema.json"
MANIFEST_DIR = REPO_ROOT / "experiments" / "manifests"
MANIFEST_SCHEMA_VERSION = "0.1.0"


class ManifestVerificationError(RuntimeError):
    """A preserved run cannot safely be skipped, resumed, or overwritten."""

    def __init__(self, kind: str, run_id: str, detail: str):
        self.kind = kind
        super().__init__(f"{kind}: {run_id}: {detail}. Preserve existing evidence; retrieve "
                         "the original artifacts or use a separately identified new execution.")


def _manifest_path(run_id: str) -> Path:
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ManifestVerificationError("identity_mismatch", run_id, "invalid run identifier")
    return MANIFEST_DIR / f"{run_id}.json"


def _read_manifest(path: Path, run_id: str) -> dict:
    if path.is_symlink():
        raise ManifestVerificationError("invalid_manifest", run_id, "manifest path is a symlink")
    try:
        manifest = json.loads(path.read_text())
        validate_manifest(manifest)
    except Exception as exc:
        raise ManifestVerificationError("invalid_manifest", run_id, str(exc)) from exc
    return manifest


def _required_artifacts(resolved: dict, *, prospective: bool) -> set[str]:
    names = {"config.resolved.json", "curve.jsonl", "metrics_summary.json"}
    if resolved.get("split") is not None:
        names.update({"transcript_tensors.pt", "transcript_inventory.json", "transcript_bytes_per_step.json"})
        if resolved.get("cohort") == "calibration":
            names.add("calibration_pairs.pt")
    if resolved.get("canary"):
        names.add("canaries.pt")
    if resolved.get("save_final_state"):
        names.add("final_state.pt")
    if prospective:
        names.update(f"provenance/{name}" for name in (
            "source.tar.gz", "source_inventory.json", "working_tree.patch", "environment.txt", "run_start.json"
        ))
    if resolved.get("r1", {}).get("semantic_validation"):
        names.update({"training_checkpoint.pt", "semantic_validation.json", "panel_binding.json"})
        if resolved.get("split") is not None:
            step = int(resolved["optim"]["steps"])
            names.update({"client-wire-audit.json",
                          "remote-process/remote/remote-wire-audit.json",
                          "remote-process/remote/step-journal.jsonl",
                          f"remote-process/remote/remote-checkpoint-step-{step}.pt",
                          f"remote-process/remote/remote-captures-step-{step}.pt",
                          "remote-process/remote-root/packet.json"})
            if resolved.get("mode") == "trained" and resolved.get("r1", {}).get("attack_validation"):
                names.update({"attack_suite.json", "attack_banks.pt"})
    return names


def _verify_artifacts(manifest: dict, resolved: dict) -> None:
    run_id = resolved["run_id"]
    output_dir = Path(resolved["output_dir"]).resolve()
    if not manifest["artifacts"]:
        raise ManifestVerificationError("artifacts_unavailable", run_id, "empty artifact list")
    verified = set()
    for item in manifest["artifacts"]:
        artifact = Path(item["path"])
        artifact = (artifact if artifact.is_absolute() else REPO_ROOT / artifact).resolve()
        if not artifact.is_relative_to(output_dir) or artifact in verified:
            raise ManifestVerificationError("identity_mismatch", run_id,
                                            f"escaping or duplicate artifact path: {artifact}")
        if not artifact.is_file():
            raise ManifestVerificationError("artifacts_unavailable", run_id, str(artifact))
        if sha256_file(str(artifact)) != item["sha256"]:
            raise ManifestVerificationError("hash_mismatch", run_id, str(artifact))
        verified.add(artifact)
    if manifest["status"] == "completed":
        for name in _required_artifacts(resolved, prospective="execution_provenance" in manifest):
            if (output_dir / name).resolve() not in verified:
                raise ManifestVerificationError("artifacts_unavailable", run_id,
                                                f"required artifact is not bound: {name}")
        if resolved.get("r1", {}).get("semantic_validation"):
            _verify_semantics(output_dir, resolved)


def _verify_semantics(output_dir: Path, resolved: dict) -> None:
    from .artifacts import (
        validate_attack_banks, validate_attack_report, validate_capture,
        validate_capture_pair, validate_capture_panel,
        validate_checkpoint_pair,
        validate_config, validate_curve, validate_inventory_file, validate_metrics,
        validate_remote_checkpoint,
        validate_semantic_record, validate_state_dict, validate_step_journal,
        validate_training_checkpoint, validate_wire_audit,
    )
    from .r1_packets import verify_process_packet
    from .r1_panels import verify_panel_manifest
    try:
        validate_config(output_dir / "config.resolved.json", resolved)
        validate_curve(output_dir / "curve.jsonl", int(resolved["optim"]["steps"]))
        validate_metrics(output_dir / "metrics_summary.json", resolved)
        initialization = json.loads((output_dir / "provenance/r1_initialization.json").read_text())
        shapes = (initialization["parameter_shapes"] if resolved.get("split") is None
                  else initialization["trusted_parameter_shapes"])
        validate_state_dict(output_dir / "final_state.pt", shapes)
        validate_training_checkpoint(output_dir / "training_checkpoint.pt",
                                     int(resolved["optim"]["steps"]), shapes,
                                     require_remote=resolved.get("split") is not None)
        validate_semantic_record(output_dir / "semantic_validation.json", output_dir)
        binding = json.loads((output_dir / "panel_binding.json").read_text())
        panel_value = Path(binding["path"])
        panel_path = (panel_value if panel_value.is_absolute() else output_dir / panel_value).resolve()
        if (panel_path.is_symlink() or not panel_path.is_file() or
                sha256_file(str(panel_path)) != binding["sha256"]):
            raise ValueError("bound panel manifest is unavailable or changed")
        panel_manifest = json.loads(panel_path.read_text())
        data_path = Path(resolved["data"]["path"])
        data_path = data_path if data_path.is_absolute() else REPO_ROOT / data_path
        verify_panel_manifest(data_path, panel_manifest)
        if resolved.get("split") is not None:
            split = resolved["split"]
            channels = ("forward_values", "returned_values", "inbound_gradients", "outbound_gradients")
            client_captures = output_dir / "transcript_tensors.pt"
            remote_captures = output_dir / f"remote-process/remote/remote-captures-step-{int(resolved['optim']['steps'])}.pt"
            validate_capture(client_captures, channels,
                             int(split["width"]), int(resolved["data"]["seq_len"]))
            validate_capture_panel(client_captures, panel_manifest)
            validate_capture_pair(client_captures, remote_captures, channels)
            validate_inventory_file(output_dir / "transcript_inventory.json")
            client_audit = output_dir / "client-wire-audit.json"
            remote_audit = output_dir / "remote-process/remote/remote-wire-audit.json"
            validate_wire_audit(client_audit)
            validate_wire_audit(remote_audit)
            if json.loads(client_audit.read_text())["total_bytes"] != json.loads(remote_audit.read_text())["total_bytes"]:
                raise ValueError("client and remote wire totals differ")
            step = int(resolved["optim"]["steps"])
            validate_step_journal(output_dir / "remote-process/remote/step-journal.jsonl", step)
            remote_checkpoint = output_dir / f"remote-process/remote/remote-checkpoint-step-{step}.pt"
            validate_remote_checkpoint(remote_checkpoint, step)
            validate_checkpoint_pair(output_dir / "training_checkpoint.pt", remote_checkpoint, step)
            verify_process_packet(output_dir / "remote-process/remote-root", "remote-service")
            if resolved.get("mode") == "trained" and resolved.get("r1", {}).get("attack_validation"):
                attack_path = output_dir / "attack_suite.json"
                banks_path = output_dir / "attack_banks.pt"
                validate_attack_report(attack_path)
                report = json.loads(attack_path.read_text())
                expected_bindings = {
                    "panel": panel_manifest["panel_sha256"],
                    "client_checkpoint": sha256_file(str(output_dir / "training_checkpoint.pt")),
                    "remote_checkpoint": sha256_file(str(remote_checkpoint)),
                    "capture": sha256_file(str(client_captures)),
                }
                bank_digests = validate_attack_banks(banks_path, report["defense"])
                if (report["input_bindings"] != expected_bindings or
                        report["panel_bindings"]["panel_sha256"] != panel_manifest["panel_sha256"] or
                        report["panel_bindings"]["checkpoint_id"] != f"step-{step}" or
                        sha256_file(str(banks_path)) != report["attack_banks_sha256"] or
                        {key: item["attack_bank_sha256"] for key, item in report["results"].items()} != bank_digests):
                    raise ValueError("attack report and retained banks differ")
    except Exception as exc:
        raise ManifestVerificationError("semantic_mismatch", resolved["run_id"], str(exc)) from exc


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def _repo_relative(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def git_commit() -> str:
    return (_run(["git", "rev-parse", "HEAD"]) or "unknown-commit").strip()


def dirty_patch_sha256() -> str | None:
    diff = _run(["git", "diff"])
    if diff is None or diff == "":
        return None
    return sha256_bytes(diff.encode("utf-8"))


def git_status_porcelain_sha256() -> str | None:
    status = _run(["git", "status", "--porcelain"])
    if status is None or status == "":
        return None
    return sha256_bytes(status.encode("utf-8"))


def environment_digest() -> str:
    freeze = _run([sys.executable, "-m", "pip", "freeze"])
    if freeze is None:
        freeze = _run(["uv", "pip", "freeze", "--python", sys.executable])
    if freeze is None:
        freeze = f"unavailable-freeze python={sys.version}"
    return sha256_bytes(freeze.encode("utf-8"))


def build_manifest(
    resolved: dict,
    status: str,
    artifact_paths: list[Path],
    captured_channels: list[str],
    extra: dict | None = None,
) -> dict:
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": resolved["run_id"],
        "protocol_version": resolved["protocol_version"],
        "cohort": resolved.get("cohort", "exploratory"),
        "pairing_block_id": resolved.get("pairing_block_id"),
        "seed": resolved["seed"],
        "git_commit": git_commit(),
        "dirty_patch_sha256": dirty_patch_sha256(),
        "environment_digest": environment_digest(),
        "resolved_config_sha256": sha256_json(resolved),
        "transcript_schema_version": TRANSCRIPT_SCHEMA_VERSION,
        "captured_channels": sorted(set(captured_channels)),
        "status": status,
        "artifacts": [
            {"path": _repo_relative(p), "sha256": sha256_file(str(p))} for p in artifact_paths
        ],
        "git_status_porcelain_sha256": git_status_porcelain_sha256(),
        "note": (f"Recorded {resolved.get('cohort', 'exploratory')} cohort; this metadata "
                 "does not by itself establish qualification, confirmation, or a pre-outcome freeze."),
    }
    if extra:
        if set(extra).intersection(manifest):
            raise ManifestVerificationError("identity_mismatch", resolved["run_id"],
                                            "extra metadata cannot override required manifest fields")
        manifest.update(extra)
    output_dir = Path(resolved["output_dir"]) if "output_dir" in resolved else None
    start = output_dir / "provenance" / "run_start.json" if output_dir else None
    if start is not None and start.is_file():
        recorded = json.loads(start.read_text())
        if (recorded["run_id"] != resolved["run_id"] or
                recorded["resolved_config_sha256"] != sha256_json(resolved) or
                recorded["output_dir"] != str(output_dir.resolve())):
            raise ManifestVerificationError("identity_mismatch", resolved["run_id"],
                                            "run-start provenance differs from the resolved config")
        try:
            provenance.verify_snapshot(REPO_ROOT, output_dir, recorded)
        except Exception as exc:
            raise ManifestVerificationError("source_mismatch", resolved["run_id"], str(exc)) from exc
        manifest.update({
            "schema_version": "0.2.0", "git_commit": recorded["git_commit"],
            "dirty_patch_sha256": (None if recorded["working_tree_patch_empty"]
                                   else recorded["working_tree_patch_sha256"]),
            "git_status_porcelain_sha256": recorded["git_status_porcelain_sha256"],
            "environment_digest": recorded["environment_digest"],
            "execution_provenance": recorded,
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        bound = {item["path"] for item in manifest["artifacts"]}
        for path in [output_dir / "config.resolved.json", *sorted(start.parent.iterdir())]:
            if path.is_file() and _repo_relative(path) not in bound:
                manifest["artifacts"].append({"path": _repo_relative(path),
                                             "sha256": sha256_file(str(path))})
                bound.add(_repo_relative(path))
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: dict) -> None:
    schema = json.loads(MANIFEST_SCHEMA_PATH.read_text())
    validate(instance=manifest, schema=schema)


def write_manifest(
    resolved: dict,
    status: str,
    artifact_paths: list[Path],
    captured_channels: list[str],
    extra: dict | None = None,
) -> Path:
    path = _manifest_path(resolved["run_id"])
    if path.exists() or path.is_symlink():
        raise ManifestVerificationError("evidence_exists", resolved["run_id"],
                                        f"refusing to replace {path}")
    manifest = build_manifest(resolved, status, artifact_paths, captured_channels, extra)
    if "execution_provenance" not in manifest:
        raise ManifestVerificationError("source_unavailable", resolved["run_id"],
                                        "new manifests require provenance captured before execution")
    _verify_artifacts(manifest, resolved)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also protects against a second writer appearing after
    # the initial check. Never replace historical records, including failures.
    with path.open("x") as handle:
        handle.write(json.dumps(manifest, indent=2, sort_keys=True))
    return path


def completed_manifest_exists(run_id: str, *, resolved: dict) -> bool:
    """True only for verified complete evidence of the requested execution.

    Missing manifests permit a new run only in an empty output directory. Every
    other incomplete or unverifiable condition raises, never returning False to
    authorize an automatic rerun under a historical identity.
    """
    path = _manifest_path(run_id)
    output_dir = Path(resolved["output_dir"]).resolve()
    if resolved["run_id"] != run_id:
        raise ManifestVerificationError("identity_mismatch", run_id, "requested run_id differs")
    if not path.exists() and not path.is_symlink():
        if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
            raise ManifestVerificationError("unbound_artifacts", run_id,
                                            f"nonempty output directory lacks a manifest: {output_dir}")
        return False
    manifest = _read_manifest(path, run_id)
    for key in ("run_id", "protocol_version", "seed", "cohort", "pairing_block_id"):
        expected = resolved.get(key, "exploratory" if key == "cohort" else None)
        if manifest[key] != expected:
            raise ManifestVerificationError("identity_mismatch", run_id, f"{key} differs")
    if manifest["resolved_config_sha256"] != sha256_json(resolved):
        raise ManifestVerificationError("identity_mismatch", run_id,
                                        "resolved config differs, including the requested output path")
    if manifest["transcript_schema_version"] != TRANSCRIPT_SCHEMA_VERSION:
        raise ManifestVerificationError("identity_mismatch", run_id, "transcript schema differs")
    if manifest["status"] != "completed":
        raise ManifestVerificationError("incomplete_evidence", run_id,
                                        f"recorded status is {manifest['status']}; no automatic overwrite")
    _verify_artifacts(manifest, resolved)
    try:
        stored_config = json.loads((output_dir / "config.resolved.json").read_text())
        if sha256_json(stored_config) != manifest["resolved_config_sha256"]:
            raise ValueError("config artifact does not match the manifest's resolved-config digest")
    except (OSError, ValueError) as exc:
        raise ManifestVerificationError("identity_mismatch", run_id, str(exc)) from exc
    recorded = manifest.get("execution_provenance")
    if recorded is None:
        raise ManifestVerificationError("source_unavailable", run_id,
                                        "legacy manifest has no retained execution source snapshot")
    try:
        if (recorded["run_id"] != run_id or recorded["output_dir"] != str(output_dir) or
                recorded["resolved_config_sha256"] != sha256_json(resolved)):
            raise ValueError("execution provenance identity differs")
        for name in ("git_commit", "environment_digest"):
            if manifest[name] != recorded[name]:
                raise ValueError(f"manifest {name} differs from its run-start record")
        expected_patch = None if recorded["working_tree_patch_empty"] else recorded["working_tree_patch_sha256"]
        if manifest["dirty_patch_sha256"] != expected_patch:
            raise ValueError("manifest patch digest differs from its run-start record")
        if json.loads((output_dir / "provenance" / "run_start.json").read_text()) != recorded:
            raise ValueError("manifest provenance differs from the bound run-start record")
        provenance.verify_snapshot(REPO_ROOT, output_dir, recorded)
    except Exception as exc:
        raise ManifestVerificationError("source_mismatch", run_id, str(exc)) from exc
    return True


def begin_run(resolved: dict) -> None:
    """Reserve a fresh output directory and retain provenance before any training."""
    if completed_manifest_exists(resolved["run_id"], resolved=resolved):
        raise ManifestVerificationError("evidence_exists", resolved["run_id"],
                                        "direct execution would overwrite verified complete evidence")
    if "data" in resolved:
        data = resolved["data"]
        try:
            data_path = Path(data["path"])
            data_path = data_path if data_path.is_absolute() else REPO_ROOT / data_path
            if sha256_file(str(data_path)) != data["corpus_sha256"]:
                raise ValueError("corpus bytes differ from the resolved configuration digest")
        except (OSError, KeyError, ValueError) as exc:
            raise ManifestVerificationError("identity_mismatch", resolved["run_id"], str(exc)) from exc
    output_dir = Path(resolved["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        provenance.capture(REPO_ROOT, output_dir, resolved, _run)
    except Exception as exc:
        raise ManifestVerificationError("source_unavailable", resolved["run_id"],
                                        f"could not retain pre-execution provenance: {exc}") from exc


def report_resume_status(runs: list[dict]) -> int:
    """Dry-run inventory: report every blocked cell without starting training."""
    blocked = 0
    for index, resolved in enumerate(runs, start=1):
        try:
            status = ("verified_complete" if completed_manifest_exists(resolved["run_id"], resolved=resolved)
                      else "pending_new_execution")
        except ManifestVerificationError as exc:
            blocked += 1
            status = f"blocked: {exc}"
        print(f"[{index}/{len(runs)}] {resolved['run_id']} cohort={resolved.get('cohort', 'exploratory')} ({status})")
    return 1 if blocked else 0
