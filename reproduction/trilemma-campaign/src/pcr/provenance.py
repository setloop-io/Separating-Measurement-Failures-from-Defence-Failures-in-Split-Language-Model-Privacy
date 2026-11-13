"""Retain a bounded filesystem source snapshot before prospective executions.

These records describe collection by the current runner. They neither reconstruct
legacy executions nor establish an independently witnessed preregistration date.
"""

from __future__ import annotations

import io
import json
import platform
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

from .utils import sha256_bytes, sha256_file, sha256_json


def source_paths(root: Path) -> list[Path]:
    """Include staged/untracked working files in explicitly bounded source roots."""
    patterns = (
        "src/**/*.py", "scripts/**/*.py", "experiments/configs/**/*",
        "experiments/schemas/**/*", "docs/PREREGISTRATION*.md",
        "docs/METHODOLOGY.md", "docs/SEED_STREAM_AND_COHORT_POLICY.md",
        "docs/RESEARCH_PLAN.md", "pyproject.toml", "uv.lock", "requirements*.txt",
        "docs/R1_*.md", "environment/r1/**/*", "experiments/remote/r1/**/*",
        "evidence/R1_EVIDENCE_POLICY.json",
    )
    paths = {p for pattern in patterns for p in root.glob(pattern) if p.is_file()}
    for path in paths:
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"source snapshot refuses a symlink or escaping path: {path}")
    return sorted(paths)


def source_inventory(root: Path) -> list[dict]:
    return [{"path": str(p.relative_to(root)), "sha256": sha256_file(str(p))}
            for p in source_paths(root)]


def capture(root: Path, output_dir: Path, resolved: dict, run_command) -> dict:
    """Write source/config/environment records to an already reserved empty run dir."""
    directory = output_dir / "provenance"
    directory.mkdir()
    sources = []
    with tarfile.open(directory / "source.tar.gz", "w:gz") as archive:
        for path in source_paths(root):
            payload = path.read_bytes()
            name = str(path.relative_to(root))
            sources.append({"path": name, "sha256": sha256_bytes(payload)})
            entry = tarfile.TarInfo(name)
            entry.size = len(payload)
            entry.mode = 0o644
            archive.addfile(entry, io.BytesIO(payload))
    (directory / "source_inventory.json").write_text(json.dumps(sources, indent=2) + "\n")
    # HEAD diff includes staged and unstaged tracked edits. The archive above
    # independently retains untracked source files; the patch alone is incomplete.
    patch = run_command(["git", "diff", "HEAD", "--binary"])
    (directory / "working_tree.patch").write_text(patch or "")
    freeze = run_command([sys.executable, "-m", "pip", "freeze"])
    if freeze is None:
        freeze = run_command(["uv", "pip", "freeze", "--python", sys.executable])
    environment_available = freeze is not None
    freeze = freeze if freeze is not None else f"unavailable-freeze python={sys.version}\n"
    (directory / "environment.txt").write_text(freeze)
    # Capture before training, including when an execution subsequently fails.
    (output_dir / "config.resolved.json").write_text(
        json.dumps(resolved, indent=2, sort_keys=True) + "\n"
    )
    recorded = {
        "version": "1", "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": resolved["run_id"], "resolved_config_sha256": sha256_json(resolved),
        "output_dir": str(output_dir.resolve()),
        "git_commit": (run_command(["git", "rev-parse", "HEAD"]) or "unknown-commit").strip(),
        "git_diff_head_available": patch is not None,
        "working_tree_patch_sha256": sha256_bytes((patch or "").encode()),
        "working_tree_patch_empty": not (patch or ""),
        "source_inventory_sha256": sha256_json(sources),
        "environment_digest": sha256_bytes(freeze.encode()),
        "environment_available": environment_available,
        "argv": list(sys.argv), "cwd": str(Path.cwd()), "python": sys.version,
        "executable": sys.executable, "platform": platform.platform(),
        "hostname": platform.node(), "requested_device": resolved.get("device", "auto"),
        "scope": "Filesystem snapshot of Python source, configs, schemas and protocol documents; "
                 "dependencies are recorded by environment.txt. No historical reconstruction "
                 "or independently witnessed protocol freeze is asserted.",
    }
    git_status = run_command(["git", "status", "--porcelain"])
    recorded["git_status_porcelain_sha256"] = (
        sha256_bytes(git_status.encode()) if git_status else None
    )
    (directory / "run_start.json").write_text(json.dumps(recorded, indent=2, sort_keys=True) + "\n")
    return recorded


def verify_snapshot(root: Path, output_dir: Path, recorded: dict) -> None:
    """Check retained archive against its inventory and the current source tree."""
    directory = output_dir / "provenance"
    for name, key in (("working_tree.patch", "working_tree_patch_sha256"),
                      ("environment.txt", "environment_digest")):
        if sha256_file(str(directory / name)) != recorded[key]:
            raise ValueError(f"{name} differs from its run-start digest")
    if ((directory / "working_tree.patch").stat().st_size == 0) != recorded["working_tree_patch_empty"]:
        raise ValueError("working_tree.patch emptiness differs from its run-start record")
    config = json.loads((output_dir / "config.resolved.json").read_text())
    if sha256_json(config) != recorded["resolved_config_sha256"]:
        raise ValueError("config.resolved.json differs from its run-start digest")
    inventory = json.loads((directory / "source_inventory.json").read_text())
    if not isinstance(inventory, list) or not inventory:
        raise ValueError("empty or malformed source inventory")
    expected = {}
    for item in inventory:
        name = item["path"]
        if Path(name).is_absolute() or ".." in Path(name).parts or name in expected:
            raise ValueError(f"invalid or duplicate source member: {name}")
        expected[name] = item["sha256"]
    if sha256_json(inventory) != recorded["source_inventory_sha256"]:
        raise ValueError("source inventory digest mismatch")
    observed = {}
    with tarfile.open(directory / "source.tar.gz", "r:gz") as archive:
        for member in archive:
            if not member.isfile() or member.name not in expected or member.name in observed:
                raise ValueError(f"unexpected source archive member: {member.name}")
            payload = archive.extractfile(member)
            if payload is None:
                raise ValueError(f"unreadable source member: {member.name}")
            observed[member.name] = sha256_bytes(payload.read())
    if observed != expected:
        raise ValueError("source archive does not match its inventory")
    if source_inventory(root) != inventory:
        raise ValueError("current source/config/protocol files differ from the retained snapshot")
