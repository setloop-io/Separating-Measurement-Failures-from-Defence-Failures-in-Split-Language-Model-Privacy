"""Deterministic, offline source and deployment bundle construction.

The builder has no cloud SDK and cannot create resources.  It accepts only
files that are either byte-identical to a named Git revision or explicitly
listed runtime artifacts, writes a deterministic tar archive, and emits
separate content-addressed manifests.  The manifests deliberately live beside
the archive rather than inside it, avoiding self-referential hashes.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile
from typing import Any, Iterable, Mapping
import uuid

from .cc_io import write_json_exclusive


SOURCE_MANIFEST_SCHEMA = "p4.cc_source_manifest.v1"
BUNDLE_MANIFEST_SCHEMA = "p4.cc_deployment_bundle_manifest.v1"
SHA256 = re.compile(r"^[0-9a-f]{64}$")
REVISION = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
RUNTIME_SOURCE_SCOPES = {
    "paper3": (
        "src/pcr",
        "scripts/check_cross_paper_evidence.py",
        "scripts/check_secure_compute_preorder.py",
        "experiments/schemas/cross_paper_secure_compute_evidence.schema.json",
    ),
    "paper4": (
        "src/p4d",
        "experiments/schemas",
        "scripts/build_cc_deployment_bundle.py",
        "scripts/build_cc_active_runtime.py",
        "scripts/advance_cc_bootstrap_lifecycle.py",
        "scripts/build_cc_bootstrap_lifecycle.py",
        "scripts/build_cc_lifecycle_launch.py",
        "scripts/consume_cc_launch.py",
        "scripts/capture_cc_runtime.py",
        "scripts/create_cc_pair_declaration.py",
        "scripts/run_cc_active_security.py",
        "scripts/run_cc_engineering_workload.py",
        "scripts/run_cc_lifecycle.py",
        "scripts/seal_cc_campaign.py",
        "scripts/validate_cc_arrival.py",
        "scripts/validate_cc_campaign.py",
        "scripts/validate_cc_lifecycle.py",
    ),
    "llm-attacker": ("attacker",),
}
REQUIRED_RUNTIME_ROLES = {
    "workload_config",
    "attack_plan_template",
    "evidence_envelope_schema",
    "checkpoint_manifest",
    "checkpoint_artifact",
    "resolved_config",
    "metrics",
    "panels_manifest",
    "panels_artifact",
    "uv_lock",
    "ttl_control",
    "deadman_control",
    "teardown_control",
    "budget_control",
    "collector_manifest",
    "collector_executable",
    "collector_public_key",
    "cc_on_journal_public_key",
    "cc_off_journal_public_key",
    "cc_on_heartbeat_public_key",
    "cc_off_heartbeat_public_key",
    "admission_verifier_adapter",
    "admission_verifier_public_key",
}


class DeploymentBundleError(ValueError):
    """A source or runtime binding is unsafe, missing, or inconsistent."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    state = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            state.update(chunk)
    return state.hexdigest()


def _safe_relative(root: Path, value: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise DeploymentBundleError("bundle paths must be nonempty POSIX-relative strings")
    relative = Path(value)
    if relative.is_absolute() or "." in relative.parts or ".." in relative.parts:
        raise DeploymentBundleError(f"unsafe bundle path: {value!r}")
    candidate = (root.resolve() / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise DeploymentBundleError(f"bundle path escapes repository: {value!r}") from exc
    return candidate


def _git_bytes(root: Path, revision: str, path: str) -> bytes:
    if not REVISION.fullmatch(revision):
        raise DeploymentBundleError("repository revision must be a full Git object ID")
    try:
        proc = subprocess.run(
            ["git", "show", f"{revision}:{path}"], cwd=root, check=False,
            capture_output=True, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise DeploymentBundleError(f"cannot resolve {revision}:{path}: {exc}") from exc
    if proc.returncode != 0:
        raise DeploymentBundleError(f"file is not present at frozen revision: {path}")
    return proc.stdout


def _git_paths(root: Path, revision: str, prefixes: Iterable[str]) -> list[str]:
    if not REVISION.fullmatch(revision):
        raise DeploymentBundleError("repository revision must be a full Git object ID")
    try:
        proc = subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", revision, "--", *prefixes],
            cwd=root, check=False, capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise DeploymentBundleError(f"cannot inventory frozen source: {exc}") from exc
    if proc.returncode != 0:
        raise DeploymentBundleError("cannot inventory frozen source revision")
    return sorted(line for line in proc.stdout.splitlines() if line)


def runtime_source_inventory(repositories: Mapping[str, Mapping[str, Any]]) \
        -> dict[str, list[str]]:
    """Return the complete executable/schema inventory at exact revisions."""
    if set(repositories) != set(RUNTIME_SOURCE_SCOPES):
        raise DeploymentBundleError(
            "runtime source requires exact paper3, paper4 and llm-attacker repositories")
    result: dict[str, list[str]] = {}
    for name, prefixes in RUNTIME_SOURCE_SCOPES.items():
        details = repositories[name]
        if set(details) != {"root", "revision"}:
            raise DeploymentBundleError(f"{name}: repository fields differ")
        root = Path(details["root"]).resolve()
        revision = details["revision"]
        paths = _git_paths(root, revision, prefixes)
        if not paths:
            raise DeploymentBundleError(f"{name}: runtime source inventory is empty")
        result[name] = paths
    return result


def _require_runtime_worktrees(repositories: Mapping[str, Mapping[str, Any]]) -> None:
    """Reject tracked drift or untracked executable/schema files at runtime."""
    for name, prefixes in RUNTIME_SOURCE_SCOPES.items():
        root = Path(repositories[name]["root"]).resolve()
        revision = repositories[name]["revision"]
        commands = (
            ["git", "diff", "--quiet", revision, "--", *prefixes],
            ["git", "ls-files", "--others", "--exclude-standard", "--", *prefixes],
        )
        try:
            changed = subprocess.run(
                commands[0], cwd=root, check=False, capture_output=True, timeout=30)
            untracked = subprocess.run(
                commands[1], cwd=root, check=False, capture_output=True, text=True,
                timeout=30)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise DeploymentBundleError(f"{name}: cannot inspect runtime worktree: {exc}") \
                from exc
        if changed.returncode != 0:
            raise DeploymentBundleError(
                f"{name}: runtime source bytes differ from the frozen revision")
        if untracked.returncode != 0 or untracked.stdout.strip():
            raise DeploymentBundleError(
                f"{name}: untracked runtime source/schema files are forbidden")


def build_runtime_source_manifest(
        repositories: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Build a manifest whose inventory cannot omit executable package bytes."""
    inventory = runtime_source_inventory(repositories)
    _require_runtime_worktrees(repositories)
    return build_source_manifest(repositories, inventory)


def validate_runtime_source_manifest(
        manifest: Mapping[str, Any],
        repositories: Mapping[str, Mapping[str, Any]]) -> Mapping[str, Any]:
    """Validate completeness plus current bytes before importing/executing work."""
    validate_source_manifest(manifest, repositories, verify_files=True)
    expected = runtime_source_inventory(repositories)
    actual = {name: [] for name in RUNTIME_SOURCE_SCOPES}
    for item in manifest["files"]:
        if item["repository"] not in actual:
            raise DeploymentBundleError("runtime source manifest repository differs")
        actual[item["repository"]].append(item["path"])
    actual = {name: sorted(paths) for name, paths in actual.items()}
    if actual != expected:
        raise DeploymentBundleError(
            "runtime source manifest omits or adds executable/schema files")
    _require_runtime_worktrees(repositories)
    return manifest


def build_source_manifest(repositories: Mapping[str, Mapping[str, Any]],
                          inventory: Mapping[str, Iterable[str]]) -> dict[str, Any]:
    """Bind current source bytes to exact committed bytes in each repository."""
    if set(repositories) != set(inventory) or not repositories:
        raise DeploymentBundleError("repository and source inventories must match")
    files: list[dict[str, Any]] = []
    revisions: dict[str, str] = {}
    archive_names: set[str] = set()
    for repository in sorted(repositories):
        details = repositories[repository]
        if set(details) != {"root", "revision"}:
            raise DeploymentBundleError(f"{repository}: repository fields differ")
        root = Path(details["root"]).resolve()
        revision = details["revision"]
        if not root.is_dir() or not isinstance(revision, str) or not REVISION.fullmatch(revision):
            raise DeploymentBundleError(f"{repository}: invalid root or revision")
        revisions[repository] = revision
        paths = list(inventory[repository])
        if not paths or len(paths) != len(set(paths)):
            raise DeploymentBundleError(f"{repository}: source inventory is empty or duplicated")
        for relative in sorted(paths):
            current = _safe_relative(root, relative)
            if not current.is_file() or current.is_symlink():
                raise DeploymentBundleError(f"{repository}:{relative}: regular file required")
            current_bytes = current.read_bytes()
            if current_bytes != _git_bytes(root, revision, relative):
                raise DeploymentBundleError(
                    f"{repository}:{relative}: current bytes differ from frozen revision")
            archive_path = f"source/{repository}/{relative}"
            if archive_path in archive_names:
                raise DeploymentBundleError("duplicate bundle archive path")
            archive_names.add(archive_path)
            files.append({
                "repository": repository,
                "path": relative,
                "archive_path": archive_path,
                "sha256": _sha256_bytes(current_bytes),
                "bytes": len(current_bytes),
            })
    body = {
        "schema": SOURCE_MANIFEST_SCHEMA,
        "repositories": revisions,
        "files": files,
    }
    body["source_tree_sha256"] = _sha256_bytes(_canonical_bytes(body))
    validate_source_manifest(body, repositories, verify_files=True)
    return body


def validate_source_manifest(manifest: Mapping[str, Any],
                             repositories: Mapping[str, Mapping[str, Any]] | None = None,
                             *, verify_files: bool = False) -> Mapping[str, Any]:
    if (not isinstance(manifest, Mapping)
            or set(manifest) != {"schema", "repositories", "files", "source_tree_sha256"}
            or manifest.get("schema") != SOURCE_MANIFEST_SCHEMA):
        raise DeploymentBundleError("source manifest fields/schema differ")
    revisions = manifest["repositories"]
    if not isinstance(revisions, Mapping) or not revisions:
        raise DeploymentBundleError("source manifest repositories are missing")
    if any(not isinstance(value, str) or not REVISION.fullmatch(value)
           for value in revisions.values()):
        raise DeploymentBundleError("source manifest revision is invalid")
    files = manifest["files"]
    if not isinstance(files, list) or not files:
        raise DeploymentBundleError("source manifest file inventory is missing")
    seen: set[tuple[str, str]] = set()
    archives: set[str] = set()
    for item in files:
        required = {"repository", "path", "archive_path", "sha256", "bytes"}
        if not isinstance(item, Mapping) or set(item) != required:
            raise DeploymentBundleError("source manifest file fields differ")
        key = (item["repository"], item["path"])
        if key in seen or item["archive_path"] in archives:
            raise DeploymentBundleError("source manifest contains duplicate paths")
        seen.add(key)
        archives.add(item["archive_path"])
        if item["repository"] not in revisions:
            raise DeploymentBundleError("source file names an unknown repository")
        if not SHA256.fullmatch(str(item["sha256"])) \
                or type(item["bytes"]) is not int or item["bytes"] < 1:
            raise DeploymentBundleError("source file digest/size is invalid")
        if verify_files:
            if repositories is None or item["repository"] not in repositories:
                raise DeploymentBundleError("repository root is unavailable for verification")
            root = Path(repositories[item["repository"]]["root"]).resolve()
            path = _safe_relative(root, item["path"])
            if (not path.is_file() or path.stat().st_size != item["bytes"]
                    or _sha256_file(path) != item["sha256"]):
                raise DeploymentBundleError(f"source file changed: {item['path']}")
    body = {key: value for key, value in manifest.items() if key != "source_tree_sha256"}
    if manifest["source_tree_sha256"] != _sha256_bytes(_canonical_bytes(body)):
        raise DeploymentBundleError("source tree digest mismatch")
    return manifest


def _runtime_records(runtime_files: Iterable[Mapping[str, Any]], *,
                     require_complete_runtime: bool) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    names: set[str] = set()
    roles: set[str] = set()
    for item in runtime_files:
        if not isinstance(item, Mapping) or set(item) != {"role", "path", "archive_path"}:
            raise DeploymentBundleError("runtime file fields differ")
        path = Path(item["path"]).resolve()
        archive_path = str(item["archive_path"])
        role = item["role"]
        if (not path.is_file() or path.is_symlink() or not archive_path
                or Path(archive_path).is_absolute() or ".." in Path(archive_path).parts):
            raise DeploymentBundleError("runtime file path is missing or unsafe")
        if (not isinstance(role, str) or not role
                or re.fullmatch(r"[a-z][a-z0-9_]*", role) is None
                or role in roles):
            raise DeploymentBundleError("runtime file roles must be unique identifiers")
        if archive_path in names:
            raise DeploymentBundleError("duplicate runtime archive path")
        names.add(archive_path)
        roles.add(role)
        records.append({
            "role": role,
            "source_path": str(path),
            "archive_path": archive_path,
            "sha256": _sha256_file(path),
            "bytes": path.stat().st_size,
        })
    if not records:
        raise DeploymentBundleError("runtime file inventory is empty")
    if require_complete_runtime and roles != REQUIRED_RUNTIME_ROLES:
        raise DeploymentBundleError(
            "runtime role inventory differs; missing "
            f"{sorted(REQUIRED_RUNTIME_ROLES - roles)}, extra "
            f"{sorted(roles - REQUIRED_RUNTIME_ROLES)}")
    return sorted(records, key=lambda value: value["archive_path"])


def _add_bytes(archive: tarfile.TarFile, archive_path: str, value: bytes) -> None:
    info = tarfile.TarInfo(archive_path)
    info.size = len(value)
    info.mode = 0o600
    info.uid = info.gid = 0
    info.uname = info.gname = "root"
    info.mtime = 0
    archive.addfile(info, io.BytesIO(value))


def build_deployment_bundle(source_manifest: Mapping[str, Any],
                            repositories: Mapping[str, Mapping[str, Any]],
                            runtime_files: Iterable[Mapping[str, Any]],
                            archive_path: str | Path,
                            manifest_path: str | Path, *,
                            require_complete_runtime: bool = True) -> tuple[Path, Path]:
    """Write a deterministic archive plus a non-self-referential byte manifest."""
    validate_source_manifest(source_manifest, repositories, verify_files=True)
    runtime = _runtime_records(
        runtime_files, require_complete_runtime=require_complete_runtime)
    archive_path = Path(archive_path)
    manifest_path = Path(manifest_path)
    if archive_path.exists() or manifest_path.exists():
        raise FileExistsError("deployment archive/manifest already exists")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive_path.parent / f".{archive_path.name}.{uuid.uuid4().hex}.pending"
    descriptor = None
    try:
        descriptor = os.open(temporary, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w+b", closefd=True) as stream:
            descriptor = None
            with tarfile.open(fileobj=stream, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for item in source_manifest["files"]:
                    root = Path(repositories[item["repository"]]["root"])
                    _add_bytes(
                        archive, item["archive_path"],
                        _safe_relative(root, item["path"]).read_bytes())
                for item in runtime:
                    _add_bytes(
                        archive, item["archive_path"],
                        Path(item["source_path"]).read_bytes())
                _add_bytes(
                    archive, "manifests/source-manifest.json",
                    json.dumps(source_manifest, indent=2, sort_keys=True).encode() + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, archive_path)
        directory = os.open(archive_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except FileExistsError as exc:
        raise FileExistsError("deployment archive/manifest already exists") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    manifest = {
        "schema": BUNDLE_MANIFEST_SCHEMA,
        "archive": {
            "path": archive_path.name,
            "sha256": _sha256_file(archive_path),
            "bytes": archive_path.stat().st_size,
            "format": "deterministic-pax-tar",
        },
        "source_manifest_sha256": _sha256_bytes(_canonical_bytes(source_manifest)),
        "source_tree_sha256": source_manifest["source_tree_sha256"],
        "runtime_files": runtime,
    }
    manifest["bundle_manifest_sha256"] = _sha256_bytes(_canonical_bytes(manifest))
    write_json_exclusive(manifest_path, manifest)
    validate_bundle_manifest(manifest, archive_path=archive_path)
    return archive_path, manifest_path


def validate_bundle_manifest(manifest: Mapping[str, Any], *,
                             archive_path: str | Path | None = None) -> Mapping[str, Any]:
    required = {"schema", "archive", "source_manifest_sha256", "source_tree_sha256",
                "runtime_files", "bundle_manifest_sha256"}
    if (not isinstance(manifest, Mapping) or set(manifest) != required
            or manifest.get("schema") != BUNDLE_MANIFEST_SCHEMA):
        raise DeploymentBundleError("deployment bundle manifest fields/schema differ")
    for key in ("source_manifest_sha256", "source_tree_sha256", "bundle_manifest_sha256"):
        if not SHA256.fullmatch(str(manifest[key])):
            raise DeploymentBundleError(f"invalid deployment bundle {key}")
    archive = manifest["archive"]
    if (not isinstance(archive, Mapping)
            or set(archive) != {"path", "sha256", "bytes", "format"}
            or archive["format"] != "deterministic-pax-tar"
            or not SHA256.fullmatch(str(archive["sha256"]))
            or type(archive["bytes"]) is not int or archive["bytes"] < 1):
        raise DeploymentBundleError("deployment archive binding differs")
    body = {key: value for key, value in manifest.items() if key != "bundle_manifest_sha256"}
    if manifest["bundle_manifest_sha256"] != _sha256_bytes(_canonical_bytes(body)):
        raise DeploymentBundleError("deployment bundle manifest digest mismatch")
    if archive_path is not None:
        path = Path(archive_path)
        if (not path.is_file() or path.name != archive["path"]
                or path.stat().st_size != archive["bytes"]
                or _sha256_file(path) != archive["sha256"]):
            raise DeploymentBundleError("deployment archive is missing or changed")
    return manifest
