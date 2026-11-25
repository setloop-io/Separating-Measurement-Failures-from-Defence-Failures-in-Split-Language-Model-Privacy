#!/usr/bin/env python3
"""Prepare the Hugging Face v1.0 root checksum manifest without fetching LFS.

The frozen Hub metadata must describe exactly the mirror's HEAD tree. Every
regular Git blob is read and hashed locally through one persistent ``cat-file``
process. For LFS files, the actual Git pointer's content SHA256 and byte count
must match the frozen Hub metadata; the immutable LFS content identity supplies
the manifest digest. This does NOT independently download or rehash LFS payloads.

The prepared overlay replaces or adds files, and only the explicitly obsolete
manuscript prefix and release-update file are deleted. Package-level manifests
remain unchanged. The root manifest excludes itself, root .gitattributes, and
recognisable OS/cache junk. No command in this script accesses a remote.

Run after the overlay is complete::

    python3 scripts/prepare_hf_v1_manifest.py
    python3 scripts/prepare_hf_v1_manifest.py --verify

``--verify`` recomputes the result and compares it with the existing manifest;
it writes no files. Defaults are relative to this repository, not the shell cwd.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from typing import BinaryIO


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RELEASE_ROOT = REPOSITORY_ROOT / "build/release-v1.0"
DELETED_PREFIXES = ("manuscript-20260930/",)
DELETED_FILES = frozenset({"RELEASE-UPDATE.md"})
CACHE_DIRECTORIES = frozenset(
    {"__MACOSX", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
)
OS_FILES = frozenset({".DS_Store", "Thumbs.db", "desktop.ini"})
SHA1 = re.compile(r"[0-9a-f]{40}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
CHUNK_SIZE = 1024 * 1024


def valid_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if (
        not name
        or path.is_absolute()
        or str(path) != name
        or ".." in path.parts
        or ".git" in path.parts
        or any(character in name for character in "\r\n\0\\")
    ):
        raise ValueError(f"Unsafe or noncanonical repository path: {name!r}")
    return path


def is_deleted(name: str) -> bool:
    return name in DELETED_FILES or name.startswith(DELETED_PREFIXES)


def exclusion_reason(name: str) -> str | None:
    path = valid_path(name)
    if name == "MANIFEST.sha256":
        return "root manifest excludes itself"
    if name == ".gitattributes":
        return "Git transport configuration"
    if (
        any(part in CACHE_DIRECTORIES for part in path.parts)
        or path.name in OS_FILES
        or path.name.startswith("._")
        or path.suffix in {".pyc", ".pyo"}
    ):
        return "OS/cache junk"
    return None


def sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        while chunk := source.read(CHUNK_SIZE):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def git(mirror: Path, *arguments: str) -> bytes:
    return subprocess.check_output(["git", "--git-dir", str(mirror), *arguments])


def read_exact(source: BinaryIO, size: int) -> bytes:
    result = bytearray()
    while len(result) < size:
        chunk = source.read(size - len(result))
        if not chunk:
            raise ValueError("Unexpected end of git cat-file output")
        result.extend(chunk)
    return bytes(result)


class GitBlobs:
    """Read and verify Git objects using one process, bounded memory per blob."""

    def __init__(self, mirror: Path):
        self.process = subprocess.Popen(
            ["git", "--git-dir", str(mirror), "cat-file", "--batch"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        assert self.process.stdin is not None and self.process.stdout is not None
        self.source = self.process.stdout
        self.requests = self.process.stdin

    def __enter__(self) -> GitBlobs:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.requests.close()
        self.source.close()
        if exc_type is not None:
            self.process.terminate()
        status = self.process.wait()
        if exc_type is None and status:
            raise ValueError(f"git cat-file failed with exit code {status}")

    def digest(self, object_id: str, expected_size: int, pointer: bool) -> tuple[str, bytes]:
        self.requests.write(object_id.encode("ascii") + b"\n")
        self.requests.flush()
        header = self.source.readline()
        expected_header = f"{object_id} blob {expected_size}\n".encode("ascii")
        if header != expected_header:
            raise ValueError(
                f"Unexpected git object header for {object_id}: {header!r}; "
                f"expected {expected_header!r}"
            )
        digest = hashlib.sha256()
        object_digest = hashlib.sha1(f"blob {expected_size}\0".encode("ascii"))
        pointer_content = bytearray()
        remaining = expected_size
        while remaining:
            chunk = read_exact(self.source, min(CHUNK_SIZE, remaining))
            digest.update(chunk)
            object_digest.update(chunk)
            if pointer:
                pointer_content.extend(chunk)
            remaining -= len(chunk)
        if read_exact(self.source, 1) != b"\n":
            raise ValueError(f"Missing Git object separator for {object_id}")
        if object_digest.hexdigest() != object_id:
            raise ValueError(f"Git object integrity mismatch for {object_id}")
        return digest.hexdigest(), bytes(pointer_content)


def validate_lfs_pointer(name: str, content: bytes, metadata: dict, logical_size: int) -> str:
    content_digest = metadata.get("sha256")
    if not isinstance(content_digest, str) or not SHA256.fullmatch(content_digest):
        raise ValueError(f"Invalid frozen LFS SHA256 for {name}")
    if metadata.get("size") != logical_size:
        raise ValueError(f"Frozen LFS size differs from file size for {name}")
    expected = (
        "version https://git-lfs.github.com/spec/v1\n"
        f"oid sha256:{content_digest}\n"
        f"size {logical_size}\n"
    ).encode("ascii")
    if content != expected or len(content) != metadata.get("pointer_size"):
        raise ValueError(f"LFS pointer does not exactly match frozen metadata for {name}")
    return content_digest


def atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as out:
            temporary = out.name
            out.write(content)
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def prepare(arguments: argparse.Namespace) -> dict:
    metadata_path = arguments.metadata.resolve()
    mirror = arguments.mirror.resolve()
    overlay = arguments.overlay.resolve()
    report_path = arguments.report.resolve()
    if not overlay.is_dir():
        raise ValueError(f"Prepared overlay does not exist: {overlay}")
    metadata_bytes = metadata_path.read_bytes()
    metadata = json.loads(metadata_bytes)
    source_head = metadata["sha"]
    if not isinstance(source_head, str) or not SHA1.fullmatch(source_head):
        raise ValueError("Frozen metadata does not contain a valid Git SHA1 commit")
    mirror_head = git(mirror, "rev-parse", "HEAD").decode("ascii").strip()
    if mirror_head != source_head:
        raise ValueError(f"Mirror HEAD {mirror_head} differs from frozen metadata {source_head}")

    frozen: dict[str, dict] = {}
    for item in metadata["files"]:
        name = item["path"]
        valid_path(name)
        if name in frozen:
            raise ValueError(f"Duplicate path in frozen metadata: {name}")
        if not SHA1.fullmatch(item["blob_id"]):
            raise ValueError(f"Invalid blob ID in frozen metadata for {name}")
        if not isinstance(item["size"], int) or item["size"] < 0:
            raise ValueError(f"Invalid file size in frozen metadata for {name}")
        frozen[name] = item

    tree_bytes = git(mirror, "ls-tree", "-r", "-z", "--full-tree", source_head)
    tree: dict[str, str] = {}
    for raw_entry in tree_bytes.split(b"\0"):
        if not raw_entry:
            continue
        raw_info, raw_name = raw_entry.split(b"\t", 1)
        mode, object_type, object_id = raw_info.decode("ascii").split(" ")
        name = raw_name.decode("utf-8")
        valid_path(name)
        if mode not in {"100644", "100755"} or object_type != "blob":
            raise ValueError(f"Unsupported Git tree entry {name}: {mode} {object_type}")
        tree[name] = object_id
    if tree.keys() != frozen.keys():
        raise ValueError(
            "Frozen metadata and mirror tree paths differ: "
            f"metadata_only={sorted(frozen.keys() - tree.keys())[:10]}, "
            f"mirror_only={sorted(tree.keys() - frozen.keys())[:10]}"
        )
    for name, object_id in tree.items():
        if object_id != frozen[name]["blob_id"]:
            raise ValueError(f"Frozen metadata and mirror blob IDs differ for {name}")

    records: dict[str, tuple[str, int, str]] = {}
    deleted: list[str] = []
    excluded: dict[str, str] = {}
    lfs_count = lfs_pointer_bytes = lfs_payload_bytes = 0
    regular_count = regular_bytes = 0
    with GitBlobs(mirror) as blobs:
        for number, name in enumerate(sorted(frozen), start=1):
            item = frozen[name]
            lfs = item.get("lfs")
            is_lfs = lfs is not None
            stored_size = lfs["pointer_size"] if is_lfs else item["size"]
            if not isinstance(stored_size, int) or stored_size < 0:
                raise ValueError(f"Invalid stored size for {name}")
            if is_lfs and stored_size > 4096:
                raise ValueError(f"Unexpectedly large LFS pointer for {name}")
            digest, content = blobs.digest(item["blob_id"], stored_size, is_lfs)
            if is_lfs:
                digest = validate_lfs_pointer(name, content, lfs, item["size"])
                lfs_count += 1
                lfs_pointer_bytes += stored_size
                lfs_payload_bytes += item["size"]
            else:
                regular_count += 1
                regular_bytes += stored_size
            if is_deleted(name):
                deleted.append(name)
            elif reason := exclusion_reason(name):
                excluded[name] = reason
            else:
                records[name] = (digest, item["size"], "frozen-lfs-identity" if is_lfs else "git-blob-sha256")
            if number % 2000 == 0:
                print(f"Verified {number:,}/{len(frozen):,} local Git blobs", file=sys.stderr)

    overlay_records: list[dict] = []
    overlay_excluded: dict[str, str] = {}
    for path in sorted(overlay.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlinks are not supported in the prepared overlay: {path}")
        if not path.is_file():
            continue
        name = path.relative_to(overlay).as_posix()
        valid_path(name)
        if is_deleted(name):
            raise ValueError(f"Overlay reintroduces an explicitly obsolete file: {name}")
        if reason := exclusion_reason(name):
            overlay_excluded[name] = reason
            continue
        digest, size = sha256_file(path)
        status = "replacement" if name in records else "addition"
        records[name] = (digest, size, "overlay-sha256")
        overlay_records.append({"path": name, "size": size, "sha256": digest, "status": status})

    manifest = "".join(f"{records[name][0]}  {name}\n" for name in sorted(records)).encode("utf-8")
    manifest_digest = hashlib.sha256(manifest).hexdigest()
    manifest_path = overlay / "MANIFEST.sha256"
    scope = {
        "method": "local Git object verification plus frozen immutable LFS content identities and hashed overlay bytes",
        "remote_network_access": False,
        "independent_full_artifact_download": False,
        "lfs_payload_bytes_downloaded": 0,
        "lfs_payload_bytes_independently_rehashed": 0,
        "limitations": (
            "LFS payload bytes were not independently downloaded or rehashed. Their SHA256 and size are taken "
            "from immutable content identifiers, after exact verification of each locally mirrored LFS pointer "
            "against the frozen Hub metadata. Regular Git blobs and overlay files are hashed from actual bytes."
        ),
    }
    report = {
        "schema_version": 1,
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "release": "v1.0",
        "source": {
            "repository": metadata.get("repo"),
            "metadata_path": str(metadata_path),
            "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
            "mirror_path": str(mirror),
            "head": source_head,
            "tree_sha1": git(mirror, "rev-parse", f"{source_head}^{{tree}}").decode("ascii").strip(),
            "ls_tree_sha256": hashlib.sha256(tree_bytes).hexdigest(),
            "tree_exactly_matches_metadata": True,
            "file_count": len(frozen),
            "logical_bytes": sum(item["size"] for item in frozen.values()),
        },
        "verification_scope": scope,
        "verified_git_content": {
            "blob_count": len(frozen),
            "all_blob_sha1_integrity_verified": True,
            "regular_blob_count": regular_count,
            "regular_blob_bytes_hashed": regular_bytes,
            "lfs_pointer_count": lfs_count,
            "lfs_pointer_bytes_verified": lfs_pointer_bytes,
            "lfs_payload_logical_bytes_represented": lfs_payload_bytes,
        },
        "deletions": {
            "prefixes": list(DELETED_PREFIXES),
            "files": sorted(DELETED_FILES),
            "matched_file_count": len(deleted),
            "matched_paths": deleted,
        },
        "manifest_exclusions": {
            "root_only": [".gitattributes", "MANIFEST.sha256"],
            "cache_directories": sorted(CACHE_DIRECTORIES),
            "os_filenames": sorted(OS_FILES),
            "other_patterns": ["._*", "*.pyc", "*.pyo"],
            "initial_paths": excluded,
            "overlay_paths": overlay_excluded,
            "package_manifests_preserved": True,
        },
        "overlay": {
            "path": str(overlay),
            "hashed_file_count": len(overlay_records),
            "hashed_bytes": sum(record["size"] for record in overlay_records),
            "files": overlay_records,
        },
        "manifest": {
            "path": str(manifest_path),
            "sha256": manifest_digest,
            "byte_count": len(manifest),
            "covered_file_count": len(records),
            "covered_logical_bytes": sum(record[1] for record in records.values()),
            "unchanged_lfs_identity_count": sum(record[2] == "frozen-lfs-identity" for record in records.values()),
            "unchanged_regular_blob_count": sum(record[2] == "git-blob-sha256" for record in records.values()),
            "overlay_hashed_file_count": len(overlay_records),
        },
    }
    if arguments.verify:
        if not manifest_path.is_file() or manifest_path.read_bytes() != manifest:
            raise ValueError(f"Existing root manifest does not match the derived final tree: {manifest_path}")
        if report_path.is_file():
            previous = json.loads(report_path.read_bytes())
            for key in ("source", "verified_git_content", "deletions", "overlay", "manifest"):
                if previous.get(key) != report[key]:
                    raise ValueError(f"Preparation report no longer matches current inputs: {key}")
    else:
        atomic_write(manifest_path, manifest)
        atomic_write(report_path, (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return {
        "mode": "verified" if arguments.verify else "prepared",
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_digest,
        "covered_file_count": len(records),
        "report": str(report_path),
        "lfs_payloads_independently_rehashed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metadata", type=Path, default=RELEASE_ROOT / "backups/hf-initial-state.json")
    parser.add_argument("--mirror", type=Path, default=RELEASE_ROOT / "backups/huggingface.git")
    parser.add_argument("--overlay", type=Path, default=RELEASE_ROOT / "prepared/huggingface")
    parser.add_argument("--report", type=Path, default=RELEASE_ROOT / "manifest-preparation.json")
    parser.add_argument("--verify", action="store_true", help="Recompute and compare without writing files")
    arguments = parser.parse_args()
    try:
        result = prepare(arguments)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print(f"Manifest preparation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
