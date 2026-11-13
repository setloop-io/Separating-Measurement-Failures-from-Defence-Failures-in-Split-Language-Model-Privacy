"""Build separate trusted-client and corpus-free remote-service packets."""

from __future__ import annotations

import gzip
import io
import json
import shutil
import tarfile
from pathlib import Path
from typing import Iterable

from . import provenance
from .utils import sha256_file, sha256_json

REMOTE_FILES = (
    "scripts/run_remote_service.py",
    "experiments/schemas/transcript.schema.json",
    "src/pcr/__init__.py", "src/pcr/data.py", "src/pcr/instrument.py",
    "src/pcr/model.py", "src/pcr/modes.py", "src/pcr/remote_service.py",
    "src/pcr/split.py", "src/pcr/train.py", "src/pcr/transcript.py",
    "src/pcr/utils.py", "src/pcr/wire.py",
)
PROHIBITED_REMOTE_PARTS = ("data", "outputs", "manifests", "checkpoint", "panel")


def _copy(root: Path, destination: Path, paths: Iterable[Path]) -> list[dict]:
    records = []
    for source in sorted(set(path.resolve() for path in paths)):
        relative = source.relative_to(root).as_posix()
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        records.append({"path": relative, "bytes": target.stat().st_size,
                        "sha256": sha256_file(str(target))})
    return records


def _archive(directory: Path, path: Path) -> None:
    with path.open("xb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as archive:
            for source in sorted(item for item in directory.rglob("*") if item.is_file()):
                relative = source.relative_to(directory).as_posix()
                payload = source.read_bytes()
                info = tarfile.TarInfo(relative)
                info.size = len(payload)
                info.mode = 0o644
                info.mtime = 0
                archive.addfile(info, io.BytesIO(payload))


def _manifest(packet: str, records: list[dict]) -> dict:
    value = {"schema_version": "r1-process-packet-1.0", "packet": packet,
             "files": records, "scientific_evidence": False}
    value["inventory_sha256"] = sha256_json(records)
    return value


def prepare_remote_service_packet(root: Path, destination: Path) -> dict:
    root, destination = root.resolve(), destination.resolve()
    if destination.exists():
        raise ValueError("remote packet destination already exists")
    remote_paths = [root / name for name in REMOTE_FILES]
    if any(not path.is_file() for path in remote_paths):
        raise ValueError("remote packet allow-list contains a missing file")
    destination.mkdir(parents=True)
    records = _copy(root, destination, remote_paths)
    if any(part in record["path"].lower().split("/") for record in records
           for part in PROHIBITED_REMOTE_PARTS):
        raise ValueError("remote packet contains prohibited trusted data or state")
    manifest = _manifest("remote-service", records)
    (destination / "packet.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    verify_process_packet(destination, "remote-service")
    return manifest


def prepare_process_packets(root: Path, destination: Path,
                            corpus_path: Path) -> dict:
    root, destination, corpus_path = root.resolve(), destination.resolve(), corpus_path.resolve()
    if destination.exists():
        raise ValueError("process packet destination already exists")
    if not corpus_path.is_file() or not corpus_path.is_relative_to(root):
        raise ValueError("trusted corpus is unavailable or outside the source root")
    destination.mkdir(parents=True)
    client_dir, remote_dir = destination / "trusted-client", destination / "remote-service"
    client_dir.mkdir()
    remote_dir.mkdir()
    client_paths = [*provenance.source_paths(root), corpus_path]
    remote_paths = [root / name for name in REMOTE_FILES]
    if any(not path.is_file() for path in remote_paths):
        raise ValueError("remote packet allow-list contains a missing file")
    client_records = _copy(root, client_dir, client_paths)
    remote_records = _copy(root, remote_dir, remote_paths)
    if any(part in record["path"].lower().split("/") for record in remote_records
           for part in PROHIBITED_REMOTE_PARTS):
        raise ValueError("remote packet contains prohibited trusted data or state")
    client_manifest = _manifest("trusted-client", client_records)
    remote_manifest = _manifest("remote-service", remote_records)
    (client_dir / "packet.json").write_text(json.dumps(client_manifest, indent=2, sort_keys=True) + "\n")
    (remote_dir / "packet.json").write_text(json.dumps(remote_manifest, indent=2, sort_keys=True) + "\n")
    _archive(client_dir, destination / "trusted-client.tar.gz")
    _archive(remote_dir, destination / "remote-service.tar.gz")
    report = {"schema_version": "r1-process-packets-1.0", "trusted_client": client_manifest,
              "remote_service": remote_manifest,
              "trusted_archive_sha256": sha256_file(str(destination / "trusted-client.tar.gz")),
              "remote_archive_sha256": sha256_file(str(destination / "remote-service.tar.gz")),
              "remote_contains_corpus": False, "scientific_evidence": False}
    (destination / "packet-set.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def verify_process_packet(directory: Path, expected_packet: str) -> dict:
    value = json.loads((directory / "packet.json").read_text())
    if value.get("schema_version") != "r1-process-packet-1.0" or value.get("packet") != expected_packet:
        raise ValueError("process packet identity differs")
    if value.get("inventory_sha256") != sha256_json(value.get("files", [])):
        raise ValueError("process packet inventory digest differs")
    seen = set()
    for record in value["files"]:
        path = directory / record["path"]
        if (record["path"] in seen or not path.is_file() or path.stat().st_size != record["bytes"] or
                sha256_file(str(path)) != record["sha256"]):
            raise ValueError("process packet file is missing, duplicated or changed")
        seen.add(record["path"])
    extras = {path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file()}
    if extras != seen | {"packet.json"}:
        raise ValueError("process packet contains unindexed files")
    return value
