#!/usr/bin/env python3
"""Build the immutable, offline CC deployment archive; never provisions hardware."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from p4d.cc_deployment import (
    DeploymentBundleError,
    build_deployment_bundle,
    build_runtime_source_manifest,
)
from p4d.cc_io import read_json_snapshot, write_json_exclusive


INVENTORY_SCHEMA = "p4.cc_runtime_inventory.v1"


def _sha256(path: Path) -> str:
    state = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            state.update(chunk)
    return state.hexdigest()


def _runtime_files(path: Path) -> list[dict[str, str]]:
    value, _ = read_json_snapshot(path)
    if set(value) != {"schema", "files"} or value.get("schema") != INVENTORY_SCHEMA:
        raise DeploymentBundleError(f"expected runtime inventory {INVENTORY_SCHEMA}")
    files = value.get("files")
    if not isinstance(files, list) or not files:
        raise DeploymentBundleError("runtime inventory must contain files")
    result = []
    for index, item in enumerate(files):
        if not isinstance(item, dict) or set(item) != {"role", "path", "archive_path"}:
            raise DeploymentBundleError(
                f"runtime inventory item {index} fields differ")
        source = Path(item["path"])
        if not source.is_absolute():
            source = path.parent / source
        result.append({
            "role": item["role"],
            "path": str(source.resolve()),
            "archive_path": item["archive_path"],
        })
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper3-root", required=True)
    parser.add_argument("--paper3-revision", required=True)
    parser.add_argument("--paper4-root", default=".")
    parser.add_argument("--paper4-revision", required=True)
    parser.add_argument("--llm-attacker-root", required=True)
    parser.add_argument("--llm-attacker-revision", required=True)
    parser.add_argument("--runtime-inventory", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    repositories = {
        "paper3": {
            "root": Path(args.paper3_root).resolve(),
            "revision": args.paper3_revision,
        },
        "paper4": {
            "root": Path(args.paper4_root).resolve(),
            "revision": args.paper4_revision,
        },
        "llm-attacker": {
            "root": Path(args.llm_attacker_root).resolve(),
            "revision": args.llm_attacker_revision,
        },
    }
    inventory_path = Path(args.runtime_inventory).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_path = output / "source-manifest.json"
    archive_path = output / "deployment-bundle.tar"
    bundle_path = output / "deployment-bundle-manifest.json"
    try:
        source = build_runtime_source_manifest(repositories)
        write_json_exclusive(source_path, source)
        build_deployment_bundle(
            source, repositories, _runtime_files(inventory_path),
            archive_path, bundle_path,
        )
    except (DeploymentBundleError, FileExistsError, OSError, ValueError) as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 1
    report = {
        "source_manifest": str(source_path),
        "source_manifest_sha256": _sha256(source_path),
        "source_tree_sha256": source["source_tree_sha256"],
        "deployment_archive": str(archive_path),
        "deployment_archive_sha256": _sha256(archive_path),
        "deployment_bundle_manifest": str(bundle_path),
        "deployment_bundle_manifest_sha256": _sha256(bundle_path),
        "hardware_provisioned": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
