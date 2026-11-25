#!/usr/bin/env python3
"""Prepare byte-pinned native-replication data inputs, without model execution.

Only Python's standard library is imported. Parquet bytes are never decoded or
rewritten here. A loader must still validate actual columns, row counts and its
native preprocessing; metadata declarations are not a substitute for that check.
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
import urllib.parse
import urllib.request
from typing import Any


LAMBADA_REPOSITORY = "cimec/lambada"
LAMBADA_REVISION = "5953bd97664b64b95754f299b2309ecfbfbe81b9"
CODEALPACA_REPOSITORY = "HuggingFaceH4/CodeAlpaca_20K"
CODEALPACA_REVISION = "798c567f69c8f4b12fc191015e59ee34e9afe00d"
CODEALPACA_ORIGIN_REPOSITORY = "sahil2801/CodeAlpaca-20k"
CODEALPACA_ORIGIN_REVISION = "152bb5e9a29651266b018106053980070a0521a1"
CODELLAMA_REPOSITORY = "codellama/CodeLlama-7b-hf"
CODELLAMA_REVISION = "6c284d1468fe6c413cf56183e69b194dcfa27fe6"
CODELLAMA_FILES = (
    "LICENSE", "README.md", "USE_POLICY.md", "config.json", "generation_config.json",
    "model.safetensors.index.json", "special_tokens_map.json", "tokenizer_config.json",
    "tokenizer.json", "tokenizer.model", "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _digest(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("LFS metadata must contain a lowercase SHA-256 digest")
    return value


def validate_lfs_entry(entry: dict[str, Any]) -> dict[str, Any]:
    path = entry.get("path")
    if not isinstance(path, str) or not path:
        raise ValueError("Missing tree file path")
    posix = PurePosixPath(path)
    if posix.is_absolute() or any(x in ("..", ".") for x in posix.parts) or "\\" in path:
        raise ValueError("Unsafe tree file path")
    if entry.get("type") != "file" or not path.endswith(".parquet"):
        raise ValueError("Expected a Parquet file tree entry")
    lfs = entry.get("lfs")
    if not isinstance(lfs, dict):
        raise ValueError("Parquet file lacks LFS SHA-256 metadata")
    size = lfs.get("size")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0 or size != entry.get("size"):
        raise ValueError("Tree/LFS sizes are invalid or disagree")
    return {"source_path": path, "sha256": _digest(lfs.get("oid")), "bytes": size,
            "git_pointer_blob_sha1": entry.get("oid")}


def fetch_verified(repository: str, revision: str, entry: dict[str, Any], target: Path,
                   *, offline: bool = False, timeout: float = 60) -> dict[str, Any]:
    """Stream immutable HF resolve bytes and require tree-attested size/SHA-256.

    Existing files are rehashed before reuse. A bad existing file is not silently
    overwritten; a failed download never becomes the final named input.
    """
    info = validate_lfs_entry(entry)
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("An exact 40-hex repository revision is required")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("Invalid Hugging Face repository identifier")
    target = target.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    url = ("https://huggingface.co/datasets/" + repository + "/resolve/" + revision + "/"
           + urllib.parse.quote(info["source_path"], safe="/"))
    if target.exists():
        if target.stat().st_size != info["bytes"] or sha256_file(target) != info["sha256"]:
            raise ValueError(f"Existing input fails pinned size/hash validation: {target}")
        mode = "REUSED_VERIFIED_EXISTING"
        final_url = None
    else:
        if offline:
            raise FileNotFoundError(f"Offline input unavailable: {target}")
        part_path = None
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "LLM-Attack-Suite-input-preparation/1"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                final_url = response.geturl()
                digest = hashlib.sha256()
                count = 0
                with tempfile.NamedTemporaryFile(dir=target.parent, prefix=target.name + ".part-", delete=False) as output:
                    part_path = Path(output.name)
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        count += len(block)
                        if count > info["bytes"]:
                            raise ValueError(f"Response exceeds pinned byte size: {url}")
                        digest.update(block)
                        output.write(block)
                    output.flush()
                    os.fsync(output.fileno())
            if count != info["bytes"] or digest.hexdigest() != info["sha256"]:
                raise ValueError(f"Downloaded input fails pinned size/hash validation: {url}")
            os.replace(part_path, target)
            part_path = None
        finally:
            if part_path is not None:
                part_path.unlink(missing_ok=True)
        mode = "DOWNLOADED_AND_VERIFIED"
    return dict(info, path=str(target), requested_url=url, resolved_url=final_url, action=mode)


def _prepare(repository: str, revision: str, metadata_path: Path, tree_path: Path,
             output_dir: Path, expected_prefix: str, expected_shards: dict[str, int],
             *, offline: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata = _load_json(metadata_path)
    if metadata.get("id") != repository or metadata.get("sha") != revision:
        raise ValueError("Dataset metadata does not match the required repository/revision")
    tree = _load_json(tree_path)
    if not isinstance(tree, list):
        raise ValueError("Expected a retained Hugging Face recursive tree JSON list")
    # The parent-retained tree receipt must separately prove the request was made
    # at this revision. Individual files may have older lastCommit identifiers.
    selected: dict[str, list[dict[str, Any]]] = {k: [] for k in expected_shards}
    pattern = re.compile(re.escape(expected_prefix) + r"/(train|test|validation)-(\d{5})-of-(\d{5})\.parquet\Z")
    for entry in tree:
        if not isinstance(entry, dict):
            raise ValueError("Invalid tree record")
        match = pattern.fullmatch(str(entry.get("path", "")))
        if match and match.group(1) in selected:
            selected[match.group(1)].append(entry)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    manifest_splits = {}
    for split, expected in expected_shards.items():
        entries = sorted(selected[split], key=lambda x: x["path"])
        names = [PurePosixPath(x["path"]).name for x in entries]
        expected_names = [f"{split}-{i:05d}-of-{expected:05d}.parquet" for i in range(expected)]
        if names != expected_names:
            raise ValueError(f"Incomplete or duplicated {split} shard set: {names}")
        files[split] = [fetch_verified(repository, revision, x, output_dir / PurePosixPath(x["path"]).name,
                                       offline=offline) for x in entries]
        manifest_splits[split] = [{"path": x["path"], "sha256": x["sha256"]} for x in files[split]]
    manifest = {"repository": repository, "revision": revision, "splits": manifest_splits}
    receipt = {
        "schema_version": "native-dataset-input-preparation.v1", "repository": repository,
        "revision": revision, "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "metadata_path": str(metadata_path.resolve()), "metadata_sha256": sha256_file(metadata_path),
        "tree_path": str(tree_path.resolve()), "tree_sha256": sha256_file(tree_path),
        "preparer_sha256": sha256_file(Path(__file__)), "files": files,
        "content_transformation": "NONE; original pinned Parquet bytes retained",
        "split_transformation": "NONE; declared source splits and shard order retained",
        "columns_or_row_counts_decoded_by_this_stdlib_module": False,
        "tree_revision_proof_requirement": "Retain official request URL/receipt for the exact revision, not only this metadata SHA field",
    }
    return manifest, receipt


def prepare_lambada(metadata_path: Path, tree_path: Path, output_dir: Path,
                    *, include_validation: bool = False, offline: bool = False) -> dict[str, Any]:
    metadata = _load_json(metadata_path)
    declared = metadata.get("cardData", {}).get("dataset_info", {})
    features = declared.get("features")
    if features != [{"name": "text", "dtype": "string"}, {"name": "domain", "dtype": "string"}]:
        raise ValueError("LAMBADA metadata has unexpected declared text/domain columns")
    expected = {"train": 2, "test": 1}
    if include_validation:
        expected["validation"] = 1
    manifest, receipt = _prepare(LAMBADA_REPOSITORY, LAMBADA_REVISION, metadata_path, tree_path,
                                  output_dir, "plain_text", expected, offline=offline)
    receipt["declared_features"] = features
    receipt["declared_splits"] = declared.get("splits")
    receipt["native_loader_contract"] = (
        "load_dataset('parquet', data_files=manifest split file lists); validate actual columns/counts, "
        "then leave native prompt/window/tokenization logic unchanged. Full train/test rows are retained."
    )
    _write_outputs(output_dir, manifest, receipt)
    return {"manifest": manifest, "receipt": receipt}


def prepare_codealpaca_parquet(metadata_path: Path, tree_path: Path, output_dir: Path,
                               *, offline: bool = False) -> dict[str, Any]:
    manifest, receipt = _prepare(CODEALPACA_REPOSITORY, CODEALPACA_REVISION, metadata_path, tree_path,
                                  output_dir, "data", {"train": 1, "test": 1}, offline=offline)
    receipt["native_json_conversion_status"] = (
        "NOT_CONVERTED. Native LoadDataset.py expects one JSON list of instruction/input/output and "
        "then a seed-dependent 90/10 split. HF train/test are not silently treated as that native split."
    )
    _write_outputs(output_dir, manifest, receipt)
    return {"manifest": manifest, "receipt": receipt}


def git_blob_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_verified_git_blob(repository: str, revision: str, entry: dict[str, Any], target: Path,
                            *, offline: bool = False, timeout: float = 60) -> dict[str, Any]:
    """Download a non-LFS JSON file and verify the tree's exact Git blob ID."""
    source_path, blob, size = entry.get("path"), entry.get("oid"), entry.get("size")
    if (entry.get("type") != "file" or source_path != "code_alpaca_20k.json"
            or "lfs" in entry or not isinstance(blob, str) or not re.fullmatch(r"[0-9a-f]{40}", blob)
            or isinstance(size, bool) or not isinstance(size, int) or size <= 0):
        raise ValueError("Unexpected original CodeAlpaca JSON tree entry")
    if repository != CODEALPACA_ORIGIN_REPOSITORY or revision != CODEALPACA_ORIGIN_REVISION:
        raise ValueError("Unpinned original CodeAlpaca source")
    target = target.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/datasets/{repository}/resolve/{revision}/{source_path}"
    final_url = None
    if target.exists():
        if target.stat().st_size != size or git_blob_sha1(target) != blob:
            raise ValueError("Existing original JSON fails pinned Git blob validation")
        action = "REUSED_VERIFIED_EXISTING"
    else:
        if offline:
            raise FileNotFoundError(f"Offline original JSON unavailable: {target}")
        part_path = None
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "LLM-Attack-Suite-input-preparation/1"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                final_url = response.geturl()
                count = 0
                digest = hashlib.sha1(f"blob {size}\0".encode())
                with tempfile.NamedTemporaryFile(dir=target.parent, prefix=target.name + ".part-", delete=False) as output:
                    part_path = Path(output.name)
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        count += len(block)
                        if count > size:
                            raise ValueError("Original JSON response exceeds pinned size")
                        output.write(block)
                        digest.update(block)
                    output.flush()
                    os.fsync(output.fileno())
            if count != size or digest.hexdigest() != blob:
                raise ValueError("Downloaded original JSON fails pinned Git blob validation")
            os.replace(part_path, target)
            part_path = None
        finally:
            if part_path is not None:
                part_path.unlink(missing_ok=True)
        action = "DOWNLOADED_AND_VERIFIED"
    return {"path": str(target), "source_path": source_path, "bytes": size,
            "git_blob_sha1": blob, "sha256": sha256_file(target), "action": action,
            "requested_url": url, "resolved_url": final_url}


def prepare_codealpaca_original(metadata_path: Path, tree_path: Path, h4_script_path: Path,
                                h4_tree_path: Path, output_dir: Path,
                                *, offline: bool = False) -> dict[str, Any]:
    """Retain the original native-schema JSON, never reverse lossy H4 prompts.

    The H4 pinned creation script is inspected as data, not executed. Its Git
    blob and source-repository reference establish the documented provenance
    chain; they do not establish an upstream experiment's unspecified data pin.
    """
    metadata, tree = _load_json(metadata_path), _load_json(tree_path)
    if metadata.get("id") != CODEALPACA_ORIGIN_REPOSITORY or metadata.get("sha") != CODEALPACA_ORIGIN_REVISION:
        raise ValueError("Original dataset metadata does not match the required pin")
    if not isinstance(tree, list):
        raise ValueError("Expected an original dataset tree list")
    matches = [x for x in tree if x.get("path") == "code_alpaca_20k.json"]
    if len(matches) != 1:
        raise ValueError("Original tree must contain exactly one native JSON file")
    h4_tree = _load_json(h4_tree_path)
    if not isinstance(h4_tree, list):
        raise ValueError("Expected a pinned H4 tree list")
    script_entries = [x for x in h4_tree if x.get("path") == "create_dataset.py"]
    if len(script_entries) != 1 or git_blob_sha1(h4_script_path) != script_entries[0].get("oid"):
        raise ValueError("H4 creation script does not match the retained pinned tree")
    syntax = ast.parse(h4_script_path.read_text())
    origins = [call.args[0].value for call in ast.walk(syntax)
               if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
               and call.func.id == "load_dataset" and call.args
               and isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str)]
    if origins != [CODEALPACA_ORIGIN_REPOSITORY]:
        raise ValueError("Pinned H4 creation script has an unexpected source repository")
    output_dir = output_dir.resolve()
    file_receipt = fetch_verified_git_blob(CODEALPACA_ORIGIN_REPOSITORY, CODEALPACA_ORIGIN_REVISION,
                                           matches[0], output_dir / "code_alpaca_20k.json", offline=offline)
    rows = _load_json(Path(file_receipt["path"]))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Native original JSON must be a nonempty record list")
    for row_id, row in enumerate(rows):
        if not isinstance(row, dict) or any(not isinstance(row.get(k), str) for k in ("instruction", "input", "output")):
            raise ValueError(f"Original JSON record {row_id} lacks native string fields")
    manifest = {"repository": CODEALPACA_ORIGIN_REPOSITORY, "revision": CODEALPACA_ORIGIN_REVISION,
                "native_json_path": file_receipt["path"], "sha256": file_receipt["sha256"],
                "git_blob_sha1": file_receipt["git_blob_sha1"], "records": len(rows),
                "native_required_columns": ["instruction", "input", "output"],
                "native_split": "Native LoadDataset.py train_test_split(test_size=0.1, random_state=current_seed); no split made in preparation"}
    receipt = {
        "schema_version": "native-codealpaca-original-input-preparation.v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "repository": CODEALPACA_ORIGIN_REPOSITORY, "revision": CODEALPACA_ORIGIN_REVISION,
        "metadata_path": str(metadata_path.resolve()), "metadata_sha256": sha256_file(metadata_path),
        "tree_path": str(tree_path.resolve()), "tree_sha256": sha256_file(tree_path),
        "h4_repository": CODEALPACA_REPOSITORY, "h4_revision": CODEALPACA_REVISION,
        "h4_script_path": str(h4_script_path.resolve()), "h4_script_sha256": sha256_file(h4_script_path),
        "h4_script_git_blob_sha1": git_blob_sha1(h4_script_path),
        "h4_tree_path": str(h4_tree_path.resolve()), "h4_tree_sha256": sha256_file(h4_tree_path),
        "preparer_sha256": sha256_file(Path(__file__)), "file": file_receipt,
        "records": len(rows), "native_string_columns_validated": ["instruction", "input", "output"],
        "content_transformation": "NONE; original JSON bytes and record order retained, no stripping, flattening or reconstruction",
        "split_transformation": "NONE; original unsplit records retained for the native seed-dependent 90/10 split",
        "source_choice": (
            "The native guide links H4, whose pinned creation script loads this original repository before "
            "lossy prompt flattening and a seed-42 split. The original native-schema JSON is selected explicitly; "
            "this is not a claim that upstream paper authors used this exact unspecified data revision."
        ),
        "tree_revision_proof_requirement": "Retain official request URLs/receipts for both exact repository revisions",
    }
    _write_outputs(output_dir, manifest, receipt)
    return {"manifest": manifest, "receipt": receipt}


def codellama_download_plan(tree_path: Path) -> list[dict[str, Any]]:
    """Select exactly the declared safetensors/tokenizer/config/licence assets."""
    tree = _load_json(tree_path)
    if not isinstance(tree, list):
        raise ValueError("Expected a retained CodeLlama model tree list")
    selected = []
    for name in CODELLAMA_FILES:
        matches = [x for x in tree if x.get("path") == name]
        if len(matches) != 1:
            raise ValueError(f"Missing or duplicate required CodeLlama file: {name}")
        entry = matches[0]
        size, oid = entry.get("size"), entry.get("oid")
        if (entry.get("type") != "file" or isinstance(size, bool) or not isinstance(size, int)
                or size <= 0 or not isinstance(oid, str) or not re.fullmatch(r"[0-9a-f]{40}", oid)):
            raise ValueError(f"Invalid model tree metadata: {name}")
        if "lfs" in entry:
            lfs = entry["lfs"]
            if not isinstance(lfs, dict) or lfs.get("size") != size:
                raise ValueError(f"Model tree/LFS size mismatch: {name}")
            _digest(lfs.get("oid"))
        if name.endswith(".safetensors") and "lfs" not in entry:
            raise ValueError("Safetensors shard lacks LFS SHA-256")
        selected.append(entry)
    return selected


def _fetch_model_file(entry: dict[str, Any], output_dir: Path,
                      *, offline: bool = False, timeout: float = 60) -> dict[str, Any]:
    name, size = entry["path"], entry["size"]
    if name not in CODELLAMA_FILES:
        raise ValueError("Model file is outside the explicit non-pickle allowlist")
    target = (output_dir / name).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    lfs = entry.get("lfs")
    expected_sha256 = lfs["oid"] if lfs else None

    def valid(path: Path) -> bool:
        return (path.stat().st_size == size
                and (sha256_file(path) == expected_sha256 if lfs else git_blob_sha1(path) == entry["oid"]))

    url = f"https://huggingface.co/{CODELLAMA_REPOSITORY}/resolve/{CODELLAMA_REVISION}/{name}"
    final_url = None
    if target.exists():
        if not valid(target):
            raise ValueError(f"Existing CodeLlama asset fails pinned validation: {target}")
        action = "REUSED_VERIFIED_EXISTING"
    else:
        if offline:
            raise FileNotFoundError(f"Offline CodeLlama asset unavailable: {target}")
        part_path = None
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "LLM-Attack-Suite-input-preparation/1"})
            digest = hashlib.sha256() if lfs else hashlib.sha1(f"blob {size}\0".encode())
            count = 0
            with urllib.request.urlopen(request, timeout=timeout) as response:
                final_url = response.geturl()
                with tempfile.NamedTemporaryFile(dir=target.parent, prefix=target.name + ".part-", delete=False) as output:
                    part_path = Path(output.name)
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        count += len(block)
                        if count > size:
                            raise ValueError(f"CodeLlama response exceeds pinned size: {name}")
                        digest.update(block)
                        output.write(block)
                    output.flush()
                    os.fsync(output.fileno())
            expected = expected_sha256 if lfs else entry["oid"]
            if count != size or digest.hexdigest() != expected:
                raise ValueError(f"Downloaded CodeLlama asset fails pinned validation: {name}")
            os.replace(part_path, target)
            part_path = None
        finally:
            if part_path is not None:
                part_path.unlink(missing_ok=True)
        action = "DOWNLOADED_AND_VERIFIED"
    return {"path": str(target), "source_path": name, "bytes": size,
            "sha256": expected_sha256 if lfs else sha256_file(target),
            "git_blob_sha1": None if lfs else entry["oid"],
            "git_lfs_pointer_blob_sha1": entry["oid"] if lfs else None,
            "lfs_sha256_verified": bool(lfs), "action": action,
            "requested_url": url, "resolved_url": final_url}


def prepare_codellama(tree_path: Path, output_dir: Path, *, offline: bool = False) -> dict[str, Any]:
    """Download pinned safetensors assets only; never import or instantiate a model."""
    plan = codellama_download_plan(tree_path)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    files = [_fetch_model_file(entry, output_dir, offline=offline) for entry in plan]
    index = _load_json(output_dir / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("CodeLlama safetensors index has no weight map")
    shards = {"model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"}
    if set(weight_map.values()) != shards:
        raise ValueError("CodeLlama safetensors index references undeclared or missing shards")
    manifest = {"repository": CODELLAMA_REPOSITORY, "revision": CODELLAMA_REVISION,
                "model_path": str(output_dir), "files": files, "weight_format": "safetensors",
                "weight_shards": sorted(shards), "pickle_bin_files_downloaded": False}
    receipt = {
        "schema_version": "pinned-codellama-asset-preparation.v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "repository": CODELLAMA_REPOSITORY, "revision": CODELLAMA_REVISION,
        "tree_path": str(tree_path.resolve()), "tree_sha256": sha256_file(tree_path),
        "preparer_sha256": sha256_file(Path(__file__)), "files": files,
        "download_bytes": sum(x["bytes"] for x in files),
        "content_transformation": "NONE; original pinned assets retained",
        "model_import_training_inference": False,
        "tree_revision_proof_requirement": "Retain official tree request URL/receipt for this exact model revision",
    }
    _write_outputs(output_dir, manifest, receipt)
    return {"manifest": manifest, "receipt": receipt}


def _write_outputs(output_dir: Path, manifest: dict[str, Any], receipt: dict[str, Any]) -> None:
    output_dir = output_dir.resolve()
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    receipt["manifest_path"] = str(manifest_path)
    receipt["manifest_sha256"] = sha256_file(manifest_path)
    (output_dir / "input-preparation-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["lambada", "codealpaca-parquet", "codealpaca-native", "codellama"])
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--include-validation", action="store_true")
    parser.add_argument("--h4-script", type=Path)
    parser.add_argument("--h4-tree", type=Path)
    args = parser.parse_args()
    if args.dataset != "codellama" and args.metadata is None:
        parser.error("--metadata is required for dataset preparation")
    if args.dataset == "lambada":
        result = prepare_lambada(args.metadata, args.tree, args.output_dir,
                                 include_validation=args.include_validation, offline=args.offline)
    elif args.dataset == "codealpaca-parquet":
        if args.include_validation:
            parser.error("CodeAlpaca has no declared validation split")
        result = prepare_codealpaca_parquet(args.metadata, args.tree, args.output_dir, offline=args.offline)
    elif args.dataset == "codealpaca-native":
        if args.include_validation or args.h4_script is None or args.h4_tree is None:
            parser.error("codealpaca-native requires --h4-script/--h4-tree and no --include-validation")
        result = prepare_codealpaca_original(args.metadata, args.tree, args.h4_script, args.h4_tree,
                                             args.output_dir, offline=args.offline)
    else:
        if args.include_validation:
            parser.error("Model preparation has no validation split")
        result = prepare_codellama(args.tree, args.output_dir, offline=args.offline)
    print(json.dumps({"dataset": args.dataset, "manifest_path": result["receipt"]["manifest_path"],
                      "manifest_sha256": result["receipt"]["manifest_sha256"],
                      "split_shards": {k: len(v) for k, v in result["manifest"].get("splits", {}).items()},
                      "native_json_records": result["manifest"].get("records")}, sort_keys=True))


if __name__ == "__main__":
    main()
