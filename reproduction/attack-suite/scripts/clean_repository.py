#!/usr/bin/env python3
"""Remove only untracked, regenerable local build intermediates.

The default is a dry run. Historical experiments, manuscript sources, PDFs,
bibliographies, validation records and the virtual environment are preserved.
Use --apply to remove the listed candidates and --report PATH to retain the
per-file inventory. No Git history or tracked file is changed.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
NEVER_TRAVERSE = {".git", ".venv", ".agents", ".codex", ".aws"}
PYTHON_EXCLUDED = {
    "results", "build", "output", "outputs", "tmp", "docs", "paper",
}
TEX_SUFFIXES = {".aux", ".blg", ".log", ".out", ".fls", ".fdb_latexmk"}
HISTORICAL_PAPER_DIRS = {
    "validation", "arxiv-build", "archive", "archives", "releases", "dist",
}


def tracked_paths() -> set[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True,
    )
    return set(result.stdout.decode("utf-8", errors="surrogateescape").split("\0")) - {""}


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def manuscript_roots() -> set[Path]:
    paper = ROOT / "paper"
    # Keep the two established manuscript roots eligible after an older source
    # has been moved into the verified manuscript archive.
    roots = {paper, paper / "paper_a", paper / "paper_b"}
    for current, directories, filenames in os.walk(paper, followlinks=False):
        directory = Path(current)
        directories[:] = [
            name for name in directories
            if name not in NEVER_TRAVERSE | HISTORICAL_PAPER_DIRS
            and not (directory / name).is_symlink()
        ]
        if "main.tex" in filenames:
            roots.add(directory)
    return roots


def package_roots() -> set[Path]:
    # These are setuptools' known intermediate directories, inspected before
    # introducing this cleaner. All other build/ trees contain retained evidence.
    roots = {ROOT / "build" / "lib", ROOT / "llm_attacker.egg-info"}
    roots.update((ROOT / "build").glob("bdist.*"))
    return {path for path in roots if path.is_dir() and not path.is_symlink()}


def inventory() -> tuple[list[dict], set[Path], list[str]]:
    tracked = tracked_paths()
    tex_roots = manuscript_roots()
    packages = package_roots()
    files: list[dict] = []
    removable_directories: set[Path] = set()
    protected: list[str] = []
    for current, directories, filenames in os.walk(ROOT, followlinks=False):
        directory = Path(current)
        directories[:] = [
            name for name in directories
            if name not in NEVER_TRAVERSE and not (directory / name).is_symlink()
        ]
        relative_directory = directory.relative_to(ROOT)
        parts = relative_directory.parts
        python_scope = not parts or parts[0] not in PYTHON_EXCLUDED
        cache = next(
            (index for index, name in enumerate(parts)
             if name in {"__pycache__", ".pytest_cache"}),
            None,
        ) if python_scope else None
        package = next(
            (path for path in packages if directory == path or path in directory.parents),
            None,
        )
        if cache is not None or package is not None:
            removable_directories.add(directory)
        for filename in filenames:
            path = directory / filename
            relative = path.relative_to(ROOT).as_posix()
            category = None
            if filename == ".DS_Store":
                category = "os_metadata"
            elif package is not None:
                category = "setuptools_intermediate"
            elif cache is not None:
                category = "pytest_cache" if ".pytest_cache" in parts else "python_bytecode"
            elif python_scope and path.suffix == ".pyc":
                category = "python_bytecode"
            elif (
                directory in tex_roots
                or directory == ROOT and filename.startswith("llm-attacker-")
            ) and (path.suffix in TEX_SUFFIXES or filename.endswith(".synctex.gz")):
                category = "current_manuscript_tex_intermediate"
            if category is None:
                continue
            if path.is_symlink() or not path.is_file():
                continue
            if relative in tracked:
                protected.append(relative)
                continue
            stat = path.stat()
            files.append({
                "path": relative,
                "category": category,
                "bytes": stat.st_size,
                "allocated_bytes": stat.st_blocks * 512,
                "sha256": digest(path),
                "mtime_ns": stat.st_mtime_ns,
                "device": stat.st_dev,
                "inode": stat.st_ino,
                "status": "candidate",
            })
    return sorted(files, key=lambda item: item["path"]), removable_directories, sorted(protected)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Remove eligible untracked intermediates")
    parser.add_argument("--report", type=Path, help="Write the detailed JSON inventory to this path")
    parser.add_argument("--list", action="store_true", help="Print every candidate as well as the totals")
    args = parser.parse_args()
    report_path = args.report.resolve() if args.report else None
    if report_path is not None and (
        not report_path.is_relative_to(ROOT)
        or any(part in NEVER_TRAVERSE for part in report_path.relative_to(ROOT).parts)
    ):
        parser.error("--report must be inside the repository and outside protected directories")
    if report_path is not None and report_path.relative_to(ROOT).as_posix() in tracked_paths():
        parser.error("--report must not overwrite a Git-tracked file")
    files, directories, protected = inventory()
    errors: list[dict] = []
    removed_directories: list[str] = []
    if args.apply:
        # Recheck Git immediately before mutation, and refuse changed files.
        tracked = tracked_paths()
        for item in files:
            path = ROOT / item["path"]
            try:
                stat = path.stat()
                if item["path"] in tracked:
                    item["status"] = "preserved_newly_tracked"
                elif path.is_symlink() or (
                    stat.st_size, stat.st_mtime_ns, stat.st_dev, stat.st_ino
                ) != (
                    item["bytes"], item["mtime_ns"], item["device"], item["inode"]
                ) or digest(path) != item["sha256"]:
                    item["status"] = "preserved_changed"
                else:
                    path.unlink()
                    item["status"] = "removed"
            except OSError as exc:
                item["status"] = "error"
                errors.append({"path": item["path"], "error": str(exc)})
        for directory in sorted(directories, key=lambda path: (-len(path.parts), str(path))):
            try:
                directory.rmdir()  # Only empty, explicitly selected intermediate directories.
                removed_directories.append(directory.relative_to(ROOT).as_posix())
            except OSError:
                pass  # Preserve directories with tracked, changed or other retained contents.
    removed = [item for item in files if item["status"] == "removed"]
    counts = Counter(item["category"] for item in files)
    report = {
        "schema": "llm-attacker.repository-cleanup.v1",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": str(ROOT),
        "mode": "apply" if args.apply else "dry_run",
        "tracked_files_protected": True,
        "scope": {
            "removed_categories": sorted(counts),
            "preserved": [
                "All Git-tracked files and Git history",
                "Manuscript sources, .bib, .bbl, PDFs and figures",
                "Research results, inputs, captures, manifests and validation records",
                "Historical build evidence, publication snapshots and staging trees",
                "Historical output/, outputs/ and tmp/ content except .DS_Store",
                "Virtual environment and managed configuration directories",
            ],
            "packaging_inspection": (
                "build/lib is setuptools' copied attacker/ package; build/bdist.* is its "
                "platform staging directory; root llm_attacker.egg-info contains generated "
                "PKG-INFO, SOURCES.txt, entry_points.txt, requires.txt, top_level.txt and "
                "dependency_links.txt. pyproject.toml declares setuptools.build_meta. "
                "Other build directories are preserved."
            ),
        },
        "summary": {
            "candidate_files": len(files),
            "candidate_bytes": sum(item["bytes"] for item in files),
            "candidate_allocated_bytes": sum(item["allocated_bytes"] for item in files),
            "category_file_counts": dict(sorted(counts.items())),
            "removed_files": len(removed),
            "removed_bytes": sum(item["bytes"] for item in removed),
            "removed_allocated_bytes": sum(item["allocated_bytes"] for item in removed),
            "removed_empty_directories": len(removed_directories),
            "errors": len(errors),
        },
        "protected_tracked_candidates": protected,
        "files": files,
        "removed_empty_directories": removed_directories,
        "errors": errors,
    }
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"mode": report["mode"], **report["summary"]}, indent=2))
    if args.list:
        for item in files:
            print(f"{item['status']:>24} {item['bytes']:>10} {item['path']}")
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
