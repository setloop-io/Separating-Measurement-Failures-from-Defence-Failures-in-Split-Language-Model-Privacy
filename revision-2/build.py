#!/usr/bin/env python3
"""Rebuild Paper 9 revision 2 without writing to the existing paper directory."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys


REVISION = Path(__file__).resolve().parent
REPOSITORY = REVISION.parent
REVIEW = REVISION / "review"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_preservation() -> dict:
    manifest = json.loads((REVISION / "preservation-manifest.json").read_text())
    results = []
    for relative, expected in manifest["files"].items():
        path = REPOSITORY / relative
        present = path.is_file()
        observed = digest(path) if present else None
        results.append({"path": relative, "expected_sha256": expected["sha256"],
                        "observed_sha256": observed,
                        "matches": present and observed == expected["sha256"]})
    report = {"checked_utc": datetime.now(timezone.utc).isoformat(),
              "scope": manifest["preservation_scope"],
              "files_checked": len(results),
              "all_match": all(item["matches"] for item in results),
              "files": results}
    REVIEW.mkdir(exist_ok=True)
    (REVIEW / "preservation-check.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["all_match"]:
        changed = [item["path"] for item in results if not item["matches"]]
        raise RuntimeError(f"Preserved files differ from the starting snapshot: {changed}")
    return report


def check_inputs() -> int:
    manifest = json.loads((REVISION / "data/source_manifest.json").read_text())
    for record in [*manifest["inputs"], manifest["script"]]:
        path = REPOSITORY / record["path"]
        if not path.is_file() or digest(path) != record["sha256"]:
            raise RuntimeError(f"Pinned analysis input or script changed: {record['path']}")
    assert len(manifest["inputs"]) == manifest["input_file_count"]
    return len(manifest["inputs"])


def executable(name: str, fallback: str) -> str:
    found = shutil.which(name)
    if found:
        return found
    if Path(fallback).is_file():
        return fallback
    raise RuntimeError(f"Required executable unavailable: {name}")


def run(command: list[str], log: Path) -> None:
    with log.open("w") as output:
        result = subprocess.run(command, cwd=REVISION, stdout=output,
                                stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"Command exited {result.returncode}; see {log}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--compile-only", action="store_true",
                      help="Use existing derived tables/figures; compile the manuscript.")
    mode.add_argument("--check-only", action="store_true",
                      help="Check source pins and preserved files; run no generators.")
    args = parser.parse_args()
    preservation = check_preservation()
    input_count = check_inputs()
    if args.check_only:
        print(f"PASS: {preservation['files_checked']} original files unchanged; "
              f"{input_count} input hashes and the analysis script match.")
        return
    if not args.compile_only:
        print("Recomputing saved-result arithmetic...", flush=True)
        run([sys.executable, "analysis/recompute.py"], REVISION / "analysis/recompute.log")
        input_count = check_inputs()
        print("Regenerating vector figures...", flush=True)
        uv = executable("uv", "/opt/homebrew/bin/uv")
        run([uv, "run", "--no-project", "figs/make_figures.py"],
            REVISION / "figs/figure-audit.log")
    pdflatex = executable("pdflatex", "/Library/TeX/texbin/pdflatex")
    bibtex = executable("bibtex", "/Library/TeX/texbin/bibtex")
    for number in range(1, 4):
        print(f"Compiling LaTeX, pass {number}/3...", flush=True)
        run([pdflatex, "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
            REVIEW / f"latex-pass-{number}.log")
        if number == 1:
            print("Building the numbered bibliography with BibTeX plain...", flush=True)
            run([bibtex, "main"], REVIEW / "bibtex.log")
    log = (REVISION / "main.log").read_text()
    failures = [line for line in log.splitlines() if re.search(
        r"undefined|multiply defined|Overfull \\[hv]box|Rerun to get cross-references", line,
        flags=re.IGNORECASE)]
    if failures:
        raise RuntimeError("Unresolved LaTeX issues: " + "\n".join(failures))
    pdf = REVISION / "main.pdf"
    if not pdf.is_file() or not pdf.stat().st_size:
        raise RuntimeError("LaTeX produced no PDF")
    preservation = check_preservation()
    input_count = check_inputs()
    analysis = json.loads((REVISION / "data/audit_results.json").read_text())
    report = {"built_utc": datetime.now(timezone.utc).isoformat(),
              "mode": "compile_only" if args.compile_only else "full_rebuild",
              "original_files_unchanged": preservation["files_checked"],
              "pinned_input_files": input_count,
              "internal_analysis_checks": analysis["validation"]["checks_passed"],
              "pdflatex_passes": 3, "bibtex_passes": 1,
              "bibliography_style": "plain", "unresolved_latex_issues": failures,
              "pdf": {"path": "main.pdf", "bytes": pdf.stat().st_size,
                      "sha256": digest(pdf)},
              "note": "Compilation and internal consistency are distinct from independent experimental validation; visual QA is recorded separately."}
    (REVIEW / "build-validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS: {pdf.name}; {input_count} pinned inputs; "
          f"{report['internal_analysis_checks']} internal checks; original files unchanged.")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, AssertionError) as error:
        print(f"BUILD FAILED: {error}", file=sys.stderr)
        raise SystemExit(1)
