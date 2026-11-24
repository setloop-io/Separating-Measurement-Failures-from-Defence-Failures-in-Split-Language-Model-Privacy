#!/usr/bin/env python3
"""Rebuild the manuscript, verify pinned inputs, and package arxiv-source.zip."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile


PAPER = Path(__file__).resolve().parent
REPOSITORY = PAPER.parent
DATA = REPOSITORY / "data"
ANALYSIS = REPOSITORY / "analysis"


ARXIV_FILES = [
    "main.tex",
    "results_macros.tex",
    "refs.bib",
    "main.bbl",
    "placeins.sty",
    "sections/admissibility-framework.tex",
    "sections/case-studies.tex",
    "sections/results.tex",
    "sections/scope-checks.tex",
    "sections/discussion.tex",
    "sections/artifact-index.tex",
    "figs/campaign-gates.pdf",
    "figs/r1-utility-boundary.pdf",
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_inputs() -> int:
    manifest = json.loads((DATA / "source_manifest.json").read_text(encoding="utf-8"))
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


def run(command: list[str], cwd: Path) -> str:
    result = subprocess.run(
        command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, check=False,
    )
    if result.returncode:
        tail = "\n".join(result.stdout.splitlines()[-40:])
        raise RuntimeError(f"Command exited {result.returncode}: {' '.join(command)}\n{tail}")
    return result.stdout


def compile_latex(workdir: Path, pdflatex: str, bibtex: str | None) -> list[str]:
    max_passes = 4
    for number in range(1, max_passes + 1):
        run([pdflatex, "-interaction=nonstopmode", "-halt-on-error", "main.tex"], cwd=workdir)
        if number == 1 and bibtex is not None:
            run([bibtex, "main"], cwd=workdir)
        log = (workdir / "main.log").read_text(encoding="utf-8", errors="replace")
        if number >= 3 and not re.search(r"Rerun to get", log, flags=re.IGNORECASE):
            break
    log = (workdir / "main.log").read_text(encoding="utf-8", errors="replace")
    failures = [
        line for line in log.splitlines()
        if re.search(
            r"undefined|multiply defined|Overfull \\[hv]box|Rerun to get cross-references",
            line, flags=re.IGNORECASE,
        )
    ]
    return failures


def build_arxiv_zip(pdflatex: str, bibtex: str) -> Path:
    bundle = PAPER / "arxiv-source.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in ARXIV_FILES:
            src = PAPER / rel
            if not src.is_file():
                raise RuntimeError(f"Missing required arXiv source file: {rel}")
            zf.write(src, arcname=rel)
    # Smoke-test the extracted bundle in an isolated temporary directory using
    # pdflatex only (matching arXiv's compilation with the prebuilt main.bbl).
    with tempfile.TemporaryDirectory(prefix="arxiv-smoke-") as tmp:
        tmp_path = Path(tmp)
        with zipfile.ZipFile(bundle, "r") as zf:
            zf.extractall(tmp_path)
        failures = compile_latex(tmp_path, pdflatex, bibtex=None)
        if failures:
            raise RuntimeError("Extracted arxiv-source.zip had LaTeX issues:\n" + "\n".join(failures))
        if not (tmp_path / "main.pdf").is_file():
            raise RuntimeError("Extracted arxiv-source.zip produced no main.pdf")
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--compile-only", action="store_true",
                      help="Use existing derived tables and figures; compile manuscript and package arxiv-source.zip.")
    mode.add_argument("--check-only", action="store_true",
                      help="Verify all pinned input hashes and exit without compiling.")
    parser.add_argument("--regen-figures", action="store_true",
                        help="Regenerate vector figures via uv before compiling.")
    args = parser.parse_args()

    input_count = check_inputs()
    if args.check_only:
        print(f"PASS: {input_count} pinned input hashes and analysis/recompute.py match.")
        return

    if not args.compile_only:
        print("Recomputing archived-evidence arithmetic...", flush=True)
        run([sys.executable, str(ANALYSIS / "recompute.py")], cwd=REPOSITORY)
        input_count = check_inputs()

    if args.regen_figures:
        print("Regenerating vector figures...", flush=True)
        uv = executable("uv", "/opt/homebrew/bin/uv")
        run([uv, "run", "--no-project", str(PAPER / "figs/make_figures.py")], cwd=REPOSITORY)

    pdflatex = executable("pdflatex", "/Library/TeX/texbin/pdflatex")
    bibtex = executable("bibtex", "/Library/TeX/texbin/bibtex")
    print("Compiling manuscript with pdfLaTeX and BibTeX...", flush=True)
    failures = compile_latex(PAPER, pdflatex, bibtex)
    if failures:
        raise RuntimeError("Unresolved LaTeX issues:\n" + "\n".join(failures))

    pdf = PAPER / "main.pdf"
    if not pdf.is_file() or not pdf.stat().st_size:
        raise RuntimeError("LaTeX produced no PDF")

    print("Packaging and smoke-testing paper/arxiv-source.zip...", flush=True)
    bundle = build_arxiv_zip(pdflatex, bibtex)
    analysis = json.loads((DATA / "reanalysis_results.json").read_text(encoding="utf-8"))
    print(
        f"PASS: {pdf.name} ({pdf.stat().st_size:,} bytes), {bundle.name} ({bundle.stat().st_size:,} bytes); "
        f"{input_count} pinned inputs; {analysis['validation']['checks_passed']:,} internal checks."
    )


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, AssertionError) as error:
        print(f"BUILD FAILED: {error}", file=sys.stderr)
        raise SystemExit(1)
