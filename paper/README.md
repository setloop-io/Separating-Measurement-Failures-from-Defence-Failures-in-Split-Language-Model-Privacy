# Manuscript source (`paper/`)

**Separating Measurement Failures from Defence Failures in Split Language-Model Privacy**

Open [main.pdf](main.pdf) for the compiled manuscript (11-point Times article,
one-inch margins, BibTeX `plain` numbered citations, 27 references).

## Files

| Path | Purpose |
| --- | --- |
| `main.tex`, `sections/` | Complete manuscript source |
| `results_macros.tex` | Generated count macros synced by `../analysis/recompute.py` |
| `refs.bib`, `main.bbl` | 27 bibliography records and generated numbered bibliography |
| `placeins.sty` | Bundled float-barrier style file |
| `figs/` | Vector figures (`campaign-gates.pdf`, `r1-utility-boundary.pdf`), plotting scripts, and provenance |
| `build.py` | End-to-end input verification, reanalysis, LaTeX/BibTeX compilation, and `arxiv-source.zip` smoke test |
| `main.pdf` | Current compiled manuscript |
| `arxiv-source.zip` | Clean, self-contained arXiv upload archive verified after isolated extraction |

## Rebuild

From the repository root (requires Python 3.10+, `pdflatex`, `bibtex`, and
optionally `uv` to regenerate figures):

```bash
python3 paper/build.py
```

Compile using existing derived tables and vector figures, or verify pinned
input hashes only:

```bash
python3 paper/build.py --compile-only
python3 paper/build.py --check-only
```

Pass `--regen-figures` to regenerate the vector PDFs and SVGs via `uv` before
compiling. The generated `arxiv-source.zip` includes `main.bbl` alongside
`refs.bib`, `results_macros.tex`, `placeins.sty`, `sections/*.tex`, and the two
vector figure PDFs, with no internal logs or scratch files.
