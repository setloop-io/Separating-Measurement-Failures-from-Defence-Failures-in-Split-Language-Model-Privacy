# Manuscript revision 2

**Auditing Privacy Claims in Split Language-Model Training**

Open [main.pdf](main.pdf) for the current manuscript. Its layout and citations
follow the user-supplied `arXiv-2609.04382v1.tar` source, with an 11-point Times
article, one-inch margins, the standard title and author blocks, and numbered
citations from BibTeX's `plain` style. The original manuscript in `../paper/`
remains unchanged.

The paper audits four experimental campaigns and two broader privacy claims.
Its secondary analysis uses existing saved results. Scientific values,
uncertainty conventions, source levels and the eight cited records are retained.
Descriptive experiment names replace the internal workload numbers. The prose
has no colons, semicolons or rhetorical questions. Exact source paths remain
intact in the reproducibility appendix.

## Files

| Path | Purpose |
| --- | --- |
| `main.tex`, `sections/` | Complete manuscript source |
| `refs.bib`, `main.bbl` | Eight bibliography records and generated numbered references |
| `main.pdf` | Current compiled manuscript |
| `build.py` | Analysis, figure and LaTeX/BibTeX build recipe |
| `placeins.sty`, `template/provenance.json` | Style file from the attachment and its source bindings |
| `analysis/`, `data/` | Rerunnable secondary analysis, derived tables and 118 hashed inputs |
| `figs/` | Vector figures, plotting scripts and provenance |
| `review/attached-style-verification-2026-11-07.md` | Independent style, citation and content check |
| `review/pdf-qa.md`, `review/build-validation.json` | Rendered-PDF and build checks |
| `arxiv-source.zip` | Clean source bundle tested after extraction |
| `REVISION_NOTES.md` | Scientific changes and publication judgment |

## Rebuild

Keep this directory inside the full experiment repository. Python 3.10+, `uv`
for plotting, and a TeX installation with `pdflatex` and `bibtex` are required.
The LaTeX packages are standard TeX Live packages. Figures already contain their
embedded fonts, so compilation using the supplied figures needs no system font
installation. Plot regeneration uses the font requirements in `figs/`.

```bash
python3 build.py
```

The full build checks original-file and input hashes, recomputes the analysis,
regenerates the figures, runs pdfLaTeX, BibTeX with `plain`, and two further
pdfLaTeX passes. All outputs stay inside `revision-2/`.

To compile using the supplied tables and figures, or to check preservation only,
use the following commands.

```bash
python3 build.py --compile-only
python3 build.py --check-only
```

The clean source ZIP includes the bibliography database and generated `.bbl`
file. It excludes evidence, logs, backups and unused style examples. No
submission or publication has been made.

## Evidence limits and preservation

The saved analysis reports 1,522 internal consistency checks over 118 hashed
inputs. These are arithmetic and disposition checks, not new experiments or
independent validation. The exploration retains two-sided 95% seed intervals
and the defence pilot a one-sided 95% upper utility limit. Source-reported
findings remain identified as such. See [analysis/README.md](analysis/README.md).

Single-program retrospective selection, shared data and overlap with the
existing attack-suite work limit the novelty and generality claims. The
manuscript makes no unsupported public-release or security-guarantee claim.

All 11 protected original files are checked against `preservation-manifest.json`.
The version immediately before this attached-paper style correction is saved
in `review/pre-attached-style-2026-11-07/`. The earlier editorial baseline is in
`review/pre-edit-2026-11-07/`. The replaced community style is archived under
`template/previous-community-style/` and is not used by the current manuscript.
