# Paper 9 — revision 2

**Auditing Privacy Claims in Split Language-Model Training: Channels, Targets, Calibration, and Scope**

This is a separate journal/conference research-paper draft based on the existing
Paper 9 artifact collection. Open [main.pdf](main.pdf) for the paper and
[REVISION_NOTES.md](REVISION_NOTES.md) for the substantive changes and publication
judgment. The original manuscript remains in `../paper/`.

The revision presents a retrospective audit of four campaigns and two broader
scope checks. It adds rerunnable secondary analysis of archived results, two
vector figures, and an explicit source index. It does not report new training,
attack fitting, or an independently validated audit cohort. Valid negative
defense results remain negative results.

## Contents

| Path | Purpose |
| --- | --- |
| `main.tex`, `sections/`, `references.tex` | Complete manuscript source; inline bibliography |
| `main.pdf` | Compiled paper |
| `analysis/recompute.py` | Standard-library secondary analysis |
| `data/audit_results.json` | Derived statistics, decisions, source levels and limitations |
| `data/r1_cells.csv`, `data/p4_cells.csv` | Recomputed figure inputs |
| `data/source_manifest.json`, `data/source-index.csv` | 118 consumed inputs with paths, byte lengths and SHA-256 hashes |
| `figs/` | Vector figures, plotting scripts, provenance and figure checks |
| `review/verification.md` | Separate claim, arithmetic and citation check |
| `review/build-validation.json`, `review/pdf-qa.md` | Build and rendered-PDF checks |
| `preservation-manifest.json`, `review/preservation-check.json` | Starting snapshot and final check of 11 existing manuscript/repository files |

## Rebuild

Keep this directory inside the full Paper 9 repository. Primary inputs are read
from paths relative to its parent directory; the source manifest lists every
consumed file. The mounted repository used for this revision is
`/Users/geo/Vol/NAS/Paper-9-Calibrated-Attacks-and-Structural-Leakage`.

Requirements are Python 3.10+, `uv` for the isolated plotting dependencies,
and a TeX installation with `pdflatex`. Plot dependencies are declared in
`figs/make_figures.py`; no archived experimental implementation is executed.
The manuscript uses standard TeX Live packages, including `placeins`, `natbib`,
`xurl`, `hyperref` and Latin Modern. The bundled inline bibliography requires
no BibTeX invocation. The recorded figure rendering used Helvetica Neue; that
font should be available for identical typography.

From this directory:

```bash
python3 build.py
```

This checks the source pins and original-file hashes, recomputes the arithmetic,
regenerates the figures, compiles the manuscript three times, checks for
unresolved citations/references and overfull boxes, and checks preservation
again. Logs and outputs are written only inside `revision-2/`.

To compile using the supplied figures and derived tables:

```bash
python3 build.py --compile-only
```

To check the saved input and original-file hashes without rebuilding:

```bash
python3 build.py --check-only
```

The analysis and figures can also be regenerated separately:

```bash
python3 analysis/recompute.py > analysis/recompute.log 2>&1
uv run --no-project figs/make_figures.py
```

## Interpretation and limits

The analysis passes 1,522 arithmetic and disposition checks against 118 hashed
inputs. These are consistency checks on the recorded evidence, not 1,522 new
experiments or an estimate of the audit's accuracy. R1 retains its two-sided
95% seed intervals; Paper 4 retains its one-sided 95% upper utility limit.
Saved scores are recomputed without retraining probes or rescoring tensor
captures. The F1 and parts of F2 findings remain source-reported, and the
MI counterexample is analytic. See [analysis/README.md](analysis/README.md).

The paper is framed as an artifact-backed audit/methodology case study. Its
single-program retrospective selection, overlap with the authors' existing
attack-suite work, and lack of an independent validation cohort limit the
novelty and generality claims. No venue acceptance or universal privacy
guarantee follows from this revision. The availability statement identifies
the companion artifact collection without inventing a public repository URL.

All new manuscript and supporting files are confined to `revision-2/`; the
original `paper/` files and repository README are checked against their starting
hashes. Existing evidence is read in place rather than copied into a new attack
implementation.
