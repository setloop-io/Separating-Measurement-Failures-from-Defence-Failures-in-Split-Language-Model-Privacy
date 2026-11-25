# Second review and frozen candidate — 22 September 2026

Candidate tag: `submission-candidate-20260922-r2`. This identifies one reviewed source, bibliography and PDF state for both papers. It supersedes the first minor revision's mutable evidence links and package hashes. It is a submission candidate awaiting author decisions, not an arXiv submission or an unrestricted artifact release.

The starting worktree was clean at `41dde59`; the previous citations, VMI definition, rounding note and gating disclosure were already committed. The review's report of an uncommitted candidate therefore described an earlier state. This revision fixes the remaining wording and records a new freeze.

## Checked corrections

- **Gradient controls, both papers:** the released depth-six/public/training means are shuffled 93.905029296875%, random 93.8873291015625%, and forward-only 94.2041015625%. The manuscripts give 93.91%, 93.89%, and 94.20%, respectively, and describe the controls as about 0.3 percentage points below forward-only. No equivalence claim is made.
- **Horizon aggregation:** Gaussian 0.3→4.2% and sparsification 17.5→25.9% compare steps 0–7 with 50–99 inside the 100-step runs. The Gaussian whole-horizon means are 0.89% at 20 steps and 2.84% at 100 steps. The excess table reports the mean per-run gate score across three seeds. These aggregations are now distinguished.
- **Accumulation scope and numbers:** the four-epoch captures have no per-epoch rotation and cannot test resistance to rotation defences. The old prose's 19.8–50.3% range was also not a complete three-seed, three-budget summary. The manuscript now gives all saved aligned-label results below. The underlying report JSON agrees exactly with the individual aligned-probe JSON files. No probe was rerun, and the old files are retained.
- **Training controls:** exact pooled counts are 154/768 (code), 48/768 (clinical), and 36/768 (WikiText). At frozen Wilson z=2.5391848136513127, the control-rate floors recompute to 3.9110904631, 2.6026956303 and 2.3423927656 percentage points. A new table gives counts, rates, formula and floors. The 768 positions pool three seeds; this remains token-binomial operational accounting, not a clustered inference claim.
- **White-box overlap:** the 198-cell grid re-executes the 36 depth-three/six configurations on the same development inputs with fresh captures and adds 162 configurations at other depths. All 36 overlapping accuracy values agree exactly, but all execution hashes differ. The grid implementation explicitly creates fresh captures. These are neither independent input replications nor 36 additional unique conditions. The separate 99 co-trained final-state cells retain their different access grant.
- **References and labels:** the v3 table and pipeline figure are referenced in prose. Two unused historical-section label aliases are replaced by one descriptive label; all label keys are unique and all references resolve. Paper B's BiSR entry now cites CCS 2024, consistent with the [primary publication record](https://arxiv.org/abs/2409.00960) and [publication DOI](https://doi.org/10.1145/3658644.3690295). Bibliography/citation counts remain 19 for A and 41 for B.

### Archived aligned accumulation results

Three-solve-seed mean percentages on the same unrotated four-epoch target; pair budget is per epoch. The nearest-neighbour surrogate decoder does not establish a rotating-defence security threshold.

| Pair budget | Undefended within epoch | Undefended pooled across epochs | Mid-lite within epoch | Mid-lite pooled |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 19.72 | 16.39 | 0.00 | 0.00 |
| 256 | 50.56 | 51.28 | 0.00 | 0.00 |
| 768 | 19.50 | 18.78 | 0.00 | 0.50 |

The summary and source digests are in [the machine-readable record](submission-candidate-2026-09-22-r2.json). They identify the saved data used for these editorial arithmetic checks; no model or scientific-analysis experiment was launched.

## Evidence links and access

All 12 HF links now fix previously verified evidence revisions. Fresh authenticated checks resolve every named directory. The Qwen link uses completed evidence revision `c0981bf75ba4a62e4388c3477c63e9f90d5d01fd`, not the incomplete closure snapshot. Existing complete manifest verifications remain the evidence-integrity records; this revision checks pinned directory metadata without repeating a full binary download.

HF reports `private=false`, `gated=manual` for both datasets. GitHub reports `PRIVATE`. Paper B now links the frozen code location and states these restrictions explicitly; source snapshots in the pinned HF releases are also gated. There is currently no unrestricted public code-and-evidence download path. No visibility, gating or licence setting was changed. Artifact eligibility must be checked against the selected venue's actual policy; no blanket acceptance or desk-rejection conclusion is made here.

## Freeze and validation

Both `.bbl` files were regenerated with BibTeX and the manuscripts rebuilt. The isolated source ZIPs compile without undefined citations/references, LaTeX warnings or overfull boxes. Every ZIP member and PDF matches its recorded SHA-256; bibliography bytes agree with the repository and extracted package PDF text agrees with the repository PDF. The new table, changed text and affected pagination were rendered and reviewed. An obsolete forced page break that left most of Paper B's second page empty was removed. Figures and empirical evidence files are unchanged.

The final packages are at `build/arxiv-preprints-20260922-freeze-r2-final/INDEX.json`: Paper A 14 pages, Paper B 25 pages. The earlier `freeze-r2` and `freeze-r2-reviewed` builds are retained preflight outputs, not the final candidate. Generated `.aux`, `.blg`, `.log` and `.out` files are excluded from the tracked submission source so stale build state cannot accompany the frozen `.tex`, `.bib`, `.bbl` and `.pdf`. Original logs remain in local build outputs and Git history. A clean compilation of the committed tag must regenerate the committed `.bbl` exactly and match the frozen PDF text; the local post-commit verification is `build/submission-freeze-20260922-r2/committed-head-check.json`.

Author decisions remain: artifact-access posture and target venue; names/order, affiliations and corresponding author; category/cross-list; manuscript licence and, for any public code release, the code licence; shared-evidence overlap disclosure and submission approval. The current tree has no LICENSE file. No history deletion, experimental rerun, HF manuscript upload or submission was performed. History cleanup must be reconciled with retaining the now-restored evidence revisions before any destructive action.
