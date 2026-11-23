# Recomputing the secondary reanalysis (`analysis/`)

Run from the repository root:

```bash
python3 analysis/recompute.py
```

The script uses Python's standard library only, resolves the repository root
relative to its own location, writes derived outputs to `data/`, and syncs
generated LaTeX count macros to `paper/results_macros.tex`. It neither imports
nor executes archived training or attacker implementations and loads no tensor
captures, checkpoints, or secret key material.

## Outputs (`data/`)

- `reanalysis_results.json`: recomputed gate arithmetic and explicitly labelled
  extractions and analytic arguments, with denominators and source levels.
- `r1_cells.csv`, `p4_cells.csv`: flat tables consumed by
  `paper/figs/make_figures.py`.
- `results_macros.tex`: generated headline-count macros (mirrored to
  `paper/results_macros.tex`).
- `source_manifest.json`, `source-index.csv`: exact repository-relative input
  paths, byte lengths, and SHA-256 digests for all 122 pinned inputs, plus the
  script's own hash.

## Statistical conventions preserved

- **19-cell exploration (`r1`):** Recomputed from per-seed losses and saved
  probe scores in `reproduction/trilemma-campaign/evidence/r1-exploration-20260917/analysis/exploratory_analysis.json`
  and cross-checked against `evidence/trilemma-campaign/r1_cell_results.csv`
  and `r1_seed_results.csv`. Uses two-sided 95% Student-$t$ intervals over four
  paired training-seed contrasts (`df=3`, critical value `3.182446305284263`).
  Rejection reasons overlap; the JSON reports both overlapping marginal counts
  and disjoint intersections.
- **Seven-cell defence-redesign pilot (`p4`):** Recomputed from 66 archived
  run-metrics records (`reproduction/defense-redesign-pilot/outputs/runs/pilot-20260922-a/`),
  including paired losses, saved candidate/null score arrays, and saved
  qualification controls. Uses a one-sided 95% upper Student-$t$ utility limit
  over three paired seeds (`df=2`, critical value `2.9199855803537242`). Both
  raw worst-seed and strictly qualified worst-seed punctuation-property
  advantages are recorded alongside next-byte structure advantages.
- **Orthogonal wrapper (`e8`) and deployed ratcheting (`er`):** Extracted from
  archived JSON records and the accepted collection manifest while retaining
  all host-key, identity, and throughput limitations.
- **Decoy gradients (`E1`), boundary review (`B1`), and variational MI (`MI`):**
  All cited primary documents are hashed in `data/source_manifest.json`; their
  experimental scores remain source-reported alongside the analytic
  Donsker--Varadhan counterexample.

This is a retrospective reanalysis of existing archived evidence. Its 1,522
internal checks verify arithmetic and disposition consistency rather than
retraining models or forming a prospective validation cohort.
