# Recomputing the revision-2 secondary analysis

Run from `revision-2/`:

```bash
python3 analysis/recompute.py > analysis/recompute.log 2>&1
```

The script uses Python's standard library, resolves the original package root
relative to its own location, and writes only `revision-2/data/`. It neither
imports nor executes archived training/attacker implementations and loads no
tensor captures, checkpoints or secret key material.

Outputs:

- `audit_results.json`: recomputed gate arithmetic and explicitly labelled
  extractions/analytic arguments, with denominators and source levels.
- `r1_cells.csv`, `p4_cells.csv`: flat tables used by the revision's figures.
- `results_macros.tex`: generated headline-count macros for the manuscript.
- `source_manifest.json`, `source-index.csv`: exact relative input paths,
  byte lengths and SHA-256 hashes, plus the script's own hash.

The privacy, utility and gain exploration is recomputed from per-seed losses and saved probe scores in the archived
analysis dossier, then checked against the provided seed/cell CSVs. Its
confidence intervals are two-sided 95% Student-t intervals over four training
seeds (`df=3`, critical value `3.182446305284263`). Rejection reasons overlap;
the JSON reports both marginal counts and disjoint reason intersections.

The defence-redesign pilot is recomputed from 66 archived run-metrics records: paired losses,
saved candidate/null score arrays, and the saved null/positive qualification
scores. Its utility gate uses a one-sided 95% upper Student-t limit over three
training seeds (`df=2`, critical value `2.9199855803537242`). The script checks
its results against the earlier generated verification/report. Recomputing
saved score arithmetic is not refitting probes or rescoring tensor captures.

E8 crossing/session results and E-R identity/timing observations are extracted
from archived JSONs and retain their original threat-model limitations. The
E-R throughput anchor `2.8/7.6` is present in a saved identity record; those are
the recorded rates, not claims of greater hardware-measurement precision.
The MI example is a logical counterexample, not an experiment. E1 and B1
documents are hashed for the manuscript's source index but their studies are
not newly reproduced here.

This is a retrospective reanalysis of existing evidence. It creates no new
independent, blinded or confirmatory validation cohort.
