# R1 exploration review — 17 September 2026

This package admits **exploratory**, finite-family results from the complete
Toula attempt `exp002`. It does not admit confirmation, a universal trilemma or
complete-transcript privacy. Pre-reset outcomes remain inadmissible.

## Results

- Calibration `cal001`: 76/76 complete; all 3,572 manifest-listed artifacts
  verified. Regenerated thresholds and all combination records exactly match
  the original proposal frozen on 16 September. Four of 24 instruments qualify.
- Exploration `exp002`: 232/232 complete, no failures or missing runs; all 10,088
  manifest-listed artifacts verified. All 19 configurations are REJECTED by the
  frozen exploratory rules: 13 qualified privacy breaches, five utility failures,
  seven gain failures (overlapping reasons). Six privacy assessments remain
  UNDETERMINED; a failing utility/gain gate rejects those cells independently.
- The 76 matched cell/seed triplets pass retained initialization and configuration
  pairing checks. Only depth-2 and depth-4 pass the separate removal-benefit gate.
- These are unadjusted four-seed exploratory gate classifications. In particular,
  failing the gain gate does not establish a population effect below its margin.

## Evidence chain and storage

The original scientific bundle is `7987e7f9…` at source freeze `5fdb9d19…`;
its complete identifiers are in the retained launch/admission records. Its
191-file bundle verification passed again on Toula during this review.
All retained source-inventory entries across all 308 scientific runs match that
frozen bundle, as recorded in [source review](bundle_source_review.json). Execution
files were not edited. The analyst code is commit **a9700e9**, with exact code
file hashes in each analysis provenance sidecar. Later reporting code has its
own file hashes and commit history.

| Evidence | Retained location |
| --- | --- |
| Original complete exploration | `toula:/home/geo/r1-bundle-sci-20260916/outputs/p3-r1-fresh-start/exploratory--exp002` |
| Independently verified exploration replica | `koula:/home/geo/r1-evidence-archive-20260917/exploratory--exp002` |
| Original calibration | `toula:/home/geo/r1-bundle-sci-20260916/outputs/p3-r1-fresh-start/calibration--cal001` |
| Calibration retrieval | `/Users/geo/r1-executions-20260916/toula/cal001` |
| Original analyzer inputs/output copy | `toula:/home/geo/r1-review-20260917/` |

Exploration comprises **10,790 files / 111,859,148,305 bytes**. Every replica
file matches the origin inventory, with no added/missing files; see
[replica verification](exp002-replica-verification.json). Calibration comprises
3,811 files / 38,870,335,106 bytes. Full tensors remain in these private owned-node
archives; the repository contains compact provenance, manifests and summaries,
not a third 105-GiB copy. Access requires the owner's SSH access; this is not a
public evidence release.

`stage-records/{cal001,exp002}/` retains the original launch, admission, claim,
panel, summary, every start/end record and every run manifest. The manifests bind
the original artifact hashes. Tree inventories in `analysis/` cover every raw
file, including files not individually needed for numerical summaries. The
interrupted `exp001` records remain at `../r1-qualification-20260916/exp001-interruption/`
and on Koula; its nine completed runs are not pooled with `exp002`.

## Analysis repair and verification

The runner binds canonical JSON for the admitted `r1-attack-spec-1.0` wrapper;
the former analyzer expected a byte-bound standalone qualification proposal.
The repaired analyzer validates the canonical launch hash, the frozen document's
file-byte bindings and the exact proposal digest in its reviewed `origin`.
The original qualification proposal is
`../r1-qualification-20260916/analysis/calibration_analysis.frozen-attack-spec.json`.
No qualification or threshold was selected anew from exploration.

Before reading outcomes the analyzer now verifies all manifest artifact hashes,
required inventories, stage admission/claim/panel bindings, run start/end links,
source archive contents, capture/snapshot digests and attack input/specification
bindings. Cross-mode checks are in [control pairing review](control_pairing_review.json).
This verifies retained bytes and recorded provenance, **not numerical training
replay**, independent re-execution of all tensor semantics or an externally
witnessed preregistration timestamp. The original records' `unknown-commit`
container field is not used as a substitute for archived source and bundle hashes.

All 105 R1 tests pass, including altered/missing artifact and wrapped-spec
regressions. Figure generation has a clean printed-size audit; PDF rendering
is inspected separately. Reproduce analysis in the accepted CPU-capable runtime:

```bash
python scripts/analyze_r1_scientific.py \
  --stage exploratory --root /path/to/exploratory--exp002 \
  --qualification evidence/r1-qualification-20260916/analysis/calibration_analysis.frozen-attack-spec.json \
  --output /new/analysis/exploratory_analysis.json \
  --analyzer-commit a9700e9 --inventory
```

Use a new output path; originals are immutable. `--stage calibration` rechecks
calibration without `--qualification`. The runner never trains. Reproduce paper
outputs with `uv run python paper/figs/r1_results.py` and `bash paper/build.sh`.
Input/output hashes accompany both analysis and figures.

## Confirmation decision

[Confirmation review](../../docs/R1_CONFIRMATION_REVIEW.md) and
[power calculation](confirmation_power.json) describe an unfrozen 464-run,
eight-seed design. The power analysis does **not** support calling that design
adequately powered for full-family confirmation. The paper therefore remains
exploratory. No confirmation or paid cloud execution was launched.
