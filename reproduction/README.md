# Reproduction archives (`reproduction/`)

Preserved primary experimental archives backing the retrospective reanalysis in
*Separating Measurement Failures from Defence Failures in Split Language-Model Privacy*
(`paper/main.tex`). All original archive files are retained byte-identical so
upstream manifests and SHA-256 digests remain verifiable.

To verify all 122 pinned inputs and 1,522 internal consistency checks offline
using only the Python standard library, run from the repository root:

```bash
python3 analysis/recompute.py
```

## Archived campaign directories

### Decoy-gradient experiment (`decoy-gradient/`)
Companion source archive for arXiv:2609.04382 (zero-support returned-gradient
metadata disclosure, forward-vs-joint gate comparison, and per-row gradient
clipping/noise mitigation). Summarised in `evidence/decoy-gradient/paper1-main.tex`
and `E1_GRADIENT_CHANNEL_2026-08-22.md`.

### Boundary-replacement review (`boundary-replacement/`)
Corrected endpoint controls (`boundary-replacement/outputs/phase08_clean_endpoints/`)
and upstream records referenced by the independent evidence review in
`evidence/boundary-replacement/independent_evidence_review_2026-09-09/VERDICTS.md`
and `recomputed_evidence.json` (shifted next-token label alignment, 6/6
zero-return comparisons, and −16.64 pp teacher-forced answer-token drop).

### 19-cell trilemma exploration (`trilemma-campaign/`)
Frozen exploratory configuration (`experiments/configs/r1_exploration.yaml`),
qualification records (`evidence/r1-qualification-20260916/attack-spec-frozen.json`,
`calibration_analysis_summary.json`), exploratory analysis JSON
(`evidence/r1-exploration-20260917/analysis/exploratory_analysis.json`), and
unexecuted eight-seed planning bound (`confirmation_power.json`). Recomputed
into `data/r1_cells.csv` and `data/reanalysis_results.json`.

### Seven-cell defence-redesign pilot (`defense-redesign-pilot/`)
Frozen protocol (`experiments/freeze/p4-defense-pilot-1.0.json`), calibration and
analysis code (`src/p4d/calibration.py`, `src/p4d/pilot_analysis.py`), pilot
run manifests (`experiments/manifests/`), and all 66 completed run records
(`outputs/runs/pilot-20260922-a/`). Recomputed into `data/p4_cells.csv` and
`data/reanalysis_results.json`.

### Orthogonal wrapper and deployed ratcheting (`structural-e8/`)
27B dense-$K$ robustness artifact
(`paper-data/collected/h100-2/issues/issue_e8_27b_20260807/e8_robustness_27b_bf16_densek_20260807T230529Z.json`)
and deployed 12B per-epoch ratcheting matrix
(`paper-data/collected/toula/er_20260811/COLLECTION_MANIFEST.json`, `summary.json`,
`identity.json`, and 56 `wire_attack_*.json` files), paired with
`evidence/structural-e8/EVIDENCE_EXECUTION_LEDGER.md` and `EXPERIMENT_STORY.md`.

### Variational mutual-information companion records (`privacy-contract/`)
Supporting estimator code and latent campaign records paired with
`evidence/privacy-contract/privacy-contract-draft.tex` and `KEY_NUMBERS.md`.

### Attack-suite companion archive (`attack-suite/`)
Companion framework and benchmark archive associated with arXiv:2610.04128
(13.71% to 37.77% exact-document recovery and +3.17 pp token recovery on GPT-2)
and arXiv:2610.22722 (LLM-Attack Suite), preserved alongside
`evidence/attack-suite/2610.04128-main.tex`.

## Provenance notes

- Every file directly consumed by `analysis/recompute.py` or cited in
  Appendix A of the manuscript is SHA-256 pinned in `data/source_manifest.json`
  and indexed in `data/source-index.csv`.
- See `evidence/EVIDENCE_MAP.md` for the full claim-to-file mapping.
