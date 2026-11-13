# P2c segment-statistics decision record

Date: 2026-09-06
Protocol: `experiments/configs/segment_stats_matrix.yaml` (`p2c-segment-statistics-1.0`, frozen
2026-09-04 after statistic selection, before any confirmatory run; preregistration
`docs/PREREGISTRATION_P2C.md` with the selection rule frozen BEFORE measurement).
Cohort: **confirmatory**, fresh seeds [3, 13, 23, 33, 43, 53, 63, 83]; statistics measured on
calibration backbone seed 929.
Artifacts: `outputs/stats/segment_stats_s929.json`, `outputs/stats/selection_table.{json,md}`,
`outputs/segment_stats/` (72 runs), `segment_stats_analysis.json`, `segment_stats_summary.md`,
`paper/figures/segment_stats_prediction.png`.

## What was done

Five candidate segment statistics were measured on the calibration backbone for all 36
contiguous segments. The frozen selection rule (highest |Pearson r| with mean Δ_U across the six
exploratory cells {2@3, 4@2, 6@1} × {64, 256}, P2 means) selected **intrinsic_dimension**
(participation ratio of segment-output activations): r = −0.9905 across the six cells,
−0.9966 in the width-256 subset. Selection table (auditable, all candidates):

| Statistic | r (6 cells) |
|---|---:|
| residual_update_magnitude | 0.007 |
| block_influence | −0.919 |
| representation_similarity (CKA) | −0.899 |
| layer_drop_sensitivity | 0.980 |
| intrinsic_dimension | −0.9905 |

## Confirmatory verdict — H2: **NOT CONFIRMED**

On the 8 held-out cells {2@1, 2@5, 3@2, 3@4, 4@3, 5@1, 5@3, 6@2} (width 256, q=8, n=7 seeds/cell
after the predeclared seed-drop, below):

- Model S (Δ_U ~ intrinsic_dimension, fit on exploratory points only): out-of-sample R² = **−7.98**,
  mean |error| 0.0971 nats.
- Model N (Δ_U ~ span only): out-of-sample R² = **0.455**, mean |error| 0.0210 nats.
- One-sided paired t on |err_S| − |err_N|: p = 0.9992. All three frozen gates fail.

**Interpretation.** The exploratory r = −0.99 was a small-sample artifact of the selection
cells: intrinsic_dimension is strongly position-degenerate — identical values for segments
ending at the same block (e.g. 2@5, 3@4, 4@3 all = 12.64) — while measured Δ_U at those cells
differs by up to 0.03 nats with tight CIs. The statistic tracked span within the three selection
segments but carries no position information, and position matters. Nominal span alone, the
baseline H2 was meant to beat, predicts held-out cells respectably (R² = 0.455).

H4 (variance beyond span+position): descriptive fit is degenerate (4 parameters on 6 selection
points; R²_oos = −11.09) — no evidence; flagged low-power per the freeze.

## Operational note (handled by the frozen missing-data rule)

Seed 33 collided with the P2 cohort's `all_local-s33` manifest/run-id. The driver skipped the
p2c all_local-s33 (a completed manifest with the same run id existed from P2); the analyzer
flagged the protocol-version mismatch as an operational failure and dropped seed 33 from every
cell mean (n=7), exactly per the frozen rule — never outcome-based. Because all_local is
deterministic given seed and fixed protocol, the collision is benign in substance, but the
tooling lesson is recorded: run ids must be protocol-qualified in future studies (candidate
amendment for P3).

## Consequences for the research program

- C2 stays **hypothesis only**. The exploratory near-perfect correlation did not survive
  out-of-sample confirmation — the exploratory/confirmatory wall worked as designed, and this
  is exactly the kind of result the previous project would have reported as a finding.
- Position effects exist: cells with equal span differ significantly by location (e.g. 2@1
  0.0390 vs 2@5 0.0346; 5@1 0.1080 vs 5@3 0.0725; CIs non-overlapping). A future predictor must
  encode position, not just segment content pooled over the span.
- The Δ_U surface over 8 positions/spans at the frontier boundary (width 256, q=8) is now
  measured with 7–8 seeds per cell — a reusable dataset for any future predictor freeze.

## Ledger update

C2 remains hypothesis-only with the stronger note that one frozen candidate failed
out-of-sample; per-cell Δ_U table becomes part of the shared evidence base.
