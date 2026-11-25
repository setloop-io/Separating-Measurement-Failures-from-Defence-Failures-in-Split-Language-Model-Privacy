# Preregistration — p2c-segment-statistics-1.0

Status: **SELECTION RULE FROZEN 2026-09-04, before any segment statistic is measured.**
The confirmatory half freezes separately (below) before any confirmatory run.

Amendment chain: p1-pilot-0.1 → p2-confirmatory-1.0 → p2b-rate-sweep-1.0 →
**p2c-segment-statistics-1.0** (H2/H4: functional segment statistics as predictors of
substitutability).

## Scientific question

H2: one primary segment statistic, frozen before confirmation, predicts the replacement penalty
\(\Delta_U\) better than nominal layer count alone, out of sample.
H4 (secondary, predictive-attenuation only, NOT causal mediation): the statistic explains
out-of-sample variance beyond span and position.

## Stage 1 (exploratory) — measurement and selection

- Statistics are computed on a calibration all-local backbone (calibration cohort, seed 929,
  2000 steps, identical fixed protocol to prior studies). No split is trained for measurement.
- Candidate statistics (per contiguous segment [start, start+span) of the 8-block backbone),
  measured on 64 fixed probe batches from the held-out region (dedicated measurement stream):
  1. **residual-update magnitude**: mean over blocks in segment of ‖h_out − h_in‖₂ / ‖h_in‖₂;
  2. **block influence**: mean over blocks of 1 − cos(h_in, h_out) (per-token, averaged);
  3. **representation similarity**: linear CKA between segment input and segment output;
  4. **layer-drop sensitivity**: increase in probe-batch loss when all blocks in the segment
     are bypassed by identity (single forward pass, no retraining);
  5. **intrinsic dimension**: participation ratio of segment-output activations,
     (Σλ)² / Σλ² over per-token covariance eigenvalues.
- Selection data: exploratory Δ_U means from P1 (seeds 101–303) and P2 (seeds 11–88) cells:
  six (segment, width) cells at q=8 — {2@3, 4@2, 6@1} × {64, 256}.
- **Selection rule (frozen now):** the primary statistic is the candidate with the highest
  absolute Pearson correlation with mean Δ_U across the six exploratory cells. Ties broken by
  the higher correlation within the width-256 subset. The selected statistic, its measured
  values, and the selection table are recorded in the p2c confirmatory config freeze; the
  selection cannot change afterward.

## Stage 2 (confirmatory) — out-of-sample predictive test

- **Held-out cells (8, frozen):** segments (span@start): {2@1, 2@5, 3@2, 3@4, 4@3, 5@1, 5@3,
  6@2} — none identical to a selection cell; boundary width 256, q_bits=8 (the frontier-plateau
  point), trained mode. Controls: all_local per seed. 8 cells × 8 seeds + 8 all_local = 72 runs.
- **Seeds (fresh, frozen):** [3, 13, 23, 33, 43, 53, 63, 83]; pairing `pb-p2c-{seed}`;
  cohort `confirmatory`; protocol version `p2c-segment-statistics-1.0`.
- Target: per-cell mean Δ_U = mean over seeds of (trained eval loss − all_local eval loss) at
  step 2000, fixed convergence.
- **Predictors (frozen):** model S: Δ_U ~ a + b·(selected statistic); model N: Δ_U ~ a + b·span.
  Both fit by OLS on the six exploratory selection points only. Confirmatory cells never enter
  the fit.
- **Primary test (frozen):** per-cell absolute out-of-sample prediction error |ŷ−y| under model
  S vs model N; one-sided paired t-test over the 8 cells (S better), α=0.05. H2 CONFIRMED iff
  p < 0.05 AND mean absolute error of S < mean absolute error of N AND out-of-sample R² of S > 0.
- **Secondary (H4, descriptive):** out-of-sample R² of Δ_U ~ span vs Δ_U ~ span + position +
  statistic (the latter fit flagged low-power, reported without gating); per-cell residuals.
- Missing data: ≤1 retry per run, identical cell identity; a cell missing its all_local partner
  seed drops that seed from the cell mean (predeclared, never outcome-based).

## Scope notes

- Statistics derive from ONE calibration backbone (seed 929): they characterize the family
  attractor, not each seed; this limitation is recorded and must accompany citations.
- Prediction targets are replacement penalties, not causal gains; G_R is reported per cell
  descriptively (trained vs frozen_random is NOT run in p2c — medium-high gain is already
  confirmed; new cells measure Δ_U only). If a future amendment wants causal gain at p2c cells,
  it must be a separate freeze.
- Privacy is not evaluated in p2c (no new boundary regimes: width 256 / q=8 was scored in P2/P2b).
