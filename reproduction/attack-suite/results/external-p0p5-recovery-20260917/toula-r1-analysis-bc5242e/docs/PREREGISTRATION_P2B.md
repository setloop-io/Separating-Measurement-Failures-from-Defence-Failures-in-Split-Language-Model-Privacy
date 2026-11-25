# Confirmatory preregistration — p2b-rate-sweep-1.0

Status: **FROZEN 2026-09-04, before any run under this protocol.**

Amendment chain: p1-pilot-0.1 (exploratory) → p2-confirmatory-1.0 (H1 confirmed at
medium-high; H5 single-cell support) → **p2b-rate-sweep-1.0** (this study: H3 rate response and
frontier shape around the confirmed cell).

## Study identity

- Protocol version: p2b-rate-sweep-1.0
- Config: `experiments/configs/rate_sweep_matrix.yaml` (protocol_status: frozen_for_confirmation)
- Freeze timestamp: 2026-09-04
- Git commit / environment digest: recorded per-run in manifests

## Design rationale

P2 confirmed causal remote-learning gain only at (medium span, width 256). P2b holds span and
width fixed at that cell and varies the boundary quantization dose q_bits ∈ {4, 8, 16, 32},
which varies the *measured operational rate* (serialized bits/token) without changing latent
width, architecture, data, or optimization. Width is therefore not confounded with rate in this
study — width is fixed; only serialization density changes.

## Hypotheses

- Primary (H3): held-out utility of the trained mode improves as measured bits/token increase,
  at fixed width and equal convergence (fixed 2000 steps, no early stop).
- Secondary: per-q causal gain G_R; per-q replacement penalty Δ_U; removed-remote reference;
  per-q privacy; the feasible frontier as a set of (bits/token, utility) points with
  uncertainty.

## Experimental units and seeds

- Inferential unit: independent training run/seed.
- Seeds (frozen, fresh, disjoint from all prior cohorts): **[7, 17, 27, 37, 47, 57, 67, 87]**.
- Runs: per seed — all_local; removed_remote at each q_bits; trained + frozen_random at each
  q_bits. Total 8 × (1 + 4 + 8) = 104 runs.
- Calibration: seed 919 width-256 artifact reused (representation-matched); probes for scoring
  retrained with calibration latents quantized on-the-fly to each cell's q_bits.

## Decision thresholds (frozen)

- H3 confirm iff: one-sided t over 8 per-seed slopes (eval loss vs q_bits) has p < 0.05 **and**
  mean eval-loss reduction q=4 → q=32 ≥ 0.01 nats.
- Per-q gain: one-sided paired t, Bonferroni α = 0.0125 over 4 cells; persistence reported.
- Privacy: token-inversion advantage ≤ 0.02 per cell (width-matched calibration).
- Frontier-feasible cell: confirmed positive G_R **and** privacy pass. Frontier reported as
  (bits/token, eval loss) points with 95% CIs.
- Missing data: ≤1 retry, identical cell identity; seed dropped from affected paired tests only
  (predeclared, never outcome-based). No outcome-based exclusions.

## Analysis

- Slope per seed by OLS of step-2000 eval loss on q_bits ∈ {4,8,16,32}; slopes tested across
  seeds (one-sided t, df = n−1).
- Paired contrasts within `pb-p2b-{seed}`; multiplicity Bonferroni as above.
- Sensitivity: Spearman rank correlation of loss vs q_bits per seed, reported not gated.
- All statistics from `scripts/stats_utils.py` (validated against known reference values).

## Freeze rule

Changes require a versioned amendment (p2b-rate-sweep-1.1, …) written before inspecting affected
outcomes. P2/P1 results may be cited as motivation and for design; no data pooling across
protocols.
