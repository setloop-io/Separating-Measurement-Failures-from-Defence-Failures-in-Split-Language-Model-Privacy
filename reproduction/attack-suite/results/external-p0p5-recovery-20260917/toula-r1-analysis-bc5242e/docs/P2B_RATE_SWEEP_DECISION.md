# P2b rate-sweep decision record

Date: 2026-09-04
Protocol: `experiments/configs/rate_sweep_matrix.yaml` (`p2b-rate-sweep-1.0`, frozen 2026-09-04
before any run; preregistration `docs/PREREGISTRATION_P2B.md`).
Cohort: **confirmatory**, fresh seeds [7, 17, 27, 37, 47, 57, 67, 87]; calibration seed 919
reused (representation-matched, disjoint).
Artifacts: `outputs/rate_sweep/` (104 runs), `rate_sweep_analysis.json`, `rate_sweep_summary.md`,
`privacy_scores.json`, `paper/figures/rate_sweep_frontier.png`.
Execution: 104/104 runs completed, 0 operational failures, 0 retries. (First driver process was
killed by its 6 h wrapper timeout after run 77 and resumed via completed manifests; every run is
backed by a manifest and artifact hashes. Recorded for audit.)

## Design

Span fixed at medium (4 blocks @ position 2), width fixed 256 — the P2-confirmed feasible cell.
Only the boundary quantization dose varies: q_bits ∈ {4, 8, 16, 32}, moving measured operational
rate (4096 → 32768 bits/token) with architecture, data, budget, and convergence rule unchanged.

## Primary verdict — H3 rate response: **CONFIRMED, with saturation**

- Per-seed OLS slopes of trained eval loss on q_bits: mean −0.000711 nats per q-bit level,
  95% CI [−0.000834, −0.000589], t(7) = −13.7, one-sided p ≈ 3e-7 < 0.05.
- Practical effect: mean q4→q32 reduction 0.0295 nats ≥ 0.01 threshold. All 8 seeds negative
  slope; Spearman sensitivity −0.8 to −1.0 per seed.
- **Shape: the entire response lives between q=4 and q=8** (1.7044 → 1.6751); q=8→16→32 are
  identical to 4 decimal places. The correct language is a **mechanism-specific surrogate
  plateau above ~8 bits/dimension**, not a monotone exchange rate. No constant bits-per-loss
  ratio exists or was claimed.

## Secondary — causal gain at every rate (one-sided paired t, α=0.0125 Bonferroni)

| q_bits | bits/token | mean G_R (nats) | 95% CI | p | confirmed |
|---:|---:|---:|---|---:|:--:|
| 4 | 4096 | **0.0264** | [0.0205, 0.0324] | 1e-5 | yes |
| 8 | 8192 | 0.0223 | [0.0187, 0.0259] | <1e-6 | yes |
| 16 | 16384 | 0.0222 | [0.0186, 0.0258] | <1e-6 | yes |
| 32 | 32768 | 0.0222 | [0.0186, 0.0258] | <1e-6 | yes |

Causal remote-learning gain persists even at 4-bit quantization — the strongest gain, in fact —
and is flat across q ≥ 8. The P2 medium-high result replicates on disjoint seeds (q=8 here:
0.0223 vs 0.0167 in P2, overlapping CIs, same direction).

## Privacy per rate (token-inversion, width-matched calibration, threshold ≤ 0.02)

All four cells pass; per-seed advantages all negative (−0.087…−0.135), **including q=32
(unquantized float passthrough)**. Scoped reading: the width-256 bottleneck plus trained
encoder defeats the per-step token-inversion probe even without quantization. This does NOT
certify privacy: membership, property, semantic, and history-accumulation attacks are
unevaluated, and the probe is one architecture on one probe batch. The omission must accompany
any citation.

## Frontier (feasible = confirmed gain AND privacy pass)

All four rate points are feasible. Empirical delegability frontier for this mechanism:
utility 1.7044 → 1.6748 held-out loss over 4096 → 8192 bits/token, then flat to 32768;
Δ_U vs all-local 0.0951 (q4) → 0.0656 (q≥8) nats; removed-remote reference tracks Δ_U closely
at q ≥ 8 (0.0643) but not at q4 (0.0732 < 0.0951), i.e. at the lowest rate the trained remote
recovers about a third of the representational penalty that an identity bypass cannot.

## Ledger consequences

- C4 (rate response improves utility envelope): **confirmed in scoped, saturated form** —
  response exists over [4096, 8192] bits/token and plateaus; no universal exchange rate.
- C3 (frontier exists): strengthened from single-cell to a **4-point feasible frontier segment**
  at (medium span, width 256), with uncertainty bands.
- C1: strengthened by disjoint-seed replication at q=8 (direction and magnitude consistent).

## Not established

H2/H4 (segment-statistic predictors) remain untested; a single span/width was swept, so the
plateau's location may move with span, width, capacity, or defense type; privacy is
single-family; no second architecture or corpus (P3) has run.
