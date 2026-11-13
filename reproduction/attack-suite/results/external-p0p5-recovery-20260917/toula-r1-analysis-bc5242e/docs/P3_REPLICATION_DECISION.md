# P3 external-replication decision record

Date: 2026-09-07
Protocol: `experiments/configs/replication_matrix.yaml` (`p3-replication-1.0`, frozen 2026-09-06
before any run; preregistration `docs/PREREGISTRATION_P3.md`).
Cohort: **confirmatory**, fresh seeds [9, 19, 29, 39, 49, 59, 69, 89].
Artifacts: `outputs/replication/` (96 runs), `replication_analysis.json`, `replication_summary.md`,
`downstream.json`, `paper/figures/replication_curves.png`.
Execution: 96/96 runs completed, 0 operational failures, 0 retries.

## Design

2×2-minus-one around the P2-confirmed corner: A2C1 (12-layer/d192 architecture, Shakespeare),
A1C2 (8-layer/d256, War and Peace), A2C2 (both new). Segment at the proportional medium position
(arch1: 4@2; arch2: 6@3), boundary width 256, q=8 — the frontier-plateau configuration.

## Verdicts (frozen gates: one-sided paired t, mean ≥ 0.01 nats, persistence)

| Setting | role | mean G_R (nats) | SD | t(7) | p | verdict |
|---|---|---:|---:|---:|---:|---|
| A2C2 | primary (α=0.05) | **0.0322** | 0.0068 | 13.42 | <1e-6 | **REPLICATED** |
| A2C1 | secondary (α=0.025) | **0.0463** | 0.0083 | 15.77 | <1e-6 | **REPLICATED** |
| A1C2 | secondary (α=0.025) | **0.0118** | 0.0057 | 5.92 | 2.9e-4 | **REPLICATED** |

All 24 pairs positive (8/8 seeds in each setting). The causal remote-learning gain replicates on
a deeper-narrower architecture, on a 3× larger corpus of different provenance, and on both
together. Effect magnitude varies by setting (0.012–0.046 nats) — expected, since settings are
not exchangeable; no pooling was performed.

## Descriptive findings

- Replacement penalty Δ_U remains positive everywhere (A2C1 0.037, A1C2 0.069, A2C2 0.045 nats):
  delegation costs utility in every setting tested; the gain is against the frozen-random
  control, not a free lunch over all-local.
- The removed-remote reference nearly matches Δ_U in A2C1/A2C2 (deeper architecture: mostly
  representational penalty), while in A1C2 the removed reference (0.059) undershoots Δ_U (0.069),
  leaving an optimization/component gap the trained remote does not close there.
- Downstream linear probes on final hidden states (A2C2): byte-property accuracies 0.985–1.000
  in both modes; trained−all_local gaps ≤ 0.006 with only is_vowel's CI excluding 0
  (+0.0058 [0.0038, 0.0077]). Descriptive only: the split representation retains token-level
  information needed by probes; no downstream harm detected at this sensitivity.

## Ledger consequences

- **C1 strengthens from "scoped to one mechanism" to "replicated across two architectures and
  two corpora"** — all under frozen preregistrations, paired controls, fresh seeds.
- C3 inherits the replication for the frontier's existence at the (width 256, q=8)
  configuration in the new settings (privacy itself was not re-scored; the token-inversion-only
  scope omission still applies and must be named).

## Not established

Privacy attacks on the new settings; segments other than the proportional medium; predictors of
effect magnitude across settings (the 0.012 vs 0.046 spread is unexplained); anything about
larger models. H2 remains not confirmed; H4 untestable at current design sizes.
