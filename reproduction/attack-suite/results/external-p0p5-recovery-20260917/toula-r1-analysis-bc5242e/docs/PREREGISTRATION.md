# Confirmatory preregistration — p2-confirmatory-1.0

Status: **FROZEN 2026-09-04, before any confirmatory run under this protocol.**

This instance freezes the P2 confirmatory study. The exploratory P1 pilot
(protocol p1-pilot-0.1, seeds 101/202/303) selected the headline cells and the
effect-size estimate; no pilot data is pooled into confirmatory estimates.

## Study identity

- Protocol version: p2-confirmatory-1.0
- Config: `experiments/configs/confirmatory_matrix.yaml` (protocol_status: frozen_for_confirmation)
- Git commit: recorded per-run in each manifest (`git rev-parse HEAD` + dirty-patch hash)
- Container/environment digest: per-run sha256 of `uv pip freeze`
- Freeze timestamp: 2026-09-04
- Owners: study authors

## Confirmatory hypotheses

- Primary (H1): trained remote modules outperform frozen-random remote modules
  by ≥ 0.01 nats mean causal gain \(G_R = L_{\text{frozen}} - L_{\text{trained}}\)
  at headline cells **medium-high** (span 4 @ block 2, width 256) and
  **long-high** (span 6 @ block 1, width 256).
- Secondary: sign/magnitude of \(G_R\) at the four remaining cells; span × width
  interaction on seed-level \(G_R\); replacement penalty \(\Delta_U\) and
  removed-remote reference at all cells.
- Explicitly exploratory: any post-hoc slicing beyond the frozen analysis;
  energy; timing.

## Experimental units

- Inferential unit: independent training run/seed.
- Confirmatory seed list (frozen): **[11, 22, 33, 44, 55, 66, 77, 88]** — 8 seeds per cell.
- Calibration seeds (disjoint): 909 (width 64, pre-existing P1 artifact), 919 (width 256, new).
- Corpus/model sampling frame: byte-level tiny-shakespeare, first-90% train / last-10% held-out
  by byte offset; 8×256 decoder-only backbone, seeded init per run.

## Fixed protocol

As in `confirmatory_matrix.yaml` `fixed:` — identical to p1-pilot-0.1 except seed list and
pairing block prefix (`pb-conf-{seed}`). Optimizer AdamW lr 3e-4 cosine to 3e-5, 2000 steps,
fixed budget, no early stop. Boundary: 8-bit per-tensor affine quantization (STE). Surrogate:
single pre-LN block at latent width, head_dim 32, d_ff 4×width. Transcript schema v1.0.0.

## Primary outcomes

1. causal remote-learning gain \(G_R\) (primary);
2. held-out replacement penalty \(\Delta_U\);
3. token-inversion attack advantage over matched shuffled null, with width-matched calibration;
4. measured remote FLOPs, serialized bytes, wall time. Energy excluded (unsupported hardware).

## Decision thresholds

- Primary per cell: one-sided paired t-test over 8 seeds, α = 0.025 (Bonferroni over 2 primary
  tests); **confirm iff** p < 0.025 **and** mean \(G_R\) ≥ 0.01 nats **and** mean \(G_R\) > 0 at
  eval steps 1800, 1900, 2000 (persistence).
- Secondary per cell: two-sided paired t-test, α = 0.0125 (Bonferroni over 4).
- Interaction: two-way ANOVA on seed-level \(G_R\), factors span(3) × width(2); reported, not gated.
- Sensitivity: Wilcoxon signed-rank per primary cell; reported, not gated.
- Privacy criterion: token-inversion advantage ≤ 0.02 (calibrated null band, P0) at the cell's
  measured bits/token with width-matched calibration. **H5 support iff** a primary-confirmed cell
  also passes the privacy criterion.

## Analysis

- Model: per-seed paired contrasts (trained vs frozen within `pb-conf-{seed}`); seed-level
  outcomes only; frames/tokens are nested observations, never inferential units.
- Planned contrasts: trained vs frozen per cell; Δ_U = trained − all_local per seed;
  removed-remote reference per span.
- Multiplicity: Bonferroni as above; primary and secondary families gatekept separately.
- Missing/failed-run policy: ≤ 1 retry with identical cell identity; a seed missing a pair
  member is dropped from that cell's paired test (predeclared, never outcome-based).
- Outlier policy: no outcome-based exclusions; operational exclusions only as above.

## Pairing and cohort separation

- Block id `pb-conf-{seed}`; identical backbone/encoder/decoder/remote init, minibatch order,
  and trusted random streams within a pair; only remote-parameter updating differs; verified by
  parameter checksums recorded in manifests.
- Calibration seeds {909, 919} disjoint from confirmatory seeds; exploratory seeds {101, 202, 303}
  spent and disjoint.
- No tuning cohort in P2; configuration selection occurred in P1 by the frozen go rule.

## Freeze rule

After this freeze, changes require a versioned amendment (p2-confirmatory-1.1, …) written before
inspecting affected outcomes. Exploratory results must never be relabelled confirmatory.
