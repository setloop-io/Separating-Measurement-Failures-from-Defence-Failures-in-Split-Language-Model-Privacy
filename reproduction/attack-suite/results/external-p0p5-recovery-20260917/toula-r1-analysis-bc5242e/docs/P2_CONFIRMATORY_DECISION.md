# P2 confirmatory decision record

Date: 2026-09-04
Protocol: `experiments/configs/confirmatory_matrix.yaml` (`p2-confirmatory-1.0`, frozen
2026-09-04 before any confirmatory run; preregistration `docs/PREREGISTRATION.md`).
Cohort: **confirmatory**, fresh seeds [11, 22, 33, 44, 55, 66, 77, 88]; calibration seeds
909 (width 64) and 919 (width 256), disjoint.
Code revision: git `be32144570e6fdb6cbdbdf367c9e71ffa685d24a` + dirty-patch hashes recorded in
per-run manifests (`experiments/manifests/`).
Artifacts: `outputs/confirmatory/` (129 runs), `confirmatory_analysis.json`,
`confirmatory_summary.md`, `privacy_scores.json`, `paper/figures/confirmatory_learning_curves.png`.
Execution: 129/129 runs completed, 0 operational failures, 0 retries needed. (The runner's shell
wrapper exited 1 because `tee` could not create its log file at startup; the experiment process
completed all runs and wrote all manifests — recorded here for audit.)

## Primary family (frozen rule: one-sided paired t, α=0.025 Bonferroni, mean ≥ 0.01 nats,
## persistence at steps 1800/1900/2000)

| Cell | n | mean G_R (nats) | SD | t(7) | p (1-sided) | 95% CI | Verdict |
|---|---:|---:|---:|---:|---:|---|---|
| medium-high (span 4 @ 2, width 256) | 8 | **0.0167** | 0.0032 | 14.62 | 3e-7 | [0.0140, 0.0194] | **CONFIRMED** |
| long-high (span 6 @ 1, width 256) | 8 | 0.0093 | 0.0104 | 2.55 | 0.0192 | [0.0007, 0.0180] | **NOT CONFIRMED** (mean < 0.01 threshold; 2/8 seeds negative) |

Wilcoxon sensitivity (not gated): medium-high p=0.0071; long-high p=0.0294 — consistent.

## Secondary family (two-sided paired t, α=0.0125 Bonferroni)

| Cell | mean G_R | p (2-sided) | significant |
|---|---:|---:|---|
| short-low | +0.0006 | 0.284 | no |
| short-high | +0.0074 | 0.0435 | no |
| medium-low | +0.0016 | 0.0132 | no |
| long-low | **−0.0093** | 0.0020 | **yes — negative** |

Two-way ANOVA on seed-level G_R: span F(2,42)=8.94 p=5.8e-4; width F(1,42)=58.3 p=4e-9;
span×width F(2,42)=3.97 p=0.026. The causal value of training the remote module depends on
both span and boundary width, with a confirmed **negative** effect at long span + narrow width.

## Privacy (token-inversion family; width-matched calibration; frozen threshold advantage ≤ 0.02)

Both instruments (widths 64 and 256) QUALIFIED. All six cells pass: per-seed advantages are all
negative (range −0.05…−0.13, i.e. below the matched null) at measured 2048 (width 64) and
8192 (width 256) bits/token. Scope: token-inversion on per-step quantized latents only;
membership, property, and history-accumulation attacks remain unevaluated — this omission must
accompany any citation.

## H5 — privacy-conditioned feasible frontier: **SUPPORTED (one cell)**

medium-high is primary-confirmed (G_R = 0.0167 nats, CI excludes 0 and the 0.01 practical
threshold) AND passes the frozen privacy criterion. The feasible set is therefore non-empty
**for this mechanism, backbone, corpus, and boundary**: a 4-layer segment at position 2 of 8
behind a 256-wide 8-bit-quantized boundary. Its replacement penalty Δ_U ≈ 0.072 nats vs
all-local remains positive — delegation is not free.

## What is confirmed, and what is not

- **Confirmed (H1, scoped):** trained remote computation contributes causally and materially
  beyond a frozen random transformation at the medium-high cell, under matched optimization
  budgets and verified pairing. Ledger claim C1 may be updated with the scoped language below.
- **Refuted as a general pattern:** the gain does not appear at short spans or narrow widths,
  and is significantly negative at long-low. Width and span jointly moderate the effect.
- **Not established:** long-high failed the practical-effect gate; H2/H3/H4 (segment
  statistics, rate response, variance explained) were not tested by this protocol; no
  second architecture or corpus (P3 replication) has run.
- Ledger C3 (frontier exists) receives single-cell confirmatory support; the frontier's
  *shape* remains unestimated.

## Scoped manuscript language now permitted (per ledger update)

"In an 8-layer byte-level backbone on tiny-shakespeare, training a remote surrogate for a
4-block segment behind a 256-wide 8-bit-quantized boundary reduced held-out loss by 0.0167 nats
(95% CI [0.0140, 0.0194], n=8 independent seeds, one-sided paired t p≈3e-7) relative to a paired
frozen-random remote with identical optimization budget, while a calibrated token-inversion
attack on the declared transcript showed no advantage over a matched null. At a 64-wide boundary
the same training had no detectable benefit, and for a 6-block segment it was significantly
harmful (−0.0093 nats, p=0.002)."

## Next steps (not part of this protocol)

P2b: estimate the frontier shape (rate sweep at medium span; finer widths) under a new freeze.
P3: replication on a second architecture and corpus. H2–H4 require the segment-statistic
predictor program. All require fresh seeds and new frozen amendments.
