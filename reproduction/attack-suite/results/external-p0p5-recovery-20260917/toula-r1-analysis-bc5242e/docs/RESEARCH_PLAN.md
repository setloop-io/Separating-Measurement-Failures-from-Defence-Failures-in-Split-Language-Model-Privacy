# Research plan

> **R1 reset (2026-09-10): historical reference only.** This document describes
> pre-reset work. All associated observations and analyses are inadmissible for
> active scientific claims. The current study has no results; follow
> [the R1 plan](R1_RESEARCH_PLAN.md) and [active ledger](CLAIM_EVIDENCE_LEDGER.md).
> The reference tag preserves the original version of this document.

## 1. Primary question

What determines the remote substitutability of a contiguous language-model segment under a
fixed privacy-constrained boundary?

The study estimates a feasible set rather than assuming a triangular trade-off. For
configuration \(c\), define:

- replacement penalty \(\Delta_U(c)=L_{\mathrm{trained\ split}}-L_{\mathrm{all\ local}}\);
- causal remote-learning gain
  \(G_R(c)=L_{\mathrm{frozen\ remote}}-L_{\mathrm{trained\ remote}}\);
- privacy risk \(R_P(c)=\max_a A_a(c)/\tau_a\), where \(A_a\) is attack family \(a\)'s
  metric-specific advantage over its matched null and \(\tau_a\) is that family's independently
  calibrated threshold;
- measured remote resources \(M_R(c)\): FLOPs, bytes, time, memory, and energy where available.

The empirical feasible set contains configurations satisfying predeclared utility, privacy,
causal-contribution, and resource constraints. Nominal layer share is not an outcome. Raw values
from heterogeneous metrics are never maximized against one another; only dimensionless threshold
ratios are aggregated, and all per-family values remain visible.

## 2. Hypotheses

### H1 — remote learning is causal

Trained remote modules outperform frozen-random remote modules by a practically meaningful
amount after matching optimization budgets.

### H2 — segment properties beat layer count

One primary segment statistic, frozen before confirmation, predicts replacement penalty better
than nominal layer count alone. Candidate statistics—residual-update magnitude, block
influence, representation similarity, layer-drop sensitivity, and intrinsic dimension—are
compared exploratorily during the pilot.

### H3 — rate response exists after comparable optimization

Increasing measured operational boundary rate in bits per token improves held-out utility at
fixed measured privacy risk after each cell reaches the same convergence rule. Latent width is
a separate factor and is not called a bitrate.

### H4 — segment statistics explain variance beyond depth

After adjustment for segment location, operational rate, and compute, a predeclared segment
statistic explains additional out-of-sample variance beyond architecture depth. This is a
predictive attenuation test, not a causal mediation claim.

### H5 — useful remote work has a measurable privacy-conditioned frontier

There exist configurations with nontrivial causal remote-learning gain that satisfy both the
utility and declared-transcript privacy criteria. This is an empirical existence question, not
an assumed result.

## 3. Stage P0 — implementation and positive-control qualification

Before the pilot:

1. enumerate every value, gradient, state update, timing signal, and accumulated history visible
   to the remote process;
2. capture a schema-valid transcript from all-local, trained-remote, frozen-random, and
   removed-remote conditions;
3. qualify every attack-family/metric pair used for a claim with a matched null and an
   independently sampled, representation-matched positive-control dose curve;
4. verify that the scorer detects its intended leak at a predeclared dose;
5. validate FLOP/byte/time instrumentation against a small hand-computable case;
6. verify that remote parameters update in trained cells and remain bit-identical in frozen
   cells.

Exit criterion: all controls resolve in the predicted direction without inspecting pilot
outcomes.

## 4. Stage P1 — go/no-go causal pilot

### Fixed factors

- one backbone checkpoint and architecture;
- one tokenizer and frozen corpus split;
- fixed sequence length and batch construction;
- one boundary-defense implementation;
- one remote surrogate family;
- fixed optimizer, scheduler, attacker budget, and stopping rule.

### Crossed factors

- segment span: short, medium, long;
- latent width: low, high; operational bits/token is measured separately;
- remote mode: trained, frozen-random;
- three independent training seeds.

The trained and frozen conditions are paired by block identifier: encoder/decoder
initialization, minibatch order, data stream, stopping budget, and trusted-side random streams
are identical; only remote-parameter updating differs. Matched all-local and removed-remote
controls are run for every seed and segment location where they are not algebraically reusable.
“Removed remote” means a prespecified dimension-matched identity/projection bypass with no
remote trainable parameters. Learning curves are retained, not only terminal values.

### Go decision

The three-seed pilot is descriptive and variance-estimating; it cannot unlock a claim. Proceed
to a separately powered, fresh-seed confirmatory study only if:

1. the trained-minus-frozen gain is positive in all three paired seeds in at least two
   span/width cells selected by a rule frozen before pilot scoring;
2. the observed magnitude supports a feasible confirmatory power calculation against a
   practical-effect criterion chosen before confirmatory execution;
3. the gain persists under the convergence rule or long-horizon confirmation;
4. resource accounting confirms that the work occurs on the remote host; and
5. at least one candidate cell is eligible for the complete-view privacy evaluation.

If these conditions fail, report the causal null: the mechanism behaves primarily as a
trusted-side adapter plus remote fixed transformation. Do not launch the full matrix.

## 5. Stage P2 — within-family response surface

Use a fractional-factorial design followed by local refinement; do not blindly execute the full
Cartesian product. Candidate factors are:

- span length: 4, 8, 12, 16, 20 layers;
- span location: early, middle, late;
- latent width: 64, 128, 256, 1024 dimensions;
- operational rate: measured bits per token after serialization, quantization, sparsity,
  framing, and repeated releases, varied independently where the mechanism permits;
- remote capacity: small, medium, large;
- privacy/noise setting: calibrated low, medium, high;
- remote mode: trained, frozen-random, removed, unconstrained oracle;
- at least five independent seeds per confirmatory frontier cell.

The design must estimate span, rate, capacity, and their interactions without changing corpus,
optimizer, or training budget at the same time.

## 6. Stage P3 — convergence and causal decomposition

For candidate frontier points:

- report learning curves by steps and cumulative training FLOPs;
- compare equal-example and equal-total-compute regimes;
- run until a predeclared convergence criterion or maximum budget;
- include trained, frozen-random, removed-remote, all-local, and unconstrained-oracle controls;
- decompose replacement penalty into representation loss, approximation error, and optimization
  error where the controls permit.

## 7. Stage P4 — replication beyond one setting

Repeat the normalized design on:

- a second architecture from a matched family, preferably with controlled depth variants;
- a second corpus and at least one downstream evaluation;
- fresh confirmatory seeds chosen before execution.

A causal depth statement requires matched parameter/FLOP budgets and separation of segment
location from span length. Cross-model anecdotes are not sufficient.

## 8. Privacy evaluation

For every headline frontier cell, evaluate:

- forward activations and returned values;
- outbound and inbound gradients;
- evolving remote state;
- per-step and accumulated history;
- joint/cross-tensor views;
- token, sequence, rare-token, semantic, membership, and property attacks where scientifically
  applicable;
- matched shuffled or label-free nulls;
- positive controls matched to representation and headline boundary, using calibration data and
  seeds disjoint from confirmation.

Every omission narrows the claim and must appear in the abstract, threat model, table caption,
and limitations. Timing is logged for systems accounting; it enters a privacy claim only under a
separately frozen clock/noise model, matched timing control, repeat structure, and attack.

## 9. Statistical analysis

- Independent seed/run is the inferential unit.
- Rows and frames are nested observations.
- Primary outcomes remain continuous; pass/fail gates are secondary deployment classifications.
- Fit a hierarchical model over seed-level outcomes with segment, location, width, operational
  rate, capacity, and architecture as factors.
- Estimate the feasible set/Pareto surface with uncertainty bands.
- Predeclare multiplicity correction for attack families and hypothesis families.
- Use a simulation-based power analysis before confirmatory execution; increase toward ten seeds
  near a claimed reversal or frontier boundary.

Any “best achievable” configuration is selected inside a nested tuning cohort and evaluated on
held-out seeds or a held-out corpus according to a frozen selection rule. The same observations
cannot both select the envelope and provide its uncorrected confirmatory estimate.

Energy is a secondary systems measure available only on a predeclared supported-hardware
cohort; it is not a primary outcome and its availability cannot depend on observed utility or
privacy.

## 10. Theory boundary

“Information-theoretic ceiling” is prohibited unless a later contribution supplies an explicit
communication/adversary model, an operational rate definition, justified channel assumptions,
a task-specific rate-distortion or communication-complexity lower bound, and empirical scaling
that approaches the predicted bound. Otherwise the correct term is **empirical delegability
frontier** or **mechanism-specific surrogate plateau**.
