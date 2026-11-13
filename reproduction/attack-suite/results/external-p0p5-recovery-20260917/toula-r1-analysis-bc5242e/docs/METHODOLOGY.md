# Scientific methodology

Status: DRAFT — subordinate to `RESEARCH_PLAN.md` and `PREREGISTRATION.md`.
This document states *how we reason*, given that the previous attempt failed to prove the
concept. It does not restate the full protocol; it defines the inferential logic that the
protocol must satisfy.

## 0. Posture after the failed attempt

The prior project failed because it confused *placement* with *contribution* and *estimates*
with *certificates*:

- A never-trained-remote ablation retained nearly all of the reported "cloud contribution" —
  nominal layer placement did not isolate useful remote learning.
- The 47.4% headline was single-seed, marginal on utility, and had no valid complete-view
  privacy measurement.
- Near-zero MINE estimates were misread as mutual-information upper bounds.
- "Capacity ceiling", "4:1 exchange rate", "delegation triangle", and "depth is delegability"
  were asserted, never established.

Therefore this methodology is built on three rules:

1. **Every claim needs a causal contrast, not an observation.** A number is evidence only
   relative to a frozen control (frozen-random remote, removed remote, all local).
2. **Nothing is proven; hypotheses survive or die.** We estimate a feasible set with
   uncertainty. Existence claims (H5) are the strongest claim type we allow, and even they
   are mechanism-specific.
3. **The exploratory/confirmatory wall is load-bearing.** Anything seen during exploration
   is hypothesis-generating only. Confirmatory estimates come from fresh seeds under a frozen
   protocol.

## 1. Question and estimands

Primary question: what determines the remote substitutability of a contiguous LM segment
under a fixed privacy-constrained boundary?

Estimands (all continuous, all seed-level):

- \(\Delta_U(c)\) — replacement penalty vs all-local;
- \(G_R(c)=L_{\text{frozen remote}}-L_{\text{trained remote}}\) — causal remote-learning gain;
- \(R_P(c)=\max_a A_a(c)/\tau_a\) — worst attack advantage over matched null, normalized by
  an independently calibrated threshold (dimensionless ratios only; never raw max across
  heterogeneous metrics);
- \(M_R(c)\) — measured remote FLOPs, bytes, time, memory.

## 2. Falsification logic per hypothesis

| Hypothesis | Proof standard | What kills it |
|---|---|---|
| H1 remote learning is causal | \(G_R>0\) beyond a predeclared practical minimum, replicated across seeds, equal optimization budget, resource accounting confirms remote work | Any paired seed where trained ≈ frozen, or gain vanishes under the convergence rule |
| H2 segment statistic beats layer count | Predeclared statistic beats nominal count out-of-sample | Frozen statistic fails to beat count on held-out seeds |
| H3 rate response exists | Utility improves with measured bits/token at fixed privacy risk, equal convergence | Flat response after comparable optimization |
| H4 statistics explain variance beyond depth | Added out-of-sample variance after adjusting for location/rate/compute | No incremental predictive value (predictive test, **not** mediation) |
| H5 feasible frontier exists | ≥1 config with nontrivial \(G_R\) passing utility + complete-transcript privacy | Empty feasible set is a publishable null result |

Key asymmetry: the pilot can only *kill* the study, never *prove* it. Proof (in the weak,
empirical sense above) requires the separately powered confirmatory cohort.

## 3. Stage gates

- **P0 instrument qualification** — enumerate the complete remote-visible transcript; qualify
  every attack/metric pair with matched nulls and positive-control dose curves; verify the
  scorer detects a known leak at a predeclared dose; validate FLOP/byte instrumentation on a
  hand-computable case; verify remote parameters update in trained cells and are bit-identical
  in frozen cells. Exit: all controls resolve in the predicted direction *blind to pilot
  outcomes*.
- **P1 go/no-go pilot** — 3 paired seeds × span × width, trained vs frozen-random vs removed
  vs all-local, with learning curves. Go requires positive gain in all 3 seeds in ≥2 cells
  chosen by a pre-frozen rule, a feasible power calculation, persistence under convergence,
  verified remote resource use, and ≥1 privacy-eligible cell. Otherwise: report the causal
  null and stop. This is the direct fix for the prior failure mode.
- **P2 response surface** — fractional factorial (span, location, width, measured rate,
  capacity, noise), ≥5 fresh seeds per frontier cell, hierarchical model, uncertainty-banded
  feasible set. Configuration selection happens in a nested tuning cohort; the same data never
  both selects and confirms.
- **P3 decomposition** — equal-example vs equal-compute curves; decompose \(\Delta_U\) into
  representation, approximation, and optimization error.
- **P4 replication** — second architecture, second corpus, downstream eval, fresh seeds.
  Cross-model anecdotes are not evidence.

## 4. Privacy methodology (the prior attempt's weakest point)

- The privacy claim covers only the **declared complete transcript**: forward values, inbound
  and outbound gradients, evolving remote state, per-step and accumulated history, joint views.
  Every omitted channel is named in the abstract, threat model, captions, and limitations.
- Each attack family gets a matched null and an independently sampled,
  representation-matched positive-control dose curve, calibrated on data/seeds disjoint from
  confirmation.
- No estimator output (e.g. MINE) is ever interpreted as an MI bound or privacy certificate.
- Timing enters a privacy claim only under a separately frozen clock/noise model with matched
  timing controls.
- Multiplicity correction across attack families is predeclared.

## 5. Statistical methodology

- Inferential unit = independent training run/seed; tokens/frames are nested observations.
- Hierarchical model over seed-level outcomes (segment, location, width, rate, capacity,
  architecture).
- Simulation-based power analysis before confirmatory execution; ~10 seeds near any claimed
  frontier boundary.
- No outcome-based exclusions; operational exclusions prespecified.
- Feasible set reported as a Pareto surface with uncertainty bands, never as a single "best"
  point selected and confirmed on the same data.

## 6. How far the concept can be pushed — and no further

In ascending order of difficulty, with the stopping rule for each:

1. **Mechanism-specific existence** (H1, H5): "under this boundary, this segment, trained
   remote work causally helps and survives the declared privacy instrument." Achievable if the
   pilot's go criteria pass. This is the realistic ceiling of "proof".
2. **Predictive law within a family** (H2–H4): a frozen segment statistic predicts
   substitutability better than depth, out of sample, within one architecture family.
3. **Generalization** (P4): the law transfers to a second architecture/corpus under matched
   budgets.
4. **Prohibited**: information-theoretic ceilings, universal exchange rates, causal depth
   laws — unless a future contribution supplies an explicit adversary/channel model and a
   proved lower bound. Until then the honest terms are *empirical delegability frontier* and
   *mechanism-specific surrogate plateau*.

A documented causal null at P1, or an empty feasible set at P2, is a successful execution of
this methodology — it is the answer the previous attempt refused to accept.

## 7. Evidence chain

Every manuscript number must resolve through `docs/CLAIM_EVIDENCE_LEDGER.md` to a committed
manifest and a generated summary. No ledger entry, no claim.
