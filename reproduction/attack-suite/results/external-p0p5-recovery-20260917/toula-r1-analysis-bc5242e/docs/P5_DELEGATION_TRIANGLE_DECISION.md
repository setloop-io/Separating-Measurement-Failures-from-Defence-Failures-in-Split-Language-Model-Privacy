# P5 delegation-sweep decision record — the empirical triangle test

Date: 2026-09-08
Protocol: `docs/PREREGISTRATION_P5.md` (`p5-delegation-sweep-1.0` + pre-cohort amendment p5-1.1,
frozen 2026-09-07 before any cohort run). External motivation: arXiv:2609.04382 (returned-
gradient channel; clip+noise mitigation), incorporated as the D1 defense axis and the
gradient-channel attack requirement.
Cohort: **confirmatory**, fresh seeds [21, 31, 41, 51, 61, 71, 81, 91].
Artifacts: `outputs/delegation/` (288 runs), `delegation_analysis.json`, `delegation_summary.md`,
`gradient_qualification.json`, `paper/figures/delegation_triangle.png`.
Execution: 288/288 runs completed, 0 operational failures, 0 retries.
Analysis incident (recorded for audit): the first analysis run truncated the fractional
`remote_share` to int (share 0.000 everywhere); root-caused to the shared `collect_run` coercion,
fixed with regression tests, and the analysis was regenerated from the untouched artifacts. All
numbers below are from the corrected analysis.

## Verdict (frozen language)

**"Within the swept space, defense options, and attack suite, no tested configuration holds all
three gates."** Cells holding privacy + utility + delegated-share: **none**.

## The measured corner structure

| Cell (span @ start, defense) | privacy | utility | share ≥ 0.15 | G_R mean | Δ_U [95% CI] | share |
|---|---|---|---|---:|---|---:|
| s1@3 d0 | FAIL | pass | FAIL (0.123) | +0.004 | 0.025 [0.020, 0.030] | 0.123 |
| s2@3 d0 | FAIL | pass | FAIL (0.140) | +0.008 | 0.035 [0.028, 0.042] | 0.140 |
| s3@2 d0 | FAIL | pass | **pass** (0.280) | +0.028 | 0.034 [0.028, 0.039] | 0.280 |
| s4@2 d0 | FAIL | pass | **pass** (0.325) | +0.044 | 0.048 [0.039, 0.056] | 0.325 |
| s5@1 d0 | FAIL | pass | **pass** (0.488) | +0.084 | 0.061 [0.054, 0.069] | 0.488 |
| s6@1 d0 | FAIL | pass | **pass** (0.582) | +0.080 | 0.077 [0.072, 0.082] | 0.582 |
| s7@0 d0 | FAIL | FAIL (0.134) | **pass** (0.776) | +0.206 | 0.134 [0.127, 0.140] | 0.776 |
| all d1 cells | FAIL | FAIL (0.82–1.56) | pass for spans ≥ 3 | −0.17…+0.11 | 0.82–1.56 | 0.12–0.78 |

Corners: **utility+share** holds at d0 spans 3–6 (privacy fails); **privacy+anything** holds
nowhere — the privacy gate fails at every cell of both defenses. So not only is no triple
feasible; no pair involving privacy is feasible in the swept space.

## Privacy detail (qualified instruments only)

- Forward latents leak at every cell, both defenses: sequence-property r-advantage
  +0.95…+1.15, structure/semantic +0.22…+0.43 (threshold 0.02). Token-inversion never leaks,
  replicating P2/P2b.
- Returned-gradient channel (the arXiv:2609.04382 surface): structure-semantic-v2 qualified at
  d0 (positive control +0.223) and found **no leak there** (−0.22); at d1 no gradient-channel
  variant qualified (no detectable signal — an instrument outcome, not a certificate).
  Property and token-inversion gradient variants were disqualified (null-band breaches / no
  positive control). The leak therefore lives in the forward latents for this mechanism.
- The D1 gradient defense (q4 + clip 1.0 + noise 0.1) suppresses the *gradient* channel but
  does not close the *forward* leak, and costs 0.82–1.56 nats of utility — 8–15× over the
  frozen gate, and it destroys the causal gain at several spans (G_R negative at s2/s3/s4/s7
  under d1). This mitigation, successful in the cited system, fails here on both counts.

## Interpretation

The old paper's triangle survives contact with valid instruments in the strongest form we can
state empirically: in this mechanism family, over 7 delegation levels × 2 defenses × 8 seeds,
delegating more compute monotonically buys utility (G_R rises to +0.21 at span 7) and measured
FLOP share (to 0.78), but the privacy gate never passes anywhere — the forward bottleneck
preserves exactly the sequence-level structure the surrogate needs to be useful. The "deep
reason" (privacy from destroying information, delegation from preserving it) is now a measured
statement about *which* information classes pass the boundary, not a narrative.

Scope limits (named): arch1/corpus1 only; history/membership attack instruments unqualified;
timing, active attackers, natural-sequence membership, cross-run attacks unevaluated; defense
constants arbitrary-but-frozen (a calibrated-noise sweep might move D1's cost, an open
amendment).

## Ledger consequences

- X3 (delegation triangle) moves from "withdrawn/historical" to an **empirically supported
  feasible-set statement** with the frozen language above — within this mechanism family,
  defense set, and attack suite only.
- C3 remains withdrawn under broadened coverage; P5 strengthens the negative: no cell even
  holds privacy+utility.
- C1 gains descriptive G_R over the full span range (gain monotonic in span at d0, up to
  +0.206 nats at span 7 where utility fails).
