# Preregistration — p5-delegation-sweep-1.0

Status: **FROZEN 2026-09-07, before any run or scoring under this protocol.**

Amendment chain: … → p4b-privacy-requalification-1.0 (frontier empty under the broadened
qualified suite) → **p5-delegation-sweep-1.0** (the empirical delegation-triangle test).

External motivation: arXiv:2609.04382 ("Privacy Failure in Split-LLM Training") demonstrated
that (a) the returned-gradient channel is a privacy surface that forward-only instruments miss,
and (b) per-row gradient clipping + noising closed their channel for ≈0.01 nats. P5 attacks
both channels and includes a gradient-clip+noise defense axis inspired by that mitigation. Their
protocol elements (injected-leak positive controls, shuffled nulls, pre-fixed thresholds) are
the ones this project already follows.

## Scientific question

Is there ANY configuration in the swept space that simultaneously satisfies three frozen gates —
privacy, utility, and delegated-compute share? ("hold any two, never all three" is the triangle
claim being tested, as an empirical feasible-set statement within this mechanism family, not a
universal law.)

## Design

- Backbone: arch1 (8 layers, d_model 256), corpus1 (tiny-shakespeare); fixed optimization
  protocol identical to prior studies (2000 steps, batch 32, seq 128, fixed budget).
- **Delegation share**: span ∈ {1, 2, 3, 4, 5, 6, 7}, centered positions frozen as
  start = (8 − span) // 2 → spans at starts {3, 3, 2, 2, 1, 1, 0}. **Remote capacity matched to
  span**: the remote surrogate is `ceil(span/2)` stacked pre-LN blocks at width 256 (previously
  always 1 block), so delegated FLOP share grows with span. Width fixed 256.
- **Defenses** (boundary): D0 = q_bits 8, gradients untouched (reference; known leaky from P4b);
  D1 = q_bits 4 + gradient defense: per-position gradient vectors crossing the boundary in BOTH
  directions clipped to norm ≤ 1.0 then Gaussian noise σ = 0.1 added (straight-through in the
  backward graph; constants frozen arbitrarily-but-before-outcomes, sensitivity deferred to a
  later amendment).
- Modes: trained, frozen_random per cell; controls all_local per seed, removed_remote per
  span per seed (width 256; removed uses the same multi-block-capacity topology minus remote
  trainable params — identity per block).
- Seeds (fresh, frozen): [21, 31, 41, 51, 61, 71, 81, 91]; pairing `pb-p5-{seed}`.
- Runs: 7 spans × 2 defenses × 2 modes × 8 seeds = 224; +8 all_local + 7×8 removed = 288 total.

## Gates (frozen per cell; a cell holds a gate iff the criterion passes)

1. **Privacy**: for each qualified attack family × channel, the cell's worst per-seed advantage
   ≤ 0.02. Families: token-inversion (P0 instrument), sequence-property-v2, structure-semantic-v2
   (P4b instruments). Channels: forward latents AND returned (inbound) gradients. Gradient-channel
   variants of families 1–2 must qualify on fresh calibration captures with the same frozen floors
   (property r-advantage ≥ 0.10, structure ≥ 0.02, |null| ≤ 0.02); a channel-variant that fails
   qualification is DISQUALIFIED and named — the privacy gate then covers the remaining qualified
   variants only, and the narrowing is declared.
2. **Utility**: mean Δ_U = L(trained) − L(all_local) ≤ 0.10 nats with the upper 95% CI ≤ 0.12.
   (Chosen from the P2/P2b Δ_U distribution ≈ 0.02–0.16 before any P5 run.)
3. **Delegated share**: measured analytic remote FLOP fraction (remote fwd+bwd ÷ total fwd+bwd,
   per the validated instrumentation) ≥ 0.15.

## Calibration runs (calibration cohort, fresh)

Gradient-channel attack qualification captures: trained mode, medium span 4@2, width 256,
600 steps, dense 4-channel probe captures, seeds {9391, 9491}, at D0 and D1 (4 runs). Forward
channels reuse existing qualified instruments/calibrations (919; P4b dense captures).

## Analysis (frozen)

- Per cell: per-seed Δ_U with 95% CI; G_R = L(frozen) − L(trained) reported descriptively
  (causal gain is established; here it contextualizes utility); privacy per family × channel;
  measured remote FLOP share.
- **Triangle verdict**: report the set of cells passing all three gates, with per-gate numbers
  always visible. Language frozen: if empty — "within the swept space, defense options, and
  attack suite, no tested configuration holds all three gates"; if non-empty — "cells {…} hold
  all three gates" with uncertainty. No universal-law language either way.
- Missing data: ≤1 retry, identical cell identity; seed dropped from affected tests only.

## Named omissions

History-accumulation and canary-membership families (instruments unqualified, P4b); timing;
optimizer state beyond remote module; active/adaptive attackers; natural-sequence membership;
cross-run joint attacks; arch2/corpus2 (single-setting sweep by design; replication deferred).

## Amendment p5-1.1 (2026-09-07, pre-cohort — no P5 cohort outcome inspected)

Instrument fixes discovered during gradient-channel qualification:
1. **Corrected r-metric null**: the P4b ridge-probe null evaluated on aligned test pairs; a
   corrected null that also permutes test-time alignment is used for P5 gradient-channel
   qualification and scoring. Direction of the P4a/P4b error is conservative (inflated nulls →
   shrunken advantages); prior family verdicts unchanged.
2. Gradient-channel probe training uses every 10th dense step (recorded as `step_stride` in
   `outputs/delegation/gradient_qualification.json`); floors unchanged.
3. Gradient features are quantized to the cell's q_bits before probing, matching the serialized
   transcript accounting.
4. Family-1 cell score aggregates as the max advantage over the three property targets
   (conservative).
5. Qualification result (pre-cohort): structure-semantic-v2 × inbound-gradients QUALIFIED at d0
   (adv +0.223, null 0.000), no gradient-channel variant qualified at d1 (no detectable signal —
   instrument outcome, not a certificate); the P5 privacy gate covers the forward trio plus
   structure-semantic-v2 × inbound-gradients at d0, narrowing declared.

## Freeze rule

Changes require a versioned amendment written before inspecting affected outcomes.
