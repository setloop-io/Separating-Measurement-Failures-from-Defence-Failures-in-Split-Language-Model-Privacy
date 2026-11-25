# P1 pilot go/no-go decision record

Date: 2026-09-03
Protocol: `experiments/configs/pilot_matrix.yaml` (`protocol_status: frozen_for_pilot`, version
`p1-pilot-0.1`)
Cohort: **exploratory** — hypothesis-generating only. Per `docs/RESEARCH_PLAN.md` §4, this pilot
cannot unlock a claim. No manuscript language changes result from this document.
Artifacts: `outputs/pilot/` (49 runs), manifests in `experiments/manifests/`, analysis in
`outputs/pilot/pilot_analysis.json`, `outputs/pilot/pilot_summary.md`,
`outputs/pilot/privacy_scores.json`, figure `paper/figures/pilot_learning_curves.png`.

## What was run

49/49 runs completed, 0 operational failures: per seed {101, 202, 303} — all-local; removed-remote
per span {short=2@3, medium=4@2, long=6@1}; trained and frozen-random per span × width
{low=64, high=256}, 2000 steps, paired by block id `pb-{seed}` (bit-identical backbone and remote
init, identical data stream; only remote-parameter updating differs). Calibration run seed 909
(disjoint). P0 self-tests (30 tests) passed before launch: parameter-update verification, pairing
bit-identity, FLOP hand-check, transcript/manifest schema validation, quantization byte exactness,
seed separation.

Pairing sanity confirmed empirically: trained vs frozen eval loss identical at step 0 (diff
0.000000 in all 18 pairs), diverging only as remote parameters update.

## Primary result: causal remote-learning gain G_R = L(frozen) − L(trained) at step 2000

| Cell | s101 | s202 | s303 | Mean | Qualifies (frozen rule) | Persists (1800–2000) |
|---|---:|---:|---:|---:|:--:|:--:|
| short-low | +0.0012 | −0.0025 | −0.0013 | −0.0009 | no | no |
| short-high | +0.0110 | +0.0213 | −0.0049 | +0.0091 | no | no |
| medium-low | −0.0027 | +0.0003 | +0.0011 | −0.0004 | no | no |
| medium-high | +0.0162 | +0.0143 | +0.0197 | +0.0168 | **yes** | yes |
| long-low | −0.0139 | −0.0168 | −0.0081 | −0.0129 | no (negative ×3) | — |
| long-high | +0.0061 | +0.0156 | +0.0326 | +0.0181 | **yes** | yes |

## Go-criteria evaluation (predeclared in RESEARCH_PLAN.md §4 and the frozen go rule)

1. **Frozen go rule** (G_R > 0 at final step for all three seeds in ≥ 2 of 6 cells): **PASS** —
   medium-high and long-high qualify.
2. **Power feasibility**: mean |G_R| ≈ 0.017 nats in qualifying cells with across-seed SD
   0.003–0.013; naive paired 80%-power n of 2–5 — a confirmatory cohort of ≥5 seeds/cell is
   feasible. **PASS** (descriptive).
3. **Persistence**: gain positive at eval steps 1800, 1900, 2000 in both qualifying cells.
   **PASS**.
4. **Remote resource accounting**: 18/18 trained runs show nonzero measured remote FLOPs
   (1.07e12 fwd at width 64; 1.40e13 at width 256) and serialized transcript bytes; work occurs
   on the remote module. **PASS**.
5. **Privacy eligibility**: instrument QUALIFIED (dose curve monotone, floor passed, null
   advantage 0.0000); eligible cells short-low, medium-low, long-low. **PASS** in the letter of
   the criterion.

## Decision: **GO** — proceed to a powered, fresh-seed confirmatory study.

## Conditions and caveats binding the confirmatory design

1. **The causal gain appears only at width 256, which is privacy-UNSCORED.** All width-64 cells —
   the ones where the token-inversion instrument verified no leak (advantage −0.07…−0.13 vs the
   matched null at 2048 bits/token) — show zero or negative gain. The confirmatory study MUST add
   a width-256-matched calibration cohort before any frontier (H5) language; otherwise the
   privacy-conditioned existence question stays open. As of this pilot, H5 is **not** supported:
   no cell simultaneously shows causal gain and a verified privacy evaluation.
2. **Long-low is consistently negative** (trained worse than frozen in 3/3 seeds, mean −0.013):
   at a narrow boundary, updating the remote module actively hurts. This is an exploratory
   interaction (span × width), not a confirmed effect; the confirmatory design must cross span
   and width and model the interaction.
3. **Replacement penalty Δ_U is positive everywhere** (0.006–0.160 nats) and grows with span;
   the removed-remote reference tracks most of it at long spans, meaning much of the long-span
   penalty is representational, not optimization failure.
4. Exploratory seeds {101, 202, 303} are spent. Confirmatory seeds must be fresh, the protocol
   amendment versioned, and the practical-effect threshold for G_R frozen before execution
   (candidate: 0.01 nats, justified by the observed qualifying-cell means).
5. Privacy results here cover only the token-inversion family on per-step quantized latents for
   one probe batch; membership, property, and history-accumulation attacks remain unevaluated.
   This omission must be named wherever these numbers are cited.

## Bottom line

The pilot survives its own kill conditions: H1 (causal remote learning) remains standing as a
hypothesis with exploratory support in two width-high cells. The concept is **not proven** — the
pilot by design cannot prove it — but the mechanism is not a trusted-side adapter plus remote
fixed transformation either. Proceed to P2 confirmatory with the conditions above.
