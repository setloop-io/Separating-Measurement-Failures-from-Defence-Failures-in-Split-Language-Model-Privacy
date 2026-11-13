# Calibration-cohort decision record: instrument qualification at the pilot regime

Recorded 2026-09-22. Evidence: campaign `qualification-20260922-a`, protocol
[p4-qualification-0.1](QUALIFICATION_PROTOCOL.md), 32 `p4-qualification-*` run
manifests (cohort calibration, `scientific_evidence: true`; per-seed execution
subsets 419101+419102 on toula, 419103+419104 on koula, CUDA deterministic
kernels, code commit `bc1f1d2` and analyzer revision `a29d15a`), the
analyzer-generated [report](../outputs/runs/qualification-20260922-a/report.md)
and [verification](../outputs/runs/qualification-20260922-a/verification.json).
Every manifest re-verified after merge (schema v2, source archive, artifact
digests, panels); all 28 split runs' candidate scores **recomputed** from saved
captures and matched. This is the instrument-qualification record required by the
pilot freeze. Candidate scores below are calibration-cohort exploratory
observations, not pilot outcomes and not defense results.

## Instrument verdicts (per target × cell; all four seeds' bindings must pass)

- **Next-byte structure × all 7 cells: qualified.** Every binding's 16 no-signal
  trials within ±0.02 and injected-code positives ≥ 0.02 floor.
- **Punctuation × 5 cells qualified** (d1-s0.1, d1-s0.5, d3-q1, d3-q2, d3-q3).
- **Punctuation × d0-q8 and × d1-s1.0-q8: not qualified** — each had exactly one
  binding with marginal no-signal excursions beyond the band (max |advantage|
  0.0217 in 2/16 trials for d0-q8/419104; 0.0245 in 1/16 trials for
  d1-s1.0-q8/419102); positive controls ≈ 1.0 in both. These are the measured
  ~1.5%-rate marginal estimator excursions seen in the engineering campaigns,
  not instrument breakage. The strict all-16-trials rule stands as frozen; the
  pilot freeze discloses this reliability evidence and the pilot's per-binding
  qualification verdicts decide coverage at execution.

## Calibration-cohort candidate scores (exploratory; 16,384-window panels)

Punctuation advantage (qualified bindings): d0-q8 0.854–0.898;
d1-s0.1 0.785–0.827; d1-s0.5 0.779–0.816; d1-s1.0 0.763–0.817;
d3-q1 0.803–0.851; d3-q2 0.803–0.825; d3-q3 0.836–0.863.
Next-byte structure advantage: d0-q8 0.094–0.104; d1 cells 0.094–0.105;
d3 cells 0.108–0.120 (descriptively above the d0 reference).

Read carefully: at this operating point the measured punctuation signal remains
enormous for **every** tested cell — the ridge probe refits in released
coordinates and finds the property regardless of rank-1 noise direction or 1-bit
precision. D1 σ=1.0 reduces the punctuation advantage by roughly 0.03–0.07 nats
of correlation versus d0; D3 does not reduce it at all (some seeds higher).
Structure advantages (~0.10) are likewise far above the 0.02 acceptance
threshold everywhere. Utility on the fixed panel: all_local 1.574–1.580;
d0 Δ_U-like +0.049–0.074; D1 adds ≤0.008; D3 reaches +0.087–0.120 depending on
q and seed (several above the proposed 0.10 utility gate value).

These are four calibration seeds' exploratory observations. The pilot's own
three-seed paired matrix with frozen/removed controls is the pre-registered test
of eligibility.

## Claim boundary

Records carry cohort `calibration` and `scientific_evidence: true`, scoped to
instrument qualification and calibration-cohort exploratory observation. No
defense-success, utility, causal-gain, or pilot eligibility claim is made here.
Unevaluated: returned values, quantizer metadata, gradients, remote/optimizer
state, timing, active/adaptive attackers, joint/cross-run attacks, history,
membership, semantics beyond five-class byte structure, token inversion. C1–C7
remain unestablished.
