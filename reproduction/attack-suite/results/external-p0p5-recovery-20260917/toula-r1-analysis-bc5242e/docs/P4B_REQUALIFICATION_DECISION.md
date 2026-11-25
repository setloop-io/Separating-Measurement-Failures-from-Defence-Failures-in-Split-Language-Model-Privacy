# P4b privacy re-qualification decision record

Date: 2026-09-07
Protocol: `docs/PREREGISTRATION_P4B.md` (`p4b-privacy-requalification-1.0`, frozen 2026-09-07
before execution; redesigns traced to the P4a diagnoses).
Cohort: calibration (12 fresh runs; seeds 9390/9490/9590/9690; canary stream 9790) + scoring of
existing P2b frontier artifacts.
Artifacts: `outputs/privacy_calibration_p4b/`, `outputs/privacy_extended_p4b/privacy_extended_p4b.json`,
`privacy_extended_p4b_summary.md`.
Execution: 12/12 calibration runs completed, 0 failures. 159/159 tests green.

## Qualification outcomes (frozen gates)

| Family | Verdict | Key numbers |
|---|---|---|
| sequence-property v2 | **QUALIFIED** | punctuation-fraction r-advantage +0.715 (floor 0.10), null +0.003 ✓ (vowel +0.754 but null +0.028 outside band; newline +0.544, null +0.090 outside band — all reported) |
| structure/semantic v2 | **QUALIFIED** | advantage +0.293 (floor 0.02), null −0.014 ✓ — stronger probe lifts the P4a signal decisively past the null band |
| history-accumulation v2 | DISQUALIFIED | history advantage +0.102/+0.117 ≥ floor, BUT step-shuffled nulls +0.032/+0.039 breach the 0.02 band — the gain is partly non-temporal (larger feature set); the temporal component is not isolable by this instrument |
| canary membership v2 | DISQUALIFIED | q=32 AUC 0.569/0.464 (min 0.464 < 0.75 floor); nulls within bootstrap bands ✓ — the band machinery works; the attack is simply weak at these settings |

## Frontier survival under the broadened suite: **the frontier shrinks to empty**

Both qualified families detect leakage far above threshold at **every** frontier cell, including
q=4 (the heaviest quantization):

- sequence-property v2: max per-seed r-advantages +0.66…+1.64 (threshold 0.02);
- structure/semantic v2: max advantages +0.080…+0.191.

**All four frontier cells: NO.** `surviving_cells: []`. Per the frozen survival rule, the H5
support from P2/P2b is revised: the feasible set is **empty under the broadened qualified
suite**. This supersedes P4a's UNDETERMINED.

## Interpretation — the paper's central tension is now measured

- Token-level inversion does not leak (P0–P2b, all widths and q). Sequence-level properties
  (punctuation/vowel/newline structure) and next-byte structure classes leak strongly through
  the same boundary — even at 4-bit quantization and even though the latent is a 256-dim
  bottleneck. The leak is in *what the surrogate needs to compute*, not in raw token identity.
- The confirmed causal remote-learning gain (H1, replicated) and the privacy failure coexist in
  the same cells: **useful remote work and information leakage are not traded off here — the
  useful cells are exactly the leaky ones.** This is the honest, measurable version of the
  tension the previous project asserted with invalid instruments.
- This is not a privacy certificate failure of the defense parameters alone: the leak persists
  at the strongest quantization tested. Rate reduction (P2b axis) does not remove it.

## Consequences for the ledger

- C3 (frontier exists): **support WITHDRAWN under the broadened suite** — the feasible set is
  empty when property/structure attacks gate. Retained only under the explicit token-inversion-
  only scope, which is now known to be insufficient coverage.
- H5: revised to "no feasible cell under the qualified broadened instrument" — a confirmed
  negative at (medium span, width 256, q ∈ {4..32}).
- Named omissions (still open channels): history-accumulation and canary membership
  (unqualified instruments), timing, optimizer state, active/adaptive attackers, natural-
  sequence membership, cross-run joint attacks.

## What this opens (next studies)

- Defense redesign: add calibrated noise or bottleneck narrowing targeted at property/structure
  channels; re-run the P2b-style sweep under the qualified suite to find whether ANY feasible
  cell exists (the H5 existence question is now the live one).
- Rate–privacy mapping at finer q with the qualified families: where between "no leak"
  (imagined) and "leak" (observed ≥4 bits/dim) does the transition sit — or does it exist at all?
- History/membership instrument redesign remains open (diagnoses recorded).
