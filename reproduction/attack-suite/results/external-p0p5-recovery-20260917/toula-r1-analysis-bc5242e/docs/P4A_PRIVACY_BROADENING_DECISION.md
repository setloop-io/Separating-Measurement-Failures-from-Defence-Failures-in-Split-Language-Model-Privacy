# P4a privacy-broadening decision record

Date: 2026-09-07
Protocol: `docs/PREREGISTRATION_P4A.md` (`p4a-privacy-broadening-1.0`, frozen 2026-09-07 before
execution).
Cohort: calibration (12 runs: dense-capture ×4, canary ×4, matched no-canary ×4; seeds
939/949/959/969, disjoint), plus scoring of existing P2b frontier artifacts.
Artifacts: `outputs/privacy_calibration/`, `outputs/privacy_extended/privacy_extended.json`,
`privacy_extended_summary.md`.
Execution: 12/12 calibration runs completed, 0 failures.

## Result: all four broadened attack families failed instrument qualification

Per the frozen gates (positive control must pass its floor; null must sit within ±0.02):

| Family | Positive control | Null | Verdict |
|---|---|---|---|
| sequence-property | advantage −0.005 / 0.000 (floor 0.05) | 0.000 ✓ | DISQUALIFIED — properties near-constant on 128-token windows (majority rate 0.97–1.00); no probe can show advantage |
| structure/semantic | advantage +0.036 (floor 0.05) | 0.000 ✓ | DISQUALIFIED — real but sub-floor signal |
| history-accumulation | history−single = 0.000 (floor 0.02) | 0.000 ✓ | DISQUALIFIED — single-step inversion already saturated (accuracy 1.0 at q=32), no headroom |
| canary membership | AUC 0.50–0.57 (floor 0.9) | up to 0.094 ✗ | DISQUALIFIED — attack too weak AND null band tighter than sampling noise at n=16v16 |

## What this does and does not mean

- This is an **instrument outcome, not a privacy failure**: we cannot claim the frontier cells
  resist these attacks, and we cannot claim they do not. Frontier survival under the broadened
  suite is **UNDETERMINED** (no qualified family gated any cell).
- The token-inversion evidence from P0–P2b stands unchanged and remains the entire privacy
  basis for H5/C3.
- Per AGENTS.md rule 6, the following are now **named omitted/unqualified channels** and must
  appear wherever privacy results are cited: sequence-property, structure/semantic,
  history-accumulation, canary membership attacks (unqualified instruments); timing; optimizer
  state beyond the remote module; active/adaptive attackers; natural-sequence membership;
  cross-run joint attacks. Families 3–4 were evaluable only on calibration captures
  (confirmatory runs lack dense transcripts).
- Scientific lessons: (a) attack metrics need enough outcome variance to be instruments —
  near-constant properties cannot leak measurably; (b) a saturated baseline leaves no room for
  accumulation effects; (c) small-n AUC metrics need wider null bands or larger samples — the
  frozen 0.02 band was tighter than sampling noise, a design flaw recorded for the next
  amendment.

## Ledger consequence

C3's privacy scope narrows explicitly: "token-inversion family on per-step latents; four further
families attempted and unqualified (see P4a)". The frontier existence support stands with that
scope. No other claim changes.

## Path to a broader claim (future amendments)

- structure/semantic: re-qualify with a lower predeclared floor justified by a power analysis,
  or a stronger probe class (the +0.036 signal was real).
- sequence-property: choose properties with outcome variance on 128-token windows (frozen
  before measurement).
- history-accumulation: evaluate at earlier training stages where single-step inversion is
  unsaturated, with a frozen saturation rule.
- canary membership: larger canary/non-member sets (n ≥ 64), null band from binomial sampling
  noise, and gradient-level features rather than probe confidence alone.
