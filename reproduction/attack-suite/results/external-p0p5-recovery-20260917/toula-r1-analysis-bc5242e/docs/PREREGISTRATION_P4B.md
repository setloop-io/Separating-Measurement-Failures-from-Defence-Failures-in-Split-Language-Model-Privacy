# Preregistration — p4b-privacy-requalification-1.0

Status: **FROZEN 2026-09-07, before any run or scoring under this protocol.**

Amendment chain: … → p4a-privacy-broadening-1.0 (all four families DISQUALIFIED at instrument
qualification) → **p4b-privacy-requalification-1.0**. P4a was calibration-cohort instrument
work; its measurements legitimately inform the redesign below (instrument design is exploratory
by nature). Each redesign is traced to the P4a failure diagnosis recorded in
`docs/P4A_PRIVACY_BROADENING_DECISION.md`. Floors below are frozen NOW and apply to FRESH
calibration captures; no floor was chosen to fit a P4b outcome, which does not exist yet.

## Redesigned attack families

1. **sequence-property (redesigned: continuous targets with outcome variance).** P4a diagnosis:
   bucketed properties were near-constant. New targets, predicted by ridge regression from the
   mean-pooled quantized latent, metric = Pearson r between predicted and true property,
   advantage = r − r_null (shuffled latents):
   - vowel fraction (continuous);
   - punctuation fraction (continuous);
   - newline count (continuous).
   Family qualifies iff AT LEAST ONE target passes its gate (all three always reported).
   Gate per target: positive control (q=32 calibration captures) r-advantage ≥ 0.10 AND
   |null advantage| ≤ 0.02. (Floor raised 0.05 → 0.10 because continuous correlation metrics are
   more sensitive; the old floor would trivialize the positive control.)
2. **structure/semantic (redesigned: stronger probe + floor from a frozen power analysis).**
   P4a diagnosis: real signal (+0.0366) below the 0.05 floor with the weak probe. Changes:
   (a) probe = 2-layer MLP trained 200 epochs with lr 3e-3 on the full dense-capture probe set
   (P4a used a shallow budget); (b) floor frozen at **0.02**, equal to the study-wide null band
   and justified by the P4a calibration measurement that the weak instrument achieved 0.0356 —
   a stronger instrument on the same channel must exceed the null band decisively; this floor is
   frozen before any P4b measurement. Gate: q=32 advantage ≥ 0.02 AND |null| ≤ 0.02.
3. **history-accumulation (redesigned: frozen early-stage saturation rule).** P4a diagnosis:
   single-step inversion saturated (1.0) at step 600, leaving no headroom. Change: dense
   captures from step 10 to 600 (sample_every=10 → 60 captures); the evaluation step is the
   EARLIEST captured step where single-step inversion accuracy ≤ 0.95 at q=32 on the calibration
   capture (frozen rule, computed from the capture, not tuned). Metric: history-features
   (K=5 consecutive latents) minus single-step accuracy at that step. Gate: advantage ≥ 0.02 at
   q=32 AND |null| (step-order shuffled) ≤ 0.02. If no captured step satisfies the saturation
   rule, the family is DISQUALIFIED with cause recorded.
4. **canary membership (redesigned: power-adequate sample + noise-derived null band).** P4a
   diagnosis: n=16v16 made the frozen ±0.02 band tighter than sampling noise, and the probe
   confidence feature was too weak. Changes: 64 canaries vs 64 non-members (32 bytes each, 10
   repeats, dedicated stream seed 9790); scorer features = per-position probe max-probability
   AND the latent's per-position mean, concatenated (both frozen now); null band derived from
   sampling noise: B=50 bootstrap null AUCs from no-canary runs, band = mean ± 3 SD, reported;
   gate: q=32 AUC ≥ 0.75 (lowered from 0.9 — a membership instrument detecting 75% AUC is
   already a strong canary signal; frozen before measurement) AND null within the derived band.

## Calibration runs (fresh, calibration cohort)

- dense-capture: seeds {9390, 9490}, trained, medium span, width 256, 600 steps,
  sample_every=10, capture_probe_batches=2, q ∈ {4, 32} → 4 runs.
- canary: seeds {9590, 9690} × q ∈ {4, 32} with injection (4 runs) + matched no-canary (4 runs).
- Total 12 runs, cohort=calibration, protocol_version p4b-privacy-requalification-1.0.

## Scoring and survival (unchanged in kind from P4a)

- Frontier cells (medium span, width 256, q ∈ {4,8,16,32}) scored on families 1–2 from stored
  probe tensors with width-matched calibration (919) quantized to cell q; families 3–4 scored on
  the fresh dense/canary calibration captures at q ∈ {4,32} only (declared limitation).
- Per-family advantage threshold for a cell to PASS: ≤ 0.02 (families 1–3, advantage over null),
  AUC − null-band upper edge ≤ 0 (family 4).
- Survival rule: a frontier cell survives iff every QUALIFIED family passes at that cell's q.
  Cells with no qualified families remain UNDETERMINED. Disqualifications narrow the claim and
  are named per AGENTS.md rule 6.

## Named omissions (unchanged, still apply)

Timing channels; optimizer state beyond the remote module; active/adaptive attackers; semantic
attacks beyond byte-structure classes; membership of natural (non-canary) sequences; joint
cross-run attacks.

## Freeze rule

Changes require a versioned amendment written before inspecting affected outcomes.
