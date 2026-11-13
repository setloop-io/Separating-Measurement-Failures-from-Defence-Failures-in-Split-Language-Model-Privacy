# Preregistration — p4a-privacy-broadening-1.0

Status: **FROZEN 2026-09-07, before any run or scoring under this protocol.**

Amendment chain: p1-pilot-0.1 → p2-confirmatory-1.0 → p2b-rate-sweep-1.0 →
p2c-segment-statistics-1.0 → p3-replication-1.0 → **p4a-privacy-broadening-1.0**.

## Scientific question

Does the confirmed feasible frontier (medium span, width 256, q ∈ {4, 8, 16, 32}) survive a
broader attack suite? All prior privacy evidence rests on one family (per-step token-inversion).
Per AGENTS.md rule 6, each family requires a declared transcript, a matched null, and a
calibrated positive control; every omitted channel must be named.

## Attack families (all operate on the declared remote-visible transcript)

1. **sequence-property**: predict a sequence-level property of the probe input (vowel fraction
   bucket, digit-presence) from the mean-pooled quantized latent. Metric: accuracy (vowel
   bucket, 4 classes) / balanced accuracy (digit presence). Null: latents shuffled w.r.t.
   sequences. Positive control: property computed from UNQUANTIZED latents of a calibration
   capture must be detectable (advantage ≥ 0.05 over null).
2. **structure/semantic**: predict the class of the NEXT byte after each position
   (newline / space / letter / digit / punctuation; 5 classes) from the quantized latent at
   that position. Metric: accuracy. Null: shuffled. Positive control: same at q=32 on
   calibration captures, advantage ≥ 0.05.
3. **history-accumulation**: token-inversion probe (as P0–P2b) given K consecutive dense
   captures (latents at steps t−K+1..t, K=5) vs single-step. Metric: per-position top-1
   accuracy advantage of history-features over single-step features. Null: history windows with
   step order shuffled. Positive control: on UNQUANTIZED dense captures, history features must
   beat single-step by ≥ 0.02. Requires dense-capture runs (below).
4. **canary membership**: 16 distinct 32-byte random canary strings, each injected 10× into the
   training stream of canary calibration runs; attacker scores candidate sequences by mean
   token-inversion-probe confidence (max-probability per position averaged) on their latents.
   Metric: AUC distinguishing member canaries from held-out non-member random strings
   (16 vs 16). Null: same scoring with latents from runs trained WITHOUT canaries matched for
   steps/budget. Positive control: at q=32 the AUC must exceed 0.9 on calibration captures.
   Requires canary-injection runs (below).

## Calibration runs (calibration cohort; new captures, no confirmatory claims)

- **dense-capture**: trained mode, medium span, width 256, seeds {939, 949}, 600 steps,
  transcript_sample_every=1, storing latents (and returned values/grads per existing channels);
  each run at q ∈ {4, 32} (two runs per seed → 4 runs) — q=32 is the positive-control dose,
  q=4 the hardest defense.
- **canary-injection**: trained mode, medium span, width 256, seeds {959, 969}, 600 steps,
  canaries injected per the schedule above; q ∈ {4, 32} (4 runs); plus matched no-canary runs
  (same seeds, no injection) for the null (4 runs). Canary byte strings drawn from a dedicated
  stream (`canary`, seed 979), fixed for the study.
- All calibration runs use cohort="calibration", protocol_version p4a-privacy-broadening-1.0.

## Qualification gates (per family, frozen)

A family is QUALIFIED iff: positive control passes its frozen floor AND |null advantage| ≤ 0.02
(on calibration data, seeds disjoint from scoring). A family failing qualification is
DISQUALIFIED and reported as such — its absence narrows the privacy claim and is named
explicitly, per AGENTS.md rule 6.

## Scoring and thresholds

- Target cells: the four frontier cells (medium span, width 256, q ∈ {4, 8, 16, 32}) from P2b,
  using their stored probe-batch transcript tensors (families 1–2) and, for family 3–4, the
  dense-capture/canary calibration runs at matching q (families 3–4 are evaluated on the
  calibration captures only, since confirmatory runs lack dense transcripts — this limitation
  is declared).
- Threshold per family: advantage ≤ 0.02 (token-inversion precedent; AUC advantage for family 4
  measured as AUC − 0.5, threshold ≤ 0.02 above the null).
- Multiplicity: family-wise; each family gated separately, no cross-family pooling; per-family
  results always visible.
- **Frontier survival rule**: a frontier cell survives iff every QUALIFIED family scores ≤
  threshold at that cell's q. H5's support is revised accordingly: if a previously feasible cell
  fails any qualified family, the frontier shrinks and the ledger is updated to say so.

## Out of scope (named omissions, per rule 6)

Timing channels; attacks on the optimizer state beyond the remote module; active/adaptive
attackers that modify queries; semantic attacks beyond byte-structure classes; membership
inference of natural (non-canary) sequences; joint cross-run attacks.

## Freeze rule

Changes require a versioned amendment written before inspecting affected outcomes.
