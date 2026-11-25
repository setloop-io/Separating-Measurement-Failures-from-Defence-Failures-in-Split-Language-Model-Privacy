# Confirmatory preregistration — p3-replication-1.0

Status: **FROZEN 2026-09-06, before any run under this protocol.**

Amendment chain: p1-pilot-0.1 → p2-confirmatory-1.0 → p2b-rate-sweep-1.0 →
p2c-segment-statistics-1.0 → **p3-replication-1.0** (external replication of H1).

## Scientific question

Does the P2-confirmed causal remote-learning gain replicate beyond the original setting?
Per `docs/RESEARCH_PLAN.md` §7: a second architecture from a matched family, a second corpus,
and at least one downstream evaluation, with fresh confirmatory seeds.

## Settings (2×2 minus the already-tested arch1+corpus1 corner)

- **arch1**: 8 layers, d_model 256, heads 8, d_ff 1024 (as P1–P2c).
- **arch2**: 12 layers, d_model 192, heads 6 (head_dim 32), d_ff 768 — matched family, deeper and
  narrower. Split segment at the proportional medium position: span 6 @ start 3 (half the depth,
  same relative location as span 4 @ 2 in arch1).
- **corpus1**: tiny-shakespeare (sha256 86c4e6…65ed). **corpus2**: War and Peace, Project
  Gutenberg #2600 (sha256 2d5bb2…f1b2b), `data/raw/war_and_peace.txt`; identical byte-level
  tokenizer and first-90%/last-10% byte-offset split.
- Boundary at the frontier-plateau point: latent width 256, q_bits = 8.
- Settings: **A2C1** (arch2+corpus1), **A1C2** (arch1+corpus2), **A2C2** (arch2+corpus2).

## Units and seeds

- Inferential unit: independent training run/seed.
- Seeds (fresh, frozen, disjoint from all prior cohorts): **[9, 19, 29, 39, 49, 59, 69, 89]**.
- Per setting: trained, frozen_random, removed_remote, all_local × 8 seeds = 32 runs;
  96 runs total. Pairing `pb-p3-{seed}`; only remote-parameter updating differs within a pair.
- Fixed optimization protocol identical to P2 (AdamW 3e-4 cosine, 2000 steps, batch 32, seq 128,
  eval every 100 on 64 held-out batches, fixed budget, no early stop).

## Decision thresholds (frozen)

- **Primary (A2C2)**: one-sided paired t over 8 seeds on G_R at step 2000, α=0.05; replicate iff
  p < 0.05 AND mean G_R ≥ 0.01 nats AND mean G_R > 0 at steps 1800/1900/2000 (persistence).
- **Secondary (A2C1, A1C2)**: same rule per setting, Bonferroni α=0.025 each.
- Δ_U (trained − all_local) and removed-remote reference per setting: descriptive with 95% CIs.
- **Downstream evaluation (descriptive, not gated)**: linear probes (multinomial logistic) from
  final-layer hidden states of A2C2 trained-mode and all-local final checkpoints predicting byte
  properties {is_uppercase_letter, is_vowel, is_digit, is_whitespace}; probe trained on
  train-split hidden states, evaluated on held-out; accuracy gaps reported with seed-level CIs.
  Final checkpoints are saved only for A2C2 trained and all_local runs (additive, no effect on
  training semantics).
- Privacy: unchanged boundary (width 256, q=8) already scored in P2/P2b; no new privacy
  evaluation in P3; the scope omission (token-inversion family only) still applies wherever P3
  results are cited.
- Missing data: ≤1 retry per run, identical cell identity; seed dropped from affected paired
  tests only (predeclared, never outcome-based).

## Analysis

- Per-setting paired contrasts within `pb-p3-{seed}`; seed-level outcomes only.
- Multiplicity: as gated above; descriptive quantities ungated and labeled.
- Statistics from `scripts/stats_utils.py` (validated against reference values).
- Cross-setting consistency reported descriptively (sign and magnitude pattern); no pooling
  across settings (architectures/corpora are not exchangeable).

## Freeze rule

Changes require a versioned amendment (p3-replication-1.1, …) written before inspecting affected
outcomes. No pooling with prior protocols' data.
