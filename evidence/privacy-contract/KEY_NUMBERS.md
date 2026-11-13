# Key verified numbers for the privacy-contract paper

All from committed JSONs in `../Distributed Training/paper-data/`. Cite as
(cell name, artifact path). Do not use numbers not on this list without
re-verifying against a committed artifact.

## The defense that works (v9.2 winner)

- Excess over label-free majority control: **+0.292 pp** (K=2 monomial D=64,
  `latent_v9/latent_v9_v92_k2_d64_06b_attacker.json`), **+0.323 pp**
  (K=2 radial D=64, `..._v92_k2_radial_d64_06b_attacker.json`)
- 3-seed re-score: +0.318 / +0.647 / +0.427 pp (`latent_v9/latent_v9_v92_s4{2,3,4}_attacker.json`)
- 35B transfer: +0.390 pp, ratio 1.106 (`latent_v9/latent_v9_v92_35b_s42_attacker.json`)
- 12B: +0.540 pp, utility −0.411, ratio 1.111 (`latent_v7_defense/latent_v7_12b_wss_chaff16_seed42_v2_attacker.json`)
- v11 matching attacks: VMA centroid 0.02–0.03% and signature matching 1.54–1.56%
  vs majority control ~5.9% (both models; `latent_v11/*_matching.json`)
- Undefended backbone reference: 58.7% top-1 (`latent_v11/mi_budget_06b_backbone.json`)

## MINE evidence (the paper's core)

- v13.0 (MINE + minimax, 0.6B): excess +0.576 pp, MINE = **−1.3039×10⁻⁶ nats**
  (`latent_v13/latent_v13_a9_mine010*.json`)
- v13.1 (MINE only, no minimax, 0.6B): +0.428 pp, MINE = −0.3272×10⁻³ nats
- v13.0/13.1 on 35B: +0.540 / +0.417 pp, MINE ≈ −0.18×10⁻³ nats both
  (`latent_v13/latent_v13_35b_v130_mine*.json`, `..._v131_mionly*.json`)
- v11.2 vacuous CE bound: prior 5.771 nats vs calibrated probe CE 5.775–6.013
  (bound = 0 bits while probes recover 55–59% top-1)

## Verdict flips under MINE (re-evaluation, seed 42)

- E2 seq-64: +1.250 → **+0.758 pp** (`latent_reval/latent_reval_e2_seq64_mine_attacker.json`)
- E2 seq-128: +1.364 → **+0.264 pp** (utility +0.380 — now binding)
- noise σ=0.30 tripwire: +1.039 (v6) → **+0.553 pp**
- E5 pub-4k/priv-1k: +1.490 → +1.553 (still FAIL — distribution shift, not MI)

## Capacity ceiling (six flat axes, 0.6B unless noted)

- v9.2 reference contribution: 1.477–1.488 (zero-cloud penalty)
- experts 8→32 (35B): 0.404 flat; depth 2→11 (35B): 0.400–0.403 flat
- width D=16→64→128: 1.452/1.488/1.488 flat
- two segments: +0.023/+0.017 per segment; on 35B: A +0.095 / B −0.012
- invariant-MLP family: 1.473; convergence 4k+10k: 1.427
- **the only mover: public pretraining 1.556** (v10.1)

## Integrity (Byzantine, v9.5)

- 3 honest CPU replicas: 2,256 frames, 0 flagged, deviation exactly 0.0
- 1 malicious of 3 (undeclared 0.5 perturbation, self-reports 0 tampered):
  **2,256/2,256 flagged (100%)**, utility +0.088 vs +0.071 honest
  (`latent_v9/latent_v9_byz_honest_3cpu_06b.json`, `..._malicious1of3_06b.json`)

## Utility/runtime anchors

- v9.2: loss delta +0.065..+0.073, eval ratio 2.30–2.52× (K=2)
- v13.x: ratio 1.84× (0.6B), 1.06–1.11× (35B)
- Byzantine on CPU nodes: 9.1–9.3× (weak hardware, not the mechanism)

## Corpus/provenance (PR-101 outcomes)

- Original corpus: sha256 78b6bfb9…, 10,316,588 B, WikiText-2 **tokenized**
  variant (12,095 `<unk>` lines, `@-@` spacing) — NOT raw-v1
- Scope caveat: all cells use first 16,896 tokens (~74.8 KB) only
- D4 settle-it: e2_seq128 on reconstructed corpus = +0.410 vs their +0.377
  (corpus drives the leak magnitude, confirmed)

## Honest limits to restate in the new paper

- MINE/DV is a LOWER bound on MI (can under-read, never over-read)
- All training evidence is 2,000-step diagnostic runs (one 10k cell)
- DP ε ≈ 2.77×10⁶ is vacuous, quoted as such
- `launchable: false` for all defense profiles
