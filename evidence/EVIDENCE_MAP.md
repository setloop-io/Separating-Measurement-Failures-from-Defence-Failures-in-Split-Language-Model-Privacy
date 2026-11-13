# Evidence Map — Paper 9

Every number quoted in `paper/main.tex` resolves to a file in this tree. Raw
training outputs (JSON per seed/frame) stay in the source repositories under
hash-pinned manifests; this folder carries the decision documents, ledgers,
and result tables that cite them.

## Failure 1 — decoy-gradient experiment (arXiv:2609.04382, published)

| Claim in main.tex | Evidence |
|---|---|
| 4,096/4,096 frames, 9 seeds, row agreement 1.000 | `decoy-gradient/paper1-main.tex` (ll. 79, 633–634) |
| +0.65 to +1.50 pp attack recovery | `decoy-gradient/paper1-main.tex` (l. 81, l. 677) |
| clip+noise closes channel at ≈0.01 nats | `decoy-gradient/paper1-main.tex` (ll. 89, 616, 783) |
| E1 protocol and reproduction | `decoy-gradient/E1_GRADIENT_CHANNEL_2026-08-22.md`, `E1_REPRODUCTION_W17_2026-08-22.md`, `METRIC_STATISTICS_AUDIT.md` |

## Failure 2 — boundary-replacement experiments

| Claim | Evidence |
|---|---|
| Zero-return controls beat trained remote in all 6/6 clean comparisons | `boundary-replacement/independent_evidence_review_2026-09-09/VERDICTS.md` (ll. 10, 38–43) |
| NLL passes utility gate (+0.034) | source: `Paper-2…/papers/delegation-frontier/generated/results.tex` (`\QualityDeltaNLL`) |
| Answer-token match collapses −16.64 pp (34.79% → 18.15%) | `boundary-replacement/independent_evidence_review_2026-09-09/VERDICTS.md` (l. 81), `adversarial_recomputed_2026-09-08.json` |
| 42.2% clean point (47.4% demoted) | `boundary-replacement/paper2-README.md` (l. 22) |

## Failure 3 — trilemma campaign (R1)

| Claim | Evidence |
|---|---|
| All 19 configurations rejected | `trilemma-campaign/2026-09-17-exploration-review.md` (l. 27) |
| 4 of 24 instruments qualified | `trilemma-campaign/2026-09-16-scientific-status.md` (l. 38), `R1_CONFIRMATION_REVIEW.md` (l. 49) |
| Confirmation power ≤ 26.4% (NO-GO) | `trilemma-campaign/2026-09-17-exploration-review.md` (l. 44) |
| depth-4 ΔU 0.0091 vs base 0.0348 | `trilemma-campaign/r1_cell_results.csv` (rows `base`, `depth-4`) |
| span-6 missed utility gate by 2.95e-5 nats | `trilemma-campaign/r1_cell_results.csv` (row `span-6`, utility 0.09997053 vs gate 0.100 in `r1_results.tex`) |

## Failure 4 — defense-redesign pilot

| Claim | Evidence |
|---|---|
| Punctuation advantage 0.79–0.89 vs gate 0.02 | `defense-redesign-pilot/PILOT_DECISION.md` (l. 29), `CLAIM_EVIDENCE_LEDGER.md` (l. 362) |
| Next-byte structure 0.097–0.112 | same (l. 31); calibration ranges in `QUALIFICATION_DECISION.md` |
| Utility + causal-gain pass every cell, privacy fails/undetermined | `defense-redesign-pilot/PILOT_DECISION.md` (ll. 41–42) |
| 0.10 property-control floor ≠ 0.02 acceptance threshold | `defense-redesign-pilot/paper4-README.md` (l. 89) |
| **OPEN:** "≥6.0% coordinate-visible leak" detection floor | quoted in `synthesis/ADVERSARIAL_REVIEW.md` (l. 55); not yet traced to a Paper 4 ledger entry — verify or drop before submission |

## Structural results — E8 / Distributed Training

| Claim | Evidence |
|---|---|
| Undefended boundary leaks 55–80% of tokens | `structural-e8/distributed-training-README.md` (l. 28), `EXPERIMENT_COMPARISON_MATRIX.md` |
| E8 ≤ 3.7% at zero utility cost (≤3.71% @27B bf16; sub-percent on 0.6B–12B) | `structural-e8/EVIDENCE_EXECUTION_LEDGER.md` (l. 38), `EXPERIMENT_STORY.md` (l. 178) |
| E9 loss unchanged: 3.9985 vs 3.9984 | `structural-e8/EXPERIMENT_COMPARISON_MATRIX.md` |
| K50 ≈ 0.09·H labeled pairs: 87.7–94.3 @H=1024 (3 W seeds); 258/356/618 @H=5120 depths 1/4/8 | `structural-e8/EXPERIMENT_STORY.md` (l. 178), ledger l. 38 |
| Live wire: 50% baseline → 0% obfuscated, unchanged loss | `structural-e8/EXPERIMENT_STORY.md` (l. 166) |
| Per-epoch ratchet: 0% top-1 wire recovery | `structural-e8/EVIDENCE_EXECUTION_LEDGER.md` (l. 42) |
| Invariant-MLP task contribution flat: 1.473 vs 1.488 | `structural-e8/LATENT_CAMPAIGN_REPORT_V7_V9.md` (l. 334) |

## Cross-program measurements

| Claim | Evidence |
|---|---|
| Gradients triple document recovery 13.71% → 37.77% (GPT-2, L6, public copy) | `attack-suite/2610.04128-main.tex` (ll. 55, 230) — published arXiv:2610.04128 |
| MINE ≈ 0 bits/token on defended boundary | `privacy-contract/privacy-contract-draft.tex` (l. 239) |
| Delegation triangle: privacy+quality @14%, privacy+delegation @28.6% | `privacy-contract/NEW_PAPER_BRIEF.md` (l. 35) |

## Audit basis

`synthesis/SYNTHESIS.md` (program-wide read), `synthesis/ADVERSARIAL_REVIEW.md`
(external positioning), `synthesis/PAPERS.md` (inventory). Numbers in
`main.tex` were cross-checked against the files above on 2026-11-06;
corrections applied: K50 units (pairs, not sessions; 2.4× depth spread, not
4×), invariant-MLP 1.473 = task contribution (not recovery), MINE in bits
(not nats), E8 baseline 55–80% (not 64–70%).

## Scripts and raw artifacts

Analysis scripts are frozen inside the source repositories (hash-recorded);
they are referenced, not duplicated, to keep provenance intact:

- Trilemma analysis: `Paper 3 - Privacy-Constrained Remote/paper/figs/` (CSV generators pinned by that repo's manifests)
- Pilot decisions: `Paper-4-Defense-Redesign/src/p4d/` + `docs/CLAIM_EVIDENCE_LEDGER.md` manifest hashes
- E8 batteries: `Distributed Training/probes/`, artifacts under `paper-data/collected/` (see `EVIDENCE_EXECUTION_LEDGER.md` artifact column)
- Gradient-gain tables: `paper-llm-attack-suite` (code in `LLM-Attacker/`, arXiv ancillary files)
