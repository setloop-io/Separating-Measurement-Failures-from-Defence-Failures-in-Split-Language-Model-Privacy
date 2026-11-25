# Evidence Map

**Manuscript:** *Separating Measurement Failures from Defence Failures in Split Language-Model Privacy* (`paper/main.tex`)

Every retained quantitative value in `paper/main.tex` resolves to a hashed
artifact in [../data/source_manifest.json](../data/source_manifest.json) (122
pinned files verified by `python3 analysis/recompute.py`, 1,522 internal
consistency checks). Primary archived records live in `evidence/` and
`reproduction/`; all original files are preserved byte-identical.

## 1. Decoy-gradient experiment (`E1`, arXiv:2609.04382)

| Claim in `paper/main.tex` | Primary evidence (all hashed in `data/source_manifest.json`) |
| --- | --- |
| Exact real/decoy partition recovery on 4,096/4,096 frames per run across 9 seeds | `evidence/decoy-gradient/paper1-main.tex` (ll. 79, 633–634) |
| Frequent-word token probe gain +0.65 to +1.50 pp over constant-guess baseline | `evidence/decoy-gradient/paper1-main.tex` (ll. 81, 667–678) |
| Six utility-compliant gradient-open cells pass forward +1.0 pp gate and breach joint gate | `evidence/decoy-gradient/paper1-main.tex` (ll. 737–747, 774–776) |
| Per-row gradient clipping + Gaussian noise closes partition leak on 1,024 frames/run (3 seeds, 0.400 agreement at 32/80 base rate) at ~0.01 nats utility cost | `evidence/decoy-gradient/paper1-main.tex` (ll. 89, 741–743, 781–783) |
| Supporting channel protocol record | `evidence/decoy-gradient/E1_GRADIENT_CHANNEL_2026-08-22.md` |

## 2. Boundary-replacement review (`B1`)

| Claim in `paper/main.tex` | Primary evidence |
| --- | --- |
| Off-by-one target shift: $(x_1,\ldots,x_{T-1})$ scored against $(x_2,\ldots,x_T)$ | `evidence/boundary-replacement/independent_evidence_review_2026-09-09/VERDICTS.md` (ll. 49–55) |
| Standalone zero-return controls beat trained remote in all 6/6 clean corpus–seed pairs | `evidence/boundary-replacement/independent_evidence_review_2026-09-09/VERDICTS.md` (ll. 38–43), `recomputed_evidence.json` |
| Reviewed checkpoint: NLL rises ~0.034 nats within rule while teacher-forced answer-token top-1 drops 16.64 pp (34.79% to 18.15%) | `evidence/boundary-replacement/independent_evidence_review_2026-09-09/VERDICTS.md` (ll. 79–83), `recomputed_evidence.json` |

## 3. 19-cell privacy, utility and gain exploration (`R1`)

| Claim in `paper/main.tex` | Primary evidence & recomputation |
| --- | --- |
| 232 runs, 19 cells × 4 seeds (76 cell–seed contrasts); all 19 rejected | `reproduction/trilemma-campaign/evidence/r1-exploration-20260917/analysis/exploratory_analysis.json`, `evidence/trilemma-campaign/r1_cell_results.csv`, `r1_seed_results.csv` |
| 4 of 24 required probes qualified (inbound/outbound gradient input-byte probes) | `reproduction/trilemma-campaign/evidence/r1-qualification-20260916/calibration_analysis_summary.json`, `attack-spec-frozen.json` |
| Overlapping failures: 13 qualified privacy breaches, 5 utility, 7 gain; disjoint intersections (10, 3, 3, 2, 1); all 6 privacy-undetermined cells independently rejected | Recomputed in `data/reanalysis_results.json` and `data/r1_cells.csv` (Table 4, Figure 1a) |
| `span-6` passes both utility gates (mean 0.0999705344 ≤ 0.10, upper 95% Student-$t$ 0.1195024415 ≤ 0.12); rejected on gain + privacy | Recomputed in `data/r1_cells.csv` (Figure 2) |
| Unexecuted 8-seed confirmation planning upper power bound 0.263772 | `reproduction/trilemma-campaign/evidence/r1-exploration-20260917/confirmation_power.json`, `evidence/trilemma-campaign/R1_CONFIRMATION_REVIEW.md` |

## 4. Seven-cell defence-redesign pilot (`P4`)

| Claim in `paper/main.tex` | Primary evidence & recomputation |
| --- | --- |
| 66 runs across 7 cells × 3 paired seeds (126 candidate probe records, 16,384-window panels); 0 eligible | `reproduction/defense-redesign-pilot/outputs/runs/pilot-20260922-a/verification.json`, `report.md`, `summary.json`, 66 `metrics.json` files, `evidence/defense-redesign-pilot/PILOT_DECISION.md` |
| 2 utility failures (`d3-q1`, `d3-q2`, one-sided 95% upper Student-$t$ $t_{0.95,2}=2.919986$), 7/7 gain passes, 4 failed / 3 undetermined composite privacy | Recomputed in `data/reanalysis_results.json` and `data/p4_cells.csv` (Table 2, Figure 1b) |
| Calibrated next-byte structure (`next_byte_structure`, floor 0.02): 0.097–0.112 across all 7 cells (21/21 bindings qualified) | `reproduction/defense-redesign-pilot/src/p4d/calibration.py`, recomputed in `data/p4_cells.csv` |
| Sequence-property punctuation fraction (`sequence_property`, floor 0.10): 0.79–0.89 across qualified bindings (17/21 bindings qualified; 4 marginal null excursions on 3 cells) | `reproduction/defense-redesign-pilot/src/p4d/calibration.py`, recomputed in `data/p4_cells.csv` |

## 5. Orthogonal wrapper (`E8`) and deployed ratcheting (`ER`)

| Claim in `paper/main.tex` | Primary evidence |
| --- | --- |
| 27B ($H=5120$) depth-specific $K_{50}$ crossings: 258.32 (d=1), 355.81 (d=4), 617.93 (d=8); ratios 0.05045–0.12069 (2.39-fold range) | `reproduction/structural-e8/paper-data/collected/h100-2/issues/issue_e8_27b_20260807/e8_robustness_27b_bf16_densek_20260807T230529Z.json` |
| Depth-1 per-session rotation: 3.71% pooled top-1 wire score vs 5.73%, 3.65%, 2.34% individual sessions | Same JSON artifact + `evidence/structural-e8/EXPERIMENT_STORY.md` |
| Deployed 12B ratcheting matrix: 8 runs, 24 sessions, 20/24 greedy-decode identity, 4/8 runs with known limitations, 0% top-1 rotated wire recovery, `N=16` zero-RTT session-1 throughput 2.8 vs 7.6 tokens/s (63.16% drop), budget arm 5.5–5.6 tokens/s at zero RTT | `reproduction/structural-e8/paper-data/collected/toula/er_20260811/COLLECTION_MANIFEST.json`, `summary.json`, `identity.json`, `wire_attack_*.json`, `evidence/structural-e8/EVIDENCE_EXECUTION_LEDGER.md` |

## 6. Variational mutual-information bounds (`MI`) & companion benchmark (`arXiv:2610.04128`)

| Claim in `paper/main.tex` | Primary evidence |
| --- | --- |
| Near-zero MINE Donsker–Varadhan readouts and companion cross-entropy lower bound (both acknowledged as lower bounds, not upper ceilings) | `evidence/privacy-contract/privacy-contract-draft.tex`, `evidence/privacy-contract/KEY_NUMBERS.md` |
| Finite invariant-MLP / radial architecture ablations (empirical saturation, not universal capacity proof) | `evidence/structural-e8/LATENT_CAMPAIGN_REPORT_V7_V9.md` |
| Controlled GPT-2 returned-gradient gain: 13.71% to 37.77% exact-document recovery and +3.17 pp token recovery (3.1714 pp unrounded) | `evidence/attack-suite/2610.04128-main.tex` (ll. 55, 176, 187, 209) |
