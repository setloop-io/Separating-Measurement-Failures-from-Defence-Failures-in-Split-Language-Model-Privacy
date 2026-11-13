# Papers Inventory — where everything stands

*Companion to [SYNTHESIS.md](SYNTHESIS.md). Updated 2026-11-05. "Status" reflects each package's own plan-of-record + the independent reviews on file.*

## The papers

| # | Paper | Where | Status | What remains |
|---|---|---|---|---|
| 1 | **Privacy Failure in Split-LLM Training: The Returned Gradient Nullifies the Decoys** | `paper1_arxiv_submission/` (submission) + `Paper-1-When-the-Privacy-Probe-Passes-but-the-Transcript-Leaks/` (dev package) | **Submitted to arXiv** (Aug 2026). JAIAI journal adaptation drafted, not uploaded. Referee remediation done (2026-08-30). | arXiv endorsement decision; F.1 protocol freeze + post-freeze seeds (W5.7); cross-Gram attack (W4.2); E1 re-run from packaged code (RELAYED → VERIFIED). |
| 2 | **How Much of an LLM Can a Protected Boundary Replace?** | `Paper-2-How-Much-of-an-LLM-Can-a-Protected-Boundary-Replace/` | **Manuscripts exist; interpretation re-scoped** by two independent Sept-2026 reviews. Not submission-ready. | Decide the narrow paper ("limits of a small protected boundary" — the controlled negative on remote necessity); fix 27B geometry; correct privacy-evaluator targets; gradient-clip (0.01 vs 1.0) sweep; deployed-path quality eval. |
| 3 | **Accuracy, Privacy and Remote Learning in Split Language-Model Training** (R1) | `Paper 3 - Privacy-Constrained Remote/` | **Exploration complete; all 19 configs rejected**; confirmation NO-GO (power ≤ 26.4%). Exploratory manuscript compiled (`paper/main.pdf`). | Public/reviewer release of the zero result; or narrower hypothesis / larger seed cohort; fresh instrument-development programme. Evidence archived (`paper3-r1-release-cache-20260917/`, `r1-*` bundles). |
| 4 | **Defense Redesign Under Calibrated Privacy Instruments** | `Paper-4-Defense-Redesign/` | **Pilot complete (2026-09-22): no cell pilot-eligible** (privacy gate failed ~40×). `paper/main.tex` still a scaffold. | Write the finite negative into the manuscript (punctuation 0.79–0.89 vs 0.02 is the headline number); active-attacker evaluation (C5) before any positive defense claim; TEE–FHE track awaits CC-capable hardware. |
| 5 | **LLM-Attack Suite: A Unified Red-Team Framework for Privacy Evaluation of Split and Distributed LLM Systems** | `paper-llm-attack-suite/2610.22722v2/` + code in `LLM-Attacker/` | **arXiv 2610.22722 (v2 replacement prepared, not yet submitted).** Journal submission drafted 2026-11-01 (Advances in KBSDS&C). | Submit v2 replacement; EuroS&P pending. |
| 6 | **What Gradients Add to Text Leakage in Split Language Models (per Token and per Document)** | `paper-llm-attack-suite/2610.04128v1/` + `journal-submission-gradients/` | **arXiv published; journal submission drafted 2026-11-01.** | Upload journal submission. |
| 7 | **Defense-in-Depth for Split LLM Training on Fully Compromised Cloud Nodes** (master synthesis) | `Distributed Training/papers/arxiv-draft/` + `papers/Privacy_in_Distributed_Training_and_Inference/` | **Draft, not submitted.** Superseded in part by Papers 1–4; its certified-budget part became Paper 8. | Fold the §5 inconsistencies (E8 0.00% vs ≤3.7%; FedAvg K\* 1600 vs 20) or archive as the program record. |
| 8 | **The Privacy Contract — an explicit leakage budget for split LLM training on compromised cloud nodes** | `Privacy-contract/` (`papers/draft/main.tex`, arXiv bundle) | **Complete draft; v14/v15 experiments done. One TODO (trim abstract to 200 words) from submission.** ← closest to done | Trim abstract; submit to arXiv. |
| 9 | **Why Everything Was Rejected: Calibrated Attacks and the Structural Nature of Leakage** | `Paper-9-Calibrated-Attacks-and-Structural-Leakage/` | **Full draft started 2026-11-05** (`paper/main.tex`, compiles; thesis §2A–B of SYNTHESIS). | Figures; real author lists in `refs.bib`; number-by-number re-check vs claim ledgers; title variant. |

## The `LLM-Attacker*` directories (yes — reviewed)

They are **one repo, not thirteen papers**: `LLM-Attacker/` is **LLM-Attack Suite v2.0** (65-module red-team framework; GitHub `setloop-io/LLM-Attack-Suite`; arXiv 2610.22722). The dated siblings are snapshot worktrees of focused tasks:

- `-qwen-reproduction-20260921`, `-qwen-anomaly-`, `-qwen-fp64[-grid|-operands]-` — reproducing the Qwen3-0.6B study incl. fp64 numerics / operand-order probes.
- `-sasa-reproduction-`, `-sasa-review-` — SASA 500-step prediction-integrity audit.
- `-repairs-`, `-closure-`, `-agentic-followup-`, `-v4-adaptation`, `-pr35` — measurement-convention corrections (2026-09-09: original defended pilot superseded; 188 tests, 30-cell corrected synthetic pilot), BISR fidelity, PR #35.

All feed Papers 5–6. They are archive candidates, not open work.

## toula / koula — remaining experiments (checked 2026-11-05)

- **Nothing is running.** Both nodes are up (`ray-head` / `ray-worker`, restarted today); no training processes.
- **Latest artifacts: 2026-08-13/14** (`er9_smoke`, `er9_alignment_retry` logs; `er_*` budget/N sweeps 2026-08-10/11).
- **Stale state:** `~/experiments/results/training_status.json` still says `"state": "running"` for `er_train_wire` dated **2026-08-14** — a dashboard leftover, not a live run.
- **Open items recorded in the docs, not yet run on the nodes:**
  - **E-R9 (full-stack prototype): data finished, analysis pending** (the Aug-13 `er9` logs are its last output).
  - **D3 chosen-plaintext attack** — designated "next GPU job", never run.
  - **E8 battery on 35B-A3B MoE** — unmeasured cell.
  - **Paper 1 post-freeze confirmatory seeds** (W5.7) — owed.
  - **h100-1 (lost 2026-08-07) re-verification** of E7/E8 JSONs — still pending.
  - One of eight THREAT_MAP seams is attacked by zero experiments.

## Suggested order of work

1. **Paper 8 (Privacy Contract)** — trim abstract, submit. *(days)*
2. **Papers 5–6** — upload the two drafted journal submissions; submit the v2 replacement. *(days)*
3. **Paper 9 (synthesis)** — draft from existing evidence. *(the big scientific payoff)*
4. **Paper 2** — write the narrow "limits of a protected boundary" version. *(weeks)*
5. **Papers 3 + 4** — release the two finite negatives as written results (3: release the zero result; 4: fill the scaffold with the pilot). *(weeks)*
6. Only then: new GPU work (D3, E8-MoE, post-freeze seeds, E-R9 analysis) — ideally as **one pooled confirmatory experiment**, not per-paper.
