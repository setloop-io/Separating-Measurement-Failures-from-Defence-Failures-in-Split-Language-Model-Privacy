# Completed single-seam adapted comparison — 20 September 2026

All 80 joint prediction cells completed successfully on toula and koula. Complete-set verification, statistical analysis on sasa and independent archive-based CPU reproduction on toula passed. The revised Paper B reports the result and its limitations; independent numerical and manuscript reviews found no scientific blockers. Both the full evidence package and the revised manuscript are now hash-verified at immutable HF revisions.

## Finding and exact scope

The prespecified depth-6/public comparison gives 93.206787% positional token accuracy for adapted BRE-bf and 97.375488% for `joint_true_k32`. The difference is **−4.168701 percentage points**, with 95% crossed-bootstrap interval **[−4.789429, −3.529053]**. Every victim-run difference is negative, and the whole interval is below the negative two-point practical margin.

The design retains the same 256 v3 documents, ten independently trained victim seeds, depths 3/6, public/exact head grants and attack seeds 17/29. Eighty joint executions produce 160 SIP/bf arm cells and 20,480 document executions. Four trained GRUs are shared across victims and grants. Public/exact SIP predictions are identical in all 40 paired cells and add no independent attacker replications.

Attack seeds are averaged before the 100,000-replicate crossed document/victim bootstrap (seed 93281). Fifteen secondary comparisons form a Bonferroni-corrected family: ten intervals are negative, one positive and four cross zero. In particular, bf beats forward at depth 6/exact, but its comparison with joint there crosses zero. This is not equivalence or a universal attack ranking. The intervals condition on the fixed training corpus and four GRUs, are approximate with ten victim runs, and do not include uncertainty over auxiliary retraining. Budgets differ; no equal-compute claim is made.

This is the separately approved single-seam GPT-2 adaptation. Earlier C-A implementation conformance qualified its implementation. Neither result reproduces the original SensiReplaced/dual-cut setting or satisfies the literal B1 requirement of the unchanged original v4 protocol.

## Provenance and verification

- Prediction source: `2d0f58b9d6912206d1070679e31853f9a344cefc`.
- Frozen launch SHA-256: `6f0dc502ff275be21bba73d5b29f291356bb60ab8a1f0e82d16febea58564d63`.
- Analysis source: `8ec09babf38adf9c9fbb597b853a2ee62b190a20`.
- Toula prediction run: `b831a06d-ecaf-4d9c-bbd1-b2d2a0ad0270`.
- Koula prediction run: `ae607e15-41e2-440e-8d25-82d199d9f8f9`.
- Sasa analysis run: `ec83b6d5-d0c8-4765-838c-c903eb5204f4`.
- Raw archive: 21,154 files, 9,652,098,570 uncompressed bytes; compressed SHA-256 `f49b3120135a0019799aab5d7ed8b7360c0626ee7d0e86ef1a578292817584d0`.

All candidate records and journal chains were checked before comparative scoring. All 240 original control prediction files were hash-verified and their metrics rescored. An independent numerical review reconstructed all 20 arm means, 16 contrasts and per-victim means from the 102,400 metric rows. A fresh extraction verified every raw file and regenerated all four deterministic analysis outputs byte-for-byte; the full result summary matched except its declared Python-version provenance (3.13.5 versus 3.12.3, both NumPy 2.2.4). This was CPU analysis reproduction, not a second GPU prediction or training campaign.

Exact-head replay matches all 20 original activation captures bit-for-bit under the original SDPA path. Attack optimisation uses the C-A-qualified eager path. The original gradient bytes are retained with the declared factor 256 undoing the original full-evaluation-set loss normalisation; full downstream-gradient replay is not claimed.

The frozen amendment's inherited description of SIP-versus-forward as isolating BRE is incorrect. The supplement's explicit erratum and manuscript interpret it as a comparison of reconstruction methods. Original protocol bytes and all prespecified contrasts remain unchanged.

## Compute

Prediction allocation across both GPU hosts was **29.700103 hours**. Fresh inverter training and mechanical preparation used another **0.583789 hours**. Earlier checks are separate. The conservative allowance accounting retains the full three-hour prior/preparation reservation, totalling 32.700103 hours against the approved aggregate ceiling of 50 hours. Reservations are not actual training measurements. Layer token-position counts are retained but are not FLOPs.

## Publication

Public destination: `Setloop/llm-attacker-paper-b`, additive prefix `v4-adaptation-20260920`.

- [Complete evidence and CPU/GPU reproduction instructions](https://huggingface.co/datasets/Setloop/llm-attacker-paper-b/tree/d415492614cb1eb760926bf2e01a33b7fb024124/v4-adaptation-20260920/evidence): **27 files, 3,186,252,756 bytes**, revision `d415492614cb1eb760926bf2e01a33b7fb024124`. Manifest SHA-256 `881bdf02d6777e29449ac3bbef9f2112725fd75dda78052757e18baf0d221e42`.
- [Reviewed manuscript, source and release checks](https://huggingface.co/datasets/Setloop/llm-attacker-paper-b/tree/93f45488e86ee73d595976b524144336b21dc370/v4-adaptation-20260920/manuscript): **19 files, 662,011 bytes**, revision `93f45488e86ee73d595976b524144336b21dc370`. Manifest SHA-256 `450bb72c3c0c4c9def71fb6bab692419dfc5fab841a818eaa6513a14d139218b`.
- [Final Paper B PDF](https://huggingface.co/datasets/Setloop/llm-attacker-paper-b/resolve/93f45488e86ee73d595976b524144336b21dc370/v4-adaptation-20260920/manuscript/main.pdf).

Every expected Git blob/LFS digest and exact prefix coverage passed, with no missing, extra or mismatched files. All 23 rendered manuscript pages were inspected. The final build compiled twice without warnings and its rendered pixels are identical to the reviewed draft after inserting the immutable evidence link. Numerical review, manuscript review, protocol erratum, CPU reproduction and final release-check records are included. Existing baseline, C-A and historical packages remain unchanged.

Local release records: `build/v4-release-20260920/`. Machine-readable status: `experiments/v4_adaptation_current.json`.

## Remaining work outside this adapted release

- Original published-setting BiSR replication remains unestablished; historical substitute runs remain excluded.
- The separate Sasa 500-step extension was outside this adapted release. Its later descriptive audit and release are complete; see the dated status reconciliation below. It does not supply the main study's paired gradient-benefit trajectory.
- Historical external deployment and defence claims lacking execution-source provenance remain withdrawn. Recovered file counts alone cannot restore those claims.
- Submission packaging and venue requirements are separate from verification of this additive release.

## Status reconciliation — 20 October 2026

The earlier statement that the separate 500-step extension still required collection, verification and analysis is superseded. The [release record](../build/sasa-extension-20260920/release/README.md), [two-host reproduction record](../build/sasa-extension-20260920/release/REPRODUCTION.json) and [later preparation status](preprint-readiness-2026-09-20.md) establish completion of its descriptive audit. The manuscript release record identifies evidence revision `f1488fbde74726635f1e8616092ca187ac051270`, prefix `sasa-500step-20260920/evidence`; this reconciliation checks local records and does not repeat remote publication verification. It changes neither the original v4 protocol nor either immutable release.

The extension comprises six GPT-2 depth-six training trajectories: no defence or a fixed random mid-lite projection, each with victim seeds 11/22/33, 500 steps and 32 tokens per sequence. The earlier audit verified 12,443 raw files and 13 journals and independently rescored 3,000 saved sequence predictions. Today's local check rehashed both hosts' `analysis.json` and `trajectory.csv`; all four files match `REPRODUCTION.json`. It did not rerun that audit, train a model or generate predictions.

These are descriptive public-surrogate activation-inversion results on training traffic. Victim training and the 64-window attacker calibration share the pilot corpus, with 1/4/4 exact calibration-row overlaps for seeds 11/22/33 in each defence condition. The decoder uses the clean public base head while the victim changes; it is not projection-adaptive, and capture-time victim checkpoints were not saved. The audit establishes neither held-out convergence nor a paired activation-versus-gradient benefit, exact-head replay, adaptive privacy guarantee or renewed privacy-gate verdict.

All six local raw capture directories retain 500 step-aligned forward/backward wire pairs and canonical scoring truth each: 3,000 pairs in total. A prospectively specified public-head comparison can reuse these captures without retraining the victim, subject to checking the declared gradient scale, FP16 capture precision and training-mode conditions. Such redecoding would be a **new attack experiment and remains unperformed**; it must remain separate from the v3 final-checkpoint, document-disjoint study. The [existing-evidence status audit](../artifacts/paper-a-review-round2-20261020/existing-evidence-status.md) records the checked paths, hashes and remaining access boundaries.
