# Paper A: current corrections and evidence index

This is the current author-facing correction record for *What Gradients Add to Text Leakage in Split Language Models*. It supplements the dated campaign logs and reviewer responses without changing their historical bytes. The [nine-report audit](../artifacts/paper-a-review-evidence-audit-20261022/REPORT.md) and [targeted revision plan](../artifacts/paper-a-review-evidence-audit-20261022/author-revision-plan.md) identify the issues against the pre-correction manuscript. The [baseline snapshot](../artifacts/paper-a-targeted-corrections-20261022/baseline-manifest.json) binds the version before this editorial pass.

The revision preserves the paper's section structure, original primary result, recorded experiment outputs and original statistical families. It corrects descriptions and independently verified table entries. No training, inference, capture, attack or bootstrap execution is part of this pass. Paper B's independent study is outside its scope.

## Corrected reporting

- Defended whitespace-normalised exact-passage recovery is **17.421875%** for replica forward, **47.1484375%** for replica joint, **0.1953125%** for ordinary mixup joint, **1.796875%** for adaptive mixup forward, and **6.171875%** for adaptive mixup joint. Raw exact-passage and token rates are different metrics and are preserved. The [independent count table](../artifacts/paper-a-stronger-20261020/review-20261021/ws-verification/counts.csv) retains the numerators and denominators.
- Defended and longer-passage victims were freshly trained for 128 steps from pinned bases/seeds. Saved observation and prediction anchors verify their specified outputs; they do not prove full weight-state identity with retained checkpoints.
- Adaptive joint beam scoring uses the per-position mean squared residual to the transformed target, minus 0.25 times the within-candidate-standardised backward score. The transformation, candidate proposals and gradient beam term are bundled changes. Adaptive-forward has no gradient term. The custom loop initially retains the first B position-zero candidates.
- The fixed-boundary redesign uses original autograd in Regime A and a straight-through estimator in Regime B. The eight selected Qwen numerical arms do not form a complete factorial or identify every causal interaction.
- The fixed-boundary table reports start-explained total variance as **Start R²**, separately from within-seed R². Its 28 outcome tests have exploratory pointwise intervals and unadjusted permutation p-values. The original four-test placement family remains separately labelled and unchanged.
- The longer-passage extension uses nested prefixes of 255 previously examined articles. Fresh 64/128-token attacks are complete; independent new articles and 256-token decoding are not. Position-slice accuracy does not establish zero prefix-error compounding.
- Three 500-step trajectories provide three training runs with serial observations. The victim has active training dropout; the public attacker prefix is deterministic. No dropout-aware expectation-over-transformation baseline was executed.

## Campaign evidence

| Campaign | Retained records | Verification and interpretation |
|---|---|---|
| Original primary, original 47 secondaries and placement rows | `experiments/validation_protocol.v3.json`; original release and dated retained-data exports | [Protocol/code audit](../artifacts/paper-a-review-round2-20261020/protocol-code-audit.md). Original primary and secondary families remain separate. |
| Candidate/beam expansion | `artifacts/paper-a-stronger-20261020/analysis/` | [Search/Exp3 audit](../artifacts/paper-a-stronger-20261020/review-20261021/search-exp3-audit/AUDIT.md). 100 physical executions including anchors; k128's adjusted difference interval includes zero, without an equivalence test. |
| Paired live trajectories | `artifacts/paper-a-stronger-20261020/trajectory-analysis/` | [Trajectory/defended verification](../artifacts/paper-a-stronger-20261020/review-20261021/defended-trajectory-audit/README.md). Paired extension complete; old status notes saying otherwise are superseded. |
| Defended/adaptive attacks | `artifacts/paper-a-stronger-20261020/defended-analysis/` | Same [verification](../artifacts/paper-a-stronger-20261020/review-20261021/defended-trajectory-audit/README.md) and independent [whitespace decoding](../artifacts/paper-a-stronger-20261020/review-20261021/ws-verification/README.md). Attack-path encoder queries differ from recorder diagnostics. |
| Fixed-boundary, numerical and longer-passage extensions | `artifacts/paper-a-stronger-20261020/exp3-analysis/` | [Source/prediction verification](../artifacts/paper-a-stronger-20261020/review-20261021/search-exp3-audit/verification.json). 128 placement cells plus eight controls,34 numerical cells, 150 longer-passage cells. Do not mix these units with sidecar files or host workers. |
| Original Qwen reconstruction | `output/paper-b-qwen3-recon-20261020/` | [Qwen provenance audit](../artifacts/paper-a-stronger-20261020/review-20261021/qwen-provenance/README.md). 304 arm cells; complete sealed predictions followed by wrapper failure, then independent offline verification. This is distinct from the completed FP64 placement grid. |
| New Qwen trained-copy k128/calibration and GPT-2 calibration | `output/paper-a-qwen3-cal-scale-20261021/` | [Calibration verification](../artifacts/paper-a-review-evidence-audit-20261022/calibration-verification.json). 48 new C32 trained-copy k128 cells, 128 larger-calibration Qwen beam cells, two repeated anchors, 320 Qwen diagnostic records, 50 GPT-2 diagnostic records and 120 GPT-2 beam summaries. |

The combined standard-calibration Qwen table has **352 C32 arm cells =304+48**. Each physical arm cell scores **256 passages/8,192 token positions**. An eight-victim aggregate scores **2,048 passage evaluations/65,536 positions**, reusing the same 256 articles. k128 decoding covers L1/L7/L14 under both public and trained grants; L21 has k32 decoding, not k128.

Calibration uses nested sets of 32/64/128/256/384 windows from 32 separate calibration articles, disjoint from training and evaluation article partitions. Qwen diagnostic curves cover all four depths and both grants. Larger-calibration Qwen beam decoding is at L14 only: C64/k32, C128/k32+k128, C256/k32. GPT-2 calibration beam decoding is L6/public. C384 has ridge top-1/candidate-recall diagnostics only. C32 beam cells use FP32 ridge solving; larger-calibration beam cells use FP64 solving followed by FP32 scoring. Both solve precisions are retained in diagnostic records. No size-only causal or measured-condition-number claim follows from this mixed-precision beam comparison.

The new Qwen predicted IDs independently reproduce token/exact means; the new GPT-2 calibration campaign retains sealed per-victim summaries but no predicted-ID/per-document beam sidecars. Saved confidence intervals retain their recorded provenance; no new resampling was performed.

## Source and run identities

| Work | Actual source | Actual runs |
|---|---|---|
| Original 304-cell Qwen reconstruction | `5d3590756d4f2fb110b71167b52aeebb40896a4b` | Toula `5bc66b03-3d8d-4d20-a19f-6bf45334617b`; Koula `85534cfb-a8ba-4813-a843-a42f109d04f4`. Both wrappers failed after sealing complete predictions. |
| Exp3 placement/numerical/length | `923d16e4d8b8cb3a74ccfd564de9d4167de8f28f` | Toula `80b28684-ce28-4138-8abd-e4ec761e6397`; Koula `e7ce0f2f-5264-4e42-b565-026265e82815`. |
| Calibration/trained-copy k128 | `ca6ccf72f0cb8d6fa3d988db49f31fbb0b0e11cd` | Toula `9cbdb131-dd19-40a9-ad9e-3d5f987ba0fd`; Koula `466d01b9-b334-426b-8523-5889514ce8e2`. Both completed. |

Inherited `source_commit` fields in later raw reports may name the earlier protocol root. Actual runner/module hashes and retained ORX/Git records identify executed source; those immutable report bytes are preserved. The local audits establish internal record consistency. They do not provide independent external hardware attestation. Historical manual totals for executions/GPU-hours mix accounting units and should not be treated as a reconciled ledger.

## Remaining scope

Strict matched-total-compute/SipIt >128, original-setting BiSR/Vec2Text, isolated adaptive component ablations, dropout-aware activation attacks, an untied-model mechanism test, L21/k128, independently sampled longer articles/256 tokens, published-defence replication, downstream utility/exploitability, stronger position-zero priors and human annotation agreement remain unperformed at their stated requested scope. Some requests can be resolved by accurate claim scope; acknowledging an experiment does not execute it.

The revised [58-page PDF](../output/pdf/paper-a-targeted-corrections-20261022/paper-a-review.pdf) and [standalone source archive](../output/pdf/paper-a-targeted-corrections-20261022/paper-a-review-source.zip) are the local author-review candidate. Independent verification passes 99 checks, including rescoring/decoding all 140 defended prediction files; see the [scientific check](../artifacts/paper-a-targeted-corrections-20261022/scientific-final-review.md). Both clean source compilation and archive recompilation pass with identical extracted text, embedded fonts and resolved references. Immutable releases and earlier review packages are preserved; no Hugging Face/arXiv publication occurs in this editorial pass.
