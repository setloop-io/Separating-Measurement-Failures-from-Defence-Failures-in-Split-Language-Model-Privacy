# Paper A: response to the three review reports

This draft revises **What Gradients Add to Text Leakage in Split Language Models, Counted per Token and per Document** using the retained experiment records. It does not add training runs, model inference, new attacks or new empirical claims about a different deployment. New rescoring, bootstrap intervals, component tests and permutation references are explicitly post hoc. The original pre-specified results and frozen evidence remain unchanged.

The reports are labelled R1 (the unsuffixed review PDF), R2 (`-2.pdf`) and R3 (`-3.pdf`). All three describe themselves as AI-generated reports. Their technical claims were checked against the paper, retained predictions and frozen source; their suggested changes were not treated as automatic instructions.

## Draft and supporting material

- [Revised manuscript](../paper/paper_a/main.pdf) and [LaTeX source](../paper/paper_a/main.tex).
- [Review PDF](../output/pdf/paper-a-review-20261020/paper-a-review.pdf).
- [Standalone source ZIP](../output/pdf/paper-a-review-20261020/paper-a-review-source.zip).
- [Reanalysis supplement](../output/pdf/paper-a-review-20261020/paper-a-review-reanalysis.zip).
- [Prediction rescoring](../artifacts/paper-a-review-20261020/metrics/report.md), [placement reanalysis](../artifacts/paper-a-review-20261020/offloading/REPORT.txt), and [frozen-source methods audit](../artifacts/paper-a-review-20261020/methods-audit.md).
- [Full-grid permutation reference](../artifacts/paper-a-review-20261020/offloading/null-calibration/analysis.json).

The existing Hugging Face release and tags have not been replaced. This is a review candidate, not a newly submitted arXiv version.

## What changed scientifically

1. **The primary comparison is explicitly conditional.** It compares these decoders at fixed candidate budget and beam width, using a final-checkpoint loss-gradient probe. It is not a universal measure of the information in gradients, a strongest-attack comparison, a matched-runtime experiment, or an observation of an evolving training trajectory.
2. **The document advantage survives normalisation.** Exact token-ID recovery remains 13.71% versus 37.77%, a 24.06-point paired difference with post-hoc 95% interval [16.88, 30.94], and ratio 2.75 [2.25, 3.58]. Removing leading ASCII spaces from the complete decoded text gives 18.91% versus 49.10%, ratio 2.60 [2.16, 3.24]. Broader whitespace normalisation gives a 2.56-fold ratio.
3. **Corrections and regressions are separated.** Of 9,496 activation-only token errors, 5,418 are corrected; 222 new errors are introduced. Gross correction is 57.06%; net error reduction is 54.72%.
4. **The partial-offloading claim is narrowed.** Moving segment start also moves observation depth. For every start at or above one, VMI is exactly invariant across seeds and downstream run lengths in both models and defences. The paper now reports associations with the coupled boundary/training configuration, not an independent causal effect of segment start.
5. **The predictor discrepancy has a concrete diagnostic.** The original predictor uses numeric degree-one features; the variance test uses categorical interactions. A post-hoc matched categorical predictor reduces Qwen D1 utility RMSE from 2.2847 to 0.7130 and index RMSE from 0.2177 to 0.1010. The original numeric results remain reported. The already-examined held-out seeds do not become a new independent validation.
6. **The composite index receives a full retained-data reference.** Small scoring JSON files were recovered read-only from the completed experiment archives; all 400 cell anchors and 800 scoring-file hashes were verified. All 384 trained-run metric sets reconstruct within 1e-12. Under a specified document-correspondence permutation reference and 48-way post-hoc adjustment, 37 of 48 placement means exceed the reference. This is not a universal leakage null. Failure to pass does not establish absence of leakage.
7. **Defence interpretation is restricted to what was tested.** Defended configurations contain activation-only decoding and SipIt, not the joint attacker. The implementation audit corrects pool history, the location of gradient correction, auxiliary decoy visibility, replica initialisation and differing utility/capture release schedules.
8. **Rare spans and selected maxima receive appropriate caution.** There are only two identifier spans. Name and number lengths differ substantially; length-stratified results replace a blanket semantic-class interpretation. The selected replica-median decrease of 7.66 points has paired interval [-4.02, 18.07], including zero and without adjustment for selecting a maximum.

## R1: point-by-point disposition

| Comment | Change in this draft | Remaining work |
| --- | --- | --- |
| Major 1: gradient value depends on attack budget | Abstract, contributions, methods and conclusion now specify fixed candidate budget and attack family. The cost appendix adds observed host-stratified recovery/cost points and work counters. | An equal-time-budget or saturation comparison requires new attack runs. Existing nonmonotonic timings are retained and not smoothed into a claimed matched-compute curve. |
| Major 2: placement and frozen-prefix confound | Section 5 and its abstract/contribution claims now describe the coupled configuration. Frozen-prefix invariance is verified across individual seeds, and the `b=0` exception is explicit. | Fresh training is needed to retain an independent causal trained-layer-placement claim. |
| Major 3: document arithmetic and length | Appendix D compares observed recovery with homogeneous and position-specific independence products, and scores retained prefixes of 8/16/24/32 positions. | These are not fresh short-input attacks or evidence about long documents. The review's inference of positive error correlation from pooled rates is not accepted. |
| Minor 1: missing BiSR comparison | Discussion limits the implication to evaluation practice; the companion's separately frozen adaptation is identified and linked under its current name. | Direct comparative BiSR strength is not claimed. |
| Minor 2: “55% corrected” | Replaced with net error reduction and a paired correction/new-error breakdown. | None for this reporting correction. |
| Minor 3: maximum leakage index | Appendix E separates VMI and all three AUCs, adds component placement tests and full document-correspondence/conditional permutation references. | These analyses do not establish an independent placement mechanism or a universal leakage definition. |
| Minor 4: leading-space differences | Added decoded leading-space and broader whitespace-normalised recovery with paired intervals. | None for the retained-data sensitivity analysis. |
| Minor 5: main training details | Added retained-source optimiser, steps, batch, learning rate, dropout, precision, objective and capture procedure. | The source audit states exactly what its fingerprints cover; hash consistency is not external authentication. |
| Minor 6: activation recovery range | The 94.20–97.81% range is explicitly qualified as the activation decoder at `k=32`. | Lower-budget and SipIt results remain in the full matrix. |
| Minor 7: Figure 7 versus table | Retained the figure label because it is a plotted heatmap, and its caption identifies its form and scope. | No conversion is necessary to address a misclassification of the graphic. |
| Minor 8: replica-median purpose and uncertainty | Described as integrity-oriented without claiming tested Byzantine correctness. Added the paired interval for its selected largest document change. | No integrity experiment or tested privacy reduction is claimed. |
| Minor 9: secret mixup origin | Identified the study's in-repository recycled-decoy implementation and documented its exact mechanism. | No claim of a faithful external defence replication or new general mixup theorem. |
| Minor 10: defended lower bounds | The abstract and main defence section identify the activation-only attack set and lower-bound interpretation. | Joint and defence-aware attacks remain new experiments. |
| Minor 11: calibration set sensitivity | Explicitly identifies the fixed 32-document/1,024-token calibration design and fitted ridge penalty. | Varying calibration size requires refitting and new attack evaluations. |
| Minor 12: beam sensitivity | Beam width two is stated in the abstract and main claims. | Wider-beam reconstruction requires new attack evaluations. |

## R2: point-by-point disposition

| Comment | Change in this draft | Remaining work |
| --- | --- | --- |
| 1: activation minimum | Qualified by `k=32` and the specific decoder. | None. |
| 2: controls sacrifice half the activation candidates | Explained the effective activation half-budget and the candidate-replacement question being tested. | The random/shuffled attack is not declared algorithmically identical to `k=16`; extra candidates can affect beam pruning. |
| 3: no defended joint attack | Explicit in the abstract, configuration count, methods, defence discussion and limitations. | New joint attack/capture experiment required to claim defence behaviour against both signals. |
| 4: two identifier instances | Removed class-level interpretation and described the two spans as examples only. | A representative identifier study requires more examples and new attack evaluation. |
| 5: observed noise versus perturbed training | Observation-only perturbations remain explicitly separate from deployed training-time defences. | No training-time result is inferred from observation corruption. |
| 6: starting-layer confound | Reframed the full placement result and verified exact VMI invariance; separately explains the within-segment observation at `b=0`. | Independent placement effects require a revised experimental design. |
| 7: tokenisation artefacts | Added decoded-text normalisations and intervals; the advantage survives. | The report's assertion that normalisation must exaggerate/reduce the ratio was not assumed before rescoring. |
| 8: document difference/ratio intervals | Added paired crossed-bootstrap intervals, labelled post hoc and pointwise. | None for this uncertainty report. |
| 9: high R² and weak prediction | Specified numeric features, penalty, intercept and categorical interactions; added matched-feature ridge and OLS diagnostics on existing outcomes. | No new untouched-seed confirmation is claimed. |
| 10: FP64 comparability | Clearly distinguishes FP64 model/backward computation, FP32 loss and FP16 capture storage from GPT-2's execution. | The failed FP32 attempt does not separately identify quantisation or learning rate as its cause; no standard-FP32 replication claim. |
| 11: index mechanism and positive bias | Added component-specific results and complete permutation reference analysis, including the fixed-VMI conditional reference. | A specified permutation reference is not a universal privacy certification. |
| 12: held-out post-training gradients | The recorder's backward procedure and absence of updates are explicit throughout. | Actual training-trajectory leakage requires appropriate captures; this draft does not claim to measure it. |

## R3: point-by-point disposition

| Comment group | Change in this draft | Remaining work |
| --- | --- | --- |
| Major 1: generalisation | Main gradient comparison is GPT-2 only; Qwen replicates the placement study. Quantitative claims are conditional on data, length, decoder, checkpoint and budget. Treating activations as sensitive is presented as an engineering precaution. | No broad quantitative transport to larger models or other domains. |
| Major 2: stronger attackers | Describes the controlled ablation and defended lower bounds without optimality claims. | Stronger/adaptive attacks are additional evaluations, not silently assumed completed. |
| Major 3: defence ranking | Replaces rank-reversal language with differences in apparent protection, states attack/metric scope and clarifies the utility pairing. | No semantic/downstream-exploit or deployment-wide defence verdict. |
| Major 4: historical audit | Introduction explicitly separates current source/input records from withdrawn historical claims. The historical appendix remains intact. | No rerun is inferred merely from historical disclosures. |
| Major 5: explanation versus prediction | Reports original and post-hoc feature representations and results separately. | Already inspected held-out seeds remain diagnostic; independent causal interpretation is removed. |
| Minor 1: phrasing | Uses precise descriptions of exact reconstruction and accuracy loss while preserving plain prose. | No need to make all readable phrasing impersonal. |
| Minor 2: presentation | Harmonises the aggregate table to two decimals, adds explicit appendix pointers and repairs crowded placement-table spacing. | Full visual QA accompanies the build. |
| Minor 3: statistical status | Labels all added intervals, null references and sensitivity tests as post hoc; preserves original test families. | No retrospective preregistration. |
| Minor 4: AI-writing percentage | Gives no evidentiary weight to a percentage inferred from writing style. | Authorship/disclosure decisions must be factual, not based on this estimate. |
| Minor 5: fraud screening | Preserves factual provenance and distinguishes integrity checks from authenticity or regeneration of predictions. | Speculative style screening is not evidence for or against experimental validity. |

## Decisions before any fresh training

| Scientific question to retain or add | What is required | Fresh victim training? |
| --- | --- | --- |
| Independent effect of server-trained segment placement at a fixed observation boundary | A new design that separates boundary, frozen/trainable prefix and trained segment. A frozen prefix with only downstream training cannot establish changing activation inversion. | **Yes**, for an independently manipulated training configuration. This causal claim is removed from the current draft. |
| Leakage while a victim actually trains | Inspect the retained training captures, establish their targets/checkpoints and score the appropriate observation surface. | **Possibly**; first assess existing trajectory evidence. A fixed-final-checkpoint probe cannot answer it. |
| Joint attacks against secret mixup or replica-median | Define which primary and auxiliary gradients are observable; locate sufficient defended observations and final model state, then run new attacks. | **Possibly**. The main runner retained client heads and observations, not a guaranteed complete final server checkpoint. Do not assume that every new defended capture can be produced without training. |
| Larger SipIt budgets, beam widths, calibration sensitivity or an adaptive baseline | New decoding/calibration/attack evaluations, with costs and stopping rules fixed beforehand. | **Not inherently**; retained observations and the required attacker-side state may suffice. |
| Longer passages, new domains or stronger architecture coverage | New inputs and capture/attack evaluation; training depends on the claim and design. | **Design dependent**. Prefix rescoring is not a substitute. |

The revised paper no longer depends on an independent causal start-position claim. Fresh training is therefore a decision about the intended contribution, rather than a prerequisite for the completed retained-data analyses. No new GPU work, external publication, repository tag movement or arXiv submission was performed for this revision.
