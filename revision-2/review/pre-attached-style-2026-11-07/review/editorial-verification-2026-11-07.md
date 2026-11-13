# Scientific and editorial verification

Reviewed 7 November 2026 against the preserved baseline in `review/pre-edit-2026-11-07/` and the user's supplied Academic Manuscript Rewriter instructions. Scope covers the current main manuscript, all six sections, references, artifact display names, and template provenance. The verifier made no changes to manuscript sources, code, data, figures, or the original paper. Final rendered-PDF punctuation and visual layout checks are handled separately by the parent agent.

## Disposition

**Pass with no remaining scientific or source-level editorial blocker identified.** The rewrite preserves the verified retrospective analysis while replacing internal campaign shorthand with descriptive names and simplifying the prose. The changed notation, prose, and template do not strengthen the scientific conclusions. The manuscript still distinguishes recomputed arithmetic, archived extraction, source-reported results, and analytic arguments.

The available evidence supports a completed retrospective case audit. The manuscript continues to make no claim of a new independent validation cohort, newly executed training, stronger attack fitting, field-wide error prevalence, or a universal privacy guarantee. The editorial work does not create a need for new GPU experiments.

## Scientific invariants

| Item | Verification |
| --- | --- |
| Exploratory campaign | 19 configurations, four training seeds each, 76 configuration–seed contrasts, 232 source-reported runs |
| Overlapping rejection reasons | 13 qualified privacy breaches, five utility failures, seven gain failures |
| Privacy-undetermined cases | Six configurations, each independently rejected on utility or gain |
| Disjoint reason intersections | 10 privacy only, three gain plus privacy, three utility plus gain, two utility only, one gain only |
| Exploration intervals | Two-sided 95% Student-t, four seeds, df 3, critical value 3.182446 |
| Six-layer-span utility example | Mean 0.0999705344 and upper endpoint 0.1195024415 both pass the 0.10 and 0.12 criteria |
| Passing utility margins | Approximately 2.9466e-5 and 4.9756e-4 nats |
| Confirmation planning | Eight proposed seeds and approximately 0.263772 conditional upper bound remain planning quantities, not confirmation outcomes |
| Defence pilot | Seven configurations, three paired seeds each, 66 run records, 63 split records, 126 candidate score records |
| Pilot outcomes | Five utility passes and two failures, seven gain passes, four failed and three undetermined composite privacy outcomes, no eligible configuration |
| Pilot utility uncertainty | One-sided 95% upper Student-t limit, three seeds, df 2, critical value 2.919986 |
| Probe units | Punctuation correlation advantage remains distinct from next-byte classification-accuracy advantage |
| Wrapper recovery crossings | 258.32, 355.81 and 617.93 are archived absolute 50% crossings, with brackets retained and no asserted crossing interval |
| Pooled versus individual sessions | 3.71% pooled remains distinct from the individual 5.73%, 3.65% and 2.34% examples |
| Deployment evidence | 20 of 24 identity outcomes, four of eight runs with known limitations, and the no-rotation status of N=256 are retained |
| Runtime example | Recorded rates 2.8 and 7.6 tokens/s yield a 63.16% reduction at recorded precision |
| Secondary-analysis scope | 118 consumed input files and 1,522 internal consistency checks remain separate from independent experimental validation |

The code and saved audit data remain byte-identical to the scientifically verified baseline. The audit JSON SHA-256 remains `98d7ffa2f73c57d761f6b61f19161eb4b39d32b44993e8d32d78d408e731ef19`. The analysis script SHA-256 remains `1a0fa7630309989c99482fa1fd7d5bd81cfbf1873d0ca65afaf6364e917446f1`. Because neither changed, the prior isolated execution and independent CI verification remain applicable. This editorial pass did not rerun training or attack code.

## Equations and interpretation

All three displayed equation blocks match the baseline after accounting for the requested mutual-information notation change.

- The exploration interval retains its two-sided critical quantile, standard deviation, square-root sample-size denominator, and degrees of freedom.
- The orthogonal wrapper and inverse retain `h′ = hQ` and `h = h′Qᵀ`. The distinction between a no-key wire observer and the compromised restoring host remains explicit.
- The variational lower-bound objective retains the same joint-versus-product expectations and inequality direction. `MI(X,Z)` is explicitly defined as mutual information. Replacing the notation `I(X;Z)` changes no mathematical relationship.

The constant-critic counterexample remains valid. A uniform discrete variable copied into its observation can reveal eight bits for 256 symbols while the constant critic has a zero objective. This remains an analytic argument rather than a synthetic experiment or a measurement of the defended system's actual leakage. The rewrite preserves the separate finite-sample warning and does not infer high leakage in every defended configuration.

The observer proposition still requires the stronger observer's admissible class to include each admissible weaker-observer attack composed with projection, within the stated resource allowance. Its proof remains an existence argument, without a guarantee that a particular optimiser discovers the attack.

The original Python slices are now expressed as representations of `(x₁,…,xT−1)` scored against `(x₂,…,xT)`. This correctly preserves the one-token shift and the interpretation that a row representing the current token was scored against the next token. It introduces no replacement experimental result.

## Scope and evidential status

The rewrite retains the important qualifications at the point where readers need them.

- Exact decoy-row partition identification remains separate from token-identity recovery. The approximately 0.01 nats measures utility cost, not residual leakage.
- The six separately trained zero-return comparisons and the 16.64-percentage-point answer-token decline remain attributed to earlier audit records. They are not promoted into new reconstruction experiments or replicated deployment evidence.
- Missing qualification still blocks a conjunctive privacy pass without erasing qualified breaches or independent utility/gain rejection reasons.
- Atomic pilot measurements remain separate from the archived composite verdict policy. A missing property binding can leave that composite outcome undetermined while an applicable structure probe records a large signal.
- Labelled-pair attacks, pooled-session versus individual-session outcomes, incomplete identity restoration, model/cadence differences, and the limits of laboratory key-lifetime crossings remain visible.
- Empirical saturation among finite tested architectures remains distinct from an information-theoretic ceiling. Non-detection remains finite and attack-relative.
- Retrospective selection, shared evidence, fixed evaluation panels, small seed cohorts, and the absence of independent diagnostic validation have not been dropped or weakened.

## Editorial and template checks

The new main file uses the vendored `arxiv` style. All three template-related files match the pinned hashes in `template/provenance.json`, including the unmodified style and its licence. The provenance correctly describes a community preprint style suitable for arXiv, rather than an arXiv-issued or mandatory template.

Artifact display names are now descriptive. All seven internal artifact IDs have display mappings, while their underlying hyperlink keys, citation keys, and immutable provenance paths remain available for source traceability. Internal campaign labels do not need to appear in the scientific narrative. The opaque quantisation configuration codes identified during this pass were replaced with the one-bit and two-bit quantisation descriptions. Meaningful span, depth, width, and cadence parameters remain legitimate scientific descriptions.

After excluding non-rendered hyperlink targets, LaTeX reference/label arguments, and comments, the current manuscript sources contain no prose colons, literal semicolons, or question marks. TeX spacing commands such as `\;` are not printed semicolons. The caption configuration uses a period separator, and the citation configuration uses commas, addressing punctuation that LaTeX could otherwise generate. Final confirmation against the rendered PDF belongs to the parent's output audit.

The source-level word-count comparison was approximately 4.5% above the preserved baseline before the final minor wording adjustments, within the supplied rewrite instructions' approximate 5% budget. This is a source-token estimate, not an exact rendered-body count. Sentence length and paragraph structure vary, and the technical vocabulary, denominators, and locally necessary scope restrictions remain intelligible. No AI-detector score or detector-evasion guarantee is asserted.

## References and preservation

The full citation-key set is preserved, with all eight bibliography entries still defined. Their descriptive public-preprint references remain separate from internal experimental records and audit notes. The appendix and limitations no longer assign published status to unpublished working documentation. The parent is checking the public bibliography metadata separately, and no new literature retrieval was performed in this pass.

All 11 original files covered by `preservation-manifest.json` still match their saved digests. The existing paper and repository README remain preserved. The pre-edit revision snapshot remains available for comparison.

## Remaining publication limits

The conclusions remain appropriate for a retrospective audit of one selected research programme. The paper does not establish a new broadly validated evaluator or a new secure defence. That scope is a publication-positioning constraint, not a missing-result placeholder. Acceptance still depends on the venue's interest in the cross-case secondary analysis and its relationship to the component studies. The style revision preserves those limits while making the actual completed contribution easier to read.
