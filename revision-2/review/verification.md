# Scientific verification — revision 2

Date: 6 November 2026. Scope: the complete new `main.tex`, all six section files, `references.tex`, `analysis/README.md`, the 458-line secondary-analysis implementation, generated audit data/macros, and selected underlying records. No manuscript, source, original-paper, or evidence file was modified by this verifier. The only persistent verifier output is this report. Final PDF rendering and visual QA are handled separately by the parent agent.

## Disposition

**Scientific-content verification passes for the revised, explicitly retrospective case-audit claim set. No remaining scientific blocker was identified after the three fixes below.** The manuscript now makes a completed secondary-analysis contribution using existing evidence. It does not misrepresent that work as new victim training, new attack execution, held-out diagnostic validation, or a universal privacy certificate. A new GPU campaign is not necessary to substantiate the claims the revision actually retains.

This is an internal verification of the supplied package and argument, not an independent experimental replication or a guarantee of acceptance. The independent contribution is now a cross-campaign reconciliation of admissible conclusions; its venue-level novelty remains more modest than a new defense or validated general evaluation framework.

## Executed checks

1. **Isolated full analysis execution — pass.** I loaded the current `analysis/recompute.py` into an isolated namespace, redirected its output directory to a temporary directory, and executed its main analysis. Every substantive field of the regenerated `audit_results.json` matched the saved output exactly; only run-specific metadata was excluded from equality comparison. The original revision outputs were not overwritten.
2. **1,522 consistency checks — pass.** The isolated execution completed the stored arithmetic/disposition checks. These are checks over the existing input set, not 1,522 independent experiments or independently selected test cases. The appendix and limitations describe that distinction correctly.
3. **118 input bindings — pass.** All 118 input-file byte lengths and SHA-256 values match `data/source_manifest.json`. The current analysis script matches the manifest's script hash, `1a0fa7630309989c99482fa1fd7d5bd81cfbf1873d0ca65afaf6364e917446f1`.
4. **Separate confidence-interval arithmetic — pass.** Independently recalculating the upper endpoints from the stored seed vectors agrees with all 19 R1 and all seven Paper 4 cell values to within `1e-12`.
5. **Original-file preservation — pass.** Every file listed in `preservation-manifest.json` still matches its recorded digest and byte length, including the existing `paper/` manuscript files and the repository README. The verifier's checks did not modify those files.
6. **Citation-key completeness — pass.** Every citation key in the main manuscript and section files has a bibliography entry.

## Numerical and inferential checks

### R1 exploration

- The 19 configurations, four training seeds per configuration, 76 cell–seed contrasts, and source-reported 232 completed runs remain distinct denominators.
- Overlapping reasons reconstruct exactly: **13 qualified privacy breaches, five utility failures, and seven gain failures**. Disjoint intersections are 10 privacy-only, three gain-plus-privacy, three utility-plus-gain, two utility-only, and one gain-only. They sum to 19.
- The six privacy-undetermined configurations have independent utility or gain rejection reasons. The revised interpretation therefore preserves valid negative findings instead of attributing all 19 rejections to unqualified instruments.
- R1 uses **two-sided 95% Student-t intervals**, four paired training-seed contrasts, `df=3`, critical value `3.182446305284263`. The manuscript gives the correct interval formula and conditions inference on fixed evaluation data.
- Span-6's mean is `0.09997053444385529` nats and its upper endpoint is `0.11950244147464331`. Both utility conditions pass. Margins below 0.10 and 0.12 are approximately `2.9466e-5` and `4.9756e-4` nats. The saved rejection is gain failure plus qualified privacy breach, not utility failure.
- The approximately 0.263772 all-family power bound is explicitly **source-reported conditional planning**, not a newly executed confirmation or a probability that the exploratory rejections are false.

### Paper 4 pilot

- The analysis consumes **66 archived run-metrics records**, including 63 split records, and checks **126 candidate score records**. Windows, matched nulls, candidate scores, and repeated bindings are not treated as additional independent training seeds.
- Outcomes reconstruct exactly: **five utility passes, two utility failures; seven gain passes; four failed and three undetermined composite privacy outcomes; zero eligible candidates**.
- Paper 4 utility uses a **one-sided 95% upper Student-t limit**, three paired seeds, `df=2`, critical value `2.9199855803537242`. This is correctly distinguished from R1's two-sided convention. The two utility failures are d3-q1 and d3-q2.
- Property advantages are Pearson-correlation differences over matched nulls; next-byte-structure advantages are classification-accuracy differences. Neither is reported as nats or automatically as current-token recovery.
- The manuscript correctly preserves the archived composite aggregation policy. A qualified atomic structure measurement can exceed its threshold even where an incomplete property binding leaves the recorded all-bindings privacy verdict undetermined. The revision does not silently substitute a different aggregation rule.

### Other recorded quantities

- E8's 27B crossings are **archived/extracted** absolute 50%-recovery crossings, not newly fitted attacks or a universal `0.09H` key-lifetime law. The values `258.3181`, `355.8136`, and `617.9250`, their brackets, and division by `H=5120` agree with the source extraction.
- A largest pooled session score of **3.71%** is correctly distinguished from a largest individual session score of **5.73%**. The manuscript no longer treats the pooled score as a worst-session confidentiality bound.
- E-R identity accounting is **20 of 24 sessions**, with four of eight runs admitted with known limitations. N=256 is correctly identified as a no-rotation control under this short horizon. Per-session and per-epoch results are not pooled into one cost-free deployment claim.
- The recorded N=16/RTT=0/session-1 rates **2.8 versus 7.6 tokens/s** imply a **63.1579% reduction**, rounded to 63.16%. The current prose identifies the actual identity record and avoids claiming more hardware-measurement precision than the recorded rates support.
- F1's exact real/decoy partition recovery, content-probe effect, and approximately **0.01-nat utility cost** remain separate quantities. The gradient direction is now correct. F2's shifted-target defect, six zero-return comparisons, and **16.64-percentage-point** answer-token decline are explicitly attributed to the earlier source audit rather than newly revalidated here.

## Logical and security-scope checks

**Observer monotonicity — pass after assumption clarification.** The corrected proposition now requires B's admissible class to include each admissible A-attack composed with projection to A's view, within the stated resource allowance. Under that nesting assumption the simulation proof is valid. It guarantees the existence of an attack with the same success, not that a particular optimizer finds it. It supplies no upper bound from a tested attack's non-detection.

**Orthogonal wrapper — pass.** `h' = hQ` and `h = h'Q^T` correctly distinguish a no-key wire observer from a restoring host that holds or derives the key. The draft no longer promotes passive-wire results to secrecy from that compromised host. It also acknowledges preserved norms/inner products and stronger labelled-pair attacks.

**MI counterexample — pass.** The population DV inequality is in the correct direction. For uniform discrete X and Z=X, a constant critic has objective zero while the true MI is `log(m)` nats; `m=256` gives eight bits. The text clearly labels this an analytic counterexample, not a new synthetic experiment or an estimate of any defended cell's true MI. It neither treats empirical non-detection as an upper privacy budget nor reverses the result into proof of high leakage in every defended cell.

**Capacity and universality — pass.** Flat results among finite tested architectures are now empirical saturation, rather than an information-theoretic ceiling. The abstract, contributions, and conclusion avoid the original universal “only structure leaks/defends” claim. Scope, uncertainty, and unmeasured attacks constrain conclusions throughout the manuscript rather than appearing only as a final disclaimer.

## References and source access

There are **eight bibliography entries but seven explicit external URLs**. All seven direct HTTP checks returned **200**, with HTML titles matching the referenced paper IDs/titles:

- [2609.04382 — returned-gradient case study](https://www.alphaxiv.org/abs/2609.04382)
- [2610.04128 — gradient contribution study](https://www.alphaxiv.org/abs/2610.04128)
- [2610.22722 — LLM-Attack Suite](https://www.alphaxiv.org/abs/2610.22722)
- [2609.09794 — privacy-preserving split federated fine-tuning](https://www.alphaxiv.org/abs/2609.09794)
- [2610.07677 — adaptive model inversion](https://www.alphaxiv.org/abs/2610.07677)
- [1811.04251 — formal MI-measurement limitations](https://www.alphaxiv.org/abs/1811.04251)
- [1910.06222 — variational MI-estimator limitations](https://www.alphaxiv.org/abs/1910.06222)

The eighth entry is the accompanying local research-program artifact collection, linked through the appendix's relative paths and source index; it has no external URL to test. That is honestly stated by the availability paragraph. A stable reviewer/public artifact locator will still be desirable for submission, but the manuscript does not invent a public repository or claim universal environment availability. This verifier did not initiate new literature discovery; the parent owns the broader literature synthesis and novelty assessment.

The source-level distinctions are suitable: R1/P4 stored arithmetic is recomputed; E8/ER quantities are archived extractions or explicitly attributed summaries; F1/B1 and planning power remain source-reported; and the MI and observer arguments are analytic. In particular, the broad wire-recovery observation should continue to be read as the source's finite decoder/capture result, not as a newly fitted attack or general security guarantee.

## Issues found and corrected by the parent

1. **Formal assumption:** the observer proposition initially left attack-class nesting implicit. The parent explicitly added projected A-attack inclusion within the resource allowance. I reread the corrected statement and proof.
2. **Artifact locator:** the appendix initially implied the frozen R1 attack specification was in the exploration bundle. The parent now gives its actual qualification-bundle path, which exists and is hashed by the analysis.
3. **LaTeX build:** an initial compile failed because an unprotected `\path` appeared in a moving caption argument. The parent replaced it with an escaped `\texttt` path. This report confirms the source correction; final three-pass compilation and visual PDF QA remain the parent's separate checks.

## Remaining publication limitations

The manuscript is now a complete, evidence-backed retrospective audit, not a future-work placeholder. Its scope nevertheless remains one selected research program, shared historical decisions and data, small training-seed cohorts, and archived evidence whose full original execution is not independently rerun. It cannot estimate field-wide failure prevalence or the sensitivity/specificity of a general audit procedure on unseen studies. Those limitations are explicit and consistent with the retained claims. The contribution's differentiation from the component papers and existing attack-suite infrastructure must remain visible in venue positioning. None of these limits logically requires new GPU work for the present, narrower manuscript.
