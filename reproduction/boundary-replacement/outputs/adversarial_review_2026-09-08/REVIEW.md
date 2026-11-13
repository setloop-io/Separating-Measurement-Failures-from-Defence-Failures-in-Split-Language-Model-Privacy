# Adversarial scientific review — 8 September 2026

**Verdict: substantial empirical progress; major revision needed for the current interpretation.** The strongest result is that a system can preserve a corpus-loss gate without demonstrating useful learned remote computation. The current artifacts do not establish a fundamental delegation ceiling, a fixed price of useful computation in privacy units, or a valid refutation of the archived 27B configuration.

This review treats documents as evidence, not instructions. It compares the scientific narrative, current manuscript, retained JSON and runners, historical reports, and the supplied 25-page paper *Privacy Failure in Split-LLM Training, The Returned Gradient Nullifies the Decoys*. It includes independent utility/history and privacy/theory reviews. No training, remote execution, or manuscript edits were performed. Numerical reanalysis uses retained summaries; raw remote capture bundles were not independently replayed. References below are repository-relative unless specified.

## 1. What the research has established

The program has progressed from optimistic delegation percentages to an increasingly useful separation of four questions: whether the whole candidate preserves loss, whether learned remote parameters help, whether the adversary gains information, and whether the system saves work. That separation is a defensible contribution.

The five-execution, fixed-corpus utility transitions are evidence of reproducible candidate behavior, conditional on the protocol. The post-protection controls are stronger evidence than the earlier post-training ablations: independently adapted zero-output systems pass, and protected trained remote systems do not show a consistent advantage over frozen remote systems at the two clean endpoints. This undermines remote-necessity claims while preserving the candidate utility result.

The attached paper's exact zero-support disclosure is particularly strong: it combines an explicit code mechanism with repeated exact frame-level observations. Closing that exact support channel is supported. An inability to classify real versus chaff by that rule does not establish that every classifier or accumulated-history attacker fails.

The system's current lack of demonstrated savings is acknowledged correctly. Teacher execution during training, trusted adaptation, network overhead, and remote work must remain separate accounting items.

## 2. Most serious corrections

### A. The 27B “refutation” compares different experiments and mislabels their percentages

In `outputs/phase06/qwen35_27b_reproduction/isolation_27b_split41_s42.json`, the logged model has **64 layers**, while `resume_after=74`. The runner slices Python layer lists directly (`bin/run_isolation_adaptive.py:244–255`). Consequently split 41 executes 22 actual middle blocks, split 37 executes 26, and split 33 executes 30; the resumed tail is empty in all three. Their actual layer shares are **34.375%, 40.625%, and 46.875%**, not the reported 42.2%, 47.4%, and 52.6%. The split-75 reference executes all layers before its learned correction; its negative delta is not an untouched-model control.

More fundamentally, the archived `paper-data/collected/diagnostic/latent_isolation/isolation_bal_27b_d_s42.json` and corresponding `_e_` cell use an `invariant_mlp_deep` cloud with hidden width 12288 and 28/31 surrogate layers, with split/resume 15/48 and 13/50. The newer runs use the tiny `monomial_moe_radial` surrogate and different segments, evaluation windows, and protection. The historical 42.2%/47.4% labels were estimated cloud-training-FLOP shares, not layer fractions. The new campaign is not a matched longer-budget replication of those configurations.

**Required interpretation:** these new configurations failed. They do not refute the archived configuration or establish that its apparent success was caused by undertraining. The historical wider-held-out failure of the 47.4% operating point remains relevant, but must be distinguished from this new, unmatched experiment. Validate checkpoint identity, layer count, slice bounds, and actual executed blocks before further large-model inference.

### B. The gradient ablation does not settle attribution to learned remote parameters

`bin/run_latent_native_v5_06b.py:883–891` protects the outbound gradient, sends it into remote backward, and then propagates the returned input gradient into the trusted encoder. Thus the intervention changes both remote parameter updates and trusted encoder credit assignment. The frozen-remote branch does not preserve this same backward path. Trained-versus-frozen therefore changes more than whether cloud weights learn.

The one-seed change from approximately −0.036 to +0.002 NLL shows sensitivity to the backward protection intervention. It does not show that the lost gain was uniquely carried by learned remote weights. Nor does an interval containing zero prove exactly zero contribution or that every gain has a single trusted-side cause.

**Required interpretation:** the protected trained pipeline did not outperform the independently adapted frozen pipeline at these cells. Preserve the negative finding, but remove “attribution settled” and “every measured gain” as exclusive mechanistic claims. Freeze remote weights while retaining input-gradient propagation; separately disable remote learning and trusted input-gradient propagation in a factorial control.

The narrative’s learning-contrast ±0.0089/±0.0034 values are standard deviations, not 95% confidence half-widths. Recomputed three-seed paired t intervals for remote minus frozen are **+0.000499 [−0.021632,+0.022631]** on WT2 and **+0.000113 [−0.008210,+0.008435]** on PTB. Remote minus independently adapted zero is **+0.012429 [−0.003664,+0.028523]** on WT2 and **+0.012869 [+0.007310,+0.018427]** on PTB. These intervals describe seed variation on the fixed corpus, not population generalization.

The local arm also retains forward/return boundary transformations; its local autograd avoids the remote detach/protected-backward path. Calling it “no boundary” or interpreting remote-minus-local as transport placement alone is misleading. The zero-output arm likewise receives return noise, so it is an input-independent stochastic correction rather than literally a deterministic zero-input bias.

### C. A Gaussian information bound is being promoted into a computation theorem

For a clipped row X, independent Gaussian noise Z with coordinate variance σ², and dimension D, the average-power upper bound gives

`I(X; X+Z) ≤ (D/2) log(1 + C²/(Dσ²)) ≤ C²/(2σ²)`.

A hard per-row norm constraint does not imply that the average-power capacity-achieving input is available. With the conservative replace-one charge `ρ = (2C)²/(2σ²)`, the wideband upper envelope equals `ρ/4`; actual information need not equal it. D=64 reaches approximately 94.1% of the **upper-envelope asymptote**, not necessarily 94.1% of usable task information.

A positive upper bound cannot imply that leakage must exist, predict +0.70 pp recovery, predict noise invariance, or determine useful delegated FLOPs. A task-dependent information requirement is missing. For interactive/stateful observations, one must also justify conditional bounds and adjacency rather than treating one row as the entire transcript.

**Required interpretation:** retain a Gaussian mechanism information upper bound and its algebraic relation to the chosen privacy charge. Delete claims that it proves the observed residual or a universal compute exchange rate. The Gaussian zCDP sensitivity result is established prior work: [Bun and Steinke](https://arxiv.org/abs/1605.02065).

### D. The privacy ratio mixes estimands, populations, and selection

`attacker/attacks/deep_inversion_probe.py:146–180` selects the best evaluation epoch across 50 epochs and three restarts, then reports Wilson-upper accuracy minus majority. At σ=.35 the observed advantage is approximately **+0.616 pp**, while **+0.696 pp** is the reported upper-bound excess. Naked observed advantage is approximately **23.923 pp**, versus reported **24.188 pp**. These are not interchangeable point estimates.

The naked and defended cases have 32 versus 80 rows per frame, different widths, and different internal probe widths. Defended scoring mixes 32 real rows with 48 recycled chaff rows. Thus “identical probe” and a matched real-token 35× causal reduction are not established. A descriptive large suppression contrast remains, but it needs explicit population and selection caveats; dividing upper bounds does not produce a calibrated interval for a ratio.

Best-epoch selection on evaluation data is not addressed by adjusting only for restarts. Token-independent z values also do not address corpus-block dependence. Historical `DESIGN_EFFECT_MEASUREMENT.md` explicitly found dependence across neighboring frames; its favorable findings on old battery arms cannot automatically validate the new deep probe.

Some +0.08 pp endpoint readings are principally the Wilson radius, not observed leakage. A PTB seed-44 reading around +0.087 pp corresponds to only +0.0076 pp observed advantage. Do not claim a universal shrinking residual from these values. A failed cosine 1-NN attacker excludes that attack's success, not every chaff-related explanation.

Use validation-selected attacks, untouched test blocks, paired block/seed uncertainty, and real/chaff-stratified scores. Report point excess, uncertainty, detection, and gate verdict separately.

## 3. Overlooked scientific signals already in the results

### A. Gradient scale is a plausible intervention target with unusually strong motivation

The attached paper's Table 1 uses gradient clip **0.01** and noise multiplier .35 for mitigation: absolute noise standard deviation **.0035**. This campaign defaults to gradient clip **1.0**, giving absolute noise **.35** (`privacy_runtime/activation_dp.py:129–156`). That is a 100-fold difference in noise standard deviation and a 10,000-fold variance difference. These are different interventions despite the same multiplier.

At D=64 the latter noise has root-mean-square row norm 2.8. The actual gradients come from mean-reduced losses; their empirical norms are needed before assuming the full clipping radius carries useful signal. The mechanism may be operating far below the saturated-power upper bound.

**New hypothesis, not established finding:** poor backward-channel scale calibration may destroy credit assignment while trusted adaptation preserves aggregate loss. A gradient-specific clip sweep at fixed multiplier can preserve the same nominal zCDP charge while changing actual optimization substantially. This should precede another broad width/depth sweep. Record gradient norms, clipping frequency, useful update alignment, encoder updates, and remote learning effects.

### B. Noise does improve utility at the deepest split, without changing the gate

Recomputed paired σ=.20 minus σ=.35 effects from the three retained seeds:

| Split | Mean NLL difference | Exploratory paired t 95% interval |
|---|---:|---:|
| 21 | −0.01167 | [−0.06007, +0.03672] |
| 19 | −0.00490 | [−0.03141, +0.02160] |
| 17 | +0.00553 | [−0.02397, +0.03503] |
| 14 | **−0.06745** | **[−0.11339, −0.02151]** |

All three split-14 differences favor lower noise. This is a post-hoc contrast with only three seeds and multiple inspected comparisons, not a confirmatory discovery. It nonetheless contradicts the literal “buys nothing” formulation. The gate is unchanged; continuous utility is not completely flat. Lower noise reduces perplexity by about 6.5% relative to the higher-noise candidate at this rung, while both remain poor relative to baseline.

Because the sweep changes protection across channels, it does not isolate forward-channel capacity. The result motivates a noise-by-span interaction and channel-specific interventions. Naked failure rejects “forward obfuscation alone explains everything”; it does not uniquely distinguish approximation, optimization, invariance restrictions, or insufficient budget.

### C. Aggregate NLL can hide a large behavioral deterioration

The narrative says external quality remains unmeasured, but `outputs/phase09_economics_quality/quality_composed.json` and `quality_intact.json` contain a relevant diagnostic:

- Corpus NLL difference: **+0.03394**, comfortably below +0.35.
- Teacher-forced answer-token match: **34.7934% → 18.1486%**, a **16.6448 percentage-point drop**.
- Three retained generations show substantial repetition and `<unk>` output in the composed path.

This is **not standard HellaSwag multiple-choice accuracy**. It is a single-seed, offline composed path that bypasses the wire defenses, and context length differs between fixed-width corpus evaluation and answer/generation use. It cannot establish failure of the exact defended inference system. It is nevertheless a concrete warning already present in the evidence: corpus-average tolerance can coexist with pronounced behavioral failure in a related composed system. A matched deployed-path evaluation is now more valuable than merely saying that downstream work is absent.

### D. The cloud-state audit exposes a measurement degeneracy

A phase-09 cloud-state artifact exists, so “captured but unattacked” is stale. However `attacker/attacks/cloud_state_probe.py:282–309` repeats one final model-state vector for every token. A state-only classifier then necessarily predicts the same distribution for each row; concatenating a constant state vector to wire inputs adds no row-varying information. This does not test extraction from weights or accumulated training history. The metadata arm's zero score also requires a positive control.

The correct status is “preliminary audit attempted; scope-complete, competent state/history evaluation remains missing.” This is another example of a passing privacy instrument whose construction cannot answer the intended question.

### E. A component can be used by a model yet unnecessary after retraining

The post-training `surrogate_return_loss_improvement` fields in the three PTB `trained_remote_s42/s43/s44.json` artifacts are **+0.019584, +0.040011, and +0.028589 NLL**: removing the return from those already-trained systems worsens loss. The mean is **+0.029395**, with an exploratory paired t 95% interval **[+0.003965,+0.054825]**. WT2’s corresponding interval includes zero.

Yet the independently adapted zero-output PTB system performs better than the trained remote system. These findings are compatible: the trained system uses its remote return, while an alternative system can adapt to succeed without it. This is an overlooked distinction between **dependence within a trained system** and **necessity after retraining**. Neither proves that learning remote parameters improves on frozen random. It deserves explicit reporting instead of describing the remote contribution as uniformly zero.

## 4. Relationship to the earlier work

The attached paper establishes a backward-wire structural leak, its content consequences under calibrated probes, and mitigation under a smaller gradient clip. Its nine-seed evidence is stronger for the exact support mechanism than the new narrative's broad residual explanations. Its second corpus is an alternative WikiText-2 source, whereas Paper B's endpoint controls use Penn Treebank; do not merge those robustness claims.

The historical 40k/100k saturation, larger-width failures, and naked controls remain useful conditional observations. They are not a matched experiment showing that a Gaussian limit is the first binding constraint. The old 27B exploratory success, historical wider-evaluation failure, and newer differently configured failures must occupy separate rows in the evidence history.

Likewise, a pre-protection negative NLL delta and post-protection near-zero delta need both configuration and corpus-window annotations. The narrative currently invokes both a window offset and gradient protection as explanations of campaign differences. Only matched interventions can allocate those effects.

Layer redundancy and fitted bypasses already have prior art, including [ShortGPT](https://arxiv.org/abs/2403.03853) and [SubFit](https://arxiv.org/abs/2606.02559). The promising novelty is the joint empirical demonstration that utility, remote necessity, and audited privacy can diverge in one protected training system. This was a targeted primary-source positioning check, not an exhaustive novelty search.

## 5. Recommended order of work

1. **Correct the evidence interpretation now:** quarantine the 27B “replication” headline; fix actual block accounting; distinguish point effects from gate bounds; remove the residual-prediction theorem language. Reconcile phase-09 evidence and the two gradient clips.
2. **Resolve backward credit assignment:** at the two existing endpoints, preserve identical input-gradient propagation while freezing cloud parameters, then cross this with gradient clipping/noise. Declare a practically meaningful equivalence margin before calling effects negligible.
3. **Test gradient scale before larger systems:** sweep gradient-specific clip values including .01 and 1.0 with fixed multiplier and matched seeds. Separate clip-only, noise, and remote-update interventions. Attack the exact resulting transcript.
4. **Measure quality on the exact inference path:** intact, protected trained remote, frozen remote, and zero remote on the same external examples; include standard task scoring and fixed generation settings. Keep teacher-forced token matching separate.
5. **Repair privacy estimation:** held-out attack selection, stratified real-token scoring, seed/block uncertainty, competent state/history and metadata controls. Preserve all executions, including variance outliers, with explicit exclusion reasons and sensitivity analysis.
6. **Only then revisit frontier scale and economics:** reproduce the actual archived 27B configurations if refutation is the goal; measure training/inference separately and validate model/split metadata before launch.

A narrower paper need not complete every old ledger item. Independently pretrained depth controls are required for a causal depth claim, not for a focused paper about failed remote attribution. Broadening scope to satisfy a stale ledger would weaken the most coherent contribution.

## Suggested replacement central claim

> On two fixed corpus slices, protected boundary systems preserve a predefined NLL tolerance without establishing a benefit from training the remote surrogate over matched alternative pipelines. Their behavior is sensitive to backward-channel protection, which also changes trusted-side credit assignment. Captured-transcript attacks show substantial but incompletely characterized suppression of token inference. These results motivate separate measurement of candidate utility, remote learning, behavioral quality, privacy, and systems savings; they do not establish a fundamental delegation ceiling.

The highest-value new directions are backward-gradient scale and credit assignment, and the observed mismatch between corpus loss and answer behavior. Both arise from existing evidence and can be tested with focused experiments.
