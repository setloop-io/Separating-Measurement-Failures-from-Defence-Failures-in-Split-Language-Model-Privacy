# Scientific and publication review of the current split-LLM paper

Date: 20 August 2026  
Artifact reviewed: `papers/arxiv-draft/main.tex` at `e8dfe48`  
Review standard: strong systems/security/ML venue; candid pre-submission review

## Executive verdict

**Recommendation: do not submit the current manuscript in its present form.** A strong
reviewer is likely to reject it because the central positive claim—private split LLM
training against a fully compromised cloud—is broader than the evaluated threat surface,
and several headline theoretical and empirical statements are contradicted by the
repository's own later audits.

This does not mean the research program has failed. The repository contains a potentially
important publication, but its strongest scientific result is different from the current
headline:

> An apparently rigorous privacy campaign can still create false confidence when the
> evaluated artifact omits part of the adversary's transcript, the attack metric is pinned
> by a degenerate constant predictor, and the positive control is also unreadable to the
> chosen attacker.

That is a sharp and consequential result. The full-view trace, gradient-channel experiment,
leakage-injection calibration, variance work, and design-effect audit are unusually candid
and could form a strong measurement/security paper.

The capacity/delegation work is also promising, but it presently establishes an
**architecture-dependent empirical association and an observed surrogate plateau**, not an
information-theoretic ceiling and not a causal result that model depth determines
delegability.

My recommended strategy is **two papers, developed sequentially rather than submitted
immediately**:

1. A privacy-evaluation and failure-analysis paper, after one decisive full-transcript
   experiment and a narrow rewrite.
2. A controlled delegability/frontier paper, after new depth-controlled, multi-seed,
   longer-horizon experiments and correct end-to-end compute accounting.

## What the paper actually studies

The manuscript says it studies private split LLM training and inference on a fully
compromised cloud. The implementation supports a narrower description:

- The pretrained LLM is loaded in evaluation mode and all of its parameters are frozen
  (`bin/run_isolation_adaptive.py:198-202`,
  `bin/run_latent_native_v5_06b.py:239-243`).
- The trusted node runs the frozen prefix, the original frozen middle segment as a teacher,
  and the frozen tail; it trains a private encoder/decoder.
- The cloud trains a replacement surrogate from distillation and language-model losses.
- The headline "training" experiments are therefore **frozen-backbone boundary-adapter and
  surrogate-distillation experiments**, not full-model or LoRA LLM fine-tuning.
- The trusted node still executes the original middle segment on every training step to
  obtain `middle_target` (`bin/run_isolation_adaptive.py:227-238`). The reported remote
  share excludes this duplicated teacher computation.

This is still a valid object of study, but it must be named precisely. "Split LLM training"
will lead reviewers to assume that meaningful LLM parameters are being trained remotely and
that local compute is reduced. Neither is currently demonstrated.

The defensible current thesis is:

> On a small diagnostic corpus, a learned narrow activation boundary resists specified
> forward-latent token probes, while restricted gauge-equivariant surrogate modules show a
> model-dependent utility plateau when replacing frozen middle-layer spans.

That statement is interesting but substantially narrower than the title, abstract, and
conclusion.

## Scientific significance

### What is genuinely significant

1. **Threat-view completeness is load-bearing.** The project traced every protocol crossing
   and discovered that the primary attacker bundle did not represent the compromised
   training node's full view. The outbound output gradient is unclipped, unnoised, absent
   from the bundle, and absent from the privacy accounting. It reveals the real/chaff split
   exactly and carries measured token information. This is the paper's strongest finding.

2. **Privacy instruments need calibrated false-negative curves.** The controlled
   leakage-injection sweep shows that the gate can report exactly the clean reading while
   hundreds or thousands of rows contain exact token codewords. The cause—the maximum over
   arms being pinned by a constant predictor—is understandable and repairable. This is a
   reusable methodology contribution.

3. **Negative results across a real engineering campaign are valuable.** Rotation-only
   protection fails; excessive noise destroys utility; long frames can expose more
   structure; short private re-hardening is inadequate; naive CE-to-MI reasoning is
   vacuous; and several attractive composition ideas fail in implementation. These are
   useful findings if presented as scoped observations rather than steps toward a foregone
   positive conclusion.

4. **Architecture may strongly affect replaceable span.** The 64-layer 27B model behaves
   differently from the 28- and 40-layer models under this surrogate mechanism. That is a
   valuable hypothesis. A controlled study could connect split-system delegability to
   residual-update magnitude, layer redundancy, intrinsic dimension, or block influence.

### What is not yet a scientific contribution

- A general privacy guarantee for a fully compromised training node.
- An information-theoretic capacity ceiling for private delegation.
- A novel privacy/quality/offload trilemma. The general privacy–utility–efficiency tension is
  already explicit in closely related work.
- A causal law that deeper models are more delegable.
- Production-ready private training or inference.
- A new Byzantine-fault-tolerance result; the current median experiment is an implementation
  check of a standard construction.

## Fatal issues for the present submission

### 1. The evaluated artifact is not the compromised cloud's full training view

The manuscript states that bundles contain the exact released view, including gradients
(`main.tex:172-178`), and Table 1 says the cloud receives clipped, noised gradients
(`main.tex:251-255`). The implementation and the project's scope audit show otherwise:

- `cloud_k.grad` is sent to the cloud at
  `bin/run_latent_native_v5_06b.py:693-709`.
- `RemoteLatentCloud.backward` transmits it directly at
  `privacy_runtime/latent_protocol.py:167-176`.
- No outbound clipping or Gaussian noise is applied to that tensor.
- The attacker bundle at `bin/run_latent_native_v5_06b.py:942-961` contains forward views
  and labels only.
- The bundle's training views are regenerated after training and were not the actual
  per-step views sent during training.

The audit measured the omitted channel rather than merely reasoning about it:

- its zero support identifies all 32 real rows among 48 chaff rows in every captured frame;
- a gradient-only attack on 65,536 rows reaches `z = +14.98`;
- a shuffled-label control reaches `z = -0.99`.

That measured cell also had a leaking forward path, so it does not prove that the gradient
breaks a headline passing cell. It does prove that the omitted channel contains private
signal and that a forward-frame pass is not a full-node privacy result.

**Required action:** either restrict the paper to inference/forward-frame resistance or
capture and attack the complete training transcript. The phrase "fully compromised" cannot
remain attached to the current training results.

### 2. The positive control establishes attacker blindness

At the actual encoder-output boundary, the standardized probe battery lands at its floor on
both:

- a fully naked `D=1024` boundary with gauges, chaff, meaningful noise, and adversarial
  training disabled; and
- the defended `D=64` operating point.

Increasing probe training and restarts by four times does not separate them. This is not
evidence that the defense adds privacy. It is evidence that the attacker family cannot read
this representation at that location.

The correct causal conclusion is:

> The measured marginal privacy effect of gauges/chaff/noise is below the sensitivity of
> this attacker family at this boundary.

The manuscript instead treats the result as a calibrated defense success
(`main.tex:941-951`, `main.tex:1131-1136`). A reviewer will reject that interpretation.

### 3. The gate has a measured false-negative region

The gate takes the maximum upper confidence bound over nine arms, but near-constant
invariant arms score exactly at the majority control. They form a pedestal that hides
movement in the coordinate arms.

The controlled injection experiment shows:

- a coordinate-visible exact leak still passes at 5.5% row coverage and fails at 6%;
- a rotation-faithful exact leak still passes at 40% row coverage and fails at 50%;
- at 4% coordinate coverage, one arm recovers 377 injected rows, while the published gate
  is identical to the undosed reading to four decimals.

Therefore "at the statistical floor" cannot be interpreted as negligible leakage. It means
only that no scored arm exceeded the constant-predictor pedestal by the declared amount.

**Required action:** score each arm against its own undosed/negative-control distribution,
exclude degenerate predictors from the maximum, and report a calibrated detection curve for
every headline attacker family.

### 4. Freezing the attacker does not prevent evaluation overfitting

Freezing is useful because a falling score cannot be explained by weakening attacker code.
But v6 through v13 repeatedly tuned the defense against the same known nine-arm benchmark.
This is adaptive test-set reuse: the fixed benchmark becomes a training target for the
defense campaign.

**Required action:** freeze the final defense, then run a sealed, independently implemented
attack suite or external red-team evaluation. The new attacks must not be used to tune the
reported operating point.

### 5. Proposition 1 is false as written

The gauge-invariance proposition (`main.tex:1251-1270`) should be removed pending a new
proof.

- The positive-scaling group is noncompact and has no uniform probability measure.
- The implementation samples a clamped signed log-normal scale, not a Haar-uniform group
  element; clamping also breaks the stated group structure.
- A posterior over gauges cannot be called uniform without specifying a prior on the latent.
- Independent per-request gauges do not imply
  `I(h_t; h_tilde_1:T) = I(h_t; h_tilde_t)` when latents are correlated across repeated
  tokens, documents, shared model state, or training history.

A valid result may state invariance/equivariance of selected functions under orthogonal and
permutation actions. It cannot currently state posterior uniformity or no cross-request
accumulation.

### 6. The 4:1 "capacity price" is not an operational identity

The manuscript says useful work bought is `clip^2/(2 sigma^2)` nats and the privacy cost is
four times that, independently of width (`main.tex:913-917`, `main.tex:1126-1130`). The
project's own adversarial review correctly withdrew this claim.

For finite width, an AWGN upper bound has the form

`C_D <= D/2 log(1 + C^2/(D sigma^2))`.

`C^2/(2 sigma^2)` is its large-width asymptote. At the actual widths, the ratio is not
exactly four. More importantly:

- channel capacity is an upper bound under a communication model;
- zCDP rho is a worst-case privacy-accounting bound;
- neither is measured information used by the task;
- cross-entropy improvement in nats is not mutual information in nats;
- no rate-distortion or communication-complexity theorem connects the bound to achievable
  LLM loss.

Lowering noise from 0.35 to 0.30 at the 39.3% point also made both utility and privacy worse,
so the proposed frontier direction is not observed there.

**Required action:** remove the identity and replace "capacity ceiling" with "observed
surrogate plateau" or "empirical delegation frontier."

### 7. The 47.4% headline does not survive the stronger utility evaluation

The 27B balance cell is one seed, 500 steps, and ends at approximately `+0.325`, barely
inside the `+0.35` gate. The later dedicated inference artifact reports:

- quick utility approximately `+0.363`, outside the gate;
- held-out forward cross-entropy delta `+0.5699`;
- visibly degraded generation.

The other dedicated inference cells also exceed the same `+0.35` threshold: `+0.3633` on
0.6B and `+0.3937` on 35B. The table calls the underlying cross-entropies "PPL," but the
code averages `F.cross_entropy` and never exponentiates it
(`bin/run_isolation_adaptive.py:875-908`). Correct perplexities are approximately:

| Cell | Composed PPL | Intact PPL | Ratio |
|---|---:|---:|---:|
| 0.6B | 186.3 | 129.6 | 1.44x |
| 27B | 60.0 | 33.9 | 1.77x |
| 35B | 52.8 | 35.6 | 1.48x |

The honest present statement is **42.2% clean on the small diagnostic; 47.4% marginal and
unconfirmed, and failing the later larger forward evaluation**. The 35B inference cell also
uses noise 0.35, not the paper's stated validated hybrid-MoE operating point of 0.40.

### 8. "Training FLOPs delegated" overstates what is remote

The base model is frozen, yet cloud share is estimated with a training-FLOP heuristic based
on parameter counts. The trusted side also runs the original frozen middle teacher on every
step. The reported percentage is therefore not the fraction of measured end-to-end training
work moved off the trusted node.

The fragmentation defect accidentally supplies an additional warning: the fragmented cloud
workers never run backward or optimizer steps, yet the reported cloud-contribution metric
hardly changes. Training the remote worker accounts for roughly 0.7% of the metric difference
in that comparison. This shows that the chosen zero-cloud/closure metric does not isolate
cloud learning; it includes the much larger trusted encoder/decoder contribution.

**Required action:** report actual measured FLOPs/time/energy on each node and add matched
ablations:

- trained remote surrogate;
- frozen random remote surrogate;
- identically sized local surrogate;
- decoder retained with cloud correction zeroed;
- all-local compute with the same trainable parameter budget;
- no teacher execution when claiming inference savings.

### 9. MINE near zero is not privacy evidence

The Donsker–Varadhan/MINE objective provides an empirical lower bound whose finite-sample
estimate can be negative. A value near zero does not upper-bound true mutual information and
cannot certify that leakage is near zero. "First non-vacuous MI reading" and "MINE reads
approximately zero nats" should not appear as positive privacy claims
(`main.tex:715-721`, `main.tex:1300-1310`).

At most, the paper can report estimator behavior alongside strong positive controls and
explicitly state that it supplies no confidentiality guarantee.

### 10. Several artifact and reporting claims are too strong

- Important figure values are hard-coded rather than loaded from artifacts. Panel C of
  `fig_capacity_ceiling.py` is hard-coded from a Markdown report, and
  `fig_privacy_by_version.py` hard-codes its result table. This contradicts
  `main.tex:1463-1472`.
- The fragmented workers do not train in the reported v13.2 path
  (`bin/run_latent_native_v5_06b.py:661-677`), invalidating the claim that fragmentation is
  free during training.
- A seed-separation bug was fixed only in commit `e8dfe48`. Existing result artifacts were
  generated by earlier code in which the cloud received a seed related to trusted
  initialization. The fix does not retroactively validate the old runs.
- Many bibliography entries use incomplete author lists or organizations in place of
  authors. This is not publication-ready.

## Claim disposition matrix

| Current claim | Verdict | Defensible replacement |
|---|---|---|
| Privacy against a fully compromised cloud during training | Unsupported | Forward-frame resistance to named probes; full transcript untested and gradient channel leaks |
| Forward latent is at the statistical floor | Numerically true, semantically overstated | No arm exceeds its current gate; calibrated gate misses specified injected leaks |
| Composite defense caused the floor result | Not identified | Naked and defended boundaries are indistinguishable to the tested probes |
| Matching attacks fail | Narrowly supported | These two adaptations fail on these bundles; positive-control sensitivity required |
| Compromise battery proves robustness | Weak | No trend under one weaker probe and incomplete cloud view |
| Controller repairs privacy breaches | One gross breach demonstrated | It responds to an ungauged leak; its monitor is blind to smaller/other leaks |
| Byzantine layer provides integrity | Narrowly supported | Contains one large tested perturbation under identical replicas; no general active security claim |
| MINE estimates approximately zero MI | Not privacy evidence | The estimator returns near zero; true MI is not upper-bounded |
| Capacity ceiling is information-theoretic | Unsupported | Tested surrogate families exhibit an empirical plateau |
| Depth is delegability | Interesting hypothesis | One deep model has a higher observed frontier; depth is confounded |
| 47.4% is a clean pass | Contradicted by later evaluation | 42.2% clean; 47.4% marginal/unconfirmed and fails later evaluation |
| Exact 4:1 useful-work/privacy identity | Incorrect | An asymptotic relation exists between two nonmatching upper bounds |
| No cross-request accumulation | Incorrect as stated | Fresh gauges prevent direct coordinate alignment under specified assumptions |
| Fragmentation adds essentially no cost | Invalid experiment | Rerun after enabling remote backward and optimizer updates |
| Real split LLM training | Misleading | Frozen-backbone boundary/surrogate distillation |

## Capacity ceiling and delegation triangle: novelty assessment

### Capacity ceiling

The words appear novel in this narrow context, but naming an observation is not scientific
novelty. Closely related work already studies low-dimensional privacy adapters,
information-bottleneck privacy/utility frontiers, and split-system privacy/utility/latency
trade-offs. The current manuscript's possible novelty is empirical:

> Under a fixed class of gauge-compatible surrogate operators and a fixed defended boundary,
> increasing width, surrogate size, depth, or training budget often fails to improve the
> replaced-span utility.

That is publishable as a scoped saturation result if the cloud contribution is isolated and
the experiments are repeated. It is not yet a universal ceiling.

The 27B observation suggests a better scientific hypothesis:

> Delegability is governed by the functional compressibility of the replaced segment—not
> simply its FLOPs or the channel width—and layerwise residual dynamics may predict that
> compressibility.

This connects the work to established LLM layer-redundancy and depth-pruning literature. A
new paper can be novel by linking those representation properties to privacy-constrained
remote surrogates, not by claiming that depth alone is the cause.

### Delegation triangle

The triangle is a useful explanatory graphic, but it is not currently a scientific result.

- Privacy–utility–efficiency/stability trade-offs are already explicit in related split-LLM
  work such as MIXGUARD.
- Information-theoretic privacy-adapter work already describes a hard privacy–utility
  frontier controlled by bottleneck dimension.
- The project's own `DELEGATION_TRIANGLE_REASSESSED.md` concludes that the privacy axis is
  underdetermined, the delegation axis is mis-scaled, and the expected frontier direction
  fails in the 0.30-noise experiment.
- The figure first says all three corners cannot be achieved and then says the 27B nearly
  achieves all three. That makes it a model-dependent design tension, not a trilemma.

Use the triangle as motivation or a hypothesis map. Do not list it as a novel contribution
until a Pareto frontier is measured with sensitive privacy metrics and actual remote-work
accounting.

## Recommended two-paper program

### Paper A: full-view privacy evaluation for split LLM systems

Possible title:

> **When the Privacy Probe Passes but the Transcript Leaks: Auditing Split-LLM Evaluations**

Central question:

> What evidence is required before an empirical privacy claim about a compromised split-LLM
> endpoint is trustworthy?

Core contributions:

1. A full transcript manifest and source-to-wire audit for split training.
2. The omitted output-gradient channel, exact chaff-partition disclosure, and controlled
   gradient leakage result.
3. Leakage-injection calibration showing the false-negative curve of a standard max-over-arm
   gate.
4. Positive-control methodology: a defense result is not attributable when naked and
   defended views are equally unreadable.
5. Correct statistical treatment: paired arm/control comparisons, per-arm calibration,
   run-level variance, and block/corpus dependence.
6. A reproducible checklist or harness for full-view empirical privacy evaluation.

Experiments still required:

- Capture a headline forward-passing configuration's complete per-step transcript.
- Attack forward frames, outbound gradients, evolving cloud state, and their joint/shared-
  gauge invariants.
- Include sequence reconstruction, rare-token conditional accuracy, cross-entropy gain,
  membership/property inference, and semantic reconstruction.
- Run defense-frozen independent/adaptive attacks with calibrated positive controls.
- Apply the audit to at least two external or baseline defenses, not only this system.

This paper is closest to publishable because most of the distinctive evidence already
exists.

### Paper B: architecture-dependent delegability under protected boundaries

Possible title:

> **How Much of an LLM Can a Protected Boundary Replace? An Empirical Delegability Frontier**

Central question:

> Under a fixed protected representation and surrogate family, what properties of a model
> segment determine how much remote approximation can preserve task quality?

Core hypotheses:

1. Segment function complexity, not nominal latent width, controls replacement error.
2. Residual-update magnitude, CKA/representation similarity, intrinsic dimension, block
   influence, or layer-pruning redundancy predicts delegability.
3. Deeper models appear more delegable because their per-layer updates may be smaller or
   more redundant—not because layer count itself is causal.

Experiments still required:

- Same-family controlled depth variants, ideally trained with matched parameter/FLOP budgets.
- At least 3–5 independent runs per headline frontier point.
- Long-horizon confirmation for the 27B 42.2% and 47.4% points.
- Multiple corpora and downstream tasks; no conclusion from one 256-block WikiText window.
- Correct held-out NLL/perplexity, task accuracy, and generation-quality evaluation.
- Actual node-level wall time, FLOPs, energy, memory, and network traffic.
- Baselines: layer dropping, local/student distillation, random/frozen surrogate, unconstrained
  surrogate, bottleneck-only, and relevant privacy adapters.
- A clean separation between inference offload and adapter/surrogate training cost.

Do not use "information-theoretic ceiling" unless a valid rate-distortion or communication-
complexity result is proved. "Empirical frontier" is accurate and still interesting.

### What to do with Byzantine integrity and adaptive monitoring

Keep these as supporting components or appendices in Paper A unless expanded substantially.
The present evidence is too small for a third paper:

- Byzantine verification tests one obvious fixed perturbation under identical CPU replicas.
- The controller catches an ungauged posture, but its quick monitor was later shown blind at
  relevant operating points.

A separate integrity paper would need subtle bounded bias, adaptive poisoning, replay/state
divergence, heterogeneous GPU numerics, collusion, WAN-separated replicas, and realistic cost.

## Highest-value experimental plan

### Tier 0: correctness before more scale

1. Fix/rerun fragmentation with real remote backward and optimizer steps.
2. Confirm every old result affected by the seed-separation fix, or mark it historical.
3. Rename NLL as NLL and exponentiate only when reporting perplexity.
4. Generate every table/figure from an allow-listed manifest; eliminate hard-coded results.
5. Remove the invalid proposition and 4:1 identity from the manuscript.

### Tier 1: decisive privacy closure

1. Capture the full transcript of a forward-gate-passing configuration.
2. Run gradient-only and joint forward+gradient attacks on that exact run.
3. Fit per-arm negative-control distributions and implement a paired advantage statistic.
4. Run calibrated injected leaks at low coverage and rare-token/semantic targets.
5. Freeze the defense, hand artifacts to an independent attacker implementation, and do not
   retune after seeing the results.

### Tier 2: delegability causality

1. Build a controlled family with the same width/parameter budget but different depth.
2. Measure per-layer residual norms, CKA, block influence, and layer-drop sensitivity.
3. Predict frontier location before running the surrogate sweep.
4. Confirm the prediction on a held-out architecture family.

That prediction/validation design would turn "depth is delegability" from a post-hoc story
into a real scientific result.

## Rewrite plan if one narrow paper must be submitted now

If new experiments are not possible, write one negative/measurement paper and do not split
the current positive story into two weak submissions.

Suggested structure:

1. Introduction: why full-view privacy evaluation in split LLMs is difficult.
2. System and transcript: enumerate every object the endpoint receives.
3. Initial campaign: how the forward-only evaluation appeared convincing.
4. Audit 1: omitted gradient channel.
5. Audit 2: positive-control failure and injection-calibrated sensitivity.
6. Correct statistical and experimental protocol.
7. Lessons across the v1–v13 negative results.
8. Limitations and checklist.

Remove from the main story:

- the information-theoretic ceiling;
- the delegation triangle as a result;
- MINE near-zero as evidence;
- the v13 fragmentation result;
- the 47.4% quality-pass headline;
- most version-by-version implementation detail;
- strong claims about training, production, or complete compromise.

Three crisp contributions are stronger than the current eight-item list.

## Writing and presentation assessment

The draft is energetic and unusually transparent, but it reads like a lab notebook merged
with a rebuttal. Phrases such as "winning defense," "full stop," "the telltale," "the deep
reason," and repeated `So what` blocks advocate before the evidence is settled. A strong
paper should use neutral language and let the result structure carry the argument.

Specific changes:

- Cut the abstract to one question, one method, three results, and two limitations.
- Move the v1–v13 ledger and implementation inventory to an appendix/artifact guide.
- Replace eight contributions with three.
- Separate confirmatory experiments from exploratory/post-hoc cells.
- State the experimental unit, number of independent runs, and selection process in every
  headline table.
- Report effect sizes and uncertainty for utility, runtime, and privacy, not only gates.
- Replace "pass/fail" rhetoric with estimates and calibrated sensitivity.
- Avoid adversarial commentary about related papers (for example, "the same disease").
- Correct incomplete bibliography metadata and compare like-for-like threat models.

## Venue outlook

As written:

- Strong security/systems venue: reject.
- Strong ML venue: reject; theory and causal depth claims are not established.
- Workshop/arXiv: acceptable only with much narrower claims.

After the recommended changes:

- Paper A could be competitive as a security measurement/methodology paper if the decisive
  full-view passing-cell attack and external-defense audits are added.
- Paper B could be competitive as a systems/ML empirical study if depth is tested causally,
  quality is evaluated properly, and actual remote savings are measured.

## Final judgment

The project contains excellent research behavior—especially its willingness to preserve
failed runs, commission adversarial reviews, and accept damaging audit results. The paper
should now follow that evidence all the way.

The publication-quality claim is not:

> We solved private split LLM training and found an information-theoretic capacity ceiling.

It is:

> We discovered that forward-only privacy evaluation can pass convincingly while omitting a
> leaking channel, quantified exactly how the gate can miss injected leakage, and derived a
> stricter methodology for evaluating split-LLM privacy. Separately, we observed a promising
> architecture-dependent frontier for protected surrogate replacement that now deserves a
> controlled study.

That version is more honest, more novel, and more likely to survive expert review.

## Related primary literature used for the novelty assessment

- Noorbakhsh, Khalili, and Sehatbakhsh, *Defense Against Prompt Inversion Attacks: An
  Information-Theoretic Approach for LLM Collaborative Inference*, arXiv:2606.11592.
  https://arxiv.org/abs/2606.11592
- Chen et al., *The Art of Mixology: Mixup-based Obfuscation for Privacy-Preserving Split
  Learning in Large Language Models*, arXiv:2606.16801.
  https://arxiv.org/abs/2606.16801
- Noorbakhsh et al., *Inf2Guard: An Information-Theoretic Framework for Learning
  Privacy-Preserving Representations against Inference Attacks*, USENIX Security 2024.
  https://arxiv.org/abs/2403.02116
- Yang et al., *Differentially Private Label Protection in Split Learning*,
  arXiv:2203.02073. https://arxiv.org/abs/2203.02073
- Men et al., *ShortGPT: Layers in Large Language Models are More Redundant Than You
  Expect*, arXiv:2403.03853. https://arxiv.org/abs/2403.03853
- Zhang et al., *A Federated Splitting Framework for LLMs: Security, Efficiency, and
  Adaptability*, arXiv:2505.15683. https://arxiv.org/abs/2505.15683
