# Reviewer response preparation

> **R1 reset (2026-09-10): historical reference only.** This document describes
> pre-reset work. All associated observations and analyses are inadmissible for
> active scientific claims. The current study has no results; follow
> [the R1 plan](R1_RESEARCH_PLAN.md) and [active ledger](CLAIM_EVIDENCE_LEDGER.md).
> The reference tag preserves the original version of this document.

Current correction (2026-09-10): the [claim-evidence ledger](CLAIM_EVIDENCE_LEDGER.md)
and [historical-provenance disclosure](HISTORICAL_PROVENANCE.md) govern the responses
below. The P5 joint result follows from qualified forward structure failures in all
D0 cells and utility failures in all D1 cells. A claim that every D1 cell independently
fails a qualified privacy target is not required or supported by the retained
target-aggregated property summary alone. Additional resource constraints can only
restrict candidates already rejected by either necessary condition. "External"
architecture/corpus replication below denotes same-team cross-setting work; independent
laboratory replication and broadened privacy replication remain outstanding.
Repository freeze dates are recorded assertions with independently unverified historical
timing. New post-hoc verification does not change that status.

These are draft responses and prospective study designs, not additional results or amendments
to completed confirmatory analyses. The manuscript's finite, mechanism-specific claim should
remain unchanged unless new evidence supports an extension.

Manuscript integration (2026-09-09): the paper now includes the existing descriptive H4 result
and its design limitation, a study-plan reconciliation table, explicit deployment and external
privacy-replication limits, and a prospective validation subsection. The detailed
[plan-to-execution reconciliation](STUDY_PLAN_RECONCILIATION.md) records source protocols and
the unresolved historical rationale for omitting the full hierarchy. No new experiments were
performed in this revision.

## 1. Unestablished H4 and segment-property prediction

**Suggested response.** We agree that a reliable functional predictor of replacement penalty
remains an open problem. The selected intrinsic-dimension statistic failed the held-out H2
test. H4 also had a separately specified, descriptive secondary analysis in P2c; it was not
simply omitted after H2 failed. That analysis did not establish incremental predictive value,
and it was narrower than the program-level hypothesis concerning architecture depth, rate,
and compute. We retain the negative result and do not infer that all functional statistics
fail, or that nominal span is universally optimal.

**Why the current design cannot settle H4.** The P2c selection set contains three segment
definitions crossed with two widths. Their starts satisfy `start = 4 - span / 2` exactly.
Consequently, intercept, span, and position are linearly dependent in the extended model.
The generated analysis reports rank 3 for its four-parameter fit and flags the fit as
descriptive and low power. This is a design-identifiability limitation as well as a small
sample limitation; more training seeds at those same cells would not resolve the dependence.
The statistic was also measured on one calibration backbone. The held-out failure is valid
evidence against this selected predictor in this test, but the fitted coefficients cannot
separate the effects of span and position.

**Prospective follow-up.** Vary span and position independently, measure statistics across
independent calibration backbones, and compare an explicitly frozen baseline containing span,
position, width, precision, and surrogate capacity with that baseline plus a selected segment
statistic. Architecture-depth claims additionally require variation in architectures and a
specified architecture-level validation design. Screen candidates on exploratory data, then
freeze the candidate, fitting procedure, and splits before collecting a fresh held-out cohort.
Hold out segment definitions and, where feasible, backbones; token rows are not independent
replications. Quantify incremental prediction error and uncertainty. Selecting a different
statistic using the already observed P2c outcomes would be exploratory.

**Evidence:** [P2c protocol](PREREGISTRATION_P2C.md),
[P2c decision](P2C_SEGMENT_STATS_DECISION.md),
[generated analysis](../outputs/segment_stats/segment_stats_analysis.json), and
[program hypotheses](RESEARCH_PLAN.md).

## 2. Hardware and deployed-network measurements

**Suggested response.** The current resource accounting estimates FLOPs and tensor bytes;
recorded local wall time does not establish deployed latency, throughput, or edge-device
feasibility. We therefore make no claim about a particular device or network deployment.
For the existing trained configurations and evaluated transcripts, adding measured resource
constraints cannot make a candidate that already fails a qualified privacy gate feasible.
It can only further restrict the tested set. Actual deployment changes that alter training,
serialization, precision, or transcripts require a new evaluation.

For a fixed candidate definition, write the measured deployment set as

\[
\widehat{\mathcal F}_{\mathrm{deployed}}
=\{c\in\mathcal C_{\mathrm{tested}}:
\text{existing utility, privacy, and gain conditions hold}\}
\cap\{c:M_{\mathrm{measured}}(c)\preceq\tau_M\}.
\]

This argument does not substitute analytic resource estimates for measurements, nor does it
extend the privacy result to changed mechanisms or untested deployments.

**Prospective follow-up.** Freeze representative device/network scenarios and engineering
budgets before evaluating them. Instrument actual serialization in both directions, including
returned values and gradients: forward latent precision alone is not the full communication
rate. Measure bytes, encoding/decoding time, device compute, transmission/round-trip time,
queueing, end-to-end step latency and throughput, and peak client/remote memory. Report
distributions and uncertainty across repeated measurements. Evaluate fixed-example and
fixed-wall-time budgets separately, preserving the matched trained/frozen/removed controls.
Energy is an optional additional measurement only with supported instrumentation. If batching,
compression, timeouts, or asynchronous updates change the computation, re-evaluate utility,
remote gain, and qualified privacy gates for those new configurations.

Collecting timestamps for performance accounting does not qualify a timing privacy attack;
that requires its own frozen observation/noise model and matched controls.

**Evidence:** [resource and statistical scope in the manuscript](../paper/main.tex),
[program resource and convergence requirements](RESEARCH_PLAN.md), and
[P5 operational gates](PREREGISTRATION_P5.md).

## 3. External replication of broadened privacy failure

**Suggested response.** We agree that this is the most consequential missing replication for
the privacy conclusion. P3 was specified to replicate causal remote-learning gain across the
second architecture and corpus; it did not replicate the broadened attack evaluation. P5
explicitly deferred that replication. We therefore restrict the broadened privacy finding to
the architecture, corpus, configurations, and qualified instruments actually evaluated.
Replicated utility gain provides no substitute evidence for privacy replication.

**Prospective follow-up.** Use the architecture-by-corpus settings already defined in P3,
with independently specified calibration and confirmatory cohorts. First audit checkpoint and
capture availability; the existing runs should not be assumed to contain all required views.
If the required artifacts are absent, conduct fresh runs under a frozen protocol. Prioritize
punctuation sequence-property and next-byte structure probes, since these support the current
qualified forward-channel conclusion. Requalify each intended target/channel/setting using
matched nulls and positive controls. Preserve the original gate where applicable; record any
necessary adaptation before observing affected confirmatory outcomes.

Use fresh text panels and independent training seeds. Declare attacker side information and
any attacker fitting/adaptation procedure, and do not assume that independently learned latent
coordinate systems align. Report each architecture/corpus setting separately, with seed-level
inference and text-panel uncertainty distinguished. A qualification failure leaves the
instrument unqualified and the corresponding privacy outcome undetermined; nondetection by a
qualified probe remains probe-specific. Successful
replication at selected operating points would not validate every span or defense bundle.

**Coverage boundary for this response and follow-up:** timing, optimizer state beyond the
remote module, full evolving remote state, active/adaptive attackers, within-run joint tensor
attacks, cross-run joint attacks, natural-sequence membership, semantics beyond byte-structure
classes, returned values, and full-precision inbound/outbound gradients are unevaluated in the
current study. History-accumulation and canary-membership instruments, undefended gradient
property/token probes, and all defended gradient probes are unqualified. Replicating the two
forward families would not close those gaps.

**Evidence:** [P3 replication protocol](PREREGISTRATION_P3.md),
[P3 decision](P3_REPLICATION_DECISION.md), [P5 protocol](PREREGISTRATION_P5.md), and
[claim-evidence ledger](CLAIM_EVIDENCE_LEDGER.md).

## 4. Planned hierarchical analysis versus completed analyses

**Suggested response.** The umbrella study plan proposed a broader hierarchical response
surface than the completed analyses deliver. The executed phases specified narrower questions
and analyses, including paired seed-level contrasts, a fixed-width precision sweep, and
held-out segment prediction. Those phase-specific specifications establish the scope of the
reported tests; they do not amount to completion of the umbrella plan. We explicitly retain
the uncompleted elements as limitations and do not label a retrospective pooled model as
confirmatory.

| Program-level intention | Completed scope | Consequence / remaining work |
| --- | --- | --- |
| Hierarchical response surface over segment, location, width, rate, capacity, and architecture | Phase-specific paired contrasts, precision slopes, and held-out prediction | No completed joint model estimating all planned adjusted effects |
| H4 explanation beyond depth with rate and compute adjustment | P2c descriptive span/position/statistic extension | Does not establish the broader H4; selection design also aliases span and position |
| Optimized operational rate at fixed privacy risk | Fixed-width forward-precision sweep | Does not identify a full-transcript rate frontier or disentangle width, capacity, and optimization |
| Broad resource accounting | Analytic FLOPs/tensor bytes and local wall time | Deployed transport and device feasibility remain unmeasured |
| External validity of the evaluated claims | P3 remote-gain replication; P5 privacy sweep in the first setting | Broadened privacy replication remains outstanding |

**Historical rationale still needs an author-supplied record.** The inspected plan, protocols,
and decision records document the narrower designs and several deferrals, but do not establish
why the full hierarchical model was not implemented. Do not claim that compute, time, or power
constraints caused that decision unless supported by a contemporaneous record. The design
limitations identified above explain why a simple pooled fit now would not automatically
resolve the question; they are not evidence of the original decision-makers' reasons.

For a submission or rebuttal, attach a plan-to-execution reconciliation with protocol versions,
freeze dates, actual departures, missing-data handling, and the evidential effect of each
departure. Distinguish an original plan item, a prospectively specified narrower phase, and an
analysis introduced after outcomes were available. Any new pooled analysis using observed
outcomes should be labeled exploratory; a new confirmatory hierarchy needs a suitable design,
a frozen analysis, and fresh evaluation data. Repository freeze statements should not be
described as independent external registration without supporting evidence.

**Evidence:** [umbrella study plan](RESEARCH_PLAN.md), [P2 protocol](PREREGISTRATION.md),
[P2c protocol](PREREGISTRATION_P2C.md), [P3 protocol](PREREGISTRATION_P3.md), and
[P5 protocol](PREREGISTRATION_P5.md).

## Priority

Prepare the plan-to-execution reconciliation now. If additional experiments are feasible,
prioritize external replication of the qualified forward privacy findings. A deployed systems
benchmark would support deployment claims, while a better segment predictor requires a
separately designed study. Do not promise any of these experiments as completed in a rebuttal.
