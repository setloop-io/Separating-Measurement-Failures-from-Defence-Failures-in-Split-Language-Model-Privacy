# R1 held-out confirmation review packet — draft 2026-09-17

**Status: prepared for owner review; not frozen, admitted, or authorized for execution.**
This document does not amend the completed exploratory protocol or turn exploratory
outcomes into confirmation. No new execution, cloud resource, or paid service is
requested by its creation. A separately reviewed scientific admission must bind
an implemented confirmation protocol, source, environment, inputs and budget before
any reserved-seed outcome is generated or inspected.

## Decision and scope

**Planning recommendation: do not launch the proposed eight-seed full-family
confirmation as an adequately powered test.** The exact marginal-power calculation
and conservative family bounds are now generated in
[`confirmation_power.json`](../evidence/r1-exploration-20260917/confirmation_power.json)
by [`plan_r1_confirmation.py`](../scripts/plan_r1_confirmation.py). Under the
observed exploratory means/SDs, the probability of confirming rejection of all
19 members is bounded above by **0.2638**. These are fragile planning assumptions,
not a prediction with calibrated uncertainty. They nevertheless do not justify
an 80%-power claim for this design. The reviewable design below remains an option
for a separately labelled precision/replication objective or for revision before
freeze; no additional training is launched.

The proposed confirmation repeats the complete 19-cell family in
`experiments/configs/r1_exploration.yaml`, using all eight reserved independent
training seeds, all three remote modes, and one matched all-local run per seed:

- Seeds: **64007, 64019, 64037, 64051, 64067, 64079, 64091, 64109**.
- Runs: **8 × (19 × 3 + 1) = 464**; 2,000 training steps each, 928,000 steps total.
- Modes: trained remote, frozen-random remote, removed remote, and matched all-local.
- Architecture, corpus, training schedule, initialization matching, capture schedule
  and causal controls: the exploratory scientific matrix, copied explicitly into
  the new packet rather than inherited from mutable files.

The complete family avoids selecting a favourable subset after exploration. Any
smaller scope requires a dated amendment naming the selected cells and explaining
that the resulting inference concerns only that subset. No historical pre-reset
outcome, checkpoint or calibration may enter this design. Completed calibration
and exploration remain separate cohorts and are never pooled with confirmation.

**Additional training is optional for an explicitly exploratory paper.** If the
owner chooses that route, finish evidence admission and report the exploration,
its uncertainty and its limitations; make no confirmatory claim. Confirmation is
not a prerequisite for honestly reporting an exploratory or undetermined result.

## Qualification limits and scientific question

The frozen calibration suite qualifies only input-byte probes on inbound and
outbound gradients in each of the two recorded defense categories: 4 of 24
instruments. For a cell, only its two applicable qualified gradient instruments
are available. Every forward/returned-value instrument and every next-byte-class
or punctuation-fraction instrument remains unqualified. Keep the frozen numerical
thresholds and explicitly report this missing coverage.

The proposed scientific question is whether the finite family contains a member
with acceptable expected utility loss, useful causal remote learning and acceptable
leakage under the required suite. **With the current qualification state no member
can receive a joint scoped pass.** Confirmation can establish a scoped rejection
or leave a member undetermined. It cannot establish complete-transcript privacy,
a universal trilemma, or impossibility outside the tested family.

An alternative instrument-development programme is a separate option: repair
null/positive-control validity on new, independently reserved calibration seeds,
freeze a replacement attack specification, then design its confirmation. Do not
use reserved confirmation seeds to develop attacks, loosen thresholds to admit
instruments, or retrospectively repair exploratory qualification. Its additional
workload and cost are not included below. Resolve whether qualification can validly
transfer across every cell within the broad `gradient-defense` category before
admission; unsupported cell-specific transfer leaves that instrument undetermined.

## Proposed estimands, gates and multiplicity

For each seed and cell, retain paired losses and compute
`ΔU = L_trained − L_all_local`, `GF = L_frozen − L_trained`, and
`GB = L_removed − L_trained`. The independent observation is a training seed;
token rows, captures and windows do not increase n. Shared all-local controls
induce cross-cell dependence; Bonferroni coverage below does not require
independence across cells.

This draft proposes population-mean thresholds of **ΔU ≤ 0.10 nats**, **GF ≥ 0.01
nats**, and **GB ≥ 0 nats**. These retain the exploratory practical margins but
introduce a new confirmatory decision rule; they do not reinterpret the original
mean-plus-descriptive-interval gates. In particular, the exploratory 0.12 upper
confidence-limit tolerance is not an alternative population threshold here.

For each applicable qualified gradient instrument, define `A` as the per-seed
input-byte accuracy advantage over its matched null, under the frozen score rule,
and the confirmatory scoped leakage threshold as **E[A] ≤ 0.02**. This expected
seed-level advantage is a new explicit inferential estimand. Also report the
original maximum-over-seeds gate separately; it is an observed engineering gate,
not a confidence statement about an unseen seed or a worst-case privacy bound.
Never replace an accuracy advantage with a leakage probability.

There are **M = 19 × (3 + 2) = 95** scalar endpoints in this draft. Freeze M even
if some endpoints later become unavailable. For every complete endpoint, compute
a two-sided interval

`mean ± t_(1 − 0.05/(2M), n−1) × sample_sd / sqrt(n)`.

Under the paired-seed Student-t assumptions, these give at least 95% simultaneous
coverage over the declared 95 means. Report the assumptions and all seed values;
eight seeds cannot establish normality. No significance-driven switch of test,
selective endpoint omission, or opportunistic increase in sample size is allowed.
If diagnostics or admissibility undermine the planned model, label the inference
undetermined and report descriptive results rather than silently substituting a
more favourable analysis.

A confirmatory **rejection** of a member requires at least one simultaneous bound
establishing a breach: lower bound of ΔU > 0.10, upper bound of GF < 0.01,
upper bound of GB < 0, or lower bound of an applicable qualified A > 0.02.
Failure to demonstrate a pass is not a demonstrated breach. A joint scoped pass
would require upper ΔU ≤ 0.10, lower GF ≥ 0.01, lower GB ≥ 0, upper A ≤ 0.02
for every required instrument, full qualification and complete admissible
outcomes. Missing qualification prevents that pass under the present suite.
All other members are **undetermined**. Empty-family inference requires a valid
confirmatory rejection for every planned member; one undetermined member prevents
that conclusion. Report the original exploratory engineering classifications
separately so these new inferential rules cannot overwrite them.

Sensitivity tables at utility margins 0.05/0.10/0.20 and gain margins
0.005/0.01/0.02 remain labelled sensitivity analyses. They cannot replace the
primary margins or rescue unqualified privacy channels. New privacy targets or
qualification changes require a new endpoint family and power calculation before
freeze.

## Power assessment and stop decision before admission

Eight reserved seeds are **not yet justified as adequately powered**. With n=8
and M=95 the critical t value is approximately **6.02998**, giving simultaneous
interval half-width **2.13192 × sample SD**. An unadjusted 95% interval would have
half-width 0.83602 × sample SD and is not the primary confirmatory interval.
These planning constants were calculated using `scripts/stats_utils.py::t_ppf`;
they are mathematical design calculations, not empirical paper results.

For a one-direction decision at a threshold, let d be the anticipated distance
from the threshold divided by the paired-seed population SD, in the direction of
the proposed decision. Under normal paired-seed differences, exact marginal power
is `P(T_(n−1, sqrt(n)*d) > t_(1 − alpha/(2M), n−1))`. The following executable
planning calculation uses SciPy (not added to the execution lock by this draft):

```python
from math import sqrt
from scipy.stats import t, nct
from scipy.optimize import brentq

alpha, M, n = 0.05, 95, 8
critical = t.ppf(1 - alpha / (2 * M), n - 1)
def power(d, n=n):
    cutoff = t.ppf(1 - alpha / (2 * M), n - 1)
    return nct.sf(cutoff, n - 1, sqrt(n) * d)
print('critical', critical)
print('d for 80% marginal power', brentq(lambda d: power(d) - .80, 0, 20))
# Replace d with each prospectively justified distance / paired-seed SD.
# A minimum-n calculation is planning only; it does not reserve or authorize seeds.
```

The executable calculation has now been run, without changing the repository
dependency lock:

```bash
uv run --no-project --with scipy==1.17.1 python scripts/plan_r1_confirmation.py
```

It consumes the verified 232/232 exploratory summary, preserves all four paired
seed values for every endpoint, and hashes its source/input in the output. Exact
noncentral-t power requires standardized distance **2.60960 SD** for 80% marginal
power at n=8 and M=95. The generated output reports both the observed signed
plug-in distance to each rejection threshold and a specified positive distance
of **0.01 nats** for each loss contrast or **0.01 accuracy advantage** for each
qualified privacy probe. The latter are design scenarios, not fitted effects or
new thresholds. SD multipliers 0.5, 1 and 2 expose variance sensitivity; they are
not a variance confidence interval.

For `width-64`, the observed GF is 0.000149712 with SD 0.00758736, giving only
**0.12371** marginal power to establish GF < 0.01 under the proposed simultaneous
rule. At a specified 0.01-nat distance that power is **0.13039**. Its stronger
plug-in rejection endpoint under this *new* confirmation protocol is GB < 0,
with power **0.14006**. Exploration rejected this cell on its GF engineering gate;
GB was reported separately, and the incomplete privacy suite remains undetermined.
Neither its observed qualified-probe values nor any planning scenario grants a
privacy pass.

Without assuming cross-endpoint independence, cell rejection probability is at
least the maximum endpoint rejection probability and at most their sum capped
at one. The all-family intersection is at least one minus the sum of cell failure
upper bounds (clipped at zero) and at most the smallest cell success upper bound:

| SD multiplier | All-19-rejected probability lower bound | Upper bound |
|---|---:|---:|
| 0.5 | 0.79594 | 1.00000 |
| 1 | 0.00000 | 0.26377 |
| 2 | 0.00000 | 0.02521 |

These are bounds conditional on the plug-in normal model, not empirical confidence
bounds on future power. Four exploratory seeds give fragile means and variance;
do not choose the optimistic half-SD row to justify execution. Before freeze,
review the intended effect distances, conservative variance assumptions and claim
scope. An 80% marginal endpoint target is not an 80% full-family target.

If n=8 is inadequate for the intended claim, either retain an explicitly limited
precision/replication objective, narrow the claim prospectively, or propose a
larger independent seed cohort in a new reviewed amendment with a new budget.
Do not begin and then add seeds after looking at confirmation outcomes. The
current runner admits calibration/exploration, not this confirmation design;
implement and validate confirmation admission and analysis before issuing a
packet. The full planning script has been executed with SciPy 1.17.1; its complete output and recorded-seed audit are retained with the exploratory evidence.

## Proposed owned-node workload and budget

Runtime basis: reported Toula exp002 execution completed 232/232 runs at
2026-09-17 16:04:38 UTC in 76,684.199 s (21.3012 h). This operational timing is a
budget input, not a scientific effect. Calibration cal001 took 26,815.616 s for
76 trained runs. The proposed workload preserves the exploratory mode mix,
so doubling exploration is the closer estimate: **153,368.398 s = 42.6023 h**.

| Resource | Proposed review envelope | Basis / condition |
|---|---:|---|
| Toula GB10 execution | 60 h / 216,000 s hard cap | 42.60 h central estimate plus 17.40 h contingency |
| Planned successful runs | 464 | Full family, eight seeds; no outcome-dependent stopping |
| Extra operational retries | At most 46, at most one per failed run | Shared 60 h and storage caps; documented device/OOM/I/O failures only |
| Active output on Toula | 320 GiB hard cap | Verified exp002 bytes imply 208.35 GiB central estimate plus headroom |
| Independent durable replica | 320 GiB reservation | Separate owned node; full inventory verification |
| Additional local analysis/workspace | 32 GiB reservation | Compact extracts, summaries and figures; no third full tree assumed |
| Aggregate new storage reservation | 672 GiB | Does not include or replace existing archives |
| Nominal accounting rate | USD 0.50 / GPU-hour | Prior owned-node amortization/power convention, not a cloud tariff |
| Nominal GPU accounting | USD 21.30 expected; USD 30.00 cap | Not an electricity measurement or a metered purchase |

The verified exp002 artifact total is **111,859,148,305 bytes**; doubling it gives
**223,718,296,610 bytes (208.35 GiB)**. This accounts for the full retained tree,
not just compact analysis exports. Operational checks supplied during this review
reported approximately 1.5 T free on Toula and 2.0 T on Koula before replica
creation. These are availability observations, not space reservations; recheck
actual free bytes after replication and immediately before budget freeze. Do not delete prior evidence to create space. Retry failures
remain append-only and consume the same aggregate caps; 46 additional attempts
are an operational ceiling, not a guarantee that all can fit. Budget exhaustion
stops execution and leaves incomplete outcomes visible and undetermined.

Use Toula only when uncontended; do not interfere with co-tenant jobs. Koula is
an alternative requiring its own reviewed environment and revised runtime plan,
not an assumed second GPU or automatic migration. Owned CPU nodes may perform
verification/analysis within separately checked capacity. CPU analysis wall time,
replication bandwidth and actual power cost are not measured by the GPU timing;
record them operationally. No paid cloud resources are within this proposal.

## Freeze and release checklist

Before confirmation can proceed, record all of the following in a committed,
timestamped packet and retain the independent review/owner decision:

1. Admitted exploratory evidence and generated summaries used only for planning;
   qualification-transfer decision, complete endpoint inventory, exact power
   output, chosen claim scope and final sample size.
2. Exact confirmation matrix and statistical protocol, implemented confirmation
   admission/analyzer, regression checks, clean source revision, source archive,
   dependency/container/environment digests and exact command.
3. The generated planning output audits 780 recorded inputs, including all 768
   legacy manifests, retained R1 execution summaries (including interrupted
   exp001 and failed engineering), inventories and calibration analysis: **no
   reserved-seed overlap**. Only seed fields/run identifiers are used from legacy
   files, never outcomes. Repeat against live attempt logs before freeze to cover
   attempts absent from these retained records. This audit does not prove absence
   of an unrecorded execution; reservation in YAML alone is insufficient.
4. Fresh initialization for every confirmation run and fresh fitted probes in
   each target checkpoint's coordinates. Bind panel manifests before fitting;
   keep fit/tune/evaluation windows disjoint. If reusing the corpus regions and
   evaluation windows, explicitly call the design held-out **training seeds**,
   not held-out documents or a new-corpus validation. Freeze any new panel
   selection seed and exact windows in the packet before execution.
5. Verified free storage, immutable archive destinations, exact inventory byte
   budget, final wall/retry caps, node reservation and reviewed budget record.
6. A stage-specific owner authorization and single-use scientific admission;
   verify retrieval/replay failure on missing or modified artifacts. This draft
   and the instruction to prepare it do not constitute that admission.

On completion, report every attempted seed and failure, verify the entire bound
artifact chain, generate summaries with analysis/source hashes, add claim-ledger
links to committed manifests and summaries, and update the manuscript only for
admitted claims. Never recast missing, failed or unqualified outcomes as privacy
passes, silently drop them, or rewrite exploration as confirmation.
