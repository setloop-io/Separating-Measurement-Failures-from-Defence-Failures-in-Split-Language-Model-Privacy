# Paper progress and `paper-data` evidence review

Date: 22 August 2026

Artifacts reviewed:

- active manuscript and evidence ledger in `papers/delegation-frontier/`;
- archival 1,477-line manuscript in `papers/arxiv-draft/main.tex`;
- current Phase 1--6 outputs in `outputs/`;
- all 635 files present under `paper-data/`, including 422 JSON artifacts;
- the 60/40, isolation, gate-calibration, design-effect, and threat-scope reports.

Review standard: strong systems, security, or ML venue.

## Corrected verdict

The previous review was incomplete because it did not reconcile the historical
`paper-data` program with the current Phase 1--6 evidence. The archive materially
strengthens the experimental history and supplies results that should appear in the
paper as exploratory, negative, or measurement-audit evidence.

It does not make the manuscript submission-ready. The active manuscript is still a
75-line scaffold, and several attractive archival claims are contradicted or narrowed
by later audits. Current recommendation remains **reject / not submission-ready**.

It is useful to separate three scores:

| Object | Assessment |
|---|---|
| Current active manuscript | about 2/10: mostly protocol prose and placeholders |
| Evidence program, including `paper-data` | about 5/10: unusually rich diagnostics, but uneven replication and incomplete endpoints |
| Submission package as a whole | about 3/10: the reliable evidence has not yet been turned into a coherent, claim-disciplined paper |

## Major findings

### 1. The archive contains substantial evidence omitted from the active manuscript

The present checkout contains 635 `paper-data` files: 422 JSON, 139 logs, 68 JSONL,
three Markdown files, two scripts, and one text file. Important experimental groups
include:

- a dense 0.6B 60/40 ladder and intervention program through 100,000 steps;
- Qwen3-0.6B, Mistral-NeMo-12B, Qwen3.5-27B, and Qwen3.5-35B-A3B isolation/frontier cells;
- three-seed 0.6B and 35B protected-boundary cells;
- five repeated 39.3% 0.6B runs for rerun-variance estimation;
- gate sensitivity, design-effect, bundle-forensics, matching-attack, and gradient-channel audits;
- three small inference demonstrations with greedy generations.

The 1,477-line archival manuscript is useful source material for methods, system
description, negative-result history, and bibliography. It cannot be restored as the
active paper unchanged because its headline security, depth, and production claims do
not survive the later audits.

### 2. The archived cross-model frontier is real exploratory evidence, not a causal law

The isolation balance sweep reports the following loss/NLL deltas against the intact
model at a 500-step diagnostic budget, all on seed 42:

| Model | Reported remote share | Covered layers | Delta | Status under +0.35 gate |
|---|---:|---:|---:|---|
| Mistral-NeMo-12B | 13.5% | 5 | -0.0045 | clean |
| Mistral-NeMo-12B | 25.9% | 11 | +0.9137 | fail |
| Qwen3.5-35B-A3B | 18.5% | 11 | +0.2067 | clean |
| Qwen3.5-35B-A3B | 27.1% | 16 | +0.4463 | marginal fail |
| Qwen3.5-27B | 42.2% | 32 | +0.0714 | clean |
| Qwen3.5-27B | 47.4% | 36 | +0.3249 | marginal pass |

The 35B 18.5% point also has one 4,000-step run ending at -0.4054. These data
support a hypothesis-generating cross-model association and justify replication of
the 27B result. They do **not** support "depth is delegability": the models differ in
width, architecture, active-parameter structure, checkpoint training, and tokenizer;
the balance sweeps are one seed; the 27B point has only 500 steps and no observed
failure rung; and no checkpoint identity/hash is embedded in the isolation artifacts.

The 47.4% result is weaker than its 64-block diagnostic suggests. A later 256-block
evaluation records +0.3626, just outside the +0.35 gate, and its separately recorded
inference-evaluation delta is +0.5699. The manuscript may retain 42.2% as a clean
single-seed observation; 47.4% must be labelled marginal and unconfirmed. A selected
three-seed 35B series also exists, but seed 42 used noise 0.40 while seeds 43--44 used
0.35, so it is not an exact replication set.

### 3. The 60/40 archive gives a strong negative utility result

For Qwen3-0.6B, the archived split ladder shows a clean utility point at 14.3%
(delta +0.0655), followed by failures at 21.4% (+0.6015), 28.6% (+1.0521),
39.3% (+1.8585 at 2k steps), and 42.9% (+1.9707 at 2k steps). Extending the
last two cells to 10,000 steps improves them but does not rescue utility:

- 39.3%: +1.2750;
- 42.9%: +1.3713.

Five repeated 10k-step 39.3% cells give mean residual 1.2615 nats with SD
0.0128 and mean closure 86.296% with SD 0.139 percentage points. All five use
seed 42, so this is rerun stability rather than between-seed uncertainty. Width,
surrogate, sublayer, data-window, noise, and optimization interventions are also
preserved. This is useful evidence that the observed failure is stable under the
tested mechanism family, not just one unlucky execution.

This archive should support a negative-results section. It still does not establish
an impossibility or capacity ceiling: the mechanism family is fixed, cells are mostly
seed 42, and the historical corpus and evaluation protocol differ from the frozen
current confirmation grid.

### 4. The privacy archive invalidates broad security wording

The later threat-scope audit establishes that the original forward-frame attacker did
not see the full compromised training node view. The outbound output-gradient frame was
unclipped, unnoised, absent from bundles, and absent from the privacy accounting. Its
zero support identifies all real versus chaff rows in 4,608/4,608 captured frames.

On 65,536 independent rows from a configuration whose forward frame also leaks, the
project's own attacker recovered token information from the gradient alone at
`z = +14.98`; a shuffled-label control gave `z = -0.99`. On a forward-passing
configuration, the gradient reading breached the legacy +1.0 pp gate but remained
statistically unresolved because the corpus window supplied too few independent rows.

The gate-calibration audit is independently damaging: the maximum-over-arms statistic
is pinned by near-constant predictors. It can remain bit-identical to clean under a 25%
rotation-faithful injected leak and first fires between 40% and 50% coverage. Therefore
an archived "privacy pass" means only that the declared attack statistic stayed below
its threshold on released forward frames. It cannot support secure delegation, a fully
compromised-cloud claim, or a complete privacy coordinate.

These are nevertheless publishable negative methodological results if presented as an
instrument and threat-scope audit rather than as successful privacy protection.

### 5. The archive does not provide downstream or human quality validation

Three inference artifacts contain three greedy continuations per model for 0.6B, 27B,
and 35B. They are qualitative examples only. The recorded fields named
`perplexity_composed` and `perplexity_intact` are on the loss scale rather than
exponentiated perplexity, and the 27B sample visibly degrades despite its marginal
gate pass. There is no instruction-following, factuality, reasoning, code, standardized
downstream-task, blinded judge, or human-preference evaluation in the archive.

### 6. Provenance and packaging remain uneven

`paper-data/DATA_MANIFEST.md` describes a historical inventory of 1,766 collected
artifacts, but this checkout contains 635 files and omits many paths listed there.
`paper-data/TABLES.md` is empty because its expected `paper-data/results/` input is not
present. The manifest itself reports resolvable embedded provenance for only 41/296 of
one historical artifact group. The present isolation and latent-native exemplars do not
embed a producing commit or checkpoint hash; importing them in the repository's initial
commit does not recover that run-level provenance.

Only 403 of the 422 JSON files parse successfully. The other 19 are 60/40 precheck
artifacts containing literal carriage returns inside JSON strings. They do not invalidate
the corresponding run artifacts, but they prevent the archive from being described as a
clean machine-readable evidence package.

The corpus manifest does bind the historical corpus bytes by SHA-256, but it states that
v6--v13 used only the first 16,896 tokens of a 10.3 MB tokenized WikiText-2 file.
Consequently those results characterize one small window, not WikiText-2 generally.

## Current confirmed evidence

The strongest paper-level evidence remains the frozen, five-seed Phase 6 frontier for
one Qwen3-0.6B configuration on two corpora:

- WikiText-2: 14.3% clean, mean NLL delta -0.0465, 95% CI
  [-0.0495, -0.0435]; 21.4% failed, +0.5161, CI [0.5104, 0.5219].
- Penn Treebank: 28.6% clean, -0.0432, CI [-0.0514, -0.0351]; 39.3%
  failed, +0.4495, CI [0.4095, 0.4895].

This supports a corpus-dependent empirical transition. The archived evidence adds a
cross-model hypothesis and an extensive negative intervention history; it does not
turn the transition into a universal percentage or causal architecture result.

## Reviewer-expectation matrix

| Expectation | Corrected status |
|---|---|
| Claim discipline | Met in the scaffold, violated in much of the archival manuscript |
| Replicated current frontier | Met for Qwen3-0.6B on two language-model corpora |
| Cross-model evidence | Exploratory: 12B/27B/35B exist, but mostly one seed and 500 steps |
| Causal architecture/depth evidence | Not met |
| Matched remote-contribution controls | Partial and not located at all clean retained endpoints |
| Robust negative/intervention evidence | Substantially met for the historical 0.6B 39.3% cell |
| Statistical measurement audit | Stronger than previously credited, but it also exposes a defective privacy gate |
| Downstream and human quality | Not met |
| Actual systems economics and trusted-work savings | Not met |
| Exact-cell full-view privacy | Not met |
| Reproducible evidence package | Partial; current phases are better packaged than historical diagnostics |
| Complete manuscript | Not met |

## Manuscript direction

The paper should now be written around evidence tiers rather than waiting for every old
claim to be restored:

1. Headline: the current replicated, two-corpus Qwen3-0.6B empirical frontier.
2. Exploratory result: the archived 12B/35B/27B balance sweeps, explicitly framed as
   cross-model hypothesis generation.
3. Negative result: the archived 60/40 intervention program and measured rerun variance.
4. Measurement result: why forward-frame privacy passes do not imply security, including
   the gate-sensitivity and omitted-gradient findings.
5. Limitations: no causal depth result, no downstream/human quality, no demonstrated
   trusted-side savings, and no scope-complete privacy coordinate.

The 27B result is important enough to prioritize, but it must be replicated at 42.2% and
47.4% with a long horizon, multiple seeds, a failure rung, current corpus provenance, and
matched controls before becoming a headline. Until then, the defensible paper is an
empirical frontier plus negative measurement study, not a capacity, depth, production,
or secure-delegation paper.

## Operational note

The latest exact forward-wire follow-up is not running. Toula completed the run and then
failed while assembling its capture because tensors with 32 and 80 rows were concatenated.
Koula's port-5129 server remained alive but idle. No output from that failed assembly should
enter the evidence ledger.
