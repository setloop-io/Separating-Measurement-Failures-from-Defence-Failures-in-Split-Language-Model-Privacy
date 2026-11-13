# Prospective BiSR fidelity specification

Status: **design specified; execution not admitted**. This additive specification
resolves the decisions left open by the 17 September audit. It does not change
`experiments/validation_protocol.v4.json`, whose SHA-256 remains
`cc23f15d5d7597ff2e73edd45e6f77b65aaf19ce3ed6f86d121352ce1ca5c4fc`.
The machine-readable companion is `experiments/bisr_fidelity_spec.v1.json`.

Historical B1 outcomes have already been inspected. These rules are prospective
for fresh runs, not a claim of outcome-blind preregistration and not a basis for
reclassifying those historical runs. No fidelity pass is issued by this document.

## 1. Claim and route decisions

| Route | Question | Permitted claim after all relevant checks | Dependency |
|---|---|---|---|
| R: published-setting replication | Can the original configuration be reproduced? | Configuration fidelity; performance replication only if its separate criterion passes | Original processed SensiReplaced and split/preprocessing identities, published-setting reference implementation |
| C: implementation conformance | Does the candidate implement the specified algorithm on identical inputs? | Implementation agreement within a declared tolerance on the validation set | Matched reference/candidate checkpoints, observations, grants, metrics and complete predictions |
| A: single-seam adaptation | How does the adapted algorithm compare on v3 captures? | BiSR-inspired single-seam comparison, never original-setting reproduction | C on the explicitly adapted reference, frozen adaptation manifest and complete v3 input coverage |

Route R remains blocked by original-data provenance. Route A is the designated
way to make progress without that dataset; it does **not** discharge either
paper's faithful-reproduction limitation. It requires a new execution amendment,
not an implicit reinterpretation of frozen v4. No route substitutes a favourable
accuracy for architectural or data fidelity. Route R and C can fail independently.

## 2. Source authority and algorithm

Primary sources: [BiSR v2, sections 3.1–3.2, 4.1 and Table 1](https://arxiv.org/html/2409.00960v2),
and [pinned SplitLLM source](https://github.com/StupidTrees/SplitLLM/tree/c376590227f9718077765824af4088665bb78687).
The paper specifies sequential backward then forward enhancement, a GRU with
256 hidden units and dropout 0.1, three-seed reporting, and SensiReplaced auxiliary
data. Its GPT2-large BRE settings are GM 32 iterations, learning rate 0.09,
beta 0.85, temperature 1.2; SM 800 iterations, learning rate 0.01, weight decay
0.01; AdamW betas (0.9, 0.999). These are source settings, not evidence that our
current implementation reproduces them.

The following operational decisions govern new validation:

- `sip`: unmodified argmax of SIP logits. `b`: SIP initialisation followed by GM.
  `f`: SIP argmax followed by SM. `bf`: SIP initialisation, GM, then SM initialised
  from GM tokens. The framework beam is a separate comparator; it must not supply
  the initial tokens or position-zero fallback for these four fidelity arms.
- GM uses fixed observed activations at the backward cut, the observed gradient,
  next-token alignment and the declared loss reduction. Record batch size,
  padding mask, shift, and every gradient-rescaling factor. Position zero remains
  the SIP prediction after GM. SM may subsequently change it.
- SM must traverse the granted embedding-to-forward-cut model, including client
  blocks and positional handling; retain autograd through that path. Decode by
  the specified nearest-embedding rule with deterministic tie-breaking.
- Save resolved AdamW epsilon, GM weight decay, all loss reductions, optimiser
  batch size, dropout placement and train/eval modes. Do not infer unspecified
  settings from a headline table. Bind them from the pinned reference source in
  the launch manifest; any disagreement requires an explicit configuration delta.
- R uses the published dual-cut target (boundaries after blocks 6 and 27, distinct
  from a count of 21 trunk blocks). A uses one cut at depth 3 or 6 and must label
  this changed observation grant. Existing single-seam captures cannot recreate
  the missing second boundary.
- Use the reference GRU architecture for the validated implementation and A's main
  SIP arm. Keep the existing MLP as a separately named adaptation/ablation; its
  results cannot satisfy GRU fidelity. NaMoE is excluded from the primary design.

## 3. Fresh validation inputs and timing

Use fresh victim/SIP seed pairs (31415, 27182, 16180), all mandatory. These are
new study choices, not asserted original-author seeds. Use the same reference
victim and SIP checkpoint bytes for each candidate/reference replay; do not
train independent victims and then subtract their aggregate scores.

For C, first reserve 256 documents from the pinned WikiText-2 test partition
by deterministic SHA-256 ordering of normalised text, salt
`bisr-conformance-v1`, excluding **all** v3 calibration/training/eval
partitions, the pilot corpus, previously used SIP auxiliary data and all
historical B1 attacked inputs. Then select fresh auxiliary articles from the
training partition, excluding these reserved validation documents. This ordered
selection avoids a circular auxiliary/validation exclusion rule. Pin document IDs, token IDs, masks and selection manifest before producing
candidate attack predictions. If original historical sample IDs cannot be
recovered, exclude their complete source partition; do not claim disjointness
from unavailable records. Absence of enough admissible documents blocks launch.
Use 128-token reference windows for R/C-dual-cut and 32-token windows for C-A;
these C window choices are validation design, not claims about original sampling.

R must reproduce the source's actual sampling rather than silently completing
missing batches. Audit and pin the exact attack event list using a control-flow
trace without attack predictions. Resolve the paper's 600-step/five-batch wording
against upstream indexing in that manifest. The historical 602-step/thirteen-batch
schedule is **not** accepted as an exact match. If this discrepancy cannot be
resolved, R remains a documented variant, not published-setting reproduction.
Do not extend training beyond the reference target merely to collect more attacks.

R must pin the original processed dataset, row order, split and auxiliary
preprocessing. Freshly sanitised or raw replacement data are acceptable only as
explicit variants. For A, select the first 112 admissible WikiText-2 training
articles under SHA-256 order salted `bisr-aux-v1`, after the exclusions above;
use all complete non-overlapping 32-token windows, no cross-article concatenation.
Missing exclusions block selection. Save the resulting IDs and hashes, not just
a requested article count. Train A's GRU for 20 complete epochs, batch size 32,
AdamW lr 0.001, betas (0.9, 0.999), epsilon 1e-8, weight decay 0, seeded shuffle,
no early stopping; these are explicit adaptation choices, superseding v4's
20,000-step MLP for route A only. C-A uses the same settings. R's auxiliary budget
must instead follow its fully resolved upstream configuration.

## 4. Mechanical conformance gate

Before aggregate recovery is inspected, both implementations must pass replay
and unit checks on identical saved inputs. Retain input/output tensors and code
hashes, not just a boolean assertion:

1. Check layer/cut identities, granted checkpoints, masks and no label access in
   the attacker process. Ground truth is supplied only to a separate scorer.
2. Verify exact stored-wire replay at capture precision and no victim parameter
   updates during replay. A failed replay blocks comparison.
3. On deterministic FP32 fixtures with TF32 disabled, compare SIP logits, GM/SM
   losses, gradients with respect to optimised variables, and one AdamW update:
   `abs(candidate-reference) <= 1e-6 + 1e-5*abs(reference)` elementwise; all values
   finite. This is a numerical implementation tolerance, not a privacy margin.
   Failure requires diagnosis, not a tolerance increase after seeing results.
4. Verify shift/position-zero semantics and `bf` ordering. Check `f` starts from
   SIP, independently of the beam arm. Perturbing scoring-only labels must not
   change any prediction. Validate both public and exact grants separately.
5. On fixed decoded strings, both metric implementations must agree to 1e-12.
   Record tokenizer revision, special-token removal, text normalisation, ROUGE
   implementation and METEOR/WordNet versions. Do not equate token-set F1 with
   position-wise token accuracy.

The FP32 one-step check does not demand bitwise identity after hundreds of
optimisation steps; that is evaluated by the separate performance gate below.

## 5. Operational statistical criterion

C has eight co-primary equivalence endpoints: reference-minus-candidate agreement
for each of SIP, b, f and bf, measured separately by positional token accuracy and
ROUGE-L F1. Store scores as fractions. Compute each seed's paired mean difference
(candidate minus reference) over the **same 256 documents**, with equal document
weight. For each endpoint form the two-sided 90% Student interval over the three
seed means: `mean +/- 2.919985580355516 * sample_sd / sqrt(3)` (df=2).

Every interval must lie **strictly inside [-0.01, +0.01]** to declare
`CONFORMANCE_ESTABLISHED`. This is an intersection-union equivalence decision:
all eight must pass, so no endpoint may be selected because it passes. Each
individual equivalence test uses alpha 0.05. A boundary crossing is
`INCONCLUSIVE`; an interval entirely outside the equivalence band is
`NON_EQUIVALENT`. Invalid or incomplete inputs give `NOT_EVALUABLE`, not a
statistical failure. Mechanical failures give `CONFORMANCE_FAILED` regardless
of recovery scores. Report all intervals and paired per-document differences.

The one-percentage-point tolerance is **our operational choice**: implementation
discrepancy should be less than half v4's two-point scientific margin. It is not
the authors' tolerance, is not estimated from historical B1 scores and does not
bound every future deployment. Three seeds can give wide intervals; an
inconclusive result must remain inconclusive. No outcome-driven extra seeds.
This test is conditional on the fixed validation documents, not a population
claim over text. Independent document-resampling analyses are sensitivity results.

For R, configuration fidelity is assessed separately from performance replication.
ROUGE-L F1 is the primary performance metric. The three fresh seed means give
a 90% Student interval by the same rule; compare it to the fixed published target
0.9979 with the same +/-0.01 operational band. Call a pass
`PUBLISHED_POINT_TARGET_COMPATIBLE`, **not** statistical equivalence to the
unknown population behind the original experiment. Original-run uncertainty
cannot be reconstructed from a reported mean and dispersion alone. Report
METEOR, token-set F1 and positional accuracy separately, without choosing the
metric closest to a published number. No R performance conclusion is allowed
before configuration provenance passes.

## 6. v4 comparison and compute claims

A is conditional on C-A passing, never on R being silently waived. Preserve the
v3 matrix: two depths, two knowledge conditions, ten victim seeds, two attack
seeds and the same 256 evaluation documents. SIP and bf produce 160 arm cells;
b/f diagnostics remain explicitly separate. Preserve paired seed/document IDs,
all frozen v3 control predictions and the crossed-bootstrap analysis design.

Public grants use pretrained mimic segments. In exact cells, grant the same
post-training client head as the v3 comparator; downstream/public segments must
be explicitly declared and held equal across compared arms. Checkpoint loaders
must enforce grants rather than merely changing a metadata label. Until that
loader and grant test exist, exact cells are not implemented and block completion.

The primary contrast remains bf minus joint_true_k32 at depth six/public.
A difference CI above +0.02 supports a practically larger recovery; below -0.02,
a practically smaller recovery. A 95% CI wholly inside [-0.02,+0.02] supports
operational equivalence at that margin; all other outcomes are inconclusive
about that three-way classification. Report zero-exclusion separately: statistical
significance is not the same as exceeding the practical margin.

`candidate_count=32` is **not** equal compute for beam search and continuous BRE.
The initial A comparison uses specified algorithm budgets, not a matched-compute
claim. Retain per-arm online forward/backward token positions, vocabulary work,
wall time and peak memory; report offline SIP training separately and state any
amortisation. Count mimic-top GM work separately from client-head SM work,
including higher-order autograd. A genuinely equal-compute comparison requires a
separate frozen resource-cap design; do not tune it on v3 recovery results.

## 7. Admission manifest and current disposition

A launch manifest must bind: route, this specification digest, v4 digest, source
commit and complete patch, model/tokenizer/checkpoint hashes, input/partition and
auxiliary hashes, resolved arguments, ordered attack events, environment lock,
metric implementation/resources, code for mechanical checks and scoring, and
fresh output locations. It must be committed before candidate outcome generation.
No null pins, placeholder paths or skipped disjointness checks are admissible.
Record host and measured canary resource use within the existing compute envelope;
this specification does not enlarge that envelope or start jobs.

All expected cells, seeds and documents are mandatory. Preserve immutable SIP/b/f/bf
per-token predictions, canonical IDs in scorer-only storage, masks, score records,
checkpoints, work counters and hash-chained journals. Missing cells block analysis;
no complete-case deletion, favourable checkpoint selection or recovery-driven retries.
Infrastructure repairs use new output directories, retain failed attempts and
record a source amendment when code changes. Qualification uses only prospective
records; historical audits stay historical.

**Resolved now:** claim labels, algorithm order and initialisation, decoder target,
data-separation rules, metric identity, equivalence decision, missing-data policy,
knowledge grants and the prohibition on an unsupported matched-compute claim.

**Still to implement or supply:** original data and resolved reference cadence for R;
GRU and cut/grant adapters, scorer-only prediction export, independent conformance
fixtures, mechanical-gate runner, performance scorer and complete hashed launch
manifests for C/A. No current historical run satisfies this specification. Neither
paper's evidence status changes until new results pass the relevant checks.
