# R1 prospective protocol — FROZEN 2026-09-16

Protocol version: `r1-exploration-1.0`.
Matrix: `experiments/configs/r1_exploration.yaml`.
Status: **frozen 2026-09-16 for scientific execution under bound single-use
scientific admissions** (`r1-scientific-admission-1.0`). It is not a claim of
preregistration, historical confirmation or a completed experiment. The
engineering stages are complete and remain non-scientific. Calibration and
exploratory execution each require an admitted, timestamped packet before their
affected outcomes are inspected; confirmation requires a separate protocol.
Attack thresholds remain draft until the calibration-qualified frozen attack
specification; any amendment is recorded before exploratory scores are inspected.

## Evidence admissibility

Legacy observations are excluded from R1 scientific evidence. Retain them under
the pre-reset reference revision as historical, inadmissible material. Do not
overwrite, relabel, pool, warm-start from or use them to qualify R1 attacks.
No old source snapshot has been recovered by this reset. Reviewed current source
and raw corpus bytes can be new declared inputs; save their complete bytes and
digests before execution. Their earlier use does not validate new outcomes.

Engineering uses `scientific_evidence: false`. A later successful test cannot
promote an engineering run into a scientific cohort. Admit new scientific runs
only after the source/environment/data/attack/compute requirements below pass.

## Implemented engineering substrate

The non-scientific local validation stage now uses separate trusted-client and
remote-surrogate processes over authenticated loopback TCP. The remote process
runs from an allow-listed packet containing no corpus, panel, checkpoint, output,
or trusted model state. Versioned frames carry actual encoded tensors and record
payload plus framing bytes. An uncertain optimizer acknowledgement fail-stops the
client; durable remote prepare/commit records and coordinated checkpoints permit
recovery only from a known common completed step.

The engineering path implements deterministic four-region manifests, sparse
capture of pre-transport/transmitted forward values, returned values and both
full-precision gradient directions, target-checkpoint attack
fitting for the proposed 3 targets x 4 channels, and semantic validation of panels,
captures, wire records, checkpoints, attack banks and manifests. These facilities
have engineering status only. Default thresholds remain draft values; CPU execution
cannot qualify a scientific instrument or admit a result.

## Questions, estimands and competing outcomes

For matched seed s and cell c, let L0, LT, LF and LB be losses for all-local,
trained-remote, frozen-remote and removed-remote models. Define:

- replacement penalty ΔU = LT − L0;
- remote-learning gain GF = LF − LT;
- gain relative to removal GB = LB − LT.

Positive GF identifies a benefit from remote updates in this controlled system.
It does not establish benefit relative to removal; GB addresses that separately.
Neither establishes necessity of remote hardware. The scientific estimands are
continuous; gate decisions are engineering classifications, not natural laws.

The open hypothesis is that no member of the finite tested mechanism family
simultaneously meets the utility, qualified privacy and causal remote-learning
requirements. A valid passing member is evidence against that hypothesis at its
operating point. Failures support only the tested set's classification. No
universal trilemma, information-theoretic bound or feature-necessity conclusion is
admissible from the proposed experiment matrix.

## Cohorts and seeds

| Cohort | Seeds | Use |
|---|---|---|
| Micro engineering | 61001, 61002 | Eight-step technical checks only |
| Full-size engineering speed check | 61103 | Runtime/resource estimate only |
| Attack calibration | 62011, 62023, 62039, 62047 | Select and qualify attacks; no pooled scientific estimate |
| Exploration | 63013, 63029, 63043, 63059 | Complete 19-cell controlled exploration |
| Reserved confirmation | 64007, 64019, 64037, 64051, 64067, 64079, 64091, 64109 | Untouched; separate protocol and power assessment required |

On 2026-09-10, the proposed seeds were checked against the seed fields in all
768 legacy manifests; none overlapped. That read-only check excludes prior seeds
without consulting their outcomes. The machine-readable matrix retains the
observed exclusion list. Validate disjointness again before packet freeze,
including all fresh run attempts. Seed reservation alone is not preregistration
or a claim that eight runs are adequately powered.

## Data and attack panels

Initial raw corpus: `data/raw/tinyshakespeare.txt`, SHA-256
`86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`;
byte-level identity vocabulary 256. Archive the corpus and tokenizer definition,
not merely a download URL. This is a fresh experiment on a reused public dataset,
not a claim of a new corpus or production-model replication.

Engineering may use the simple first-90%/last-10% loader because its outputs are
inadmissible scientifically. The scientific design requires a new explicit panel
loader, **not yet admitted**, with four byte-offset regions:

1. first 80% for model training only;
2. next 5% for attacker-known auxiliary text and attack fitting;
3. next 5% for attacker tuning, positive controls and matched null calibration;
4. last 10% for held-out utility and attack evaluation.

Publish exact integer offsets, sequences and target definitions. Every sequence
and its next-byte label must be fully inside its assigned region. Drop a full
sequence length at each region boundary; use nonoverlapping windows within an
evaluation panel. Separate windows used for attack fitting and testing, and bind
the fixed panel inventory before affected fits. Independence is at the training
seed level; windows in one play and repeated text across seeds are not new
documents. Fresh-document or external-corpus claims require an additional design.

Learned coordinates belong to a particular model checkpoint. Fit each declared
known-plaintext attack in that target model's coordinates using only its
permitted auxiliary panel. Calibration-seed results can set algorithmic choices;
they do not justify direct coordinate transfer of a fitted classifier. Validate
the precise attacker knowledge and fitting procedure in each representation and
defense. Do not read held-out attack scores while selecting probes.

## Controlled runs and instrumentation

Run every matrix cell in trained, frozen-random and removed-remote modes for all
four exploratory seeds, plus one matched all-local model per seed. The 232 runs
share budgets and data but are separate attempts with separate outputs. The
calibration proposal has 19 trained cells × four seeds = 76 runs, each with the
same 2000-step endpoint budget. There is no outcome-dependent early stopping.

Use component-specific initialization streams. Common trusted layers must
match across modes by original layer identity; encoder/decoder initialization
must not drift when remote depth or removal changes RNG consumption. Verify
initial tensor digests, minibatch identity, mode-specific trainability and
dedicated defense randomness. Capture-side evaluation must not advance training
RNG state, update model parameters or alter the training schedule. These are
executable engineering requirements, not assumed facts about historical code.

Preserve actual forward latent values, returned values, both directions of
boundary gradients and their dtype/processing metadata. A low-precision probe
copy is a separate derived artifact from the full-precision value visible to the
remote side. Record step and checkpoint identity. Store a full final model and
optimizer checkpoint with RNG states for regeneration and explicitly inventory
which remote parameters, optimizer state and history observations were captured.
A checksum is an integrity observation, not an attack on the state it describes.

Timing, actual network transport, full evolving remote-state attacks, optimizer
state beyond the remote module, active/adaptive attackers, joint-tensor attacks,
cross-run joint attacks, history-accumulation attacks, canary/natural-sequence
membership and unrestricted semantics are outside the first proposed qualified
suite. They remain explicitly unevaluated; storing their ingredients does not
qualify an attack. If returned values or full-precision gradients have no
qualified implemented probe, name those exclusions too and leave their privacy
status undetermined. No complete-transcript privacy claim is permitted.

## Attack qualification and provisional gates

Proposed required probes cover punctuation fraction, next-byte character class
and known-plaintext input-byte classification separately on forward values,
returned values, full-precision outbound gradients and full-precision inbound
gradients. This is an intended 12 target/channel combinations per defense and
capture setting, **not a claim that the suite is implemented or qualified**.
Vowel/newline targets, temporal attacks and different attacker knowledge would
require separately named additions. Attack fits, positive injections, null
permutations, captures, side information and classifier hyperparameters must all
be freshly recorded. Nulls must preserve the sampling/sequence dependence
relevant to the target; token shuffling alone is not a general matched null.

Draft minimum positive-control advantages are .20 for correlation targets and
.10 for accuracy targets; draft null absolute-advantage band is .02. Draft
privacy limits are .10 correlation advantage and .02 accuracy advantage. These
are provisional engineering targets for calibration, not thresholds validated
by legacy observations. Calibration can reveal they are unsuitable. Any change
must be recorded and frozen before affected exploratory scores are inspected.
Qualification is per target, channel, defense, side-information setting and
frozen score rule. One qualified target cannot qualify another target.

Compute per-seed advantages against the declared matched null. Report all seeds,
all targets and null/positive-control diagnostics, with a seed-level uncertainty
summary. The proposed classification uses the maximum across retained seeds
within each qualified combination, followed by comparison to that combination's
limit. Do not maximize raw correlation differences and accuracy differences
together; do not interpret either as a leakage probability.

The draft utility requirement is mean ΔU ≤ .10 nats and upper 95% seed-level
confidence limit ≤ .12. The draft causal requirement is mean GF ≥ .01 nats with
lower 95% confidence limit > 0, alongside GB ≥ 0 with lower 95% limit ≥ 0 when
claiming benefit over removal. These margins are explicit provisional utility
choices, not estimated success rates. The four-seed exploratory intervals are
descriptive; simultaneous confirmatory inference requires a later power and
multiplicity plan. Sensitivity tables should show decisions near each margin.

The utility margin expresses a chosen tolerance on average prediction loss;
the gain margin asks for a nonzero practically useful improvement rather than
a nominal remote layer count. No accepted margin follows from quarantined
data. Report planned sensitivity to utility margins .05/.10/.20 nats and gain
margins .005/.01/.02 nats without selecting whichever table supports a preferred
story. Attack thresholds require fresh calibration and an explicit target
definition; sensitivity cannot make an unqualified instrument qualified.

Apply three-valued logic:

- **Rejected:** any valid required gate is breached. A qualified privacy attack
  may reject despite other attacks being unavailable; a valid utility failure
  may reject even if privacy is undetermined.
- **Scoped candidate pass:** every required utility/causal gate and every
  required qualified privacy combination passes, all planned seeds are complete
  and evidence is admissible. This is a pass under the declared suite, not a
  certificate against omitted attacks.
- **Undetermined:** there is no established rejection, but a required result,
  qualification, paired run or evidence link is missing.

An empty finite joint feasible set requires a valid rejection reason for every
planned candidate. It does not require every candidate to fail privacy. A
mixture of utility and privacy rejections must be reported as that mixture.
Undetermined candidates prevent declaring the entire planned set empty.

## Statistical reporting and resources

The independent training seed is the inferential unit. Keep token rows and
windows nested. Report paired contrasts per seed, complete loss curves, effect
sizes and uncertainty; no outcome-based deletion, best-seed selection or pooling
of cohorts. Report all predeclared cells even if training is unstable. No
confirmatory p-value or H1/H5 confirmation is produced by this exploratory draft.

Record wall time, measured GPU identity/memory, execution precision, actual
serialized transcript bytes if implemented, and separately labelled analytical
FLOPs. A minimum remote FLOP-share constraint may be added as an independent
deployment requirement before freeze; it is not a substitute for GF or GB.
No deployed latency, energy or edge-device advantage follows from a local tensor
formula or one GPU timing run. Forward q is not complete-transcript bits/token.

## End-to-end evidence and admission checklist

Before any scientific stage, retain and validate:

- a clean committed execution revision plus source archive of all runtime files,
  the protocol, matrix, analysis scripts and complete source inventory;
- dependency lock, interpreter/framework versions, immutable container digest,
  GPU/driver/CUDA identity and relevant determinism settings;
- corpus bytes, tokenizer, model/configuration identity, panel offsets/windows,
  preprocessing code, seeds and independent random-stream specification;
- exact executable command, bound workload/cost budget, UTC start/end records,
  hostname or privacy-safe stable machine identifier and append-only attempts;
- resolved configuration, initialization comparisons, curves, raw captures,
  checkpoints, RNG states, qualification inputs/outputs and artifact hashes;
- downstream summaries recording source artifact hashes, analysis revision and
  parameters, followed by figure/table provenance and a current claim ledger;
- a fresh-checkout retrieval and replay check that rejects missing, changed or
  mismatched source/input/output artifacts before resume can skip a run;
- reviewed scientific panel and qualification implementations, and a stage
  admission record signed by the responsible authors before execution.

The freeze packet must exist before execution; retaining it afterwards is not
equivalent. Preserve immutable content-addressed artifacts with private reviewer
access. Never rewrite a failed or completed attempt. An operational retry uses
the same intended scientific identity, a new attempt identifier and an explicit
failure record. The permitted retry count is one for a documented device, OOM
or I/O failure. Configuration or code changes create a new packet and cohort
amendment. If a planned seed remains incomplete, show it and classify affected
joint claims as undetermined; do not silently drop it to manufacture a pass.

## Status of requirements at the 2026-09-16 freeze

The scientific stack is implemented and tested at source commit `0bcf568`
(352 tests + 12 subtests pass): the four-region scientific panel loader, the
12-combination attack suite with fresh matched nulls and positive-control dose
curves, sparse training-frame captures with RNG invariance, remote initial/final
snapshots, the scientific admission/executor and the three-valued analysis. The
engineering stages (local CPU validation, GPU micro and full-size performance
engineering) are complete on the owned GB10 nodes and remain non-scientific;
their acceptance records are retained under `evidence/r1-gpu-engineering-20260916/`.
Calibration and exploratory execution each still require their own bound
scientific admission before outcomes are inspected, and a successful engineering
run is not evidence that scientific requirements pass. Reserved confirmation
still needs a separate design, independent timestamp, sample-size justification
and frozen multiplicity/selection rules. No scientific experiment result is
supplied by this document.
