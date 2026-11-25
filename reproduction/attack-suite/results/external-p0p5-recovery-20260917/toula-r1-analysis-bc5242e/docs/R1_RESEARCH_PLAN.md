# R1: fresh experiments with complete evidence tracing

Status: **research design; no R1 scientific result exists.** The executable first
stage is engineering only. Remote GPU execution requires a prepared workload,
cost estimate and bound budget review. No cloud resources are provisioned by
this document, and no scientific run is authorized by creating this plan.

The systems implementation now has a non-scientific local CPU gate using separate
trusted and corpus-free remote processes, authenticated framed tensors, actual wire
accounting, coordinated checkpoints, deterministic panels, draft attacks and
semantic artifact validation. The same loopback path is wired into admitted GPU
engineering execution but has not run on CUDA. This implementation status does not
activate a scientific claim or freeze the draft protocol.

## Reset and scientific question

The legacy observations are inadmissible for new Paper 3 claims because their
exact execution source cannot be established. Keep the originals under their
reference revision for inspection; do not repair them into new scientific
evidence. Reuse of reviewed source code and the original raw public corpus is
permitted only as explicitly identified, newly archived inputs. Reuse of legacy
model checkpoints, captures, fitted attacks, calibrated thresholds, summaries,
statistics or figures is forbidden in the R1 evidence chain.

The question is whether a tested split system can simultaneously provide:

1. acceptable task utility relative to a matched all-local model;
2. useful causal remote learning relative to frozen and removed controls; and
3. leakage below qualified, explicitly scoped privacy gates.

The proposed trilemma is a hypothesis to challenge. A passing candidate would
count against it at that operating point. Failure by every fully evaluated
candidate would establish only an empty feasible set in the finite tested
family. Unqualified attacks, incomplete runs or inadequate evidence leave
questions undetermined. An empirical sweep cannot prove a universal trilemma or
that a particular leaked feature is necessary for learning.

## Architecture and controlled interventions

Use a newly initialized byte-level transformer: eight blocks, hidden width 256,
eight attention heads, feed-forward width 1024 and sequence length 128. A trusted
prefix feeds an encoder, a quantized latent crosses to a remote surrogate, and
a trusted decoder/suffix produces the prediction. The initial reference replaces
four blocks beginning at block 2, using latent width 256, forward precision 8
and one remote transformer block. These settings are design choices rather than
inherited evidence that this configuration succeeds.

For every exact cell and seed run trained, frozen-random and removed-remote
modes. Match encoder/decoder, trusted components, optimizer budget, minibatches,
precision and defense settings. Only the trained/frozen comparison isolates
remote parameter updates. The removed control is a dimension-matched identity
bypass with no remote trainable parameters. Its trusted initialization must be
identical despite a different number of remote parameters: assign independent
initialization streams by component and absolute original layer index. Compare
common trusted tensors by digest before training. Supply an all-local model for
each seed under the same backbone, data and optimizer budget.

The 19 cells in `experiments/configs/r1_exploration.yaml` separate interventions:

| Intervention | Levels or comparison | What it examines |
|---|---|---|
| Forward precision | 4, 8, 16, 32 bits/dimension | Precision response at fixed width/capacity |
| Latent width | 64, 128, 256 | Boundary capacity, with remote parameter count reported |
| Remote depth | 1, 2, 4 | Surrogate capacity at fixed replaced span |
| Centered replaced span | 2, 4, 6 | Replacement extent; both endpoints move around a common center |
| Position for span 4 | starts 0, 2, 4 | Location at fixed span |
| Backward clipping | absent versus norm 1 | Effect of clipping without added noise |
| Backward noise | 0, .01, .03, .1, .3 at clip norm 1 | Noise response without changing precision |
| Limited interaction | q4/q8 crossed with noise 0/.1 at clip norm 1 | Whether noise response depends on precision |

The matrix is a controlled exploratory design, not a full factorial or a complete
response-surface model. Shared reference cells are executed once. Width changes
also change remote parameter count, and centered span changes both endpoints;
neither contrast isolates an abstract information quantity. Longer training,
alternative encoders, target-removal interventions, cryptographic mechanisms,
external architectures/corpora and a full hardware deployment are later studies.

## Stages and workload

| Stage | Proposed runs | Intended output | Scientific status |
|---|---:|---|---|
| Local CPU systems validation | 14 × 2 training steps | Real two-process transport, panels, attacks, checkpoints and semantic evidence checks | Engineering only |
| Micro engineering | 14 × 8 training steps | Pairing, gradient/RNG, capture and provenance checks on CUDA | Engineering only |
| Full-size speed calibration | 4 × 100 steps | GPU runtime, memory, storage and capture overhead estimate | Engineering only |
| Attack calibration | 76 × 2000 steps | Fresh qualification and frozen attack specification | Blocked draft |
| Exploratory matrix | 232 × 2000 steps | Per-seed behavior and gate classifications | Blocked draft |
| Reserved confirmation | Not scheduled | Independently registered held-out test | No protocol or claims yet |

The scientific draft totals 308 training runs and 616,000 optimization steps.
These counts exclude attack fits, controls for their calibration, evaluation,
data transfers, output verification and repeated engineering measurements. A
representative GPU benchmark must precede a credible runtime/cost estimate.
Cell runtime varies with width, surrogate depth and span; a reference benchmark
is an estimate, not a measured bound on the complete matrix. Use measured peak
memory and bytes to size storage and concurrency. Training is staged only after
the next workload is reviewable; engineering throughput is never evidence of
utility or privacy.

The micro stage has a proposed five-minute watchdog and 256 MiB output stop
threshold; the full-size four-mode benchmark has a separate 30-minute watchdog
and 2 GiB threshold. These are not measured completion times or approved spending.
The output check runs between and after runs, so one run can overshoot. A Python
watchdog can also wait for a native CUDA call to return. An external provider
controller must enforce paid lifetime; strict storage bounds require a quota.
Capture training views at step 0, every 200 steps and at step 2000,
at most 11 frames per run, eight examples per frame. Four float32 tensor views
of shape 8 × 128 × 256 occupy about 4 MiB per frame: approximately 3.3 GiB for
76 calibration runs and 10 GiB for 232 exploratory runs at this width, before
checkpoints, attack-specific panels, metadata and other overhead. These are
illustrative size estimates; widths and all-local captures differ. Save remote
parameter states at initial/final checkpoints and a full optimizer checkpoint
at the end; do not silently enable per-step state dumps. Budget separate attack
panel captures and actual checkpoint sizes after measuring them. Unbounded
dense capture is outside this draft and requires a reviewed amendment.

## Expected evidence and analysis

Save every seed's all-local loss, trained loss, frozen loss, removed-control
loss, loss curve, initialization checks, capture metadata and resource records.
Compute replacement penalty, gain against freezing and gain against removal
separately. Report nominal layer share, analytic FLOPs, actual runtime and
serialized bytes as different quantities. A nominal share does not establish a
causal contribution or hardware advantage.

Plot seed-level responses and complete qualified gate outcomes before creating
headline prose. The four exploratory seeds support descriptive uncertainty and
candidate generation; they do not establish a powered confirmatory test.
Calibration cannot contribute samples to exploratory or confirmatory estimates.
Any subsequent statistic-selection, noise tuning or changed attack is recorded
as a new exploratory amendment. Reserve untouched seeds for a later frozen test.

Privacy evaluation must be target-, channel-, defense- and side-information
specific. Fresh positive controls and matched nulls must establish the
instruments' ability to detect the declared signal. Fit known-plaintext probes
on auxiliary captures from the target model itself, using disjoint evaluation
windows. A classifier trained in one independently learned coordinate system
cannot be assumed valid in another. The complete panel and attack implementation
is a scientific admission blocker, not something supplied by this design file.

## Admission before scientific execution

Follow `docs/PREREGISTRATION_R1.md`. The minimum acceptable chain is:

`reviewed code and inputs → archived execution packet → attempts and manifests
→ verified raw observations → generated analysis → table/figure → claim ledger`.

Archive every dependency and command needed to reproduce each arrow. Verify
restoration in a disposable clean checkout. Freeze the packet before the affected
scientific stage, record its timestamp externally or in a reviewer-accessible
immutable release, and preserve every failed attempt. Historical labels, new
file timestamps and later checksums do not substitute for a prospective record.

The eventual paper must follow the observations. If there is a feasible point,
report it. If a qualified attack defeats only some cells, report exactly those
cells. If utility alone rejects others, that is a different reason for joint
infeasibility. If evidence is incomplete, the result remains undetermined.
