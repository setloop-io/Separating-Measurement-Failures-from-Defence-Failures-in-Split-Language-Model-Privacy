# Paper closure schedule — 20 September 2026

The user authorised all six outstanding workstreams and selected **arXiv
preprints**, with a possible later IEEE or USENIX submission. This schedule
prepares upload packages; it does not submit to arXiv or select a final venue.
The machine-readable plan is
[`experiments/closure_queue_20260920.json`](../experiments/closure_queue_20260920.json).

On 21 September, steps 1, 2 and 4 are complete; step 5 retains reviewed
external exclusions. Qwen's complete 200-cell grid and primary analysis passed.
CPU reproduction and evidence archiving are running before HF publication and
final manuscript integration. The table below preserves the original scheduling
state; dated continuation records give subsequent progress.

## Work and dependencies

| Step | Work | Allocation | State at scheduling | Next admission condition |
| --- | --- | --- | --- | --- |
| 1 | Paper A corrected P6 raw release | koula, CPU and network | All 200 source cells verified; 33 GB archive uploading | Full immutable HF hash verification before updating release links |
| 2 | Sasa 500-step extension | sasa audit; CPU analysis | Integrity verified: 13 jobs, 13 evidence journals, 12,443 raw files | Capture/checkpoint/grant review before descriptive analysis and publication |
| 3 | Qwen P6 correction | soula diagnosis; koula GPU | Non-finite training losses confirmed; cause investigation queued | Finite training/control canaries, frozen amendment and resource projection before a fresh 200-cell grid |
| 4 | Five unmeasured agentic cells | voula diagnosis; toula GPU | Audit complete; repair queued | Valid action-production/parsing canaries before the complete matched ten-cell follow-up |
| 5 | Original BiSR setting and historical provenance | Bounded inventory review on all five nodes | Prerequisite review queued; known dependencies absent | Exact original assets and execution provenance; otherwise `BLOCKED_EXTERNAL` with explicit exclusions |
| 6 | Integration and preprint preparation | Local editing/build/review | Publication branch merged; two provisional source archives compiled | Qualified evidence updates, claim review, final rendered PDFs and author metadata checklist |

Step 5 does not prevent a scoped preprint that explicitly excludes the missing
claims. It cannot be represented as a runnable experiment until its inputs and
provenance are recovered. The existing single-seam BiSR adaptation remains
distinct from original dual-cut/SensiReplaced fidelity.

Qwen retains the original **24-hour total envelope**, including 1,117.259 seconds
already spent and all new canaries/retries; the canary ceiling is 1,800 seconds.
Agentic follow-up is capped at **2.5 GPU-hours**, including canaries with a
900-second ceiling. One GPU job per host may run at a time. The separate v4
50-hour authorisation is not reused. All new model and scientific-analysis work
uses committed ORX child experiments. Failed/invalid originals stay preserved;
errors are never converted into successful zero-violation measurements.

## Dispatched work

- Paper A publisher: koula PID `2877565`, root
  `/home/geo/paper-a-p6-release-20260920`. Check `status.json`, `publisher.log`
  and the final `status-verification.json`. Source raw directory:
  `/home/geo/llm-attacker-p6-v2-d5ec03e/results/p6-corrected-gpt2-v2`.
  Destination: `Setloop/llm-attacker-paper-a`, `revision-20260917/p6-raw`.
  Do not start a second publisher while this one is alive.
- CPU audit experiment: `29874f46-1739-4e76-86e4-0e1dbd12d974`, source
  `eb98eea90fff34d272520b4fc7b965f000d82b8b`, isolated worktree
  `/Users/geo/LLM-Attacker-closure-20260920`.
- Voula audit `170c8b36-339f-4906-9ebd-31d2dbdaa6e2` passed and identified
  all five unmeasured cells. The amended full ten-cell follow-up has now
  completed and been published; the original errors remain historical.
- Sasa audit `148c2140-c947-4b38-9595-95dbac00cd6d` passed: all 13 jobs and
  journals verified, with 12,443 raw files hashed. Integrity verification does
  not replace the pending scientific grant and checkpoint review.
- Soula audit `3677e766-dfda-415f-9fa9-f03eb5c0e8c3` completed and located
  non-finite mean/final training losses in both result and summary records.
  Subsequent backward probes found FP32 gradient overflow. The FP64 amendment
  and its admission checks are described below.
- Earlier audit dispatches `d084dde7-a2b0-45ff-81e7-b9459f8b5797` and
  `46594a3a-c59c-4407-a033-08f2906ddd80` failed because `.gitattributes`
  excluded a directory named `results` from the source archive. Corrected source
  uses `qwen/raw`; all 11 pinned input hashes were checked in the actual archive.
  These failed setup runs are retained and have no scientific result.
- Publication safeguard integrated into main at `24e68be`; frozen experiment
  refs remain unchanged. The diagnostic entrypoint remains on its own branch.

## Automatic continuation

LaunchAgent `com.geo.paper-closure-20260920` is installed on the user's Mac.
The original `exec resume` transport failed because this conversation retained
an active writer: 66 attempts did not start any work. That transport has been
removed. The scheduler now uses **`codex queue`**, which delivers to the existing
owner without opening a second writer. A real transport probe was accepted as
message `01a0bfeb-6d5b-7bf2-b817-74cae9f3551b` while this conversation was active.
The probe was consumed in this same conversation at
`2026-09-20T17:56:46.273505+00:00`, recorded in
`build/closure-queue-20260920/delivery-ack-20260920.txt`. This verifies delivery
and execution of the acknowledgement; it does not imply scientific completion.

Each queued continuation must acknowledge its ticket when it starts and record
completion before ending. The next continuation is due 30 minutes after that
completion. A file lock and one pending ticket prevent duplicate deliveries.
Missing acknowledgement after 30 minutes, missing completion after two hours,
or three rejected deliveries pause scheduling and record a blocker. An ambiguous
delivery timeout also pauses rather than risking a duplicate. The current next
check time is in the local configuration and runtime queue state below.

Configuration, prompt and logs are under
`/Users/geo/.codex/scheduled-checks/paper-closure-20260920/`.
The mutable task state is
`build/closure-queue-20260920/queue-state.json`; set `active` to `false` to pause
continuation. The service can be inspected with:

```sh
launchctl print gui/501/com.geo.paper-closure-20260920
cat /Users/geo/.codex/scheduled-checks/paper-closure-20260920/scheduler-status.json
```

`scheduler-status.json` distinguishes queued, running, completed and paused
continuations. `receipt-<ticket>.json` records execution acknowledgements.
The old failed-resume status is retained separately. The trigger depends on this
Mac being awake and the user session being available; already dispatched remote
jobs continue independently. The continuation stops once executable work is
complete and only external dependencies or author decisions remain.

Eight isolated scheduler tests passed, covering scheduling, queue delivery,
receipt/completion timing, missing receipts, rejection and uncertain-delivery
pauses, and overlap prevention. The installed service's initial invocation
returned `WAITING_FOR_SCHEDULE`. The real queue probe was accepted and its
execution acknowledgement was subsequently verified.

## Repair execution started after the scheduling failure

Qwen and agentic investigations were dispatched directly through ORX experiment
`29da2edc-17d5-44a6-880d-a7aedff96b76`, frozen source `e39b2e0`, from
`/Users/geo/LLM-Attacker-repairs-20260920`.

- Koula: instrument the original D1 failure and D0 control, preserving pinned
  inputs, 128 training steps, parameters and arithmetic; check the earliest
  non-finite forward/gradient/optimizer stage. Ceiling: 1,800 seconds.
- Toula: capture original planner replies for two failing configurations, then
  test non-thinking decoding and multiline ACTION parsing on the five excluded
  cells. Original semantics and failure categories remain enforced. Ceiling:
  900 seconds. These are excluded one-trial canaries, not the matched follow-up.

These initial diagnostic runs did not automatically admit a full grid.

Koula run `a7e7c2d0-987c-4b70-8dda-5d24ac6b8a18` completed in 60.494 seconds
of worker time. D1 reproduced a non-finite parameter gradient at training step
99 in `model.layers.3.mlp.down_proj.weight`; D0 completed all 128 steps with
finite training and before/after utility. Outputs live under each remote ORX
run's `repo/` directory.

### Qwen precision amendment and admission

Further probes retained all failures. Run
`5dc3d673-a5ff-4db5-8a2f-597aadb0a795` observed backward magnitudes around
`4.1e28`, `1.6e32` and `3.5e35` entering successive tail layers, then non-finite
gradients entering layer 17. This identifies FP32 range overflow in the backward
pass. Input activations and weights remained finite at the captured operations.

A prospective FP64 amendment preserves inputs, seeds, quantisation, noise,
clipping, learning rate and training steps. Model arithmetic and the otherwise
hard-coded FP32 RMSNorm and attention-softmax accumulators use FP64; the existing
cross-entropy conversion and FP16 capture format remain explicit. This is a
different precision condition, not reproduction of the failed FP32 run.

Both 128-step D1 and D0 training/utility canaries passed with finite values in
run `7c357032-83ec-47a1-a88c-20cf6a2719d6`, source `9c0cd45`, consuming
207.247 seconds. The preceding four diagnostics consumed 191.514 seconds.
Including the original failed run, 1,516.020 seconds precede the full-cell
admission checks.

Experiment `70fa3a32-1dd8-4772-bbfb-c4ff236ae5f2`, frozen source `e2d4268`,
is running on koula as `766a85bc-0bfe-40d2-aa13-51df27f5e926`. Its isolated
worktree is `/Users/geo/LLM-Attacker-qwen-fp64-grid-20260920`. The full formerly
failing cell passed in 158.485 seconds and the all-local full control passed in
253.206 seconds. The resource gate passed and the fresh 200-cell grid started.
Its 48,682-second projection (13.52 hours, including a 50% margin) fits the
84,472 seconds remaining before the 1,800-second analysis reserve. This is an
admission estimate from two canaries; completion time and later-cell numerical
validity remain subject to the actual run.
Read `repo/qwen-fp64-v2/status.json` and `resource-gate.json` under the remote
run. The controller admits a **fresh complete 200-cell grid** only after both
full-cell checks pass and `1.5 × (192 × trained time + 8 × control time)` fits
the remaining original 24-hour envelope with 1,800 seconds reserved for analysis.
All new canaries share the original 1,800-second canary allowance. Live and
persisted captures must be finite. Canaries and the old partial grid are excluded
from the new grid. A failure or exhausted budget stops the controller without
partial-grid inference. The complete grid still requires review, reproduction
and immutable HF verification before paper claims change.

### Agentic measurement blocker closed

The excluded diagnostic run `905b561b-e4df-474a-8349-bb2595e836d0` found
thinking-mode replies exhausting the 768-token limit before a complete action.
All five affected cells passed non-thinking canaries. The candidate replies
also parsed identically with the original parser, which was retained.

Fresh full run `fe79474d-361e-4585-a6c5-b88de0d50333` on toula, experiment
`4f10d085-7b9a-4227-8187-4223ba45bc1f`, source `edf0df5`, measured all ten
cells and 20 fixed scenario trials. There were zero policy-violating trials;
each of the three sanctioned-peer-request cells recorded two false blocks.
This closes missing measurements for the amended decoding condition. It does
not validate the original failures or establish a defensive contrast or a
population security rate. Canaries and the full run used 234.786 seconds of
worker wall time. No new parser was promoted into the production code.

- [Verified evidence](https://huggingface.co/datasets/Setloop/llm-attacker-paper-b/tree/0a07b9d35ad517269b26e1d187c45017feeaedd1/agentic-amendment-20260920/evidence):
  16 files, 1,439,578 bytes; manifest SHA-256
  `a6b85acd43e08f3fa4f8b6302faa081dd01ce934f0d4d60b94f11e329c1eff57`.
- [Verified manuscript package](https://huggingface.co/datasets/Setloop/llm-attacker-paper-b/tree/30fda02272bcd9b2c3110855978528e40cc76e97/agentic-amendment-20260920/manuscript):
  9 files, 653,981 bytes; manifest SHA-256
  `ca8661d4da14601bf1d5ee61e5bbfb8b9f91749ca221570dde20eb4d414c0161`.

Both tiers passed immutable remote hash verification. Source, protocol, raw
replies, timing, environment, evidence journals and reproduction instructions
are included. Paper B now reports the amendment separately from the historical
failures. Its refreshed 23-page PDF passed compilation and visual review,
including the changed live-agentic section.

## Provisional arXiv packages

The current author-facing checklist is
[preprint-readiness-2026-09-20.md](preprint-readiness-2026-09-20.md). It separates
pending evidence work from authorship, category, licence and overlap decisions.

`build/arxiv-preprints-20260920/INDEX.json` indexes the two packages. Each paper
directory contains its source ZIP, compiled PDF, hashes and `METADATA.json`.
Both packages passed two PDFLaTeX passes, with no undefined references/citations
or overfull boxes, and archive payload hashes were independently checked.
Only necessary source, bibliography and figures enter the source ZIP; compiled
PDFs and build logs remain outside it, following
[arXiv's TeX submission guidance](https://info.arxiv.org/help/submit_tex.html).

These are provisional. Following the evidence updates, build into a fresh
directory with:

```sh
python3 experiments/prepare_arxiv_preprints.py --output build/arxiv-preprints-FINAL-DATE
```

The agentic amendment was incorporated into a fresh pair of provisional packages
at `build/arxiv-preprints-20260920-agentic/INDEX.json`. Both compiled without
undefined references/citations or overfull boxes. Paper B's source archive and
PDF are also included in the verified manuscript supplement above.

Final tasks are evidence-link/claim review, visual review of the rendered PDFs,
author and affiliation confirmation, category/cross-list and licence selection,
and an overlap/disclosure checklist. The draft primary category is `cs.CR`;
`cs.LG` is only a possible cross-list. No final licence has been chosen and no
submission has been made.

## First acknowledged continuation: Sasa and prerequisite review

Follow-up ticket `2148bfef-c4eb-4e32-aa24-02ab491caab2` was acknowledged by the
existing conversation. No second scheduler or agent was created.

The Sasa archive was collected intact: all 12,443 files match the earlier raw
inventory. Original source commit `1ee12e111f7bc6306bd0a4249b129b7c2611522f`
independently reconstructs the source digest recorded in all 13 evidence
journals. The capture/checkpoint review qualifies the declared **public surrogate
head** measurement. `mid-lite` is a fixed seeded projection, not a co-trained
encoder; these runs do not claim an exact capture-time head grant.

The initial descriptive audit ran on Sasa at source `3d8858f` as
`4c192da0-0e74-4985-9473-52a85f72437c`. It passed all six trajectories. A
generic statement about cycling text was corrected on a child after the audit
found 500 unique rows per trajectory. Numerical analysis code and all cell
metrics were unchanged. Frozen source `bd4aabf` then produced byte-identical
`analysis.json` and `trajectory.csv` on two CPU hosts:

| Host | ORX run | Worker wall seconds |
| --- | --- | ---: |
| sasa | `39ae413f-8203-4ef7-b81a-83c07952d6b1` | 39.378 |
| soula | `608d1a39-dc29-4353-b14a-f61b49f22eac` | 39.126 |

Experiment: `7e61e901-54cd-4532-a76a-02ed13d6eefa`. Analysis SHA-256:
`f5aed623868978f2a427cb712f2c4ed38b979843971b090b2dda6a428c29b770`;
trajectory SHA-256:
`0d0dfcf2821883d142b26d5b36bc4f43a761350adedbd0f1ea2825810399b034`.
The complete evidence package passed immutable remote hash verification:
[Paper B Sasa supplement](https://huggingface.co/datasets/Setloop/llm-attacker-paper-b/tree/f1488fbde74726635f1e8616092ca187ac051270/sasa-500step-20260920/evidence),
18 files and 223,541,131 bytes. Its manifest SHA-256 is
`34ad17cc76af13e6e527070f6b72d90356e0c2ea44982f25bd626ed34b636083`.
Paper B now links this pinned evidence and reports the descriptive scope.

Training and calibration share a corpus; 1, 4 and 4 of each seed's 500 inputs
exactly overlap the 64 public calibration windows. The changing victim differs
from the public attacker head and embedding, and the decoder is not adaptive to
the projection. The measurements are descriptive training trajectories, not
held-out convergence or an adaptive privacy guarantee. Legacy gate/seed fields
are retained as raw history without promoting their verdicts.

The bounded prerequisite refresh inspected `/home/geo` to depth six on all five
nodes, pruning caches, environments, Git metadata and ORX runs. It found only the
known raw-article Sensi substitute and two release copies, all with SHA-256
`5da54ac04f01937061a7aa5f5a55aec3d4f28b71375d00f6f56815667637183d`.
The owning external project's HEAD remains
`525de9c3037abd6235af83a1add1480453107ae0`; its current ledger admits no
empirical scientific claims and its policy still excludes pre-reset results.
The disposition is **BLOCKED_EXTERNAL, exclusions retained**, recorded in
[closure-prerequisites-2026-09-20.json](closure-prerequisites-2026-09-20.json).
This bounded check makes no assertion about other accounts, unmounted media or
arbitrarily named assets. It does not block the scoped preprints. Revisit only
when a new asset, location or evidence-policy change is provided.

The Sasa manuscript supplement also passed immutable verification at
[revision 6a853d04eb50d7dd639cb0bf08354cc4a032278d](https://huggingface.co/datasets/Setloop/llm-attacker-paper-b/tree/6a853d04eb50d7dd639cb0bf08354cc4a032278d/sasa-500step-20260920/manuscript): 11 files, 664,880 bytes. Both provisional arXiv packages were
rebuilt at `build/arxiv-preprints-20260920-sasa/INDEX.json`; their archive payload
and PDF hashes match the metadata. The updated 24-page Paper B was rendered and
visually reviewed, including the revised discussion on pages 12 and 17. Sasa
closure is complete for the declared descriptive scope. Qwen remains running
(five of eight controls verified at this check, followed by 192 trained cells);
Paper A still has two raw files remaining at its last upload commit.

## Second acknowledged continuation: Paper A release verified

Ticket `c8735dfa-1acf-467e-bdad-2814400128f2` was acknowledged in the existing
conversation. The original publisher completed at 19:00:18 UTC. Its complete
raw archive passed pinned verification at
[`0bec5c37b960fdb5be5b2bd8fab1cdd05a4bbe4c`](https://huggingface.co/datasets/Setloop/llm-attacker-paper-a/tree/0bec5c37b960fdb5be5b2bd8fab1cdd05a4bbe4c/revision-20260917/p6-raw):
11 files, 34,703,146,309 bytes. All 200 cells had passed the publisher's source
and journal checks before archive creation. The independent pinned-tree check
matched the exact inventory, downloaded small-file hashes and large-object
LFS digests; it did not repeat the entire archive download and extraction.

Paper A now links that release and accurately distinguishes completed
single-seam BiSR conformance/adaptation from the still-excluded original
published setting. Public HF access was confirmed without changing visibility.
The 12-page PDF passed compilation and visual review. The additive manuscript
supplement contains the current PDF, six LaTeX/bibliography/figure source files,
provisional arXiv ZIP, metadata, reproduction index and raw-verification reports.
All 14 files (573,793 bytes) passed immutable verification at
[`2540f5904e5aaeb019179f9f528209a10f746ea1`](https://huggingface.co/datasets/Setloop/llm-attacker-paper-a/tree/2540f5904e5aaeb019179f9f528209a10f746ea1/closure-20260920/manuscript).
The [durable verification record](paper-a-release-verification-2026-09-20.json)
includes both release anchors and the independent raw check. Old HF prefixes
and failed/invalid evidence were preserved.

At 19:34:27 UTC, the same Qwen run
`766a85bc-0bfe-40d2-aa13-51df27f5e926` had **21/200 verified cells**: all eight
controls and 13 trained cells, with no recorded failed cell. GPU utilisation
was 92%. The controller remains `FRESH_FULL_GRID_RUNNING`; its transition-time
`wall_seconds` field is not current total elapsed time. No partial inference
was performed and no second job was started. Existing budget limits remain.

Both provisional arXiv packages were rebuilt into
`build/arxiv-preprints-20260920-paper-a-raw/INDEX.json`. Their PDFs and every
source ZIP entry match the recorded hashes. Paper B source is unchanged from
the previously reviewed Sasa revision. Figures and empirical analyses were
unchanged in this manuscript-link update. Remaining executable work is Qwen
completion, verification/reproduction, publication and final cross-paper
integration. The existing scheduler stays active with its 30-minute interval
after this ticket's completion receipt. Author metadata/licence choices and
the reviewed external exclusions remain explicit.

## Third acknowledged continuation: Qwen progresses within budget

Ticket `fbf79376-4eba-4c77-bfda-d9b2345d3841` was acknowledged in the existing
conversation. At 20:08:24 UTC, the same ORX run had **34/200 verified cells**:
eight controls and 26 trained cells, with no failures. ORX still reports the
run as running; controller PID `2971383` was alive, GPU utilisation was 91%,
and the evidence filesystem had approximately 1.54 TB free. The complete-grid
marker and final analysis were absent, as expected at this stage.

Completed trained cells took 149.9–164.8 seconds, with a median of 157.0 seconds.
Applying that median to the remaining 166 cells gives approximately 7.24 hours
for the grid alone. A practical 7–8 hour estimate is conditional on later cells
maintaining this pace and succeeding; it excludes analysis, reproduction and
publication. Prior work plus the controller process's elapsed time totals an
upper bound of 8,127 seconds (2.26 hours), within the original 86,400-second
ceiling. The running controller still enforces that ceiling and its analysis
reserve; no budget was added and no duplicate job was launched.

The frozen controller was inspected: after the complete grid succeeds it
invokes `experiments.qwen_p6.analyze` and writes `qwen-fp64-v2/grid.analysis.json`.
Its terminal success status is `COMPLETE_REVIEW_AND_PUBLICATION_REQUIRED`.
Future CPU reproduction must use this Qwen analyser, whose analytic layer
fraction uses the protocol's 28 layers, with the same full matrix and protocol
hash `981889c4df947496fbfaa6e138510ee031dc4bf65ab26e5169763c6131595085`.
Only after complete-grid qualification should a committed ORX child reproduce
that analysis from sealed evidence. Compare numerical content and cell hashes
while accounting explicitly for relocated absolute evidence-anchor paths.
No partial scientific analysis was run during this check.

All six completed Paper A, Sasa and agentic release manifests were downloaded
again at their immutable revisions and matched the earlier complete Git/LFS
verification reports. This was a manifest recheck, not another full archive
download or upload. Runtime reports are
`build/closure-queue-20260920/progress-fbf79376.json` and
`build/closure-queue-20260920/hf-manifest-check-fbf79376.json`. The existing
scheduler remains active; final manuscript integration waits for qualified
Qwen results. Original-setting and historical exclusions are unchanged.

## Subsequent monitoring checks

| Checked UTC | Ticket | Verified cells | Failures | Runtime and next action |
| --- | --- | --- | --- | --- |
| 20 September 20:43:21 | `f734b76a-ca9a-49cc-8e8a-c5d9c6a4fcdc` | 48/200: eight controls, 40 trained | 0 | Same ORX run/controller; GPU 94%; approximately 1.54 TB free. Prior work plus controller elapsed: 10,225 seconds, within 86,400. Median trained cell 157.3 seconds implies about 6.64 hours for the remaining grid, excluding later qualification/publication. |
| 20 September 21:16:18 | `d9c55409-4e33-401e-b9fe-3c3c8e346017` | 60/200: eight controls, 52 trained | 0 | Same run/controller; GPU 92%; 1.53 TB free. Prior work plus controller elapsed: 12,202 seconds (3.39 hours), within 86,400. Median trained cell 157.1 seconds implies about 6.11 hours for the remaining grid. No complete-grid marker or analysis yet; no new jobs. |

For these checks, ORX and live controller/grid/cell processes agree that the run
is active. The complete-grid marker and analysis are not yet present. The
completed HF dispositions were reconciled against the pinned-manifest check
at 20:09 UTC; no repeat upload or experiment was started. The runtime snapshots
are `build/closure-queue-20260920/progress-f734b76a.json` and
`build/closure-queue-20260920/progress-d9c55409.json`. The same scheduler
remains active, with complete-grid analysis/reproduction and final publication
still pending. No scientific claims changed.

## 21 September: complete Qwen grid, reproduction and archive dispatched

Ticket `f87a94bc-0417-46af-a90e-580592046ee7` followed the user-requested six-hour
interval. ORX run `766a85bc-0bfe-40d2-aa13-51df27f5e926` completed all 192 trained
cells and eight controls, then the frozen Qwen analyser. Its terminal status is
`COMPLETE_REVIEW_AND_PUBLICATION_REQUIRED`. The completion marker and analysis
contain the same protocol hash and all 200 cell anchors. Primary analysis SHA-256:
`ae7b2d03f88856dcbb0cae7e9d99a97e634059ebaa20e604c91180398af16e20`.
The controller used 32,829.601 seconds; with prior failures/canaries, recorded
wall time is 34,345.621 seconds (9.54 hours) within the 86,400-second envelope.
No model rerun is needed. Scientific claims remain unchanged pending release
qualification.

The complete run occupies approximately 142 GB. The CPU nodes have less free
space than this: soula 85 GB, voula 35 GB and sasa 1.4 GB when inspected. A
one-cell-at-a-time streaming attempt was frozen at `9e2198b` and launched as
`4420555d-f029-4057-a464-e7fb36376af9` on soula. Approximately 2.5 MB/s transfer
would exceed its existing two-hour ceiling, so this unanswered transport attempt
was cancelled and its process group stopped before any complete reanalysis.
Its source and partial evidence remain intact; 461 ORX seconds are recorded and
600 seconds conservatively charged to the budget.

The same unanswered reproduction node was repaired without altering the
completed execution branch. Source `61b8df5` uses the full-file hash inventory
produced by the archive writer on koula. On soula it checks those digests against
all sealed cell manifests, downloads every JSON/JSONL scoring and journal input,
verifies their actual bytes and evidence chains, and runs the unchanged Qwen
statistics through an explicit inventory-backed cell verifier. This does not
claim that soula independently downloaded every raw binary byte. Categories,
integers and all 200 normalised anchor hashes must agree exactly; numerical
outputs use prospective absolute and relative tolerances of `1e-10`. Seven
integrity/comparison boundary tests passed; an additional synthetic check
accepted an intact journal and rejected altered binary inventory/scoring bytes.
The repaired entrypoint, plan and analyser inputs were checked in a Git export.

- Reproduction experiment: `bdab9989-b662-479a-97ab-0fcd63a6c053`.
- Current soula run: `169c1e94-e8a5-4b99-bb83-eadadd73046a`, source `61b8df5`.
- Outputs: `/home/geo/.orx/runs/169c1e94-e8a5-4b99-bb83-eadadd73046a/repo/qwen-reproduction/`.
- The worker waits at most 5,400 seconds for the archive inventory; its complete
  prerequisite/transfer/reanalysis process is capped at 7,200 seconds.
- Archive worker: koula PID `3336987`, source helper `f39cc9a`, outputs under
  `/home/geo/qwen-release-20260921/archive/`; log is the sibling `archive.log`.
  `experiments/freeze_qwen_release.py` hashes every file while writing gzip
  archive parts bounded to 4 GiB. The parts round-trip and digest check passed.
  The two excluded canary directories are retained separately inside the archive.

The archive ends at `ARCHIVED_PENDING_CPU_REPRODUCTION`; it does not upload by
itself. Require the CPU `REPRODUCTION.json` to pass and its inventory digest to
match `ARCHIVE.json` before an additive HF evidence upload. All archive parts,
raw inventory, source commits, protocol, inputs, environment, primary and CPU
analyses and reproduction instructions must be included and pinned hashes
verified before changing manuscript claims. The auxiliary CPU report field
`independently_downloaded_scoring_files` currently counts directory entries too;
count regular files from the retained tree/inventory for publication rather than
using that field as a file-count claim. This does not affect numerical analysis.

The [durable execution record](qwen-fp64-closure-2026-09-21.json) and local
`build/qwen-fp64-closure-20260921/` preserve the collected completion reports,
source archives and frozen inputs. Post-run environment inspection records
Python 3.12.3, PyTorch 2.11.0+cu128, Transformers 5.15.0 and NumPy 2.5.2 on koula;
this is labelled as a post-run observation. Existing verified Paper A, Sasa and
agentic tiers remain unchanged. The existing scheduler stays active for
reproduction, HF verification and final preprint integration.

## 21 September follow-up at 2026-09-21T04:35:56.966203+00:00

The existing archive worker has processed 313,000 files and
148,605,461,381 source bytes in 2835.6 seconds.
Soula reproduction remains `WAITING_FOR_COMPLETE_ARCHIVE_INVENTORY`.
No duplicate jobs or HF uploads were started. Publication still requires the
completed archive inventory, CPU reproduction PASS and matching inventory hashes.
The queue remains active; manuscript claims remain unchanged.

## 21 September: Qwen archive and CPU qualification passed

Archive completed in 2,881.335 seconds: 317,552 files / 151,027,715,029 raw
bytes, compressed into 32 parts totalling 136,726,540,704 bytes. Soula run
`169c1e94` completed successfully, verifying all 200 cells and 1,400 journals.
Its raw inventory SHA256 matches the archive:
`99f692283b16a221caaa9774871d4575118a81f0d41a17c5b6366874c825693e`.
Actual downloaded regular scoring files number 160,003; the auxiliary report
counter includes directories. Three floating-point differences are at most
5.551115123125783e-17, within the frozen tolerance. All other numerical values,
categorical/integer values and normalised anchors agree. Full binaries were
hashed on koula, not independently downloaded to soula.

The complete qualified evidence is being staged for additive publication under
`Setloop/llm-attacker-paper-a/qwen-fp64-20260921/evidence`. The publisher uses
the existing immutable-prefix and pinned-hash verifier. Manuscript claim updates
remain gated on successful HF verification. No GPU work was repeated.

## 21 September: HF publisher authentication repaired

The initial publisher retried RepositoryNotFoundError because koula’s cached
HF credential was invalid (401 on identity verification). It uploaded no
verified Qwen release. PID 3382119 was stopped, and the existing local account
credential was supplied securely to its replacement process environment.
Remote identity and Paper A repository access passed. Publisher PID 3384533
resumes the same staged package with immutable-prefix checks; its log is
`/home/geo/qwen-release-20260921/publisher-auth-retry.log`. The helper now permits
resuming an existing stage. Source experiments and scientific results are
unchanged. HF verification and manuscript integration remain pending.

## 21 September 06:14 UTC: HF upload progressing

Publisher PID 3384533 remains active. Pinned HF revision
`b52df809748fe2207335ac11884ed9803737a2ea` contains 44 committed files
(344,012,545 bytes) under the Qwen evidence prefix. No archive part is committed
at that revision; the log shows part031 uploading. The last commit reports
33 files remaining. These are committed-file counts, not byte progress or an ETA.
Complete-release verification and manuscript updates remain pending. No restart
or duplicate job was needed.

## 21 September 06:46 UTC: first archive part pinned

Publisher 3384533 remains active. HF revision
`355bedc42d1703f191bc44fce11942109d0ae526` contains 45 files / 3,926,567,073
bytes. The committed part031 LFS SHA256 and size match ARCHIVE.json. The log
shows part000 transfer at 100%, but it is not in this pinned inventory yet.
The last commit reports 32 remaining files; complete-release verification is
still pending. No process was restarted and manuscript claims remain unchanged.

## 21 September 07:18 UTC: three archive parts pinned

Publisher 3384533 remains active. Revision
`2945b4d22d3236b09d209ccbe26707ba8cfc0569` contains 47 files / 12,516,501,665
bytes. Parts000,001,031 match ARCHIVE.json in LFS SHA256 and size. Part002
is uploading; the last commit reports 30 files remaining. Complete-release
verification and manuscript integration remain pending. No duplicate work started.

## 21 September 07:50 UTC: four archive parts pinned

Publisher 3384533 is active, uploading part003. Revision
`109324369bb90e38169fd7445b85fbf574b38b02` contains 48 files / 16,811,468,961
bytes. Parts000,001,002,031 match ARCHIVE.json in LFS SHA256 and size.
The last commit reports 29 remaining files; full-release verification and
manuscript integration remain pending. Existing jobs were preserved.

## 21 September 08:22 UTC: five archive parts pinned

Publisher 3384533 remains active, uploading part004. Pinned revision
`8cf7f75c3dc61427029090fcbad0efbb6cbdaae7` contains 49 files / 21,106,436,257
bytes. Parts000–003 and031 match ARCHIVE.json in LFS SHA256 and size.
The last commit reports 28 files remaining. Full verification and manuscript
integration remain pending; no duplicate jobs or restarts were needed.

## 21 September 08:54 UTC: six archive parts pinned

Publisher 3384533 remains active, uploading part005. Pinned revision
`3e8ce7b07a73f255d78a21fddcd910cae5731c55` contains 50 files / 25,401,403,553
bytes. Parts000–004 and031 match ARCHIVE.json in LFS SHA256 and size.
The last commit reports 27 files remaining. Full verification and manuscript
integration remain pending; no duplicate jobs or restarts were needed.

## 21 September 09:26 UTC: seven archive parts pinned

Publisher 3384533 remains active, uploading part006. Pinned revision
`512bdde238ce215495aa68e5ee94cc680ffe270a` contains 51 files / 29,696,370,849
bytes. Parts000–005 and031 match ARCHIVE.json in LFS SHA256 and size.
The last commit reports 26 files remaining. Full verification and manuscript
integration remain pending; no duplicate jobs or restarts were needed.

## 21 September 09:58 UTC: eight archive parts pinned

Publisher 3384533 remains active, uploading part007. Pinned revision
`041f918e7279431067f17b52b01acad68ffd6241` contains 52 files / 33,991,338,145
bytes. Parts000–006 and031 match ARCHIVE.json in LFS SHA256 and size.
The last commit reports 25 files remaining. Full verification and manuscript
integration remain pending; no duplicate jobs or restarts were needed.

## 21 September 10:30 UTC: nine archive parts pinned

Publisher 3384533 remains active, uploading part008. Pinned revision
`31e311390b9576f6436d248031c856787d0b2a71` contains 53 files / 38,286,305,441
bytes. Parts000–007 and031 match ARCHIVE.json in LFS SHA256 and size.
The last commit reports 24 files remaining. Full verification and manuscript
integration remain pending; no duplicate jobs or restarts were needed.

## 21 September: user requested paper integration before HF completion

Paper A now reports the completed 200-cell FP64 amendment, four variance tests,
98.75% intervals, D0-only held-out predictive improvement, compute and exact
CPU verification scope. Failed FP32 runs remain excluded. HF staging is explicitly
labelled incomplete and not an immutable verified release. Both preprint packages
were rebuilt and reviewed; see `qwen-manuscript-review-2026-09-21.json`.
The remaining work is final HF verification, immutable-link replacement and
publication of the updated manuscript release; author decisions remain open.

## 21 September 11:03 UTC: ten archive parts pinned

Publisher 3384533 remains active, uploading part009. Pinned revision
`52e780b443282de747b7f6af171fc2ab6821e136` contains 54 files / 42,581,272,737
bytes. Parts000–008 and031 match ARCHIVE.json in LFS SHA256 and size.
The last commit reports 23 files remaining. The Qwen paper update and reviewed
packages are complete at user request; full HF verification, final immutable-link
replacement and manuscript-tier publication remain pending. No restart was needed.

## 21 September 11:35 UTC: eleven archive parts pinned

Publisher 3384533 remains active, uploading part010. Pinned revision
`a8a3f97a5af0d6d27964d59d586d1df7ab244480` contains 55 files / 46,876,240,033
bytes. Parts000–009 and031 match ARCHIVE.json in LFS SHA256 and size.
The last commit reports 22 files remaining. Paper integration and package reviews
are complete; full HF verification, immutable-link replacement and manuscript
publication remain pending. No jobs were restarted.

## 21 September 12:07 UTC: twelve archive parts pinned

Publisher 3384533 remains active, uploading part011. Pinned revision
`ff5507b141e4ba28f6602088def57d4512cafe99` contains 56 files / 51,171,207,329
bytes. Parts000–010 and031 match ARCHIVE.json in LFS SHA256 and size.
The last commit reports 21 files remaining. Paper integration and package reviews
are complete; full HF verification, immutable-link replacement and manuscript
publication remain pending. No jobs were restarted.

## 21 September 12:39 UTC: thirteen archive parts pinned

Publisher 3384533 remains active, uploading part012. Pinned revision
`ff5bba33eea48183c6077a8d46c6162025f4040f` contains 57 files / 55,466,174,625
bytes. Parts000–011 and031 match ARCHIVE.json in LFS SHA256 and size.
The last commit reports 20 files remaining. Paper integration and package reviews
are complete; full HF verification, immutable-link replacement and manuscript
publication remain pending. No jobs were restarted.

## 21 September 13:11 UTC: fourteen archive parts pinned

Publisher 3384533 remains active, uploading part013. Pinned revision
`69e0da8929565cf0f2eb38b65f791327aa7fd64f` contains 58 files / 59,761,141,921
bytes. Parts000–012 and031 match ARCHIVE.json in LFS SHA256 and size.
The last commit reports 19 files remaining. Paper integration and package reviews
are complete; full HF verification, immutable-link replacement and manuscript
publication remain pending. No jobs were restarted.

## 21 September 13:46 UTC: fifteen archive parts pinned

Publisher 3384533 remains active, uploading part014. Pinned revision
`ec48a39bd1142b499d40a3820f93f2ed9449a66e` contains 59 files / 64,056,109,217
bytes. Parts000–013 and031 match ARCHIVE.json in LFS SHA256 and size.
The last commit reports 18 files remaining. Paper integration and package reviews
are complete; full HF verification, immutable-link replacement and manuscript
publication remain pending. No jobs were restarted.

## 21 September 14:18 UTC: seventeen archive parts pinned

Publisher 3384533 remains active, uploading part016. Pinned revision
`9363164c93e2839fd05141250624a5318e18036e` contains 61 files / 72,646,043,809
bytes. Parts000–015 and031 match ARCHIVE.json in LFS SHA256 and size.
The last commit reports 16 files remaining. Paper integration and package reviews
are complete; full HF verification, immutable-link replacement and manuscript
publication remain pending. No jobs were restarted.

## 21 September 14:50 UTC: eighteen archive parts pinned

Publisher 3384533 remains active, uploading part017. Pinned revision
`8ec86efbc93ded1cdffadf840f9f89bb39b6d329` contains 62 files / 76,941,011,105
bytes. Parts000–016 and031 match ARCHIVE.json in LFS SHA256 and size.
The last commit reports 15 files remaining. Paper integration and package reviews
are complete; full HF verification, immutable-link replacement and manuscript
publication remain pending. No jobs were restarted.

## 21 September 15:22 UTC: nineteen archive parts pinned

Publisher 3384533 remains active, uploading part018. Pinned revision
`333c5ed4882e4bacc855d1b4983d1fb4c887ac36` contains 63 files / 81,235,978,401
bytes. Parts000–017 and031 match ARCHIVE.json in LFS SHA256 and size.
The last commit reports 14 files remaining. Paper integration and package reviews
are complete; full HF verification, immutable-link replacement and manuscript
publication remain pending. No jobs were restarted.

## 21 September 15:54 UTC: twenty archive parts pinned

Publisher 3384533 remains active, uploading part019. Pinned revision
`13c0ee6199159fe2e1fd958e44f39e94fc3d518c` contains 64 files / 85,530,945,697
bytes. Parts000–018 and031 match ARCHIVE.json in LFS SHA256 and size.
The last commit reports 13 files remaining. Paper integration and package reviews
are complete; full HF verification, immutable-link replacement and manuscript
publication remain pending. No jobs were restarted.

## 21 September 16:26 UTC: twenty-one archive parts pinned

Publisher 3384533 remains active, uploading part020. Pinned revision
`8945b22e7c8b6b3a9c27c3922c367bdc3c4f2d73` contains 65 files / 89,825,912,993
bytes. Parts000–019 and031 match ARCHIVE.json in LFS SHA256 and size.
The last commit reports 12 files remaining. Paper integration and package reviews
are complete; full HF verification, immutable-link replacement and manuscript
publication remain pending. No jobs were restarted.

## 21 September 16:58 UTC: twenty-two archive parts pinned

Publisher 3384533 remains active, uploading part021. Pinned revision
`f2762071931fb0db435618eeb9fb9ec7c3a71377` contains 66 files / 94,120,880,289
bytes. Parts000–020 and031 match ARCHIVE.json in LFS SHA256 and size.
The last commit reports 11 files remaining. Paper integration and package reviews
are complete; full HF verification, immutable-link replacement and manuscript
publication remain pending. No jobs were restarted.

## 21 September 17:30 UTC: twenty-three archive parts pinned

Publisher 3384533 remains active, uploading part022. Pinned revision
`f69743173d265277eb36e2695ce924a6947efd63` contains 67 files / 98,415,847,585
bytes. Parts000–021 and031 match ARCHIVE.json in LFS SHA256 and size.
The last commit reports 10 files remaining. Paper integration and package reviews
are complete; full HF verification, immutable-link replacement and manuscript
publication remain pending. No jobs were restarted.

## 21 September 20:38 UTC: twenty-nine archive parts pinned

After the user-requested three-hour delay, publisher 3384533 remains active,
uploading part028. Pinned revision `1e89bd1f37655d373cb09a5720925125e8195bcc`
contains 73 files / 124,185,651,361 bytes. Parts000–027 and031 match
ARCHIVE.json in LFS SHA256 and size. The last commit reports four files
remaining: three archive parts and the release manifest. Paper integration
and package reviews are complete; full HF verification, immutable-link
replacement and manuscript publication remain pending. No jobs were restarted.

## 21 September 21:10 UTC: thirty archive parts pinned

Publisher 3384533 remains active, uploading part029. Pinned revision
`62be6298144689d0128c084c8c7f4a92b8ef136d` contains 74 files / 128,480,618,657
bytes. Parts000–028 and031 match ARCHIVE.json in LFS SHA256 and size.
The last commit reports three files remaining: two archive parts and the
release manifest. Paper integration and package reviews are complete; full
HF verification, immutable-link replacement and manuscript publication
remain pending. No jobs were restarted.

## 21 September 21:42 UTC: thirty-one archive parts pinned

Publisher 3384533 remains active. Pinned revision 28c6c921f3ed315f1f099b08d81a37fd91fef788 contains 75 files / 132,775,585,953 bytes; parts000–029 and031 match ARCHIVE.json in LFS SHA256 and size. Two files remain at the last commit: part030 and the manifest. The log displays part030 at 100% transfer, but it is not committed in the checked revision. Full release verification, immutable-link replacement and manuscript publication remain pending. Paper integration/review is already complete. Preserve the existing publisher.

## 21 September: executable closure complete

Qwen evidence passed full pinned verification at `c0981bf75ba4a62e4388c3477c63e9f90d5d01fd`: 77 files / 137,070,560,283 bytes. Independent verification checked exact inventory, manifest SHA256, all LFS SHA256 metadata and downloaded non-LFS bytes. No second full archive download was claimed. Publisher exited.

Paper A now links the immutable evidence release. Both final packages compiled twice without warnings and passed source ZIP/PDF hash checks. Only Paper A page 9 text changed; its rendered layout was reviewed. Paper B scientific source is unchanged. Final manuscript tier: `c120b3b21b53e029bf7cb615dfc576637b7d8f76`, 7 files / 456,091 bytes, all pinned hashes verified. Packages: `build/arxiv-preprints-20260921-qwen-final/INDEX.json`.

Queue active=false: executable closure work is complete. Author metadata, category/cross-list, licence, overlap disclosure and separately authorised submission remain. Later venue formatting is undecided. Original-setting and historical external exclusions remain unchanged; no submission was made.
