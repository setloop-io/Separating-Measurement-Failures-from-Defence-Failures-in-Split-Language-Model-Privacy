# Unrecorded run outcomes — 20 September 2026

**This document records an absence of evidence, not a result.**

Two experiment runs were launched, recorded a running state, and then left no
completion record in this checkout. This note states only what the committed
artefacts prove and where they stop. It does **not** state how either run
ended, because no artefact in this checkout says how either run ended. Nothing
here should be cited as an outcome, a failure, a success, or a cause.

Both runs executed on the GPU host **koula**, in isolated checkouts outside
this repository. This repository holds only the launch-time collections that
were copied back. The actual terminal state of each run — including any
partial outputs, logs, or error traces — exists (if at all) on koula, and must
be confirmed there before anything is claimed about either run.

This note was written on host `devbox`. Neither run's process table is
reachable from here.

---

## 1. Qwen P6 replication grid

### Last recorded state

`grid_running`, written at **2026-09-17T14:23:15Z**.

**Proving file:** `results/p6-qwen-launch-20260917/qwen-queue-status.json`

```json
{
  "status": "grid_running",
  "pid": 562873,
  "utc": "2026-09-17T14:23:15Z",
  "wall_s": 141.69530735199805,
  "projection_s": 15810.807652080635,
  "remaining_s": 86258.304894089,
  "canaries_excluded": true
}
```

The append-only queue log `results/p6-qwen-launch-20260917/qwen-queue.log`
contains exactly three lines — `canary_running` (14:20:53Z),
`control_canary_running` (14:21:45Z), `grid_running` (14:23:15Z). The third
line is identical to the status file above. **There is no fourth line.** The
log stops at the moment the grid started, 141.7 seconds into the run.

Supporting launch record: `results/p6-qwen-launch-20260917/launch.json` gives
`"node": "koula"`, checkout `/home/geo/llm-attacker-p6-qwen-20260917`,
`"grid_started_utc": "2026-09-17T14:23:15Z"`, `"grid_driver_pid": 565319`,
`"maximum_gpu_wall_hours": 24`, and `"status_at_collection": "grid_running"`.

For scale: `projection_s` is 4.39 hours of projected grid work; `remaining_s`
is 23.96 hours of remaining budget. The queue enforced a 24-hour total
deadline, which would have expired at **2026-09-18T14:23:15Z**. Per
`docs/p6-qwen3-followup-2026-09-17.md`, on expiry the queue terminates its own
process group and preserves partial outputs.

### What does not exist in this checkout

- No status line after `grid_running`.
- No `results/qwen-grid-v1/` directory — the declared grid-success artefact
  `results/qwen-grid-v1/grid-complete.json` is absent.
- No `results/qwen-grid-v1.analysis.json`.
- No error log, traceback, abort record, or non-finite-value record of any
  kind.

### Note on PID 562873

PID 562873 is recorded in `qwen-queue-status.json`, `qwen-queue.log`, and
`launch.json` as `queue_pid`. **That PID belongs to koula, not to this host.**
This note was written on `devbox`. Checking the local process table for 562873
is meaningless and proves nothing about the run; no such check is offered here
as evidence. Whether the process is still alive can only be determined on
koula.

### Discrepancy with Paper A

Paper A (`paper/paper_a/main.tex`) states that the Qwen extension stopped
because of "a non-finite training result". The same cause appears in
`docs/paper-a-closure-2026-09-17.md` and
`docs/open-commitments-plan-2026-09-16.md`.

**No artefact in this checkout records that cause.** The last recorded state is
`grid_running` with no subsequent entry. The claim may well be correct and
supported by evidence on koula, but it is not supported by anything committed
here. This is a gap between the manuscript and the committed artefacts, and it
is recorded here as such — not as a contradiction of the manuscript, and not as
a correction to it. No scientific claim is changed by this note.

### To confirm on koula

Checkout `~/llm-attacker-p6-qwen-20260917` (as `/home/geo/...` per
`launch.json`):

```bash
ssh koula 'cat ~/llm-attacker-p6-qwen-20260917/results/qwen-queue-status.json'
ssh koula 'cat ~/llm-attacker-p6-qwen-20260917/results/qwen-queue.log'
ssh koula 'ls -la ~/llm-attacker-p6-qwen-20260917/results/qwen-grid-v1/'
ssh koula 'cat ~/llm-attacker-p6-qwen-20260917/results/qwen-grid-v1/grid-complete.json'
```

### Files checked for this section

All ten files in `results/p6-qwen-launch-20260917/`:
`canary-grid-complete.json`, `canary-run-index.jsonl`, `collection-sha256.json`,
`control-canary-cell.json`, `environment.json`, `launch.json`,
`qwen-budget-gate.json`, `qwen-queue-status.json`, `qwen-queue.log`,
`trained-canary-cell.json`.

Also checked: `results/p6-qwen-input-v1/`, `docs/p6-qwen3-launch-2026-09-17.md`,
`docs/p6-qwen3-followup-2026-09-17.md`, `docs/p6-corrected-followup-2026-09-17.md`,
`experiments/crossed_span_protocol.p6.qwen3.v1.json`, and the whole of
`results/` for any `qwen-grid` output.

Note: `canary-grid-complete.json` reads `"status": "complete"` with
`"canary": true, "count": 1`. That is the **excluded integrity canary**, not the
grid. It is not a completion record for the replication grid.

---

## 2. BiSR fidelity campaign

### Last recorded state

Checkpoint preparation **RUNNING**; prediction campaign
**WAITING_FOR_CHECKPOINTS**.

**Proving files:**

- `experiments/bisr_fidelity_running.20260918.json` —
  `"checkpoint_preparation": "RUNNING"`,
  `"prediction_campaign": "WAITING_FOR_CHECKPOINTS"`, `"host": "koula"`,
  `"root": "/home/geo/llm-attacker-bisr-fidelity-20260918"`,
  `"fidelity_established": false`, `"expected_prediction_cells": 24`,
  `"source_commit": "8f8a4c4aade436aa31ef79ac294d97f1d392b4d3"`.
- `docs/bisr-fidelity-running-2026-09-18.md` — the launch snapshot: first GRU
  complete, second at epoch 10/20; preparation worker 1100771, supervisor
  1100770, campaign supervisor 1102383. It states plainly: "No fidelity pass
  has yet been established."

These were committed in `5a192eb` at **2026-09-18 07:46:44 +0100**
(2026-09-18T06:46:44Z). That is the most recent commit on this branch's history
for either run.

### Deadline

`docs/bisr-fidelity-running-2026-09-18.md` sets the deadline at "preparation
start plus 35.9 hours", reserving 0.1 hours within the approved 36 GPU-hour
ceiling recorded in `experiments/bisr_execution_admission.v1.json`.

Preparation was already running when `5a192eb` was committed, so preparation
started at or before 2026-09-18T06:46:44Z. The deadline therefore expired **no
later than 2026-09-19T18:40:44Z**, and in fact earlier by however long
preparation had already been running. Either way it has expired. The exact
preparation start timestamp is not recorded in this checkout; it is in
`run-v1/status.json` on koula.

### What does not exist in this checkout

- No result, completion, verdict, or scoring document. The most recent BiSR
  document of any kind is dated 2026-09-18.
- No `results/bisr-*` directory. `results/` contains only
  `heldout_access.journal`, `p6-qwen-input-v1`, `p6-qwen-launch-20260917`,
  `validation-corpus-v1`, and the sixteen `validation-heldout-v3-r3-shard*`
  directories.
- No record of preparation completing or failing, no campaign status, no
  deadline-expiry record, no scoring output.
- `"fidelity_established": false` remains the last recorded value, and no later
  record supersedes it.

### To confirm on koula

Root `~/llm-attacker-bisr-fidelity-20260918`, using the commands already given
in `docs/bisr-fidelity-running-2026-09-18.md`:

```bash
ssh koula 'cat ~/llm-attacker-bisr-fidelity-20260918/run-v1/status.json'
ssh koula 'tail -5 ~/llm-attacker-bisr-fidelity-20260918/run-v1/preparation.log'
ssh koula 'cat ~/llm-attacker-bisr-fidelity-20260918/campaign-v1/status.json'
```

Preparation journals are under `run-v1/checkpoints/`; campaign events, trained
gate traces, per-cell logs, prediction journals and any final analysis are
under `campaign-v1/`. A complete-set run record should also exist in the
separate run-record Git repository under `campaign-v1/records/`.

### Files checked for this section

`docs/bisr-fidelity-running-2026-09-18.md`,
`docs/bisr-fidelity-progress-2026-09-18.md`,
`docs/bisr-execution-plan-2026-09-18.md`, `docs/bisr-fresh-inputs-2026-09-18.md`,
`docs/bisr-fidelity-audit-2026-09-17.md`,
`docs/bisr-fidelity-specification-2026-09-17.md`,
`docs/bisr-conformance-amendment-v2-2026-09-17.md`,
`docs/bisr-fresh-input-amendment-v3-2026-09-17.md`,
`docs/bisr-implementation-2026-09-17.md`,
`docs/bisr-upstream-preflight-2026-09-17.md`,
`docs/bisr-upstream-repair-2026-09-17.md`,
`experiments/bisr_fidelity_running.20260918.json`,
`experiments/bisr_fidelity_progress.20260918.json`,
`experiments/bisr_execution_admission.v1.json`,
`experiments/bisr_full_campaign.v1.json`, and the full listing of `results/`.

---

## Status of both runs

| Run | Last recorded state | Timestamp of that record | Completion record here | Outcome |
|---|---|---|---|---|
| Qwen P6 grid | `grid_running` | 2026-09-17T14:23:15Z | none | **unknown — must be confirmed on koula** |
| BiSR fidelity | preparation `RUNNING`, campaign `WAITING_FOR_CHECKPOINTS` | 2026-09-18T06:46:44Z (commit `5a192eb`) | none | **unknown — must be confirmed on koula** |

"Unknown" means unknown. Neither run should be described as completed, failed,
aborted, or budget-stopped on the strength of this checkout. Until the koula
artefacts are retrieved, the honest statement for both is that the last
committed evidence shows the run in progress and nothing more.
