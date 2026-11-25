# R1 GPU engineering execution — 2026-09-16 (owned GB10 nodes)

First R1 execution on the owned GPU nodes, authorized by the owner on 2026-09-16.
**Engineering evidence only: every stage is `scientific_evidence: false`; no
scientific claim is admitted and the claim ledger is unchanged.**

## What ran

| Stage | Node | Execution | Runs | Bundle | Outcome |
|---|---|---|---|---|---|
| performance_calibration (Stage B) | koula | engB001 | 4/4 | v1 `45d7682e` | completed, analysis verified 4/4 |
| engineering (Stage A, attempt 1) | toula | engA001 | 4/14 | v1 `45d7682e` | **failed**: remote-service defense device bug; admission consumed, records preserved |
| engineering (Stage A, attempt 2) | toula | engA002 | 14/14 | v2 `e123089f` (fix `fcd0922`) | completed in ~65 s, analysis verified 14/14 |

engA001 failed on `smoke-gradient-noise--s61001--trained` (CUDA/CPU device mismatch
in `remote_service._defend_outbound`, fixed by moving noise to the gradient device in
`fcd0922`; R1 systems battery 57/57 after the fix; new packet per the code-change
rule). Stage B's only cell carries no gradient defense, so its measurements are
unaffected by the fix; it was not re-run.

## Measured engineering numbers (not scientific results)

- Stage B full-size (8 blocks, d256, batch 32, seq 128, 100 steps): split runs
  12.6–15.3 s wall (~0.13–0.15 s/step including evals/captures), all-local 7.0 s;
  peak client ~686 MiB, remote ~174 MiB; ~1.43 GB actual wire per split run;
  84–124 MiB outputs per run. Basis for scientific budget sizing (~26 GPU-hours
  for the 308-run draft at 2000 steps/run, before attack fits).
- Stage A micro (d32, 8 steps): 14 runs in ~65 s whole-stage (cap 300 s); process
  overhead measured ~54 s in the acceptance smoke. Total wire 7.4 MB.
- Environment acceptance: torch 2.14.0+cu130 / CUDA 13.0 on both GB10 nodes
  (Blackwell confirmed by real forward/backward/optimizer/checkpoint smoke, not
  just wheel-tag reads); driver 580.159.03; one GPU per container; images built
  from `environment/r1/cuda/uv.lock` on arm64 base
  `python:3.12.12-slim-bookworm@sha256:228390ec…`. dcgmi unavailable (no sudo):
  recorded deviation, nvidia-smi is the GPU health evidence.
- The engineering descriptive contrasts in the analyses are engineering behavior
  only; privacy and joint feasibility are UNDETERMINED by construction.

## Contents

- `acceptance-toula/`, `acceptance-koula/` — budget reviews, environment reviews,
  admissions, freeze logs, `inspect-environment` records (the attempt-1 and
  attempt-2 acceptance sets, including the consumed attempt-1 approval).
- `stage-records/engA001-failure/` — launch, admission, claim, panel manifest,
  execution summary and the failure record of the consumed attempt.
- `stage-records/engA002/`, `stage-records/engB001/` — launch, admission, claim,
  panel manifest, execution summary and verified analyses.
- `smoke-results.json` — environment acceptance smoke on both nodes.
- `retrieved-tree-inventory.json` — SHA-256 of every retrieved file (855 files,
  ~445 MB). Full raw trees (runs, checkpoints, captures, wire records, manifests)
  are retained on the orchestrator at `/Users/geo/r1-executions-20260916/` and on
  the origin nodes; this package binds them by hash without copying bulk tensors
  into git.
- Preparation evidence: `../r1-preparation-20260916/` (v1, corrected transport) and
  `../r1-preparation-20260916-v2/` (v2 after the defense fix).
- Node inventory: `../node-inventory-20260916.json`.

Two-host operation remains unbuilt and was not attempted; all execution is
same-host two-process loopback inside one container per node.
