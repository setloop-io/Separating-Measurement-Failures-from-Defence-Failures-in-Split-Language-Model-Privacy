# Status — 2026-09-16, GPU engineering executed on owned nodes

Supersedes nothing; extends `2026-09-13-2150-status.md`. The owner authorized the
owned nodes toula/koula (NVIDIA GB10) and Soula/Voula/Sasa (CPU) for R1 work on
2026-09-16. Metered cloud remains unauthorized.

## What changed today

- Node inventory measured and retained (`evidence/node-inventory-20260916.json`):
  toula/koula are GB10 Grace-Blackwell (aarch64, 121 GiB, driver 580.159.03 /
  CUDA 13.0 — the R580+ requirement is met; GPUs idle at start). CPU nodes are
  4-core aarch64 Debian 13 (Voula 15 GiB; Soula/Sasa 7 GiB).
- CUDA runtime empirically validated: torch 2.14.0+cu130 runs on GB10 — real
  forward/backward/optimizer/checkpoint smoke in the acceptance container on both
  nodes (previously only a static wheel-tag read). One GPU per container;
  `nvidia-smi` healthy. **Deviation:** dcgmi unavailable (no sudo); recorded in
  the environment reviews.
- Images `r1-cuda-runtime:20260916` built per node from
  `environment/r1/cuda/uv.lock` on arm64 base
  `python:3.12.12-slim-bookworm@sha256:228390ec…`; per-node image IDs recorded in
  the environment reviews (no registry digests; operator-supplied local identity).
- **Stage B (`performance_calibration`) complete on koula, engB001, 4/4 runs.**
  Measured full-size behavior: ~0.13–0.15 s/step (100-step runs, 12.6–15.3 s),
  peak client ~686 MiB / remote ~174 MiB, ~1.43 GB actual wire per split run,
  84–124 MiB outputs/run. This is the measured basis for scientific budget sizing
  (~26 GPU-hours for the 308-run draft at 2000 steps/run, before attack fits).
- **Stage A attempt 1 failed (engA001, 4/14).** Run
  `smoke-gradient-noise--s61001--trained` hit a CUDA/CPU device mismatch in
  `remote_service._defend_outbound` (noise moved by dtype only). The admission was
  consumed exactly as designed; failure records preserved.
- Fix committed `fcd0922` (noise moves to the gradient device; matches both
  sibling defense sites). R1 systems battery 57/57 after the fix. New source seal
  `c96d16ecde0c8982eb0cf1761310b349ba4f1611e92e73c54be4db0c30f097e0` (97 files;
  supersedes `9e39b3db…` for future work). New v2 packet per the code-change rule
  (`evidence/r1-preparation-20260916-v2/`, bundle `e123089f…`).
- **Stage A attempt 2 complete on toula, engA002, 14/14 runs in ~65 s** (whole-stage
  cap 300 s; measured process overhead ~54 s).
- Execution evidence curated at `evidence/r1-gpu-engineering-20260916/` with the
  full retrieved-tree hash inventory (855 files, ~445 MB raw retained on the
  orchestrator and origin nodes).
- `AGENTS.md` and this handoff series now record the owned-node authorization;
  the metered-cloud budget draft in `docs/R1_REMOTE_COMPUTE.md` is superseded by
  the owned-hardware execution (that file is provenance-bound and is updated at
  the scientific freeze, not piecemeal).

## Execution defects found and fixed today

- macOS tar emitted xattr PAX records that materialized as 218 AppleDouble files
  on GNU-tar extraction; the Linux `verify-bundle` correctly refused the polluted
  copy (fail-closed working as intended). Archive rebuilt with
  `COPYFILE_DISABLE=1`; preparation package corrected and re-committed.

## Where the study stands

Six stages, 308 scientific runs plus 32 engineering runs. Engineering: local CPU
validation (14/14, sealed 2026-09-13), Stage A (14/14, engA002), Stage B (4/4,
engB001) — **all engineering stages are now complete.** Zero scientific runs have
executed and zero claims are admitted; the claim ledger is unchanged.

The scientific stack — scientific panel loader, 12-combination attack suite with
fresh nulls/positive controls and qualification, three-valued held-out admission,
and the frozen protocol/thresholds — remains the long pole and gates everything
downstream. It needs no GPU. Then: scientific freeze (re-seal, bound-doc updates
including `docs/R1_REMOTE_COMPUTE.md`), calibration (76 runs), exploration
(232 runs). Optional stretch: genuine two-host engineering validation
(toula↔koula over a protected tunnel; needs a wire bind-address change).
