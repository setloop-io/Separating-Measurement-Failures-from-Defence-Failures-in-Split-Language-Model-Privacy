# Paper 3 R1 — GPU authorization handoff

Status: **CPU implementation complete and sealed against the final source; committed,
pushed and bundled; GPU authorization not yet requested.**
Scientific calibration, exploration, confirmation and claims remain prohibited.

Updated 2026-09-16: **the GPU engineering lane is executed on owned hardware.** The
owner authorized the owned GB10 nodes toula/koula on 2026-09-16; Stage B
(`performance_calibration`, engB001, 4/4 runs) completed on koula and Stage A
(`engineering`, engA002, 14/14 runs) completed on toula after the engA001 attempt
failed on a defense-device bug (fixed in `fcd0922`, new packet per the
code-change rule, consumed admission preserved). All four authorization blockers
below are now closed **for the two engineering stages only**. The metered-cloud
budget section remains a superseded draft; budgets were executed as owned-hardware
wall-clock caps with a nominal USD 0.50/h accounting rate. Scientific stages stay
blocked: no scientific stack, no frozen protocol, no scientific admission exists.
See `docs/status/2026-09-16-gpu-engineering-status.md`.

Updated 2026-09-13: the first two authorization blockers are cleared. The stack is
committed as `6ebc2bf..5f9fcb0` on branch `r1-cpu-systems-seal` and pushed to origin,
and a bundle prepared from that clean commit was verified from a disposable extraction
and retained at `evidence/r1-preparation-20260913/`.

## Verified local evidence (sealed 2026-09-13 against final source)

All items below were measured from the final unchanged source tree
(source inventory `9e39b3db26b21a2e975d8159c0f6e6f62faca8449060fff2ad227afb36f8b2a0`,
working-tree patch `7dbc0aa7e9d5a36d77902928e4561eda1963cae285a54d5c50b48736d90ed1af`
over commit `fd4b014`) and are retained in the replacement evidence package.
The superseded package `evidence/r1-cpu-validation-20260913/` binds an earlier
working tree and is preserved for audit only.

- Real two-process trusted/remote execution over authenticated loopback TCP.
- Remote process packet contains no corpus, panels, checkpoints, outputs or trusted model state
  (single remote-packet inventory digest `12c662e15f414573…` across all runs).
- 14/14 retained CPU runs completed (`execution-id cpu-seal-20260913`); zero failures,
  interruptions or missing runs; 14 prospective manifests with semantic checks.
- 24 committed remote steps with durable prepare/commit records.
- Client and remote wire totals match for every split run
  (total 2,995,027 wire bytes; per-run range 249,148–249,978).
- Four trained runs retained 48 draft target/channel attack results and fitted banks.
- Full relocated replay of all 14 runs verified against the current tree before packaging.
- Locked host full suite: **312 tests + 12 subtests passed** (277.5s).
- R1 systems targeted battery: **57 tests passed** (104.1s).
- Pinned CPU container systems battery: **57 tests passed** (260.6s) in local image
  `sha256:b635a440496dc832533c203c4ae660ed5df4792195edb12d57129aaf208378ec`
  (1,992,572,703 bytes); no registry digest exists.
- Manuscript: 6 pages, zero LaTeX warnings.
- CPU evidence package: `evidence/r1-cpu-validation-20260913-v2/`
  - 528 files, 33,707,654 extracted bytes, 24,286,417 compressed bytes
  - archive SHA-256 `7c76754f19b8479a2536c8cf2b362a28769f8ce6f7f004aab1d9fb0ab3e60a59`
  - fresh-extraction verification: 528/528 files, zero failures
  - retained logs: `full-test-suite.log`, `r1-targeted-tests.log`, `container-validation.log`

The sealed run used a then-uncommitted working tree over `fd4b014`; every run binds the
same source inventory, patch and environment digest. That tree is now committed as
`6ebc2bf..5f9fcb0`, and the source inventory still recomputes to
`9e39b3db26b21a2e975d8159c0f6e6f62faca8449060fff2ad227afb36f8b2a0` over 97 bound files,
so the seal survived the commit unchanged. Note that
`outputs/r1-cpu-validation-20260913-v2/validation_summary.json` still records
`working_tree_clean: false` and `git_commit: fd4b014`, which was accurate at capture;
`master` also remains at `fd4b014`. The binding is intact; only that metadata is
historical.

## Proposed CUDA environment

`environment/r1/cuda/uv.lock` resolves PyTorch `2.14.0+cu130` and its wheel hashes.
No GPU or image has validated it.

Architecture coverage, determined 2026-09-13 by reading the lock (not by installing):
the lock is arm64-complete. `torch 2.14.0+cu130` ships
`manylinux_2_28_aarch64`, `manylinux_2_28_x86_64` and `win_amd64` for cp312, and across
all 52 locked packages none has Linux wheels without an aarch64 wheel; the single
sdist-only entry is the project's own `pcr-r1-cuda-runtime`, which is
architecture-agnostic. So an ARM64 host such as NVIDIA GB10 Grace-Blackwell needs no
source builds. This is a static read of wheel tags only; it does not establish that the
CUDA 13 build runs correctly on Blackwell, which item 5 below still has to prove.

Before acceptance:

1. Select an immutable `IMAGE@sha256:DIGEST`; mutable tags are rejected.
2. Require a CUDA-13-compatible host driver, currently R580 or newer, and recheck
   provider compatibility at launch.
3. Require exactly one exclusive GPU with at least 24 GiB advertised memory.
4. Pass `nvidia-smi`, `dcgmi health -c`, and `dcgmi diag -r 1`.
5. Inside the image, verify PyTorch/CUDA/cuDNN/NCCL versions, one visible GPU and a
   real forward/backward/optimizer/checkpoint smoke operation.
6. Retain client and remote peak allocated/reserved memory, allocator retries,
   device free memory, step time, evaluation time, output bytes and wire bytes.
7. Use a provider-side shutdown independent of the Python watchdog.

## Budget proposed for later authorization

### Stage A — micro engineering gate

- Workload: 14 runs x 8 steps, engineering only.
- Hardware: one exclusive >=24 GiB NVIDIA GPU.
- Python process cap: 5 minutes.
- Provider paid-lifetime cap: 45 minutes, including setup and teardown.
- Maximum total charge: USD 2.00.
- Output cap: 256 MiB; ephemeral volume cap: 5 GiB.
- Stop immediately on health, CUDA, source, environment, packet-isolation,
  checkpoint, semantic-artifact, wire-accounting, timeout or budget failure.

### Stage B — performance calibration, separately authorized after Stage A

- Workload: four modes x 100 steps, engineering only.
- Provider paid-lifetime cap: 90 minutes.
- Maximum total charge: USD 4.00.
- Output cap: 2 GiB; ephemeral volume cap: 10 GiB.
- Purpose: measure per-mode runtime, client/remote GPU memory, capture overhead,
  output growth and actual transport bytes.

No budget for the 76 calibration or 232 exploration runs should be authorized until
Stage B produces measured per-cell cost and storage estimates and the scientific
packet is separately frozen.

## Authorization blockers

- ~~Working tree must be reviewed and committed.~~ **Cleared 2026-09-13**: 8 commits
  `6ebc2bf..5f9fcb0` on `r1-cpu-systems-seal`, pushed; working tree clean.
- ~~A new workload bundle must be prepared from that clean commit and restored in a
  disposable directory.~~ **Cleared 2026-09-13**: prepared from `5f9fcb0` with
  `working_tree_clean: true`, extracted to a disposable directory and verified from the
  extracted copy (`status: verified`, 183 files); retained at
  `evidence/r1-preparation-20260913/`.
- ~~Provider, region, exact hourly/all-in rate, image digest and external
  shutdown mechanism must be entered into the signed budget/environment reviews.~~
  **Cleared 2026-09-16 for engineering stages**: executed on owned nodes toula/koula
  (no provider, no metered rate); reviews recorded the per-node local image digests,
  measured environment digests, nominal USD 0.50/h accounting rate and wall-clock
  caps (300 s Stage A, 1800 s Stage B). External shutdown on owned hardware is the
  owner-attended session plus the runner watchdog; no billed lifetime exists.
- ~~Stage A and Stage B require separate admissions; failure consumes an
  admission.~~ **Cleared 2026-09-16**: three admissions executed exactly as designed —
  engB001 admitted and completed (4/4), engA001 admitted and consumed by the
  defense-device failure (records preserved), engA002 admitted under a fresh attempt-2
  review and completed (14/14). Scientific admissions remain uncreated.

## Compatibility references

- PyTorch 2.14 release support: https://pytorch.org/blog/pytorch-2-14-release-blog/
- PyTorch installation selector: https://pytorch.org/get-started/locally/
- GPU framework/container guidance: https://ai-infrastructure.net/gpu-frameworks/
- NVIDIA container runtime: https://ai-infrastructure.net/container-toolkit/
- GPU health gating: https://ai-infrastructure.net/gpu-health-gating/
- PyTorch allocator measurements: https://ai-infrastructure.net/pytorch-cuda-memory-allocator/
