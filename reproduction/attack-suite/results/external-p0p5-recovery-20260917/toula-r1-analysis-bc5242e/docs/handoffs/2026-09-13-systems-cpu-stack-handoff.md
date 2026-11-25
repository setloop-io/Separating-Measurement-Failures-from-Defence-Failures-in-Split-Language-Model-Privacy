# Handoff — Paper 3 R1 systems path: finish and re-seal the CPU stack

> **Point-in-time record, superseded 2026-09-13.** Superseded first by
> `2026-09-13-cpu-sealed-handoff.md` and then by the commit itself: the CPU stack was
> committed as 8 commits `6ebc2bf..5f9fcb0` on branch `r1-cpu-systems-seal` and pushed.
> Statements below that the tree is uncommitted were true when written and are now
> false. The body is preserved unedited as the record of that session.

You are taking over the systems-paper implementation of Paper 3 R1 in `/home/epappas/workspace/spacejar/paper-3-privacy-constrained-remote`. The goal is to test the privacy–quality–useful-distribution triangle through a real trusted-client/remote-service boundary, not merely an in-process model cut. A substantial CPU implementation exists but is **uncommitted and still in validation**. Your immediate job is to review the current diff, finish validation, regenerate evidence from the final unchanged source, and stop at the GPU authorization boundary. Read the files below; do not re-derive the research decision.

## 0. How to work

- Work from evidence, not existing “complete” prose. Current source has moved beyond the retained CPU evidence package.
- Keep source unchanged while `run_local_validation()` is executing: provenance deliberately fails completion if any source/config/protocol file changes mid-run.
- Use narrow test groups while editing; run the expensive full suite and final retained validation only after the source is stable.
- Use real models, real Tiny Shakespeare bytes, real sockets and real artifacts. Do not restore mock training or placeholder `.pt` files.
- Do not provision a GPU, approve spend, freeze scientific stages, or admit claims.

## 1. Read these first, in order

1. `/home/epappas/.claude/plans/fizzy-whistling-engelbart.md` — approved CPU systems implementation plan and GPU stop boundary.
2. `AGENTS.md` — R1 reset, zero active claims, historical evidence restrictions.
3. `docs/PREREGISTRATION_R1.md` — draft scientific protocol, panels, attacks, gates and admission blockers.
4. `docs/R1_RESEARCH_PLAN.md` — 19-cell design and staged workload.
5. `docs/GPU_AUTHORIZATION_HANDOFF.md` — prior CPU evidence and proposed GPU micro/performance budgets; currently needs regeneration against final source.
6. Core systems code:
   - `src/pcr/wire.py` — authenticated framed tensor transport and exact wire accounting.
   - `src/pcr/remote_service.py` — corpus-free remote surrogate state machine, optimizer and checkpoints.
   - `src/pcr/distributed_split.py` — trusted client, custom autograd boundary, global clipping and fail-stop commits.
   - `src/pcr/local_remote.py` / `src/pcr/r1_packets.py` — subprocess lifecycle and separate client/remote packets.
7. Scientific-support code:
   - `src/pcr/r1_panels.py` — deterministic four-region panels.
   - `src/pcr/r1_attacks.py` — 3 targets × 4 channels, qualification and three-valued status.
   - `src/pcr/artifacts.py` — semantic validators.
   - `src/pcr/r1_local_validation.py` — real 14-run CPU validation and replay.
8. Integration/provenance:
   - `src/pcr/manifest.py`, `src/pcr/r1.py`, `scripts/run_r1.py`.
9. Tests: `tests/test_wire.py`, `test_distributed_split.py`, `test_r1_panels.py`, `test_r1_attacks.py`, `test_artifacts.py`, `test_r1_packets.py`, `test_r1_cpu_e2e.py`, `test_r1.py`, `test_manifest_resume.py`.

## 2. Where you are starting

Branch `master`, HEAD and origin both at:

```text
fd4b014 Archive and verify the prepared R1 remote workload
9b45323 Capture source checkout identity before generating R1 bundle
3707df2 Reset Paper 3 claims and prepare traceable R1 remote experiments
```

There is no PR. GitHub issues are disabled. **Nothing from the CPU systems implementation is committed.** Current tree:

```text
27 modified tracked files
28 untracked entries
tracked diff: +611 / -293 lines
```

### PROVEN against the current working source

Commands run after the latest source edits:

```text
uv run pytest -q \
  tests/test_wire.py tests/test_distributed_split.py tests/test_artifacts.py \
  tests/test_r1_panels.py tests/test_r1_attacks.py tests/test_r1_packets.py -x
=> 50 passed in 65.76s

uv run pytest -q tests/test_r1.py tests/test_manifest_resume.py -x
=> 64 passed in 145.09s
```

The second gate executes the real 14-run two-process local validation once, verifies relocation, and exercises provenance/resume behavior. Also verified:

```text
.venv/bin/python -m compileall -q src scripts tests  => passed
git diff --check                                   => passed
```

Current source inventory:

```text
29ad046e536c595be89a83ddcdc73fa5684c0a84cc68745e5849cbee52984e25
```

### RETAINED EVIDENCE, BUT FOR AN EARLIER WORKING TREE

`evidence/r1-cpu-validation-20260913/` is internally verified:

- 554/554 files; archive SHA-256 `025b32e78c9423a774e00f4e0c43eed77a2a39e8a09798b58ae3888cd0a7d50a`.
- 14/14 runs, 24 committed remote steps, 48 draft attack results.
- 312 tests + 12 subtests, 71 targeted tests, 57 container tests.

But it binds source inventory:

```text
e2307616a302269a7904185936ee1f7ddc1fb942688c3430030f56fee1090afc
```

That differs from current source. Do not use the retained package as validation of the latest hardening changes. The current 50+64 targeted tests are valid; the old full-suite/container/evidence claims are not current-tip proof.

### NOT DONE / NOT PROVEN

- Full `uv run pytest -q` has not been rerun against current source.
- `tests/test_r1_cpu_e2e.py` has not been rerun after the final hardening changes, although `tests/test_r1.py` exercised the same 14-run function.
- The CPU Docker image has not been rebuilt/retested against current source.
- No new retained CPU output/evidence package binds current source inventory.
- The updated `execute_engineering()` distributed GPU orchestration is not runnable locally because CUDA admission correctly blocks CPU fallback.
- No GPU, second host, provider lifecycle, scientific calibration, exploration or confirmation has run.
- Active claim ledger remains empty and must remain so.

## 3. Guardrails and security caveats

- Never claim completion without current-source tests and retained evidence.
- Never weaken tests or substitute mock artifacts. Historical data and results remain inadmissible for R1 claims.
- GPU/cloud spend is prohibited until the user explicitly approves the budget after the clean CPU handoff.
- `wire.secret` is generated per local service, mode `0600`, and deleted after service completion. Never commit or print it.
- The service binds loopback only; future cross-host use is through a protected tunnel. HMAC authenticates frames but is not transport encryption.
- The local subprocess runs from a verified corpus-free source packet, but same-host testing is not an OS security sandbox. Actual host separation is still untested.
- A lost/uncertain optimizer-step acknowledgement fail-stops the client. Recovery is only from a coordinated client/remote checkpoint; there is no transparent mid-step continuation.
- Do not commit/push unless the user asks. If asked, branch first from `master`; never use `git add -A`; no generated-by trailer.

## 4. What is next

**Goal:** produce a reviewed, current-source CPU evidence package and an honest GPU authorization request. Exit means every current source byte is bound to a passing full suite, real retained 14-run validation, current container result, fresh archive, updated docs and zero active scientific claims.

1. Review `git diff` and every untracked implementation file for correctness and scope.
2. Run current dedicated e2e:
   ```bash
   uv run pytest -q tests/test_r1_cpu_e2e.py -x
   ```
3. Run the complete current suite without editing source during execution:
   ```bash
   uv sync --frozen --all-groups
   uv run pytest -q
   ```
4. Build and run `environment/r1/Dockerfile.cpu` against current source. Record image ID and exact test output; do not claim a registry digest.
5. Run a final retained local validation with a new immutable execution ID and no concurrent edits.
6. Verify relocation, all manifests, semantic artifacts, panel/attack bindings, checkpoint pairs, step journals, remote-packet isolation and client/remote wire totals.
7. Rebuild/package `outputs/r1-cpu-validation-<date>/` into a fresh `evidence/r1-cpu-validation-<date>/` archive and verify a clean extraction.
8. Update `docs/GPU_AUTHORIZATION_HANDOFF.md`, active R1 docs, manuscript and evidence policy only from the new result.
9. Inspect generated files. `paper/SHA256SUMS` is currently an empty untracked file and should not be silently included; determine its origin and remove it only after confirming it is disposable.
10. Present the diff and evidence to the user. Commit/push only if requested. Then wait for explicit GPU authorization.

### Exit criteria before GPU discussion

- Full suite passes on current source.
- Real 14-run CPU validation completes with zero failures/interruption/missing runs.
- Remote packet contains no corpus, panels, checkpoints, outputs or trusted model state.
- Current source inventory and patch are identical across all retained runs.
- Current Docker systems gate passes.
- Fresh evidence archive restores and verifies completely.
- GPU handoff names only engineering budgets; no scientific stage is authorized.

## 5. Lessons learned

- Source provenance is intentionally fail-closed. Editing any source/config/protocol file while a validation run is active causes manifest completion to fail with `source_mismatch`; wait for the run before editing.
- Running the remote process directly from the repository allowed corpus visibility and generated `__pycache__`. The current code prepares/copies a corpus-free packet and sets `PYTHONDONTWRITEBYTECODE=1`.
- Attack events must bind the logical `panel_sha256`, not the JSON file’s byte hash. Confusing the two caused an e2e failure.
- Panel verification must recompute deterministic window selection; checking only non-overlap and hashes permits post-selection.
- Semantic validation records originally stored absolute paths and broke relocation. Current code stores output-relative paths and maps manifest artifacts during replay.
- A remote `STEP` can succeed before the trusted update. The chosen behavior is fail-stop plus durable remote prepare/commit records and recovery only from coordinated checkpoints—not false per-step atomicity.
- Pytest cleanup emits unrelated warnings from root-owned garbage directories under `/tmp/user/1000/pytest-of-epappas`. Use a unique `--basetemp` when needed; do not treat those warnings as project failures.

## 6. Decisions made — do not relitigate

- **Systems-paper path:** physical client/server separation is part of the third corner. Honored.
- **Useful distribution:** require both gain over frozen and gain over removal; nominal placement is insufficient. Honored.
- **Raw-data ownership:** corpus and trusted model state remain client-side; remote packet is allow-listed and corpus-free. Honored.
- **Historical reset:** pre-reset observations remain audit material, never R1 evidence. Honored.
- **CPU first, GPU later:** complete/review/seal every CPU-capable task before requesting spend. Honored.
- **Fail-stop uncertainty:** an uncertain distributed step invalidates the attempt; resume only from coordinated checkpoints. Chosen over pretending per-step cross-host atomicity.
- **Three-valued privacy:** missing/unqualified evidence is `UNDETERMINED`, never a pass. Honored.
- **No scientific execution:** calibration, exploration, confirmation and claims remain blocked. Honored.

## 7. Known debt and cut corners

- **Current retained CPU evidence is stale relative to current source. Regenerate it before any “CPU complete” claim or GPU request.**
- **Current full suite and CPU container are unverified after the latest hardening.**
- The current tree is large and uncommitted; review it before adding more scope.
- The remote service is process-separated and packet-isolated locally, but not yet validated on two physical hosts.
- `execute_engineering()` now routes through the distributed cell runner, but the exact CUDA orchestration cannot be exercised without the later GPU admission.
- Per-step fail-stop is safe for evidence admission but sacrifices availability; durable transparent reconnect/recovery is deliberately not implemented.
- Existing docs and `GPU_AUTHORIZATION_HANDOFF.md` describe the earlier CPU evidence and may overstate current-tip validation until regenerated.

## 8. Validation and rollout

Fast edit loop:

```bash
uv run pytest -q tests/test_wire.py tests/test_distributed_split.py \
  tests/test_artifacts.py tests/test_r1_panels.py tests/test_r1_attacks.py \
  tests/test_r1_packets.py -x
```

R1/provenance gate:

```bash
uv run pytest -q tests/test_r1.py tests/test_manifest_resume.py -x
```

Final gates:

```bash
uv sync --frozen --all-groups
uv run pytest -q
docker build -f environment/r1/Dockerfile.cpu -t paper3-r1-cpu:validation .
docker run --rm paper3-r1-cpu:validation
cd paper && ./build.sh
```

Use unique temporary directories. Final retained validation must run from unchanged source. Inspect `git status`, `git diff --check`, source inventory hashes, generated manifests and archive verification before reporting done.

## 9. Dev ops and contribution

- Environment: `.venv`, managed by `uv`; `torch==2.14.0+cpu` currently installed.
- Lock: `uv.lock` is untracked and part of the intended deliverable.
- Docker daemon was available earlier; current image must be rebuilt after the latest changes.
- Historical private release `evidence-2026-09-10-v1` is complete but unrelated to R1 claim admission.
- No PR exists. Work is directly on dirty `master`; do not discard or reset.
- Preserve `docs/status/`, historical output trees/tags, and the prior CPU evidence until the replacement archive is verified.

## 10. North star

Deliver a defensible systems paper that tests whether privacy, model quality and causally useful physical distribution coexist. The remaining arc is:

```text
current CPU hardening
  -> current-source CPU evidence seal
  -> user-approved 14-run GPU micro gate
  -> separately approved 4-run performance calibration
  -> implement/freeze scientific packet
  -> 76-run attack calibration
  -> 232-run exploration
  -> held-out confirmation
  -> claim ledger, manuscript, independent audit and submission
```

The next session must stop at the CPU evidence/GPU authorization boundary. A clean CPU stack authorizes asking the user for a budget; it does not authorize spending it.
