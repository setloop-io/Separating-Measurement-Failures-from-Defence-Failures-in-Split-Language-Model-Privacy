# Handoff — Paper 3 R1 CPU stack: sealed against final source; commit review and GPU authorization next

> **Point-in-time record, partly overtaken 2026-09-13.** Jobs (1) and (2) below are
> done: the stack was reviewed and committed as 8 commits `6ebc2bf..5f9fcb0` on branch
> `r1-cpu-systems-seal`, pushed to origin, and a fresh bundle was prepared from the
> clean commit, verified from a disposable extraction and retained at
> `evidence/r1-preparation-20260913/`. Statements below that nothing is committed were
> true when written and are now false. Job (3), the GPU authorization boundary, still
> stands. The body is preserved unedited as the record of that session.

You are taking over the systems-paper implementation of Paper 3 R1 in
`/home/epappas/workspace/spacejar/paper-3-privacy-constrained-remote`. The goal is to
test the privacy–quality–useful-distribution triangle through a real
trusted-client/remote-service boundary, not merely an in-process model cut. **The CPU
seal described in the previous handoff (`2026-09-13-systems-cpu-stack-handoff.md`, now
superseded) is COMPLETE as of this session: the retained 14-run validation, every test
gate, the container and the evidence archive all bind one identical final source tree.**
Nothing is committed. Your job is: (1) review/present the diff and evidence to the
user, (2) commit/push ONLY if the user asks, (3) stop at the GPU authorization
boundary and wait for the user's explicit budget decision. Read the files below; do
not re-derive the research decision.

## 0. How to work

- Work from evidence. Every number in this handoff was re-measured this session; the
  evidence package is self-verifying (`index.json` + `verification.json` + `SHA256SUMS`).
- Source provenance is fail-closed: any provenance-bound file edited while
  `run_local_validation()` runs, or between the run and its replay, makes completion
  and replay verification raise `source_mismatch`. This was proven live this session:
  a packaging dry-run correctly REFUSED to repackage the older evidence tree because
  the current source had moved.
- The evidence packager refuses to write over existing outputs (`validation_summary.json`,
  evidence dir, copied logs) — seal steps are exclusive by construction.
- Keep heavy jobs serialized on this host: it is a shared, loaded box (two pytest
  attempts were OOM-killed at ~46% with no test failures — that is environmental, not
  code). Set `OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2`
  and use a unique `--basetemp` inside the repo (`.validate-tmp/`, disposable,
  untracked, NOT provenance-bound).
- Do not provision a GPU, approve spend, freeze scientific stages, or admit claims.

## 1. Read these first, in order

1. `/home/epappas/.claude/plans/fizzy-whistling-engelbart.md` — approved CPU systems
   plan and GPU stop boundary (unchanged, still governing).
2. `AGENTS.md` — R1 reset, zero active claims, historical evidence restrictions.
3. `docs/GPU_AUTHORIZATION_HANDOFF.md` — REWRITTEN this session from sealed numbers;
   this is the document the user will read to decide GPU spend.
4. `outputs/r1-cpu-validation-20260913-v2/validation_summary.json` — the single JSON
   that aggregates every gate result and binding digest of the seal.
5. `docs/PREREGISTRATION_R1.md`, `docs/R1_RESEARCH_PLAN.md` — draft protocol and 19-cell
   design (reviewed; unchanged this session).
6. Core systems code (reviewed, unchanged this session except item 7 below):
   `src/pcr/wire.py`, `src/pcr/remote_service.py`, `src/pcr/distributed_split.py`,
   `src/pcr/local_remote.py`, `src/pcr/r1_packets.py`, `src/pcr/r1_panels.py`,
   `src/pcr/r1_attacks.py`, `src/pcr/artifacts.py`, `src/pcr/r1_local_validation.py`.
7. Changed by this session (review these first in any diff pass):
   - `src/pcr/remote_service.py` — readiness-file race fix (see §5).
   - `tests/test_distributed_split.py` — startup-timeout parameter fix (see §5).
   - `scripts/package_cpu_evidence.py` — NEW: replay-verify + package + extract-check
     the retained tree into an evidence archive (bound source; do not edit without
     re-sealing).
   - `environment/r1/Dockerfile.cpu` + `environment/r1/README.md` — container CMD
     scoped to the git-free systems battery (see §5).
   - `docs/GPU_AUTHORIZATION_HANDOFF.md` — regenerated from the seal.
8. Tests: `tests/test_wire.py`, `test_distributed_split.py`, `test_r1_panels.py`,
   `test_r1_attacks.py`, `test_artifacts.py`, `test_r1_packets.py`,
   `test_r1_cpu_e2e.py`, `test_r1.py`, `test_manifest_resume.py`.

## 2. Where you are starting

Branch `master`, HEAD = origin = `fd4b014`; no PR; GitHub issues disabled.
**The whole CPU systems implementation remains uncommitted** on a dirty tree:

```text
27 modified tracked files (+615 / -293)
32 untracked entries (all implementation, tests, config, docs, evidence)
```

### PROVEN against the FINAL working tree — measured this session

Final source inventory (97 provenance-bound files), identical across every retained run:

```text
source inventory      9e39b3db26b21a2e975d8159c0f6e6f62faca8449060fff2ad227afb36f8b2a0
working-tree patch    7dbc0aa7e9d5a36d77902928e4561eda1963cae285a54d5c50b48736d90ed1af (over fd4b014)
environment digest    ddd933efa5d4181090b7c52e76061758177eeb3ca91750e0568c53154c1af688
remote packet digest  12c662e15f4145734a9e1b00a8cf38fedb48a09e1b2fe4993e5f67563a725d19 (single value, all runs)
```

- Full locked suite: **312 passed + 12 subtests, 0 failed** (277.5s).
- R1 systems battery (7 files incl. the real 14-run e2e): **57 passed** (104.1s).
- Dedicated e2e: `test_r1_cpu_e2e.py` ran standalone once (7 passed, 172.9s, pre-seal
  inventory) and inside every targeted battery since (57 passed at each inventory
  revision, final battery 104.1s).
- Retained 14-run validation `execution-id cpu-seal-20260913`:
  14/14 completed, 0 failures/interrupted/not_started, 14 manifests, 14 semantic
  records, 24 committed remote steps, 48 draft attack results (4 reports x 12),
  client/remote wire totals match every run (total 2,995,027 B; per-run 249,148–249,978),
  run wall-time total 40.1s. Verified independently: all 14 manifests' bound inventory
  == current tree digest (recomputed this session).
- Pre-package relocated replay of all 14 runs: 14/14, zero anomalies.
- CPU container image `sha256:b635a440496dc832533c203c4ae660ed5df4792195edb12d57129aaf208378ec`
  (1,992,572,703 B; built from the final tree): systems battery **57 passed** (260.6s).
  No registry digest exists — local image ID only.
- Evidence package `evidence/r1-cpu-validation-20260913-v2/`:
  528 files, 33,707,654 extracted B, 24,286,417 compressed B, archive SHA-256
  `7c76754f19b8479a2536c8cf2b362a28769f8ce6f7f004aab1d9fb0ab3e60a59`;
  fresh-extraction verify 528/528; independent `sha256sum -c SHA256SUMS` 4/4 OK;
  retained logs inside: `full-test-suite.log`, `r1-targeted-tests.log`,
  `container-validation.log`.
- Manuscript rebuilt from current tree: 6 pages, 0 LaTeX warnings (main.pdf re-modified
  in the working tree — expected).
- Hygiene: `compileall src scripts tests` OK; `git diff --check` OK; stray empty
  `paper/SHA256SUMS` investigated (0-byte file created 03:43 after the 03:05 paper
  build; no repo script writes that path; `paper/` is outside provenance patterns and
  outside the tracked hash index) and removed; inventory proven unchanged by the
  removal (recomputed identical afterward).
- Zero active claims: `docs/CLAIM_EVIDENCE_LEDGER.md` admits nothing; every artifact
  carries `scientific_evidence: false`; attack status is `ENGINEERING_ONLY`.

### Exit criteria from the previous handoff — all met (evidence above)

Full suite on current source; real 14-run validation with zero gaps; corpus-free
remote packet (single digest, allow-listed); one inventory/patch across all runs;
container gate green on final tree; fresh archive restores and verifies; GPU handoff
names engineering budgets only.

### NOT DONE / NOT PROVEN (do not claim)

- Nothing is committed or pushed. The user has not yet reviewed this diff.
- No GPU, second physical host, provider lifecycle, paid execution.
- `execute_engineering()` distributed CUDA orchestration unexercised (CUDA admission
  correctly blocks local CPU fallback).
- Transparent mid-step reconnect/recovery deliberately not implemented (fail-stop chosen).
- Scientific stages (calibration 76, exploration 232, confirmation) remain blocked;
  thresholds are draft values, unfrozen.
- Superseded package `evidence/r1-cpu-validation-20260913/` (inventory
  `e2307616…`) is audit-only history; keep it, never cite it as current-tip proof.

## 3. Guardrails and security caveats

- Never claim completion without current-source tests and retained evidence. Never
  weaken tests or substitute mock artifacts. No fake/stub/placeholder code.
- GPU/cloud spend prohibited until the user explicitly approves the budget AFTER a
  reviewed clean commit and freshly prepared bundle. A sealed CPU stack authorizes
  ASKING, not spending.
- `wire.secret`: generated per service, 0600, deleted on completion; never commit or
  print. Service binds loopback only; HMAC authenticates frames but is NOT transport
  encryption; cross-host use must go through a protected tunnel.
- The remote subprocess runs from a verified corpus-free packet, but same-host testing
  is not an OS security sandbox.
- Uncertain optimizer-step acknowledgement fail-stops the client; recovery only from a
  coordinated client/remote checkpoint pair.
- Commit hygiene (user rules): never `git add -A`; branch first if asked to commit on
  master; no "Generated with Claude Code" trailer; commit message ends with
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`; no author name otherwise.
  Do not commit/push unless the user asks.
- Preserve: `docs/status/`, historical output trees/tags, both CPU evidence packages.
- `.validate-tmp/` in the repo root is disposable scratch from this session (logs +
  basetemps); delete or regenerate freely; it is untracked and not provenance-bound.
  Do not commit it.

## 4. What's next

**Goal:** present the sealed diff + evidence to the user; on request, commit; then
hold at the GPU authorization boundary.

1. Review with the user: `git diff` + the untracked list (this handoff §1.7 names what
   changed since the last review point). Suggested commit slices: (a) transport +
   service + client + packets + lifecycle, (b) panels/attacks/artifacts/validators,
   (c) local-validation stage + scripts + config, (d) tests, (e) docs/manuscript/
   evidence policy, (f) environment (Docker + cuda proposal + uv.lock/.dockerignore),
   (g) evidence packages (large; confirm the user wants binaries in git — prior
   practice committed `evidence/r1-preparation-20260910/`, and
   `evidence/r1-cpu-validation-20260913/` was left untracked).
2. Only after commit: prepare a fresh workload bundle from the clean commit and verify
   it in a disposable directory (`scripts/run_r1.py prepare` / `verify-bundle`) —
   already a stated authorization blocker in `docs/GPU_AUTHORIZATION_HANDOFF.md`.
3. Present the Stage A (14x8 micro, USD 2.00 cap) and Stage B (4x100 performance,
   USD 4.00 cap) budgets for the user's explicit decision. Enter provider, region,
   rate, image digest and external shutdown mechanism into the budget/environment
   review templates — do not fill approvals on the authors' behalf.
4. Do not run any GPU stage, scientific freeze, calibration, exploration, or ledger
   update without separate explicit authorization.

Exit for this phase: user has the reviewed commit + the sealed evidence chain + the
GPU handoff, and has made (or explicitly deferred) the budget decision.

## 5. Lessons learned (this session)

- **Readiness-file race (fixed):** the remote service wrote `ready.json` with a
  buffered `open("x")+write`, while the client polls `is_file()` then reads — the
  create is visible before content flush, so under host load the client once parsed an
  empty file (7 JSON-decode errors in a full-suite pass). Fix: write
  `ready.json.partial`, `flush` + `os.fsync`, then `os.replace` (atomic). All
  create-then-poll file handoffs in this codebase must follow this pattern.
- **Container cold import (fixed):** `import pcr.remote_service` (pulls torch) takes
  ~8.0s on the container's fresh overlay fs; `test_partial_client_is_terminated_by_server_read_timeout`
  capped service startup at 5.0s and failed there (passed on host). Fix: that test now
  uses the production 30s startup default; the 0.2s READ timeout being asserted is
  unchanged.
- **Container scope (fixed):** `Dockerfile.cpu` CMD had been widened to all `tests/`,
  but the image excludes `.git` while `prepare_bundle` calls git unconditionally — 13
  errors that were structural, not flaky. CMD restored to the 7-file git-free systems
  battery (wire/distributed/artifacts/panels/attacks/packets/cpu_e2e = the 57-test
  gate the prior package used); git-bound orchestration tests belong to the host full
  suite. `environment/r1/README.md` now states this split.
- **Seal discipline:** the packager itself is provenance-bound (`scripts/**/*.py`).
  Any fix to it moves the inventory and invalidates the seal — budget for the full
  ~35-min chain rerun (suite 278s + battery 104s + container build/run ~300s +
  validation ~250s + package ~120s). The chain was run serially and green 3 times
  this session; the numbers above are from the final chain only.
- **Packager bugs found before they could lie:** (1) `package()` clobbered
  `summary["execution"]` (planned/completed/failure counts vanished) — now `update()`;
  (2) hard-coded image-ID flag would have recorded a stale digest after any rebuild —
  now the chain writes `container-image-spec.txt` post-build and the packager reads it.
  Both fixed before the final seal; `outputs/r1-cpu-validation-20260913-v2/validation_summary.json`
  carries the corrected shape.
- **Proven fail-closed:** a deliberate dry-run against a stale evidence tree was
  REFUSED by replay with "current source/config/protocol files differ from the
  retained snapshot" — the replay-before-package step is a real gate, not theater.
- OOM kills during earlier attempts were the shared box (other users at ~70GB), not
  the code; identical suites passed after waiting for memory. Check `free -g` before
  launching heavy runs.

## 6. Decisions made (carry forward, honored — do not relitigate)

All decisions from the previous handoff §6 stand (systems-paper path; useful
distribution needs gain vs frozen AND removed; raw-data ownership client-side;
historical reset; CPU-first; fail-stop over fake atomicity; three-valued privacy;
no scientific execution). Added this session:

- **Loopback container gate = systems battery only:** the container proves transport/
  panels/attacks/artifacts/e2e without git; the host proves the full suite including
  git-bound bundle/resume behavior. Chosen over shipping `.git` into the image or
  stubbing git. Honored.
- **Evidence packaging is a scripted, replay-first pipeline** (`scripts/package_cpu_evidence.py`):
  relocate/identity replay must pass, logs are copied into the tree, then archive +
  index + fresh-extraction byte check + SHA256SUMS. Manual prose summaries are not
  evidence. Honored.
- **The v1 CPU evidence package stays in the tree, uncommitted, as audit history**
  alongside v2; docs must cite v2 (final inventory) as the current seal. Honored.

## 7. Known debt and cut corners

- **Everything is uncommitted.** The seal binds a dirty tree over `fd4b014`; GPU
  admission additionally requires a clean commit + newly prepared + restored bundle.
  This is the critical path.
- `paper/main.pdf` and `paper/figures/README.md` changed as part of the doc pass; PDF
  diff is binary and was verified by rebuild + 0-warning log, not visual review.
- Per-run remote dirs retain full captures (~33.7 MB sealed tree); storage grows
  linearly with runs — relevant to Stage A/B output caps.
- `run_local_validation` seeds (65001/65002) are local-only engineering seeds,
  documented non-scientific; no reserved confirmation seed touched.
- The container lacks git; a future image that runs the FULL suite in CI must mount a
  checkout (open, not blocking).
- Legacy drivers only had their retry loops retired (`max_retries = 0`); they remain
  single-process paths kept for historical tooling.

## 8. Validation & rollout (commands, all proven green this session)

```bash
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
uv sync --frozen --all-groups
uv run pytest -q --basetemp=$PWD/.validate-tmp/full            # asserts 312+12 pass
uv run pytest -q tests/test_wire.py tests/test_distributed_split.py \
  tests/test_artifacts.py tests/test_r1_panels.py tests/test_r1_attacks.py \
  tests/test_r1_packets.py tests/test_r1_cpu_e2e.py --basetemp=$PWD/.validate-tmp/t  # 57
docker build -f environment/r1/Dockerfile.cpu -t paper3-r1-cpu:validation .
docker image inspect paper3-r1-cpu:validation --format '{{.Id}} {{.Size}}' > spec.txt
docker run --rm paper3-r1-cpu:validation                       # 57 in container
uv run python scripts/run_r1.py local-validate \
  --output $PWD/outputs/<new-tree> --execution-id <new-id>     # 14-run seal (frozen tree!)
uv run python scripts/package_cpu_evidence.py \
  --output-root outputs/<new-tree> --evidence evidence/<new-pkg> \
  --execution-id <new-id> --full-suite-log ... --targeted-log ... \
  --container-log ... --container-image-spec spec.txt \
  --paper-pages 6 --paper-latex-warnings 0                     # replays, refuses stale tree
cd paper && ./build.sh                                          # 6 pages, 0 warnings
```

Order matters: gates -> container -> retained validation -> packaging, on a frozen
source tree. Editing ANY provenance-bound file (`src/**/*.py`, `scripts/**/*.py`,
`experiments/configs/**`, `experiments/schemas/**`, `pyproject.toml`, `uv.lock`,
`docs/PREREGISTRATION*.md`, `docs/R1_*.md`, `environment/r1/**`,
`evidence/R1_EVIDENCE_POLICY.json`) invalidates the seal and the recorded container
image; anything outside those patterns (e.g. `docs/GPU_AUTHORIZATION_HANDOFF.md`,
`README.md`, `tests/**` for host gates) does not move the inventory — but `tests/**`
DOES change the container image content, so rebuild the image after test edits before
quoting its ID.

## 9. Dev ops

- `.venv` via uv; `uv.lock` (untracked, part of the deliverable) + pyproject pin
  torch 2.14.0+cpu index. Host CPU-only. Docker 29.3.1 present.
- Disk `/` has ~212 GB free (it hit 0 earlier this week from unrelated caches — check
  `df -h /` if a run dies weirdly).
- This box is shared and load is unpredictable; prefer unique in-repo basetemps and
  serialized runs; clean `.validate-tmp/*` basetemps after (~400 MB each).
- No PR workflow exists (issues disabled); user reviews diffs in chat.

## 10. North star

A defensible systems paper testing whether privacy, model quality and causally useful
physical distribution coexist. The CPU corner of the arc is done and sealed:

```text
current CPU hardening               DONE (this session: race/timeout/container fixes)
  -> current-source CPU evidence seal  DONE (inventory 9e39b3db, v2 package verified)
  -> user review + clean commit        NEXT (your step)
  -> fresh bundle from clean commit    then GPU ask
  -> user-approved 14-run GPU micro gate (Stage A, USD 2.00 cap)
  -> separately approved 4-run performance calibration (Stage B, USD 4.00 cap)
  -> implement/freeze scientific packet -> 76-run calibration -> 232-run exploration
  -> held-out confirmation -> claim ledger, manuscript, independent audit, submission
```

Stop at the GPU authorization boundary. Present the evidence, let the user decide.
