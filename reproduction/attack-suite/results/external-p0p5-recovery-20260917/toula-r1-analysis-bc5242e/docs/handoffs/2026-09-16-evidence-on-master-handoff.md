# Handoff — Paper 3 R1: CPU stack sealed and merged to master; the scientific stack is next

You are taking over the systems-paper implementation of Paper 3 R1 in
`/home/epappas/workspace/spacejar/paper-3-privacy-constrained-remote`. The goal is to test
the privacy–quality–useful-distribution trilemma through a **real trusted-client /
remote-service process boundary**, not an in-process model cut. **The CPU systems stack is
complete, sealed, committed, and merged to `master` as of 2026-09-16 — verified against the
server, not just locally.** Zero scientific runs have executed and zero claims are admitted;
that is by design, not by omission. Your job is **the scientific stack** (panel loader, fresh
attack qualification, held-out admission, frozen protocol and thresholds) — it needs no GPU
and it gates all 308 scientific runs. Do **not** provision a GPU, approve spend, freeze
scientific stages, or admit claims. The research decision, the design and the plan are already
written — read them, do not re-derive them.

## 0. How to work

- **Work from evidence, and date every number.** Say which figures you measured this session
  and which you are citing from a retained log. A number without a provenance date is a
  liability in this repo.
- Source provenance is **fail-closed**: any provenance-bound file edited while
  `run_local_validation()` runs, or between a run and its replay, makes completion and replay
  raise `source_mismatch`. This is a feature — it has fired correctly before.
- Keep heavy jobs serialized. This is a shared, loaded box; pytest has been OOM-killed at
  ~46% with zero test failures, which is environmental. Set
  `OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2` and use a unique
  `--basetemp` inside the repo (`.validate-tmp/`, disposable, untracked, **not** bound).
- Use subagents for breadth; keep your own context for the gates and the judgement calls.

## 1. Read these first, in order

1. `AGENTS.md` — the R1 reset of 2026-09-10 and the eight standing rules. **This governs
   everything.** Pre-reset observations are inadmissible, not disproven.
2. `/home/epappas/.claude/plans/fizzy-whistling-engelbart.md` — the approved CPU systems plan
   and the GPU stop boundary. Still governing.
3. `docs/GPU_AUTHORIZATION_HANDOFF.md` — the document the owner will read to decide GPU spend.
   Two of its four blockers are cleared; two are open.
4. `docs/status/2026-09-13-2150-status.md` — the most recent status, including four corrections
   to the 2026-09-12 status and one deliberately-unfixed defect.
5. `outputs/r1-cpu-validation-20260913-v2/validation_summary.json` — the single JSON
   aggregating every gate result and binding digest of the seal.
6. `docs/PREREGISTRATION_R1.md`, `docs/R1_RESEARCH_PLAN.md` — draft protocol and the six-stage,
   308-scientific-run design.
7. `docs/CLAIM_EVIDENCE_LEDGER.md` — every manuscript number must resolve through this.
8. Core systems code: `src/pcr/wire.py`, `remote_service.py`, `distributed_split.py`,
   `local_remote.py`, `r1_packets.py`, `r1_panels.py`, `r1_attacks.py`, `artifacts.py`,
   `r1_local_validation.py`, `provenance.py`.
9. `docs/handoffs/2026-09-13-cpu-sealed-handoff.md` — the prior handoff. Point-in-time; its
   "nothing is committed" statements are now false and it says so in a banner.
10. `~/.claude/CLAUDE.md` — the owner's hard rules (evidence, no stubs, commit hygiene).

## 2. Where you're starting / what's delivered

`master` and `r1-cpu-systems-seal` both at `836743e`, pushed, **confirmed from the server via
`git ls-remote` matching local `rev-parse`** — not from the push's own output. Working tree
clean, in sync. Base before this effort was `fd4b014`.

```
836743e Correct stale evidence READMEs and repair their checksum manifests
f19e384 Correct documentation overtaken by the CPU seal commit
19ab841 Retain the prepared R1 workload bundle as preparation evidence
5f9fcb0 Retain the sealed CPU validation evidence packages
1b530cf Pin the CPU validation container and lock the environment
7eae349 Document the CPU systems stack without creating a claim
a208648 Test the process boundary against real execution instead of mocks
7280a85 Retire same-identity retry loops in the legacy drivers
14ae275 Add the permanent CPU local-validation stage and evidence packager
100bf8d Add scientific panels, draft attacks and semantic artifact validation
6ebc2bf Add the trusted-client and corpus-free remote process boundary
```

914 files changed, +159,562 / −297.

**PROVEN — re-derived on 2026-09-16 (this session):**

- Source inventory seal **`9e39b3db26b21a2e975d8159c0f6e6f62faca8449060fff2ad227afb36f8b2a0`
  over 97 bound files**, computed through the canonical
  `sha256_json(source_inventory(Path('.')))`. Unchanged across the entire commit and merge.
- All four evidence packages self-verify: **17 files, zero checksum warnings.**
- Server refs match local exactly.

**PROVEN — measured 2026-09-13, retained as evidence, NOT re-run since:**

- Full suite **312 passed + 12 subtests** in 277.50s (`full-test-suite.log`, exit 0).
- R1 targeted battery **57 passed** in 104.09s (`r1-targeted-tests.log`, exit 0).
- Pinned CPU container battery **57 passed** in 260.60s (`container-validation.log`, exit 0),
  local image `sha256:b635a440496d…`; no registry digest exists.
- 14/14 retained CPU runs completed, execution-id `cpu-seal-20260913`, zero failures.
- Remote packet contains no corpus/panels/checkpoints/trusted state (digest `12c662e15f41…`).
- Client and remote wire totals match on every split run (2,995,027 bytes total).
- Manuscript: 6 pages, zero LaTeX warnings.

**DEFERRED — your job. Not run. Do not claim any of this works:**

- The entire scientific stack: panel loader, fresh target/channel attack qualification,
  held-out admission, prospectively accepted protocol and thresholds.
- All 308 scientific runs. Zero executed.
- Any GPU execution. The CUDA lock has never been installed or run anywhere.
- Genuine cross-host operation. Everything so far is same-host loopback.

**Load-bearing invariants — do not break:**

- The seal must keep recomputing to `9e39b3db…`. If it changes, something bound was edited.
- `evidence/**` is **unbound** except `evidence/R1_EVIDENCE_POLICY.json`. Never edit that file
  casually — it is the one evidence path that moves the seal.
- Engineering runs carry `scientific_evidence: false` and are **never** promotable.
- Admissions are consumable: one per launch, and a failure burns it.

## 3. Guardrails & security caveats

- **Never claim something works without end-to-end validation and retained evidence.** Never
  weaken a test, never substitute a mock artifact, no stubs or placeholders.
- **No GPU or cloud spend until the owner explicitly approves a budget**, and only after a
  reviewed clean commit and a freshly prepared bundle. A sealed CPU stack authorizes *asking*,
  not spending.
- The owner owns **two NVIDIA GB10 Spark machines** intended for this work instead of metered
  cloud. As of 2026-09-13 they were busy and **no go-ahead has been given.** Hold.
- `wire.secret`: generated per service, mode 0600, deleted on completion. **Never commit or
  print it.** The service binds loopback only. HMAC authenticates frames; it is **not**
  transport encryption — any cross-host use needs a protected tunnel.
- The remote subprocess runs from a verified corpus-free packet, but same-host testing is
  **not** an OS security sandbox. Do not describe it as one.
- Git: never `git add -A`; stage explicit paths. Branch first if asked to commit on `master`.
  No "Generated with Claude Code" trailer. Do not commit or push unless asked.
- Preserve `docs/status/`, historical output trees and tags, and both CPU evidence packages.
- `.validate-tmp/` is untracked and not provenance-bound. Never commit it.

## 4. What's next: the build

**Goal:** the scientific stack. **Exit criteria:** a panel loader that serves declared complete
transcripts; fresh attack qualification with matched nulls and calibrated positive controls;
held-out admission wired to the three-valued PASS / REJECTED / UNDETERMINED outcome; and a
protocol + thresholds + seed list frozen *before* any confirmatory execution.

**Why this and not the GPU lane:** the scientific stack needs no GPU and gates all 308
scientific runs. The GPU lane, fully unblocked, buys only the 18 engineering runs of Stages A
and B. The scientific stack is the long pole — start there.

**Watch out:** a missing or unqualified required privacy view is **UNDETERMINED**, never an
automatic pass (`AGENTS.md`). Do not optimize reporting toward an empty feasible set.

**Defer:** GPU provisioning; the budget rewrite; freezing any scientific stage.

## 5. Lessons learned

- **Compute the seal only via `sha256_json(source_inventory(root))` from `src/pcr/provenance.py`.**
  A hand-rolled hash loop produces a different digest and looks exactly like a seal break. This
  cost this session a false alarm. The ad-hoc script did not raise, so the correct fallback
  never ran.
- **`provenance.py` needs the project venv.** It imports `utils.py`, which imports `torch` at
  module load. Bare `python3` fails with `ModuleNotFoundError`. Use `.venv/bin/python`.
- **Editing an evidence package's `README.md` breaks its `SHA256SUMS`.** These packages
  instruct reviewers to run exactly that check. If you edit one, repair the manifest's README
  line in the same commit — replace only that line, so the archive / `index.json` /
  `verification.json` digests are *provably* untouched rather than merely re-asserted.
- **The aarch64 wheel tag is `manylinux_2_28_aarch64`,** not `linux_aarch64`. Grepping the
  wrong one returns zero hits and reads as "no ARM support".
- **Three different tool failures look alike and need opposite responses:** a classifier
  *timeout* ("temporarily unavailable") is infrastructure — retry, and read-only tools still
  work; a *stage-2 denial* says retrying often succeeds — retry a few times; a genuine *user*
  denial is a boundary — do not retry or reroute.
- **Keep `git push` as a bare command.** A compound `checkout && merge --ff-only && push` was
  refused, while the identical merge with the push removed went straight through.
- **`.gitignore` admits only `.json/.jsonl/.md/.txt/.py/.csv/.log/.png` under `outputs/`.** The
  82 `.pt` tensor payloads live only inside the evidence archive — which was verified to
  contain exactly the 82 that exist on disk. `data/raw/` is untracked; the corpus survives
  inside both preparation archives.

## 6. Decisions made (ADRs) — don't relitigate

- **Real two-process boundary** over an in-process model cut, because the paper's claim is
  about a process boundary. Honored.
- **Three-valued privacy outcomes** (PASS / REJECTED / UNDETERMINED). Missing views are never
  automatic passes. Honored.
- **Engineering/scientific one-way door.** Engineering runs can never be promoted to
  scientific evidence. Honored.
- **R1 reset (2026-09-10).** Pre-reset results are inadmissible because their exact execution
  source is unrecoverable — *not* because they were shown false. Post-hoc recalculation does
  not restore admissibility. Honored — don't reopen.
- **`experiments/configs/r1_exploration.yaml:261-262` left broken deliberately.** It declares
  `outputs_root: outputs/r1` and `manifests_root: experiments/manifests/r1`, neither of which
  exists. That file **is** provenance-bound, so fixing it re-seals the whole stack. Deferred to
  the scientific freeze, when a re-seal is expected anyway.
- **No manifests generated for the pre-reset evidence directories.** Minting fresh provenance
  records for inadmissible material is worse than leaving them plainly unverifiable.

## 7. Cut corners / known debt

- **HEADLINE: `two_host_validated: false`. Every result so far is same-host loopback.** The
  two Sparks make this genuinely testable for the first time, but the wire is loopback-only by
  design and HMAC does not encrypt — a real cross-host split needs a protected tunnel. This is
  unbuilt.
- **`experiments/configs/r1_exploration.yaml:261-262` points at two directories that do not
  exist.** Bound file; see §6. Fix it at the scientific freeze, not before.
- **`evidence/audit-20260910/` and `evidence/source-search-20260910/` have no checksum manifest
  at all.** They predate the convention and were already on `master`. They are the only
  evidence directories that cannot be verified. Owner's call: withdraw, annotate, or leave.
- `outputs/r1-cpu-validation-20260913-v2/validation_summary.json` still records
  `working_tree_clean: false` and `git_commit: fd4b014`. That was accurate at capture and is
  now historical metadata only; the binding itself is intact.
- The CUDA environment (`environment/r1/cuda/uv.lock`) is **arm64-complete by a static read of
  wheel tags only** — `torch 2.14.0+cu130` ships cp312 `manylinux_2_28_aarch64`, and none of
  the 52 locked packages lacks aarch64 coverage. This does **not** establish that the CUDA 13
  build runs on Blackwell. Nothing has installed or executed it.
- The container image has a local digest but **no registry digest**, so it is not yet an
  immutable `IMAGE@sha256:DIGEST` of the kind the GPU checklist demands.

## 8. Validation & rollout

Gates, all runnable on CPU:

```bash
# seal — must print 97 files and 9e39b3db26b21a2e975d8159c0f6e6f62faca8449060fff2ad227afb36f8b2a0
.venv/bin/python -c "from pathlib import Path; from src.pcr.provenance import source_inventory, sha256_json; \
inv=source_inventory(Path('.')); print(len(inv), sha256_json(inv))"

# every evidence package self-verifies
for d in evidence/*/; do [ -f "$d/SHA256SUMS" ] && (cd "$d" && sha256sum -c SHA256SUMS); done

# test gates (serialize these; see §0)
OMP_NUM_THREADS=2 .venv/bin/python -m pytest --basetemp=.validate-tmp
```

After any push, verify from the server with `git ls-remote --heads origin` against local
`git rev-parse` rather than trusting the push output.

"Done" for a scientific stage requires more than green tests: a bound chain from protocol,
source, environment and corpus through execution, raw artifacts, qualification, analysis and
claim. Anything short of that is engineering evidence.

## 9. Dev ops & contribution

- Work in `/home/epappas/workspace/spacejar/paper-3-privacy-constrained-remote`. Interpreter is
  `.venv/bin/python` — required for anything importing `src/pcr/`.
- Evidence packages are self-verifying by construction: `index.json` binds path/size/mode/
  sha256, `verification.json` records a fresh-extraction check, `SHA256SUMS` covers the
  package's own files.
- Traps: pytest OOM on this shared host (environmental, not a failure); `.validate-tmp/` must
  stay untracked; the evidence packager deliberately refuses to overwrite an existing output
  tree, so seal steps are exclusive by construction.
- Commits: explicit paths only, conventional subject line, no generated-by trailer.

## 10. North star / roadmap

The paper asks whether privacy, quality and useful remote work can hold simultaneously across a
real process boundary — as a **falsifiable finite-family hypothesis**, accepting
counterexamples, failures and undetermined outcomes. The manuscript currently states plainly:
*"This document contains no admitted empirical results"* and closes by calling itself a research
protocol with no empirical conclusion about the trilemma. Keep it that way until evidence earns
otherwise.

Arc: **CPU systems seal (done)** → scientific stack (your job) → Stage A/B engineering GPU runs
(18 runs, separately admitted) → 76 calibration runs → 232 exploration runs → confirmation.

Two strategic checkpoints to make explicit before spending anything:

- The ledger itself records that a **universal** trilemma is not establishable by this design.
  Know what the strongest honest claim actually is before optimizing toward it.
- **Segment-property prediction — one of the three novelty legs — is deferred out of R1
  entirely**, and its "held-out failure" support was a pre-reset result that is now
  inadmissible. If that leg matters to the paper's contribution, decide deliberately when it
  comes back, rather than discovering the gap at writing time.
