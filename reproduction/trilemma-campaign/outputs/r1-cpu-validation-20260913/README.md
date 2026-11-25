# R1 local CPU systems validation — 2026-09-13

Status: **verified engineering evidence only; not scientific evidence**.

This retained run exercised 14 real training/control executions over an authenticated
two-process loopback boundary. The trusted process owned corpus bytes, prefix/suffix,
loss and trusted optimizer. Each remote process ran from the same verified corpus-free
packet and owned only its surrogate, optimizer, transaction journal and remote captures.

Verified outcomes:

- 14/14 runs completed; no failed, interrupted or missing attempts.
- 14 prospective manifests and 14 semantic-validation records passed.
- 24 remote optimizer steps have durable prepare/commit pairs.
- Client and remote wire totals match for every split run.
- Four trained runs retained 48 draft target/channel attack results and their fitted banks.
- Relocating the complete output tree and replaying analysis preserved all 14 runs.
- Locked host suite: 312 tests plus 12 subtests passed.
- Pinned CPU container: 57 systems tests passed; local image ID is recorded in
  `validation_summary.json` but has no registry digest.
- The active six-page protocol manuscript builds without LaTeX warnings.

The numerical contrasts are tiny two-step engineering observations and must not be
used as utility, privacy, causal-learning or trilemma evidence. The attack thresholds
and qualification outputs are draft pipeline checks only.

The source checkout was intentionally uncommitted while this implementation was being
validated. Every run retains the exact source inventory and binary patch, but GPU
admission still requires a reviewed clean commit and a newly prepared bundle.

Primary files:

- `analysis.json` — engineering-only contrasts and completion state.
- `validation_summary.json` — aggregate verification and explicit boundaries.
- `panel_manifest.json` — deterministic real-corpus four-region panel.
- `manifests/` — 14 prospective manifests.
- `runs/` — real model states, captures, wire records, remote packets and checkpoints.
- `full-test-suite.log`, `r1-targeted-tests.log`, `container-validation.log` — retained checks.
