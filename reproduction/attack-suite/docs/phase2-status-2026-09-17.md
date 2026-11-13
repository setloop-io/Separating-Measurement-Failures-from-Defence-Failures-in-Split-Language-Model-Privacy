# Phase 2 status — 17 September 2026

Checked the execution hosts directly; collected all three P6 result trees.

**Scope correction after reviewing commit 8bb1452:** P6 audit: these 192 successful cells are a pilot-corpus, VMI-only Mode-2 grid. Frozen-protocol held-out utility controls/evaluations and gradient label-scoring are missing, and the deterministic order differs from the randomisation rule. Full confirmatory P6 evaluation is incomplete. See p6-driver-audit-8bb1452.md.

| Node | Span | Completed cells | Execution failures | Finished (UTC) |
|---|---:|---:|---:|---|
| soula | 2 | 64 | 0 | 17 Sep 02:35:35 |
| voula | 4 | 64 | 0 | 16 Sep 22:17:44 |
| sasa | 6 | 64 | 0 | 17 Sep 05:01:26 |

All **384 capture/attack evidence manifests** verify locally, including their
hash chains and artifact digests. Detailed results are in
`results/phase2-status-2026-09-17/p6-evidence-verification.json`.

Each span has 48 privacy-gate FAIL and 16 PASS outcomes. These are recorded
privacy outcomes, not process failures. D0 fails all 32 cells per span; D1
passes 16/32. A separately labelled descriptive analysis is now saved as
`results/phase2-status-2026-09-17/p6-pilot-analysis.json`, including seed
permutations and seed/window bootstrap checks. Original article identifiers
are absent, so these are not article-cluster confidence intervals. The missing
held-out endpoints cannot be recovered from this pilot. These Mode-2 results do not settle the
external Mode-1 empty-set claim. Paper conclusions have not been changed.

On koula, both the SplitLLM reference log and our BiSR attack log end with
exit code zero. The local artifact records 12 cells / 3,072 tokens, 99.0234%
main-arm recovery, and 99.9674% optimisation-arm recovery. The reference
aggregate records 13 batches and 98.6478% positional accuracy for BiSR(b+f).
Different samples and metrics prevent treating these as a fidelity pass.
No documented fidelity-tolerance verdict was found. The frozen v4 protocol
requires that verdict before the matched evaluation can proceed.

The BiSR JSON evidence and reference records have been collected in
`results/phase2-status-2026-09-17/`. The full 79 GB BiSR capture directory is
still on koula and is not part of this release snapshot. Completion of the
matched v4 comparison, Qwen3 grid, and agentic trials=2 rerun was not verified.
External P0–P5 records were found on toula/voula and recovered after searching
all five nodes. The owning project retired them from current claims on
10 September because execution-source provenance was missing. Of the recovered
records, 3,720 manifest references match with zero mismatches; 772 mostly binary
references were not downloaded. Current R1 calibration completed 76/76, but
exploration stopped after nine cells and no confirmation was found. See
`docs/external-p0p5-recovery-2026-09-17.md`. No original SensiReplaced was found;
the explicit BiSR verdict remains NOT_ESTABLISHED. Fleet deployment evidence
remains outstanding.

## Corrected prospective P6 follow-up

Commit `d5ec03eaa561241bf71564bdf08e7fb1501dde78` freezes the new protocol,
driver and analysis separately from the historical pilot. The full local suite
passed: 2,563 tests, 36 skips. A GPU canary passed in 28.37 seconds and is
excluded from the analysis. The full 192-cell grid plus eight all-local
controls completed on koula (former PID 487553) in
`~/llm-attacker-p6-v2-d5ec03e/results/p6-corrected-gpt2-v2`.
The launch record is `results/full-grid-launch.json` in that checkout.
The conservative projected duration was 3.28 GPU hours, below the 20-hour
launch budget. This is an estimate, not a completion claim.

A separate worker, PID 488834, waits for the complete-grid marker and then
verifies all cells and runs the frozen analysis. Its state is
`results/p6-corrected-gpt2-v2.analysis-status.json` and its log is
`results/p6-corrected-gpt2-v2.analysis.log`. Any failed cell blocks analysis.
Both completion markers are now present. All 200 cells returned zero; strict
verification and frozen analysis completed at 2026-09-17 13:43:09 UTC
(14:43:09 BST). Summed cell wall time was 1.369 hours. The analysis, launch
record, complete marker, grid manifest and full run journal are collected in
`results/p6-corrected-followup-20260917/`; bulk cell artifacts remain on koula.
The corrected evidence has not yet been published to HF.

The start-position test passes the prespecified criterion for D0 utility,
D0 normalised leakage and D1 utility. D1 leakage does not pass the corrected
significance threshold (p=.01930 versus alpha=.0125). Held-out-seed prediction
improves by at least 10% only for the two D0 endpoints. This is the prospective
Mode-2 follow-up; it does not validate Mode-1/P5 claims.

No matching Phase-2 experiment processes were found on toula, koula, soula,
voula or sasa at the follow-up check. The remaining runs are not queued.
Qwen3 and agentic trials=2 remain outstanding; the BiSR matched comparison is
blocked by the fidelity prerequisite.

The HF release worker is documented in `docs/evidence-release-hf.md`.

## Subsequent execution authorisation

The user subsequently requested starting all pending items. New CPU queues
are now active on soula, voula and sasa; corrected Qwen P6 is running on koula,
with the trials=2 agentic rerun queued behind it. See
`docs/phase2-execution-queue-2026-09-17.md` for exact PIDs, frozen protocols,
status paths and blocked dependencies. A broader process check also found the
external R1 exp002 exploratory run already active on toula; the earlier narrow
experiment-name filter missed it. It has not been duplicated or interrupted.
