# PR35 corrected white-box execution status

## Superseding status — Paper B revision

All 198 stable cells and 99 fresh co-trained final-inference cells completed
on koula at 18:56:48 UTC; the full package passed source-side verification.
The original36 package is verified on HF at
`ba37427807a29f217e6fefb565c5e468e9bdac81`. The full package is still being
collected by the existing publisher; remote verification is not yet complete.
The code is integrated into the Paper B revision branch with source hashes
preserved. See `docs/paper-b-closure-2026-09-17.md` for current claims and tests.

The launch-time record below is retained as history.

Implemented in isolated worktree `/Users/geo/LLM-Attacker-pr35`, branch
`codex/pr35-corrected-sweep-20260917`. The existing Phase 2 checkouts and
historical results are unchanged. PR35's fix is applied locally; this does
not merge the upstream GitHub PR.

- Validation: 20 PyTorch tests passed, zero skipped. This includes the PR's
  compute counters, exact client-layer gradient traversal and five co-trained
  final-state capture tests.
- Original matched rerun: 36 corrected cells completed, zero failures,
  230.020 seconds; all original input hashes matched after execution.
  The old missing-head results remain invalid. Historical 18 co-trained
  training cells remain NOT_EVALUABLE.
- All internal GPT2 depths 1–11: all eleven excluded canaries passed.
  The 198-cell stable-defence grid is running on koula.
- Repair tier: 99 new co-trained final-state inference cells queued afterward.
  Matching victim/head/encoder checkpoints and exact wire replay are required;
  this is a separately labelled stronger-access experiment, not a repair of
  the old training observations.
- HF: verified original-36 package is being collected. Publication worker
  waits for existing baseline verification and then publishes each verified
  stage to private `Setloop/llm-attacker-paper-b`, under
  `whitebox35-20260917/{original36,full}`. Remote publication is still pending.

Koula checkout: `~/llm-attacker-whitebox35-20260917`.
Follow-up PID: 629969. Live state:
`results/whitebox35-followup-v1/status.json`; per-job records:
`results/whitebox35-followup-v1/journal.jsonl`. Each failed job is preserved
and blocks dependent stages; there are no automatic outcome-based retries.

Local publication PID: 88820. State and immutable remote verification records:
`/Users/geo/LLM-Attacker-pr35/build/whitebox35-publication/`.
Full execution notes:
`/Users/geo/LLM-Attacker-pr35/docs/whitebox35-execution-2026-09-17.md`.

Source anchors: PR fix ac88fbd; frozen rerun 2f75440; follow-up/capture repair
209bcd6; publication automation 88d6f97. Depth 12 is excluded from this internal
boundary experiment; no endpoint control is claimed.

The prior Qwen queue had already stopped on a NaN cell, and both queued
agentic runs had completed, before the corrected white-box rerun began.
Neither was interrupted; Qwen's failed evidence remains intact for diagnosis.
