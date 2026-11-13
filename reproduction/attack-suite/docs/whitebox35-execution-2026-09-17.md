# Corrected PR #35 execution — 17 September 2026

User authorised the complete five-step sequence: isolate and validate PR #35,
rerun the original 36, expand to every internal GPT2 boundary, repair cotrain
captures separately, and publish corrected evidence to HF.

## Source and validation

Isolated worktree: `/Users/geo/LLM-Attacker-pr35`, branch
`codex/pr35-corrected-sweep-20260917`. Upstream PR35 commit
`19ae26bae9c3229fe2fc6d26a69ea90c492e0a58` was cherry-picked as `ac88fbd`.
No merge into GitHub main or modification of the earlier execution checkouts
was performed. The frozen rerun is `2f75440`; the all-depth/cotrain/package
queue is `209bcd6`; HF publication automation is `88d6f97`.

PyTorch validation: **20 tests passed, zero skipped** across the white-box
module, explicit client-layer/input-gradient checks and co-trained capture
integration tests. The first test invocation had two missing-corpus fixture
failures; the complete rerun passed after copying the exact ignored corpus.
Logs are `build/pr35-regression-with-corpus.log`, `build/pr35-head-gradient.log`
and `build/pr35-combined-tests.log` in the isolated worktree.

## Completed original 36

On koula, `~/llm-attacker-whitebox35-20260917` contains the isolated snapshot.
`original36.log` records **36 complete, zero failed, 18 NOT_EVALUABLE**.
All original input hashes matched again after execution. Wall time was
230.020 seconds. The original historical results were not overwritten;
`HISTORICAL-INVALIDATION.md` in the new result tree marks their omitted-head
objective invalid, not a leakage bound.

The new root is `results/whitebox35-original36-v1`. Its
`verified-complete.json` requires all per-cell head-forward/backward counters
and evidence chains, the36/18 counts, and unchanged input hashes.

Descriptive mean exact-token recovery across three seeds and three widths:

| Split depth | PCA compression | Frozen bottleneck |
|---:|---:|---:|
| 3 | 58.32% | 58.00% |
| 6 | 50.89% | 52.16% |

These are fixed-corpus averages, not universal attack guarantees or a new
privacy-gate decision. The old approximately 1.6% result remains invalid.

## Follow-up execution

Worker PID629969 on koula runs `experiments.whitebox35.queue`:

1. Eleven excluded canaries, one per boundary 1–11, with fixed small attack budgets.
2. **198 stable-defence cells**: 11 boundaries × 3 seeds × 2 methods × 3 widths.
   Fresh training/captures use the original settings. No unrelated baseline
   attacks are claimed. Paired undefended utility controls are retained.
3. **99 new co-trained final-state inference cells**: same boundaries/seeds/widths.
   These save the final victim, head and encoder; verify exact stored-wire replay;
   verify no observation-time parameter/gradient changes; and persist predictions.
   This changes the observation phase and grants the final head, so it is a
   separate stronger-access tier. The historical 18 training exclusions remain.
4. Full matrix verification, descriptive analysis and chunked evidence packaging.

Depth12 is excluded from the internal-boundary comparison; no endpoint control
is represented as executed. The all-depth order is seeded before outcomes.
Every stage preserves failure records and stops instead of outcome-based retries.
Each subprocess has a two-hour timeout; missing inputs or invalid evidence
prevent completion markers. No paper claims are automatically changed.

Live state: `results/whitebox35-followup-v1/status.json`; per-job commands,
return codes and durations: `journal.jsonl`; stdout: `followup.log`.
The original 36 precedes the queue and both existing GPU predecessors were
terminal before it started. The previous Qwen grid failed on a NaN cell;
its evidence remains intact. The two agentic runs completed successfully.
Neither was interrupted by this work.

## HF publication

The corrected original 36 package has been verified and compressed (about 1.7GB)
under `results/whitebox35-release-original36` on koula. The full package is
created only after 198 stable and 99 repaired cells verify. Packages preserve
source, corpus, captures, checkpoints, hashes, layer coverage, measured work,
exclusions and reproduction instructions. Large gzip archives are split into
parts of at most4GB; `manifest.json` records every published file digest.

Local publisher PID 88820 uses cached local HF credentials; none were sent to
nodes. Its status is `build/whitebox35-publication/status.json` in the isolated
worktree. It collects and checks each package, waits for the currently running
baseline HF release to verify, then uploads to private
`Setloop/llm-attacker-paper-b/whitebox35-20260917/{original36,full}`.
A root `WHITEBOX35-UPDATE.md` indexes results and invalidates the old objective.
Each stage gets a pinned remote revision only after all remote Git/LFS hashes
match. `original36-remote-verification.json` and `full-remote-verification.json`
are the publication success records. Queued publication is not completed upload.
