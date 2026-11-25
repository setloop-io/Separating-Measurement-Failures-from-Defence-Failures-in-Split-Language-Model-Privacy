# BiSR fidelity campaign started — 18 September 2026

**Current status (19 September):** see the [completed C-A conformance and release record](bisr-fidelity-completed-2026-09-19.md). The dated observations below are preserved as historical snapshots.

The user approved the requested **36 GPU-hour ceiling** and reaffirmed proceeding.
This clears the resource-approval blocker recorded in the earlier progress report.
The frozen scientific inputs, seeds, iteration budgets and equivalence margin are
unchanged. No fidelity pass has yet been established.

## Live execution

Host: **koula**, directory `~/llm-attacker-bisr-fidelity-20260918/`.

| Stage | State at verification | Process |
| --- | --- | ---: |
| Checkpoint preparation | Running; first GRU complete, second at epoch 10/20 | worker 1100771; supervisor 1100770 |
| Matched campaign | Waiting automatically for all prepared checkpoints | supervisor 1102383 |

Both auxiliary activation caches completed. Preparation trains six GRUs, three
shared victim models, exports six exact heads and captures/replays all 256
validation inputs at both depths for each seed. The public downstream checkpoint
is separately retained. Preparation has a six-hour ceiling; a failure stops the
dependent campaign without retry.

After preparation, the queued worker automatically:

1. Verifies all checkpoint hashes, input identity, package inventory, metric
   resources and source hashes.
2. Checks trained SIP logits and GM/SM losses, gradients and one-step updates on
   the fixed first and last observations in all twelve depth/grant/seed strata.
3. Commits the complete checkpoint/prediction manifests to a separate run-record
   Git repository under `campaign-v1/records/`, before any full-budget heldout
   predictions. This does not modify the source working branch.
4. Runs candidate and upstream in separate label-free prediction subprocesses:
   **24 prediction cells × 256 documents**, each retaining SIP, b, f and bf.
5. Verifies every immutable prediction and hash-chained journal before scoring.
   A separate scorer evaluates all eight endpoints in each depth/grant stratum.

Any failed gate, incomplete cell or exhausted time allowance stops the campaign.
There is no favourable-cell selection or automatic retry. The deadline is
preparation start plus 35.9 hours, reserving 0.1 hours for completed canaries
within the approved total of 36 GPU-hours. This conservative wall-clock limit
also counts waiting and CPU overhead. No other GPU jobs were running at launch.

## Verification and monitoring

Before queuing, an excluded full-budget CPU fixture passed the new trained gate,
both prediction engines and journal validation; all four predicted arms matched.
Two additional regression tests passed for journal integrity and rejection of
truth-bearing prediction manifests. Previous real-model/metric canaries and the
54-test regression suite remain recorded in the earlier progress report.

Read current status (the table above is only a launch snapshot):

```bash
ssh koula 'cat ~/llm-attacker-bisr-fidelity-20260918/run-v1/status.json'
ssh koula 'tail -5 ~/llm-attacker-bisr-fidelity-20260918/run-v1/preparation.log'
ssh koula 'cat ~/llm-attacker-bisr-fidelity-20260918/campaign-v1/status.json'
```

Preparation journals are under `run-v1/checkpoints/`; campaign events, trained
gate traces, per-cell logs, prediction journals and final analysis are under
`campaign-v1/`. Online work records contain per-stage wall time, allocated memory,
and observed transformer-layer forward/reverse token-position counts, including
reverse visits during higher-order differentiation. These are not FLOP estimates
or an equal-compute claim; offline training remains separate.

Source commits: `179cafc` (approved admission and supervised preparation),
`8f8a4c4` (gated paired execution and complete-set scoring). Commits are local;
no new evidence has been uploaded to HF. Once the campaign finishes, inspect its
complete verdict and retained exclusions before publishing or changing papers.

The permitted eventual claim is C-A single-seam implementation conformance on
the fixed WikiText-103 holdout. Original SensiReplaced and published dual-cut
reproduction remain outside this campaign's scope.
