# Phase 2 execution queue — 17 September 2026

Authorisation: the user requested scheduling and starting all pending and blocked
Phase 2 items. Runs use existing nodes and environments; no paid infrastructure
or external services were provisioned. Frozen historical protocols remain intact.

## Started and scheduled

| Node | Queue | PID | State at launch verification |
|---|---|---:|---|
| soula | Sequence lengths 64/128 on WikiText reference and code; then matched order recovery and chaff-retention battery on three paired synthetic reference seeds | 2794423 | Canary passed; first sequence-length experiment running |
| voula | Independent dialogue corpus; paired non-adaptive/adaptive Gaussian noise-whitening comparison | 505063 | Canary passed; dialogue experiment running |
| sasa | 500-step none/mid-lite-k64 trajectories for seeds 11/22/33, followed by VMI on each capture | 1247753 | Canary passed; first 500-step control running |
| koula | Corrected Qwen3-0.6B P6: excluded canaries, then 192 cells and eight controls under a 24-hour budget | 562873 | Both canaries passed; full grid started 14:23:15 UTC |
| koula | Agent-goal-hijack and cross-agent-deputy, trials=2 | 563086 | Waiting for Qwen queue to release GPU; automatically starts on its terminal status |
| toula | External R1 exploratory execution exp002 | existing process 232185 | Already running; preserved without duplicate launch |

The CPU/agentic queues are frozen at `1ee12e111f7bc6306bd0a4249b129b7c2611522f`
in `experiments/phase2_pending_20260917.json`. Protocol SHA-256:
`b57a36c2bd44bfcc38e97a6a539a151ff4f7eb66d46841c8aa12930606fb2d40`.
The Qwen protocol is frozen at `875cb2c0dc0cbec50e41bc3a2f4edc0a478cd2db`;
see `experiments/crossed_span_protocol.p6.qwen3.v1.json` and its Qwen-specific
driver, tokenizer preparation and analysis.

Qwen canaries took 51.125 seconds (trained cell) and 90.569 seconds
(all-local control). The 1.5-times projected full-grid cost is 4.392 hours;
the worker enforces the 24-hour ceiling and excludes both canaries from
analysis. Projection is not a completion guarantee.

CPU/agentic checkouts: `~/llm-attacker-phase2-pending-20260917` on each assigned
node. `queue.log`, `queue.pid`, `pending-source-commit.txt` and
`results/phase2-pending-v1/NODE/{status.json,journal.jsonl}` retain commands,
timings and outcomes. A failed prerequisite prevents dependent jobs; unrelated
jobs continue. Failed outcomes are retained and never silently retried.
Successful jobs require their expected artifacts and valid evidence chains.

Qwen checkout: `~/llm-attacker-p6-qwen-20260917`. Its terminal queue state is
`results/qwen-queue-status.json`; the agentic worker waits for that state,
not merely a grid marker. The GPU has one experimental queue at a time.

The dialogue input contains 96 unique documents selected before outcomes from
the cached official DialogSum training split. Corpus identities and selection
rules are pinned. All three extension corpora have zero exact normalised-text
hash overlap with the frozen validation partitions and at least 128 GPT-2
tokens per document. The queue uses fixed prefixes, not overlength documents.

## Blocked dependency register

| Item | Admission condition | Current action/status |
|---|---|---|
| B1 faithful BiSR validation | Original SensiReplaced version/split hashes, aligned dual-cut implementation and metrics, prospective scientifically justified fidelity rule | Prepared reference configuration retained in the BiSR audit. Five-node search found only raw-article substitution. No faithful run can currently be admitted. |
| B2 matched BiSR v4 | B1 fidelity passes; amendment resolves backward/forward order wording | Registered dependency on B1; no computation falsely presented as fidelity-qualified. |
| P6 Mode-1 / current external programme | Valid current external protocol and completed admissible upstream evidence | Existing R1 exp002 on toula remains running. Its discovery corrects the earlier narrow process check; no confirmatory completion is asserted. |
| Repeated-session external v13 / composite long horizon | Actual external bundle/release mapping and frozen session/key-reuse threat model | Remains a design/data dependency. The synthetic order/chaff jobs do not impersonate this experiment. |
| August deployment evidence | Original fleet journals/manifests matched to headline cells | Read-only discovery inspected toula captures/evidence/pilot trees and koula result roots; no matching August deployment release identified in those locations. Collection remains open. |

These are explicit blocked tasks, not background jobs that are secretly running.
Original BiSR data cannot be reconstructed by relabelling a newly generated
substitute. No author contact or paid sanitisation request has been sent.

## Interpretation and follow-up

Two greedy agentic trials are deterministic repetitions, not independent model
seeds. Order/chaff results concern synthetic reference bundles and the
connectivity-selected retained rows, not a deployed v13 guarantee. The 500-step
probe measures training trajectories; it does not by itself prove held-out
convergence. Adaptive whitening is one declared attacker adaptation.

Collect and verify completed evidence before analysis/paper changes, then stage
new results as a separate HF supplement. The existing HF upload workers continue;
these new executions are not yet included in the uploaded snapshots.

## BiSR specification update

See [the prospective fidelity specification](bisr-fidelity-specification-2026-09-17.md)
and `experiments/bisr_fidelity_spec.v1.json`. The specification decisions are
resolved. Original-setting replication still requires the original processed
data and reference-cadence reconciliation. The designated alternative is an
explicit single-seam adaptation with its own conformance gate and execution
amendment; it cannot qualify the frozen v4 as a faithful reproduction. No new
BiSR job has been launched and neither route is currently admitted.
