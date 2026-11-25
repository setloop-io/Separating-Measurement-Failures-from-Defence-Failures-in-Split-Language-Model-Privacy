# Study-plan reconciliation for the revised manuscript

Prepared 2026-09-09 from the existing plan, cohort protocols, configurations, and generated
analyses. This document reconciles the reviewer concerns addressed in the manuscript; it is
not an exhaustive audit of every umbrella-plan requirement, a retrospective amendment, or
evidence that proposed follow-up experiments have been performed.

## Plan and protocol identities

The [umbrella research plan](RESEARCH_PLAN.md) describes convergence/decomposition as its
Stage P3 and external replication as Stage P4. Executed cohort names follow later protocols;
executed P3 is external H1 replication. These phase labels are not a one-to-one mapping.

| Executed protocol | Recorded freeze | Specified analysis relevant to this revision | Evidence |
|---|---|---|---|
| `p2-confirmatory-1.0` | 2026-09-04 | Primary trained/frozen paired contrasts; secondary contrasts and span-by-width analysis | [P2 protocol](PREREGISTRATION.md), [decision](P2_CONFIRMATORY_DECISION.md) |
| `p2b-rate-sweep-1.0` | 2026-09-04 | Fixed-width precision sweep with seed-level slopes and secondary contrasts | [P2b protocol](PREREGISTRATION_P2B.md), [decision](P2B_RATE_SWEEP_DECISION.md) |
| `p2c-segment-statistics-1.0` | Selection rule and separate confirmatory configuration recorded on 2026-09-04 | Frozen selected-statistic held-out H2 test; descriptive secondary H4 extension | [P2c protocol](PREREGISTRATION_P2C.md), [confirmatory configuration](../experiments/configs/segment_stats_matrix.yaml), [decision](P2C_SEGMENT_STATS_DECISION.md) |
| `p3-replication-1.0` | 2026-09-06 | H1 replication with separate architecture/corpus contrasts; no new privacy evaluation | [P3 protocol](PREREGISTRATION_P3.md), [decision](P3_REPLICATION_DECISION.md) |
| `p5-delegation-sweep-1.0`, amendment `p5-1.1` | 2026-09-07 | Fixed-width span/defense sweep with per-cell gates; gradient-qualification amendments; external privacy replication deferred | [P5 protocol and amendment](PREREGISTRATION_P5.md), [decision](P5_DELEGATION_TRIANGLE_DECISION.md) |

Dates above are freeze statements in the repository records. They do not independently
establish external registration or prove timing from a later committed snapshot. Per-run
code/configuration digests and protocol versions remain in the [manifests](../experiments/manifests/).

## Planned versus completed work

| Umbrella-plan intention | Completed evidence | Manuscript treatment and remaining work |
|---|---|---|
| Hierarchical seed-level response surface over segment, location, width, operational rate, capacity, and architecture | Separate phase-specific contrasts, slopes, and prediction tests | Explicitly incomplete; no claim of jointly adjusted effects across all planned factors. A new pooled fit chosen after observing outcomes would be exploratory. |
| H4 prediction beyond architecture depth after location/rate/compute adjustment | P2c descriptive model extends span with position and the selected statistic; existing output reports poor held-out prediction and a rank-deficient selection fit | State that the narrower descriptive test did occur, while the program-level H4 remains unestablished. Vary span and position independently and design architecture-level validation prospectively. |
| Optimized full operational-rate envelope with nested tuning and uncertainty | P2b varies forward precision at fixed width/capacity and uses matched stopping budgets | Avoid historical shorthand equating forward precision to measured full-transcript rate or fixed steps to demonstrated equal convergence. Nested envelope estimation remains incomplete. |
| Equal-example and equal-total-compute comparisons; unconstrained oracle and error decomposition | Matched controls and learning curves under the phase budgets | Do not claim the full equal-total-compute or representation/approximation/optimization decomposition was completed. |
| Broad measured resource vector | Analytic FLOPs, tensor-size byte accounting, and local wall time | Explicitly distinguish estimates from deployed transport, end-to-end throughput/latency, and peak memory measurements. |
| Replication beyond a single setting | P3 replicates remote-learning gain across architecture/corpus settings; broadened privacy evaluation remains in the original setting | Separate external utility evidence from missing external privacy replication. |

## H4 design diagnostic and missing-data handling

The H4 extension was prespecified as descriptive and ungated in P2c. Its selection set has
three segment definitions, `{span@start} = {2@3, 4@2, 6@1}`, each observed at two widths.
Thus `start = 4 - span / 2`; intercept, span, and start are linearly dependent. The existing
[analysis](../outputs/segment_stats/segment_stats_analysis.json) reports rank three for a
four-parameter fit and out-of-sample R² of −11.08939. These results and artifact hashes are
recorded in the [claim-evidence ledger](CLAIM_EVIDENCE_LEDGER.md). The rank diagnosis limits
coefficient interpretation; it does not establish why every candidate statistic would fail.

The [P2c decision record](P2C_SEGMENT_STATS_DECISION.md) documents the seed-33 all-local run-ID
collision with P2. The analyzer excluded the protocol-mismatched partner under the recorded
missing-data handling, leaving seven retained seeds per held-out cell. This operational
exclusion is preserved in the manuscript; token rows or frames are not substituted for
independent runs. No exclusion or analysis rule was changed by this revision.

## Historical explanation versus current design diagnosis

The inspected records specify narrower phase analyses and several deferrals but do not
document a separate historical rationale for omitting the full hierarchy. No claim is made
that time, compute, or power constraints caused that decision. The confounded selection
design, fixed factors in the precision sweep, and bundled interventions in P5 are objective
constraints on a retrospective model; they are not evidence of the original reason for
deferral. An author-supplied contemporaneous decision record could resolve that historical
question without changing completed analyses.

## Prospective studies and privacy boundaries

The manuscript proposes external forward-probe replication, a deployed systems benchmark,
and a fresh segment-prediction design. These proposals are not newly frozen protocols; each
would require its own calibration, analysis specification, cohorts, and artifact inventory.
See [reviewer response preparation](REVIEWER_RESPONSE_PLAN.md) for the proposed designs.

Current unevaluated views/threats remain timing, optimizer state beyond the remote module,
full evolving remote state, active/adaptive attackers, within-run joint tensor attacks,
cross-run joint attacks, natural-sequence membership, semantics beyond byte-structure classes,
returned values, and full-precision inbound/outbound gradients. History-accumulation and
canary-membership instruments, undefended gradient property/token probes, and all defended
gradient probes remain unqualified. External replication of two forward probes would not
establish coverage of these omitted or unqualified views.
