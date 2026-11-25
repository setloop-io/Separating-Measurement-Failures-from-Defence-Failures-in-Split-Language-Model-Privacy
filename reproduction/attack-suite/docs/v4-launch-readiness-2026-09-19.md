# v4 fleet readiness — 19 September 2026

**Current status (20 September):** all 80 adapted prediction cells and the complete CPU analysis have finished. Archive-based statistical reproduction passed. See the [completion and release record](v4-adaptation-completed-2026-09-20.md). All execution-status entries below are historical snapshots.

**Current execution:** full adapted comparison launched on both GPUs on19 September13:19UTC. Source`2d0f58b9d6912206d1070679e31853f9a344cefc`; frozen launch SHA`6f0dc502ff275be21bba73d5b29f291356bb60ab8a1f0e82d16febea58564d63`. Toula run`b831a06d-ecaf-4d9c-bbd1-b2d2a0ad0270`; koula run`ae607e15-41e2-440e-8d25-82d199d9f8f9`. Both verified49 staged assets and matching environments and emitted first prediction records. Each executes40 joint cells×256 documents, SIP/bf outcomes plus b diagnostic. Each process group is limited to23.5h; together with1h earlier-checks and2h preparation reservations this respects50 aggregate GPU-hours. No automatic retries or partial-set scoring. CPU prerequisites and80 trained gates passed. Earlier status entries below are historical.

**Parallel work:** soula/voula/sasa audits completed successfully at source `047d7e6`. Soula verified112 auxiliary articles have no exact document-hash/ID overlap with v3 partitions; voula verified qualified source and20 replay records; sasa verified240 control cells/61440 document pairings. Reports: `build/v4-readiness-20260919/parallel-audits/`. Toula canary run `38f63c65-f2ae-4f54-bca3-d44bb6e0be36` is running at source `a83522a`, with a300-second process-group limit charged to the existing1-hour prior/checks reserve. The interrupted laptop-relayed data copy was replaced by successful direct koula-to-toula rsync; canary inputs remain hash-checked before use. Koula preparation continues without interruption. Full comparison remains unlaunched.

**Current:** koula recovered; fresh SIP preparation and subsequent80 mechanical gates are running as `91effbd8-9efb-4489-9299-9611380b67ac`. Frozen auxiliary verification passed. This stage has a2-hour limit within the50 aggregate GPU-hour allowance. Full comparison is not launched. Older outage/failure entries below are historical.

**Resource update:** user approved50 aggregate GPU-hours across toula/koula. Budget approval is no longer pending. Koula is currently unreachable (SSH host-down/timeout); four other nodes respond. Next-stage training has not started. Authorisation is recorded in the isolated worktree at `experiments/v4_resource_authorization_20260919.json`.

**Latest:** exact-head activation replay is now resolved. SDPA reproduces all 20 original captures (5,120 document-depth rows) bit-for-bit on toula/koula. The prior eager mismatch was a replay-kernel mismatch. Correction source `63a88c1`; no tolerance relaxation or original-wire replacement. Full campaign admission remains pending. See `/Users/geo/LLM-Attacker-v4-adaptation/docs/v4-capture-replay-correction-2026-09-19.md`. Historical failure details below are retained.

The user requested toula/koula for GPU and soula/voula/sasa for CPU while the independent HF release continues. All five nodes are reachable. Both NVIDIA GB10 devices were idle at inspection; each GPU node has 20 CPU threads and about 116 GiB available RAM. Soula/voula/sasa each have four CPU threads and approximately 6.5/14/6.3 GiB available RAM. The HF upload on koula remains running.

No new v4 comparison outcomes have been generated. The frozen v4 bytes remain unchanged. Its binding gate says: “B1: fidelity gate - reproduce the published BiSR reference setting”. The completed C-A result qualifies the explicitly adapted single-seam implementation, not that original-setting prerequisite. Original v4 also names the old pilot MLP inverter and an algorithm order superseded by the audited executable reference. Launching the repaired GRU implementation as unchanged “faithful v4” would therefore misstate the protocol.

The existing prospective fidelity specification defines a separate adaptation route A, conditional on C-A, using the same comparison matrix. The user approved the separately named adaptation amendment on 19 September 2026. The amendment must preserve original v4 and all historical outcomes, explicitly retain the original-setting limitation, and bind corrected algorithm, grants, inputs, source, environment, compute and complete-set analysis before candidate outcomes.

## Proposed dispatch after admission

| Node | Work |
|---|---|
| toula | GPU prediction cells for victim seeds 101, 202, 303, 404, 505 |
| koula | GPU prediction cells for victim seeds 606, 707, 808, 909, 1010; preserve HF upload |
| soula | Input identity, reserved partitions and auxiliary disjointness checks |
| voula | Capture/checkpoint coverage, grant validation and prediction journals |
| sasa | Frozen-control pairing, completeness and final crossed-bootstrap statistics |

Each GPU allocation includes both depths (3/6), both grants (public/exact), both attack seeds (17/29), and all 256 frozen v3 evaluation documents. SIP and bf produce 160 arm cells from 80 joint executions. Existing v3 control measurements are reused rather than rerun. Original execution shards are available under `llm-attacker-pilots/gpt2-wikitext2-v01/code/results/validation-heldout-v3-r3-shard*`; original shards0–7 belong to toula and8–15 to koula. Presence of original observation and exact-head files was checked; launch still requires their full per-file hashes and grant conversion/replay checks.

## Budget and prerequisites

Scaling the completed C-A candidate measurements for SIP + GM + bf (excluding the f diagnostic) gives **30.21 serial GPU-hours online**, or **15.11 hours with an ideal even two-GPU split**. This is a planning extrapolation, not a measured v4 ETA. It excludes fresh SIP training, source/checkpoint preparation, cold loads, I/O, CPU scoring and contingency. Original v4's10–30GPU-hour envelope is therefore not yet sufficient for full admission. A resource canary must resolve throughput and budget without reducing documents, seeds or iterations after outcomes.

The original v3 gradient capture used its own loss normalisation and saved head format. Repackaging for the qualified worker must preserve the original observation bytes and account explicitly for that gradient scaling; exact/public grants must be enforced by actual weights. No final-head substitute, fresh victim or stronger grant may silently replace a missing original checkpoint.

The local readiness snapshot is `build/v4-readiness-20260919/readiness.json`. This is a dispatch proposal and blocker inventory, not a frozen amendment, experiment launch or claim that original B1 passed.


## Admission execution update

The approved scope is frozen in the isolated worktree `/Users/geo/LLM-Attacker-v4-adaptation`, experiment `2ff348ab-19d9-4620-9b09-348e85ece571`, initial commit `f27daaf`; original v4 is unchanged. This is a scope freeze, not full execution admission. Five-node preflight is complete; all 60 original shared binary hashes pass independently on toula and koula. CPU hosts pass declaration checks only; task-specific data staging and audits remain pending. A hostname-case infrastructure error was corrected in `4f12951`; failed attempts remain recorded.

Resource canary run `a7f9ecaa-bb9a-440a-90db-d8f36f41c101` at `b3e90ad` completed four fixed excluded C-A observations (depth3/6 × public/exact), full GM32/SM800, retaining SIP/bf and omitting the unneeded f diagnostic. Measured online times: 5.1283, 5.0324, 6.8041, 7.3779 seconds. Equal-weight full-matrix extrapolation is **34.6207 GPU-hours online**, excluding fresh training, setup and contingency. It is a small-sample timing estimate, not a guaranteed ceiling or population runtime estimate. User approval for a 40 GPU-hour total ceiling is pending; the existing ceiling remains 30.

Original exact-head replay is blocked. Runs `f22bd6a6-e0ef-495e-983f-d01cfd882c4c` and `439ca245-f1b7-4fcb-b071-201ec527379d` failed the unchanged conformance tolerance on toula/koula. Diagnostic run `7f7fb161-1084-4db4-a526-1753ce9bb569` at `8be7a42` reports seed101/depth3/first8 rows: max absolute difference 0.000244140625, RMS difference 1.4923332400940126e-6, reference RMS16.60871696472168, 1376 elements outside `1e-6 + 1e-5*abs(reference)`. This suggests a small numerical discrepancy but its cause is not established. Do not call it a pass, silently relax the threshold, or discard the original captures. Inspect the original attention kernel, numerical settings, environment and execution source before any revised compatibility decision. Full gradient replay has not been performed because original full-victim downstream weights have not been established; the saved exact client head alone is insufficient for that check.

No v3 recovery outcomes, fresh adaptation GRUs, or full comparison jobs have been produced. Remaining: resolve replay compatibility; bind auxiliary/control/environment assets; train seeds17/29 GRUs and run their mechanical gates; freeze complete prediction manifests; admit resources; dispatch full matrix; collect complete-set statistics. The C-A HF evidence upload continues independently and its publication worker remains guarded.
