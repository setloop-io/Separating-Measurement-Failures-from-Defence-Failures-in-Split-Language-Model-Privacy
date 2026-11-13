# Minimum cross-paper TEE/FHE/CURE experiment plan

Date: 23 September 2026
Status: planning and evidence comparison only. This document authorises no paid
compute, scientific execution, cloud provisioning, or manuscript claim.

## Executive decision

Run one gated secure-compute programme, not another broad defence sweep.

1. Do not rerun Paper 3's 19-cell R1 family, Paper 4's 66-run D1/D3 pilot, or
   LLM-Attacker's GPT-2 v3/P6 matrices.
2. Use Paper 4's prepared `d0-q8` workload as the initial common anchor: 8 byte
   transformer blocks, width 256, span 4 starting at block 2, one remote block,
   per-token q=8, sequence 128, batch 32, 2,000 steps, and `gutenberg_p4`.
3. Treat TEE/CC as the only current candidate for a full protected-training
   cohort. Existing measurements already reject full training with the current
   per-token CPU-CKKS/TenSEAL layout.
4. Only if a materially different CURE-class backend, packing or layout remains
   technically plausible, put it behind one representative forward + backward +
   parameter/optimizer-update feasibility gate. Do not buy a full FHE run until
   that gate passes a frozen time, memory, numerical and security threshold.
5. Before science, use one short paid CC engineering campaign to admit the
   hardware, verify attestation and key release, establish same-workload
   correctness, measure CC-on/off cost, and execute active security canaries.
6. If that gate passes, the smallest protected scientific pilot is 12 logical
   jobs: 3 fresh seeds x {TEE trained, TEE frozen, removed, all-local}.
7. The arm-minimal cross-paper design is 18 logical jobs: add plaintext trained
   and plaintext frozen for the same 3 seeds. Those six extra jobs provide a
   fresh, same-seed protection-off comparison and make the evidence useful to
   the trilemma, defence, and attacker papers without relying on historical
   cohorts. The 12-job matrix is sufficient only for protected feasibility; it
   does not estimate the full protection-by-learning interaction. If the intent
   is to align the exploratory learning evidence with Paper 3's current
   four-seed convention, use 4 seeds x 6 arms = 24 training jobs. Eighteen is an
   arm-complete three-seed screen; 24 is the preferred cross-paper exploratory
   cohort. Neither is confirmation.
8. Do not launch confirmation automatically. If and only if a candidate is
   eligible, freeze a new, single-cell power analysis, seed list, attack suite,
   budget and evidence packet. The existing eight reserved seeds are not by
   themselves a power justification.

The programme must be declared as one shared cohort. Reuse across papers is
efficient, but it is not three independent replications.

## Why this is the minimum useful programme

The three scientific jobs overlap but are not identical:

| Scientific job | What the new cohort can establish | What it cannot establish |
| --- | --- | --- |
| Returned-gradient failure | Whether the backward channel and equivalent exports are actually inside the protection boundary; whether an external operator can obtain the zero/support pattern or token signal; whether active mutations are rejected | A universal statement about all TEEs/FHE systems; output privacy; inaccessible physical or host-side attacks |
| Paper 3 trilemma | Utility, gain over frozen, benefit over removal, operator-confidentiality evidence and measured cost for one protected cell | A universal trilemma result; a claim that representation privacy and operator confidentiality are the same; an exact replication of R1 |
| Paper 4 defence redesign | A separate protected-deployment result and active-attacker evaluation under an explicit trust model | A repair or reinterpretation of the completed D1/D3 pilot; a positive D1/D3 representation-defence claim |
| LLM-Attacker framework | A real confidential-compute profile, observation schemas, live adapters, controls, journals and evidence chain | Validation of every registered module or an independent replication of the older GPT-2 campaign |

The shared physical execution should generate utility, causal-learning,
correctness, performance and complete boundary observations once. Paper-specific
analyses may consume that immutable evidence bundle, but each paper must retain
its own claim boundary.

## Anchor choice

### Recommended: Paper 4 `d0-q8`

This is the cost-minimising choice because the repository already has:

- a hash-bound checkpoint and panel;
- a same-checkpoint serialized correctness runner;
- an observation contract;
- actual-shape CKKS forward-projection measurements;
- per-shape and selective-policy cost accounting;
- a corpus, seed audit, manifest and verification stack;
- a CC hardware-arrival specification; and
- a draft adversarial protocol.

It is also scientifically useful: the completed Paper 4 pilot reports that the
cell passes utility and gain-over-frozen and has a qualified forward-structure
breach; its overall privacy status remains undetermined because property-probe
qualification is incomplete. A protected version therefore asks a clean new
question: can operator confidentiality change the feasibility result without
claiming that the representation itself became non-informative?

Relevant records:

- `/Users/geo/Paper-4-Defense-Redesign/docs/TEE_FHE_TRACK.md`, lines 8-42.
- `/Users/geo/Paper-4-Defense-Redesign/docs/TEE_FHE_EXEC_PROTOCOL.md`, lines 14-60.
- `/Users/geo/Paper-4-Defense-Redesign/docs/PILOT_DECISION.md`, lines 1-69.
- `/Users/geo/Paper-4-Defense-Redesign/experiments/configs/pilot.yaml`.

### Alternative: Paper 3 `depth-2`

Paper 3's post-exploration `depth-2` cell is the stronger candidate if a positive
benefit-over-removal result is required: it passed exploratory utility,
gain-over-frozen and benefit-over-removal, and was rejected only by a qualified
privacy breach. It is also cheaper than Paper 3's `depth-4` alternative.

Do not run both anchors initially. Choosing `depth-2` would require amending the
current TEE/FHE protocol, rebuilding the observation contract, using Paper 3's
Tiny Shakespeare split and quantizer conventions, and requalifying the new
attacks. The gain in claim strength does not justify that duplication for the
first cost screen. If the owners make benefit-over-removal a mandatory headline
gate, select `depth-2` before implementation and drop `d0-q8`; do not add it as a
second factorial level.

## Stage 0: no-paid-compute admission packet

Complete this before requesting a cloud quote.

### Freeze the deployment contract

- adversary roles: host OS, hypervisor, orchestration administrator, network
  observer, malicious accelerator/server, and physical operator in or out;
- trusted components: exact CPU TEE, GPU CC, verifier, application measurement,
  key-release service, client and cryptographic libraries;
- permitted leakage: message length, timing, failures, resource counters,
  outputs and model release, each explicitly allowed or forbidden;
- protected phases: ingress, tokenisation, forward, return, backward, update,
  optimizer state, checkpointing, logs, crash/restart and egress;
- exact export policy for checkpoints, captures, diagnostics and raw secrets;
- hardware, firmware, VBIOS, driver, TEE module, verifier, collateral/CRL and
  dependency versions;
- workload, seeds, panels, tolerances, metrics, stop rules and evidence schema.

TEE, hybrid TEE-FHE and CURE/FHE require separate threat-model records. Their
security outcomes must never be pooled into one generic privacy score.

### Implement the shared two-view recorder

Every event must be tagged as one of:

1. `trusted_diagnostic`: internal plaintext used only for correctness, scoring
   truth and positive controls; or
2. `external_attacker`: exactly what the declared operator/network attacker can
   obtain.

Record, with common request/step IDs:

- forward and returned values;
- both gradient directions;
- remote parameters, updates and optimizer state;
- history and within-run joint views;
- serialized ciphertext/packet bytes, headers, shapes, timing and access events;
- CPU-GPU staging and memory inventories;
- quote, collateral, verifier policy, nonce, application measurement, endpoint
  key and key-release decision;
- protocol errors and all FHE feedback;
- logs, checkpoints, dumps, telemetry, retry/restart and export decisions; and
- every active mutation and the resulting accept/reject/abort outcome.

The trusted view must never be silently treated as attacker-visible. Raw MITM
sidecars and trusted captures need separate access control because the evidence
system itself can otherwise become the leak.

### Implement and qualify the missing attacker adapters

Reuse LLM-Attacker's surface registry, artifacts/journals, evidence chain,
wire-shape, MITM, artifact-canary, timing, protocol-integrity, key-session,
malicious-server and reconstruction components. Add real target adapters rather
than a new family of toy attacks.

Required for the TEE campaign:

1. **Attestation and key release.** One valid path plus wrong measurement,
   modified evidence, stale/revoked collateral, replayed nonce/quote, wrong-machine
   relay, endpoint-key substitution and channel-binding mismatch. Invalid paths
   must release zero secrets; the valid path must succeed.
2. **Live protocol mutation.** Replay, reorder, duplicate, drop/truncate,
   cross-session swap, downgrade, tamper, stale-step/result and request-result
   substitution against the real attested connection. The current
   `protocol_integrity` path is a simulator/journal consumer, not a live injector.
3. **Malicious computation.** Stale/substituted weights, skipped/scaled/zeroed
   operators or gradients, replayed outputs and deliberately wrong results, with
   request/step binding and measured training/utility consequences.
4. **Boundary completeness.** Independent canaries in forward, backward, update,
   CPU/GPU staging, IPC, VRAM, logs, checkpoints, dumps, environment and telemetry.
5. **Backward and history attacks.** Paper 1's zero/support detector and token
   attack, joint activation-gradient reconstruction, step-history accumulation,
   remote/optimizer-state attacks and matched positive/null controls.
6. **Host-visible side channels.** Packet shape, timing, accessible counters,
   crash/restart and access traces. TDXRay-like host-side or physical TEE.fail
   experiments remain `UNASSESSED` if the cloud guest cannot execute them.
7. **New schemas and gates.** Add attestation/key-release, confidential-host trace,
   malicious-computation and external/trusted-view surface types. Do not reuse a
   token-recovery `+1 pp` gate or a generic 5% integrity tolerance for secret
   release: invalid secret releases have a zero-tolerance gate.

Conditional FHE/CURE adapter:

- record parameters, security mode, keys/contexts, ciphertexts, packing and extra
  slots, refreshes, decryption outputs, error/status/timing feedback and query
  budget;
- inject malformed, modified, replayed and wrong-context ciphertexts;
- test output/verification-feedback and extra-slot preconditions with an
  intentionally leaky positive-control oracle;
- distinguish failure to expose a required oracle from generic proof of FHE
  security; and
- test result integrity separately from ciphertext confidentiality.

Do not implement a physical DDR5 interposer, reconstruct patched historical
exploits, run a broad CVE sweep, or adapt all registered LLM-Attacker modules.

## Stage 1: conditional CURE/FHE representative-training-operation gate

This is a conditional alternative to TEE, not an automatic cloud stage. Do not
execute it merely because CURE exists: first identify a materially different
backend, packing or layout that could change the decision already reached for
per-token CPU CKKS. The proposed first envelope is at most 2 aggregate CPU
hours, after its packet is specified; that is a cap proposal, not authorisation.

The current TenSEAL work encrypts per-token activations and evaluates a plaintext
weight projection. It does not implement encrypted input gradients, weight
gradients, parameter updates or optimizer state, and therefore is not CURE
training.

First inventory the selected remote block's operations and establish which are
supported, approximated or forced back to the trusted side. Then implement the
smallest exact representative CURE-class training operation that includes:

- one actual-shape remote projection rather than a toy dimension;
- input-gradient computation;
- weight-gradient computation;
- parameter and optimizer update;
- client encryption/decryption/refresh work;
- the required output-disclosure security mode;
- all communication, key/ciphertext memory and extra-slot handling; and
- the complete error/feedback interface.

Use a matching plaintext numerical control and at least five retained
repetitions. Measure bounded token counts first; execute a 4,096-token canary
only if the measured projection fits the frozen CPU-hour cap. A forward-only
matrix multiplication is not sufficient because it omits the training work,
but this elimination screen need not reproduce the full transformer. A pass
only admits a later full-block canary: attention, LayerNorm, GELU and their
placement must still be accounted for before any FHE training claim.

Freeze a feasibility threshold before observing the result. At minimum it must
cover:

- projected complete-step time and total 2,000-step projection;
- peak client/server/key/ciphertext memory;
- numerical error in forward, gradients and update;
- training-trajectory risk from approximation;
- serialized bytes and client work; and
- presence or absence of malicious-feedback/oracle protections.

Stop if a required operation or placement has no defensible design, the required
security mode makes precision unusable, or the projected campaign exceeds the
frozen budget. Do not substitute CURE's published MLP/convolutional result for
this transformer test: its server
model is semi-honest and it supplies no `G_F`/`G_B` evidence for this workload.

Why this is a hard gate: the completed Paper 4 measurements put the four remote
linears at about 775 ms/token. A 4,096-token forward is therefore roughly 53
minutes before backward/update. Slot packing did not provide the expected
asymptotic improvement: one encrypted linear at 128 tokens was about 15 seconds
and the remote linear set about 102 seconds. Repeating full current-backend FHE
training would spend money to confirm a decision already supported by the
engineering evidence.

Relevant records:

- `/Users/geo/Paper-4-Defense-Redesign/docs/TEE_FHE_BATCHED_DECISION.md`, lines 12-77.
- `/Users/geo/Paper-4-Defense-Redesign/docs/TEE_FHE_TRACK.md`, lines 93-149.
- `/Users/geo/Paper 3 - Privacy-Constrained Remote/artifacts/privacy-mechanisms-20260922/evidence-register.md`, lines 45-55 and 129-163.

## Stage 2: one paid CC engineering and adversarial campaign

This stage remains engineering and uses no scientific seed. The existing
adversarial protocol is a draft: freeze its own identities/nonces, attack
budgets, controls and cost allocation before executing its sessions. Admission,
engineering correctness/performance, adversarial tests and post-mitigation
remeasurement may share one rental, but remain separately labelled stages.

### Hardware admission

Run the existing arrival specification first. Record GPU/CPU identity, CC mode
and protected memory, firmware/VBIOS/driver/TEE/verifier versions, quote,
collateral/CRL freshness, measured workload and channel key. If the platform is
affected, unpatched, debug-enabled, unable to attest, or cannot bind release to
the workload and endpoint, stop before performance.

### Active-security admission

A compact admission matrix is one valid condition plus seven invalid
attestation/release classes, each under three fresh nonces/session identities:
24 verifier sessions, not 24 training runs. Reuse their connections and recorded
traffic for protocol mutation tests where valid; do not retrain for every attack.

Run boundary canaries, backward/history attacks, live mutations, malicious
result tests, artifact/log scans and host-visible side-channel probes in this
same engineering campaign. Each arm needs an intentionally vulnerable positive
control and a clean/null control. Performance must be remeasured after any
mitigation found necessary.

Only an exact all-observed `PASS` may authorise measurement. A signed external
controller state must bind the bootstrap, active-security manifest and CC-on
launch; a missing view is `UNASSESSED`, never an automatic pass.

### Same-workload correctness and cost

Use the exact final topology: wrap the selected remote segment, not every model
linear. The supported minimum lifecycle uses matched separate instances, not a
claimed same-host mode switch. Execute CC-on first, verify deletion/absence,
then provision CC-off. Match provider, SKU/GPU, CPU family, region/zone,
immutable image, driver, software, checkpoint, inputs, precision, batch,
transfer pattern and diagnostic policy. Record the unavoidable distinct-host
limitation.

Run exactly three measured repetitions per arm under the signed campaign
contract, plus one warm-up that is not counted as a repetition. Include
same-checkpoint inference on the frozen panel, repeated complete training steps
covering forward/backward/update, and cold setup/attestation separately from
warm operation.

Retain the existing correctness gates unless amended before execution: max
absolute logit deviation <= 1e-4, absolute loss delta <= 1e-5, and 100% token
agreement. Add gradient, update, optimizer-state and checkpoint-digest deltas.
Measure complete-step latency/throughput, setup, CPU-GPU bridge/staging time,
serialized bytes, peak trusted CPU/GPU and host-visible memory, failure/abort
rate, and cost after all required mitigations are enabled.

### Optional Paper 1 bridge canary

If the three-paper scope includes the Qwen decoy system, prepare its pinned
checkpoint and dependencies before renting the instance. During the same CC
session, run a fixed microbatch/short-step canary in protection-off and
protection-on modes:

- the trusted diagnostic view must reproduce the known real-row zero/support
  pattern and token instrument;
- the external view must contain only the declared protected transcript; and
- staging, logs, checkpoints, errors and timing must be included.

This is not another 2,000-step training cohort and does not repeat Paper 1's
nine-seed result. It directly checks that the new boundary covers the channel
that motivated the series.

### Budget rule

Use the existing proposal as the first cap: no more than 4 aggregate CC/GPU
hours for admission, warm-up, failures, paired sessions, mitigations and teardown.
This is a proposed cap, not authorization. Obtain the actual instance quote and
billing granularity first.

## Stage 3: one shared scientific pilot

Run this only if Stage 2 passes every admission, correctness and budget gate.
All model initializations, training seeds and attack-calibration data must be
fresh and globally collision-checked. Existing checkpoints may support Stage 2
engineering only; they must not initialize or qualify the new scientific cohort.

### Minimum protected-feasibility matrix: 12 jobs

| Per fresh seed | Purpose |
| --- | --- |
| TEE trained remote | Protected candidate |
| TEE frozen-random remote | `G_F = L_frozen - L_trained` |
| Removed remote | `G_B = L_removed - L_trained` and retention check |
| All-local | `Delta_U = L_trained - L_all-local` |

Three seeds produce 12 logical training jobs. This is an exploratory pilot, not
a confirmation. It answers whether a protected candidate jointly has acceptable
utility and useful remote learning under the declared TEE adversary. It does not
directly estimate how protection changes the learned trajectory relative to a
fresh plaintext cohort.

### Arm-minimal cross-paper matrix: 18 jobs

Add, per seed:

| Additional arm | Added value |
| --- | --- |
| Plaintext trained remote | Fresh same-seed protection-off trajectory, leakage-positive control and utility/cost comparator |
| Plaintext frozen-random remote | Separates the protection x remote-update interaction and supplies matched plaintext `G_F` |

The full per-seed set is therefore:

`{all-local, removed, plaintext-trained, plaintext-frozen, TEE-trained, TEE-frozen}`.

This is the smallest balanced arm matrix that lets all
papers compare protected and unprotected learning without treating historical
Paper 3/Paper 4 observations as a new matched control. All-local and removed may
be shared across protection worlds only when their trusted computation,
initialization, optimizer, data order and numerical environment are identical
and that equivalence is prospectively verified.

Three seeds make this an exploratory feasibility screen. Paper 3's present R1
descriptive design uses four independent seeds. If the result is intended to
enter Paper 3 as aligned new exploratory learning evidence rather than only as a
deployment screen, run the same six arms for four fresh seeds: **24 training
jobs**. This is the preferred maximum-overlap design; it is still not
confirmation.

Final-checkpoint inference, cold/warm serving and multiple offline attacks reuse
these checkpoints and captures; they are not new training cohorts.

The 12/18/24 counts cover model-training jobs only. Attestation/key-release
negative sessions, protocol mutations, positive/null controls, attacker
qualification and offline attack scoring are additional experiments, although
they should reuse recorded traffic and checkpoints rather than multiply
training.

### Conditional CURE addition

Only if Stage 1 passes, add CURE/FHE trained and frozen arms for the same seeds.
With shared all-local/removed controls, that adds two jobs per seed to any
selected matrix. Controls are shareable only if the trusted architecture,
initialization, optimizer and schedule remain identical; otherwise CURE needs
its own four-mode cohort.

Do not construct a mechanism x D1 x D3 x depth x precision factorial. The new
question is protected computation, not another representation-defence search.

## Stage 4: conditional confirmation

Confirmation is earned by an eligible exploratory result, not by unused seed
availability.

Before confirmation:

- freeze the selected mechanism and exact cell;
- use an independently justified power analysis for utility, `G_F`, any required
  `G_B`, active-security and performance endpoints;
- freeze fresh seeds, multiplicity family, attack budgets and qualification;
- freeze the total currency cap and evidence release plan; and
- keep exploratory and confirmation summaries separate.

Paper 3's proposed eight-seed full-family confirmation was underpowered and is
not a template for this single-cell study. A single-cell design may need fewer
or more seeds; determine this before execution. Do not pool token rows, frames,
queries or timing repetitions as independent training replicates.

## Shared metrics and gates

Emit these once from the canonical evidence bundle.

### Learning and utility

- final held-out loss and training trajectory;
- `Delta_U = L_trained - L_all-local`;
- `G_F = L_frozen - L_trained`;
- `G_B = L_removed - L_trained`;
- seed-level paired intervals and all prespecified gate decisions; and
- convergence/checkpoint divergence under protection.

Compute both Paper 3 and Paper 4 decision summaries from the same per-seed raw
values if their rules differ; do not silently merge thresholds.

### Correctness

- max/mean logit and loss delta;
- token agreement;
- activation, gradient, parameter-update and optimizer-state error;
- checkpoint/state digests and replay equivalence; and
- clean false-reject, crash, hang and abort rates.

### Security

- valid release success and invalid release count (required zero);
- workload/endpoint/session/request/step binding;
- replay, tamper, downgrade, substitution and malicious-result acceptance;
- canary exposure by location and view;
- external reconstruction/property/membership/history advantage with matched
  nulls and calibrated positive controls;
- oracle/key-recovery outcome with query budget, when applicable;
- accessible side-channel advantage/AUC and false positives; and
- named `UNASSESSED` classes.

Nondetection by an empirical probe is not a cryptographic or hardware security
certificate. The claim shape is: the named deployment resisted the named attacks
under the named resources and trust assumptions, at the measured cost.

### Systems and cost

- cold attestation/key setup;
- complete warm training-step latency and throughput;
- time to first token, prefill, decode/token and total inference latency if the
  inference add-on is retained;
- trusted-client work and CPU-GPU bridge time;
- serialized wire/ciphertext bytes;
- trusted CPU memory, protected GPU memory, host-visible memory and key/ciphertext
  memory;
- retries, failures and teardown; and
- actual billed time, storage and egress.

Use paired session uncertainty for performance. Independent training seed is the
inferential unit for learning. Tokens, windows, frames, steps and repeated
requests are nested measurements.

## Claim map after a successful campaign

### Returned-gradient paper

Permitted claim: the tested protected boundary covered the previously exposed
backward channel, and the specified external operator did not receive its
plaintext under the admitted, attested deployment; the same internal plaintext
remained attackable in the trusted positive-control view.

Do not say the zero pattern ceased to exist, that all side channels were closed,
or that a TEE makes the representation private.

### Paper 3

Permitted claim after all gates: a scoped protected-computation candidate in the
new finite family met or failed utility, `G_F`, separately reported `G_B`,
operator-confidentiality/security and cost criteria. This is a counterexample,
failure or undetermined result under the specified trust model.

Do not rewrite the existing 19-cell R1 verdict, claim universal feasibility, or
call the Paper 4 anchor an exact Paper 3 replication.

### Paper 4

Permitted claim: a separate TEE/FHE deployment track met or failed its
same-checkpoint, protected-training, active-security and cost gates.

Do not relabel it D1/D3, reopen the completed pilot, or treat operator
confidentiality as a passing representation-privacy probe.

### LLM-Attacker

Permitted claim: the framework gained and exercised a real confidential-compute
profile with explicit surfaces, real adapters, controls and evidence-chain
outputs on one deployment.

Do not imply that registry presence validates every module, that inaccessible
attacks passed, or that the shared case study is an independent replication of
the other papers.

## Stop rules

Stop before paid performance if:

- hardware/firmware/topology is outside the admitted policy;
- attestation or workload/channel binding cannot be verified;
- an invalid quote or stale/revoked state releases a secret;
- debug/plaintext exports cannot be disabled;
- the exact external transcript cannot be enumerated; or
- positive controls cannot demonstrate that the attacker instruments work.

Stop before scientific training if:

- protected execution violates correctness tolerances;
- required active mutations fail open;
- a required plaintext/staging/log/checkpoint exposure remains;
- measured protected step cost exceeds the frozen campaign budget; or
- required views or attacks remain unqualified without an explicit
  `UNDETERMINED` claim plan.

Stop FHE/CURE before full training if:

- the full remote block/backward/update is unsupported;
- the required security mode breaks precision or feedback safety;
- projected time or memory exceeds the cap; or
- the implementation only benchmarks a linear projection.

Stop after the exploratory cohort if no candidate is eligible. Do not add seeds,
mechanisms or attack budgets post hoc to search for a publishable success.

## Cost request format

No price estimate is defensible until an eligible SKU and billing unit are
identified. Request a quote only after Stage 0 and calculate:

`total = billed CC-host units + CPU/verifier units + storage + egress`,

where billed host units include admission, setup, warm-up, failed attempts,
mitigation reruns, scientific jobs and teardown. If resources are independently
billed, also report summed device-hours; if the host is inseparable, use the
provider's real host billing unit rather than multiplying an advertised GPU-hour
rate.

The first request should fund Stage 2 only, capped provisionally at 4 aggregate
CC/GPU hours. Use its measured complete-step time to price the 12-, 18- or
24-job scientific cohort. Do not reserve the scientific allocation at the same
time.

## Immediate implementation order

1. Freeze whether `d0-q8` or `depth-2` is the sole anchor; default to `d0-q8`.
2. Write one cross-repository protocol ID and canonical evidence-bundle schema.
3. Implement the two-view recorder and real LLM-Attacker adapters.
4. Qualify adapters locally using vulnerable and clean fixtures.
5. If a materially different CURE-class design is credible, implement its
   bounded forward/backward/update feasibility gate; retain the current FHE
   measurements as inputs rather than rerunning them.
6. Freeze the CC engineering packet, hardware admission policy, mutations,
   repetitions, tolerances, stop rules and provisional cap.
7. Obtain an actual quote; request authorization for Stage 2 only.
8. After Stage 2, issue a measured cost and evidence review for the scientific
   matrix.
9. Freeze and run 12 jobs for protected feasibility, 18 for the arm-complete
   three-seed screen, or 24 for the preferred four-seed cross-paper exploratory
   design.
10. Decide on confirmation only after the exploratory verdict.

## Principal repository evidence used

- Paper 3 protected-computation plan:
  `artifacts/privacy-mechanisms-20260922/README.md`, especially lines 129-207.
- Paper 3 protected-computation manuscript section:
  `paper/sections/privacy_mechanisms.tex`, lines 10-112.
- Paper 3 R1 constraints: `AGENTS.md` and `docs/R1_CONFIRMATION_REVIEW.md`.
- Paper 4 deployment track: `docs/TEE_FHE_TRACK.md`.
- Paper 4 engineering and FHE decisions:
  `docs/TEE_FHE_EXEC_PROTOCOL.md`, `docs/TEE_FHE_FLEET_DECISION.md`,
  `docs/TEE_FHE_BATCHED_PROTOCOL.md`, and `docs/TEE_FHE_BATCHED_DECISION.md`.
- Paper 4 hardware/adversarial plans:
  `docs/TEE_FHE_HARDWARE_SPEC.md` and `docs/TEE_FHE_ADVERSARIAL_PROTOCOL.md`.
- Paper 4 scientific status: `docs/CLAIM_EVIDENCE_LEDGER.md` and
  `docs/PILOT_DECISION.md`.
- LLM-Attacker capabilities and limits: `README.md`, `attacker/surfaces.py`,
  `attacker/threat_model.py`, `attacker/attacks/protocol_integrity.py`,
  `attacker/attacks/key_session_abuse.py`, `attacker/attacks/wire_shape.py`,
  `attacker/attacks/mitm.py`, `attacker/attacks/malicious_server.py`, and
  `attacker/attacks/gpu_memory.py`.
- Returned-gradient paper:
  `/Users/geo/paper1_arxiv_submission/main.tex`, especially the declared-channel
  table and scope/limitations.
