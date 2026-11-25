# Claim-evidence ledger

Status: **Paper 4 design/P0 engineering; no scientific claim unlocked.**
This ledger replaces the initial scaffold's unsupported universal-impossibility and already
successful/failed defense language. Diagnostic outputs, if produced, establish implementation
behavior only. Pilot and confirmatory studies remain governed by their own recorded freezes.

The current Paper 4 quantizer changes per-tensor statistics to per-token min/max over the
latent dimension. Record that mechanism change in every new manifest/configuration. Fresh
D0 controls are required; equal forward precision is not grounds for pooling Paper 3 and
Paper 4 outcomes. D1 shared-warmup mechanics are implemented and checked; the scientific
warmup duration and basis-drift policy remain pending validation.
Paper 4's 80/10/5/5 training/calibration/probe-fit/evaluation partition also differs from
Paper 3. The validated configurations enumerate 10 engineering and 66 draft pilot runs;
these are configuration counts only, not completed scientific evidence.

## Engineering evidence recorded 2026-09-10

The 10-job `p4-p0-engineering-0.1` matrix completed with zero failed jobs at campaign
`p0-20260910-a`. Its [generated report](../outputs/runs/p0-20260910-a/report.md),
[verification record](../outputs/runs/p0-20260910-a/verification.json), and individual
`experiments/manifests/p4-p0-20260910-a-*.json` records establish execution and artifact
integrity only. Source tree hash:
`e143a79dc0874b146a62b1858df1fe20673b0ef3ccea77dccc132edf13b58aa0`.
The source snapshot includes uncommitted files; the repository had no HEAD at execution.
This is not a scientific freeze or externally registered cohort. All diagnostic probes
remain unqualified, privacy remains undetermined, and C1–C7 below remain unestablished.
The 79-test suite additionally checked implementation semantics; it is not research evidence.

## Prospective Paper 4 claims

| ID | Candidate claim | Required evidence | Current status / permitted language |
|---|---|---|---|
| C1 | D1 satisfies the joint gates at a specified cell | Frozen fresh-seed confirmation: mean Δ_U≤0.10, upper 95% limit≤0.12; mean G_R≥0.01 with corrected paired test; every required qualified attack/view worst seed≤0.02; full declared transcript and qualified active evaluation | Not established; “We test whether…” |
| C2 | D2 adversarial encoder yields an eligible cell | Separately specified minimax training, held-out evaluation attacks, same joint gates and active/transcript prerequisites | Subsequent extension; not in the initial pilot |
| C3 | An eligible cell exists at lower forward precision | Fixed other factors, qualified exact-q attacks, held-out joint-gate evaluation; no assumed monotonicity | Not established; precision is not full-transcript bitrate |
| C4 | No evaluated cell satisfies the joint criteria | Completed frozen finite matrix, all cell-level outcomes, operational failures and unqualified coverage disclosed; no missing score treated as zero | **Supported only as a finite empirical statement about the completed tested cells**: the frozen pilot matrix `pilot-20260922-a` contains no eligible cell; not a universal reading |
| C5 | Eligibility persists under a specified active attacker | Qualified protocol-manipulating attack with frozen budget, side information, integrity assumptions, matched controls, and training/utility consequences | Pending; stronger passive probes do not satisfy this requirement |
| C6 | A result replicates beyond the reference architecture/corpus | Fresh setting-specific cohorts, new qualification and complete outcomes under a frozen replication protocol | Pending; Paper 3 utility replication supplies no privacy evidence |
| C7 | D1+D3 improves outcomes relative to its components | Controlled interaction cells with matched D1-only, D3-only and reference runs; independent confirmation for a positive claim | Optional exploratory extension, not established |
| X1 | All representation-preserving boundaries face universal impossibility | Explicit task/adversary/communication model and a justified task-specific converse | Not a claim available from this empirical study |
| X2 | Leaked punctuation or other properties are necessary for useful remote work | An identified intervention or applicable formal result separating the claimed necessity from correlation | Not established by prior or planned finite sweeps |

Calibration floors are not acceptance gates. Punctuation is the primary sequence-property
target; vowel/newline targets remain exploratory until individually qualified. Next-byte
structure is byte-class prediction. No activation-only result unlocks a positive defense claim.

## Inherited evidence: cite, do not rerun

Authority: final Paper 3 ledger and manuscript at pinned revision
`52191c430110bd8d9dee80538fde22bf1726e56a`. Its current clarification takes precedence over
historical decision language. See [PRIOR_EVIDENCE.md](PRIOR_EVIDENCE.md) for the complete scope
and [INHERITED_ASSETS.md](INHERITED_ASSETS.md) for reusable interfaces.

| Inherited item | Evidence within Paper 3 | Permitted interpretation |
|---|---|---|
| Reference remote gain: 0.0167 nats, n=8, p≈8.35e−7 | `outputs/confirmatory/confirmatory_analysis.json`, `docs/P2_CONFIRMATORY_DECISION.md`, current ledger clarification, P2 manifests | Trained-versus-frozen gain in the specified cell, with width/span moderation; not remote necessity or superiority over removal |
| Replicated remote gain: 24 positive pairs, means 0.012–0.046 nats | `outputs/replication/replication_summary.md`, P3 protocol/manifests | Remote-gain replication, not broadened privacy replication |
| Four P2b candidates rejected under qualified forward attacks | `outputs/privacy_extended_p4b/privacy_extended_p4b.json`, P4b protocol and calibration/P2b manifests | Punctuation and next-byte structure evidence; secondary property maxima do not independently qualify vowel/newline |
| P5 finite sweep has no eligible tested privacy candidate; 288 runs | `outputs/delegation/delegation_analysis.json`, P5 protocol/manifests | Qualified forward property scores reject all candidates; do not claim both forward families fail every cell |
| P5 D1 Δ_U=0.82–1.56 nats | Same P5 sources | Relative to all-local for bundled precision/backward perturbation; not isolated noise cost or Paper 4 D1 evidence |
| Utility responds to fixed-width forward precision with a practical plateau above q=8 | `outputs/rate_sweep/rate_sweep_analysis.json`, P2b protocol/manifests | No validated complete-transcript rate or extrapolated monotonic law below q=4 |

Generated Paper 3 summaries are local ignored artifacts linked by its ledger to committed
protocols, source, configurations, and manifests. A current source snapshot is not independent
proof of historical pre-outcome registration. This ledger imports neither unverified venue
claims nor numerical results from the earlier decoy-system paper.

## Coverage carried forward

Until Paper 4 independently closes and records each gap, name timing; optimizer state beyond
the remote module; full evolving remote state; active/adaptive attackers; within-run joint
tensor attacks; cross-run joint attacks; natural-sequence membership; semantics beyond
byte-structure classes; returned values; and full-precision inbound/outbound gradients as
unevaluated. History-accumulation/canary-membership instruments, undefended gradient
property/token probes, and all defended gradient probes were unqualified in Paper 3.

A captured or declared channel is not an evaluated attack. Nondetection by a qualified probe
is probe-specific. Missing or unqualified scores are undetermined. No post-hoc maximum or
stronger passive evaluation silently expands qualified scope or satisfies the active requirement.

## Requirements for new entries

Before adding a manuscript result, record the exact Paper 4 code and pinned dependency
revision, environment, protocol/freeze/amendments, matrix hash, seed cohort, data/capture and
calibration hashes, target/channel qualification, selection rule, inferential unit, uncertainty
and multiplicity procedures, generated summary, and corresponding manifests. Distinguish
engineering diagnostics, exploratory pilot, and confirmation. Record failures and omissions
alongside successes. Actual matrix-derived counts replace planned counts only after validation.

`Deep-research.md` remains supplied background, not a source of completed experiments,
universal impossibility, or comparative failures of unimplemented defenses.

## External literature addition — 2026-09-10

Added `chen2026bifrost`, `tramer2019slalom`, and `volos2018graviton` to `paper/refs.bib`
and a scoped related-work paragraph to the manuscript. Source links and verification
boundaries are in [TEE-FTE.md](../TEE-FTE.md). Bifrost is recorded as arXiv v1;
Slalom and Graviton use verified ICLR/USENIX publication records. No external
performance number becomes a Paper 4 result. The [proposed deployment track](TEE_FHE_TRACK.md)
has no executed secure-deployment cohort or verified Bifrost implementation artifact,
and unlocks none of C1–C7.

## External literature addition — 2026-09-23

Imported from the Paper 3-era security review plus a fresh retrieval pass. Added to
`paper/refs.bib`: `asad2026benchmarking` (B200+TDX CC benchmark: ~1–3% tuned
inference overhead, 11–24% training-step overhead in its training arm),
`yin2026serialized` (Serialized Bridge: GPU-local compute ~0.998× but 13–27%
end-to-end serving loss and >2× KV-restore latency from the serialized VM↔GPU
bridge), `batan2026splitee` (TDX split inference with DP-masked intermediates),
`fan2026odin` (open GPU-CKKS Llama-3-8B, 366 s per 128-token forward),
`yu2026cachemir` (FHE LLM inference with KV cache), `chen2026veriattn`
(key `lee2026veriattn`), `hornetz2026tdxray` (host microarchitectural tracing of
TDX guests incl. prompt reconstruction), `teefail2025` (DDR5 interposition
extracts attestation keys incl. NVIDIA GPU CC), `kim2026shuffling` (permutation
"model confidentiality" broken in hybrid FHE), `li2020ckks` (CKKS
decryption-output key recovery), `chaturvedi2022fhewverification` (TFHE/FHEW
decryption-error feedback key recovery). All are references only; none is a
Paper 4 measurement, and no external performance number is adopted as one.

Wording discipline recorded for the manuscript: overhead ratios are valid only
scoped to their exact configuration (hardware, backend, layout, batching,
precision); our measured per-token CKKS/plaintext ratios apply to the recorded
stage-B/C/D configurations and are not a universal FHE constant. The three-way
distinction — representation privacy, operator confidentiality, output privacy —
is imported into [TEE_FHE_TRACK.md](TEE_FHE_TRACK.md); the adversarial evaluation
program (version/CVE mapping, attestation and key-release attacks, communications
attacks, applicable leakage/oracle tests with matched controls, performance after
mitigation, unassessed classes named) is drafted in
[TEE_FHE_ADVERSARIAL_PROTOCOL.md](TEE_FHE_ADVERSARIAL_PROTOCOL.md), unfrozen and
unexecuted. The deployment hardware spec gained the security-mapping and
measurement-discipline requirements in [TEE_FHE_HARDWARE_SPEC.md](TEE_FHE_HARDWARE_SPEC.md).

## Paper 1 backward-channel import — 2026-09-23

The Paper 1 demonstration (the returned gradient nullifies decoy rows:
exactly-zero gradients reveal real rows, 4,096/4,096 per run across nine seeds,
plus a token-content attack at +0.65 to +1.50 pp) is imported as the decisive
requirement for any training-stage deployment: protection must cover the backward
pass, parameter updates, and all equivalent disclosures — forward-only protection
repeats the Paper 1 mistake. Recorded in the training section of
[TEE_FHE_TRACK.md](TEE_FHE_TRACK.md) and as explicit arms of
[TEE_FHE_ADVERSARIAL_PROTOCOL.md](TEE_FHE_ADVERSARIAL_PROTOCOL.md). The FHE-side
precedent is verified as CURE (`kanpak2026cure`, PoPETS 2026(2) 259–276):
HE-encrypted server parameters, encrypted gradient return, homomorphic weight
updates under a semi-honest server — evidence that the approach exists, not a
replication of our system and not malicious-server coverage. Our measured cost
reality (`tee-fhe-batched-20260922-a`: ~775 ms/token for the remote's four
linears in per-token CPU CKKS) implies FHE-covered remote training at the pilot
scale is not viable with this backend/layout; the TEE-covered training path is
the realistic route, measured on our per-step boundary pattern per the Serialized
Bridge finding. The pilot's forward-only instrument coverage and its declared
gradient-channel omissions remain unchanged; the training stage, when specified,
must qualify its own backward-channel instruments (Paper 1's zero-pattern and
token-recovery attacks are the reference designs).

## Scenario engineering evidence — 2026-09-10

Execution used committed code `a8dff2126832f5fbb2048437c1e0af6a646082a9` and source tree
`4701c7aaccbe8083597fc6181a652d6e9ac1e7d49bda17378626199d303507ed`. All scenario manifests
retain these identities. Source archives and output hashes were independently checked
by `scripts/analyze_scenarios.py`, whose own hash is in each verification record.

| Engineering claim | Evidence | Limit |
|---|---|---|
| All 36 control scenarios completed | [Control report](../outputs/scenarios/controls-20260910-a/report.md), verification JSON and `p4-calibration-controls-20260910-a-*` manifests | Existing three-step P0 captures; no new training cohort |
| All 72 positive-dose floors passed; 17/36 distinct null checks failed | Same generated report; every per-target/view/dose result retained | Sensitivity diagnostics only; every instrument is still unqualified |
| All six routing cases matched logits with maximum error 0.0 and exact generated bytes | [Routing report](../outputs/scenarios/routing-20260910-a/report.md), verification JSON and `p4-routing-routing-20260910-a-*` manifests | Seeded untrained fixture, fp32 loopback, full-prefix recomputation; no secure hardware or cryptography |
| Implementation checks passed | 110 pytest cases in `tests/` at execution revision | Logic/correctness checks, not privacy qualification |

Operational completion is distinct from a calibration gate passing. The null failures
remain in the record and motivate larger independent panels and a justified control
domain before scientific freeze. They do not motivate retrospectively changing the
acceptance threshold. C1–C7 remain unestablished; no manuscript result is unlocked.

## Warmup/stability engineering evidence recorded 2026-09-10

Protocol: [WARMUP_STABILITY_PROTOCOL.md](WARMUP_STABILITY_PROTOCOL.md).
Execution commit `ba8b23d0e84a3a29392973fc81f413f3eb7846a5`, source hash
`4ea238153bc7fd5bee1eaec091b4342a7ce09c7b181c921b7d1260497f1809e3`.
The [report](../outputs/scenarios/warmup-20260910-a/report.md),
[verification](../outputs/scenarios/warmup-20260910-a/verification.json), and 22
`p4-warmup-warmup-20260910-a-*` manifests bind the results to source and artifacts.
See [the engineering assessment](WARMUP_STABILITY_DECISION.md) for follow-up decisions.

| Engineering finding | Evidence | Permitted interpretation |
|---|---|---|
| 22/22 records completed: two setups, twenty branches | Generated report and manifests; source/archive/artifact/pairing/panel checks passed | Operational execution, not scientific results |
| Frozen remote parameters and D1 bases stayed fixed; shared initial trusted weights matched | Branch/setup checksums; 122-test implementation suite at execution revision | Shared-warmup/fork mechanics work under the tested schedule |
| 95th percentile absolute null advantage fell from 0.6793 at 16 sequences to 0.1031 at 1,024 | All sixteen donor repetitions per size in eighteen split branches retained | Descriptive variability reduction; repetitions are dependent |
| 117/288 null checks within ±0.02 at 1,024; 171 failed | Same generated verification and per-branch stability artifacts | Tested panel enlargement does not establish null stability or qualification |
| Final direction cosine to warmed basis 0.7475--0.8659, versus 0.3504--0.4454 to initial basis | Report's eighteen branch rows and saved basis hashes | Warmed coordinates are a useful design starting point; ongoing drift remains |

Each model received sixteen common-warmup and sixteen branch updates. The removed
control is a post-warmup identity intervention. The all-local reference receives its
own warmup. These losses are not evidence of convergence or causal-gain eligibility.
All records remain `scientific_evidence: false`, `qualified: false`, privacy
`undetermined`. The omitted views and attacks are explicitly listed in the protocol,
generated report, and assessment. C1–C7 remain unestablished.

## Matched-null precision engineering evidence recorded 2026-09-10

Protocol: [NULL_PRECISION_PROTOCOL.md](NULL_PRECISION_PROTOCOL.md).
Execution revision `e29fd6376d0133c3340b26ad92c9715fe4682094`, source archive hash
`180f81bbb4d529f45f7d6f89f34240b45c4c0d470ce913e27f003afd421c2caa`.
The [report](../outputs/scenarios/precision-20260910-a/report.md),
[verification](../outputs/scenarios/precision-20260910-a/verification.json), and six
`p4-null_precision-precision-20260910-a-*` manifests bind control-only outcomes to
the two saved common warmup checkpoints. See [the assessment](NULL_PRECISION_DECISION.md).

| Engineering finding | Evidence | Permitted interpretation |
|---|---|---|
| Six scenarios completed, source/artifacts/estimates verified; 137 tests passed | Generated verification and tests at execution revision | Correctness and execution, not instrument qualification |
| Zero-dose releases match exactly; all codebook checks passed, max normalized rounding error 0.00003052 | Per-control support records and verification | Valid quantizer support for synthetic forward controls; metadata not attacked |
| All 72 positive-score checks exceeded 0.10 | Report's six estimator/panel-size rows, 12 scores each | Sensitivity under these controls only; dependent repeated checks |
| At 1,024 sequences and K=64, 107/192 within ±0.02, 85 outside, 37 above +0.02 | Raw no-signal trials and generated report | Conditional donor-score variability remains; not a population error rate |
| At 1,024, SD of mean null correlation falls from 0.0327 (K=1) to 0.0040 (K=64), while fitted-probe r SD is 0.0305 | Generated variance-component table | Averaging inner nulls leaves fitted-probe variability in this procedure |

No normal candidate was scored and no model was trained in this stage. D1/D3
were diagnostic release conditions on a common checkpoint warmed under D0 q8,
not independently trained defense cells. Privacy remains undetermined and all
instruments remain unqualified. The report and assessment explicitly name omitted
views and attacks. C1–C7 remain unestablished.

The assessment's text-capacity arithmetic derives from the source setup's
`panels.json` declarations: each probe region has 55,770 bytes; a disjoint window
uses sequence length plus one byte. Proposed 8,192/16,384-window budgets are data
allocation scenarios, not empirical sample-size guarantees or scientific results.

## Calibration text resource recorded 2026-09-18

Protocol: [CALIBRATION_TEXT_PROTOCOL.md](CALIBRATION_TEXT_PROTOCOL.md), assembled by
`scripts/fetch_gutenberg.py` at commit `076cd1b`. The 75-document, 59,619,533-byte
public-domain Project Gutenberg resource (`gutenberg_p4`, SHA-256
`af3d0245c0ec27d7021f5fd29256bc845bb20cf9c562b37bbe23663b6ebe7436`) is partitioned
by author-disjoint document assignment (train 35.8MB, calibration 9.0MB,
probe-fit 7.5MB, evaluation 7.3MB), with per-document digests in the committed
`data/gutenberg_p4.manifest.json`. Sampled overlap with the historical corpus:
0/512 distinct 32-byte n-grams. This resolves the panel-capacity prerequisite
(≈56–58k disjoint windows per probe region at sequence length 128) and supplies
the new-corpus prerequisite named by the precision assessment. It allocates data
only: no instrument is qualified and no scientific run is authorized by it.

## Warmup-at-budget engineering evidence recorded 2026-09-18

Protocol: [WARMUP_BUDGET_PROTOCOL.md](WARMUP_BUDGET_PROTOCOL.md). Execution commits
`076cd1b`/`0bf707f`, source tree hash
`cff937493ff02de3…` (full value in each manifest). Campaign
`warmup-budget-20260918-a`: sixteen `p4-warmup_budget-*` manifests (two recorded
per-seed execution subsets, seeds 410501/410502, CUDA on toula/koula), all
re-verified after merge (schema, per-file source archive, artifact digests).
See [the engineering assessment](WARMUP_BUDGET_DECISION.md).

| Engineering finding | Evidence | Permitted interpretation |
|---|---|---|
| 16/16 scenarios completed at seq 128, 500+1,500 steps, full integrity checks | Manifests, per-branch metrics | Warmup/fork mechanics valid at the pilot operating point |
| Single-null p95 fell to 0.018–0.026 at 16,384 sequences (14–16/16 in band) vs 0.073–0.108 at 1,024 | Per-branch stability.json, 16 donor repetitions per size | Large panels markedly reduce null variance; matched-null measurement at 8,192/16,384 is the separate precision campaign |
| G_R-like +0.0075/+0.0088; removed-remote ahead of trained by 0.006–0.008; Δ_U-like +0.065/+0.076 (d0) | Fixed-panel final losses, two seeds | At this budget the gain-gate margin is absent in engineering; motivates the budget scan, not a pilot outcome |
| Warm basis vs post-training refit cosine 0.256–0.362 | Post-training basis refits, four trained branches × two seeds | Fixed warm basis is stale at budget end; freeze must choose a controlled basis policy |
| d3 q=1 fork rebuild valid; eval +0.018/+0.036 over d0 | Release-support records, branch losses | 1-bit release is codebook-valid; utility margin thin/adverse in one seed |

All records remain `scientific_evidence: false`, `qualified: false`, privacy
`undetermined`. Unevaluated: next-byte classes, returned values, quantizer
metadata, gradients, remote/optimizer state, timing, active/adaptive attackers,
joint/cross-run attacks, history, membership, semantics. C1–C7 remain
unestablished.

## Matched-null precision at the pilot operating point — 2026-09-22

Protocol: [NULL_PRECISION_128_PROTOCOL.md](NULL_PRECISION_128_PROTOCOL.md).
Execution commit `f14a014`; campaign `precision-128-20260918-a`: eight
`p4-null_precision_128-precision-128-20260918-a-*` manifests in two recorded
per-seed execution subsets (410601 on toula, 410602 on koula, CUDA deterministic
kernels). Analyzer [report](../outputs/scenarios/precision-128-20260918-a/report.md)
and [verification](../outputs/scenarios/precision-128-20260918-a/verification.json);
every manifest re-verified after merge. See
[the engineering assessment](NULL_PRECISION_128_DECISION.md).

| Engineering finding | Evidence | Permitted interpretation |
|---|---|---|
| Replayed d0-trained control models match bound branch checksums bit for bit on both nodes | Replay-verification blocks in both prepare records | Chained control-model binding to full-budget trained models is exact |
| 63–64/64 no-signal trials inside ±0.02 at 16,384 sequences (all three release conditions, K 16/64; SD 0.007–0.008; p95 ≤ 0.017) | Verification's pooled control table, 64 dependent trials per row | Control-only precision requirement met at the pilot operating point; not instrument qualification |
| 57–61/64 inside band at 8,192 sequences | Same table | 8,192 is the documented insufficient panel size here |
| All nonzero-dose positive controls ≈ +1.0 vs the 0.10 floor; zero-dose exact; support error ≤ 0.000031 | Per-control support and score records | Sensitivity under these controls only; valid codebook support |

All records remain `scientific_evidence: false`, `qualified: false`, privacy
`undetermined`. No normal candidate was scored. Unevaluated: next-byte classes
(pending the structure campaign), returned values, quantizer metadata, gradients,
remote/optimizer state, timing, active/adaptive attackers, joint/cross-run
attacks, history, membership, semantics. C1–C7 remain unestablished.

## Budget response, warmup policy, and structure precision — 2026-09-22

Protocols: [BUDGET_SCAN_PROTOCOL.md](BUDGET_SCAN_PROTOCOL.md),
[STRUCTURE_PRECISION_PROTOCOL.md](STRUCTURE_PRECISION_PROTOCOL.md), plus the
no-warmup run-matrix checks (`p4-p0-nowarmup-b{2000,8000}-0.1`). Execution
commits `1a9ee07`/`675ac6a`; campaigns `budget-scan-20260918-a` (12 manifests,
12/12 verified), `structure-precision-20260922-a` (6 manifests, 6/6 verified),
and `nowarmup-{b2000,b8000}-20260922-a` (8 run manifests each, verified).
Assessments: [BUDGET_SCAN_DECISION.md](BUDGET_SCAN_DECISION.md) (including
disclosed operational incidents and bitwise re-execution checks) and
[STRUCTURE_PRECISION_DECISION.md](STRUCTURE_PRECISION_DECISION.md).

| Engineering finding | Evidence | Permitted interpretation |
|---|---|---|
| G_R-like at the anchor: +0.0185/+0.0126 (2,000 steps, no warmup); ≤+0.0103 and as low as +0.0023 at longer budgets or with a 500-step shared warmup | Budget-response table across ten regime/seed rows | The 2,000-step no-warmup regime is the only measured point where both seeds clear the proposed gain gate; freeze input, not a pilot result |
| Fixed warm basis drifts to \|cos\| 0.26–0.36 by budget end; init basis near-orthogonal | Warmup-budget refit cosines | Motivates a scheduled D1 basis refit instead of a fixed or warmup basis |
| Structure nulls exactly zero at 8,192/16,384 (192/192 trials); injected-code advantage ≈ +0.254 vs 0.02 floor; natural next-byte advantage ≈ +0.044 at the boundary (side check) | Structure verification table; assessment note | Structure estimator controls pass on control models; natural-signal sensitivity confirmed informally; not qualification |
| Re-executions reproduce first-run losses and remote checksums bitwise on both nodes | Incident disclosure in the scan assessment | Same-device deterministic replay is exact; provenance chain intact after incidents |

All records remain `scientific_evidence: false`, `qualified: false`, privacy
`undetermined`. Unevaluated: returned values, quantizer metadata, gradients,
remote/optimizer state, timing, active/adaptive attackers, joint/cross-run
attacks, history, membership, semantics beyond five-class byte structure, token
inversion. C1–C7 remain unestablished.

## Calibration-cohort instrument qualification — 2026-09-22

Protocol: [QUALIFICATION_PROTOCOL.md](QUALIFICATION_PROTOCOL.md). Execution commits
`bc1f1d2`/`a29d15a`; campaign `qualification-20260922-a`: 32 `p4-qualification-*`
run manifests (cohort **calibration**, `scientific_evidence: true` scoped to
instrument qualification; seeds 419101–419104; toula/koula CUDA). Analyzer
[report](../outputs/runs/qualification-20260922-a/report.md) and
[verification](../outputs/runs/qualification-20260922-a/verification.json);
all manifests re-verified after merge; all 28 split runs' candidate scores
recomputed from saved captures. See
[the decision record](QUALIFICATION_DECISION.md).

| Qualification outcome | Evidence | Scope |
|---|---|---|
| Next-byte structure instrument **qualified** on all 7 pilot cells | Per-binding null/positive checks, four seeds each | Instrument qualification only; not a privacy verdict on candidates |
| Punctuation instrument **qualified** on 5 cells; not qualified on d0-q8 and d1-s1.0-q8 (one marginal null excursion each: max 0.0217 and 0.0245) | Same checks; disclosed excursion rates | Strict frozen rule retained; reliability evidence disclosed to the freeze |
| Calibration-cohort candidate advantages: punctuation 0.76–0.90, structure 0.09–0.12, on every tested cell; d0 reference punctuation 0.85–0.90 | Recomputed candidate table, 16,384-window panels | Exploratory calibration observations; the pilot is the pre-registered test |
| D1 σ=1.0 reduces punctuation advantage by ~0.03–0.07 vs d0; D3 does not reduce it; D3 Δ_U-like reaches 0.087–0.120 | Same table, four calibration seeds | Descriptive only; no cell selection |

No defense-success, utility, causal-gain, or pilot-eligibility claim follows.
Unevaluated: returned values, quantizer metadata, gradients, remote/optimizer
state, timing, active/adaptive attackers, joint/cross-run attacks, history,
membership, semantics beyond five-class byte structure, token inversion. C1–C7
remain unestablished.

## Exploratory pilot p4-defense-pilot-1.0 — 2026-09-22

Protocol and freeze: [PILOT_FREEZE.md](PILOT_FREEZE.md), freeze record
`experiments/freeze/p4-defense-pilot-1.0.json` effective at commit `764d3f2`.
Campaign `pilot-20260922-a`: 66 `p4-pilot-*` run manifests (cohort
**exploratory**, `scientific_evidence: true`; seeds 411101–411103; toula/koula
CUDA deterministic kernels), analyzer [report](../outputs/runs/pilot-20260922-a/report.md)
and [verification](../outputs/runs/pilot-20260922-a/verification.json). All 66
runs completed; manifests re-verified after merge; captures hashes verified;
all 63 split-run candidate scores recomputed exactly. Independent CPU-node
verification: 236 manifests schema-valid (soula), freeze digests and pilot seed
audit OK (voula), 198 small-artifact hashes verified (sasa). Decision record:
[PILOT_DECISION.md](PILOT_DECISION.md).

| Pilot outcome (frozen gates) | Evidence | Permitted interpretation |
|---|---|---|
| **No cell pilot-eligible** (0/7) | Gate table: utility, gain, privacy per cell over 3 paired seeds | Finite empirical statement about the completed tested cells only |
| Utility: pass d0/D1 all strengths/d3-q3 (mean Δ_U 0.062–0.094); fail d3-q1 (0.109) and d3-q2 (0.109, upper 0.123) | Per-seed Δ_U with frozen t-interval | Same-cell utility evaluation at the frozen point |
| Causal gain: pass all 7 cells (mean G_R 0.013–0.037; every pair positive) | Per-seed paired trained/frozen losses | Remote gain at the frozen point; not remote necessity |
| Privacy: fail or undetermined on all 7 cells; punctuation worst-seed 0.79–0.89 where qualified, structure worst-seed 0.097–0.112 | Worst-seed candidate advantages on qualified bindings, 16,384-window panels | Qualified forward-view privacy evaluation rejects every tested cell; named omissions remain unevaluated |
| D1 σ=1.0 lowers punctuation advantage by ~0.03–0.08 vs d0; D3 does not lower it at any tested q | Per-cell advantage table | Descriptive relative comparison; not a privacy pass |

The hypothesis that the tested D1/D3 family contains an eligible cell at this
operating point is **not supported** by the exploratory pilot; the finite
alternative (no eligible tested cell, H4 in the research plan's finite sense) is
supported within the completed frozen matrix. This is not a universal
impossibility claim (X1) and says nothing about property necessity (X2). C1, C3:
not established. C4: supported only as "no eligible cell in the completed tested
matrix". Confirmatory cohorts 412101–412108 remain unused; they exist for a
future *eligible* cell, of which there are none. Follow-ups requiring new
prospective protocols: D2 adversarial encoder (C2), D1+D3 interaction cells
(C7), active-attacker evaluation (C5), replication beyond this architecture and
corpus (C6), and the complete-transcript/active prerequisites for any positive
claim.

## TEE–FHE deployment track, fleet-side stages — 2026-09-22

Track: [TEE_FHE_TRACK.md](TEE_FHE_TRACK.md). Fleet-side protocol
[TEE_FHE_EXEC_PROTOCOL.md](TEE_FHE_EXEC_PROTOCOL.md); hardware procurement spec
[TEE_FHE_HARDWARE_SPEC.md](TEE_FHE_HARDWARE_SPEC.md). Milestone 1 recorded: no
attributable Bifrost artifact located; all fleet work is independently inspired,
not a reproduction. Hardware finding: the fleet has no CC-capable GPU (GB10s
report no confidential compute and 0 MiB protected memory), so every fleet
record carries `hardware_attested: false`. Campaign `tee-fhe-20260922-a` (two
`p4-tee_fhe-*` manifests, toula, `.venv-tee` + TenSEAL 0.3.18), all re-verified.
Assessment: [TEE_FHE_FLEET_DECISION.md](TEE_FHE_FLEET_DECISION.md). Every record
is engineering (`scientific_evidence: false`), privacy `undetermined`.

| Fleet-side finding | Evidence | Permitted interpretation |
|---|---|---|
| Same-checkpoint emulated split is operator-correct: max \|Δlogit\| 0.0, loss Δ 0.0, 100% token agreement on the frozen 512-window prefix | Stage A metrics + verified stage_a block | Split routing mechanics work; not a security property |
| Observation contract: 5.64 GB serialized boundary traffic for 512 windows under an all-linears loopback policy | Contract events and totals, hash-bound | Traffic inventory the attested stages must match; a selective policy should shrink it |
| Real CKKS per-token projection (256→1024, poly 8192, scale 2^40): ~117 ms/token homomorphic vs ~13 µs/token batched plaintext (≈10⁴×), 331 KB ciphertext/token (≈330×), error ≤ 7e-7 | Stage B five-repetition records on the GB10 CPU (TenSEAL 0.3.18) | Real FHE cost at the boundary scale; slot-batching (~30×) and selective policies are the measured next levers; not a secure-serving latency |

No confidentiality, integrity, or attestation claim follows from any fleet
record. Milestone 3's TEE half and milestone 4 (serving matrix) require the
CC-capable instance; the procurement/arrival-verification spec is in
[TEE_FHE_HARDWARE_SPEC.md](TEE_FHE_HARDWARE_SPEC.md).

## TEE–FHE batched corrective and selective policy — 2026-09-22

Protocol [TEE_FHE_BATCHED_PROTOCOL.md](TEE_FHE_BATCHED_PROTOCOL.md); campaign
`tee-fhe-batched-20260922-a` (two `p4-tee_fhe-*` manifests, toula, `.venv-tee`),
all re-verified; assessment [TEE_FHE_BATCHED_DECISION.md](TEE_FHE_BATCHED_DECISION.md).

| Fleet-side finding | Evidence | Permitted interpretation |
|---|---|---|
| Slot-batching gives no asymptotic win: packed/per-token ratios 0.726/0.860/1.142 at block counts 1/2/4; ~118 ms/token is the floor (keygen amortizes to ~2.8 ms/token) | Stage C five-repetition records, block-diagonal packed form verified vs fp32 | An earlier ~30× packing estimate is measured wrong; op counts scale with output dimension |
| Selective policies cost ~15 s (one encrypted linear) to ~102 s (all four remote linears) per 128-token sequence; traffic is small (0.66–2.1 MB) | Stage D per-shape costs and policy table against the verified stage-A contract | No fleet-executable FHE policy is interactive-conceivable at this scale; bottleneck is homomorphic time, not transfer |
| GPU CKKS has no pip-installable aarch64 backend today | Probe on toula (phantom-fhe, libfhe unavailable) | GPU-CKKS is a separate source-build project; literature GPU numbers stand as the reference |

Same boundaries as before: engineering only, `hardware_attested: false`, privacy
`undetermined`. The deployment track's open paths are the CC-capable instance
(attested TEE baseline), a GPU-CKKS source build, or Odin-class layout
co-design — each a separate prospective protocol.
