# Evidence and execution ledger

Status date: 2026-08-14. This ledger controls how existing results are used while
corrected and extension experiments continue. It does not delete historical
artifacts and it does not treat every rerun as invalidating prior evidence.

## Evidence states

| State | Meaning | May support a headline? | Aggregation rule |
|---|---|---|---|
| **Confirmatory** | Correct protocol, declared replication unit, adequate repetitions, uncertainty, controls, and provenance | Yes, within the tested scope | Pool only exchangeable repetitions from the same implementation and measurement path |
| **Supporting** | Valid observation with limited scale, incomplete provenance, incomplete controls, or an explicitly documented confound | Only as qualified supporting evidence | Show separately; never silently pool with confirmatory evidence |
| **Supporting with known limitation** | Valid observation whose artifact and collection manifest name a specific limitation | Only with the limitation visible | Never silently pool with clean supporting or confirmatory evidence |
| **Superseded metric** | Raw observations remain useful, but a derived threshold, uncertainty estimate, or interpretation is defective | No for the affected metric | Preserve the raw artifact and point to the corrected derivative or rerun |
| **Fallback baseline** | Valid result from a slower or alternative implementation path | Yes, but only as that named implementation | Never pool fallback and optimized-kernel timings |
| **Pending** | Required evidence has not yet been produced | No | Launch only after dependencies and peer health checks pass |
| **Diagnostic** | Focused trace or partial result retained to explain a failure | No | Keep under `collected/diagnostic`; never aggregate into experiment results |
| **Invalid** | A correctness, identity, provenance, or protocol gate failed | No | Keep under `collected/invalid`; exclude all affected metrics from claims |

Every new result should record `evidence_status`, `measurement_kind`,
`supersedes`, `superseded_by`, and `known_limitations` in its manifest entry.
These fields are additive metadata; historical JSON files should remain
immutable unless their schema explicitly permits a derived addendum.

## Paper 1 evidence disposition

| Track | Existing evidence and result | Current state | What remains | Effect on old evidence | Issue |
|---|---|---|---|---|---|
| 7B/12B/27B WAN inference | RTT sweeps, identity, and transport artifacts; 7B lookahead gains are positive on the emulated profiles | **Supporting**, pending prompt-level intervals and final allow-list | Aggregate paired repetitions with 95% intervals and provenance audit | Retain all valid cells; downgrade only cells that fail the stop/go gate | — |
| 35B-A3B inference | Three repetitions across 0/20/80/hostile plus three identity checks; sequential arm is stable, but lookahead is non-identical in 3/3 checks with max logit delta 38.617 and pathological perplexity | **Sequential supporting; lookahead invalid** | Preserve sequential scale baselines; diagnose and correct lookahead before rerunning only that arm | Do not pool or cite current lookahead throughput; smaller-model evidence is unchanged | — |
| Real-WAN validation | Existing real-WAN run changes both network and cloud host | **Supporting/confounded** | Same-host real-WAN versus netem comparison, retransmit counters, counterbalanced arm order | Preserve as exploratory deployment evidence; exclude from causal netem-validation claim | #42 |
| Split training systems | Existing sync/overlap/FedAvg timings and corrected E-A4 repetitions | **Supporting** | Aggregate uncertainty; add deterministic equality self-test | Throughput remains usable; self-test adds semantic evidence | #45 |
| Trained inversion | Multi-model/depth attack artifacts | **Supporting to confirmatory by cell** | Allow-list cells with document-disjoint victims, utility, and complete provenance | Retain passing cells; do not generalize beyond tested models | — |
| Sequence-aware inversion | Corrected joint attacker completed at depths 1/4/8 with three seeds; it underperforms the position-wise MLP and gains only +0.25/+0.22/+0.07 pp from LM rescoring | **Confirmatory null within 0.6B scope** | Stop at 0.6B under the predeclared gate; do not spend a 7B run on this hypothesis | Existing non-joint result remains the comparator; joint result closes #40 empirically | #40 |
| SipIt | 0.6B exact-inversion result plus completed 7B depths 1/4/8 × bf16/fp32 × 10-document scale validation; recovery ranges from 77.03% to 84.53% | **Supporting scale evidence; effective n=10 documents per cell** | Do not count the three byte-identical seed labels as independent repetitions; use new victim sets or a genuinely stochastic attacker before making repeatability/CI claims | 0.6B evidence remains valid within scope; 7B extends scale but not independent-seed uncertainty | #41 |
| DLG++ | 90 cells: 10 documents × depths 1/4/8 × 3 attacker seeds; low recovery and near-zero gradient cosine | **Supporting** | Derive chance/floor annotations and clustered intervals locally; add victim-drift axis if claimed | Raw 90 cells remain valid; interpretation is limited to optimizer difficulty at the tested step-0 condition | #43 |
| E-A4 membership/property | Three corrected split-FT and FedAvg repetitions; strong ROC-AUC but zero TPR at 1% FPR | **Confirmatory within current dataset** | No rerun unless scope changes; keep ranking and low-FPR conclusions distinct | Preserve and do not rerun | — |
| E8 passive defense | Large reduction against fixed-key passive trained attacker; leaked-key control restores baseline. The 27B bf16 battery extends this pattern across depths 1/4/8: label-free recovery is at most 3.71%, and dense-grid K50 is approximately 258/356/618 labeled pairs at depths 1/4/8 | **Supporting to confirmatory by cell** | Pair each headline cell with utility/correctness and provenance checks; add fp32 only if precision sensitivity is claimed | Retain passing cells; dense K evidence supplies the guarded crossing estimates | — |
| E8 D3 oracle repeatability | Accepted depth-1, K=100 artifact: 3 independent probe selections/solves × 3 nested decoder seeds; top-1 25.43%, between-probe SD 3.93 pp, within-decoder SD 2.26 pp | **Confirmatory within the declared probe pool** | No rerun unless the probe pool, model, depth, or threat model changes | Existing oracle curve remains supporting; old decoder-only error bars at K=100 are superseded by the nested variance result | — |
| E8 known-prefix | Corrected depth-1 run uses three independent pair resamples/solves with three nested decoder seeds and guarded interpolation; K90 crossings are about 234/232/240 for prefixes 32/64/128, while prefix 256 is left-censored at K<=256 | **Confirmatory within depth-1/0.6B scope** | Add deeper/model-scale cells only if those claims remain in the manuscript | Preserve raw curves; corrected crossings and variance supersede old derived headlines | #34 |
| E8 deployment rotation | Three matched sessions per arm: rotation has negligible throughput difference, preserves the same identity controls, keeps obfuscated wire recovery at 0%, and uses three distinct key hashes | **Confirmatory within Mistral-7B deployment scope** | No rerun unless model, rotation cadence, or threat model changes; key material remains excluded | Fixed-key historical results remain valid; matched fixed arm controls harness drift | #44 |
| E-R ratcheted per-epoch rotation (deployed) | Eight completed runs on Mistral-NeMo-12B: request-scoped cadence N=16/64/256 and an evidence-triggered budget arm, each at 0 and 80 ms with three matched sessions. Identity correct in 20/24 sessions; epochs advanced as designed (N=16 ~9-10/session, N=64 2, N=256 0 as the no-rotation control, budget exactly 1 per run); per-epoch wire attack on rotated epochs 0% top-1; ratchet throughput 2.8 vs 7.6 tok/s static at N=16, ~5.5-5.6 for the budget arm | **Supporting; 4 of 8 runs supporting_with_known_limitation** | Lab rotation-lifetime suite (E-R1a/3/4/5) and E-R6 fold running; pair with the lab per-epoch budget curve before any headline claim | Four sessions (`n16` sess2 and `budget` sess1 across both RTT twins) show exactly one flipped token at the same position in both RTT twins — deterministic bf16 borderline-logit rounding across the W round-trip, not a ratchet race; flagged in `paper-data/collected/toula/er_20260811/COLLECTION_MANIFEST.json` | — |
| E-R3 fixed versus jittered scaffold (27B laboratory) | Koula completed depths 1/4/8, three seeds, and 16 sessions per arm. A fixed 64-token scaffold reached 90% of baseline after 14.33/14.50/14.67 sessions on average at depths 1/4/8; the jittered arm had no K90 crossing and ended at 0.80/0.76/0.00% top-1 | **Primary within the declared oracle-label and jitter model** | Add approximate-alignment attacker before generalizing jitter beyond pad insertion and slot permutation | Establishes that repeated fixed scaffolds exhaust a static key, while the tested jitter disrupts exact label accumulation | — |
| E-R4 stale-key transfer (27B laboratory) | Koula completed depths 1/4/8 with three seeds. Stale-only best recovery was 1.98/2.45/0.94% top-1 and pairwise key error was 1.4141; fresh pairs at E=1024 restored 65.31/63.02/61.20% | **Primary within the declared stale-key attacker** | None unless the attacker gains cross-epoch alignment or side information | Stale keys give no useful bootstrap across independently rotated epochs; fresh same-epoch pairs remain the binding risk | — |
| E-R5 static versus staggered sharding (27B laboratory) | Koula completed depths 1/4/8, block counts 1/4/16, and three seeds. Static sharding retained high composed recovery (56.98-67.86%); staggered schedules ranged 47.58-64.51% and did not approach the label-free band | **Primary negative/control evidence; not a successful defense claim** | Revisit schedule/cadence before deployment; do not launch static sharding as a privacy defense | Confirms that sharding alone shrinks per-block solve budgets and does not solve accumulation; the tested staggering is insufficient for a headline privacy claim | — |
| E-R5 static versus staggered sharding (12B laboratory) | Toula completed depths 1/4/8, block counts 1/4/16, and three seeds (27 cells). Static recovery was 58.54-65.68%; staggered recovery was 54.95-65.13%; the label-free band was 0.16-0.89% | **Primary negative/control replication; not a successful defense claim** | Reject static sharding and partial/staggered rotation; use dense full-width atomic rotation for follow-up designs | Independently reproduces the 27B failure pattern at 12B scale; collected under `paper-data/collected/toula/er5_12b_20260812/` | — |
| E9 deployment | Three audited paired sessions contain baseline, local-only, shared-key, and aligned wire-attack artifacts; local-only breaks identity and shared-key restores it in all three | **Supporting with repeatability** | No fixed-key rerun; use the paired baseline as the comparator for deployed rotation | Preserve all three valid sessions | — |
| E-R9 full privacy-stack smoke | Toula/Koula completed matched plain/full1 training at direct RTT. Full1 eval loss was 2.596 versus 4.258 plain, but known-plaintext recovery was 21.52% mean per epoch and 39.06% pooled versus a 9.22% label-free band; the required E-R8 alignment attacker then crashed before producing evidence | **Diagnostic / failed validation** | Fix the real-capture alignment driver, strengthen the privacy gate, and rerun the two-cell smoke before launching the matrix | Preserve safe aggregate JSON/logs under `paper-data/collected/diagnostic/toula/er9_smoke_20260813/`; exclude timings and privacy values from accepted claims | #90–#95 follow-up |
| E-R9 corrected-budget smoke | Toula/Koula completed plain/full1 at direct RTT on `c5dfe00`, including known-plaintext and E-R8 alignment attacks. Full1 eval loss was 2.526 versus 4.258 plain; recovery was 21.74% mean per epoch, 29.69% pooled, and 15.21% approximate alignment versus a 9.71% label-free band. No E-R8 epoch produced an accepted solved outcome. Correct B=64 accounting caused 164 dense folds at roughly 205–215 seconds each | **Diagnostic / completed, privacy target not met** | Optimize folding before another matrix; strengthen the defense against partial recovery; retain the zero-solved result as a narrow structural gate rather than a privacy claim | Safe evidence is under `paper-data/collected/diagnostic/toula/er9_smoke_main_20260814/`; raw captures, seeds, checkpoints, tensors, and private corpora are excluded | #100 |
| Qwen3-0.6B v2 strict-budget PoC | Toula/Koula completed a one-step plain/protected split-training diagnostic at depth 8. Protected label-free recovery fell to 1.56% from 40.62% plain, but known-plaintext recovery remained 33.79% mean and 93.75% worst per epoch; a corrected disjoint K3-512 pooled solve recovered 21.88%. Strict B=64 accounting served 1,216 rows with zero oversized forwards, and local-update DP-SGD reported epsilon 10.15 at delta 1e-6. Protected/plain held-out loss was 7.309/3.634 and step time was 91.76/2.87 seconds, but the corpus variants differ, so the utility delta is diagnostic rather than causal | **Diagnostic / completed, privacy target not met** | Count both forward and return observations in the adversarial budget; prevent within-epoch low-rank/repeated-template recovery; add kernel-optimized folding; repeat only after the mechanism changes | Safe aggregates and provenance are under `paper-data/collected/diagnostic/toula/poc06b_v2_20260814/`; all raw captures and seeds were removed | — |
| Qwen3-0.6B v3 adaptive-budget PoC | Toula/Koula completed a matched one-step plain/protected split-training diagnostic with trusted layers 0-15 and 26-27. Four-direction B=256 rotation produced 16 training epochs. Protected pooled recovery stayed at the label-free floor: 3.12% at 8 and 16 rows/epoch (128/256 total) and 0% at 32 rows/epoch (512 total), while the plain combined-direction control recovered 42.97% at 32 rows. The emitted boundary-DP epsilon 2.972 is **superseded** because it counted sequences rather than exposed token rows | **Diagnostic / privacy pass, utility and cost fail; DP metric superseded** | Rerun with corrected token-row adjacency; tune clipping/noise and measure convergence; revalidate the fast structured transform before any production claim | Protected/plain held-out loss was 16.508/3.688 and step time was 60.74/2.66 s. The cloud-held seed means this is capture resistance, not compromised-host secrecy. Safe evidence is under `paper-data/collected/diagnostic/toula/poc06b_v3_20260814T2115/` | — |
| Qwen3-0.6B v4 structured B=256 PoC | The fast signed-permuted Hadamard wire sandwich passed cost and utility: 3.563 s versus 2.657 s plain (1.34x), held-out loss 3.963 versus 3.688 (delta +0.275), and zero pending cloud graphs. It failed adaptive privacy: pooled recovery was 18.75/21.88/21.88% at 8/16/32 solve rows per epoch (128/256/512 total), versus 1.17-1.76% label-free | **Diagnostic / rejected privacy candidate** | Do not promote the structured transform. Preserve the corrected lifecycle and token-level accountant; investigate dense-transform caching or a fused dense fold that retains the v3 attack result | Safe aggregates are under `paper-data/collected/diagnostic/toula/poc06b_v4_b256_20260814T2145/`; raw captures and seeds were removed | — |
| Qwen3-0.6B v4 dense-sandwich B=256 PoC | Dense sandwich retained utility (held-out loss 3.772 versus 3.688 plain; delta +0.084) and removed weight refolding, but 17 portable QR derivations still cost 21.31 s (8.02x plain). Rotation alone failed privacy: pooled recovery was 21.88/21.88/18.75% at 8/16/32 rows per epoch versus 5.47-6.25% label-free | **Diagnostic / rejected privacy and cost candidate** | Do not increase B: the lowest tested accumulation already fails. The v3 pass depended on destructive DP noise; search for a learned information bottleneck or formally corrected token-DP frontier rather than rotation alone | Safe aggregates are under `paper-data/collected/diagnostic/toula/poc06b_v4_dense_b256_20260814T2200/`; raw captures and seeds were removed | — |
| Qwen3-0.6B latent-native v5 | The trusted residual-correction pilot restored utility, then a real Toula/Koula D-only protocol completed 2,000 remote training steps. Remote loss was 4.681 versus 5.031 baseline (delta -0.350), time ratio 2.709, and the zero-cloud control was 0.868 loss worse. All 6,128 temporary captures were width 128 and Koula had no H=1024 parameter/frame. Privacy still failed: recovery was 5.566% versus a 3.857% band (+1.709 pp). Low-noise DP epsilon remained in the millions | **Diagnostic / two-node utility and topology pass; privacy not accepted** | Implement real-capture ICA/subspace/active drivers in `/attacker`, score without giving labels to Koula, and run three remote seeds. Do not claim formal DP or malicious-computation integrity | Safe aggregates are under `paper-data/collected/diagnostic/latent_v5_{06b,remote,zero}_*.json`; temporary raw captures were audited then deleted, never collected | — |
| Provenance | Recent privacy artifacts have richer provenance; historical corpus remains sparse | **Supporting with declared limitation** | Add emitter metadata and CI enforcement; publish historical coverage | Do not rerun solely to add metadata unless the result is a headline with unreconstructable origin | #32, #46 |
| Paper claims | Two-paper structures and reviewer guide now narrow several unsafe legacy claims | **Pending manuscript audit** | Trace every final abstract/table claim to an allow-listed artifact | Legacy prose is superseded, not experimental data | #47 |

## Paper 2 evidence disposition

| Track | Existing evidence | Current state | Required new evidence | Effect on old evidence | Issue |
|---|---|---|---|---|---|
| Draft speculation | P2-D0/P2-D4 and repaired P2-D1 passed. P2-D2 passed 72/72 identity at 0/20/80 ms and hostile WAN, but draft throughput remained below sequential in every profile because full-history rebuilds dominate | **Correctness passed; all declared RTT profiles are valid negative performance evidence** | Analyze/report the completed profile series; any new mechanism must be evaluated separately against this baseline | The repaired results supersede the original 69/72 sweep for correctness. Keep failed attempts under invalid and focused traces under diagnostic | #5 |
| FLA/causal-conv | Logs show torch fallback | **Fallback baseline** | Kernel build, per-layer numerical comparison, warm timed benchmark, then end-to-end rerun | Keep fallback results as a named baseline; never pool with fast-path timings | #19 |
| Adaptive speculation | Design and hypotheses only | **Pending** | Tune/test-disjoint static versus adaptive experiment after E-D2 | No old numerical result to dismiss | — |
| fp8 | Design and hypotheses only | **Pending** | Numerical/quality controls and paired performance repetitions | No old numerical result to dismiss | — |
| 1F1B | Design and hypotheses only | **Pending** | Toy exactness test, 27B smoke, then WAN repetitions and memory accounting | Existing AFAB/sync results remain baselines | #45 where semantic test overlaps |

## Execution queue

The queue is ordered by scientific dependency, not by the desire to keep every
GPU occupied. Never interrupt a running job or launch more than one GPU job per
node.

| Priority | Resource | Action | Launch gate | Completion artifact / decision |
|---:|---|---|---|---|
| 1 | toula + koula | Finish 35B-A3B inference reproduction | **Completed and collected** | Sequential cells retained as supporting scale baselines; lookahead cells rejected after 3/3 identity failures |
| 2 | h100-2 | E8 D3 depth 1, K=100 | **Completed and collected** | Accepted: three distinct token hashes, W-hat errors, nested decoder scores, two variance components |
| 3 | Local host | DLG++ clustered analysis, chance/floor addendum, E9 session audit, provenance audit | Can run while GPU jobs execute | Decide whether E9 needs reruns and which DLG++ claims pass |
| 4 | h100-2 | Fix/test known-prefix harness, then dense corrective cells | **Completed and collected** | Corrected depth-1 crossings and nested uncertainty accepted; old derived headlines superseded |
| 5 | h100-2 | Joint sequence inversion at 0.6B | **Completed and collected; stop gate triggered** | Null/negative result retained; no 7B joint sweep |
| 6 | h100-2 | SipIt 7B | **Finished; artifact collected** | Scale-validation artifact complete; seed labels are deterministic duplicates, effective n=10 documents per cell |
| 7 | toula + koula | Audit E9 paired sessions | **Completed: three valid paired sessions** | No rerun required; proceed to deployed rotation |
| 8 | toula + koula | Deployed E8 rotation experiment | **Completed and collected** | Rotation preserves identity/privacy controls with negligible measured throughput difference |
| 9 | koula same host/path | Corrected real-WAN versus netem validation | Same cloud host available on both paths; arm order counterbalanced | Unconfounded transfer estimate with retransmit counters |
| 10 | Local + toula/koula | Implement distributed draft protocol | **Completed: self-tests and 35B-target/27B-draft smoke passed** | Two identity-passing runs; replay path exercised |
| 11 | toula, then koula | FLA numerical and performance microbenchmarks | **Fallback complete and collected on both nodes** | Fast-path/fallback numerical delta and timing, kept separate |
| 12 | toula + koula | Repeat E-D1/E-D2 on validated fast path | Kernel numerics accepted and distributed protocol valid | Measured throughput-vs-RTT break-even resolving #5 |

h100-1 remains outside the queue until the changed SSH host key is independently
verified. The currently presented ED25519 fingerprint must not be trusted solely
because the replacement host supplied it.

## Stop/go decisions

1. **D3:** accept only if all probe hashes differ and both variance components
   are present. Otherwise fix the run; do not substitute decoder seeds.
2. **Joint inversion:** proceed from 0.6B to 7B if the joint result materially
   changes recovery or falsifies a paper hypothesis; otherwise report the null
   and avoid an automatic scale sweep.
3. **E9:** rerun only if the local audit finds fewer than three complete paired
   sessions with aligned captures and utility/correctness outputs.
4. **Known-prefix:** if the corrected curve does not bracket a crossing densely
   enough, add points inside that bracket rather than rerunning the full grid.
5. **FLA:** do not run end-to-end performance experiments until numerical
   equivalence and the active fast path are demonstrated in logs.
6. **Paper freeze:** only confirmatory allow-listed artifacts may populate a
   headline table. Supporting and superseded evidence belongs in ablations,
   limitations, or audit appendices with its status visible.

## Result collection checklist

For every completed run:

1. Verify process exit status and peer health; archive logs before cleaning only
   stopped experiment containers.
2. Copy JSON/JSONL/log evidence without checkpoints, model weights, credentials,
   or SSH material.
3. Validate schema, repetition count, distinct seeds/hashes, provenance, model,
   hardware, WAN profile, and implementation path.
4. Assign an evidence state and record whether any prior metric is superseded.
5. Update `EXPERIMENT_COMPARISON_MATRIX.md`, the allow-list manifest, and the
   relevant issue with exact artifact paths.
6. Commit evidence only after validation; never rewrite or delete the historical
   artifact that motivated a corrective rerun.
