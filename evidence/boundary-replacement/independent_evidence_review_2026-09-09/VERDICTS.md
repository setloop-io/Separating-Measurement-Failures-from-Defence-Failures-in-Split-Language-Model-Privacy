# Independent evidence verdicts — 9 September 2026

**Overall verdict: the experiments support a narrow result about trusted boundary adaptation on a frozen small language model. They do not yet establish private token reconstruction resistance, useful remote compute delegation, or a general replaceable fraction of an LLM.** The remote-free controls are scientifically valuable: they suggest that the useful adaptation largely resides in the trusted boundary. The privacy evaluation has two concrete target/selection problems that require corrected attacks before confidentiality claims can be accepted.

This assessment was formed from experiment JSON, manifests and implementation before reading the current paper, its narrative, evidence ledger or previous reviews. The separate comparison document was written afterwards. No experiments were rerun, and no paper or implementation files were changed. Numerical checks are reproducible with `python3 outputs/independent_evidence_review_2026-09-09/recompute.py` from the repository root. [Recomputed evidence](recomputed_evidence.json) includes SHA-256 hashes for 66 input artifacts. The `/review` expansion mentioned by the peer-review skill was not exposed in this session; its researcher/reviewer roles were used for independent evidence audits.

| Question | Verdict | Confidence and scope |
|---|---|---|
| Can a trained trusted boundary compensate for removing some frozen-model layers? | **Supported** | Strong across the tested seeds and fixed short-context corpus windows. |
| Does the learned remote module account for the successful endpoints? | **Not demonstrated; controls argue against necessity** | Zero-return training passes and beats trained remote in all six clean corpus–seed comparisons. |
| Do the recorded attacks establish input-token confidentiality? | **No** | They predict shifted next-token labels; the purported real-only selection also fails after gradient noise. |
| Did gradient noise remove the observed exact chaff-support leak? | **Supported narrowly** | All 80 gradient rows become nonzero in recorded repaired runs; other leakage remains unexcluded. |
| Is there a general maximum replaceable percentage or depth law? | **Unsupported** | Sampled corpus-specific brackets, truncated-model confounds, and invalid larger-model geometry. |
| Is training or inference economically improved? | **Unproven** | Original teacher compute remains in training; FLOP scopes differ; timing includes diagnostics; no matched energy baseline. |
| Is general language-model quality preserved? | **Not established; observed degradation** | Similar short-window NLL coexists with a large answer-token accuracy drop and repetitive samples. |

**1. The utility result survives, with a narrower interpretation.**

Five-seed historical confirmation reruns give the following candidate-minus-intact mean token NLL. Intervals are Student-t intervals across training seeds, conditional on the same evaluation text:

| Corpus and split | Nominal removed layers | Mean ΔNLL | 95% seed interval |
|---|---:|---:|---:|
| WikiText-2, split21 | 4/28 (14.29%) | −0.04653 | [−0.04953, −0.04353] |
| WikiText-2, split19 | 6/28 (21.43%) | +0.51613 | [+0.51038, +0.52188] |
| Penn Treebank, split17 | 8/28 (28.57%) | −0.04323 | [−0.05141, −0.03505] |
| Penn Treebank, split14 | 11/28 (39.29%) | +0.44949 | [+0.40951, +0.48948] |

Sources: individual cells in [WT2 reruns](../../outputs/phase06/wikitext2_confirmation_rerun/) and [PTB reruns](../../outputs/phase06/penn_treebank_confirmation_rerun/). Original and rerun batches reuse seed IDs; they must not be counted as ten independent training seeds.

These establish successful and unsuccessful sampled spans under the historical protocol. They do not locate an exact maximum. PTB split15 was borderline in exploration (mean +0.35854, with seeds on both sides of the +0.35 gate) and is not the confirmed failing endpoint. The +0.35 threshold permits approximately 41.9% higher perplexity; passing this threshold is not synonymous with retaining general capability.

All original backbone weights are frozen. The trusted node learns an encoder/decoder and retains the prefix hidden state through residual addition. Every training step still evaluates the original removed segment as a distillation teacher; the segment is skipped in candidate evaluation. This is boundary adaptation/distillation, not evidence that the corresponding share of base-model training was outsourced. See [runner](../../bin/run_latent_native_v5_06b.py), especially lines 306–307, 354–357, 506–527 and 715–835; [boundary](../../privacy_runtime/latent_native.py), lines 532–563.

**2. The strongest mechanism result is that the cloud is unnecessary for meeting this utility target.**

The repaired endpoint controls give these three-seed mean ΔNLL values:

| Corpus endpoint | Trained remote | Frozen random remote | Separately trained zero return | Trained local |
|---|---:|---:|---:|---:|
| WT2 split21 | −0.00012 | −0.00062 | −0.01255 | −0.04604 |
| PTB split17 | +0.00109 | +0.00097 | −0.01178 | −0.04217 |

Sources: individual files in [clean controls](../../outputs/phase08_clean_endpoints/clean_controls/). Zero return beats trained remote in all six matched seed comparisons, by approximately 0.008–0.020 nats. Trained remote does not show a consistent advantage over a frozen random remote module. This directly challenges a claim that *learned remote computation* causes the successful endpoint performance.

Do not overcorrect to “the remote return has literally no effect.” Removing the return from an already-trained PTB model raises NLL by about 0.020–0.040. That model uses its return, but independently training without it achieves the target anyway. Also, `cloud_correction_loss_improvement` removes the entire trusted decoder in its comparator; it cannot attribute the resulting improvement to the remote module. `surrogate_return_loss_improvement` is the relevant within-model diagnostic.

Historical confirmation has forward/return DP accounting; repaired remote controls additionally account for outbound-gradient protection. The historical ~0.04-nat improvements therefore cannot simply be attached to the repaired protocol. The single-seed prefix-gradient ablation supports a possible gradient-channel effect but lacks the replication and immutable treatment provenance needed for a strong causal claim.

**3. Token reconstruction privacy is not established by the implemented target.**

In [runner lines 417–419](../../bin/run_latent_native_v5_06b.py), `tensors()` returns `input_ids = ids[:, :-1]` and `labels = ids[:, 1:]`. The prefix representations are calculated from the former, but attacks, chaff labels and exact-wire label joins receive the latter. `attack_targets()` permutes labels; it does not remove this shift. The post-training probe uses the same convention.

Thus a row representing input token x[t] is scored against x[t+1]. This measures next-token predictive information, not direct reconstruction of the token represented by that row. A representation could reveal its current input token yet perform poorly on this test. Sequence structure might permit some attacks to exploit the shift, but that does not make this metric a test of direct input-token secrecy, particularly with token permutation and short independent frames.

**Required verdict:** rename existing scores as next-token probe performance. Claims that these scores bound current-token recovery must wait for correct labels and reruns. A large naked-control score demonstrates predictive power under the current metric, not validation of an input-reconstruction evaluator. This finding applies to the inspected runner-generated evidence; it does not assert that every historical attack implementation in the repository shares the same defect.

**4. The repaired “real-token-only” arms are also misselected.**

[Bundle construction](../../bin/deleg6040_grad_bundle.py), lines 177–194, selects `keep=32` rows by nonzero gradient support. After noise is applied to every row, all 80 rows are nonzero. The selector consequently takes the first 32 permuted wire rows, including an unknown mix of current and recycled chaff tokens.

The [split21 seed42 bundle report](../../paper-data/collected/diagnostic/clean_endpoint_audits/audit_split21_s42_bundles.json) records 80 nonzero rows, 32 actual real rows, 0.4 mask agreement and zero exactly matching frames. The same condition occurs across the six clean endpoint bundle reports. Hence `wire_real`, `grad_real`, and their joint variants cannot be described as real-only attacks for these runs.

The trusted `is_real` mask should identify genuine rows for an oracle de-chaffed audit, with the all-row view retained as the actual cloud observation. This need not disclose the mask to the operational attacker: trusted-side labels/masks are appropriate for scoring. Because chaff consists of recycled corpus rows, mixed-row averages and nominal token counts cannot establish confidentiality of current private tokens.

**5. A narrow security repair is supported, but full-compromise security remains open.**

The historical [gradient bundle report](../../paper-data/collected/diagnostic/threat_model_scope/grad_power_10k_w4096_split14_bundles.json) shows exact correspondence between zero support and the real/chaff partition in all 4,096 frames. Repaired reports lose this exact support signal. This is valid mechanistic evidence of a repaired disclosure, independent of the shifted prediction labels.

The phase09 [cloud-state probe](../../attacker/attacks/cloud_state_probe.py) supplies an MLP with forward rows, constant final model weights and metadata features. Optimizer state, backward gradients and state history are not attack inputs, despite broad threat-model wording. Constant weights repeated for every row do not test how changing state across training or sessions might reveal data, and appending constants changes classifier normalization/optimization.

[Recorded scores](../../outputs/phase09_economics_quality/clean21_cloud_state_probe.json) are 3.138% for wire alone and 3.212% for wire plus state, against 4.153% majority; metadata alone scores 0% in all three restarts. These are weak diagnostic results, not a demonstration that cloud state or metadata is safe. A null result from an underperforming classifier is particularly inconclusive.

Further limitations: deep probes select the best evaluation epoch before reporting a restart-adjusted Wilson interval; row-level Wilson intervals ignore frame/text/chaff dependence; joint concatenation's Gram matrix is WWᵀ+GGᵀ and does not explicitly expose WGᵀ to the invariant arm; the +1 percentage-point tolerance is an operational threshold, not absence of leakage. The original phase09 recorded composed DP ε is about 16.25 million at δ=10⁻⁶, providing no useful practical confidentiality bound.

**6. Lower noise did not extend the WT2 utility result in the recorded sweep.**

All 48 [phase07 cells](../../outputs/phase07_noise_sweep/) cover four splits, four noise levels and three seeds. Across noise levels, mean ΔNLL is 1.458–1.525 at split14, 0.982–0.998 at split17, 0.539–0.559 at split19, and −0.005–0.008 at split21. All 12 split21 cells meet +0.35; the other 36 do not. This is a useful negative result for this tested noise range and budget. It is not proof that no different objective, boundary capacity or training budget could succeed.

**7. Quality preservation is much narrower than NLL preservation.**

The matched phase09 [intact](../../outputs/phase09_economics_quality/quality_intact.json) and [composed](../../outputs/phase09_economics_quality/quality_composed.json) evaluations give NLL 5.26889 versus 5.30284 and perplexity 194.20 versus 200.91 (+3.45%). Yet teacher-forced answer-token top-1 falls from 34.79% to 18.15%, a **16.64 percentage-point drop**, across 6,849 answer tokens in 500 rows. Both exact-row rates are zero. The three composed generations are repetitive or contain repeated `<unk>` tokens; intact samples are also weak.

This is not standard HellaSwag multiple-choice accuracy. The [quality evaluator](../../bin/eval_composed_quality.py) uses prompt/answer token matching and runs an offline composition without wire defenses. Therefore it demonstrates a concerning metric mismatch in one checkpoint, not a measured deployed-protocol task-accuracy drop across seeds. Larger, standard task evaluations and matched defended inference are needed before claiming useful language-model capability is preserved.

**8. Neither depth scaling nor the stated 27B replication is established.**

The 20/24-layer models in [phase03](../../outputs/phase03/) are truncations of the same frozen 28-layer checkpoint. Their intact baselines are already badly degraded: NLL 9.167 and 7.079, versus 4.930 for the original model. Boundary training improves those damaged baselines, but candidates remain around 6.267–6.324 and 5.766–6.115. No failing endpoint brackets these studies. They show partial repair of truncation damage, not a general depth law or a comparison of independently pretrained depths.

The [27B run files](../../outputs/phase06/qwen35_27b_reproduction/) report actual depth 64 but request `resume_after=74`. Split75 requests `split_after=75`, logs **−2 covered layers**, and removes no segment. Other splits33/37/41 log 40/36/32 removed layers, although Python slices can contain only 30/26/22 layers, leaving an empty tail. The nominal percentages and purported intact control are not valid geometry-matched replication evidence. Quarantine these results pending corrected bounds, model identity and reruns; do not use them to support either successful scaling or a cleanly established failure of the stated larger-model design.

**9. Economics remains an open question.**

The clean endpoint records 161 remote parameters, about 497 MB of total session wire traffic, 243.70 billion remote profiler-counted operations across a session, and 50.79 billion trusted operations for one representative training step. Dividing the former by the latter extrapolated over 4,000 steps gives about 0.12%, but these scopes differ and profiler coverage is incomplete. It is only a scale diagnostic, not a measured remote-work fraction. Nominal 14.29% layer coverage cannot be substituted for that fraction.

The [energy reprofile](../../outputs/phase09_economics_quality/energy_reprofile_summary.json) supplies sampled-power estimates: approximately 45.0 kJ trusted and 28.8 kJ remote for the clean run. These are whole-run GPU estimates, not total-system energy or savings. No matched intact/local/zero-return energy baseline establishes a benefit.

In [runner lines 939–1062](../../bin/run_latent_native_v5_06b.py), the candidate evaluation timer includes both zero-cloud and decoder-zero diagnostic losses and data preparation/collection. Baseline timing contains only intact forward evaluation. The reported ratios (including 2.744 in the original phase09 run and 1.277 in its reprofile) must not be described as isolated inference slowdown or speedup. Benchmark matched candidate-only execution before judging deployment latency.

**10. What the paper can responsibly conclude now.**

Suggested core conclusion:

> On a frozen Qwen3-0.6B model and fixed short-context corpus windows, a trained trusted residual boundary compensates for removing selected layer spans. At tested repaired endpoints, separately trained zero-return controls retain comparable or better token loss than a trained remote surrogate, so the results do not establish that useful remote computation is required. Broader language quality, economic benefit and input-token confidentiality remain unestablished; the existing privacy evaluator requires corrected current-token targets and real-row selection.

As a paper about private compute delegation, this requires **major revision and corrected experiments**. As a paper about the limitations of using a small protected boundary to replace frozen-model computation, it has a credible empirical core. The strongest contribution is the controlled negative finding about remote necessity, with the utility and quality mismatch reported alongside it.

Priority work before submission:

1. Correct current-input-token attack targets and real-row selection; rerun naked and defended controls, gradients, explicit cross-Gram attacks and state-history attacks on matched protocol/checkpoint/capture identities. Select attacks on validation data, then freeze them for test scoring with frame/block-aware uncertainty.
2. Repeat utility brackets under the complete repaired protocol with stronger remote-free/local controls, document held-out splits, and evaluate standard downstream tasks and longer-context generation.
3. Repair 27B geometry and validate model/depth identity before further scaling claims; interpret truncated-depth repair separately.
4. Measure candidate-only latency and equal-quality energy/compute against intact, local and zero-return baselines, separating teacher training from inference.

The archived JSON and code suffice for this audit, but most raw attack captures, corpus files and backbone weights are not in this checkout. Only two phase09 `.pt` checkpoint/state assets were found. Consequently this review validates arithmetic and implementation logic, not independent reproduction of the GPU experiments. PTB is a disjoint slice of `ptb.train.txt`, not the official test split, and new seeds do not create new held-out text.
