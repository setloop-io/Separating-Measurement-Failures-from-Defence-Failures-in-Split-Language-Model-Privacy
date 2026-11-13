# Adversarial evidence review — 8 September 2026

**Overall assessment: major revision. There are useful results, but the current scientific interpretation overstates privacy protection, remote necessity, and what the delegation sweep establishes. The most important new finding is that input bytes are readily recoverable from the saved 4-bit activations under a within-run known-plaintext attack.**

This review uses the supplied Paper 3 repository and the attached `2609.04382v1.pdf`. Documents were treated as evidence to assess, not instructions to adopt their conclusions. Original data, manuscript, and claim ledger were not changed. New analyses are explicitly post-hoc, not new confirmatory experiments. The peer-review skill's `/review` launcher was unavailable; its researcher/reviewer roles were used for independent checks, followed by direct recalculation here. No backbone training was run.

## 1. New finding: the token-inversion pass is not robust to within-run calibration

I fitted an untuned linear ridge classifier (lambda 1, 256 input features, one-hot byte targets) to **existing final-step forward captures** from all eight P2b seeds at q=4 and q=32. Each fold trains on seven complete sequences and evaluates the eighth; no token rows from the held-out sequence enter the fit. The eight byte windows do not overlap. A shuffled-training-label classifier and training-majority classifier accompany each fold.

| Existing condition | Mean input-byte accuracy | Range across 8 training seeds | Shuffled-label accuracy, mean | Majority baseline |
|---|---:|---:|---:|---:|
| q=4 | **96.72%** | 95.90–97.17% | 7.41% | 13.77% |
| q=32 | **97.80%** | 97.66–98.05% | 9.25% | 13.77% |

This is input-byte classification from released activations, not merely punctuation prediction. It directly contradicts unqualified statements such as “Token-inversion never leaks” in `docs/P5_DELEGATION_TRIANGLE_DECISION.md:42` and “not in raw token identity” in the P4b interpretation. The new attack was applied to P2b captures, not every P5 cell; do not silently extend its numerical result to P5.

**Threat-model scope:** this attack assumes the adversary obtains 896 known plaintext byte/activation pairs (seven 128-byte sequences) from the target model state. It does not establish recovery without known plaintext, an active exploit, or generalization to arbitrary corpora. All seeds reuse the same eight text windows, so this is model-seed replication on a fixed text panel, not eight independent text populations. It is not preregistered; no headline population p-value is claimed. A fresh document-disjoint capture cohort should confirm it.

Why the old probe missed it: `scripts/score_privacy.py:155` trains on calibration-model activations and `:182` applies that probe to independently initialized target models. Width matching does not align learned latent coordinates. The original negative advantages (approximately −9 to −14 percentage points in P2b) are compatible with failed cross-model transfer. The within-run result demonstrates that the representations are decodable under the stated side information. It does not isolate seed mismatch as the sole cause, since classifier and training distribution also differ. A same-classifier within-run versus cross-run comparison would isolate that explanation.

**Scientific significance:** the sharper result is an evaluation failure caused by insufficient attacker adaptation, not evidence that the boundary selectively conceals tokens while retaining only useful sequence structure. At q=4 the forward message still allocates 256×4=1,024 bits per input byte, versus 8 bits in the raw byte itself; low precision per coordinate is not a low total information budget or a privacy guarantee.

Reproduction: `within_run_probe.py`; exact fold scores and 16 verified capture hashes: `within_run_probe.json`.

## 2. New finding: beating frozen random is different from beating removal

Define **bypass gain = L(identity bypass) − L(trained remote)**. Positive means the trained remote improves quality over retaining the trusted encoder/decoder and using an identity in place of remote processing. This contrasts two complete training procedures; it is not a post-training ablation of one checkpoint.

| Existing configuration | Bypass gain, nats | Paired 95% CI | Two-sided p | Holm-adjusted p* |
|---|---:|---|---:|---:|
| P2 medium/high | **−0.01543** | [−0.02142, −0.00944] | .000495 | .00692 |
| P2b q4 | **−0.02190** | [−0.02638, −0.01741] | .00000823 | .000132 |
| P2b q8 | −0.00158 | [−0.00805, 0.00489] | .581 | 1 |
| P3 12×192 / Shakespeare | **+0.01614** | [0.00829, 0.02398] | .00182 | .02188 |
| P3 12×192 / War and Peace | **+0.00644** | [0.00319, 0.00970] | .00226 | .02490 |
| P5 span5, d0 | **+0.05457** | [0.04555, 0.06358] | .00000194 | .0000348 |
| P5 span6, d0 | **+0.05973** | [0.04955, 0.06991] | .00000239 | .0000406 |

*Holm across the 19 contrasts explicitly enumerated in `reanalyze.py`, not across all possible analyses one could search. All intervals are unadjusted, seed-level t intervals. These are exploratory reanalyses of existing cohorts, even where the underlying runs were confirmatory. Exact sign-flip sensitivity values are in the JSON; with eight seeds their resolution is coarse and they do not provide equally strong multiplicity-adjusted evidence. Do not present the small parametric p-values as assumption-free proof.*

The original frozen-random gains can be numerically correct while the practical interpretation is wrong. In P2 medium/high, remote training repairs some harm from the frozen transformation, but removing that transformation does better still. In P2b q4 the decision record says the learned remote recovers a penalty that an identity cannot (`docs/P2B_RATE_SWEEP_DECISION.md:57–59`); the stored losses give the opposite ordering.

**Positive result worth developing:** P5 d0 spans 5–6 beat both controls and pass the declared utility gate, with analytic remote FLOP shares around 49% and 58%. Their bypass gains are substantial compared with the earlier 0.01-nat practical threshold. These are the strongest tested candidates for useful remote computation before privacy is considered. Span7 gains more over removal (+0.14042 nats) but fails the utility gate. This suggests an intermediate useful operating range, not a universal depth law.

P3 also contains a useful distinction: the 12×192 configurations beat the identity bypass on both corpora; the 8×256 / War and Peace configuration does not (−0.01018 nats, unadjusted p=.0113, Holm .112). This supports a hypothesis that architectural allocation affects the necessity of remote processing. Depth, width, relative segment size and capacity change together, so it is not evidence that depth causes delegability.

## 3. Additional overlooked results

**Position matters in the held-out segment study.** At span 5, moving the replaced segment from start 1 to start 3 reduces loss by 0.03552 nats, paired CI [0.02501, 0.04604], n=7 under the study's seed-exclusion convention; Holm p=.00255. Span2 and span3 location contrasts are much smaller and do not survive correction. This is a specific position-by-span hypothesis, not a universal preference for late segments. It also explains why a statistic determined only by a segment's output endpoint is a poor candidate for the whole segment's replacement difficulty.

**The rate plateau is practical, not literally flat.** q8→q32 improves loss by 0.000311 nats, CI [0.000186, 0.000436], Holm p=.00800. This is statistically detectable but only about 1% of the q4→q32 improvement and far below the study's 0.01-nat practical scale. “Negligible additional improvement above 8 bits” is better than “the entire response is between 4 and 8.” Statistical significance and scientific importance diverge here.

**Increasing causal gain is not improving absolute utility.** In P5 d0, replacement penalty rises from approximately .025 to .134 nats as spans increase. Gain against a worsening frozen reference can grow while the trained model gets worse against all-local. The reported claim of monotonic gain is also literally false at spans5→6 (.0843→.0796). No conclusion about a necessity of preserving the leaked feature follows from simultaneous leakage and utility: feature-removal interventions were not performed.

## 4. Implementation and inference issues that need correction

### A. Reported operational rate does not match the implemented computation

`src/pcr/split.py:220–232` quantizes the forward latent z only. Returned y and both gradient channels are full precision; the STE passes gradients without quantization. `src/pcr/transcript.py:77–103` nevertheless counts every channel at q-bit packed size, through a size formula rather than actual network serialization. `src/pcr/privacy_attacks_p4b.py:228` and P5 scoring additionally quantize captured gradients before attacking them.

Consequences:

- The gradient probe discards information available in the simulated remote view; its failure cannot establish absence of gradient leakage.
- At width256/q4, forward q-bit codes plus three fp32 channels require approximately **25,600 bits/token**, not the reported 4,096; at q8 approximately **26,624**, not 8,192, before transport overhead. This comparison assumes packing the forward codes as proposed. The actual experiments execute locally on MPS and do not benchmark a deployed packed-wire system.
- Utility versus forward precision remains an interpretable experiment. Utility versus the claimed complete-transcript serialized communication rate must be relabeled or rerun using an implemented transport.
- FLOPs are an analytic estimate with excluded operations, not direct measurement of useful cloud work, network throughput, or energy.

### B. D1 is not a replication of the attached paper's mitigation

P5 simultaneously changes q8→q4, adds gradient clipping and Gaussian noise, and applies its backward defense at **both** z and y. It uses clip 1.0 and noise std .1; the attached paper's returned-gradient mitigation uses clip .01 and noise std .0035 on a different model, loss scaling and topology. Arbitrary frozen constants establish an operating-point result, not an optimal defense result.

The P5 .82–1.56 nats are penalties versus all-local, not clean isolated estimates of the incremental cost of gradient noise. Say “this bundled defense configuration performs poorly”; do not claim the previous mitigation has been refuted or that a gradient channel has been suppressed when no D1 gradient instrument qualifies.

### C. Measurement changes the D1 random stream

`capture_probe_tensors` (`src/pcr/train.py:247`) runs a backward pass. `_GradDefense.backward` (`src/pcr/split.py:174`) draws from the same generator used by subsequent training, and probe capture does not save/restore it. Thus the claimed measurement-only pass changes later D1 noise realizations. Identically scheduled trained/frozen pairs can still share the stream, but “no RNG interaction” is false and calibration with different capture frequency has a different trajectory. Fix prospectively and assess sensitivity; this does not erase all existing paired results.

### D. P5 conflates missing attack scores with a privacy failure

`scripts/analyze_delegation.py:495–504` constructs one list containing defense-specific qualified gradient variants; `:555–558` applies the list to every defense. D1 then lacks scores for the qualified d0 variant. The missing-score handling produces a failing gate. Per-defense qualification and an explicit “not evaluated” status are required. Existing positive forward-channel readings independently support rejecting the current tested privacy candidates, so this bookkeeping defect alone does not rescue the frontier.

### E. Privacy coverage and statistical units are narrower than the inventory implies

The inventory advertises accumulated history and remote state with no omissions (`src/pcr/transcript.py:120`), but remote state is represented by checksums, and confirmatory attack scoring uses a final capture of eight fixed sequences. Checksums cannot substitute for attacks on the evolving weights and optimizer state. The same fixed text panel appears across model seeds and in dense calibration captures. Repeated training states of the same plaintext do not create new independent sequences.

Property r-advantage is r minus one shuffled-model r, so values above 1 are mathematically possible and are **not** correlations above 1 or near-perfect privacy risk probabilities. With only eight sequences and a max over seeds/targets, a negative null correlation can enlarge the reported advantage substantially. Report raw r, raw null, uncertainty and task-specific qualification, including that only punctuation qualified cleanly in P4b. Positive attack evidence can reject a candidate; failure of these attacks cannot certify privacy. The new within-run byte probe supplies much more direct evidence for the evaluated P2b panel.

### F. “Confirmatory” provenance is not independently established by the repository snapshot

The current git history contains the September 3 scaffold commit. Most executed source files, later protocols and manifests are untracked. `dirty_patch_sha256()` hashes `git diff`, which omits untracked file contents; the status hash records names/status, not those contents. Therefore commit + dirty-patch hash does not reconstruct the executed implementation, and self-dated freeze notes do not independently prove pre-outcome registration. This is an auditability gap, not evidence of fabrication or retrospective tuning.

All **282 metric files** used in the new contrast analysis matched their current manifest hashes, as did all **16 captures** used for the byte probe. This verifies current artifact integrity against current manifests, not historic execution or freeze timing. Preserve the snapshot now, publish code/config/data fingerprints, and obtain externally timestamped registration for follow-up confirmation. Do not retroactively mark old runs as newly preregistered.

## 5. Relationship to the attached paper

The attached paper, *Privacy Failure in Split-LLM Training, The Returned Gradient Nullifies the Decoys*, establishes an exact real/decoy partition disclosure in its implemented system: 4,096/4,096 frames on nine seeds. It separately reports modest frequent-token advantages and successful suppression of scored attacks at a four-layer mitigation topology. Those are distinct evidence levels.

| Dimension | Attached paper | Current Paper 3 |
|---|---|---|
| Model | Frozen Qwen3-0.6B backbone setup | Small byte-level models trained from seeded initialization |
| Remote mechanism | Tiny gauge-equivariant surrogate, rotations/permutations and decoys | One or more ordinary Transformer blocks; quantized forward boundary |
| Primary privacy failure | Returned-gradient zeros reveal real versus decoy rows | Forward activations expose properties; new within-run analysis recovers bytes |
| Defense evidence | Matched returned-gradient mitigation within that system | Bundled q/gradient perturbation with different constants and two backward locations |
| Generalization | Scoped systems-security case study | Separate mechanism study, not a replication of the same experiments |

Do not pool their losses, token accuracies, or mitigation costs. Tokens in Qwen's vocabulary and bytes here are different prediction units. The older paper's oracle-partitioned forward arm is explicitly not an independently available forward-only attack until the gradient exposes the partition (Table8). Its strong structural result must not be presented as full text reconstruction. Table9 shows mitigation on one corpus and unmitigated joint leakage on a second; it does not independently show the mitigation transferring to both corpora.

The most defensible connection is **two distinct ways an evaluation can miss accessible information**: the attached study omitted a channel; the new study's token probe insufficiently adapts to the target representation. This is an interpretation of these two case studies, not a claim of literature-wide novelty. No broad novelty search was performed. The attached paper itself is not evidence for a universal delegation triangle.

## 6. Recommended scientific position and next checks

A defensible current thesis is:

> Trained remote modules can improve utility beyond both frozen and identity controls at specific capacity allocations. However, the tested quantized boundary is not robust to a simple within-run known-plaintext token probe. Earlier negative token results from cross-model probes and incomplete communication accounting overstated the privacy/utility evidence.

Prioritize these bounded next steps:

1. **Independently reproduce the new byte attack on fresh document-disjoint captures**, preregistering side information, held-out text, per-run adaptation and nulls. Compare within-run and cross-run training using the same classifier. Include q4, q8 and useful P5 spans5–6.
2. **Repair transcript fidelity and accounting** before further rate claims: attack the actual fp32 gradients, implement transport quantization if intended, capture state, and separate absent scores from observed failure.
3. **Make both controls primary** for useful remote computation. Confirm the P5 intermediate operating range with fresh seeds and a crossed span/location/remote-depth design, avoiding the current co-variation of those factors.
4. **Isolate defense effects** with a small q × gradient-noise factorial and noise scaled to clean gradient norms. Freeze it before inspecting outcomes; restore measurement RNG state.
5. **Revise interpretation now:** withdraw broad token-protection language, soften the triangle to an observed absence of feasible tested cells, correct the identity-bypass sign, and distinguish practical saturation from exact flatness.

The existing results do contain scientifically interesting information. The strongest contribution is the combination of a better causal control comparison and a newly exposed attack-evaluation blind spot, with a narrow useful-compute regime that still lacks demonstrated privacy.

## Reproducibility files

- `reanalyze.py`: 19 seed-level contrasts, t intervals, Holm correction and exact sign-flip sensitivity.
- `contrasts.json`: raw per-seed differences, statistics and source hashes.
- `hash_checks.json`: 282 metric-to-manifest checks.
- `within_run_probe.py`: no-tuning, leave-one-sequence-out ridge attack using saved forward captures.
- `within_run_probe.json`: all fold results, 16 capture hashes and text-window overlap check.
- `attached_paper.txt`, `attached-12.png`: local extraction and visual check of the attached mitigation table.

To reproduce, run the two Python scripts from the Paper 3 repository with its `.venv/bin/python`. The scripts write only into this review directory. Their torch loads of captures use `weights_only=True`.
