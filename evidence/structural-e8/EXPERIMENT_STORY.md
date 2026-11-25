# The Experiment Story: Distributed Training and Inference, Privacy, and Rotation

Status: living draft, 21 August 2026

## The question that started the project

Large language models do not always fit, economically or operationally, on one trusted machine. A natural alternative is to distribute the model: one node computes the trusted prefix and suffix, another node computes the middle layers, and intermediate activations cross the network between them. The same boundary supports both inference and training: inference exchanges activations for generation, while split fine-tuning also sends gradients back across the boundary. A second scale-out option, FedAvg, distributes training through model updates rather than a continuously split forward/backward pass.

That design creates four questions, and the experiments in this repository follow them in order:

1. **Can a split model produce the same answer as the original model?**
2. **Can it do so fast enough across a real network?**
3. **Can the same architecture train across nodes, and what do synchronization, overlap, and FedAvg cost?**
4. **What does the intermediate activation or training update reveal, and can we protect it without breaking the model?**

The project did not move through these questions in a straight line. Several seemingly successful optimizations failed exact identity checks. Some privacy defenses worked for a single session but raised a key-lifetime problem. A later rotation matrix completed operationally but was rejected because its evidence protocol was flawed. Those failures are part of the scientific result: they tell us which claims the evidence can and cannot support.

## How to read each experiment

Every experiment below has five parts:

- **Question:** the uncertainty being tested.
- **Method:** what was changed and what remained controlled.
- **Result:** the measured outcome.
- **So what:** the decision or scientific meaning.
- **Evidence status:** accepted, diagnostic, invalid, blocked, or running.

An experiment is not accepted merely because a process completed or produced a JSON file. Correctness and provenance gates come first. Throughput is excluded when token identity fails; privacy measurements are excluded when keys or raw captures contaminate the evidence; fallback kernels are not presented as optimized-kernel results.

## Act I — Establishing that split execution works

### Experiment 1: Exact split-model identity

**Question.** Does moving a middle portion of the model to another node change the generated output?

**Method.** Run the same prompt through sequential target decoding and through the split path. Compare generated token IDs, rather than relying on similar-looking text. Repeat under controlled network profiles and across model sizes.

**Result.** The sequential split path produced valid identity evidence for the tested dense models and for the 35B-A3B target baseline. The 35B lookahead implementation did not: all three identity repetitions failed, with a maximum logit difference of 38.617 and lookahead perplexity near 5.02 million.

**So what.** Splitting the model is technically viable, but an optimization cannot inherit that correctness claim. The failed lookahead arm taught us to treat identity as a prerequisite for performance reporting. Its throughput numbers are excluded.

**Evidence status.** Sequential path accepted; original 35B lookahead path invalid.

### Experiment 2: Latency and throughput under emulated and real WAN conditions

**Question.** Once correctness is established, how much does network delay reduce generation speed?

**Method.** Measure sequential split inference at 0, 20, and 80 ms RTT and under a hostile profile with loss and jitter. Separately compare emulated WAN behavior with a real Internet path of approximately 100–121 ms.

**Result.** For the 35B-A3B sequential target, throughput declined from 14.63 ± 0.29 tok/s at 0 ms to 11.20 ± 0.00 at 20 ms, 6.57 ± 0.06 at 80 ms, and 1.80 ± 0.00 under the hostile profile. The real-WAN experiment confirmed that network conditions, not only synthetic netem settings, materially affect split execution.

**So what.** Split inference remains functional over a WAN, but communication becomes the dominant systems constraint. This motivates methods that reduce round trips, while the identity failure above warns that fewer round trips are useful only if exact target behavior is preserved.

**Evidence status.** Accepted sequential systems evidence.

## Act II — Training across nodes: synchronization, overlap, and aggregation

### Experiment 3: WAN split fine-tuning baseline

**Question.** Can a model make training progress when its forward and backward passes cross a real or emulated WAN?

**Method.** Run synchronous pipeline-split fine-tuning with the trusted prefix and suffix on one node and the middle layers on the other. Compare the synchronous schedule with local controls across 7B and Qwen3.6-35B-A3B configurations under WAN profiles. Check parameter/update semantics and bit-exact outputs before interpreting timing.

**Result.** Split fine-tuning completed with bit-exact semantic agreement in the validated baseline cells. The experiments established a usable systems baseline for dense and hybrid-MoE models, while WAN delay increased step time as expected.

**So what.** Distributed execution is not limited to serving: the repository demonstrates training progress across nodes as well. The training boundary has a different traffic pattern from inference because gradients return across the wire, so inference-only latency results cannot be used as training results.

**Evidence status.** Accepted baseline training evidence, scoped to the tested models, schedules, and short systems runs.

### Experiment 4: Overlap scheduling and FedAvg comparison

**Question.** Can communication be hidden during training, and when is update-based FedAvg preferable to a continuously split step?

**Method.** Compare synchronous split training with asynchronous overlap scheduling, then compare both with FedAvg. Overlap changes the execution schedule; FedAvg changes the communication and optimization protocol. Measure step time, throughput, memory, and exact update/output controls under WAN profiles.

**Result.** Validated overlap cells achieved approximately 6.7–9.3× exact speedups over the synchronous split-training baseline in the measured settings. With naive transport, FedAvg’s estimated break-even point is roughly K*=1600 local steps between synchronizations. The corrected FedAvg training-seed repetitions used later for E-A4 were independent after seeded shuffling was repaired.

**So what.** There is no single “distributed training” speed number. Pipeline overlap amortizes network waits within a split step, while FedAvg amortizes communication across local work but changes the training protocol. Claims about convergence or final quality require longer horizons than these systems microbenchmarks.

**Evidence status.** Accepted systems evidence for the measured schedules; the FedAvg break-even is a transport-model estimate, not a universal threshold.

## Act III — Trying to hide network latency with distributed drafting

### Experiment 5: Distributed draft-protocol smoke test (P2-D0)

**Question.** Can a smaller remote draft model propose tokens while a larger target model remains the final authority?

**Method.** Pair a tokenizer-compatible Qwen3.5-35B-A3B target with a 27B draft. Use two prompts and require exact agreement with sequential target decoding.

**Result.** Both 16-token runs matched the target exactly. One accepted 12 draft tokens without replay; the other accepted 14 and exercised two replay forwards.

**So what.** The protocol could work end to end, including rollback and replay. This was a correctness smoke test, not evidence of repeatability or speedup.

**Evidence status.** Accepted smoke evidence.

### Experiment 6: Prompt-category coverage (P2-D4)

**Question.** Was the smoke-test success specific to two convenient prompts?

**Method.** Test eight prompts spanning code, structured output, creative writing, and conversation. Require exact identity and exercise replay where disagreement occurs.

**Result.** All eight runs matched sequential decoding. Five runs invoked replay, totaling 14 replay forwards.

**So what.** The protocol was not limited to one prompt style, but it still lacked repeated trials and a throughput claim.

**Evidence status.** Accepted one-repetition correctness coverage.

### Experiment 7: Repeated K sweep and the cost of exactness (P2-D1)

**Question.** Does the protocol remain exact across draft lengths K=2, 4, and 8, prompts, and repetitions?

**Method.** Run 72 cells: three K values, eight prompts, and three repetitions. After an earlier 69/72 result exposed state-reconstruction errors, repair the protocol with a full-history rebuild policy and repeat the complete matrix.

**Result.** The repaired policy passed 72/72 identity cells. It required 291 exactness-fallback rounds, 6,057 rebuild forwards, and 6,114 replay forwards.

**So what.** Exactness was recoverable, but it was expensive. The result transformed the next question from “does it work?” into “does the cost ever pay for itself?”

**Evidence status.** Repaired run accepted for correctness. Earlier failed and focused diagnostic runs remain excluded from performance claims.

### Experiment 8: Throughput versus RTT (P2-D2)

**Question.** Can distributed drafting break even when network latency becomes severe?

**Method.** Measure the repaired K=2/4/8 protocol and sequential decoding at 0, 20, and 80 ms plus hostile WAN. Require 72/72 token identity in every profile before interpreting speed.

**Result.** Every 80-token identity cell passed. Sequential throughput was 15.43, 11.57, 6.75, and 1.89 tok/s across the four profiles. The repaired draft path reached only 1.55–1.62, 1.37–1.47, 1.07–1.18, and 0.47–0.65 tok/s. Under hostile WAN, K=2/4/8 averaged 0.529, 0.470, and 0.646 tok/s.

**So what.** This is an important negative result. The intuitive argument for distributed drafting was strongest under bad networks, yet the exact full-history protocol never broke even. Correctness recovery consumed more computation than latency hiding saved. Future work needs a different state-management or verification design, not merely a different K.

**Evidence status.** Accepted negative systems evidence. Hostile evidence was merged in PR #84.

## Act IV — Discovering what the split representation leaks

### Experiment 9: Trained inversion across model sizes and split depths

**Question.** Can an attacker reconstruct private tokens from intermediate activations?

**Method.** Train decoders on activation/token pairs, then evaluate on held-out victim data across models and split depths. Follow with stronger attacks that exploit known prefixes, sequence structure, output-side representations, or geometric structure.

**Result.** Leakage was not removed simply by moving the split deeper. At 7B scale, the SipIt attack recovered approximately 82.35/82.97% at depth 1, 84.38/84.53% at depth 4, and 77.03/78.44% at depth 8 in bf16/fp32. Conversely, DLG++ gradient inversion was weak on the 0.6B experiment: mean recovery was 1.77% at depth 1, 0.31% at depth 4, and 0.21% at depth 8. A joint sequence-aware attacker also failed to beat the simpler position-wise MLP.

**So what.** “Privacy attack” is not one capability. Some threat models are weak while others recover most tokens. A defense cannot be justified by defeating only gradient inversion or one decoder architecture. The strongest applicable attacker must define the claim.

**Evidence status.** Accepted attack evidence, with effective sample-size and provenance caveats recorded in the experiment matrix.

### Experiment 10: Membership and property inference during distributed training (E-A4)

**Question.** Even when exact text is not reconstructed, can the representation reveal whether a sample was in training or disclose a sensitive property?

**Method.** Compare split fine-tuning and FedAvg over independent training seeds. Archive earlier repetitions when identical feature hashes reveal that the seed did not actually change data order; rerun with corrected seeded shuffling.

**Result.** Across corrected seeds 42/43/44, membership ROC-AUC was 0.875 ± 0.013 for split fine-tuning and 0.874 ± 0.008 for FedAvg. Property ROC-AUC was 0.979 ± 0.005 and 0.984 ± 0.003, respectively.

**So what.** Distributed training exposed strong membership and especially property signals under both protocols. The invalid first attempt also established a general rule for this project: nominal seed repetition is not independence unless the resulting data or feature hashes differ.

**Evidence status.** Corrected runs accepted; earlier zero-variance runs archived as invalid.

## Act V — Defending the wire without changing the model

### Experiment 11: Orthogonal activation obfuscation (E8/E9)

**Question.** Can the local node transform an activation before transmission so that a passive wire attacker fails, while an authorized cloud node restores the original computation?

**Method.** Multiply the boundary activation by an orthogonal matrix W before transmission. Evaluate three legs: unprotected baseline, local-only obfuscation, and shared-key restoration. Compare generated token identity, throughput, utility, and wire-decoder recovery.

**Result.** In the initial live 7B validation, baseline wire recovery was 50% top-1 and obfuscated recovery was 0%, while held-out loss was unchanged within measurement precision. Three paired sessions later reproduced baseline, local-only, and shared-key behavior. Every local-only output differed from baseline, every shared-key leg restored identity, and obfuscated wire recovery was 0% in all sessions. Fixed versus per-session rotated keys had essentially unchanged throughput: baseline means 10.53/10.57 tok/s, local-only 11.40/11.30, and restored 11.33/11.30.

**So what.** Orthogonal wrapping is a practical wire defense against the evaluated fixed passive decoder and can preserve exact model behavior when both endpoints share the key. But the result does not protect against a compromised cloud holding the key, and a fixed or long-lived key gives an attacker time to gather labeled pairs.

**Evidence status.** Accepted for the stated fixed passive-wire threat model. It is not a claim of protection from the cloud itself.

### Experiment 12: How many labeled pairs defeat the defense?

**Question.** If an attacker can collect labels under one key, how long does the key remain useful?

**Method.** Sweep the number of labeled activation/token pairs K and measure when recovery returns. Repeat across depths, seeds, and stronger known-prefix conditions.

**Result.** For the 27B robustness battery, passive label-free recovery stayed at or below 3.71% top-1. The estimated K50 values were approximately 258.3, 355.8, and 617.9 labeled pairs at depths 1, 4, and 8. At 90% row leakage, recovery returned to 65.17–69.92%. In the corrected known-prefix experiment, the 90%-of-baseline crossing occurred around K=234–240 for prefix lengths 32–128.

**So what.** Obfuscation buys time rather than permanent secrecy. This finding motivated rotation: change W before an attacker accumulates enough matched evidence under one key.

**Evidence status.** Accepted laboratory security evidence.

## Act VI — Rotating keys before the attacker catches up

### Experiment 13: Deployed per-epoch ratchet matrix

**Question.** Can the deployed two-node path rotate W frequently, preserve exact outputs, and keep each attacker window below its learning budget?

**Method.** Run N=16/64/256 at RTT 0/80 ms and evidence-budget B=500 at RTT 0/80 ms, with three sessions and 60 generated tokens. Measure baseline, static-key, and ratcheted legs and run wire attacks per epoch.

**Result.** The matrix completed operationally, but it did not produce acceptable scientific evidence. N=16 and both budget configurations had shared-key identity failures in at least one session. Sessions 2–3 lacked baseline measurements by construction. The summaries exposed `key_seed_base=77000`, which allows regeneration of every epoch key. N=256 advanced zero epochs in the 60-token runs, so those cells did not exercise rotation at all.

**So what.** Completion is not validity. The deployed ratchet remains unproven. The next deployed matrix must rotate secret material, omit secrets from artifacts, run a baseline and all required attacks for every session, stop or quarantine a cell on identity failure, and ensure the token horizon actually crosses an epoch boundary.

**Evidence status.** Entire matrix invalid for timing and attack claims. Retain only as diagnostic evidence about the harness and failure modes.

### Experiment 14: Rotation-lifetime laboratory study

**Question.** Before spending another two-node matrix, does frequent rotation actually reduce a stronger attacker’s usable sample budget?

**Method.** On Toula, run `rotation_lifetime.py` offline on Qwen3.5-27B and Mistral-NeMo-12B at depths 1/4/8. Compare within-epoch recovery with recovery from captures pooled across independently transformed epochs. This isolates the security mechanism from network and orchestration failures.

**Current result.** Earlier 27B attempts reached depth-1 measurements but failed with `AssertionError: corrupted index slice (memory stomp during solve?)`. A corrected run using commit `63179b9` started at 09:46:55 BST on 11 August 2026 and is still running. The 12B run is queued behind it. No result from the current run is accepted yet.

**So what.** This experiment is a prerequisite, not a substitute, for two-node validation. If it passes, it will estimate a defensible rotation interval. A subsequent clean Toula–Koula deployment must still demonstrate identity, throughput, synchronization, and safe evidence collection.

**Evidence status.** Running on Toula. Koula is reachable and idle because this stage is deliberately single-node.

## Where the evidence leaves us

The project now supports five conclusions:

1. **Correct split execution is possible**, including across WAN conditions, but latency substantially reduces throughput.
2. **A correct distributed drafting protocol can still be slower than sequential decoding.** The repaired full-history method is a valid negative result.
3. **Distributed training is feasible across nodes.** Synchronous split fine-tuning is bit-exact in the validated cells; overlap can substantially reduce measured step cost, while FedAvg trades fewer synchronizations for a different optimization protocol.
4. **Intermediate representations and training surfaces can reveal substantial private information.** The strength depends strongly on the attacker and cannot be summarized by one inversion number.
5. **Orthogonal wrapping can defeat the evaluated passive wire attacker with little measured overhead, but its protection has a lifetime.** Rotation is scientifically motivated, yet the deployed ratchet has not passed its evidence gates.

The next decisive milestone is therefore not “run more cells.” It is a staged validation:

1. complete and validate the single-node rotation-lifetime study;
2. fix the deployed key schedule and evidence contract;
3. rerun a small two-node correctness gate;
4. only then run the full Toula–Koula rotation and performance matrix;
5. validate whether the privacy mechanism remains correct and useful during longer, converged distributed-training runs; the repository already contains multi-node training evidence, but the rotation matrix is not yet validated for that setting.

## Evidence map

The principal indexes behind this narrative are:

- [`EXPERIMENT_COMPARISON_MATRIX.md`](../EXPERIMENT_COMPARISON_MATRIX.md): experiment status, numerical summaries, and interpretation notes.
- [`paper-data/DATA_MANIFEST.md`](../paper-data/DATA_MANIFEST.md): inventory of committed artifacts and available producing commits.
- [`papers/EVIDENCE_EXECUTION_LEDGER.md`](EVIDENCE_EXECUTION_LEDGER.md): execution and acceptance ledger.
- [`paper-data/collected/toula/paper2/`](../paper-data/collected/toula/paper2/): distributed drafting correctness artifacts.
- [`paper-data/collected/toula/e9/`](../paper-data/collected/toula/e9/): deployed obfuscation and key-rotation artifacts.
- [`paper-data/collected/invalid/`](../paper-data/collected/invalid/): runs retained for diagnosis but excluded from claims.
