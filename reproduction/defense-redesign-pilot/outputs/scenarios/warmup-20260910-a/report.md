# Warmup/stability engineering report: warmup-20260910-a

Completed 22/22 records; failures: 0.
Execution commit: `ba8b23d0e84a3a29392973fc81f413f3eb7846a5`. Source, artifacts, pairing and panel separation verified.

Two shared setups and twenty branch runs; 16 warmup + 16 branch updates per model.
These are engineering diagnostics, not convergence results, defense selection, or instrument qualification.
Privacy remains **undetermined**. Null repeats share text and models; pooling below is descriptive.

| Sequences per panel | Null checks within ±0.02 | 95th percentile absolute advantage | Maximum absolute advantage |
|---|---|---|---|
| 16 | 13/288 | 0.6793 | 0.7786 |
| 64 | 46/288 | 0.2613 | 0.4374 |
| 256 | 74/288 | 0.1876 | 0.2582 |
| 1024 | 117/288 | 0.1031 | 0.1215 |

All repetitions are retained. No null-band change or scientific qualification is inferred.

| Seed | Initial–warm basis absolute cosine | Initial basis validation r | Warm basis validation r |
|---|---|---|---|
| 410301 | 0.4821 | 0.7583 | 0.7797 |
| 410302 | 0.5717 | 0.8272 | 0.8144 |

| Seed | Variant | Mode | Final loss (nats/byte) | Final basis cosine to initial | Final basis cosine to warm |
|---|---|---|---|---|---|
| 410301 | all_local | all_local | 3.3452 | — | — |
| 410301 | d0 | frozen_random | 3.3159 | 0.3547 | 0.8069 |
| 410301 | d0 | removed_remote | 3.3342 | 0.3962 | 0.8654 |
| 410301 | d0 | trained | 3.3178 | 0.3518 | 0.8021 |
| 410301 | d1_initial_basis | frozen_random | 3.3159 | 0.3537 | 0.8065 |
| 410301 | d1_initial_basis | removed_remote | 3.3342 | 0.3957 | 0.8654 |
| 410301 | d1_initial_basis | trained | 3.3176 | 0.3504 | 0.8015 |
| 410301 | d1_warm_basis | frozen_random | 3.3159 | 0.3560 | 0.8080 |
| 410301 | d1_warm_basis | removed_remote | 3.3341 | 0.3967 | 0.8659 |
| 410301 | d1_warm_basis | trained | 3.3177 | 0.3533 | 0.8037 |
| 410302 | all_local | all_local | 3.3204 | — | — |
| 410302 | d0 | frozen_random | 3.2642 | 0.4225 | 0.7490 |
| 410302 | d0 | removed_remote | 3.2702 | 0.4453 | 0.7686 |
| 410302 | d0 | trained | 3.2672 | 0.4289 | 0.7527 |
| 410302 | d1_initial_basis | frozen_random | 3.2643 | 0.4231 | 0.7484 |
| 410302 | d1_initial_basis | removed_remote | 3.2703 | 0.4454 | 0.7686 |
| 410302 | d1_initial_basis | trained | 3.2674 | 0.4300 | 0.7522 |
| 410302 | d1_warm_basis | frozen_random | 3.2643 | 0.4215 | 0.7475 |
| 410302 | d1_warm_basis | removed_remote | 3.2702 | 0.4451 | 0.7682 |
| 410302 | d1_warm_basis | trained | 3.2674 | 0.4285 | 0.7515 |

D1 uses a fixed basis throughout each branch. Final re-fitting is diagnostic only.
Removed controls bypass the remote module after shared warmup. All-local has its own warmup.
No inference is made about converged utility, causal-gain eligibility, or defense effectiveness.

Unevaluated in this stage: next-byte classes; returned values; quantizer metadata; full-precision inbound/outbound gradients; evolving remote state; optimizer state; timing; active/adaptive attackers; within-run joint tensors; cross-run attacks; history accumulation; canary membership; natural-sequence membership; unrestricted semantics.

See [summary](summary.json), [source](source.json), and [verification](verification.json).
