# Stronger reconstruction baseline comparison

300 completed cells; all prediction files and source evidence verified. Scores independently recomputed from the sealed pilot truth. 40 paired forward/joint comparisons have identical public-head prefix work.

This is an exploratory replay of eight fixed pilot documents per mode, two depths, two defenses and five seeds. Each document has 32 tokens. The protocol was frozen before new outcomes; the source pilot had already been inspected.

## Findings

- At matched prefix-evaluation budgets, joint-minus-forward effects have adjusted intervals above zero in 8/8 conditions, below zero in 0, and including zero in 0. These are exploratory conditional estimates.
- At K=32, union proposals improve joint recovery over the legacy backward-only shortlist by 28.67–60.16 percentage points across the four training conditions. This is a descriptive implementation comparison, not evidence that adding observations necessarily helps every decoder.
- The 128-step SIPIT adaptation recovers 90.16–97.73% across conditions, using 7.69–10.68 times the online wall time of the corresponding K=32 forward beam. The two algorithms have different work budgets; the paired table must not be interpreted as an equal-cost ranking.
- High token accuracy is not full-prompt recovery: joint K=32 exact recovery of complete 32-token training sequences is 0.0–5.0% across conditions. Sensitive-span recovery remains unmeasured.
- These results strengthen the empirical attack baseline. They do not validate a scalar risk score, establish architectural generalization, or demonstrate a novel reconstruction principle.

## Access and methods

All arms are granted the same frozen public prefix weights, embeddings, sequence lengths and calibration documents. The observation ablation gives forward-only arms activations, backward-only direction decoding gradients, and joint arms both. No private weights, dropout masks or evaluation tokens enter reconstruction functions. Prediction files are persisted before scoring.

- **Ridge forward:** original calibration-based activation decoder.
- **Forward beam:** K forward proposals per position, beam width 2, activation residual ranking.
- **Joint union:** forward top K/2 plus backward top K/2, deduplicated and filled from forward proposals to exactly K. Position zero uses forward proposals. Same prefix evaluation budget as forward beam.
- **Legacy joint:** K backward proposals after position zero, reproduced as a diagnostic reference.
- **Direction backward:** shifted negative-gradient/embedding cosine heuristic, public majority at position zero. This is not the complete published TAG algorithm.
- **SIPIT:** independently implemented published sequential continuous/discrete reconstruction, at 32/128 iterations per token. Best verified residual fallback, stale public checkpoint and batching are explicit adaptations. No exact-recovery guarantee is claimed here.

## Recovery and measured compute

Accuracy is five-seed mean ± SD. Time is measured for a batch of eight documents; dividing by eight gives amortized throughput cost, not single-request latency. Setup-plus-attack cost includes the shared full-prefix loading/warmup and applicable ridge calibration; online cost excludes them. The backward heuristic could use a cheaper embedding-only loader, so shared startup is not its optimized standalone cost. CUDA event elapsed time includes host launch gaps. Counts are executed prefix token positions, not FLOP estimates.

| Depth | Defense | Mode | Arm | Recovery % ± SD | Exact sequence % | Online seconds / 8 docs | Shared setup + attack seconds | Peak MiB |
|---|---|---|---|---:|---:|---:|---:|---:|
| 3 | gaussian | inference | forward_beam_k32 | 95.47 ± 0.21 | 27.50 | 3.492 | 5.426 | 1663.9 |
| 3 | gaussian | inference | forward_beam_k8 | 95.39 ± 0.17 | 27.50 | 0.837 | 2.771 | 873.1 |
| 3 | gaussian | inference | ridge_forward | 87.89 ± 0.28 | 12.50 | 0.010 | 1.944 | 757.9 |
| 3 | gaussian | inference | sipit_steps128 | 97.73 ± 0.85 | 45.00 | 33.089 | 34.998 | 707.4 |
| 3 | gaussian | inference | sipit_steps32 | 92.89 ± 1.08 | 0.00 | 7.985 | 9.894 | 707.4 |
| 3 | gaussian | training | direction_backward | 19.61 ± 1.08 | 0.00 | 0.011 | 1.920 | 806.7 |
| 3 | gaussian | training | forward_beam_k32 | 92.66 ± 1.08 | 0.00 | 3.490 | 5.424 | 1664.6 |
| 3 | gaussian | training | forward_beam_k8 | 92.58 ± 1.14 | 2.50 | 0.837 | 2.771 | 873.9 |
| 3 | gaussian | training | joint_union_k32 | 95.08 ± 0.65 | 0.00 | 3.506 | 5.440 | 1713.7 |
| 3 | gaussian | training | joint_union_k8 | 94.92 ± 0.62 | 2.50 | 0.850 | 2.784 | 922.7 |
| 3 | gaussian | training | legacy_joint_k32 | 34.92 ± 1.87 | 0.00 | 3.500 | 5.434 | 1713.7 |
| 3 | gaussian | training | legacy_joint_k8 | 27.81 ± 3.24 | 0.00 | 0.847 | 2.781 | 922.7 |
| 3 | gaussian | training | ridge_forward | 87.19 ± 1.05 | 0.00 | 0.010 | 1.944 | 758.7 |
| 3 | gaussian | training | sipit_steps128 | 92.58 ± 0.48 | 0.00 | 35.712 | 37.621 | 708.1 |
| 3 | gaussian | training | sipit_steps32 | 90.86 ± 0.35 | 0.00 | 8.846 | 10.755 | 708.1 |
| 3 | none | inference | forward_beam_k32 | 95.62 ± 0.51 | 27.50 | 3.495 | 5.441 | 1621.9 |
| 3 | none | inference | forward_beam_k8 | 95.23 ± 0.17 | 25.00 | 0.836 | 2.782 | 831.1 |
| 3 | none | inference | ridge_forward | 87.81 ± 0.70 | 12.50 | 0.011 | 1.957 | 715.9 |
| 3 | none | inference | sipit_steps128 | 97.73 ± 1.05 | 50.00 | 37.314 | 39.223 | 665.4 |
| 3 | none | inference | sipit_steps32 | 93.28 ± 0.97 | 0.00 | 9.114 | 11.023 | 665.4 |
| 3 | none | training | direction_backward | 19.06 ± 2.64 | 0.00 | 0.011 | 1.920 | 764.7 |
| 3 | none | training | forward_beam_k32 | 92.73 ± 0.59 | 0.00 | 3.509 | 5.455 | 1622.6 |
| 3 | none | training | forward_beam_k8 | 92.58 ± 0.68 | 0.00 | 0.836 | 2.782 | 831.9 |
| 3 | none | training | joint_union_k32 | 95.47 ± 0.59 | 2.50 | 3.501 | 5.447 | 1671.7 |
| 3 | none | training | joint_union_k8 | 94.69 ± 0.71 | 2.50 | 0.850 | 2.796 | 880.7 |
| 3 | none | training | legacy_joint_k32 | 37.34 ± 2.88 | 0.00 | 3.501 | 5.447 | 1671.7 |
| 3 | none | training | legacy_joint_k8 | 29.69 ± 2.46 | 0.00 | 0.847 | 2.794 | 880.7 |
| 3 | none | training | ridge_forward | 86.64 ± 1.12 | 0.00 | 0.013 | 1.959 | 716.7 |
| 3 | none | training | sipit_steps128 | 93.59 ± 1.02 | 0.00 | 31.870 | 33.779 | 666.1 |
| 3 | none | training | sipit_steps32 | 91.64 ± 1.62 | 0.00 | 8.164 | 10.072 | 664.5 |
| 6 | gaussian | inference | forward_beam_k32 | 91.25 ± 0.45 | 12.50 | 6.872 | 8.830 | 1662.6 |
| 6 | gaussian | inference | forward_beam_k8 | 90.62 ± 0.55 | 12.50 | 1.623 | 3.581 | 871.6 |
| 6 | gaussian | inference | ridge_forward | 79.06 ± 0.59 | 0.00 | 0.010 | 1.968 | 756.7 |
| 6 | gaussian | inference | sipit_steps128 | 96.56 ± 1.82 | 42.50 | 57.859 | 59.789 | 707.1 |
| 6 | gaussian | inference | sipit_steps32 | 91.64 ± 1.55 | 2.50 | 14.512 | 16.441 | 707.1 |
| 6 | gaussian | training | direction_backward | 53.12 ± 1.24 | 0.00 | 0.011 | 1.941 | 805.0 |
| 6 | gaussian | training | forward_beam_k32 | 89.38 ± 0.89 | 0.00 | 6.864 | 8.822 | 1663.4 |
| 6 | gaussian | training | forward_beam_k8 | 88.91 ± 0.81 | 0.00 | 1.622 | 3.580 | 872.4 |
| 6 | gaussian | training | joint_union_k32 | 95.94 ± 0.35 | 5.00 | 6.896 | 8.854 | 1712.5 |
| 6 | gaussian | training | joint_union_k8 | 95.62 ± 0.58 | 5.00 | 1.634 | 3.592 | 921.5 |
| 6 | gaussian | training | legacy_joint_k32 | 67.27 ± 1.69 | 0.00 | 6.878 | 8.837 | 1712.5 |
| 6 | gaussian | training | legacy_joint_k8 | 62.34 ± 2.10 | 0.00 | 1.629 | 3.587 | 921.5 |
| 6 | gaussian | training | ridge_forward | 82.19 ± 1.06 | 0.00 | 0.010 | 1.969 | 757.5 |
| 6 | gaussian | training | sipit_steps128 | 90.16 ± 1.52 | 0.00 | 55.623 | 57.553 | 707.8 |
| 6 | gaussian | training | sipit_steps32 | 86.56 ± 1.25 | 0.00 | 14.525 | 16.455 | 707.8 |
| 6 | none | inference | forward_beam_k32 | 90.94 ± 0.75 | 12.50 | 6.870 | 8.845 | 1621.6 |
| 6 | none | inference | forward_beam_k8 | 90.62 ± 0.73 | 10.00 | 1.627 | 3.602 | 830.5 |
| 6 | none | inference | ridge_forward | 78.52 ± 0.48 | 0.00 | 0.010 | 1.985 | 715.7 |
| 6 | none | inference | sipit_steps128 | 96.48 ± 1.93 | 32.50 | 55.110 | 57.040 | 665.1 |
| 6 | none | inference | sipit_steps32 | 91.17 ± 1.09 | 0.00 | 13.797 | 15.726 | 665.1 |
| 6 | none | training | direction_backward | 51.48 ± 2.20 | 0.00 | 0.011 | 1.941 | 762.3 |
| 6 | none | training | forward_beam_k32 | 90.16 ± 0.75 | 0.00 | 6.868 | 8.843 | 1622.3 |
| 6 | none | training | forward_beam_k8 | 89.53 ± 0.58 | 0.00 | 1.618 | 3.594 | 831.3 |
| 6 | none | training | joint_union_k32 | 96.25 ± 0.45 | 5.00 | 6.871 | 8.846 | 1669.8 |
| 6 | none | training | joint_union_k8 | 96.02 ± 0.51 | 2.50 | 1.705 | 3.680 | 878.7 |
| 6 | none | training | legacy_joint_k32 | 67.50 ± 1.22 | 0.00 | 6.866 | 8.842 | 1669.8 |
| 6 | none | training | legacy_joint_k8 | 61.02 ± 1.36 | 0.00 | 1.629 | 3.604 | 878.8 |
| 6 | none | training | ridge_forward | 83.28 ± 1.28 | 0.00 | 0.010 | 1.986 | 714.8 |
| 6 | none | training | sipit_steps128 | 91.09 ± 1.73 | 0.00 | 52.801 | 54.731 | 664.2 |
| 6 | none | training | sipit_steps32 | 88.52 ± 1.02 | 0.00 | 13.735 | 15.665 | 665.8 |

## Paired effects

Intervals use the frozen family of 24 comparisons, Bonferroni-adjusted Student t over five seeds, conditional on these documents. Joint/forward beam comparisons match prefix evaluations, with gradient proposal overhead measured. SIPIT/beam comparisons have different compute and must be read alongside the timing curves.

| Depth | Defense | Mode | Attack | Baseline | Difference pp [adjusted CI] |
|---|---|---|---|---|---:|
| 3 | none | training | joint_union_k8 | forward_beam_k8 | 2.11 [1.00, 3.22] |
| 3 | none | training | joint_union_k8 | direction_backward | 75.62 [67.05, 84.20] |
| 3 | none | training | joint_union_k32 | forward_beam_k32 | 2.73 [1.22, 4.25] |
| 3 | none | training | joint_union_k32 | direction_backward | 76.41 [67.88, 84.93] |
| 3 | none | training | sipit_steps128 | forward_beam_k32 | 0.86 [-3.73, 5.45] |
| 3 | none | inference | sipit_steps128 | forward_beam_k32 | 2.11 [-1.24, 5.46] |
| 3 | gaussian | training | joint_union_k8 | forward_beam_k8 | 2.34 [0.38, 4.30] |
| 3 | gaussian | training | joint_union_k8 | direction_backward | 75.31 [70.64, 79.98] |
| 3 | gaussian | training | joint_union_k32 | forward_beam_k32 | 2.42 [0.39, 4.46] |
| 3 | gaussian | training | joint_union_k32 | direction_backward | 75.47 [70.80, 80.14] |
| 3 | gaussian | training | sipit_steps128 | forward_beam_k32 | -0.08 [-3.29, 3.13] |
| 3 | gaussian | inference | sipit_steps128 | forward_beam_k32 | 2.27 [0.23, 4.30] |
| 6 | none | training | joint_union_k8 | forward_beam_k8 | 6.48 [3.25, 9.72] |
| 6 | none | training | joint_union_k8 | direction_backward | 44.53 [37.86, 51.21] |
| 6 | none | training | joint_union_k32 | forward_beam_k32 | 6.09 [2.52, 9.66] |
| 6 | none | training | joint_union_k32 | direction_backward | 44.77 [38.06, 51.48] |
| 6 | none | training | sipit_steps128 | forward_beam_k32 | 0.94 [-4.92, 6.79] |
| 6 | none | inference | sipit_steps128 | forward_beam_k32 | 5.55 [-1.21, 12.30] |
| 6 | gaussian | training | joint_union_k8 | forward_beam_k8 | 6.72 [3.51, 9.93] |
| 6 | gaussian | training | joint_union_k8 | direction_backward | 42.50 [37.67, 47.33] |
| 6 | gaussian | training | joint_union_k32 | forward_beam_k32 | 6.56 [4.18, 8.95] |
| 6 | gaussian | training | joint_union_k32 | direction_backward | 42.81 [38.38, 47.25] |
| 6 | gaussian | training | sipit_steps128 | forward_beam_k32 | 0.78 [-2.83, 4.40] |
| 6 | gaussian | inference | sipit_steps128 | forward_beam_k32 | 5.31 [-0.61, 11.23] |

## Candidate coverage diagnostic

An oracle allowed perfect selection within these candidate lists cannot exceed the ceiling below. This scoring-only diagnostic does not tune proposals.

| Depth | Defense | Mode | Arm | Candidate ceiling % | Actual recovery % |
|---|---|---|---|---:|---:|
| 3 | gaussian | inference | forward_beam_k32 | 95.47 | 95.47 |
| 3 | gaussian | inference | forward_beam_k8 | 95.39 | 95.39 |
| 3 | gaussian | training | forward_beam_k32 | 94.06 | 92.66 |
| 3 | gaussian | training | forward_beam_k8 | 93.59 | 92.58 |
| 3 | gaussian | training | joint_union_k32 | 96.48 | 95.08 |
| 3 | gaussian | training | joint_union_k8 | 96.09 | 94.92 |
| 3 | gaussian | training | legacy_joint_k32 | 36.17 | 34.92 |
| 3 | gaussian | training | legacy_joint_k8 | 28.75 | 27.81 |
| 3 | none | inference | forward_beam_k32 | 95.62 | 95.62 |
| 3 | none | inference | forward_beam_k8 | 95.23 | 95.23 |
| 3 | none | training | forward_beam_k32 | 93.83 | 92.73 |
| 3 | none | training | forward_beam_k8 | 93.59 | 92.58 |
| 3 | none | training | joint_union_k32 | 96.56 | 95.47 |
| 3 | none | training | joint_union_k8 | 95.70 | 94.69 |
| 3 | none | training | legacy_joint_k32 | 38.36 | 37.34 |
| 3 | none | training | legacy_joint_k8 | 30.62 | 29.69 |
| 6 | gaussian | inference | forward_beam_k32 | 91.33 | 91.25 |
| 6 | gaussian | inference | forward_beam_k8 | 90.70 | 90.62 |
| 6 | gaussian | training | forward_beam_k32 | 90.94 | 89.38 |
| 6 | gaussian | training | forward_beam_k8 | 90.00 | 88.91 |
| 6 | gaussian | training | joint_union_k32 | 97.58 | 95.94 |
| 6 | gaussian | training | joint_union_k8 | 96.80 | 95.62 |
| 6 | gaussian | training | legacy_joint_k32 | 69.22 | 67.27 |
| 6 | gaussian | training | legacy_joint_k8 | 63.52 | 62.34 |
| 6 | none | inference | forward_beam_k32 | 91.33 | 90.94 |
| 6 | none | inference | forward_beam_k8 | 90.70 | 90.62 |
| 6 | none | training | forward_beam_k32 | 91.17 | 90.16 |
| 6 | none | training | forward_beam_k8 | 90.39 | 89.53 |
| 6 | none | training | joint_union_k32 | 97.34 | 96.25 |
| 6 | none | training | joint_union_k8 | 96.88 | 96.02 |
| 6 | none | training | legacy_joint_k32 | 68.91 | 67.50 |
| 6 | none | training | legacy_joint_k8 | 62.42 | 61.02 |

## Fidelity and prior art

Ten fidelity cases passed: eight tiny-model fixtures and two pretrained public-calibration prefixes, spanning both depths. Recovered IDs and iteration counts matched the pinned author implementation; the prefix adapter also matched full-model hidden states. The sealed fidelity artifact is linked in analysis.json.

[SIPIT paper](https://arxiv.org/abs/2510.15511), Algorithm 1 and Appendix E; [author code](https://github.com/giorgosnikolaou/SIPIT) pinned at `820683156b7257313046a4fb3c492e52519525b7`. `experiments/verify_sipit_reference.py` checks predictions, iteration counts and head equivalence on noiseless fixtures against that external checkout. This establishes kernel fidelity on those fixtures, not reproduction of the published large-model benchmark.

[BiSR/SIP](https://arxiv.org/abs/2409.00960) is a relevant additional baseline with public-checkpoint mismatch; its trained inversion model requires a substantially larger auxiliary training study than the pilot calibration sample. [Hidden No More](https://arxiv.org/abs/2505.18332) provides sequential vocabulary matching, but its proceedings-linked author repository returned 404 during this review.

## Runtime provenance correction

The frozen manifests used a distribution inventory that overwrote earlier packages with later shadowed installations. They consequently list Transformers 5.15.0 from the shared base environment. Separate post-run inspections using the same task interpreters resolve Transformers 4.57.6 from the task virtual environment, alongside PyTorch 2.11.0+cu128 and SciPy 1.18.1. Imported paths, versions, duplicate-distribution order and module hashes are retained in analysis.json. These inspections are not telemetry from the original running processes. The inventory precedence bug is fixed for future evidence; frozen manifests are preserved.

## Limitations

- Exploratory reuse of a previously inspected pilot; requires fresh held-out replication
- Training captures include dropout and evolving private weights; inference uses final weights in eval mode
- No private client checkpoint, labels, server-tail weights or dropout masks granted to attacks
- Backward direction is a heuristic, not full published TAG gradient matching
- Finite beam search is approximate and candidate construction is under test
- SIPIT exact-recovery guarantee does not apply to noisy/stale public-checkpoint adaptation
- Depth is assigned to host; no causal depth claim
- Eight fixed documents and five seeds do not establish population generalization
