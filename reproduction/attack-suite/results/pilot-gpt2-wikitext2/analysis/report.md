# Pretrained GPT-2 / WikiText-2 pilot

Completed 100/100 cells: 2 split depths × 2 defenses × 5 seeds × 5 mode/attack combinations.

96 articles: 32 calibration, 32 victim-training, 32 held-out evaluation. Each scored window has 32 tokens. The official validation/test splits were reserved.

The engine is the clean V.0.1 release. All input and artifact hashes verified; all runs used identical initial public model weights.

## Pilot findings

1. Activation-only VMI recovered roughly 80–89% of position-aligned tokens across the tested conditions, compared with majority controls of 3.42% (training) and 4.30% (inference).
2. At Gaussian sigma=0.1, every paired defense-minus-undefended interval includes zero after the frozen multiplicity correction. This pilot does not demonstrate a recovery reduction at that setting; it does not establish equivalence or evaluate stronger noise.
3. The joint decoder improves on its backward-only diagnostic but performs below forward-only VMI. The post-hoc candidate-set audit below identifies the limiting mechanism.
4. Moving the split from layer 3 to layer 6 decreases VMI recovery while increasing the backward/joint decoder recovery in this sample. Treat this as a follow-up hypothesis: depth is assigned to host, and these are particular implementations.
5. Held-out perplexity improves after training. The undefended runs produce matching utility at both depths; this is useful evidence that changing the split preserves the underlying training computation.

The next methodological priority is a stronger joint candidate construction and an attack-budget comparison. This pilot does not support a general claim that joint observations are harmless or that the Gaussian family is ineffective.

## Recovery and paired uncertainty

Accuracy is mean ± sample SD across five seeds. Difference intervals use the frozen 60-comparison family across both depths; they are conditional on this corpus, not population guarantees. Negative defense-minus-none values mean reduced recovery.

| Split | Mode | Defense | Arm | Accuracy % ± SD | Majority % | Attack − majority pp [CI] | Defense − none pp [CI] | Attack seconds |
|---|---|---|---|---:|---:|---:|---:|---:|
| 3 | inference | gaussian | vmi | 88.95 ± 0.35 | 4.30 | 84.65 [83.22, 86.08] | 0.18 [-1.90, 2.25] | 1.73 |
| 3 | inference | gaussian | wbi | 58.05 ± 0.05 | 4.30 | 53.75 [53.53, 53.97] | 0.06 [-0.47, 0.59] | 6.14 |
| 3 | inference | none | vmi | 88.77 ± 0.34 | 4.30 | 84.47 [83.11, 85.84] | 0.00 [0.00, 0.00] | 1.71 |
| 3 | inference | none | wbi | 57.99 ± 0.16 | 4.30 | 53.69 [53.04, 54.34] | 0.00 [0.00, 0.00] | 5.30 |
| 3 | training | gaussian | joint | 25.90 ± 1.04 | 3.42 | 22.48 [18.28, 26.69] | -0.64 [-5.61, 4.32] | 6.89 |
| 3 | training | gaussian | vmi | 86.11 ± 0.59 | 3.42 | 82.70 [80.32, 85.07] | 0.55 [-2.68, 3.77] | 1.72 |
| 3 | training | gaussian | wbi | 54.77 ± 0.24 | 3.42 | 51.35 [50.37, 52.33] | 0.02 [-0.49, 0.53] | 6.06 |
| 3 | training | none | joint | 26.54 ± 0.92 | 3.42 | 23.12 [19.39, 26.86] | 0.00 [0.00, 0.00] | 6.97 |
| 3 | training | none | vmi | 85.57 ± 0.37 | 3.42 | 82.15 [80.66, 83.63] | 0.00 [0.00, 0.00] | 1.74 |
| 3 | training | none | wbi | 54.75 ± 0.13 | 3.42 | 51.33 [50.80, 51.86] | 0.00 [0.00, 0.00] | 5.31 |
| 6 | inference | gaussian | vmi | 80.04 ± 0.39 | 4.30 | 75.74 [74.15, 77.33] | 0.47 [-2.28, 3.22] | 1.75 |
| 6 | inference | gaussian | wbi | 56.13 ± 0.11 | 4.30 | 51.84 [51.40, 52.27] | -0.04 [-0.76, 0.68] | 6.33 |
| 6 | inference | none | vmi | 79.57 ± 0.36 | 4.30 | 75.27 [73.81, 76.73] | 0.00 [0.00, 0.00] | 1.74 |
| 6 | inference | none | wbi | 56.17 ± 0.17 | 4.30 | 51.88 [51.17, 52.58] | 0.00 [0.00, 0.00] | 5.34 |
| 6 | training | gaussian | joint | 59.51 ± 1.13 | 3.42 | 56.09 [51.55, 60.64] | 0.35 [-2.51, 3.22] | 8.55 |
| 6 | training | gaussian | vmi | 79.75 ± 0.34 | 3.42 | 76.33 [74.97, 77.68] | -0.16 [-2.94, 2.63] | 1.76 |
| 6 | training | gaussian | wbi | 53.44 ± 0.23 | 3.42 | 50.02 [49.08, 50.96] | -0.06 [-1.81, 1.70] | 6.54 |
| 6 | training | none | joint | 59.16 ± 0.92 | 3.42 | 55.74 [52.01, 59.48] | 0.00 [0.00, 0.00] | 8.63 |
| 6 | training | none | vmi | 79.90 ± 0.45 | 3.42 | 76.48 [74.65, 78.32] | 0.00 [0.00, 0.00] | 1.78 |
| 6 | training | none | wbi | 53.50 ± 0.21 | 3.42 | 50.08 [49.22, 50.93] | 0.00 [0.00, 0.00] | 5.38 |

## Held-out utility

Each model condition is counted once per seed, despite appearing in several attack records. Perplexities below are arithmetic means of per-run perplexities.

| Split | Defense | CE before → after | Perplexity before → after | Training seconds |
|---|---|---:|---:|---:|
| 3 | none | 4.524 → 3.932 | 92.19 → 51.03 | 2.22 |
| 3 | gaussian | 4.527 → 3.944 | 92.49 → 51.63 | 2.14 |
| 6 | none | 4.524 → 3.932 | 92.19 → 51.03 | 2.41 |
| 6 | gaussian | 4.526 → 3.945 | 92.35 → 51.69 | 2.41 |

## Joint observation diagnostics

These are descriptive comparisons with the implemented single-surface decoders, not equal-compute comparisons to the published strongest attacks.

| Split | Defense | Joint − forward pp | Joint − backward pp |
|---|---|---:|---:|
| 3 | gaussian | -60.21 | 6.31 |
| 3 | none | -59.02 | 7.11 |
| 6 | gaussian | -20.23 | 7.64 |
| 6 | none | -20.74 | 7.87 |

## Post-hoc candidate-filter diagnostic

The joint decoder considers only backward top-4 candidates after position zero. The following oracle ceiling grants perfect position-zero recovery and perfect selection among all remaining candidates. It is computed from the original saved captures, without changing the frozen attacks or budgets.

| Split | Defense | Candidate-set oracle ceiling % | Actual joint % | Forward-only % |
|---|---|---:|---:|---:|
| 3 | none | 28.98 | 26.54 | 85.57 |
| 3 | gaussian | 28.48 | 25.90 | 86.11 |
| 6 | none | 61.86 | 59.16 | 79.90 |
| 6 | gaussian | 62.21 | 59.51 | 79.75 |

This identifies a limitation of candidate construction in this implementation. Increasing beam width alone cannot overcome a missing correct token at fixed top-k. It does not show that the combined observation contains less information or establish an optimal joint attack.


## Scope and limitations

- Training and inference score different document partitions and training-time states; their contrast is descriptive, not causal
- Split depth is confounded with host; same GPU type and package versions are checked but not a host crossover
- Published attacks are local adaptations, not certified reproductions
- Beam search and optimizer steps are not matched compute
- Small pilot corpus and five seeds do not establish generalization or equivalence

## Primary model and data sources

- [GPT-2 model card](https://huggingface.co/openai-community/gpt2)
- [WikiText dataset card](https://huggingface.co/datasets/Salesforce/wikitext)

Regenerate: `python experiments/analyze_pilot.py --root results/pilot-gpt2-wikitext2`
