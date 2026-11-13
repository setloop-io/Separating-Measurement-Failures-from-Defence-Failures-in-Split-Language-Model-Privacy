# Defended development pilot (trained splitbench defenders)

Tier: development_pilot. Matrix: none x gaussian x trained-bottleneck x secret-mixup x replica-median defenses x splits [3, 6] x seeds [11, 22, 33] on the 96 development wikitext-2 articles.

Frozen-corpus separation: development corpus sha256 `d11d88a6c0331024683e462814aff07d8158ae3829487d714bf77086bfbb7c77`; article ids disjoint from every frozen-corpus partition (asserted fail-closed against `experiments/validation_corpus_manifest.json`); identity checked against excluded_pilot_sha256: True.

Attacker access is matched across defenses: public calibration partition, pretrained public head only (final victim head not granted), identical query budgets. Per-cell access specs were verified byte-identical.

## Recovery (common seam-inversion baselines)

Accuracy is position-aligned token recovery on captured training-phase wires; intervals are percentile bootstrap over paired defense-minus-none seed differences (B=10000), conditional on this fixed article sample. Negative means the defense reduced recovery. Fewer than three seeds gives no interval (INSUFFICIENT_EVIDENCE), not equivalence.

| Split | Defense | Forward % | SIPIT % | Forward − none pp [CI] | SIPIT − none pp [CI] | Attack s | Wall s |
|---|---|---:|---:|---:|---:|---:|---:|
| 3 | none | 91.60 | 93.46 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 47.63 | 4.77 |
| 3 | gaussian | 92.32 | 93.16 | 0.72 [0.20, 1.66] | -0.29 [-0.49, -0.20] | 47.39 | 4.68 |
| 3 | trained-bottleneck | 0.00 | 0.00 | -91.60 [-92.09, -91.02] | -93.46 [-93.65, -93.36] | 47.16 | 15.45 |
| 3 | secret-mixup | 85.81 | 84.90 | -5.79 [-6.74, -4.49] | -8.56 [-9.28, -7.13] | 244.86 | 5.57 |
| 3 | replica-median | 24.90 | 4.17 | -66.70 [-67.68, -64.94] | -89.29 [-89.94, -88.67] | 53.65 | 10.60 |
| 6 | none | 87.60 | 89.29 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 88.92 | 4.72 |
| 6 | gaussian | 87.79 | 88.77 | 0.20 [-1.66, 1.76] | -0.52 [-0.88, 0.10] | 89.19 | 4.73 |
| 6 | trained-bottleneck | 0.00 | 0.00 | -87.60 [-88.57, -86.52] | -89.29 [-90.14, -88.67] | 89.85 | 15.48 |
| 6 | secret-mixup | 80.37 | 78.03 | -7.23 [-8.59, -5.66] | -11.26 [-13.57, -10.06] | 286.19 | 5.72 |
| 6 | replica-median | 16.37 | 3.35 | -71.22 [-71.68, -70.51] | -85.94 [-86.62, -84.86] | 95.95 | 9.31 |

## Held-out utility

Perplexity deltas are defended-minus-undefended, paired over seeds (positive = the defense cost utility).

| Split | Defense | PPL before → after | ΔPPL vs none [CI] |
|---|---|---:|---:|
| 3 | none | 92.19 → 48.32 | 0.000 [0.000, 0.000] |
| 3 | gaussian | 92.59 → 48.66 | 0.346 [-1.426, 1.874] |
| 3 | trained-bottleneck | 90.57 → 66407696.94 | 66407648.625 [38439400.796, 101033823.113] |
| 3 | secret-mixup | 68.77 → 83.39 | 35.078 [11.014, 67.963] |
| 3 | replica-median | 90.57 → 97.43 | 49.113 [6.703, 97.342] |
| 6 | none | 92.19 → 48.32 | 0.000 [0.000, 0.000] |
| 6 | gaussian | 92.41 → 48.39 | 0.077 [-1.221, 0.780] |
| 6 | trained-bottleneck | 90.57 → 124342446.73 | 124342398.410 [16933224.632, 260444642.698] |
| 6 | secret-mixup | 68.77 → 84.64 | 36.321 [15.961, 71.126] |
| 6 | replica-median | 90.57 → 93.64 | 45.323 [7.191, 93.578] |

## Defense-specific consumers

representation-privacy (bottleneck + gram_flattening arms) on converted bottleneck releases; view-privacy de-mixing on converted mixup releases; replica-integrity clean / one-malicious / two-colluding arms with perturbation injected at the aggregation point.

| Split | Defense | Seed | Consumer summary |
|---|---|---:|---|
| 3 | trained-bottleneck | 11 | bottleneck: label_accuracy=None; gram_flattening: label_accuracy=None |
| 3 | secret-mixup | 11 | calibrated_mean: mse=51.84; naive_zero_mean: mse=55.76; oracle_pool_mean: mse=52.78; calibrated_mean: mse=36.69 |
| 3 | replica-median | 11 | clean: accepted_error_rate=0.0; one-malicious: accepted_error_rate=0.0; two-colluding: accepted_error_rate=1.0 |
| 3 | trained-bottleneck | 22 | bottleneck: label_accuracy=None; gram_flattening: label_accuracy=None |
| 3 | secret-mixup | 22 | calibrated_mean: mse=67.49; naive_zero_mean: mse=73.7; oracle_pool_mean: mse=68.92; calibrated_mean: mse=46.57 |
| 3 | replica-median | 22 | clean: accepted_error_rate=0.0; one-malicious: accepted_error_rate=0.0; two-colluding: accepted_error_rate=1.0 |
| 3 | trained-bottleneck | 33 | bottleneck: label_accuracy=None; gram_flattening: label_accuracy=None |
| 3 | secret-mixup | 33 | calibrated_mean: mse=58.07; naive_zero_mean: mse=61.34; oracle_pool_mean: mse=57.96; calibrated_mean: mse=39.67 |
| 3 | replica-median | 33 | clean: accepted_error_rate=0.0; one-malicious: accepted_error_rate=0.0; two-colluding: accepted_error_rate=1.0 |
| 6 | trained-bottleneck | 11 | bottleneck: label_accuracy=None; gram_flattening: label_accuracy=None |
| 6 | secret-mixup | 11 | calibrated_mean: mse=52.85; naive_zero_mean: mse=59.6; oracle_pool_mean: mse=54.12; calibrated_mean: mse=35.88 |
| 6 | replica-median | 11 | clean: accepted_error_rate=0.0; one-malicious: accepted_error_rate=0.0; two-colluding: accepted_error_rate=1.0 |
| 6 | trained-bottleneck | 22 | bottleneck: label_accuracy=None; gram_flattening: label_accuracy=None |
| 6 | secret-mixup | 22 | calibrated_mean: mse=77.85; naive_zero_mean: mse=89.31; oracle_pool_mean: mse=81.16; calibrated_mean: mse=51.17 |
| 6 | replica-median | 22 | clean: accepted_error_rate=0.0; one-malicious: accepted_error_rate=0.0; two-colluding: accepted_error_rate=1.0 |
| 6 | trained-bottleneck | 33 | bottleneck: label_accuracy=None; gram_flattening: label_accuracy=None |
| 6 | secret-mixup | 33 | calibrated_mean: mse=64.68; naive_zero_mean: mse=70.89; oracle_pool_mean: mse=64.06; calibrated_mean: mse=40.61 |
| 6 | replica-median | 33 | clean: accepted_error_rate=0.0; one-malicious: accepted_error_rate=0.0; two-colluding: accepted_error_rate=1.0 |

## Failures and nulls

All 30 planned cells completed with provenance, utility, matched access spec and both common baselines. None dropped.

## Scope and limitations

- development pilot for effect sizes and plumbing; not a confirmatory security claim
- three victim seeds; intervals are conditional on this fixed article sample
- trained defenders are in-repo re-implementations, not the original defense projects' code or weights
- the MINE reading in bottleneck provenance is a diagnostic, never a certificate
- split depth and defense are confounded with cell ordering on a single host

Regenerate: `python experiments/analyze_defended_pilot.py --root results/pilot-defended-v1`
