# Defended development pilot (trained splitbench defenders)

Tier: engineering_smoke. Matrix: none x gaussian x trained-bottleneck x secret-mixup x replica-median defenses x splits [3, 6] x seeds [11, 22, 33] on a SYNTHETIC smoke corpus (engineering tier; not security evidence).

Frozen-corpus separation: development corpus sha256 `65d868b97d59d44a7298319abf67e920f92042965f5a7e5c41f7b8a6771f42ba`; article ids disjoint from every frozen-corpus partition (asserted fail-closed against `experiments/validation_corpus_manifest.json`); identity checked against excluded_pilot_sha256: False.

Attacker access is matched across defenses: public calibration partition, pretrained public head only (final victim head not granted), identical query budgets. Per-cell access specs were verified byte-identical.

## Recovery (common seam-inversion baselines)

Accuracy is position-aligned token recovery on captured training-phase wires; intervals are percentile bootstrap over paired defense-minus-none seed differences (B=10000), conditional on this fixed article sample. Negative means the defense reduced recovery. Fewer than three seeds gives no interval (INSUFFICIENT_EVIDENCE), not equivalence.

| Split | Defense | Forward % | SIPIT % | Forward − none pp [CI] | SIPIT − none pp [CI] | Attack s | Victim + evaluation s |
|---|---|---:|---:|---:|---:|---:|---:|
| 3 | none | 98.96 | 16.67 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.07 | 0.18 |
| 3 | gaussian | 8.33 | 4.17 | -90.62 [-93.75, -87.50] | -12.50 [-21.88, 3.12] | 0.07 | 0.16 |
| 3 | trained-bottleneck | 1.04 | 0.00 | -97.92 [-100.00, -93.75] | -16.67 [-25.00, -3.12] | 0.25 | 0.18 |
| 3 | secret-mixup | 98.96 | 13.54 | 0.00 [0.00, 0.00] | -3.12 [-6.25, 0.00] | 0.59 | 0.19 |
| 3 | replica-median | 98.96 | 16.67 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.18 | 0.29 |
| 6 | none | 98.96 | 15.62 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.12 | 0.16 |
| 6 | gaussian | 8.33 | 7.29 | -90.62 [-93.75, -87.50] | -8.33 [-15.62, 6.25] | 0.11 | 0.16 |
| 6 | trained-bottleneck | 1.04 | 2.08 | -97.92 [-100.00, -93.75] | -13.54 [-21.88, 0.00] | 0.30 | 0.19 |
| 6 | secret-mixup | 97.92 | 12.50 | -1.04 [-3.12, 0.00] | -3.12 [-6.25, 0.00] | 0.64 | 0.19 |
| 6 | replica-median | 98.96 | 15.62 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.22 | 0.24 |

## Matched held-out utility

Perplexity deltas are defended-minus-undefended, paired over seeds (positive = the defense cost utility).

| Split | Defense | PPL before → after | ΔPPL vs none [CI] |
|---|---|---:|---:|
| 3 | none | 65.13 → 64.61 | 0.000 [0.000, 0.000] |
| 3 | gaussian | 65.15 → 65.04 | 0.440 [-1.403, 1.864] |
| 3 | trained-bottleneck | 63.66 → 63.64 | -0.961 [-2.163, -0.155] |
| 3 | secret-mixup | 65.25 → 64.63 | 0.029 [-0.196, 0.158] |
| 3 | replica-median | 65.13 → 64.69 | 0.084 [-0.006, 0.210] |
| 6 | none | 65.13 → 64.61 | 0.000 [0.000, 0.000] |
| 6 | gaussian | 65.26 → 65.04 | 0.438 [-1.326, 1.819] |
| 6 | trained-bottleneck | 63.72 → 63.71 | -0.896 [-2.289, 0.161] |
| 6 | secret-mixup | 65.21 → 64.64 | 0.031 [-0.188, 0.195] |
| 6 | replica-median | 65.13 → 64.65 | 0.041 [0.015, 0.080] |

## Defense-specific consumers

representation-privacy (bottleneck arm; no separate Gram-flattening defense claim) on public auxiliary/fresh scoring bottleneck releases; view-privacy de-mixing on converted mixup releases; replica-integrity clean / one-malicious / two-colluding arms with perturbation injected at the aggregation point.

| Split | Defense | Seed | Consumer summary |
|---|---|---:|---|
| 3 | trained-bottleneck | 11 | bottleneck/coordinate/fitted: accuracy=0.0000; bottleneck/coordinate/shuffled_calibration_labels: accuracy=0.0000; bottleneck/norm/fitted: accuracy=0.0000; bottleneck/norm/shuffled_calibration_labels: accuracy=0.0000; bottleneck/gram/fitted: accuracy=0.0000; bottleneck/gram/shuffled_calibration_labels: accuracy=0.0000; bottleneck/coordinate_mlp/fitted: accuracy=0.0000; bottleneck/coordinate_mlp/shuffled_calibration_labels: accuracy=0.0000 |
| 3 | secret-mixup | 11 | calibrated_mean: mse=0.0001319; naive_zero_mean: mse=0.0001361; oracle_pool_mean: mse=0.0001288; calibrated_mean: mse=8.006e-05 |
| 3 | replica-median | 11 | clean: accepted_error_rate=0.0; one-malicious: accepted_error_rate=0.0; two-colluding: accepted_error_rate=1.0 |
| 3 | trained-bottleneck | 22 | bottleneck/coordinate/fitted: accuracy=0.0000; bottleneck/coordinate/shuffled_calibration_labels: accuracy=0.0000; bottleneck/norm/fitted: accuracy=0.1667; bottleneck/norm/shuffled_calibration_labels: accuracy=0.0000; bottleneck/gram/fitted: accuracy=0.1667; bottleneck/gram/shuffled_calibration_labels: accuracy=0.0000; bottleneck/coordinate_mlp/fitted: accuracy=0.0000; bottleneck/coordinate_mlp/shuffled_calibration_labels: accuracy=0.0000 |
| 3 | secret-mixup | 22 | calibrated_mean: mse=0.0001174; naive_zero_mean: mse=0.0001535; oracle_pool_mean: mse=0.0001313; calibrated_mean: mse=7.426e-05 |
| 3 | replica-median | 22 | clean: accepted_error_rate=0.0; one-malicious: accepted_error_rate=0.0; two-colluding: accepted_error_rate=1.0 |
| 3 | trained-bottleneck | 33 | bottleneck/coordinate/fitted: accuracy=0.1667; bottleneck/coordinate/shuffled_calibration_labels: accuracy=0.0000; bottleneck/norm/fitted: accuracy=0.0000; bottleneck/norm/shuffled_calibration_labels: accuracy=0.0000; bottleneck/gram/fitted: accuracy=0.0000; bottleneck/gram/shuffled_calibration_labels: accuracy=0.0000; bottleneck/coordinate_mlp/fitted: accuracy=0.0000; bottleneck/coordinate_mlp/shuffled_calibration_labels: accuracy=0.0000 |
| 3 | secret-mixup | 33 | calibrated_mean: mse=0.0001294; naive_zero_mean: mse=0.0001601; oracle_pool_mean: mse=0.0001363; calibrated_mean: mse=8.223e-05 |
| 3 | replica-median | 33 | clean: accepted_error_rate=0.0; one-malicious: accepted_error_rate=0.0; two-colluding: accepted_error_rate=1.0 |
| 6 | trained-bottleneck | 11 | bottleneck/coordinate/fitted: accuracy=0.0000; bottleneck/coordinate/shuffled_calibration_labels: accuracy=0.0000; bottleneck/norm/fitted: accuracy=0.0000; bottleneck/norm/shuffled_calibration_labels: accuracy=0.0000; bottleneck/gram/fitted: accuracy=0.0000; bottleneck/gram/shuffled_calibration_labels: accuracy=0.0000; bottleneck/coordinate_mlp/fitted: accuracy=0.0000; bottleneck/coordinate_mlp/shuffled_calibration_labels: accuracy=0.0000 |
| 6 | secret-mixup | 11 | calibrated_mean: mse=0.0001419; naive_zero_mean: mse=0.0001471; oracle_pool_mean: mse=0.0001403; calibrated_mean: mse=8.751e-05 |
| 6 | replica-median | 11 | clean: accepted_error_rate=0.0; one-malicious: accepted_error_rate=0.0; two-colluding: accepted_error_rate=1.0 |
| 6 | trained-bottleneck | 22 | bottleneck/coordinate/fitted: accuracy=0.0000; bottleneck/coordinate/shuffled_calibration_labels: accuracy=0.0000; bottleneck/norm/fitted: accuracy=0.0000; bottleneck/norm/shuffled_calibration_labels: accuracy=0.0000; bottleneck/gram/fitted: accuracy=0.0000; bottleneck/gram/shuffled_calibration_labels: accuracy=0.0000; bottleneck/coordinate_mlp/fitted: accuracy=0.0000; bottleneck/coordinate_mlp/shuffled_calibration_labels: accuracy=0.0000 |
| 6 | secret-mixup | 22 | calibrated_mean: mse=0.0001232; naive_zero_mean: mse=0.0001577; oracle_pool_mean: mse=0.0001367; calibrated_mean: mse=7.832e-05 |
| 6 | replica-median | 22 | clean: accepted_error_rate=0.0; one-malicious: accepted_error_rate=0.0; two-colluding: accepted_error_rate=1.0 |
| 6 | trained-bottleneck | 33 | bottleneck/coordinate/fitted: accuracy=0.1667; bottleneck/coordinate/shuffled_calibration_labels: accuracy=0.0000; bottleneck/norm/fitted: accuracy=0.0000; bottleneck/norm/shuffled_calibration_labels: accuracy=0.1667; bottleneck/gram/fitted: accuracy=0.0000; bottleneck/gram/shuffled_calibration_labels: accuracy=0.1667; bottleneck/coordinate_mlp/fitted: accuracy=0.0000; bottleneck/coordinate_mlp/shuffled_calibration_labels: accuracy=0.0000 |
| 6 | secret-mixup | 33 | calibrated_mean: mse=0.0001384; naive_zero_mean: mse=0.0001659; oracle_pool_mean: mse=0.0001445; calibrated_mean: mse=8.889e-05 |
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
