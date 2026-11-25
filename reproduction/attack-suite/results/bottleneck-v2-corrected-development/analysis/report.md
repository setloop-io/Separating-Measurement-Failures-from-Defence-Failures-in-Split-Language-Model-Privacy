# Defended development pilot (trained splitbench defenders)

Tier: development_pilot. Matrix: none x trained-bottleneck defenses x splits [3] x seeds [11, 22, 33] on the 96 development wikitext-2 articles.

Frozen-corpus separation: development corpus sha256 `d11d88a6c0331024683e462814aff07d8158ae3829487d714bf77086bfbb7c77`; article ids disjoint from every frozen-corpus partition (asserted fail-closed against `experiments/validation_corpus_manifest.json`); identity checked against excluded_pilot_sha256: True.

Attacker access is matched across defenses: public calibration partition, pretrained public head only (final victim head not granted), identical query budgets. Per-cell access specs were verified byte-identical.

## Recovery (common seam-inversion baselines)

Accuracy is position-aligned token recovery on captured training-phase wires; intervals are percentile bootstrap over paired defense-minus-none seed differences (B=10000), conditional on this fixed article sample. Negative means the defense reduced recovery. Fewer than three seeds gives no interval (INSUFFICIENT_EVIDENCE), not equivalence.

| Split | Defense | Forward % | SIPIT % | Forward − none pp [CI] | SIPIT − none pp [CI] | Attack s | Victim + evaluation s |
|---|---|---:|---:|---:|---:|---:|---:|
| 3 | none | 92.19 | 3.65 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 9.29 | 15.07 |
| 3 | trained-bottleneck | 56.51 | 1.56 | -35.68 [-39.06, -33.59] | -2.08 [-4.69, 0.00] | 9.45 | 53.79 |

## Matched held-out utility

Perplexity deltas are defended-minus-undefended, paired over seeds (positive = the defense cost utility).

| Split | Defense | PPL before → after | ΔPPL vs none [CI] |
|---|---|---:|---:|
| 3 | none | 92.19 → 48.11 | 0.000 [0.000, 0.000] |
| 3 | trained-bottleneck | 289.54 → 179.58 | 131.478 [125.631, 135.681] |

## Defense-specific consumers

representation-privacy (bottleneck arm; no separate Gram-flattening defense claim) on public auxiliary/fresh scoring bottleneck releases; view-privacy de-mixing on converted mixup releases; replica-integrity clean / one-malicious / two-colluding arms with perturbation injected at the aggregation point.

| Split | Defense | Seed | Consumer summary |
|---|---|---:|---|
| 3 | trained-bottleneck | 11 | bottleneck/coordinate/fitted: accuracy=0.4375; bottleneck/coordinate/shuffled_calibration_labels: accuracy=0.2188; bottleneck/norm/fitted: accuracy=0.1875; bottleneck/norm/shuffled_calibration_labels: accuracy=0.0938; bottleneck/gram/fitted: accuracy=0.2188; bottleneck/gram/shuffled_calibration_labels: accuracy=0.1875; bottleneck/coordinate_mlp/fitted: accuracy=0.4375; bottleneck/coordinate_mlp/shuffled_calibration_labels: accuracy=0.0938 |
| 3 | trained-bottleneck | 22 | bottleneck/coordinate/fitted: accuracy=0.4375; bottleneck/coordinate/shuffled_calibration_labels: accuracy=0.1562; bottleneck/norm/fitted: accuracy=0.1250; bottleneck/norm/shuffled_calibration_labels: accuracy=0.1562; bottleneck/gram/fitted: accuracy=0.1875; bottleneck/gram/shuffled_calibration_labels: accuracy=0.0938; bottleneck/coordinate_mlp/fitted: accuracy=0.4375; bottleneck/coordinate_mlp/shuffled_calibration_labels: accuracy=0.1562 |
| 3 | trained-bottleneck | 33 | bottleneck/coordinate/fitted: accuracy=0.4375; bottleneck/coordinate/shuffled_calibration_labels: accuracy=0.1875; bottleneck/norm/fitted: accuracy=0.1562; bottleneck/norm/shuffled_calibration_labels: accuracy=0.0312; bottleneck/gram/fitted: accuracy=0.2500; bottleneck/gram/shuffled_calibration_labels: accuracy=0.1562; bottleneck/coordinate_mlp/fitted: accuracy=0.4375; bottleneck/coordinate_mlp/shuffled_calibration_labels: accuracy=0.1875 |

## Failures and nulls

All 6 planned cells completed with provenance, utility, matched access spec and both common baselines. None dropped.

## Scope and limitations

- development pilot for effect sizes and plumbing; not a confirmatory security claim
- three victim seeds; intervals are conditional on this fixed article sample
- trained defenders are in-repo re-implementations, not the original defense projects' code or weights
- the MINE reading in bottleneck provenance is a diagnostic, never a certificate
- split depth and defense are confounded with cell ordering on a single host

Regenerate: `python experiments/analyze_defended_pilot.py --root results/pilot-defended-v1`
