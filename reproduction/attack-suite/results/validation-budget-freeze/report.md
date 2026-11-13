# Validation budget review

2,720 cells × 256 documents = 696,320 document reconstructions; 87,040 batches.

Physical executions: **2,240 cells**, 573,440 document reconstructions, 71,680 batches.
Nominal all-executed occupancy: **169.7 GPU-hours**; duplicate execution savings: **63.3 GPU-hours**.
Estimated effective online GPU occupancy: **106.4 GPU-hours**.
Planning scenarios with 25% / 50% contingency: **133.0 / 159.6 GPU-hours**.

| Split depth | Nominal GPU-hours | Effective GPU-hours | +25% | +50% |
|---|---:|---:|---:|---:|
| 3 | 62.3 | 38.4 | 48.0 | 57.6 |
| 6 | 107.4 | 68.1 | 85.1 | 102.1 |

| Attack family | Online GPU-hours |
|---|---:|
| forward | 6.4 |
| joint | 44.9 |
| backward | 0.0 |
| sipit | 55.1 |

## Assumptions and limitations

- Source summary reports five-seed means per eight-document batch on NVIDIA GB10; average the available none/Gaussian defense means.
- New protocol uses the same architecture, sequence length and batch size as source timing measurements.
- When reuse_identical_activation_modes is true, inference forward/SIPIT rows reference the matching training execution with identical observations, head, arm and seed. Logical coverage remains unchanged; no second timing measurement is claimed.
- Nominal all-executed estimates use the original training/inference timings; effective estimates retain training timing and remove inference duplicate execution. Old training/inference differences are not present in the new paired probe.
- Scale linearly with batch count; a partial final batch is conservatively treated as full.
- Interpolate unmeasured K16 and SIPIT64 between measured budgets, without claiming exact linear runtime.
- Use joint true-gradient timings for shuffled, random, noisy and quantized controls; transformation and sampling overhead is unmeasured.
- Use the same timing estimate for exact/public head access and all new victim seeds; exact-head SIPIT may terminate sooner.
- Totals are online measured wall-time occupancy, not GPU busy time, FLOPs, or elapsed time on multiple devices.
- 25% and 50% contingencies are explicit planning scenarios, not confidence bounds or guaranteed upper limits.
- Exclude model downloads/loading, calibration, training, capture/export, scoring, report serialization, evidence hashing, and statistical analysis; these require separate measurement.
- Only ten independently trained victim runs and the selected documents are sampling units. Attack seeds and repeated arms are not independent observations.

## Provenance

```json
{
  "protocol_path": "experiments/validation_protocol.json",
  "protocol_sha256": "3a13b88ac6cd707c55a7866d81a2bb836392711a8e132b09c671e102d6db4635",
  "source_summary_path": "results/baseline-strengthening/analysis/summary.csv",
  "source_summary_sha256": "37e4273622b63e21bc4e0d143b37b3e4189ecb9ce1cdbe7a6c55d4bc3af4ca66",
  "script_sha256": "a845e300b538e43faf2b0a87ac36b3959694d372bc5977b2124d34b40f8e412b"
}
```
