# Validation budget review

2,720 cells × 256 documents = 696,320 document reconstructions; 87,040 batches.

Estimated online GPU occupancy: **169.7 GPU-hours**.
Planning scenarios with 25% / 50% contingency: **212.2 / 254.6 GPU-hours**.

| Split depth | Online GPU-hours |
|---|---:|
| 3 | 62.3 |
| 6 | 107.4 |

| Attack family | Online GPU-hours |
|---|---:|
| forward | 12.7 |
| joint | 44.9 |
| backward | 0.0 |
| sipit | 112.1 |

## Assumptions and limitations

- Source summary reports five-seed means per eight-document batch on NVIDIA GB10; average the available none/Gaussian defense means.
- New protocol is unmodified, with the same architecture, sequence length and batch size as source timing measurements.
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
  "protocol_sha256": "6e52150e60799d668316dcfd27232ad752191ddaf005e989188c3ea39a28a9bf",
  "source_summary_path": "results/baseline-strengthening/analysis/summary.csv",
  "source_summary_sha256": "37e4273622b63e21bc4e0d143b37b3e4189ecb9ce1cdbe7a6c55d4bc3af4ca66",
  "script_sha256": "04435ce74284ccbe88859c808e8ad87cbc686c2261720fcb9999c1b324c4b8e5"
}
```
