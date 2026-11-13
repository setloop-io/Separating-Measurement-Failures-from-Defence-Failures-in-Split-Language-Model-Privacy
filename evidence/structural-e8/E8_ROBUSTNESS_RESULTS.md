# E8 Robustness Battery — Committed Results

This note replaces the previously hand-tabulated E8 robustness values with the
committed, provenance-bearing artifacts for Qwen3-0.6B.

Artifacts:

- `paper-data/results-h100-2/v2/privacy/e8_robustness_06b_fp32.json`
- `paper-data/results-h100-2/v2/privacy/e8_obfuscation_06b_fp32.json`

## Oracle K curve (fp32 robustness rerun)

The sweep explicitly includes the hidden dimension `H=1024`; the K=1024 point
is measured rather than inferred. Values below are top-1 recovery means from
the three decoder seeds.

| Depth | K=10 | K=100 | K=1000 | K=1024 | K=5000 |
|---:|---:|---:|---:|---:|---:|
| 1 | 18.1% | 51.0% | 69.9% | 69.9% | 69.9% |
| 4 | 5.7% | 28.7% | 67.2% | 67.3% | 67.3% |
| 8 | 3.2% | 24.9% | 66.2% | 66.2% | 66.2% |

The exact per-seed values, top-5 values, W-recovery error, and all other attack
settings remain in the JSON artifact. The transition is therefore reported as
an observed curve, not as “full recovery at K≈H” inferred from neighboring
points.

## Label-free and session-rotation results

The fp32 robustness artifact includes label-free whitening/nearest-neighbor,
moment alignment, and session-pooling attacks at depths 1, 4, and 8. The
maximum label-free top-1 result is recorded in the artifact's `summary`; depth 8
is no longer omitted.

## Utility and exactness control

The companion fp32 utility artifact reports two prompts at each depth:

- greedy token identity: 100% at depths 1, 4, and 8;
- maximum round-trip deviation: recorded per prompt/depth;
- baseline, obfuscated, and leaked-key attacker controls: recorded for three
  seeds;
- boundary matmul overhead: recorded by the harness.

The exactness claim is now scoped to the tested prompts, dtype, model, and
artifact provenance rather than generalized from the earlier bf16 smoke run.

## Reporting status

Issue #24's artifact, K=1024, depth-8 label-free, fp32 utility, and provenance
requirements are satisfied. Broader claims still require the protocol's
separate repeat-session and victim-set expansions; those are not prerequisites
for the narrowly scoped E8 robustness issue closure.
