# Next-byte structure precision at the pilot operating point (engineering report): structure-precision-20260922-a

Completed 6/6 scenarios; failures: 0.
Execution revision: `675ac6ae6ec9c9caa46a241de1f53221efaf5e8c`. Source archives, artifact bindings, replay checksums, and stored estimates verified.

Two verified precision-128 control-model preparations; three release conditions per control model.
The scorer is the five-class MLP on per-token released features; labels are the panels' own next bytes.
No normal candidate was scored or defense selected. All instruments remain unqualified and
privacy **undetermined**.

## Release condition `d0-q8`

| Sequences | Matched nulls K | Within ±0.02 | Above +0.02 | Mean advantage | SD of advantage | Positive floors passed |
|---|---|---|---|---|---|---|
| 8192 | 1 | 32/32 (1.00) | 0/32 | 0.0000 | 0.0000 | 4/4 |
| 8192 | 4 | 32/32 (1.00) | 0/32 | 0.0000 | 0.0000 | 4/4 |
| 16384 | 1 | 32/32 (1.00) | 0/32 | 0.0000 | 0.0000 | 4/4 |
| 16384 | 4 | 32/32 (1.00) | 0/32 | 0.0000 | 0.0000 | 4/4 |

| Sequences | K | SD of fitted accuracy | SD of mean null accuracy | 95th percentile absolute advantage |
|---|---|---|---|---|
| 8192 | 1 | 0.0001 | 0.0001 | 0.0000 |
| 8192 | 4 | 0.0001 | 0.0001 | 0.0000 |
| 16384 | 1 | 0.0002 | 0.0002 | 0.0000 |
| 16384 | 4 | 0.0002 | 0.0002 | 0.0000 |

## Release condition `d1-warm-s1.0-q8`

| Sequences | Matched nulls K | Within ±0.02 | Above +0.02 | Mean advantage | SD of advantage | Positive floors passed |
|---|---|---|---|---|---|---|
| 8192 | 1 | 32/32 (1.00) | 0/32 | 0.0000 | 0.0000 | 4/4 |
| 8192 | 4 | 32/32 (1.00) | 0/32 | 0.0000 | 0.0000 | 4/4 |
| 16384 | 1 | 32/32 (1.00) | 0/32 | 0.0000 | 0.0000 | 4/4 |
| 16384 | 4 | 32/32 (1.00) | 0/32 | 0.0000 | 0.0000 | 4/4 |

| Sequences | K | SD of fitted accuracy | SD of mean null accuracy | 95th percentile absolute advantage |
|---|---|---|---|---|
| 8192 | 1 | 0.0001 | 0.0001 | 0.0000 |
| 8192 | 4 | 0.0001 | 0.0001 | 0.0000 |
| 16384 | 1 | 0.0002 | 0.0002 | 0.0000 |
| 16384 | 4 | 0.0002 | 0.0002 | 0.0000 |

## Release condition `d3-q1`

| Sequences | Matched nulls K | Within ±0.02 | Above +0.02 | Mean advantage | SD of advantage | Positive floors passed |
|---|---|---|---|---|---|---|
| 8192 | 1 | 32/32 (1.00) | 0/32 | -0.0000 | 0.0000 | 4/4 |
| 8192 | 4 | 32/32 (1.00) | 0/32 | 0.0000 | 0.0000 | 4/4 |
| 16384 | 1 | 32/32 (1.00) | 0/32 | -0.0000 | 0.0000 | 4/4 |
| 16384 | 4 | 32/32 (1.00) | 0/32 | -0.0000 | 0.0000 | 4/4 |

| Sequences | K | SD of fitted accuracy | SD of mean null accuracy | 95th percentile absolute advantage |
|---|---|---|---|---|
| 8192 | 1 | 0.0001 | 0.0001 | 0.0000 |
| 8192 | 4 | 0.0001 | 0.0001 | 0.0000 |
| 16384 | 1 | 0.0002 | 0.0002 | 0.0000 |
| 16384 | 4 | 0.0002 | 0.0002 | 0.0000 |

Counts pool dependent conditional donor trials across both setups. They are not population false-positive
rates or independent replications. Positive checks exclude dose zero and retain both nonzero doses;
the floor is the declared 0.02 structure positive floor, not an acceptance gate.
No-signal trials derange the label panel at the sequence level; matched nulls permute fit labels at the
token-row level. Neither reuses an observed association, and neither certifies detection power.
All retained quantizer-support checks passed; maximum normalized rounding error: 3.05176e-05.
Zero-dose controls match ordinary donor releases exactly. Class-code signals were injected after defense
and before final quantization. Valid codebook support is not a claim of naturalistic control
distributions or defense efficacy.

Unevaluated: next-byte classes; returned values; quantizer metadata; full-precision inbound/outbound gradients; evolving remote state; optimizer state; timing; active/adaptive attackers; within-run joint tensors; cross-run attacks; history accumulation; canary membership; natural-sequence membership; unrestricted semantics.

See [verification](verification.json), [summary](summary.json), and [source record](source.json).
