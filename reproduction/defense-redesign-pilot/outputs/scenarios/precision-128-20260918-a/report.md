# Matched-null precision at the pilot operating point (engineering report): precision-128-20260918-a

Completed 8/8 scenarios; failures: 0.
Execution revision: `f14a014877ea93ac223ef54757311aff4b8c5015`. Source archives, artifact bindings, replay checksums, and stored estimates verified.

Two verified warmup-budget shared setups; each d0-trained branch was replayed inside this campaign
and the replayed remote/trusted checksums matched the bound branch records bit for bit.
Three release conditions per control model. No normal candidate was scored or defense selected.
All instruments remain unqualified and privacy **undetermined**.

## Release condition `d0-q8`

| Sequences | Matched nulls K | Within ±0.02 | Above +0.02 | Mean advantage | SD of advantage | Positive floors passed |
|---|---|---|---|---|---|---|
| 8192 | 16 | 58/64 (0.91) | 3/64 | -0.0009 | 0.0119 | 4/4 |
| 8192 | 64 | 60/64 (0.94) | 2/64 | -0.0009 | 0.0112 | 4/4 |
| 16384 | 16 | 63/64 (0.98) | 0/64 | -0.0001 | 0.0073 | 4/4 |
| 16384 | 64 | 63/64 (0.98) | 0/64 | -0.0001 | 0.0072 | 4/4 |

| Sequences | K | SD of fitted-probe r | SD of mean matched-null r | 95th percentile absolute advantage |
|---|---|---|---|---|
| 8192 | 16 | 0.0111 | 0.0029 | 0.0214 |
| 8192 | 64 | 0.0111 | 0.0014 | 0.0202 |
| 16384 | 16 | 0.0071 | 0.0018 | 0.0154 |
| 16384 | 64 | 0.0071 | 0.0011 | 0.0134 |

## Release condition `d1-warm-s1.0-q8`

| Sequences | Matched nulls K | Within ±0.02 | Above +0.02 | Mean advantage | SD of advantage | Positive floors passed |
|---|---|---|---|---|---|---|
| 8192 | 16 | 58/64 (0.91) | 3/64 | -0.0009 | 0.0120 | 4/4 |
| 8192 | 64 | 61/64 (0.95) | 1/64 | -0.0008 | 0.0114 | 4/4 |
| 16384 | 16 | 63/64 (0.98) | 0/64 | -0.0001 | 0.0076 | 4/4 |
| 16384 | 64 | 63/64 (0.98) | 0/64 | -0.0002 | 0.0074 | 4/4 |

| Sequences | K | SD of fitted-probe r | SD of mean matched-null r | 95th percentile absolute advantage |
|---|---|---|---|---|
| 8192 | 16 | 0.0112 | 0.0028 | 0.0218 |
| 8192 | 64 | 0.0112 | 0.0014 | 0.0195 |
| 16384 | 16 | 0.0072 | 0.0019 | 0.0167 |
| 16384 | 64 | 0.0072 | 0.0011 | 0.0149 |

## Release condition `d3-q1`

| Sequences | Matched nulls K | Within ±0.02 | Above +0.02 | Mean advantage | SD of advantage | Positive floors passed |
|---|---|---|---|---|---|---|
| 8192 | 16 | 58/64 (0.91) | 3/64 | -0.0002 | 0.0128 | 4/4 |
| 8192 | 64 | 57/64 (0.89) | 4/64 | 0.0003 | 0.0122 | 4/4 |
| 16384 | 16 | 64/64 (1.00) | 0/64 | -0.0002 | 0.0077 | 4/4 |
| 16384 | 64 | 64/64 (1.00) | 0/64 | -0.0001 | 0.0076 | 4/4 |

| Sequences | K | SD of fitted-probe r | SD of mean matched-null r | 95th percentile absolute advantage |
|---|---|---|---|---|
| 8192 | 16 | 0.0121 | 0.0023 | 0.0284 |
| 8192 | 64 | 0.0121 | 0.0012 | 0.0247 |
| 16384 | 16 | 0.0075 | 0.0020 | 0.0142 |
| 16384 | 64 | 0.0075 | 0.0010 | 0.0133 |

Counts pool dependent conditional donor trials across both setups. They are not population false-positive
rates or independent replications. Positive checks exclude dose zero and retain both nonzero doses.
Increasing K averages only the matched-null term; fitted-probe r remains shared across K in each trial.
All retained quantizer-support checks passed; maximum normalized rounding error: 3.05176e-05.
Zero-dose controls match ordinary donor releases exactly. Positive signals were injected after defense
and before final quantization. Metadata were recomputed and hashed, but were not scored as an attack view.
Valid codebook support is not a claim of naturalistic control distributions or defense efficacy.

Unevaluated: next-byte classes; returned values; quantizer metadata; full-precision inbound/outbound gradients; evolving remote state; optimizer state; timing; active/adaptive attackers; within-run joint tensors; cross-run attacks; history accumulation; canary membership; natural-sequence membership; unrestricted semantics.

See [verification](verification.json), [summary](summary.json), and [source record](source.json).
