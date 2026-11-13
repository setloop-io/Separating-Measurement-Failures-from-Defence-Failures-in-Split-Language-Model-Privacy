# Matched-null precision engineering report: precision-20260910-a

Completed 6/6 scenarios; failures: 0.
Execution revision: `e29fd6376d0133c3340b26ad92c9715fe4682094`. Source archives, artifact bindings, and stored estimates verified.

Two saved common warmup checkpoints, three release conditions each; no new model training.
All scoring uses control copies. No normal candidate was scored or defense selected.
The models were warmed under D0 q8 with frozen remote weights; D1/D3 here change only the diagnostic release.
All instruments remain unqualified and privacy **undetermined**.

| Sequences | Matched nulls K | Within ±0.02 | Above +0.02 | 95th percentile absolute advantage | Positive floors passed |
|---|---|---|---|---|---|
| 256 | 1 | 32/192 | 82/192 | 0.1604 | 12/12 |
| 256 | 16 | 56/192 | 65/192 | 0.1397 | 12/12 |
| 256 | 64 | 53/192 | 68/192 | 0.1338 | 12/12 |
| 1024 | 1 | 78/192 | 66/192 | 0.0753 | 12/12 |
| 1024 | 16 | 110/192 | 37/192 | 0.0608 | 12/12 |
| 1024 | 64 | 107/192 | 37/192 | 0.0650 | 12/12 |

Counts pool dependent conditional donor trials. They are not population false-positive rates or independent replications.
Positive checks exclude dose zero and retain both nonzero doses. These floors alone do not qualify an instrument.

| Sequences | K | SD of fitted-probe r | SD of mean matched-null r | SD of advantage |
|---|---|---|---|---|
| 256 | 1 | 0.0665 | 0.0606 | 0.0872 |
| 256 | 16 | 0.0665 | 0.0181 | 0.0711 |
| 256 | 64 | 0.0665 | 0.0084 | 0.0678 |
| 1024 | 1 | 0.0305 | 0.0327 | 0.0389 |
| 1024 | 16 | 0.0305 | 0.0079 | 0.0302 |
| 1024 | 64 | 0.0305 | 0.0040 | 0.0304 |

Increasing K averages only the matched-null term; fitted-probe r remains shared across K in each trial.
All retained quantizer-support checks passed; maximum normalized rounding error: 3.05176e-05.
Zero-dose controls match ordinary donor releases exactly. Positive signals were injected after defense
and before final quantization. Metadata were recomputed and hashed, but were not scored as an attack view.
Valid codebook support is not a claim of naturalistic control distributions or defense efficacy.

Unevaluated: next-byte classes; returned values; quantizer metadata; full-precision inbound/outbound gradients; evolving remote state; optimizer state; timing; active/adaptive attackers; within-run joint tensors; cross-run attacks; history accumulation; canary membership; natural-sequence membership; unrestricted semantics.

See [verification](verification.json), [summary](summary.json), and [source record](source.json).
