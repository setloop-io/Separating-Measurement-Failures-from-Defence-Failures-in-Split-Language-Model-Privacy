# Instrument qualification dress rehearsal (calibration cohort): qualification-20260922-a

Completed 32/32 runs; failures: 0.
Execution revision: `bc1f1d22a5a6d65fee887f6607b323d787f82a3d`. Manifests, source archives, artifact bindings,
capture hashes, stored control scores, and candidate-score recomputation verified.

Cohort `calibration`, scientific_evidence true, scoped to instrument qualification only.
No defense-success, utility, or causal-gain claim; privacy remains **undetermined**.
Candidate advantages on calibration models are exploratory observations, not pilot gate outcomes.

| Cell | Target | Seeds | Verdict |
|---|---|---|---|
| d0-q8 | property | 4/4 | unqualified: unqualified bindings on seeds: [419104] |
| d0-q8 | structure | 4/4 | qualified |
| d1-s0.1-q8 | property | 4/4 | qualified |
| d1-s0.1-q8 | structure | 4/4 | qualified |
| d1-s0.5-q8 | property | 4/4 | qualified |
| d1-s0.5-q8 | structure | 4/4 | qualified |
| d1-s1.0-q8 | property | 4/4 | unqualified: unqualified bindings on seeds: [419102] |
| d1-s1.0-q8 | structure | 4/4 | qualified |
| d3-q1 | property | 4/4 | qualified |
| d3-q1 | structure | 4/4 | qualified |
| d3-q2 | property | 4/4 | qualified |
| d3-q2 | structure | 4/4 | qualified |
| d3-q3 | property | 4/4 | qualified |
| d3-q3 | structure | 4/4 | qualified |

A binding is qualified iff all 16 no-signal trials stay within the ±0.02 band and both
nonzero positive doses clear the target floor (property 0.10, structure 0.02). A cell-target
verdict is qualified iff all four calibration seeds' bindings qualify; unqualified coverage
leaves the corresponding pilot coverage undetermined.

## Candidate advantages per seed (descriptive)

| Cell | Target | Seed | Candidate advantage |
|---|---|---|---|
| d0-q8 | property | 419101 | 0.8540 |
| d0-q8 | property | 419102 | 0.8978 |
| d0-q8 | property | 419103 | 0.8685 |
| d0-q8 | property | 419104 | 0.8859 |
| d0-q8 | structure | 419101 | 0.1043 |
| d0-q8 | structure | 419102 | 0.0987 |
| d0-q8 | structure | 419103 | 0.0942 |
| d0-q8 | structure | 419104 | 0.0959 |
| d1-s0.1-q8 | property | 419101 | 0.8026 |
| d1-s0.1-q8 | property | 419102 | 0.8271 |
| d1-s0.1-q8 | property | 419103 | 0.8062 |
| d1-s0.1-q8 | property | 419104 | 0.7852 |
| d1-s0.1-q8 | structure | 419101 | 0.1041 |
| d1-s0.1-q8 | structure | 419102 | 0.1016 |
| d1-s0.1-q8 | structure | 419103 | 0.0976 |
| d1-s0.1-q8 | structure | 419104 | 0.0944 |
| d1-s0.5-q8 | property | 419101 | 0.8164 |
| d1-s0.5-q8 | property | 419102 | 0.8147 |
| d1-s0.5-q8 | property | 419103 | 0.7907 |
| d1-s0.5-q8 | property | 419104 | 0.7793 |
| d1-s0.5-q8 | structure | 419101 | 0.1054 |
| d1-s0.5-q8 | structure | 419102 | 0.0983 |
| d1-s0.5-q8 | structure | 419103 | 0.0937 |
| d1-s0.5-q8 | structure | 419104 | 0.0947 |
| d1-s1.0-q8 | property | 419101 | 0.8143 |
| d1-s1.0-q8 | property | 419102 | 0.8167 |
| d1-s1.0-q8 | property | 419103 | 0.7815 |
| d1-s1.0-q8 | property | 419104 | 0.7632 |
| d1-s1.0-q8 | structure | 419101 | 0.1054 |
| d1-s1.0-q8 | structure | 419102 | 0.1044 |
| d1-s1.0-q8 | structure | 419103 | 0.0952 |
| d1-s1.0-q8 | structure | 419104 | 0.0949 |
| d3-q1 | property | 419101 | 0.8106 |
| d3-q1 | property | 419102 | 0.8506 |
| d3-q1 | property | 419103 | 0.8034 |
| d3-q1 | property | 419104 | 0.8292 |
| d3-q1 | structure | 419101 | 0.1204 |
| d3-q1 | structure | 419102 | 0.1081 |
| d3-q1 | structure | 419103 | 0.1135 |
| d3-q1 | structure | 419104 | 0.1117 |
| d3-q2 | property | 419101 | 0.8251 |
| d3-q2 | property | 419102 | 0.8032 |
| d3-q2 | property | 419103 | 0.8174 |
| d3-q2 | property | 419104 | 0.8074 |
| d3-q2 | structure | 419101 | 0.1201 |
| d3-q2 | structure | 419102 | 0.1113 |
| d3-q2 | structure | 419103 | 0.1139 |
| d3-q2 | structure | 419104 | 0.1144 |
| d3-q3 | property | 419101 | 0.8385 |
| d3-q3 | property | 419102 | 0.8633 |
| d3-q3 | property | 419103 | 0.8361 |
| d3-q3 | property | 419104 | 0.8557 |
| d3-q3 | structure | 419101 | 0.1166 |
| d3-q3 | structure | 419102 | 0.1131 |
| d3-q3 | structure | 419103 | 0.1093 |
| d3-q3 | structure | 419104 | 0.1106 |

Instrument qualification records under the exact candidate pilot regime, calibration cohort only; no defense-success, utility, or causal-gain claim; privacy undetermined. Candidate scores on calibration models are calibration-cohort exploratory observations, not pilot gate outcomes.

Unevaluated: returned values; quantizer metadata; full-precision inbound/outbound gradients; evolving remote state; optimizer state; timing; active/adaptive attackers; within-run joint tensors; cross-run attacks; history accumulation; canary membership; natural-sequence membership; semantics beyond five-class byte structure; token inversion.

See [verification](verification.json), [summary](summary.json), and [source record](source.json).
