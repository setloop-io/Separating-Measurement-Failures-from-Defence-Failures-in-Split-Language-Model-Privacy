> **Verification scope warning (2026-09-10):** This new analysis was run with `--no-privacy-scoring`. Its privacy FAIL flags and triangle verdict are missing-score placeholders, not independently recomputed privacy results. Only utility, gain, and analytical-share fields are validated; see `comparison.json` and `README.md`. Historical template wording below does not establish independently verifiable preregistration.

# P5 delegation-sweep analysis (protocol p5-delegation-sweep-1.0)

> P5 delegation-sweep analysis under protocol p5-delegation-sweep-1.0 (frozen 2026-09-07, fresh seeds [21..91]); per docs/PREREGISTRATION_P5.md. Empirical feasible-set statement within this mechanism family, not a universal law.

## Triangle verdict

**within the swept space, defense options, and attack suite, no tested configuration holds all three gates**

Cells holding all three gates: none

## Per-cell gate table

| Cell | privacy | utility | share >= 0.15 | G_R mean | Delta_U [95% CI] | share | holds all |
|---|---|---|---|---:|---|---:|---|
| s1at3-d0 | FAIL | pass | FAIL | 0.0044 | 0.0251 [0.0203, 0.0298] | 0.123 | no |
| s1at3-d1 | FAIL | FAIL | FAIL | 0.0196 | 1.3289 [1.1678, 1.4900] | 0.123 | no |
| s2at3-d0 | FAIL | pass | FAIL | 0.0077 | 0.0351 [0.0281, 0.0421] | 0.140 | no |
| s2at3-d1 | FAIL | FAIL | FAIL | -0.0384 | 1.5595 [1.4184, 1.7005] | 0.140 | no |
| s3at2-d0 | FAIL | pass | pass | 0.0276 | 0.0338 [0.0283, 0.0392] | 0.280 | no |
| s3at2-d1 | FAIL | FAIL | pass | -0.1415 | 1.3172 [1.1735, 1.4609] | 0.280 | no |
| s4at2-d0 | FAIL | pass | pass | 0.0436 | 0.0476 [0.0391, 0.0562] | 0.325 | no |
| s4at2-d1 | FAIL | FAIL | pass | -0.0924 | 1.4496 [1.3322, 1.5671] | 0.325 | no |
| s5at1-d0 | FAIL | pass | pass | 0.0843 | 0.0614 [0.0537, 0.0691] | 0.487 | no |
| s5at1-d1 | FAIL | FAIL | pass | 0.0492 | 0.9831 [0.9519, 1.0143] | 0.487 | no |
| s6at1-d0 | FAIL | pass | pass | 0.0796 | 0.0767 [0.0716, 0.0817] | 0.582 | no |
| s6at1-d1 | FAIL | FAIL | pass | 0.1052 | 1.2042 [1.1475, 1.2609] | 0.582 | no |
| s7at0-d0 | FAIL | FAIL | pass | 0.2064 | 0.1339 [0.1274, 0.1404] | 0.776 | no |
| s7at0-d1 | FAIL | FAIL | pass | -0.1694 | 0.8191 [0.7996, 0.8386] | 0.776 | no |

## Privacy qualification (gradient channel, fresh captures)

- sequence_property_v2_x_inbound_gradients @ d0: DISQUALIFIED
- structure_semantic_v2_x_inbound_gradients @ d0: QUALIFIED
- token_inversion_x_inbound_gradients @ d0: DISQUALIFIED
- sequence_property_v2_x_inbound_gradients @ d1: DISQUALIFIED
- structure_semantic_v2_x_inbound_gradients @ d1: DISQUALIFIED
- token_inversion_x_inbound_gradients @ d1: DISQUALIFIED

## Named omissions

- history-accumulation and canary-membership families (instruments unqualified, P4b)
- timing channels
- optimizer state beyond the remote module
- active/adaptive attackers
- natural-sequence membership
- cross-run joint attacks
- arch2/corpus2 (single-setting sweep by design; replication deferred)

## Operational failures

- none
