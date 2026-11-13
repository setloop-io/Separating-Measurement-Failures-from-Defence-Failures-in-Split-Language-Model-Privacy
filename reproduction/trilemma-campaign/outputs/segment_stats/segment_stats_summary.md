# P2c segment-statistics analysis (protocol p2c-segment-statistics-1.0)

> Confirmatory analysis under protocol p2c-segment-statistics-1.0 (frozen 2026-09-04, fresh seeds [3..83]); per docs/PREREGISTRATION_P2C.md. Statistics derive from ONE calibration backbone (seed 929).

## Verdict

- **H2 (segment statistic beats nominal span out of sample): NOT CONFIRMED**
  - p(one-sided paired t on |err_S| - |err_N|) = 0.99917 (alpha 0.05); mean |err| S 0.0971 vs N 0.0210; R^2_oos S -7.9780, N 0.4550
  - failed conditions: ['p_value', 'mean_absolute_error', 'r2_S_positive']

## Predictor fits (OLS on the six exploratory selection points only)

- model S: Delta_U = 0.2847 + -0.0095 * intrinsic_dimension
- model N: Delta_U = -0.0425 + 0.0328 * span

## Per-cell table (held-out cells)

| Cell | Segment | stat value | measured Delta_U [95% CI] | pred S | pred N | |err S| | |err N| |
|---|---|---:|---|---:|---:|---:|---:|
| s2at1 | 2@1 | 9.95460 | 0.0390 [0.0296, 0.0483] | 0.1898 | 0.0230 | 0.1508 | 0.0160 |
| s2at5 | 2@5 | 12.64074 | 0.0346 [0.0259, 0.0433] | 0.1642 | 0.0230 | 0.1296 | 0.0116 |
| s3at2 | 3@2 | 26.00592 | 0.0527 [0.0400, 0.0653] | 0.0368 | 0.0558 | 0.0159 | 0.0031 |
| s3at4 | 3@4 | 12.64074 | 0.0479 [0.0362, 0.0596] | 0.1642 | 0.0558 | 0.1163 | 0.0079 |
| s4at3 | 4@3 | 12.64074 | 0.0639 [0.0573, 0.0705] | 0.1642 | 0.0885 | 0.1003 | 0.0246 |
| s5at1 | 5@1 | 23.07830 | 0.1080 [0.0990, 0.1170] | 0.0647 | 0.1213 | 0.0433 | 0.0133 |
| s5at3 | 5@3 | 8.68676 | 0.0725 [0.0599, 0.0850] | 0.2019 | 0.1213 | 0.1294 | 0.0488 |
| s6at2 | 6@2 | 8.68676 | 0.1110 [0.1028, 0.1192] | 0.2019 | 0.1541 | 0.0909 | 0.0430 |

## Secondary (H4, descriptive, low-power)

- R^2_oos Delta_U ~ span: 0.4550
- R^2_oos Delta_U ~ span + position + statistic: -11.0894 (4 parameters fit on 6 exploratory points; descriptive only, not gated)

## Operational failures

- `all_local-s33`: manifest_field_mismatch:protocol_version='p2-confirmatory-1.0'
