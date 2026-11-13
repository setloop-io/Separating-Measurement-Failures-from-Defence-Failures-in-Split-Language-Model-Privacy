# P2b rate-sweep analysis (protocol p2b-rate-sweep-1.0)

> Confirmatory rate-sweep analysis under protocol p2b-rate-sweep-1.0 (frozen 2026-09-04, fresh seeds [7..87]); per docs/PREREGISTRATION_P2B.md. Privacy scope: token-inversion family only; membership, property, and history-accumulation attacks remain out of scope.

## Verdicts

- **H3 (rate response): CONFIRMED** (p=0.00000, mean q4->q32 reduction 0.0295 nats, threshold 0.01)
- frontier-feasible cells: ['medium-high-q4', 'medium-high-q8', 'medium-high-q16', 'medium-high-q32']

## Primary: per-seed OLS slope of trained eval loss on q_bits

- slope seeds used: [7, 17, 27, 37, 47, 57, 67, 87] (dropped: none)
- mean slope -0.000711 nats/bit, SD 0.000147, t=-13.715, df=7, p(one-sided, slope<0)=0.00000, 95% CI [-0.000834, -0.000589]
- q4->q32 reduction: mean 0.0295, 95% CI [0.0245, 0.0346]

| Seed | slope | q4->q32 reduction | Spearman |
|---:|---:|---:|---:|
| 7 | -0.000666 | 0.0276 | -0.800 |
| 17 | -0.000827 | 0.0344 | -0.800 |
| 27 | -0.000592 | 0.0247 | -1.000 |
| 37 | -0.000669 | 0.0278 | -1.000 |
| 47 | -0.000597 | 0.0247 | -1.000 |
| 57 | -0.001027 | 0.0426 | -1.000 |
| 67 | -0.000669 | 0.0277 | -1.000 |
| 87 | -0.000641 | 0.0268 | -1.000 |

## Secondary: per-q causal gain G_R (one-sided paired t, alpha 0.0125)

| Cell | n | dropped | mean G_R | SD | t | df | p | 95% CI | persists | significant |
|---|---:|---|---:|---:|---:|---:|---:|---|---|---|
| medium-high-q4 | 8 | none | 0.0264 | 0.0071 | 10.5136 | 7 | 0.00001 | [0.0205, 0.0324] | yes | yes |
| medium-high-q8 | 8 | none | 0.0223 | 0.0043 | 14.5122 | 7 | 0.00000 | [0.0187, 0.0259] | yes | yes |
| medium-high-q16 | 8 | none | 0.0222 | 0.0043 | 14.4376 | 7 | 0.00000 | [0.0186, 0.0258] | yes | yes |
| medium-high-q32 | 8 | none | 0.0222 | 0.0043 | 14.4377 | 7 | 0.00000 | [0.0186, 0.0258] | yes | yes |

## Privacy per q (width-256 calibration, quantized to the cell's q_bits; threshold 0.02)

| Cell | status | per-seed advantage | cell max | pass |
|---|---|---|---:|---|
| medium-high-q4 | scored | s7:-0.0908, s17:-0.1318, s27:-0.1152, s37:-0.1250, s47:-0.1230, s57:-0.1289, s67:-0.1338, s87:-0.1250 | -0.0908 | yes |
| medium-high-q8 | scored | s7:-0.0879, s17:-0.1094, s27:-0.1318, s37:-0.1309, s47:-0.1104, s57:-0.1250, s67:-0.1357, s87:-0.1172 | -0.0879 | yes |
| medium-high-q16 | scored | s7:-0.0869, s17:-0.1094, s27:-0.1348, s37:-0.1309, s47:-0.1064, s57:-0.1240, s67:-0.1348, s87:-0.1162 | -0.0869 | yes |
| medium-high-q32 | scored | s7:-0.0869, s17:-0.1094, s27:-0.1348, s37:-0.1309, s47:-0.1064, s57:-0.1240, s67:-0.1348, s87:-0.1162 | -0.0869 | yes |

## Frontier (feasible = G_R confirmed AND privacy pass)

| q_bits | bits/token | eval loss mean [CI] | G_R mean [CI] | Delta_U mean [CI] | removed ref mean [CI] | privacy max adv | feasible |
|---:|---:|---|---|---|---|---:|---|
| 4 | 4096.1 | 1.7044 [1.6967, 1.7120] | 0.0264 [0.0205, 0.0324] | 0.0951 [0.0853, 0.1049] | 0.0732 [0.0651, 0.0814] | -0.0908 | YES |
| 8 | 8192.1 | 1.6751 [1.6696, 1.6807] | 0.0223 [0.0187, 0.0259] | 0.0659 [0.0586, 0.0732] | 0.0643 [0.0553, 0.0734] | -0.0879 | YES |
| 16 | 16384.1 | 1.6748 [1.6693, 1.6803] | 0.0222 [0.0186, 0.0258] | 0.0656 [0.0583, 0.0728] | 0.0643 [0.0552, 0.0733] | -0.0869 | YES |
| 32 | 32768.0 | 1.6748 [1.6693, 1.6803] | 0.0222 [0.0186, 0.0258] | 0.0656 [0.0583, 0.0728] | 0.0643 [0.0552, 0.0733] | -0.0869 | YES |

## Operational failures

- none
