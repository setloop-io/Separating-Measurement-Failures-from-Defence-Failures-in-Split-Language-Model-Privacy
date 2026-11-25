# P1 pilot analysis (exploratory cohort)

> Exploratory-cohort analysis of the P1 pilot; hypothesis-generating only, not validated evidence for any manuscript claim.

**Verdict: GO**

final eval step: 2000 (persistence steps: 1800, 1900, 2000)

## Go-criteria checklist

| # | Criterion | Result | Detail |
|---|---|---|---|
| 1 | frozen go rule: G_R > 0 at final step for all three seeds in >= 2 of 6 cells | PASS | qualified cells (2): long-high, medium-high |
| 2 | magnitude supports a confirmatory power calculation (descriptive) | PASS | cells with feasible naive n: 3 |
| 3 | gain persists at the final three eval points | PASS | short-low: no; short-high: no; medium-low: no; medium-high: persists; long-low: no; long-high: persists |
| 4 | remote resource accounting nonzero in trained cells | PASS | 18/18 trained runs with nonzero remote FLOPs and bytes |
| 5 | >= 1 cell privacy-eligible per privacy_scores.json | PASS | eligible cells: long-low, medium-low, short-low; instrument qualified: True |

## Causal remote-learning gain G_R = L(frozen) - L(trained), final step

| Cell | s101 | s202 | s303 | mean | SD | naive n (80%) | qualifies | persists |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| short-low | 0.0012 | -0.0025 | -0.0013 | -0.0009 | 0.0019 | - | no | no |
| short-high | 0.0110 | 0.0213 | -0.0049 | 0.0091 | 0.0132 | 17 | no | no |
| medium-low | -0.0027 | 0.0003 | 0.0011 | -0.0004 | 0.0020 | - | no | no |
| medium-high | 0.0162 | 0.0143 | 0.0197 | 0.0168 | 0.0028 | 2 | yes | yes |
| long-low | -0.0139 | -0.0168 | -0.0081 | -0.0129 | 0.0044 | - | no | no |
| long-high | 0.0061 | 0.0156 | 0.0326 | 0.0181 | 0.0134 | 5 | yes | yes |

## Replacement penalty Delta_U = L(trained) - L(all_local), final step

| Cell | s101 | s202 | s303 |
|---|---:|---:|---:|
| short-low | 0.0064 | 0.0159 | 0.0229 |
| short-high | 0.0285 | 0.0240 | 0.0312 |
| medium-low | 0.0432 | 0.0638 | 0.0587 |
| medium-high | 0.0633 | 0.0653 | 0.0622 |
| long-low | 0.1381 | 0.1502 | 0.1597 |
| long-high | 0.1390 | 0.1351 | 0.1396 |

## Removed-remote reference: L(removed) - L(all_local), final step

| Span | s101 | s202 | s303 |
|---|---:|---:|---:|
| short | 0.0114 | 0.0062 | 0.0018 |
| medium | 0.0326 | 0.0535 | 0.0419 |
| long | 0.1300 | 0.1325 | 0.1379 |

## Privacy: token-inversion advantage over matched null

source: /Users/geo/Paper 3 - Privacy-Constrained Remote/outputs/pilot/privacy_scores.json; instrument qualified: True

| Run | Cell | Advantage | Null acc | Bits/token | Eligible |
|---|---|---:|---:|---:|---|
| frozen_random-long-low-s101 | long-low | -0.1348 | 0.1377 | 2048.1 | True |
| frozen_random-long-low-s202 | long-low | -0.1143 | 0.1377 | 2048.1 | True |
| frozen_random-long-low-s303 | long-low | -0.0967 | 0.1377 | 2048.1 | True |
| frozen_random-medium-low-s101 | medium-low | -0.1162 | 0.1377 | 2048.1 | True |
| frozen_random-medium-low-s202 | medium-low | -0.1279 | 0.1377 | 2048.1 | True |
| frozen_random-medium-low-s303 | medium-low | -0.0850 | 0.1377 | 2048.1 | True |
| frozen_random-short-low-s101 | short-low | -0.0811 | 0.1377 | 2048.1 | True |
| frozen_random-short-low-s202 | short-low | -0.1250 | 0.1377 | 2048.1 | True |
| frozen_random-short-low-s303 | short-low | -0.1045 | 0.1377 | 2048.1 | True |
| removed_remote-long-s101 | long-low | -0.1367 | 0.1377 | 2048.1 | True |
| removed_remote-long-s202 | long-low | -0.1084 | 0.1377 | 2048.1 | True |
| removed_remote-long-s303 | long-low | -0.1143 | 0.1377 | 2048.1 | True |
| removed_remote-medium-s101 | medium-low | -0.0850 | 0.1377 | 2048.1 | True |
| removed_remote-medium-s202 | medium-low | -0.0645 | 0.1377 | 2048.1 | True |
| removed_remote-medium-s303 | medium-low | -0.0898 | 0.1377 | 2048.1 | True |
| removed_remote-short-s101 | short-low | -0.1250 | 0.1377 | 2048.1 | True |
| removed_remote-short-s202 | short-low | -0.1230 | 0.1377 | 2048.1 | True |
| removed_remote-short-s303 | short-low | -0.1162 | 0.1377 | 2048.1 | True |
| trained-long-low-s101 | long-low | -0.1289 | 0.1377 | 2048.1 | True |
| trained-long-low-s202 | long-low | -0.1221 | 0.1377 | 2048.1 | True |
| trained-long-low-s303 | long-low | -0.0674 | 0.1377 | 2048.1 | True |
| trained-medium-low-s101 | medium-low | -0.1133 | 0.1377 | 2048.1 | True |
| trained-medium-low-s202 | medium-low | -0.1094 | 0.1377 | 2048.1 | True |
| trained-medium-low-s303 | medium-low | -0.1016 | 0.1377 | 2048.1 | True |
| trained-short-low-s101 | short-low | -0.1035 | 0.1377 | 2048.1 | True |
| trained-short-low-s202 | short-low | -0.1230 | 0.1377 | 2048.1 | True |
| trained-short-low-s303 | short-low | -0.0986 | 0.1377 | 2048.1 | True |

## Remote resource accounting (trained cells)

| Run | fwd FLOPs | fwd+bwd FLOPs | bytes | bits/token | wall (s) | nonzero |
|---|---:|---:|---:|---:|---:|---|
| trained-long-high-s101 | 13958643712000 | 41875931136000 | 8388672000 | 8192.1 | 97.4 | True |
| trained-long-high-s202 | 13958643712000 | 41875931136000 | 8388672000 | 8192.1 | 96.9 | True |
| trained-long-high-s303 | 13958643712000 | 41875931136000 | 8388672000 | 8192.1 | 98.5 | True |
| trained-long-low-s101 | 1073741824000 | 3221225472000 | 2097216000 | 2048.1 | 73.9 | True |
| trained-long-low-s202 | 1073741824000 | 3221225472000 | 2097216000 | 2048.1 | 73.7 | True |
| trained-long-low-s303 | 1073741824000 | 3221225472000 | 2097216000 | 2048.1 | 73.9 | True |
| trained-medium-high-s101 | 13958643712000 | 41875931136000 | 8388672000 | 8192.1 | 151.5 | True |
| trained-medium-high-s202 | 13958643712000 | 41875931136000 | 8388672000 | 8192.1 | 154.7 | True |
| trained-medium-high-s303 | 13958643712000 | 41875931136000 | 8388672000 | 8192.1 | 155.3 | True |
| trained-medium-low-s101 | 1073741824000 | 3221225472000 | 2097216000 | 2048.1 | 126.7 | True |
| trained-medium-low-s202 | 1073741824000 | 3221225472000 | 2097216000 | 2048.1 | 129.0 | True |
| trained-medium-low-s303 | 1073741824000 | 3221225472000 | 2097216000 | 2048.1 | 131.0 | True |
| trained-short-high-s101 | 13958643712000 | 41875931136000 | 8388672000 | 8192.1 | 203.8 | True |
| trained-short-high-s202 | 13958643712000 | 41875931136000 | 8388672000 | 8192.1 | 211.8 | True |
| trained-short-high-s303 | 13958643712000 | 41875931136000 | 8388672000 | 8192.1 | 213.1 | True |
| trained-short-low-s101 | 1073741824000 | 3221225472000 | 2097216000 | 2048.1 | 179.7 | True |
| trained-short-low-s202 | 1073741824000 | 3221225472000 | 2097216000 | 2048.1 | 185.6 | True |
| trained-short-low-s303 | 1073741824000 | 3221225472000 | 2097216000 | 2048.1 | 187.5 | True |

## Pairing sanity: trained vs frozen at the first two eval points (report only)

| Cell | Seed | Step | trained | frozen | frozen - trained |
|---|---|---:|---:|---:|---:|
| short-low | 101 | 0 | 5.5924 | 5.5924 | 0.000000 |
| short-low | 101 | 100 | 2.5998 | 2.5998 | 0.000027 |
| short-low | 202 | 0 | 5.6041 | 5.6041 | 0.000000 |
| short-low | 202 | 100 | 2.6026 | 2.6024 | -0.000251 |
| short-low | 303 | 0 | 5.5554 | 5.5554 | 0.000000 |
| short-low | 303 | 100 | 2.5959 | 2.5959 | 0.000068 |
| short-high | 101 | 0 | 5.5960 | 5.5960 | 0.000000 |
| short-high | 101 | 100 | 2.6010 | 2.6019 | 0.000977 |
| short-high | 202 | 0 | 5.5464 | 5.5464 | 0.000000 |
| short-high | 202 | 100 | 2.5998 | 2.5985 | -0.001207 |
| short-high | 303 | 0 | 5.5769 | 5.5769 | 0.000000 |
| short-high | 303 | 100 | 2.6096 | 2.6057 | -0.003858 |
| medium-low | 101 | 0 | 5.5928 | 5.5928 | 0.000000 |
| medium-low | 101 | 100 | 2.6029 | 2.6028 | -0.000190 |
| medium-low | 202 | 0 | 5.6389 | 5.6389 | 0.000000 |
| medium-low | 202 | 100 | 2.5957 | 2.5957 | -0.000023 |
| medium-low | 303 | 0 | 5.5846 | 5.5846 | 0.000000 |
| medium-low | 303 | 100 | 2.6006 | 2.6005 | -0.000013 |
| medium-high | 101 | 0 | 5.5810 | 5.5810 | 0.000000 |
| medium-high | 101 | 100 | 2.6011 | 2.6004 | -0.000671 |
| medium-high | 202 | 0 | 5.5910 | 5.5910 | 0.000000 |
| medium-high | 202 | 100 | 2.5970 | 2.5994 | 0.002365 |
| medium-high | 303 | 0 | 5.5653 | 5.5653 | 0.000000 |
| medium-high | 303 | 100 | 2.6036 | 2.6030 | -0.000607 |
| long-low | 101 | 0 | 5.6034 | 5.6034 | 0.000000 |
| long-low | 101 | 100 | 2.6013 | 2.6015 | 0.000228 |
| long-low | 202 | 0 | 5.6786 | 5.6786 | 0.000000 |
| long-low | 202 | 100 | 2.6119 | 2.6117 | -0.000142 |
| long-low | 303 | 0 | 5.5715 | 5.5715 | 0.000000 |
| long-low | 303 | 100 | 2.6168 | 2.6172 | 0.000343 |
| long-high | 101 | 0 | 5.6004 | 5.6004 | 0.000000 |
| long-high | 101 | 100 | 2.6010 | 2.6037 | 0.002725 |
| long-high | 202 | 0 | 5.6227 | 5.6227 | 0.000000 |
| long-high | 202 | 100 | 2.6111 | 2.6136 | 0.002491 |
| long-high | 303 | 0 | 5.5813 | 5.5813 | 0.000000 |
| long-high | 303 | 100 | 2.6092 | 2.6188 | 0.009543 |

## Operational failures

- none

## Non-pilot manifests ignored

- `smoke-trained-s101`
