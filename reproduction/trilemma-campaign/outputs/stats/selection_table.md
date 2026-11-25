# P2c stage-1 statistic selection table

Frozen rule: the primary statistic is the candidate with the highest absolute Pearson correlation with mean Delta_U across the six exploratory cells {2@3, 4@2, 6@1} x {64, 256}; ties broken by the higher correlation within the width-256 subset

**Selected statistic: intrinsic_dimension**

## Selection cells (Delta_U means: P2 primary, P1 sanity)

| Cell | Segment | Width | Delta_U P2 mean | Delta_U P1 mean | residual_update_magnitude | block_influence | representation_similarity | layer_drop_sensitivity | intrinsic_dimension |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| short-low | 2@3 | 64 | 0.025719 | 0.015057 | 0.522799 | 0.14598 | 0.823944 | 0.835767 | 26.0059 |
| short-high | 2@3 | 256 | 0.037767 | 0.027901 | 0.522799 | 0.14598 | 0.823944 | 0.835767 | 26.0059 |
| medium-low | 4@2 | 64 | 0.069747 | 0.055241 | 0.513524 | 0.136455 | 0.381881 | 1.17198 | 23.0783 |
| medium-high | 4@2 | 256 | 0.072421 | 0.063587 | 0.513524 | 0.136455 | 0.381881 | 1.17198 | 23.0783 |
| long-low | 6@1 | 64 | 0.170908 | 0.149348 | 0.520681 | 0.131181 | 0.182261 | 1.58407 | 12.6407 |
| long-high | 6@1 | 256 | 0.154668 | 0.137881 | 0.520681 | 0.131181 | 0.182261 | 1.58407 | 12.6407 |

## Correlations with mean Delta_U (P2 means)

| Statistic | r (6 cells) | |r| | r (width 256) | r (width 64) | r vs P1 means (sanity) |
|---|---:|---:|---:|---:|---:|
| residual_update_magnitude | 0.0071 | 0.0071 | 0.0113 | 0.0038 | -0.0042 |
| block_influence | -0.9192 | 0.9192 | -0.9229 | -0.9257 | -0.9234 |
| representation_similarity | -0.8990 | 0.8990 | -0.9023 | -0.9055 | -0.9037 |
| layer_drop_sensitivity | 0.9803 | 0.9803 | 0.9852 | 0.9864 | 0.9820 |
| intrinsic_dimension | -0.9905 | 0.9905 | -0.9966 | -0.9959 | -0.9894 |
