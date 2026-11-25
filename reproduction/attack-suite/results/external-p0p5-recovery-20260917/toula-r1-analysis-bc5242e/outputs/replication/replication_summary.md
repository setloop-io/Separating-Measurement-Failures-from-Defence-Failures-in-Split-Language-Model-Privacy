# P3 external-replication analysis (protocol p3-replication-1.0)

> Confirmatory external-replication analysis under protocol p3-replication-1.0 (frozen 2026-09-06, fresh seeds [9..89]); per docs/PREREGISTRATION_P3.md. No pooling across settings; privacy not re-evaluated (token-inversion-family scope omission still applies).

## Verdicts

- **primary A2C2: REPLICATED**
- secondary A2C1: **REPLICATED**
- secondary A1C2: **REPLICATED**
- cross-setting pattern: all settings positive mean G_R: **True** (descriptive; architectures/corpora are not exchangeable, no pooling)

## Per-setting tests (one-sided paired t on G_R, step 2000)

| Setting | role | n | dropped | mean G_R | SD | t | df | p | alpha | >=0.01 | persists | verdict |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|---|
| a2c1 | secondary | 8 | none | 0.0463 | 0.0083 | 15.7718 | 7 | 0.00000 | 0.025 | yes | yes | **REPLICATED** |
| a1c2 | secondary | 8 | none | 0.0118 | 0.0057 | 5.9180 | 7 | 0.00029 | 0.025 | yes | yes | **REPLICATED** |
| a2c2 | primary | 8 | none | 0.0322 | 0.0068 | 13.4181 | 7 | 0.00000 | 0.05 | yes | yes | **REPLICATED** |

## Descriptive: replacement penalty and removed-remote reference

| Setting | Delta_U mean [95% CI] | removed ref mean [95% CI] |
|---|---|---|
| a2c1 | 0.0372 [0.0304, 0.0439] | 0.0533 [0.0457, 0.0608] |
| a1c2 | 0.0691 [0.0629, 0.0752] | 0.0589 [0.0552, 0.0626] |
| a2c2 | 0.0447 [0.0398, 0.0496] | 0.0511 [0.0461, 0.0561] |

## Downstream probes (descriptive, not gated)

| Property | gap (trained - all_local) mean [95% CI] |
|---|---|
| is_digit | 0.0000 [-0.0001, 0.0001] |
| is_uppercase_letter | 0.0005 [-0.0006, 0.0016] |
| is_vowel | 0.0058 [0.0038, 0.0077] |
| is_whitespace | -0.0001 [-0.0005, 0.0004] |

## Operational failures

- none
