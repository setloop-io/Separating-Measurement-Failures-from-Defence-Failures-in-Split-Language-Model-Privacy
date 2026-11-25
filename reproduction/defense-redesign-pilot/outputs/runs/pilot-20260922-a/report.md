# Exploratory pilot gate report (p4-defense-pilot-1.0): pilot-20260922-a

Completed 66/66 runs; failures: 0.
Execution revision: `764d3f2fd60bd04cccbbb099478e6088a9bf7af9`. Manifests, source archives, artifact bindings,
capture hashes, stored control scores, and candidate-score recomputation verified.

Cohort `exploratory`, scientific_evidence true. Pilot-eligibility is exploratory evidence only:
it unlocks no positive defense claim and privacy remains **undetermined** beyond the named
qualified coverage. A captured or declared channel is not an evaluated attack.

| Cell | Pilot-eligible | Mean Δ_U | Upper-95 Δ_U | Mean G_R | All pairs G_R > 0 | Privacy |
|---|---|---|---|---|---|---|
| d0-q8 | no | 0.0619 | 0.0675 | 0.0179 | True | undetermined |
| d1-s0.1-q8 | no | 0.0620 | 0.0675 | 0.0177 | True | failed |
| d1-s0.5-q8 | no | 0.0643 | 0.0683 | 0.0155 | True | undetermined |
| d1-s1.0-q8 | no | 0.0670 | 0.0720 | 0.0132 | True | failed |
| d3-q1 | no | 0.1094 | 0.1163 | 0.0369 | True | undetermined |
| d3-q2 | no | 0.1086 | 0.1226 | 0.0322 | True | failed |
| d3-q3 | no | 0.0941 | 0.1043 | 0.0181 | True | failed |

Gates (frozen): utility mean Δ_U ≤ 0.10 and mean + t(0.95, df=2)·sd/√3 ≤ 0.12; gain
mean G_R ≥ 0.01 with all seed pairs positive; privacy worst-seed candidate advantage ≤ 0.02
with every required binding qualified. A missing matched partner removes the pair and blocks
the complete-cohort gate; no outcome-based exclusion. Removed-remote comparisons are reported
separately per cell and are not gate inputs.

## Candidate advantages per seed/mode/target (descriptive)

| Seed | Cell | Mode | Target | Candidate advantage | Binding qualified |
|---|---|---|---|---|---|
| 411101 | d0-q8 | frozen_random | property | 0.8907 | False |
| 411101 | d0-q8 | frozen_random | structure | 0.1049 | True |
| 411101 | d0-q8 | removed_remote | property | 0.8743 | False |
| 411101 | d0-q8 | removed_remote | structure | 0.1014 | True |
| 411101 | d0-q8 | trained | property | 0.8324 | False |
| 411101 | d0-q8 | trained | structure | 0.0971 | True |
| 411101 | d1-s0.1-q8 | frozen_random | property | 0.7410 | True |
| 411101 | d1-s0.1-q8 | frozen_random | structure | 0.1031 | True |
| 411101 | d1-s0.1-q8 | removed_remote | property | 0.6872 | True |
| 411101 | d1-s0.1-q8 | removed_remote | structure | 0.0974 | True |
| 411101 | d1-s0.1-q8 | trained | property | 0.6740 | True |
| 411101 | d1-s0.1-q8 | trained | structure | 0.0958 | True |
| 411101 | d1-s0.5-q8 | frozen_random | property | 0.7456 | True |
| 411101 | d1-s0.5-q8 | frozen_random | structure | 0.1034 | True |
| 411101 | d1-s0.5-q8 | removed_remote | property | 0.6755 | True |
| 411101 | d1-s0.5-q8 | removed_remote | structure | 0.0980 | True |
| 411101 | d1-s0.5-q8 | trained | property | 0.6716 | True |
| 411101 | d1-s0.5-q8 | trained | structure | 0.0984 | True |
| 411101 | d1-s1.0-q8 | frozen_random | property | 0.7284 | True |
| 411101 | d1-s1.0-q8 | frozen_random | structure | 0.1049 | True |
| 411101 | d1-s1.0-q8 | removed_remote | property | 0.6745 | True |
| 411101 | d1-s1.0-q8 | removed_remote | structure | 0.1003 | True |
| 411101 | d1-s1.0-q8 | trained | property | 0.6700 | True |
| 411101 | d1-s1.0-q8 | trained | structure | 0.0969 | True |
| 411101 | d3-q1 | frozen_random | property | 0.8535 | True |
| 411101 | d3-q1 | frozen_random | structure | 0.0950 | True |
| 411101 | d3-q1 | removed_remote | property | 0.8041 | True |
| 411101 | d3-q1 | removed_remote | structure | 0.1257 | True |
| 411101 | d3-q1 | trained | property | 0.8405 | False |
| 411101 | d3-q1 | trained | structure | 0.1105 | True |
| 411101 | d3-q2 | frozen_random | property | 0.8375 | True |
| 411101 | d3-q2 | frozen_random | structure | 0.1010 | True |
| 411101 | d3-q2 | removed_remote | property | 0.8212 | True |
| 411101 | d3-q2 | removed_remote | structure | 0.1080 | True |
| 411101 | d3-q2 | trained | property | 0.7891 | True |
| 411101 | d3-q2 | trained | structure | 0.1123 | True |
| 411101 | d3-q3 | frozen_random | property | 0.8138 | True |
| 411101 | d3-q3 | frozen_random | structure | 0.1071 | True |
| 411101 | d3-q3 | removed_remote | property | 0.8379 | True |
| 411101 | d3-q3 | removed_remote | structure | 0.1022 | True |
| 411101 | d3-q3 | trained | property | 0.8037 | True |
| 411101 | d3-q3 | trained | structure | 0.1072 | True |
| 411102 | d0-q8 | frozen_random | property | 0.8641 | True |
| 411102 | d0-q8 | frozen_random | structure | 0.0956 | True |
| 411102 | d0-q8 | removed_remote | property | 0.8701 | True |
| 411102 | d0-q8 | removed_remote | structure | 0.0987 | True |
| 411102 | d0-q8 | trained | property | 0.8617 | True |
| 411102 | d0-q8 | trained | structure | 0.0907 | True |
| 411102 | d1-s0.1-q8 | frozen_random | property | 0.8048 | True |
| 411102 | d1-s0.1-q8 | frozen_random | structure | 0.0962 | True |
| 411102 | d1-s0.1-q8 | removed_remote | property | 0.8021 | False |
| 411102 | d1-s0.1-q8 | removed_remote | structure | 0.0940 | True |
| 411102 | d1-s0.1-q8 | trained | property | 0.8212 | True |
| 411102 | d1-s0.1-q8 | trained | structure | 0.0876 | True |
| 411102 | d1-s0.5-q8 | frozen_random | property | 0.8110 | True |
| 411102 | d1-s0.5-q8 | frozen_random | structure | 0.0953 | True |
| 411102 | d1-s0.5-q8 | removed_remote | property | 0.8032 | False |
| 411102 | d1-s0.5-q8 | removed_remote | structure | 0.0998 | True |
| 411102 | d1-s0.5-q8 | trained | property | 0.8152 | False |
| 411102 | d1-s0.5-q8 | trained | structure | 0.0898 | True |
| 411102 | d1-s1.0-q8 | frozen_random | property | 0.8089 | True |
| 411102 | d1-s1.0-q8 | frozen_random | structure | 0.0914 | True |
| 411102 | d1-s1.0-q8 | removed_remote | property | 0.8013 | True |
| 411102 | d1-s1.0-q8 | removed_remote | structure | 0.0973 | True |
| 411102 | d1-s1.0-q8 | trained | property | 0.8071 | True |
| 411102 | d1-s1.0-q8 | trained | structure | 0.0891 | True |
| 411102 | d3-q1 | frozen_random | property | 0.7893 | False |
| 411102 | d3-q1 | frozen_random | structure | 0.0842 | True |
| 411102 | d3-q1 | removed_remote | property | 0.8035 | False |
| 411102 | d3-q1 | removed_remote | structure | 0.1215 | True |
| 411102 | d3-q1 | trained | property | 0.8148 | True |
| 411102 | d3-q1 | trained | structure | 0.0994 | True |
| 411102 | d3-q2 | frozen_random | property | 0.8170 | False |
| 411102 | d3-q2 | frozen_random | structure | 0.0887 | True |
| 411102 | d3-q2 | removed_remote | property | 0.7857 | True |
| 411102 | d3-q2 | removed_remote | structure | 0.1021 | True |
| 411102 | d3-q2 | trained | property | 0.7360 | True |
| 411102 | d3-q2 | trained | structure | 0.1016 | True |
| 411102 | d3-q3 | frozen_random | property | 0.8102 | True |
| 411102 | d3-q3 | frozen_random | structure | 0.0965 | True |
| 411102 | d3-q3 | removed_remote | property | 0.8562 | True |
| 411102 | d3-q3 | removed_remote | structure | 0.0964 | True |
| 411102 | d3-q3 | trained | property | 0.7798 | True |
| 411102 | d3-q3 | trained | structure | 0.1003 | True |
| 411103 | d0-q8 | frozen_random | property | 0.8968 | True |
| 411103 | d0-q8 | frozen_random | structure | 0.1014 | True |
| 411103 | d0-q8 | removed_remote | property | 0.8834 | True |
| 411103 | d0-q8 | removed_remote | structure | 0.1017 | True |
| 411103 | d0-q8 | trained | property | 0.8863 | True |
| 411103 | d0-q8 | trained | structure | 0.0986 | True |
| 411103 | d1-s0.1-q8 | frozen_random | property | 0.8092 | False |
| 411103 | d1-s0.1-q8 | frozen_random | structure | 0.0997 | True |
| 411103 | d1-s0.1-q8 | removed_remote | property | 0.7369 | True |
| 411103 | d1-s0.1-q8 | removed_remote | structure | 0.0976 | True |
| 411103 | d1-s0.1-q8 | trained | property | 0.8075 | True |
| 411103 | d1-s0.1-q8 | trained | structure | 0.0972 | True |
| 411103 | d1-s0.5-q8 | frozen_random | property | 0.8126 | True |
| 411103 | d1-s0.5-q8 | frozen_random | structure | 0.0972 | True |
| 411103 | d1-s0.5-q8 | removed_remote | property | 0.7441 | True |
| 411103 | d1-s0.5-q8 | removed_remote | structure | 0.0993 | True |
| 411103 | d1-s0.5-q8 | trained | property | 0.8027 | True |
| 411103 | d1-s0.5-q8 | trained | structure | 0.0991 | True |
| 411103 | d1-s1.0-q8 | frozen_random | property | 0.8178 | True |
| 411103 | d1-s1.0-q8 | frozen_random | structure | 0.0966 | True |
| 411103 | d1-s1.0-q8 | removed_remote | property | 0.7447 | True |
| 411103 | d1-s1.0-q8 | removed_remote | structure | 0.1006 | True |
| 411103 | d1-s1.0-q8 | trained | property | 0.7992 | True |
| 411103 | d1-s1.0-q8 | trained | structure | 0.1011 | True |
| 411103 | d3-q1 | frozen_random | property | 0.7785 | True |
| 411103 | d3-q1 | frozen_random | structure | 0.0758 | True |
| 411103 | d3-q1 | removed_remote | property | 0.8050 | False |
| 411103 | d3-q1 | removed_remote | structure | 0.1168 | True |
| 411103 | d3-q1 | trained | property | 0.8247 | False |
| 411103 | d3-q1 | trained | structure | 0.0967 | True |
| 411103 | d3-q2 | frozen_random | property | 0.8519 | True |
| 411103 | d3-q2 | frozen_random | structure | 0.0912 | True |
| 411103 | d3-q2 | removed_remote | property | 0.8007 | True |
| 411103 | d3-q2 | removed_remote | structure | 0.1031 | True |
| 411103 | d3-q2 | trained | property | 0.7582 | True |
| 411103 | d3-q2 | trained | structure | 0.1072 | True |
| 411103 | d3-q3 | frozen_random | property | 0.7850 | False |
| 411103 | d3-q3 | frozen_random | structure | 0.0998 | True |
| 411103 | d3-q3 | removed_remote | property | 0.8451 | True |
| 411103 | d3-q3 | removed_remote | structure | 0.1030 | True |
| 411103 | d3-q3 | trained | property | 0.8321 | True |
| 411103 | d3-q3 | trained | structure | 0.1082 | True |

## Undetermined privacy coverage

- d0-q8: property x mean_forward_values
- d1-s0.5-q8: property x mean_forward_values
- d3-q1: property x mean_forward_values

Named omissions (freeze §5): timing; optimizer state beyond the remote module; full evolving remote state; active/adaptive attackers; within-run joint tensors; cross-run joint attacks; natural-sequence membership; canary membership; semantics beyond byte-structure classes; returned values; full-precision inbound/outbound gradients; quantizer metadata attacks; history accumulation; token inversion (secondary, unqualified here).

Exploratory pilot records under the frozen p4-defense-pilot-1.0 protocol. Pilot-eligibility, when evaluated by the analyzer, is exploratory evidence only and unlocks no positive defense claim; privacy remains undetermined. A captured or declared channel is not an evaluated attack.

See [verification](verification.json), [summary](summary.json), and [source record](source.json).
