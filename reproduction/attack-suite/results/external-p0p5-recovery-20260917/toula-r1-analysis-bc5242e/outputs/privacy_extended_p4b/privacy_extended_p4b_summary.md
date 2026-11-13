# P4b privacy re-qualification (protocol p4b-privacy-requalification-1.0)

## Qualification verdicts (frozen floors)

| Family | verdict | key numbers |
|---|---|---|
| sequence_property_v2 | QUALIFIED | newline_count: r_adv +0.5440, null +0.0896; punctuation_fraction: r_adv +0.7148, null +0.0029; vowel_fraction: r_adv +0.7543, null +0.0278 |
| structure_semantic_v2 | QUALIFIED | adv +0.2931 (floor 0.02), null -0.0140 |
| history_accumulation_v2 | DISQUALIFIED | min adv +0.1024 (floor 0.02), max |null| 0.0390 (band 0.02) |
| canary_membership_v2 | DISQUALIFIED | min AUC 0.4636 (floor 0.75), nulls within band: True |

## Frontier survival

| Cell | qualified-family checks | survives |
|---|---|---|
| medium-high-q16 | sequence_property_v2: FAIL; structure_semantic_v2: FAIL | NO |
| medium-high-q32 | sequence_property_v2: FAIL; structure_semantic_v2: FAIL | NO |
| medium-high-q4 | sequence_property_v2: FAIL; structure_semantic_v2: FAIL | NO |
| medium-high-q8 | sequence_property_v2: FAIL; structure_semantic_v2: FAIL | NO |

## Named omissions (per AGENTS.md rule 6)

- timing channels
- attacks on optimizer state beyond the remote module
- active/adaptive attackers that modify queries
- semantic attacks beyond byte-structure classes
- membership inference of natural (non-canary) sequences
- joint cross-run attacks
- families 3-4 (history accumulation, canary membership) evaluated on calibration captures only: confirmatory runs lack dense transcripts
- family-4 scalar score = mean of the frozen 64-dim feature vector (feature set frozen; reducer fixed by construction and documented)
- family-3 evaluation step at q=4 reuses the step selected by the frozen saturation rule on the q=32 capture (rule is defined at q=32)
