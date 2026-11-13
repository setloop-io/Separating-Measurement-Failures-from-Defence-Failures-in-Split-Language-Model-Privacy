# P4a extended privacy scoring (protocol p4a-privacy-broadening-1.0)

## Qualification verdicts (frozen floors)

| Family | verdict | key numbers |
|---|---|---|
| sequence_property | DISQUALIFIED | vowel_fraction_bucket: adv -0.0049 (floor 0.05), null pipeline +0.0000; digit_presence: adv +0.0000 (floor 0.05), null pipeline +0.0000 |
| structure_semantic | DISQUALIFIED | adv +0.0356 (floor 0.05), null pipeline +0.0000 |
| history_accumulation | DISQUALIFIED | min adv +0.0000 (floor 0.02), max |null adv| 0.0000 (band 0.02) |
| canary_membership | DISQUALIFIED | min AUC 0.5039 (floor 0.9), max |null AUC - 0.5| 0.0938 |

## Frontier survival

| Cell | qualified-family checks | survives |
|---|---|---|
| medium-high-q4 | none (no qualified family) | UNDETERMINED |
| medium-high-q8 | none (no qualified family) | UNDETERMINED |
| medium-high-q16 | none (no qualified family) | UNDETERMINED |
| medium-high-q32 | none (no qualified family) | UNDETERMINED |

## Named omissions (per AGENTS.md rule 6)

- timing channels
- attacks on optimizer state beyond the remote module
- active/adaptive attackers that modify queries
- semantic attacks beyond byte-structure classes
- membership inference of natural (non-canary) sequences
- joint cross-run attacks
- families 3-4 (history accumulation, canary membership) evaluated on calibration captures only: confirmatory runs lack dense transcripts
