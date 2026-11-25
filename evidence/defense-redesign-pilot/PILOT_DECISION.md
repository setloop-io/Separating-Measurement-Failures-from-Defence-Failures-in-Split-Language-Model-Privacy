# Pilot decision record — Paper 4 defense pilot 1.0

Recorded 2026-09-22. Evidence: campaign `pilot-20260922-a`, frozen protocol
[p4-defense-pilot-1.0](PILOT_FREEZE.md) with freeze record
`experiments/freeze/p4-defense-pilot-1.0.json` (effective at commit `764d3f2`),
66 `p4-pilot-20260922-a-*` run manifests (cohort **exploratory**,
`scientific_evidence: true`; per-seed subsets 411101+411102 on toula, 411103 on
koula, CUDA deterministic kernels), the analyzer-generated
[report](../outputs/runs/pilot-20260922-a/report.md) and
[verification](../outputs/runs/pilot-20260922-a/verification.json). All 66 runs
completed; every manifest re-verified after merge; captures hashes verified; all
63 split runs' property and structure candidate scores recomputed exactly.
Pilot outcomes are **exploratory**: they are not confirmation, not a positive
defense claim, and not a universal statement.

## Gate outcomes (frozen rule; 3 paired seeds per cell; 16,384-window panels)

| Cell | mean Δ_U | upper 95% limit | utility | mean G_R | min pair | gain | privacy | Pilot-eligible |
|---|---|---|---|---|---|---|---|---|
| d0-q8 | 0.0619 | 0.0675 | pass | 0.0179 | 0.0126 | pass | undetermined | **No** |
| d1-s0.1-q8 | 0.0620 | 0.0675 | pass | 0.0177 | 0.0127 | pass | failed | **No** |
| d1-s0.5-q8 | 0.0643 | 0.0683 | pass | 0.0155 | 0.0120 | pass | undetermined | **No** |
| d1-s1.0-q8 | 0.0670 | 0.0720 | pass | 0.0132 | 0.0111 | pass | failed | **No** |
| d3-q1 | 0.1094 | 0.1163 | **fail** | 0.0369 | 0.0306 | pass | undetermined | **No** |
| d3-q2 | 0.1086 | 0.1226 | **fail** | 0.0322 | 0.0235 | pass | failed | **No** |
| d3-q3 | 0.0941 | 0.1043 | pass | 0.0181 | 0.0168 | pass | failed | **No** |

Privacy detail (worst-seed candidate advantage, 0.02 acceptance): punctuation
0.79–0.89 on every cell where bindings are qualified (four cells additionally
have one marginal unqualified binding each → coverage *undetermined*, never a
pass); next-byte structure 0.097–0.112 on every cell. The punctuation signal
remains decodable at roughly 40× the acceptance threshold on every tested
defense; D1 σ=1.0 lowers it by only ~0.03–0.08 versus d0 (0.81 vs 0.86); D3
does not lower it at any tested q. Removed-minus-trained means: −0.004 to
+0.015 across cells (descriptive; reported separately as frozen).

## Decision

**No cell in the frozen tested family is pilot-eligible.** Utility passes the
reference, all D1 strengths, and d3-q3 (and fails only d3-q1/d3-q2); the causal
remote-gain gate passes every cell with all-positive pairs; the privacy gate
fails outright or remains undetermined on every cell. Within the finite tested
family (D0 q8; D1 σ∈{0.1,0.5,1.0} at q8 with scheduled rank-1 refits; D3
q∈{1,2,3}; span4@2, width 256, 2,000 steps, the pinned `gutenberg_p4` corpus,
16,384-window qualified instruments), the exploratory pilot supports the finite
alternative: the tested family contains **no eligible cell** (H4 in the research
plan's finite sense). H1 and H3 are not supported by these measurements. This
establishes nothing about untested mechanisms, strengths, segments, budgets,
targets, channels, or threat models, and nothing about universal impossibility
or about whether leaked properties are necessary for useful remote learning.

## What this does and does not unlock

- Unlocks: per the frozen rules, only engineering or prospectively amended
  follow-up. The measured cell-level outcomes justify new engineering on
  genuinely different mechanisms (e.g., the separately specified D2 adversarial
  encoder, higher-rank or multi-target D1 variants, or D1+D3 interaction cells
  — each requiring its own prospective protocol and, for positive claims, its
  own qualification and confirmation).
- Does not unlock: any privacy, security, or defense-effectiveness statement;
  any confirmatory claim; any statement beyond the tested family. Confirmatory
  work on a future *eligible* cell would require the confirmatory freeze
  (fresh seeds 412101–412108, a frozen paired test and multiplicity family).
  The named omissions of the freeze §5 remain unevaluated everywhere:
  timing, optimizer state, evolving remote state, active/adaptive attackers,
  joint/cross-run attacks, history, membership, semantics beyond byte classes,
  returned values, full-precision gradients, quantizer metadata attacks, token
  inversion.

## Operational and analysis disclosures

- Four pilot bindings (one seed each on d0-q8, d1-s0.1-q8, d3-q1, d3-q3
  property views) were unqualified by the frozen all-16-trials rule on marginal
  no-signal excursions (max 0.021–0.026), consistent with the estimator's
  measured ~1.5% marginal excursion rate; their coverage is reported
  *undetermined*. No cell's eligibility depends on any of these bindings.
- The pilot analyzer was revised after execution (device-override tolerance),
  hash-recorded in verification.json; it changed no executed record. The pilot
  runner and freeze record are unchanged since `764d3f2`; any future pilot
  re-execution would require a new freeze record (post-freeze source edits
  correctly refuse).
- All campaign incidents from the engineering stages were disclosed in
  [BUDGET_SCAN_DECISION.md](BUDGET_SCAN_DECISION.md); no pilot-stage incidents
  occurred.
