# R1 status — 17 September 2026: exploration verified and reported

Supersedes the running-state line in the 16 September scientific status.
The historical stage packets and frozen protocol documents are unchanged.

## Completed

- Toula `exp002`: **232/232 complete**, no failures, finish
  `2026-09-17T16:04:38.180890+00:00`, wall time 76,684.199 s
  (21 h 18 min), within 86,400 s cap. No new training was launched in this review.
- Original 191-file scientific bundle verified. Repaired analyzer `a9700e9`
  handles the runner's canonical JSON wrapper and hash-bound qualification
  proposal, and verifies all manifest artifacts and provenance links.
- Calibration reanalysis: 76/76, 3,572 artifact bindings verified; all thresholds
  and qualification combination records exactly reproduce the frozen proposal.
- Exploration: 10,088 artifact bindings verified; 76 matched cell/seed triplets
  pass retained initialization and configuration checks.
- Full exploration archive copied from Toula to Koula over the direct link:
  **10,790 files / 111,859,148,305 bytes**, all bytes and the complete file set
  independently match the origin inventory. Origin remains retained.
- All 105 R1 tests pass. Generated all-cell/per-seed data, uncertainty figure,
  results table, claim ledger and exploratory manuscript. Analysis and figures
  carry source/input/output hash bindings.

## Scientific outcome

All **19 configurations are REJECTED under the frozen exploratory rules**:
13 qualified privacy breaches, five utility gate failures and seven learning-gain
failures, with overlapping reasons. This is an empty tested feasible set under
those decision rules, **not a confirmed population-level or universal trilemma**.
Four seeds give descriptive unadjusted Student-t intervals. Failure to satisfy
a gain gate is not proof that the population effect is zero.

Calibration qualified only four of 24 required instruments. Six configurations
have undetermined privacy assessments but are rejected independently on utility
or gain. Width-64 is gain-only rejected; its privacy is undetermined. Only depth-2
and depth-4 satisfy the separate benefit-over-removal rule.

## Still pending / decision

[Confirmation review](../R1_CONFIRMATION_REVIEW.md) is concrete but **NO-GO as an
adequately powered eight-seed full-family design**. It proposes 464 runs, 60 h
owned-Toula cap and 672 GiB storage reservation. Exact conditional power analysis
puts the probability of confirming all 19 rejections at no more than 26.4% under
plug-in exploratory effect/SD estimates; those estimates are themselves fragile.
The declared power target is not met. No confirmation admission or launch exists.

The current paper is an explicitly exploratory report. To claim confirmation,
first choose a scientifically justified narrower hypothesis or larger cohort,
revise power and budget, implement the corresponding confirmation stage, freeze
and independently review its packet, then obtain stage-specific authorization.
Do not run the eight reserved seeds merely to complete a checklist.

Unqualified privacy instruments and omitted views limit claims. Qualifying a
broader suite requires fresh instrument development/calibration and a prospective
amendment; it is not required to report the current exploratory failures honestly.
A public/reviewer-accessible release and external manuscript review remain
publication tasks; the present evidence repository and owned-node archives are private.

See [evidence package](../../evidence/r1-exploration-20260917/README.md) and
[claim ledger](../CLAIM_EVIDENCE_LEDGER.md). Older R1 plan/runbook text describes
its historical preparation state; it must not be mistaken for today's status.
