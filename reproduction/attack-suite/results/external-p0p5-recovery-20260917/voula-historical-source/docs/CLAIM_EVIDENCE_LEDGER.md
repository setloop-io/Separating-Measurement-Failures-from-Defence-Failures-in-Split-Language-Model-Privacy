# Claim–evidence ledger

## Current evidence and interpretation — 2026-09-10

This section governs the manuscript and supersedes broader statements in the dated
records below. Those records are preserved as historical documentation. This revision
does not amend the original protocols, change original observations, or turn post-hoc
analysis into confirmation.

The small analysis inputs and original results are now committed. All 4,492
manifest-referenced artifacts were recovered locally and hash-verified; the private
release distributes the larger files. Follow [REVIEWER_EVIDENCE.md](REVIEWER_EVIDENCE.md)
for retrieval and the release verification record for upload completeness. Statements
below that summaries are ignored or unavailable describe the earlier repository state.

The exact historical execution snapshots and independent pre-outcome freeze timing
remain unestablished. Searching both Paper 1 repositories and Paper 2 did not recover
Paper 3's missing patches. Paper 1's E1 archive and original patch do exist, but belong
to its own experiments. See [HISTORICAL_PROVENANCE.md](HISTORICAL_PROVENANCE.md).
The manuscript therefore uses documented cohort-specific protocols and treats
historical "confirmatory" and "frozen" labels as recorded study designations, not
independently verified preregistration. Cross-setting replication was by the same team.

| Current statement | Retained evidence | Interpretation |
|---|---|---|
| P2 medium-high gain over frozen weights remains 0.01670185127 nats; the available bypass has removed-minus-trained loss −0.01542811445 nats | `outputs/confirmatory/confirmatory_analysis.json`; matching `outputs/confirmatory/*/metrics_summary.json` and `config.resolved.json`; original P2 manifests | Trained/frozen width is 256; the removed baseline uses width 64. The latter is a descriptive comparison with a lower-width alternative, not an isolated removal intervention. |
| P5: all seven D0 cells fail the qualified forward structure gate (maxima 0.2705078125–0.4833984375); all seven D1 cells fail utility (mean Δ_U 0.819109–1.559480 nats) | `outputs/delegation/delegation_analysis.json`, `outputs/delegation/gradient_qualification.json`, P4b qualification, original P5 manifests and final metrics | These necessary-condition failures establish the empty **joint** tested feasible set. No claim that each D1 cell independently fails a separately qualified privacy target is needed. This is not a test of every proposed gain/resource condition. |
| D1 spans 1, 3, 7 pass the stored qualified forward structure gate with maxima 0, 0.009765625, −0.00390625 | `outputs/delegation/delegation_analysis.json` cell-level structure fields | A structure-probe pass is not a privacy certificate. Stored property-family maxima include targets whose null qualification failed; the maximum alone does not identify a separately qualified rejecting target. Original family-rule classifications are preserved. |
| Retrospective target-resolved P5 property verification: all 112 aggregate scores reproduced exactly; punctuation worst-seed advantage 0.4496975535–1.2419896686 across 14 cells, each above 0.02 | `outputs/revision_validation_20260910/p5_property_refit.json`, `refit_p5_property.py`; original P4b calibration/capture hashes, P5 manifests, and unchanged `pcr.privacy_attacks_p4b` property scorer | New post-hoc verification supplies the missing target detail. Punctuation's original qualification is reused; vowel/newline remain individually unqualified. No new model training, requalification, or protocol amendment. Stored structure scores are not independently refitted. The manuscript retains the independent D0-privacy/D1-utility route to the joint result. |
| P5 removal control scope | Original P5 resolved configurations and `scripts/run_delegation.py` | D0 and the bypass use width 256 and q=8; D1 uses q=4 plus gradient perturbation while the available bypass remains D0. No isolated defended-removal effect is inferred. |
| Exploratory known-plaintext input-byte accuracy: q4 96.71630859375%; q32 97.802734375% | `outputs/adversarial_review_2026-09-08/within_run_probe.py`, `within_run_probe.json`, and `REVIEW.md`; sixteen manifest-bound P2b captures; separate `outputs/revision_validation_20260910/` verification | Post-hoc leave-one-sequence-out ridge classification: seven fixed 128-byte windows provide 896 known byte/activation pairs, the eighth is tested; eight training seeds at each precision. Shared text windows do not establish fresh-document recovery. Validation refits the probe on existing captures; it does not retrain models, establish historical source identity, or extend the result to P5. |

The P5 table now reports qualified forward **structure** outcomes instead of presenting
aggregated property-family failures as target-specific privacy rejections. Original
numeric analysis inputs and their provenance hashes are unchanged. Current presentation
code and documentation identify the revised display interpretation.

Privacy boundaries for the rows above: timing, optimizer state beyond the remote module,
full evolving remote state, active/adaptive attackers, within-run joint tensor attacks,
cross-run joint attacks, natural-sequence membership, unrestricted semantics, returned
values, and full-precision inbound/outbound gradients remain unevaluated. History-accumulation
and canary-membership instruments, undefended gradient property/token probes, and all
defended gradient probes remain unqualified. The known-plaintext refit adds no coverage
of these channels. Its attacker fitting setting differs from the original cross-model
token probe, so the comparison does not isolate coordinate mismatch as the sole cause.

Historical statements such as "Token-inversion never leaks", "H5: SUPPORTED", "H3:
CONFIRMED", "no cell holds privacy + anything", and "externally replicated" must be
read with this correction and the final manuscript. Neither the preserved historical
wording nor its presence in an archive overrides these current boundaries.

## Manuscript clarification — 2026-09-09

This is a documentation-only clarification of the existing P2–P5 results, not a new
experiment, retrospective protocol amendment, or promotion of exploratory review results.
It governs the revised `paper/main.tex` where historical decision prose below is broader.
The inspected source/configuration/manifest snapshot is commit `9aac3d0`; the referenced
generated summaries are present locally but ignored by Git. A later source snapshot does
not independently establish historical pre-outcome registration timing.

| Manuscript evidence | Existing provenance | Interpretation retained in the manuscript |
|---|---|---|
| C1: 0.0167 nats, p≈8.35e-7, n=8; long-low −0.0093, p=.002; long-high practical-gate failure | `docs/P2_CONFIRMATORY_DECISION.md`; `outputs/confirmatory/confirmatory_analysis.json` (`primary_family.medium-high.t_test`); `confirmatory_summary.md`; P2 manifests | Trained versus frozen contrast, not superiority over removal or necessity of remote work. P-value follows the current generated JSON rather than the rounded historical decision prose. |
| C1 replication: 12×192, 24 positive pairs, setting means 0.012–0.046 nats | `docs/P3_REPLICATION_DECISION.md`; `outputs/replication/replication_summary.md`; P3 manifests | Replication of remote-learning gain, not broadened privacy. |
| C2: selection r=−.99; held-out R² −7.98 versus .455; eight cells, seven retained seeds, calibration seed 929 | `docs/PREREGISTRATION_P2C.md`; `docs/P2C_SEGMENT_STATS_DECISION.md`; `outputs/segment_stats/segment_stats_summary.md`; `outputs/stats/selection_table.md`; `scripts/select_statistic.py`; P2c manifests | Selection used P2 cell means with P1 as sanity check. One calibration backbone supplies the statistic. No result supports superiority over span. |
| C4: q={4,8,16,32}; slope −.000711, t(7)=−13.7, p≈1.29e-6 | `docs/P2B_RATE_SWEEP_DECISION.md`; `outputs/rate_sweep/rate_sweep_analysis.json` (`primary.t_test`); `rate_sweep_summary.md`; P2b manifests | Forward precision response, practical plateau above 8; not literal equality or validated total serialized communication rate. P-value follows the current generated JSON. Full H3 at fixed broadened privacy risk remains unestablished. |
| C3/P4b: four frontier cells rejected; property maxima .66–1.64; structure maxima .08–.19 | `docs/PREREGISTRATION_P4B.md`; `outputs/privacy_extended_p4b/privacy_extended_p4b.json`; `privacy_extended_p4b_summary.md`; P4b calibration and P2b target manifests | Property range includes vowel/newline targets with failed null checks; punctuation alone qualifies cleanly. Family qualification follows the frozen at-least-one-target rule. Correlation differences are not probabilities or significance tests. |
| X3/P5: 288 runs; seven spans × two defense bundles; width 256; remote depth ceil(span/2); eight paired seeds; q8/D0 versus q4/clip1/noise.1/D1; D1 Δ_U .82–1.56 nats | `docs/PREREGISTRATION_P5.md`; `outputs/delegation/delegation_analysis.json`; `delegation_summary.md`; P5 manifests | Fixed-width, bundled-defense sweep on arch1/corpus1. D1 penalty is versus all-local, not isolated noise cost. G_R is descriptive; the third gate is analytic FLOP share. |
| P5 gates: worst-seed qualified advantage ≤.02; mean Δ_U≤.10 and upper 95% CI≤.12; analytic share≥.15 | `docs/PREREGISTRATION_P5.md`; `outputs/delegation/delegation_analysis.json` | Positive forward property scores reject every cell independently of missing gradient scores. The current artifact does not support historical prose that both forward families fail every cell. |

Additional interpretation boundaries:

- Privacy exclusions accompany the abstract, contribution summary, results, caption,
  limitations, and conclusion. They name timing; optimizer state beyond the remote module;
  full evolving remote state; active/adaptive attackers; within-run joint tensor attacks;
  cross-run joint attacks; natural-sequence membership; semantics beyond byte-structure
  classes; returned values; and full-precision inbound/outbound gradients. History-accumulation
  and canary-membership instruments are unqualified and were tested on calibration captures
  only. Sources: P4b/P5 preregistrations and `src/pcr/transcript.py`.
- Only the D0 structure probe qualifies on the processed inbound-gradient view. D0
  property/token variants and every D1 gradient variant are unqualified. No gradient-security
  conclusion follows for D1. Source: `outputs/delegation/gradient_qualification.json`.
- P5 scoring additionally quantizes gradient captures (`scripts/analyze_delegation.py`);
  only forward z is transport-quantized in `src/pcr/split.py`. FLOP/share estimates are
  analytic (`src/pcr/instrument.py`), and byte accounting uses a size formula
  (`src/pcr/transcript.py`), not a measured deployed transport. Memory measurement is not
  claimed. D1 probe captures consume the training noise stream (`src/pcr/train.py`,
  `src/pcr/split.py`); conclusions are conditional on the capture schedule.
- The original token probe transfers from calibration-model coordinates to target-model
  coordinates on a fixed final text panel (`scripts/score_privacy.py`, `src/pcr/data.py`).
  Its nondetection does not establish protection of byte identity. No post-hoc byte-recovery
  or control-contrast numbers from the adversarial review have been promoted into this
  manuscript's confirmatory results.
- No intervention shows that leaked properties are necessary for useful learning. The
  empirical feasible-set statement quantifies over tested operating points only. A finite
  sweep does not establish a universal communication bound or exclude other defenses.

Bibliographic verification and the limited novelty claim are documented in
`docs/LITERATURE_REVIEW_NOTES.md`.

## Presentation assets — 2026-09-09

`scripts/build_manuscript_visuals.py` renders the following displays without training or
new inferential tests. Regenerate with `make figures`. Exact source hashes and protocol
versions are in `paper/figures/manuscript_visuals_provenance.json`.

| Display | Existing evidence and selection | Uncertainty and scope |
|---|---|---|
| `study_controls.pdf` | Definitions in the problem formulation and matched control design | Schematic only; no empirical numbers. |
| `remote_gain_summary.pdf` | All six P2 primary/secondary cells, from `confirmatory_analysis.json`; all three P3 settings, from `replication_analysis.json`; corresponding P2/P3 committed manifests | Eight paired seeds per cell/setting, original unadjusted 95% intervals and individual gains. No pooling, privacy pass, or inferred superiority over removal. |
| `precision_and_prediction.pdf` | All four `rate_sweep_analysis.json.frontier` entries, using only `eval_loss`; all eight `segment_stats_analysis.json.cell_rows`, using frozen `pred_S`/`pred_N` and measured `delta_u`; corresponding P2b/P2c committed manifests | P2b n=8; P2c n=7 retained per cell; original 95% intervals. Fixed predictors have no displayed parameter-uncertainty estimate. Historical token-only frontier labels and byte-accounting estimates are not displayed. |
| Study-stage table | P1, P2, P2b, P2c, P3, P4a/P4b, P5 protocols and decision records linked above | Cohort inventory, not an aggregate inferential population or new result. |
| `paper/tables/p5_outcomes.tex` | All 14 cells of `outputs/delegation/delegation_analysis.json`, grouped into seven spans; mean gain, utility intervals, analytic share, and forward property-family gate; corresponding P5 committed manifests | Eight paired seeds per cell. Means/intervals copied from the JSON; property failure uses its forward score alone, not a missing D1 gradient score. Full exclusions accompany the caption. |

P2 and P2b p-values in the manuscript were corrected to the current generated JSON values
(8.354585995240527e-7 and 1.2914306934867137e-6). This changes neither frozen decision.
Historical decision prose below is preserved as a record and does not override this correction.

## Reviewer-response manuscript revision — 2026-09-09

This revision reports an existing descriptive result and clarifies scope; it adds no training,
attack execution, inferential test, or retrospective protocol amendment. Historical decision
language remains subordinate to the manuscript clarification above.

| Added statement/display | Evidence and provenance | Claim boundary |
|---|---|---|
| H4 descriptive extended-model out-of-sample R² = −11.09; rank three for four parameters; three segment definitions crossed with two widths | `docs/PREREGISTRATION_P2C.md`, Stage 2 secondary H4; `experiments/configs/segment_stats_matrix.yaml`; `outputs/segment_stats/segment_stats_analysis.json` fields `h4.r2_out_of_sample_full`, `h4.fit_full.rank`, `h4.fit_full.n`, `h4.fit_full.k`; `scripts/analyze_segment_stats.py` (`evaluate_h4`); committed P2c manifests in `experiments/manifests/`, including `trained-s3at4-high-s23.json` | Previously generated descriptive secondary analysis, not a new confirmatory test. Same held-out cells and retained-seed scope as C2. It does not establish the program-level H4. |
| Span and start position cannot be separately identified in that fit | `outputs/stats/selection_table.json`; P2c protocol selection design `{2@3, 4@2, 6@1} × {64, 256}`; `segment_stats_analysis.json.h4.fit_full.rank` | Design diagnostic: `start = 4 - span / 2`, so intercept/span/start columns are linearly dependent. More seeds at those same cells do not remove the dependence. This does not identify a unique cause of the H2 predictor's failure. |
| Table `tab:planexecution` and study-plan departure subsection | `docs/RESEARCH_PLAN.md`, §§5–9; phase protocols P2, P2b, P2c, P3, P5; detailed mapping in `docs/STUDY_PLAN_RECONCILIATION.md` | Selected planned-versus-completed scope, not an exhaustive preregistration compliance audit. Phase freezes do not retroactively complete the umbrella plan. |
| External broadened privacy evaluation remains unreplicated | `docs/PREREGISTRATION_P3.md` explicitly scopes execution to H1 replication with no new privacy evaluation; `docs/PREREGISTRATION_P5.md` explicitly defers arch2/corpus2 replication | No transfer of P3 utility evidence to privacy. All existing named unevaluated/unqualified channels remain excluded. |
| Reference model/corpus explicitly named: 8 layers, d=256, tiny-Shakespeare; P3 adds 12 layers, d=192, and War and Peace | `docs/PREREGISTRATION_P3.md`, Settings; `experiments/configs/replication_matrix.yaml`; P3 manifests, including `experiments/manifests/a1c2-trained-s19.json` and `a2c2-trained-s19.json`; `outputs/replication/replication_analysis.json` | Existing design identifiers, added to make the abstract and model description self-contained. The broadened privacy result concerns the reference setting only. |
| Resource constraints restrict otherwise eligible fixed candidates | Feasible-set definition in `paper/main.tex`; implementation evidence under Additional interpretation boundaries above | Set restriction, not a new deployment measurement. Changed behavior or transcripts require a new configuration evaluation. |
| Prospective validation priorities | `docs/REVIEWER_RESPONSE_PLAN.md` | Proposed studies only; no promised or completed new results and no newly frozen confirmatory protocol. |

The existing H4 analysis file has SHA-256
`18a0b55958bf5a95656fc771231d7cec24717aa1ec582abc8901c71bf1e083a2`;
its recorded selection-table SHA-256 is
`1bcc48ecc2738a013dde88783fe84b187dd8101f7f4d8192e10612afac998064`.
Generated outputs remain in their existing ignored output locations; the ledger links them to
the committed protocol, configuration, analyzer, and manifests without treating the current
documentation revision as a historical freeze.

## Historical study decisions

2026-09-08 (P5): delegation-share sweep under frozen preregistration `p5-delegation-sweep-1.0`
(+ amendment p5-1.1), 288/288 runs, 0 failures, fresh seeds 21–91. Decision record:
`docs/P5_DELEGATION_TRIANGLE_DECISION.md`. **Triangle verdict (frozen language): within the
swept space, defense options, and attack suite, no tested configuration holds all three gates** —
and no cell holds privacy + anything. utility+share holds at d0 spans 3–6. The D1 gradient
defense costs 0.82–1.56 nats and does not close the forward leak. (Analysis incident: a share-
truncation bug in the analysis layer was found and fixed with regression tests; verdict
unchanged.)

2026-09-07 (P4b): privacy re-qualification under frozen preregistration
`p4b-privacy-requalification-1.0` (12/12 fresh calibration runs, 0 failures). Decision record:
`docs/P4B_REQUALIFICATION_DECISION.md`. sequence-property-v2 and structure/semantic-v2
**QUALIFIED**; history-accumulation-v2 and canary-membership-v2 remain DISQUALIFIED.
**Under the qualified broadened suite, ALL FOUR frontier cells FAIL (leakage far above
threshold at every q, including 4-bit); the feasible set is empty.** C3's support is withdrawn
under broadened coverage; H5 revised to a confirmed negative at the tested configurations.

2026-09-07 (P4a): privacy-instrument broadening under frozen preregistration
`p4a-privacy-broadening-1.0` (12/12 calibration runs, 0 failures). Decision record:
`docs/P4A_PRIVACY_BROADENING_DECISION.md`. All four additional attack families
(sequence-property, structure/semantic, history-accumulation, canary membership) were
**DISQUALIFIED at instrument qualification** — frontier survival under the broadened suite is
UNDETERMINED. The privacy basis of C3/H5 remains the token-inversion family alone; the
unqualified families and omitted channels are named in the decision record and must accompany
any citation.

2026-09-07 (P3): external replication completed under frozen preregistration
`p3-replication-1.0` (96/96 runs, 0 failures, fresh seeds 9–89). Decision record:
`docs/P3_REPLICATION_DECISION.md`. **H1 REPLICATED in all three settings** (A2C2 primary
0.0322 nats p<1e-6; A2C1 0.0463 p<1e-6; A1C2 0.0118 p=2.9e-4; 24/24 pairs positive).
Downstream probes: no detectable representation harm (gaps ≤ 0.006).

2026-09-06 (P2c): segment-statistic study completed under frozen preregistration
`p2c-segment-statistics-1.0` (72/72 runs, 0 failures; seed 33 dropped per frozen missing-data
rule after a P2 run-id collision — see decision record). Decision record:
`docs/P2C_SEGMENT_STATS_DECISION.md`. **H2 NOT CONFIRMED**: the selected statistic
(intrinsic_dimension, exploratory r = −0.99) failed out of sample (R²_oos = −7.98 vs 0.455 for
nominal span). C2 remains hypothesis only.

2026-09-04 (P2b): rate sweep completed under frozen preregistration `p2b-rate-sweep-1.0`
(104/104 runs, 0 failures, fresh seeds 7–87). Decision record: `docs/P2B_RATE_SWEEP_DECISION.md`.
Analysis: `outputs/rate_sweep/rate_sweep_summary.md` + `rate_sweep_analysis.json`. H3/C4
confirmed in scoped saturated form; C3 strengthened to a 4-point feasible frontier segment;
C1 replicated on disjoint seeds at q=8.

2026-09-04: P2 confirmatory study completed under frozen preregistration
`p2-confirmatory-1.0` (129/129 runs, 0 failures, fresh seeds 11–88). Decision record:
`docs/P2_CONFIRMATORY_DECISION.md`. Analysis: `outputs/confirmatory/confirmatory_summary.md`
+ `confirmatory_analysis.json`; per-run manifests in `experiments/manifests/`; code revision
`be32144` + per-run dirty-patch hashes. C1 is unlocked in the scoped form below; C3 receives
single-cell support; C2, C4, C5 remain hypothesis only.

2026-09-03: P1 exploratory pilot completed (49/49 runs, 0 failures); go/no-go decision **GO**
recorded in `docs/P1_PILOT_DECISION.md`. Exploratory results cannot unlock any claim below;
statuses unchanged. Confirmatory prerequisites added there (width-256 privacy calibration, fresh
seeds, frozen practical-effect threshold).

| ID | Candidate claim | Required evidence | Current status | Allowed manuscript language |
|---|---|---|---|---|
| C1 | Trained remote computation contributes materially beyond a frozen/random transformation. | Predeclared trained-vs-frozen contrasts, matched budgets, ≥5 confirmatory seeds at headline cells, convergence check. | **Confirmed and externally replicated**: medium-high cell of the 8×256/Shakespeare mechanism (G_R = 0.0167 nats, p≈3e-7, n=8), replicated across q ∈ {4,8,16,32} (P2b) and across a 12×192 architecture, a second corpus, and both combined (P3: 0.012–0.046 nats, 24/24 pairs positive). Negative at long-low of arch1 (−0.0093, p=0.002); long-high failed the practical gate. | The scoped paragraph in `docs/P2_CONFIRMATORY_DECISION.md` plus replication numbers from `docs/P3_REPLICATION_DECISION.md` — must include the width/span moderation and the negative long-low result. |
| C2 | A predeclared segment property predicts substitutability better than nominal layer count. | Operationally defined predictor measured before the outcome, within-family spans/locations, nested/held-out predictive comparison with uncertainty, external replication. | **Tested and NOT confirmed (P2c)**: intrinsic_dimension selected under a frozen rule (r = −0.99 exploratory) failed out of sample on 8 held-out cells (R²_oos −7.98 vs 0.455 for span alone); position-degeneracy identified as the failure mode. Other statistics remain untested as primary. | “A first frozen candidate failed out-of-sample; we continue to hypothesize…” |
| C3 | A privacy-conditioned empirical delegability frontier exists. | Complete-view calibrated privacy evaluation, continuous utility and causal remote-gain outcomes, replicated feasible cells and uncertainty bands. | **Support WITHDRAWN under broadened coverage (P4b)**: all four P2b frontier cells fail the qualified sequence-property and structure/semantic families at every q ∈ {4,8,16,32} (leakage 4–80× threshold); feasible set empty at the tested configurations. Token-inversion-only pass retained as historical scope. History/membership instruments unqualified; timing/active/natural-membership channels unevaluated. | “Under the qualified broadened instrument, no tested configuration is simultaneously useful and private…” |
| C4 | Increasing operational boundary rate improves the optimized utility envelope. | Factorial rate sweep, comparable convergence, multiple seeds, monotonic/envelope analysis. | **Confirmed in scoped, saturated form** (P2b): slope −0.000711 nats/q-bit level, t(7)=−13.7, p≈3e-7, n=8 seeds; q4→q32 reduction 0.0295 nats; **entire response between q=4 and q=8, plateau above 8 bits/dim**. No constant exchange rate. | “Utility responds to measured boundary rate up to a mechanism-specific plateau near 8 bits per latent dimension…” |
| C5 | Depth effects are mediated by functional compressibility. | Matched-family depth variants, controlled budgets, mediation/predictive analysis, second setting. | Hypothesis only | “We test a possible mediation…” |
| X1 | Universal information-theoretic capacity ceiling. | Task-specific converse plus matching empirical scaling. | Withdrawn/not in scope | Prohibited |
| X2 | Exact 4:1 privacy/utility exchange rate. | Valid operational identity under matched extrema. | Withdrawn | Prohibited |
| X3 | Universal delegation triangle. | Identified feasible region across representative mechanisms and architectures. | Universal form remains **withdrawn/prohibited**. A scoped empirical form is now supported (P5): within the tested surrogate family, two boundary defenses, and the qualified attack suite, no tested configuration (7 spans × 2 defenses × 8 seeds) holds privacy + utility + delegated-share simultaneously, and no configuration holds privacy with either other axis. | Only: “In the tested mechanism family and attack coverage, we found no feasible triple…” with the P5 scope notes. |
| X4 | Depth is delegability. | Causal matched-depth experiment. | Withdrawn as prior claim | Prohibited as a result |

For every unlocked claim, add exact artifact paths, code revision, seed cohort, selection rule,
inferential unit, and limitations before manuscript inclusion.
