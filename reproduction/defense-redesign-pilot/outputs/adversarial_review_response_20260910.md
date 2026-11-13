# Assessment of the adversarial review of Paper 3

Reviewed on 2026-09-10. The supplied review concerns Paper 3 at `52191c4`.
The current Paper 3 revision inspected is `983598ec4725c1d86dcfde3c814f9d935cdc2843`;
Paper 4's evidence release targets `c0eb5094b049294c717e7b212e908da1c9506004`.
The review is evidence to assess, not authorization to run its proposed experiments.

## Overall assessment

The review is substantially fair about the scientific scope, historical provenance,
and deployment limitations. Its recommendation for major revision is reasonable.
However, some evidence-availability findings describe the older checkout, and its
proposed minimum repair path mixes necessary corrections with optional new research.

The paper can defensibly present a finite empirical evaluation of learned remote
contribution and measured leakage. It cannot currently claim independently auditable
preregistration, exact historical training reproduction, a deployed remote system,
or complete-transcript privacy. Publishing recovered files repairs access; it does
not repair the missing execution-source history.

Two additional qualifications emerged from checking the review:

1. The headline P2 bypass comparison changes latent width as well as removing the
   remote module. The bypass is a useful cheaper alternative, but the contrast does
   not isolate removal alone.
2. P5's claim that every cell independently fails a separately qualified privacy
   target is stronger than the retained target-aggregated scores establish. The
   empty **joint** feasible set nevertheless follows through a simpler route:
   all seven D0 cells fail the qualified structure gate; all seven D1 cells fail utility.

## What has changed since the reviewed commit

| Review assertion | Current assessment |
|---|---|
| All 4,492 referenced artifacts are unavailable | Accurate for the old Git-only checkout. All were recovered locally and matched their original SHA-256 hashes. Currently 3,720 referenced artifacts are tracked in Git; 772 tensor artifacts require the release archive. |
| Five display-source analysis JSONs are absent | Resolved for access: all five are now committed, unchanged, and match the previously committed display provenance. Four stored-curve analyses were recomputed in a fresh checkout with matching numeric values. Privacy refitting and P5 recomputation were not part of that check. |
| The 96.72%/97.80% attack cannot be recovered | The original script, result JSON, and review are now committed. Sixteen captures were recovered and hash-verified. Stored aggregation matches those percentages; classifier fitting has not been independently rerun in these audits. |
| README still says there are no results | Correct at `52191c4`; the current README describes the completed empirical analyses and provenance limitations. |
| No reviewer retrieval path exists | Guides, archive indexes, checksums, and a validating downloader now exist for both private repositories. Paper 4's release and fresh-checkout checks completed. Paper 3's 39.22 GB compressed archive is still uploading; full reviewer access to its tensors remains pending. |
| Historical source and preregistration timing are not auditable | Still unresolved. Neither artifact publication nor the later source snapshot supplies the missing dirty-tree patches or independently establishes pre-outcome timing. |

Paper 3's archive indexes **5,308 files**, including every retained output, exact
corpus inputs, and original run manifests. Paper 4 indexes **289 files**, including
all 173 manifest-referenced artifacts and five historical engineering source archives.
Both repositories remain private, as requested; reviewers require access to each.

See the dated [initial evidence audit](paper3_evidence_audit_20260910/report.md),
Paper 3's [reviewer guide](https://github.com/geopolitis/paper-3-privacy-constrained-remote/blob/983598ec4725c1d86dcfde3c814f9d935cdc2843/docs/REVIEWER_EVIDENCE.md),
and [reproduction check](https://github.com/geopolitis/paper-3-privacy-constrained-remote/blob/983598ec4725c1d86dcfde3c814f9d935cdc2843/evidence/reproduction-check-20260910.json).
Paper 4's [verification record](../evidence/reproduction-check-20260910.json) covers
all five engineering stored-output analyzers. These checks do not convert engineering
diagnostics into scientific results.

GitHub subsequently returned API rate-limit errors during release-status checks.
The Paper 3 finalizer has been restarted under a supervisor that waits before retrying
and respects the reported rate-limit reset. An API lookup failure is not evidence that
the already-published Paper 4 release disappeared. Release availability must be distinguished
from scientific reproducibility and from the temporary status of the upload.

## Numerical claims: agreement, not an arithmetic dispute

Independent SciPy checks agree with the repository's t-test implementation for
these contrasts. The review used rounded table entries; the original full-precision
values explain its small numerical differences exactly.

| Quantity | Full-precision result | Interpretation |
|---|---|---|
| P2 medium-high, frozen minus trained | Mean 0.01670185127; t(7)=14.62314077; one-sided p=8.354586e-7; unadjusted 95% CI [0.01400109, 0.01940261] | Improvement over freezing, within this matched intervention. |
| P2 removed minus trained | Mean -0.01542811445; two-sided p=0.000494631; unadjusted 95% CI [-0.02141589, -0.00944034] | The retained cheaper bypass has lower loss; see the width mismatch below. |
| P2b per-seed loss slope against forward precision | Mean -0.000711125473; t(7)=-13.71461257; one-sided p=1.291431e-6 | A precision response, not a complete-transcript rate theorem. |

The current manuscript **and `52191c4`** already use the full-precision P2 and P2b
p-values. These numbers do not need another correction. Twenty-four raw P2 final-loss
records and 32 P2b trained-run records independently reproduce the corresponding
stored contrasts and slopes. The selected intrinsic-dimension predictor's held-out
failure remains the appropriate finding; it does not refute every possible segment statistic.

Sources: `outputs/confirmatory/confirmatory_analysis.json` (cell gains and primary test),
`outputs/rate_sweep/rate_sweep_analysis.json` (slopes and primary test),
`scripts/stats_utils.py:129`, and `paper/main.tex:343,394` in Paper 3.

## The bypass critique needs stronger control disclosure

The reviewer is right that gain over frozen-random weights does not establish that
remote computation is necessary or superior to removal. The manuscript already
states this distinction, but should show the alternative explicitly beside the headline gain.

The missing qualification is that the P2 medium-high trained surrogate uses latent
width **256**, whereas the retained medium removed-remote control uses **64**.
Encoder/decoder dimensions differ, and shared random streams do not guarantee identical
later parameter draws once the construction dimensions change. Therefore, report the
comparison as performance against the available lower-width bypass, not as a
width-matched isolated removal intervention. The blanket phrase “dimension-matched
bypass” at `paper/main.tex:223` needs cohort-specific clarification.

This is not a universal finding that a bypass wins. For example, the retained P5 D0
span-4 comparison, with width 256 and q=8 on both sides, gives removed minus trained
loss **+0.013458 nats**, unadjusted paired 95% CI **[0.004140, 0.022775]** across eight
seeds. These are descriptive checks of existing observations, not new primary tests.
P5 D1 is compared with the existing D0 bypass, which retains q=8 and no gradient
defense; that is another practical alternative, not an isolated defended-removal control.

Do not retroactively insert a new primary bypass gate into P2 or P5. Display and label
the existing contrasts now. For future Paper 4 cohorts, define a fully matched bypass
and its acceptance rule before execution if superiority over removal is an intended claim.

Sources: P2 trained/removed `config.resolved.json` files, `scripts/pilot_common.py:133`,
`src/pcr/train.py:99`, P5 raw trained/removed metrics, and `scripts/run_delegation.py:57,69`.

## P5: preserve the joint result and narrow the privacy-only statement

The review correctly distinguishes the formal utility/privacy/causal-gain/resource
definition from P5's utility/privacy/analytic-share gates. This does not invalidate
an empty-set conclusion when a necessary condition fails, but P5 must not be presented
as a completed operational test of every component of the formal definition.

A further issue appears in `scripts/analyze_delegation.py:245–248`: per-target property
scores are reduced to the maximum advantage and the target details are discarded from
the retained cell summary. Punctuation qualified, while the vowel/newline calibration
null checks failed. A large maximum across all targets does not establish which
separately qualified target caused a cell's rejection.

For D1 spans **1, 3, and 7**, the qualified forward structure maxima are respectively
**0, 0.009765625, and -0.00390625**, all within the 0.02 gate. Their privacy-only
rejection therefore relies on the aggregated property family result. The frozen
family-rule classification can be described as such; the stronger claim of rejection
by an individually qualified property target needs retained target-resolved scores
or a separately labelled recomputation. This audit did not refit those probes.

The joint finite-set finding has an independent, narrower justification:

- **All seven D0 cells:** the qualified forward next-byte structure maximum exceeds
  0.02, ranging from 0.2705078125 to 0.4833984375.
- **All seven D1 cells:** mean replacement penalties range from **0.819109 to 1.559480**
  nats, far above the 0.10 utility gate. All 56 trained-minus-all-local seed differences
  and all seven means match the raw final metrics.

Consequently, none of these 14 points satisfies the joint gates, conditional on the
retained structure-probe evidence and its stated qualification. This does not require
claiming that every D1 configuration independently fails a qualified privacy target.
It also does not show that another encoder, defense budget, width, objective, or deployment
would fail. Publishing the artifacts permits further scrutiny; it does not independently
validate every stored attack score.

The harm-model criticism is also fair: punctuation fractions and byte classes are
evidence of information exposure under chosen diagnostic targets, not a direct measure
of sensitive-document disclosure or membership harm. State the target and disclosure
event explicitly. Public literary corpora and a contiguous held-out portion of the same
work do not validate generalization to private workloads or new documents.

The recovered known-plaintext byte probe is a more direct reconstruction diagnostic,
but it changes attacker knowledge and fitting conditions relative to the original
cross-model probe. That contrast does not isolate coordinate mismatch as the sole cause
of nondetection. Stronger known-plaintext performance is not automatically an active
attack or complete-transcript coverage.

## Corrections warranted before submission

1. **Qualify preregistration in the manuscript itself.** “Staged empirical study with
   documented cohort-specific protocols” is supportable. Add an explicit statement
   that independent pre-outcome freeze timing and the exact historical execution
   source could not be recovered. The ledger disclosure does not justify unqualified
   “preregistered” in the abstract (`main.tex:41`) and introduction (`:74`). Lack of
   timestamped evidence is not, by itself, proof that protocols were chosen after outcomes.
2. **Use “cross-setting replication by the same team.”** The changed architecture and
   corpus are useful checks, but “external” can imply an independent laboratory.
   No broadened privacy replication occurred in those settings.
3. **Make the control differences and joint-gate reasoning visible.** Add the available
   bypass comparison with its width/defense limitations, and use the D0-privacy/D1-utility
   reasoning above. Preserve original manifests and historical classifications.
4. **Make the recovered byte attack visible as exploratory.** Report its known-plaintext,
   fixed-window, leave-one-sequence-out setting; do not relabel it as confirmation or
   extend its percentages to P5. A fresh document-disjoint study is a valuable future
   test, not a retrospective change to the original evidence.
5. **Repair the reproduction entry points prospectively.** The original training helper
   still treats `status == completed` as sufficient (`src/pcr/manifest.py:121`), and
   drivers skip missing artifacts. The new evidence verifier does not repair that
   old driver behavior. Preserve historical records and reject missing/corrupt/wrong-identity
   evidence rather than silently replacing it with a newly trained run.
6. **Replace the stale scaffold assertion with an appropriate manuscript check.** The
   targeted test run produced **1 failure and 2 passes**: `tests/test_scaffold.py:19`
   still demands `PLANNED---NO DATA`. Do not restore false “no data” text just to pass it.
   The full suite was not run for this review.
7. **Finish dataset and reviewer-permission documentation.** Exact corpus bytes and
   hashes are now packaged, but the data-policy provenance manifests are still absent.
   Record source/version, retained notices, byte preprocessing, and explicit split ranges,
   while preserving the unrecovered original split-digest derivation as a limitation.
   The current `LICENSE` restricts redistribution/modification/reuse; clarify the intended
   reviewer-use permissions through an owner decision. The user's private-repository
   requirement does not imply permission to make the repositories public or select an
   open-source license automatically.

A more balanced title would be **“Beyond Layer Count: Causal Learning Gain and Measured
Leakage in Small Split Language Models.”** This is an editorial recommendation, not a
claim that the current finite-set title is inherently false. The reference-only privacy
scope is already explicit in the abstract and conclusion.

## Related work: the omissions are relevant, the count is not the argument

The fact that there are 13 bibliography entries does not measure scientific quality.
The stronger criticism is that the following close comparisons are absent as exact
entries. Primary-source checks support adding them with explicit distinctions:

| Work | Verified source/status | Comparison to add |
|---|---|---|
| Yao and Li, *Is Split Learning Privacy-Preserving for Fine-Tuning Large Language Models?* | [Author-hosted accepted manuscript](https://iqua.ece.toronto.edu/papers/dixiyao-tbd24.pdf), IEEE Transactions on Big Data, DOI 10.1109/TBDATA.2024.3524101 | Direct split-LLM fine-tuning privacy precedent; distinguish pretrained client-weight knowledge, attack exclusions, and production model families from seeded byte models. |
| *SIMBA: Split Inference—Mechanisms, Benchmarks and Attacks* | [ECCV 2024 paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09762.pdf), [publisher record](https://link.springer.com/chapter/10.1007/978-3-031-73116-7_13) | Benchmark integration of mechanisms, attacks, and privacy/utility metrics is established. The different proposed contribution is the remote-learning intervention plus held-out segment prediction. |
| *SAP: Privacy-Preserving Fine-Tuning on Language Models with Split-and-Privatize Framework* | [IJCAI 2025 Main Track proceedings](https://www.ijcai.org/proceedings/2025/57), pp. 502–510 | A direct split fine-tuning/DP defense comparison. Use the proceedings title and authors rather than relabelling its differently titled older preprint. P5 does not implement SAP. |
| *Differentially Private Label Protection in Split Learning* (TPSL) | [Primary paper, arXiv:2203.02073](https://arxiv.org/pdf/2203.02073) | A direct precedent for transcript privacy, with a specified label-adjacency DP guarantee. Empirical attack advantages on text do not supply that guarantee. Conference acceptance was not verified. |
| Gupta and Raskar, *Distributed learning of deep neural network over multiple agents* | [Journal of Network and Computer Applications 116 (2018), 1–8](https://www.sciencedirect.com/science/article/pii/S1084804518301590) | Foundational SplitNN attribution; add beside the already-cited Vepakomma paper, rather than implying there is no early split-learning citation at all. |

These comparisons neither establish priority for the exact proposed combination nor
prove it was already published. Retain the modest combination claim and avoid a “first”
claim without a wider search. Hardware/network measurements and fresh full-transcript
attacks would strengthen a larger study, but they are not prerequisites for every
properly scoped negative empirical paper. They become prerequisites for the stronger
deployment or positive privacy claims they are intended to establish.

## Recommended response to the reviewer

Accept the provenance, control-disclosure, systems-scope, data-diversity, and close-literature
criticisms. Explain that the original evidence was omitted from the checkout rather than
lost, identify the newly committed results and archive, and state honestly that the full
Paper 3 upload is pending. Keep historical-source recovery separate from artifact access.
Clarify the rounded numerical comparisons, the unmatched P2 bypass, and the narrower
independent justification for the joint P5 result. Separate immediate manuscript/software
corrections from prospectively registered new research, preferably in Paper 4.

## Method and changes made for this assessment

The paper-code-audit skill was applied manually; its `/audit` launcher is not exposed.
A verifier independently checked retained raw metrics and statistical calculations;
a researcher checked the named literature against primary sources. The main audit
inspected the manuscript, controls, P5 score aggregation, Git history, artifact inventories,
and reproduction records, and ran only the three scaffold tests.

No model training, attack fitting, historical manifest changes, manuscript changes,
bibliography edits, or scientific source changes were performed. This report and a
companion JSON record were created in Paper 4's outputs. Separately, the already-authorized
Paper 3 publication job was given a temporary retry supervisor after GitHub rate limiting;
that operational recovery changes no experimental evidence or repository visibility.
