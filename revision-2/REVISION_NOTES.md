# Manuscript revision 2 notes

6 November 2026

The second revision recasts the paper as a retrospective, artifact-backed audit of which split-language-model privacy conclusions the recorded evidence supports. Its title is **Auditing Privacy Claims in Split Language-Model Training**. It preserves genuine negative defense results while correcting unsupported explanations and security claims in the original synthesis.

All new material is under `revision-2/`. The existing manuscript, compiled paper, bibliography, and repository README remain unchanged. Their recorded hashes were checked against `preservation-manifest.json`; all 11 preserved-file entries matched when these notes were written.

## Substantive changes

The original claim that four episodes all demonstrated instrument failure is replaced by distinct outcomes: a qualified breach, qualified non-detection, undetermined evidence, invalid support for the claimed target, and an out-of-scope claim. The revision retains each campaign's aggregation rule. In particular, an individual qualified probe's leakage evidence does not silently overwrite Paper 4's recorded all-bindings composite verdict.

The contribution is the reconciliation of claims across campaigns and abstraction levels. It is separate from the existing LLM-Attack Suite's observation contracts, controls, saved predictions, missing-attack disclosures, and measurement checks. The paper now discloses that overlap and does not claim a newly validated attack suite, a first-for-training result, a universal value/structure dichotomy, or a new defense.

The manuscript adds reconstructed campaign tables, source-linked figures, explicit repetition units and uncertainty conventions, a threat-model scope argument, and an analytic explanation of why a small mutual-information lower-bound readout is not a privacy ceiling.

## Corrected claims and provenance

| Evidence | Correction retained in revision 2 | Basis of this revision |
|---|---|---|
| **F1 / E1: decoy gradients** | The client holding the private loss returns gradients to the cloud. Exact zeros identify real/decoy rows, not token identities by themselves. The source reports 4,096/4,096 frames per run across nine seeds; content gain is separately +0.65–1.50 percentage points. Approximately 0.01 nats is mitigation's held-out utility cost. | Source-reported Paper 1 mechanism and results; the new analysis does not recount every frame or refit probes. |
| **F2 / B1: boundary replacement** | Next-token labels do not test current-token reconstruction. Six clean zero-return comparisons do not demonstrate remote necessity. A roughly +0.034-nat NLL change coexists with answer-token match falling 34.79% to 18.15%, with the source's checkpoint and metric limits retained. | Existing independent artifact review and associated records; these comparisons are not newly recomputed in this revision. |
| **R1: trilemma exploration** | All 19 configurations remain rejected: 13 qualified breaches, five utility failures, and seven gain failures, with overlapping reasons. Four of 24 instruments qualify; missing coverage blocks a joint pass without erasing qualified breaches. Span-6 passes both utility criteria. Confirmation remains proposed and unexecuted; 26.4% is conditional planning. | Newly reconstructed 76 cell–seed contrasts, saved qualification/probe rules, intervals, and decision arithmetic, checked against the archived seed/cell tables. |
| **P4: defense pilot** | No eligible cell; two utility failures, seven gain passes, four failed and three undetermined composite privacy outcomes. Qualified signals well above a gate do not establish instrument breakage. Positive-control floors, null bands, and acceptance thresholds remain distinct; the untraced 6% floor is removed. | Newly recomputed paired-loss and saved-score arithmetic from 66 run-metrics records and 126 candidate score records; no tensor rescoring or probe fitting. |
| **E8: orthogonal wrapper** | Shared-key restoration concerns the wire observer, not confidentiality from a host that holds the key. Pooled 3.71% is not a worst-session bound; an individual session reaches 5.73%. At width 5,120, depth-specific absolute-50% crossings are 258.32, 355.81, and 617.93 pairs, not a universal 0.09H law. | Extraction and checks of the existing dense-K artifact; no new attack curve or model run. |
| **ER: deployment/rotation** | Per-session rotation and per-epoch ratcheting are separated. Later admitted supporting evidence supersedes an older blanket-invalid narrative, while retaining 20/24 identity restoration and four runs with known limitations. The recorded 2.8 versus 7.6 tokens/s anchor implies a 63.16% throughput reduction, not zero cost. | Extraction of eight admitted runs and 24 sessions, with arithmetic on recorded timing values; no newly measured paired runtime estimate. |
| **MI and capacity** | Near-zero lower-bound scores do not certify small leakage. Similar finite ablations do not prove an information-theoretic capacity ceiling. | Source interpretation plus an analytic counterexample: a zero constant-critic objective can coexist with eight bits of true MI when a uniform 256-symbol input is fully revealed. This is not a new model experiment. |

## Evidence and reproducibility limits

The secondary-analysis manifest hashes **118 input files**, and the stored run reports **1,522 internal arithmetic and disposition checks**. These establish consistency and identify the files consumed. They are not 1,522 independent experiments, a verification of all historical capture collection, or a measurement of the audit's diagnostic accuracy.

No new training, capture collection, GPU campaign, or stronger attacker evaluation was performed. The script uses saved records rather than executing archived training/attacker implementations. R1 retains descriptive two-sided 95% Student-t intervals across four training seeds; Paper 4 retains a one-sided 95% upper utility limit across three paired seeds. The campaigns' correlated tokens, windows, shared controls, and related implementations do not create extra independent replications.

This is one research program selected retrospectively after its outcomes were known. There is no independent, blinded, or confirmatory validation cohort. The revision does not estimate field-wide defect prevalence or establish that the audit reliably diagnoses unseen studies. Hash bindings also do not establish public accessibility of the complete primary collection; the manuscript makes no unsupported public-release claim.

## Publication judgment

The revised paper is worth developing and considering for submission as a restrained artifact-backed audit, evaluation-methodology, or research-program case-study paper. Its strongest result is a reproducible account of why some claims must be narrowed while genuine negative results survive. The existing evidence is sufficient to produce this revision; a new experiment campaign is not a prerequisite for completing the stated retrospective analysis.

That judgment concerns the substance and defensibility of this draft, not predicted venue acceptance. A journal or conference will still assess whether the cross-case reconciliation adds enough beyond the component papers and the authors' own attack-suite work. The single-program retrospective selection, substantial methodological overlap, and absence of independent validation constrain that publication case. If the intended contribution becomes a general diagnostic method, a new defense, or a security guarantee, those stronger claims require separate evidence. They are not claimed here.

The revision is a concrete manuscript for coauthor and external scientific review. It does not turn draft correction, internal consistency, or improved presentation into proof of novelty, generalization, or acceptance.

## Editorial update, 7 November 2026

The manuscript now follows the pinned `arxiv.sty` community preprint template.
Descriptive experiment names replace the internal numbered-workload shorthand.
The rewrite removes rhetorical questions, prose colons and semicolons, and
repetitive qualifications while retaining the scientific values and scope.
The equations retain their meaning. Mutual information is written as
`MI(X,Z)` to avoid a semicolon in the notation, and the shifted-token alignment
is stated as the sequences `(x_1,...,x_{T-1})` and `(x_2,...,x_T)`.

The former version is preserved in `review/pre-edit-2026-11-07/`. The current
scientific and editorial check is `review/editorial-verification-2026-11-07.md`.

## Attached-paper style correction, 7 November 2026

The latest formatting follows the supplied `arXiv-2609.04382v1.tar`, replacing
the community preprint style used in the previous editorial update. The current
manuscript uses its 11-point Times article layout, one-inch margins, standard
bold title and author blocks, empty date, float spacing and dark-blue linked
numeric citations. BibTeX's `plain` style generates the alphabetically ordered,
numbered bibliography from the same eight cited records. The bibliography now
follows the appendices, as in the attachment.

The earlier punctuation request is retained through a period after caption
numbers. The scientific text changes only where the two textual author-year
citations become explicit author names followed by numeric references. Values,
equations, source levels, evidence and the original manuscript remain intact.
The previous PDF and sources are saved in `review/pre-attached-style-2026-11-07/`.


## Peer-review positioning, literature expansion and human-cadence rewrite, 11 November 2026

Following the adversarial peer-review and stylometric audit, the manuscript incorporates three substantive improvements while preserving every verified numerical invariant, confidence convention, equation, and pinned file digest:

1. **Cross-campaign admissibility summary (`Table 1`):** Section 2 now includes a unified summary table mapping all six audited investigations (decoy gradients, boundary replacement, 19-cell exploration, seven-cell defence pilot, orthogonal wire wrapper and ratcheting, and variational mutual-information bounds) from their evaluated observation surfaces and initial readings to their audited dispositions and retained quantitative bases.
2. **Expanded verified bibliography (26 references):** `refs.bib` expands from 8 to 26 verified records covering foundational split learning (`1812.00564`), the 2026 split-LLM fine-tuning survey (`2604.24468`), modern gradient and internal-state reconstruction (`2206.07284`, `2405.15586`, `2604.06297`, `2507.16372`, `2609.30258`), evaluator and layer-wise vulnerability studies (`2210.10880`, `2610.07677`, `2610.32079`, `2605.02255`, `2605.23158`, `2607.24556`), split-LLM obfuscation, masking, permutation breaks, and MI regularisers (`2609.09794`, `2606.16801`, `2609.38339`, `2505.18332`, `2606.11592`, `2606.14210`), and formal MI / empirical DP auditing bounds (`1811.04251`, `1910.06222`, `2006.07709`).
3. **Natural scholarly cadence and anti-detector pass:** Every section, paragraph, and float caption has been rewritten for sharp sentence-length contrast (combining crisp 5-to-9-word anchors with medium analytical clauses and multi-clause technical sentences), varied grammatical subjects, consolidated scope limitations in Section 6, and zero trailing participial `-ing` clauses, repetitive `"X, not Y"` disclaimers, em-dashes, colons, semicolons, or rhetorical questions. The pre-rewrite Nov 7 state is archived in `review/pre-humanize-2026-11-11/`.
