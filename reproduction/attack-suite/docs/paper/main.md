> # PARKED DRAFT — NOT A CURRENT PAPER, NOT PAPER A, NOT PAPER B
>
> **This file is an abandoned stage-1 skeleton of a separate, third paper**
> ("adversary-conditioned assurance for split-training privacy defenses").
> It is **not related to `paper/paper_a/` or `paper/paper_b/`**, it is not part
> of either submission, and nothing in it is current.
>
> **The `[placeholder]` markers below belong to this parked draft only.** They
> are the only literal `[placeholder]` markers in the repository. They do
> **not** indicate unfinished content in Paper A or Paper B, which are complete
> manuscripts with their own evidence index at `docs/evidence-release-v3.md`.
> This file has repeatedly been misread as evidence that the real papers
> contain placeholders. It is not.
>
> Retained for history only. Do not cite it, do not treat its numbers as
> current, and do not use it to assess the status of Paper A or Paper B.

# Paper skeleton — adversary-conditioned assurance for split-training privacy defenses

Status: **PARKED — skeleton / stage-1 assembly of a third, unrelated paper**
(see the banner above). Quantitative claims are pulled from
committed artifacts; every number below cites its artifact path and SHA-256.
Sections marked *[placeholder]* await the frozen v2 confirmatory study.

Artifact references used throughout (SHA-256):

| Artifact | SHA-256 |
|---|---|
| `experiments/assurance_matrix_v2.json` | `209917b1902b7686e6e00f19da723b6a4a38c390ae6b4269c9a97755f48e941f` |
| `experiments/assurance_matrix_pilot_v1.json` | `f59d2ce2e57f23368d7ed87431d68245a8e174b11799cecc395fe7bfd4fda9e2` |
| `experiments/bottleneck_sweep_selection_v1.json` | `76d2ff7f6e2c501fa2283d1ebf704e9cad530855d26459d7c5bb28a328f18aa7` |
| `results/pilot-defended-v1/analysis/analysis.json` | `e51320eeb89deecc0763fa31b44a3a919372f0d0eb6c8bd2254c81153a8190c8` |
| `results/bottleneck-sweep-v1/analysis/analysis.json` | `ec5993be03d648f4f81bbd47967ff946538fc3c8a2a06182cde6ce51bf84aeef` |
| `results/bottleneck-sweep-v1/analysis/selection.json` | `76d2ff7f6e2c501fa2283d1ebf704e9cad530855d26459d7c5bb28a328f18aa7` |

## Abstract

*[draft]* We present an attacker-centric framework for validating privacy
defenses in split (split-stitch) model training: instead of proposing a new
defense, we re-implement published defense families inside one harness,
subject them to a matched attacker under an explicit access taxonomy, and
report the result as an adversary-conditioned assurance matrix — defense ×
attacker access × measured leakage × utility cost × evidence strength. The
stance is impartial evaluation: the framework exists to measure defenses, not
to advocate for or against any of them, and it structurally refuses to emit a
universal aggregate risk score or maturity model. A defended development
pilot over four trained defenses and a strengthened-attacker development
sweep over the bottleneck family (with a separately labeled white-box tier)
are reported as per-row measurements with paired bootstrap confidence
intervals; the bottleneck operating-point selection was NEGATIVE at its
predefined utility gates.

## Threat model and attacker access taxonomy

From the repository protocol (README, "threat model" sections):

- The victim fine-tunes a split model; the attacker captures training-phase
  forward/backward wires at the split point.
- The corpus is deterministically split by document into public calibration,
  private victim training, and held-out utility partitions. Calibration
  examples are never selected from scoring truth; training-capture attacks
  evaluate private training examples (distinct from attacker calibration).
- The public head/embedding snapshot is frozen **before** victim training;
  attack arms are not granted the victim's final private head. Model mismatch
  across training steps is part of the measurement.
- Access conditions in the assurance matrix
  (`experiments/assurance_matrix_v2.json`, sha256 `209917b1…`, `meta.access_conditions`):
  - `split-training-wire` — matched wire capture, public calibration,
    public head only, query budgets matched across defenses.
  - `split-training-wire-whitebox` — additionally grants the TRUE defense
    checkpoint (encoder weights / PCA basis + provenance). Strictly stronger
    access; white-box recovery neither bounds nor is bounded by black-box
    recovery of the same wire.

## Defense re-implementations and the adapter contract

All defenses are **our re-implementations**, not the original defense
projects' code. They are exercised through a single adapter contract, the
`llm_attacker.defense_bundle.v1` envelope (`attacker/defense_bundles.py`),
which also accepts externally exported artifacts. Defended-pilot defenses:
`replica-median` (K in-process body replicas, coordinate-median aggregation,
deviation-flagging journal), `secret-mixup` (recycled real latent decoy rows,
CSPRNG per-block draws without replacement, decoy-forward return-gradient
correction), additive `gaussian`, and `trained-bottleneck` (MLP encoder
minimax-trained against embedded probes; the MINE-style MI penalty is logged
as a reading, not a certificate). Bottleneck-sweep family: `pca-compress`
(control), `tb-frozen`, `tb-cotrain` at dims {64, 128, 256} × splits {3, 6}.

## Method

- **Matched-access protocol.** Every defense in a cell faces the same
  attacker under the same access condition, identical starting weights, and
  paired document order per seed.
- **Paired bootstrap.** Defense-minus-none (pilot) or candidate-minus-control
  (sweep) per-seed differences with percentile bootstrap CIs (10,000
  replicates, seed 2026; `results/bottleneck-sweep-v1/analysis/analysis.json`,
  sha256 `ec5993be…`, `bootstrap` block).
- **Fail-closed reporting.** Rows with missing leakage or utility fields are
  emitted as `INCOMPLETE`; cells declared non-evaluable by the source
  analysis (e.g. `tb-cotrain` white-box, whose final co-trained checkpoint
  matches no captured wire) are emitted as `NOT_EVALUABLE` with the source
  reason. Nothing is dropped or imputed
  (`experiments/assurance_matrix_v2.json`, `meta.fail_closed_policy`).
- **Frozen corpus governance.** The sweep corpus (96 articles, sha256
  `d11d88a6…`) is identity-checked and disjoint from the frozen v2 validation
  manifest (`experiments/validation_corpus_manifest.json`);
  `results/bottleneck-sweep-v1/analysis/analysis.json`, `disjointness` block.
- **No aggregate score.** The matrix schema
  (`llm_attacker.assurance_matrix.v1`) refuses any combined/weighted scalar
  risk score or maturity rating at any depth.

## Results

### Defended development pilot (evidence: `defended_pilot_paired`)

Source: `experiments/assurance_matrix_pilot_v1.json`, sha256 `f59d2ce2…`
(built from `results/pilot-defended-v1/analysis/analysis.json`, sha256
`e51320ee…`). Forward recovery − none, percentage points [95% CI]; ΔPPL vs
none [95% CI]; 3 seeds.

| Defense | Split | Leakage (pp) | ΔPPL |
|---|---|---:|---:|
| gaussian | 3 | +0.72 [+0.20, +1.66] | +0.346 [−1.426, +1.874] |
| gaussian | 6 | +0.20 [−1.66, +1.76] | +0.077 [−1.221, +0.780] |
| replica-median | 3 | −66.7 [−67.68, −64.94] | +49.1 [+6.70, +97.3] |
| replica-median | 6 | −71.2 [−71.68, −70.51] | +45.3 [+7.19, +93.6] |
| secret-mixup | 3 | −5.79 [−6.74, −4.49] | +35.1 [+11.0, +68.0] |
| secret-mixup | 6 | −7.23 [−8.59, −5.66] | +36.3 [+16.0, +71.1] |
| trained-bottleneck | 3 | −91.6 [−92.09, −91.02] | +6.64e7 [3.84e7, 1.01e8] |
| trained-bottleneck | 6 | −87.6 [−88.57, −86.52] | +1.24e8 [1.69e7, 2.6e8] |

### Bottleneck-family sweep, strengthened attacker (evidence: `strengthened_attacker_development_sweep`)

Source: `experiments/assurance_matrix_v2.json`, sha256 `209917b1…` (built
from `results/bottleneck-sweep-v1/analysis/analysis.json`, sha256
`ec5993be…`, and `results/bottleneck-sweep-v1/analysis/selection.json`,
sha256 `76d2ff7f…`). Leakage is the **strongest observed recovery across the
measured arms** (base forward/SIPIT + defense-aware surrogate; white-box rows
reported separately) — a measured statistic, not a ceiling. 60/60 cells
COMPLETE, 3 seeds.

Headline base-tier rows (strongest observed, utility gate 1.5×/2.0×):

| Defense | Dim | Split | Strongest observed | ΔPPL [95% CI] | Gates |
|---|---:|---|---:|---:|---|
| pca-compress | 64 | 3 | 59.28 | +144.1 [130.1, 153.8] | FAIL/FAIL |
| pca-compress | 128 | 3 | 81.90 | +42.0 [39.6, 44.6] | FAIL/PASS |
| pca-compress | 256 | 3 | 89.36 | +15.5 [14.1, 16.3] | PASS/PASS |
| tb-frozen | 256 | 3 | 89.32 | +14.6 [12.4, 16.0] | PASS/PASS |
| tb-cotrain | 256 | 3 | 89.36 | +14.6 [12.4, 16.0] | PASS/PASS |
| pca-compress | 64 | 6 | 54.82 | +169.5 [156.0, 182.3] | FAIL/FAIL |
| pca-compress | 128 | 6 | 75.29 | +69.2 [65.2, 72.3] | FAIL/FAIL |
| pca-compress | 256 | 6 | 83.85 | +35.8 [24.5, 49.3] | FAIL/FAIL |

Full 46-row table (including all white-box rows and the 6 `NOT_EVALUABLE`
`tb-cotrain` white-box rows): `experiments/assurance_matrix_v2.md`.

**White-box tier.** Granting the true defense checkpoint did not
meaningfully change strongest-observed recovery at any cell where it was
evaluable (e.g. split-3 dim-64 pca-compress: 59.28 base vs 59.83 white-box;
split-6 dim-64 tb-frozen: 55.34 vs 55.53) — a stronger-access measurement,
not a bound (`results/bottleneck-sweep-v1/analysis/selection.json`,
`access_labels.whitebox_tier`).

**Selection outcome: NEGATIVE.** No candidate operating point was selected
at either split: every gated candidate failed the predefined dual utility
gates (1.5×/2.0× PPL ratio, no retuning) and/or failed to establish
strongest-observed-leakage improvement over the `pca-compress` control under
BOTH the bootstrap condition and the all-seeds-positive condition. v3
remains UNFROZEN (`experiments/bottleneck_sweep_selection_v1.json`, sha256
`76d2ff7f…`, `negative_statement`).

### BiSR narrowed comparison

BiSR remains a separate published-baseline gap: its SIP inversion model is
trained on a substantially larger auxiliary corpus than anything in our
matrix, and its white-box grant differs from our cells, so no frozen-matrix
cell cites BiSR numbers as comparable. A pilot-scale development-only
adaptation exists as the `bisr-baseline` attack module
(`attacker/attacks/bisr_baseline.py`, clearly labeled an adaptation, not a
faithful reproduction). Training its inversion model on this small
calibration sample would not constitute a strong reproduction; published
BiSR numbers must not be cited as comparable to any matrix cell.

## Limitations

- **3 seeds** per cell; development-tier evidence only.
- **Development corpus**: sweep results are conditional on a fixed 96-article
  sample, disjoint from but much smaller than the frozen v2 corpus.
- **Re-implementations, not original defense code**: all defenses are
  in-repo re-implementations; fixture-level positives do not falsify deployed
  defenses.
- **Surrogate-arm weakness**: the defense-aware arm fits a PCA surrogate
  (stand-in) inverse on public calibration; it is not a faithful decoder of
  the true defense transform, and its numbers are an approximation, not a
  certificate (`results/bottleneck-sweep-v1/analysis/selection.json`,
  `access_labels.surrogate_arm`).
- **White-box ≠ bound**: white-box recovery is a stronger-access measurement;
  it does not constrain black-box recovery of the same wire, nor the reverse.
- **No universal risk score / no maturity model**: the current evidence does
  not validate either; the matrix reports per-row measurements only.

## Frozen v2 confirmatory study

*[placeholder]* The frozen crossed-bootstrap confirmatory study under the v2
protocol (`experiments/validation_protocol.v2.json`, runbook
`experiments/frozen_run_v2_runbook.md`) is in flight as of this draft; ETA
~4.5–5 days from launch. Evidence counts at the `frozen_crossed_bootstrap`
level only once executed; this section will pull from the frozen study's
analysis artifact and will add its path + SHA-256 here at that time.
