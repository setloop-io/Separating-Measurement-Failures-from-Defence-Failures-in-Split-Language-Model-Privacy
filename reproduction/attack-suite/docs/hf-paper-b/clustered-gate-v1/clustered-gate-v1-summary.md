# clustered-gate-v1: cluster-aware paired re-analysis of the frozen gate

Reviewer points answered with data (no new attack runs): (a) tokens within a document are not independent binomial trials -> one cluster per evaluation document, documents resampled with replacement within each seed; (b) the majority control is itself an estimate -> the excess is recomputed as a PAIRED per-document difference, so the control's own sampling error is resampled jointly with the attack's. Per-document control correctness is the attack's own recorded `control_correct` (no reconstruction assumption; pooled counts reproduce every artifact's published recovery/control exactly, asserted at load).

Gate: frozen-gate.v2 (z=2.5392, threshold +1.0 pp; control = label_free_majority). Cluster bootstrap: 100000 replicates, pinned seed 93281, two-sided 95% percentile interval. Clustered verdict analog: PASS iff interval upper <= +1.0 pp.

## domain-shift-v1 (GPT-2 @6/12, vmi arm)

| cell | docs (seed x seeds) | Wilson excess pp (published) | Wilson verdict | paired excess pp | clustered 95% CI pp | per-seed excess range pp | contains +1.0? | verdict flips? |
|---|---|---|---|---|---|---|---|---|
| code training/none | 8x3 | +52.90 | FAIL | +48.83 | [+43.49, +54.17] | [+44.53, +52.34] | no | no |
| code inference/none | 32x3 | +46.78 | FAIL | +44.63 | [+42.42, +46.88] | [+43.16, +46.68] | no | no |
| code training/gaussian | 8x3 | +52.77 | FAIL | +48.70 | [+43.36, +54.04] | [+46.48, +51.17] | no | no |
| code inference/gaussian | 32x3 | +47.52 | FAIL | +45.38 | [+42.97, +47.79] | [+44.82, +46.19] | no | no |
| clinical training/none | 8x3 | +93.75 | FAIL | +93.75 | [+93.75, +93.75] | [+93.75, +93.75] | no | no |
| clinical inference/none | 32x3 | +93.75 | FAIL | +93.75 | [+93.75, +93.75] | [+93.75, +93.75] | no | no |
| clinical training/gaussian | 8x3 | +93.75 | FAIL | +93.75 | [+93.75, +93.75] | [+93.75, +93.75] | no | no |
| clinical inference/gaussian | 32x3 | +93.75 | FAIL | +93.75 | [+93.75, +93.75] | [+93.75, +93.75] | no | no |
| wikitext-reference training/none | 8x3 | +65.27 | FAIL | +61.07 | [+56.77, +65.36] | [+57.81, +65.62] | no | no |
| wikitext-reference inference/none | 32x3 | +62.79 | FAIL | +60.61 | [+58.24, +62.96] | [+58.79, +62.01] | no | no |
| wikitext-reference training/gaussian | 8x3 | +65.52 | FAIL | +61.33 | [+56.51, +66.02] | [+58.59, +66.02] | no | no |
| wikitext-reference inference/gaussian | 32x3 | +64.11 | FAIL | +61.95 | [+59.80, +64.13] | [+60.94, +63.96] | no | no |

## long-horizon-v1 (defense matrix, vmi arm)

| cell | docs (seed x seeds) | Wilson excess pp (published) | Wilson verdict | paired excess pp | clustered 95% CI pp | per-seed excess range pp | contains +1.0? | verdict flips? |
|---|---|---|---|---|---|---|---|---|
| none h20 | 20x3 | +63.02 | FAIL | +58.18 | [+56.30, +60.00] | [+56.88, +58.91] | no | no |
| none h100 | 100x3 | +58.89 | FAIL | +56.68 | [+55.72, +57.62] | [+55.09, +57.59] | no | no |
| gaussian-s15 h20 | 20x3 | +2.42 | FAIL | +0.89 | [+0.47, +1.35] | [+0.62, +1.25] | yes | SENSITIVE |
| gaussian-s15 h100 | 100x3 | +3.69 | FAIL | +2.84 | [+2.47, +3.23] | [+2.41, +3.25] | no | no |
| sparsify-10 h20 | 20x3 | +23.52 | FAIL | +19.27 | [+17.50, +21.04] | [+17.97, +21.09] | no | no |
| sparsify-10 h100 | 100x3 | +25.03 | FAIL | +23.08 | [+22.19, +23.98] | [+22.38, +23.78] | no | no |
| mid-lite-k64 h20 | 20x3 | +1.00 | PASS | +0.00 | [+0.00, +0.00] | [+0.00, +0.00] | no | no |
| mid-lite-k64 h100 | 100x3 | +0.20 | PASS | +0.00 | [+0.00, +0.00] | [+0.00, +0.00] | no | no |
| santext-30 h20 | 20x3 | +46.15 | FAIL | +41.15 | [+39.06, +43.18] | [+38.91, +43.28] | no | no |
| santext-30 h100 | 100x3 | +42.67 | FAIL | +40.45 | [+39.51, +41.39] | [+39.22, +41.47] | no | no |
| at-lite-e15 h20 | 20x3 | +63.17 | FAIL | +58.33 | [+56.41, +60.21] | [+56.88, +59.38] | no | no |
| at-lite-e15 h100 | 100x3 | +59.00 | FAIL | +56.79 | [+55.83, +57.74] | [+55.03, +57.81] | no | no |

## Verdict sensitivity

Cells whose clustered 95% interval contains the +1.0 pp threshold or whose verdict moves: 1 of 24.
- **long-horizon/gaussian-s15/h20**: clustered 95% interval contains the +1.0 pp threshold (Wilson +2.42 pp FAIL; clustered CI [0.4688, 1.3542] pp).

## Degenerate intervals (no cross-document variance)

- domain-shift/clinical/training/none: every document shows the identical paired outcome (+93.75 pp) -> degenerate interval [93.75, 93.75] pp; there is no cross-document variance to resample, and the verdict is unaffected.
- domain-shift/clinical/inference/none: every document shows the identical paired outcome (+93.75 pp) -> degenerate interval [93.75, 93.75] pp; there is no cross-document variance to resample, and the verdict is unaffected.
- domain-shift/clinical/training/gaussian: every document shows the identical paired outcome (+93.75 pp) -> degenerate interval [93.75, 93.75] pp; there is no cross-document variance to resample, and the verdict is unaffected.
- domain-shift/clinical/inference/gaussian: every document shows the identical paired outcome (+93.75 pp) -> degenerate interval [93.75, 93.75] pp; there is no cross-document variance to resample, and the verdict is unaffected.
- long-horizon/mid-lite-k64/h20: zero correct and zero control tokens in every document -> degenerate interval [0.0, 0.0] pp. A percentile bootstrap cannot manufacture events that never occurred; the frozen gate's PASS here is its statistical floor (Wilson upper at 0 events), and clustering does not move it.
- long-horizon/mid-lite-k64/h100: zero correct and zero control tokens in every document -> degenerate interval [0.0, 0.0] pp. A percentile bootstrap cannot manufacture events that never occurred; the frozen gate's PASS here is its statistical floor (Wilson upper at 0 events), and clustering does not move it.

Reconciliation: recomputed Wilson excesses match the published tables within 0.0049 pp on all 24 cells (tolerance 0.051 pp).

Provenance: `['experiments/analyze_clustered_gate.py', '--replicates', '100000']` @ 2026-09-16T10:33:48Z, git f4d1f133f3a8c42116101001bdb006a4c78d715c.
