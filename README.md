# Separating Measurement Failures from Defence Failures in Split Language-Model Privacy

**Georgios Politis and Evangelos Pappas** (Setloop.io)

Companion manuscript, secondary reanalysis scripts, derived tables, and archived
experimental evidence for our retrospective evidentiary study of split
language-model privacy evaluations.

- **Compiled manuscript:** [paper/main.pdf](paper/main.pdf)
- **arXiv source bundle:** [paper/arxiv-source.zip](paper/arxiv-source.zip)
- **Reanalysis script:** [analysis/recompute.py](analysis/recompute.py) (Python standard library only)
- **SHA-256 input manifest:** [data/source_manifest.json](data/source_manifest.json) (122 pinned inputs, 1,522 internal consistency checks)

## Overview

Split-training privacy evaluations fail for distinct reasons. Omitted
client-returned gradient channels, off-by-one target alignments, and
uncalibrated probes can produce false assurances before a defence is ever
properly tested, while calibrated attacks can also refute real candidate
defences. Separating these failure modes changes which conclusions survive:

1. **Decoy-gradient experiment (`evidence/decoy-gradient/`):** Omitting the
   client-returned backward gradient channel gave a passing forward check while
   zero gradients exposed the exact real/decoy partition on 4,096/4,096 frames
   per run across nine seeds (+0.65 to +1.50 pp token-probe gain). Per-row
   gradient clipping and Gaussian noise eliminated the scored partition leak on
   all 1,024 frames per run across three mitigation seeds at roughly 0.01 nats
   of held-out cross-entropy cost.
2. **Boundary-replacement review (`evidence/boundary-replacement/`):** Hidden
   states for $(x_1,\ldots,x_{T-1})$ were scored against shifted labels
   $(x_2,\ldots,x_T)$, invalidating current-token reconstruction claims. Six
   clean zero-return controls outperformed the split-trained remote model, and
   on one reviewed checkpoint held-out NLL rose by roughly 0.034 nats while
   teacher-forced answer-token top-1 match dropped 16.64 pp (34.79% to 18.15%).
3. **19-cell trilemma exploration (`evidence/trilemma-campaign/`, `reproduction/trilemma-campaign/`):**
   Only 4 of 24 required probes qualified, yet those four calibrated gradient
   byte probes still established qualified privacy breaches in 13
   configurations; independent utility (5) or learning-gain (7) failures
   eliminated the remaining 6 configurations. Exact floating-point checking
   confirms `span-6` passed both utility criteria (mean 0.099971 nats, upper
   95% Student-$t$ limit 0.119502 nats) and failed on gain and privacy.
4. **Seven-cell defence-redesign pilot (`evidence/defense-redesign-pilot/`, `reproduction/defense-redesign-pilot/`):**
   Two quantised configurations failed utility, all seven passed causal remote
   gain, calibrated next-byte structure probes breached the 0.02 gate on all
   seven cells (0.097–0.112, 21/21 bindings qualified), and qualified
   punctuation-property probes reached 0.79–0.89 ($\approx 40\times$ the 0.02
   gate), producing four failed and three undetermined composite verdicts.
5. **Orthogonal wrapper and per-epoch ratcheting (`evidence/structural-e8/`, `reproduction/structural-e8/`):**
   Shared-key orthogonal rotation protects passive wire captures (3.71% pooled
   depth-1 wire score; 5.73% max constituent session; 0% rotated wire recovery
   in deployed ratcheting) without hiding activations from a compromised
   restoring host that holds or derives $Q$ in RAM. Rapid `N=16` ratcheting cut
   zero-RTT session-1 throughput by 63.16% (7.6 to 2.8 tokens/s; 20/24
   greedy-decode identity).
6. **Variational mutual-information bounds (`evidence/privacy-contract/`):**
   Because the Donsker--Varadhan objective satisfies $L_T \le \operatorname{MI}(X,Z)$,
   near-zero MINE readouts bound leakage only from below and cannot certify an
   upper leakage ceiling.

## Repository layout

| Directory | Contents |
| --- | --- |
| `paper/` | LaTeX manuscript (`main.tex`, `sections/`, `refs.bib`, `main.bbl`, `figs/`), `build.py`, compiled `main.pdf`, and self-contained `arxiv-source.zip` |
| `analysis/` | Standalone secondary reanalysis script (`recompute.py`) and documentation |
| `data/` | Recomputed JSON/CSV outputs (`reanalysis_results.json`, `r1_cells.csv`, `p4_cells.csv`, `results_macros.tex`) and SHA-256 provenance manifests (`source_manifest.json`, `source-index.csv`) |
| `evidence/` | Curated decision documents, ledgers, review verdicts, and summary CSVs for all investigations (`EVIDENCE_MAP.md`) |
| `reproduction/` | Preserved primary experimental archives and source records backing the recomputation (`reproduction/README.md`) |

## Reproduce checks and rebuild the paper

Run from the repository root:

```bash
# Recompute all 1,522 consistency checks and verify all 122 input SHA-256 digests
python3 analysis/recompute.py

# Verify pinned inputs, run reanalysis, compile paper/main.pdf, and smoke-test paper/arxiv-source.zip
python3 paper/build.py

# Check pinned inputs only, or compile using existing tables/figures
python3 paper/build.py --check-only
python3 paper/build.py --compile-only
```

To regenerate the vector figures from `data/r1_cells.csv` and `data/p4_cells.csv`:

```bash
uv run --no-project paper/figs/make_figures.py
```

## Scope and limitations

This repository performs a retrospective, read-only secondary reanalysis of
existing archived evidence from one research programme. The 1,522 consistency
checks verify internal arithmetic and decision-rule reproducibility across 122
SHA-256-pinned inputs; they do not retrain victim models, refit attackers, or
constitute a prospective validation cohort on external papers.
