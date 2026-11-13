# Paper 2 Working Package

## Start here

Read `PLAN.md` first. It defines the active thesis, controlled experiment program, and submission gate for **How Much of an LLM Can a Protected Boundary Replace?**

This is a working research package, not a submission-ready artifact. The paper studies an empirical frontier for a frozen-backbone remote surrogate. It is not presently a capacity theorem, a general LLM-training result, or a validated secure-delegation result.

## What is active

- `PLAN.md`: authoritative roadmap and claim discipline.
- `outputs/PAPER_SCIENTIFIC_REVIEW_2026-08-20.md`: scientific basis for the two-paper split.
- `docs/experiments/DELEGATION_FINDINGS_REPORT.md` and `DELEGATION_TRIANGLE_REASSESSED.md`: reassessment of the ceiling and delegation metrics.
- `docs/experiments/ISOLATION_ADAPTIVE_EXPERIMENT.md`: current architecture-frontier evidence, interpreted subject to the warning below.
- `docs/experiments/METRIC_VARIANCE.md` and `DESIGN_EFFECT_MEASUREMENT.md`: current uncertainty evidence.
- `paper-data/collected/diagnostic/deleg_60_40/` and `latent_isolation/`: current sweeps, long-run, and inference results.
- `privacy_runtime/`, `attacker/`, `split-training/`, and `bin/`: relevant implementation and runners.
- `PACKAGE_CONTENTS.txt`: exact inventory of this folder.

## Current result interpretation

- 42.2% is the present clean empirical point.
- 47.4% is marginal/unconfirmed and fails the larger held-out evaluation; it must not be called a clean pass.
- Current “delegated compute” values are analytic surrogate-compute estimates, not measurements of the fraction of real LLM training performed remotely.
- Cross-model differences motivate an architecture/depth hypothesis, but they do not identify depth causally.
- The delegation triangle is a hypothesis-generating visualization until every axis is measured on matched cells with uncertainty.

## Archival source material

`papers/arxiv-draft/` is the old combined manuscript, preserved only as source material. Its abstract, 47.4% headline, capacity-ceiling theorem language, and delegated-training-FLOP wording are superseded by `PLAN.md`; do not edit or submit it as the active Paper 2 manuscript.

Some retained experiment reports, especially `ISOLATION_ADAPTIVE_EXPERIMENT.md`, also predate the stronger held-out inference evaluation and therefore describe 47.4% more positively than is now justified. Treat raw JSON as evidence and `PLAN.md` as the current interpretation.

`REPRODUCTION_V2.md`, `EXPERIMENT_COMPARISON_MATRIX.md`, and `paper-data/DATA_MANIFEST.md` are repository-wide historical indexes. They deliberately mention experiments and paths not copied into this focused package. Use `PACKAGE_CONTENTS.txt` for the actual package inventory.

## Data policy

The package contains frontier/inference JSON and logs but intentionally excludes `.pt` attack bundles, which belong to Paper 1's privacy-instrument work. Privacy claims in this paper must be imported only after Paper 1 validates the relevant metric and adversary scope on the exact retained frontier cells.

## Immediate next action

Implement Phases 0 and 1 of `PLAN.md`: define measured delegation estimands, correct the NLL/perplexity terminology, and run matched trained-remote, frozen-random, zero-output, local-surrogate, and no-replacement controls.
