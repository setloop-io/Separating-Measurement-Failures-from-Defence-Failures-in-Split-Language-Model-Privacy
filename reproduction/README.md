# Reproduction Package — Paper 9

Original code, scripts, manifests, and result artifacts for every experiment
quoted in `paper/main.tex`, collected from the program's source repositories.
Each subtree keeps the layout of its source repo so existing docs and
manifest paths resolve unchanged.

## Layout

| Directory | Source repository | Contents |
|---|---|---|
| `decoy-gradient/` | `paper1_arxiv_submission` | `anc/` attack code, experiment scripts, frozen data, provenance; `output/` run outputs |
| `boundary-replacement/` | `Paper-2-…` | `attacker/`, `split-training/`, `privacy_runtime/` code + `outputs/` incl. the two independent reviews |
| `trilemma-campaign/` | `Paper 3 - Privacy-Constrained Remote` | `src/`, `experiments/` (frozen matrix `configs/r1_exploration.yaml`), `evidence/` (verified R1 bundle), `data/`, `outputs/` (original runs) |
| `defense-redesign-pilot/` | `Paper-4-Defense-Redesign` | `src/p4d`, `experiments/` (freeze record `p4-defense-pilot-1.0.json`), `evidence/`, `outputs/` (runs + scenarios) |
| `structural-e8/` | `Distributed Training` | `probes/`, `attacker/`, `privacy_runtime/`, `split-training/`, `split-inference/` code + `paper-data/` (collected artifacts, ledgers) |
| `privacy-contract/` | `Privacy-contract` | `attacker/`, `experiments/`, `paper-data/` (latent v14/v15), runtime code |
| `attack-suite/` | `LLM-Attacker` (LLM-Attack Suite v2.0) | `attacker/` (65 modules), `scripts/`, `experiments/`, `docs/`, `tests/` + `results/` (original result artifacts) |

## How to reproduce each quoted result

### Failure 1 — decoy-gradient channel (arXiv:2609.04382)
See `decoy-gradient/anc/README.md` for the ancillary package layout; the E1
protocol is documented in `../evidence/decoy-gradient/E1_GRADIENT_CHANNEL_2026-08-22.md`
and its reproduction in `E1_REPRODUCTION_W17_2026-08-22.md`. Frozen data and
attack code ship inside `anc/`.

### Failure 2 — boundary-replacement experiments
Harness fixes and re-run rules: source repo `REPRODUCTION_V2.md` (fixed
harness: corpus filter >500 chars, lr 1e-5, batched eval). The corrected
endpoint controls are under `boundary-replacement/outputs/phase08_clean_endpoints/`;
the independent recompute of the −16.64 pp answer-token drop is in
`outputs/independent_evidence_review_2026-09-09/recomputed_evidence.json`
(script inputs referenced from that bundle's VERDICTS.md).

### Failure 3 — trilemma campaign (R1)
Executable frozen matrix: `trilemma-campaign/experiments/configs/r1_exploration.yaml`.
Runbook: source repo `docs/R1_RUNBOOK.md`. Verified evidence bundle:
`trilemma-campaign/evidence/r1-exploration-20260917/README.md`. The 19-cell
verdicts and per-seed numbers are in `../evidence/trilemma-campaign/r1_cell_results.csv`
and `r1_seed_results.csv`. Note: pre-reset (pre-2026-09-10) observations are
preserved for audit but are inadmissible evidence per that repo's AGENTS.md.

### Failure 4 — defense-redesign pilot
Freeze record: `defense-redesign-pilot/experiments/freeze/p4-defense-pilot-1.0.json`
(66-run matrix, protocol p4-defense-pilot-1.0, pinned `gutenberg_p4` corpus).
Environment setup: source repo `docs/ENVIRONMENT.md`; architecture:
`docs/ARCHITECTURE.md`. Original runs: `outputs/runs/`; scenario sweeps:
`outputs/scenarios/`.

### Structural results — E8 boundary transform
Batteries and controls: `structural-e8/paper-data/collected/` (per-cell JSONs,
collection manifests) with the claim mapping in
`../evidence/structural-e8/EVIDENCE_EXECUTION_LEDGER.md`. Attack/probe code:
`probes/` and `attacker/`. Re-run discipline (seed/victim replication rules):
source repo `REPRODUCTION_V2.md`.

### Privacy contract (MINE budget, delegation triangle)
v14/v15 experiment data: `privacy-contract/paper-data/latent_v14/`,
`latent_v15/`; estimator code under `attacker/`; numbers summarized in
`../evidence/privacy-contract/KEY_NUMBERS.md`.

### Attack suite (13.71% → 37.77% document recovery)
The 13.71→37.77 tables come from the published arXiv:2610.04128 study; module
code in `attack-suite/attacker/`, original result artifacts in `results/`
(pilot and corrected batteries). The frozen measurement-validation package is
described in `attack-suite/docs/` (v2.0 release notes: archive-based runner,
123 pinned source hashes).

## Provenance notes

- Numbers were cross-checked against these artifacts on 2026-11-06; see
  `../evidence/EVIDENCE_MAP.md` for the claim→file mapping and corrections.
- Source repos keep their own `AGENTS.md` evidence rules; the copies here are
  byte-identical (rsync archive mode) so manifest hashes still verify.
