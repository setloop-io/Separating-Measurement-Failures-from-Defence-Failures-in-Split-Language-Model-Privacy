# v14 experiment — certified leakage-budget sweep

Part of the **Privacy-contract** paper (see `../../NEW_PAPER_BRIEF.md`).
Self-contained: this repo carries its own copy of everything the experiment
needs, so `Distributed Training` stays untouched.

## Layout (mirrors the dtraining runtime layout)

- `bin/run_latent_native_v5_06b.py` — v14-patched campaign runner (upstream:
  `Distributed Training/bin/run_latent_native_v5_06b.py` @ bfc398f). Adds:
  - `--mine-budget-bits B` — Lagrangian dual controller: lambda tracks the
    lightest MINE-penalty weight holding estimated I(z;x) <= B bits/token
  - `--mine-budget-lr` / `--mine-budget-ema` — controller tuning
  - `--mine-measure-only` — estimator trained, no penalty (frontier's
    unconstrained endpoint still gets a certified MI reading)
  - JSON fields `mine_budget_bits`, `mine_lambda_final`,
    `mine_mi_bits_held_out`, gate `mine_budget_satisfied`
- `bin/latent_v14_stage.sh` — cells v14.0-v14.4 (B = 0.25 / 0.50 / 1.00 /
  2.00 / unconstrained) on the v9.2 winner base (D=64, radial-MoE E8 L2,
  chaff-48, noise 0.35, rotation+permutation gauges, K=1, seed 42,
  2000 steps)
- `bin/latent_v14_byzantine_stage.sh` — K=3 `--byzantine-verify` integrity
  cell at B=0.50 on the CPU replicas
- `privacy_runtime/` — full copy of the runtime package @ bfc398f
  (latent_native, latent_protocol, mine, ratchet_v2, activation_dp, ...)
- `attacker/` — full copy of the frozen attacker suite @ bfc398f (17 arms)
- `split-training/` — only the two files the runner/server import
  (`split_trainer.py`, `latent_cloud_server.py`)

Verified import-clean with torch 2.13 CPU (`IMPORTS_OK, attacks: 17`).

## Nodes

- `toula` (GB10 GPU): training client, docker `split-inference:spark`;
  deployment `rsync -a bin privacy_runtime attacker split-training \
  toula:privacy-contract/`
- `koula` (GB10 GPU): latent cloud server, `wss://koula:5025` (D=64 radial)
- `soula`/`voula`/`sasa` (CPU, 192.168.1.120/121/122): Byzantine replicas,
  `latent_cloud_server.py` on port 5013 with `byz-<ip>` certs

Models/corpus/TLS stay on the nodes at `~/experiments/` (mounted into the
container as `/workspace/experiments`); only code ships with the paper.

## Artifacts

Cell JSONs land on toula at `~/experiments/results/training/latent_v14_*.json`
and are collected into `paper-data/latent_v14/` (this repo). They are **not**
committed to `Distributed Training/paper-data/`.
