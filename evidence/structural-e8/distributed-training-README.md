# Split Learning over Hostile WANs

Reproduction + extension of *Privacy-Aware Split Inference with Speculative Decoding
for LLMs over WANs* (arXiv:2602.16760), and the follow-on decentralized-training and
privacy research, on **2× NVIDIA DGX Spark (GB10)** nodes (toula = trusted/local,
koula = untrusted/cloud, direct ConnectX link + `tc netem` WAN emulation).

## What's inside

| Doc | Content |
|---|---|
| `REPORT.md` | Full experiment report (all phases, all numbers) |
| `SUMMARY.md` | Discussion: what reproduced, what's new, findings |
| `COMPARISON_MATRIX.md` | Every experiment in six matrices |
| `PAPER_DRAFT.md` | Paper 1 draft (foundations) |
| `PAPER2_PLAN.md` | Paper 2 plan (scaling at ≥20B: draft speculation, 1F1B, FLA, fp8) |
| `PAPER3_PLAN.md` | Paper 3 plan (systematic adversarial privacy analysis) |
| `PLAN.md` | Original reproduction plan + profiles |

## Headline results

- Paper reproduced: lookahead acceptance to 3 decimals; identity modulo 1-ULP fp16
  ties; robust at 458ms–1s RTT with loss and contention.
- First split inference + lookahead on hybrid linear-attention models (Qwen3.6-27B,
  35B-A3B MoE); lookahead is sub-break-even on hybrids; MoE beats dense 7B.
- Split fine-tuning over WAN, **bit-exact**; async overlap scheduling gives 6.7–9.3×
  exact speedups; FedAvg break-even K*≈1600 (naive transport).
- Privacy: trained attackers recover 55–80% of tokens; noise/quantization/LoRA are
  useless; private fine-tuning appears to defend only when it breaks the model
  (at utility-safe settings it changes nothing); **secret obfuscation (E8) gives 0.00% recovery at zero
  utility cost**, with measured oracle break-even (~H known pairs).

## Repo layout

- `split-inference/` — vendored upstream system (see `split-inference/UPSTREAM.md`)
  + our patches ([QWEN36] hybrid support, streamed loading, model-agnostic inversion)
- `split-training/` — new system: pipeline-split fine-tuning, overlap scheduler,
  FedAvg, all privacy attackers and defenses (`DESIGN.md`, `PROTOCOL.md` inside)
- `bin/` — all drivers + the `exp` orchestrator (see `bin/README.md`)
- `docker/` — container definition (`split-inference:spark`: torch 2.11+cu130,
  transformers 5.13, GB10/ARM64)
- `patches/` — upstreamable diffs + design docs
- `paper-data/` — raw JSONs (repo schema), publication figures, TABLES.md
- `probes/` — analysis probes (e.g., the fp16 near-tie margin probe)

## Quick start

```bash
# everything is orchestrated from the Mac/driver host over ssh:
ssh toula 'bin/exp list'                    # available experiments
ssh toula 'bin/exp smoke-7b'                # sanity check
ssh toula 'bin/exp sweep-7b'                # example full run
```

The dashboard on each node (`http://<node>:8090`) shows live Training + Experiments
panels during runs.

## Requirements / environment

Nodes: 2× DGX Spark (GB10, 121GB) + 2× H100 NVL (94GB, x86) for attack grids/scale runs; docker, the `split-inference:spark` image
(build from `docker/Dockerfile`), models under `~/experiments/models/`, sudoers rule
for `tc` (`bin/set_rtt.sh` / `bin/set_wan.sh` need it; see PLAN.md). All experiment
state lives under `~/experiments/results/` on toula and is synced to `paper-data/`.

## Live coverage trackers

- `COVERAGE.md` — Paper 1 methods × models (ALL×ALL audit)
- `COVERAGE_P2.md` — Paper 2 scaling experiments (checklist + headline criteria)
- `COVERAGE_P3.md` — Paper 3 attacker × defense × model frontier
