# Phase-2 Fleet Runbook — GPU/CPU workload schedule

**Date:** 2026-09-16 · **Status:** ready to execute (Phase-1 engineering complete)
**Nodes:** `koula` (GPU; holds the v3 binary tier and a full shard copy), `soula`, `voula`, `sasa` (CPU).
**Ground rules (all jobs):** frozen protocol or named runbook entry before execution; `--evidence` on every run; fresh output directories only; fail-closed journaling; corpus disjointness checks fail-closed against `experiments/validation_corpus_manifest.json`; restart = fresh directory with the same protocol (v3 stopping rule), never outcome-based retries; every protocol freeze is a git commit anchor (precedent: `064a64f`).

---

## 0. Mandatory canary gate (before any scheduling)

Per `experiments/crossed_span_protocol.p6.json` and the BiSR Phase-B plan:

- [ ] **CPU canary:** 3 Mode-2 P6 cells (one per span) on ONE of soula/voula/sasa; record victim-train wall-clock; extrapolate before splitting the grid. Anchor: long-horizon CPU captures ran 33.2 s/100-step GPT-2@6 (seq 32) + ~2.4 s attack overhead.
- [ ] **GPU canary:** 1 BRE-`bf` cell (`--opt-arm bf`) on koula against a tiny GPT-2 seam; record wall-clock and peak memory. v3 GPU attack cells averaged 378.1 s.
- [ ] **Aux-corpus canary:** stream 100k windows through the BiSR aux pipeline (measured here: 281 windows/s end-to-end, 96.1 KiB/window) on the target node to confirm the disk ceiling (~1–2M windows overnight CPU).

Do not schedule B–D until the canaries are journaled.

## A. Protocol freezes (git-anchored, before execution)

- [ ] `experiments/crossed_span_protocol.p6.json`: flip `status: draft` → `frozen`, import Mode-1 parameters from the external P5 spec (B, D0/D1 application points, `all_local` definition, FLOP-share model), commit the freeze.
  - Naming reconciled 2026-09-20: the P5 protocol identifier is **`p5-delegation-sweep-1.0`**, the only form used in a frozen artifact. The short form `p5-delegation-1.0` appears in prose only. No edit is possible here anyway: this file now reads `status: frozen` and its SHA-256 `e404b2e6702c5de66b82fcb8d30e1dcb59b6f4f11117eaa82fa6babb1b9f07fb` is anchored as `historical_protocol_sha256` in both `experiments/crossed_span_protocol.p6.v2.json` and `experiments/crossed_span_protocol.p6.qwen3.v1.json`, so changing its bytes would invalidate two downstream frozen protocols.
- [ ] `experiments/validation_protocol.v4.json` (new amendment of v3): adds the `bisr-faithful` arms (depths {3,6} × {public, exact} × k32-matched budget) with the predeclared contrast family (BiSR vs `joint_true_k32` primary; vs `forward_k32` and SIPIT-128 secondary; 100k crossed bootstrap; Bonferroni over the family). v3 stays immutable. Commit the freeze.

## B. koula (GPU) job queue

| # | Job | Budget anchor | Output |
|---|---|---|---|
| B1 | **WS3 validation gate**: reproduce the published BiSR reference setting (SplitLLM `demo_bisr` gpt2-large/wikitext config; Apache-2.0, attribution recorded) and check our arms land within documented tolerance of the paper's numbers | hours | `results/bisr-fidelity-v1/` |
| B2 | **WS3 matched comparison** under the frozen v4 protocol (uses koula's v3 binary tier + shard copies) | ~10–30 GPU-h | `results/bisr-matched-v4/` |
| B3 | **P6 Mode-2 Qwen3-0.6B replication grid** (4.9× GPT-2 cost; ~20 CPU-h equivalent — GPU preferred) | ~1 GPU-day worst case | `results/p6-qwen3-v1/` |
| B4 | **Live-planner agentic re-run** `--llm-trials 2` on `cross-agent-deputy` + `agent-goal-hijack` (CPU cost was 4.5 min/cell; GPU much faster) | <1 h | `results/agentic-live-v2/` |
| B5 | *(optional)* **Deployment-campaign evidence collection**: locate the August fleet journals/manifests for the Qwen3/Mistral/GPT-2 headline cells on the host and package per `docs/evidence-release-v3.md` conventions | hours | `docs/evidence-release-deploy.md` |

## C. CPU node job queues (shard by span/task)

| Node | Jobs | Budget |
|---|---|---|
| soula | P6 Mode-2 GPT-2 grid, span **s=2** (64 cell-evals) · then item-11 sequence-length sweep (64/128 tokens) through the Track-A domain-shift harness | ~2 h + ~1 h |
| voula | P6 Mode-2 GPT-2 grid, span **s=4** (64 cell-evals) · then item-11 independent-domain corpus (dialogue/external code) | ~2 h + ~1 h |
| sasa | P6 Mode-2 GPT-2 grid, span **s=6** (64 cell-evals) · then the 500-step convergence probe on mid-lite k=64 | ~2 h + ~1.5 h |
| any free | Matched order-recovery pilot on a locally produced composite bundle (`--evidence`; the bundle emits `train_order`/`eval_order` via `attacker.defenses.ReferenceDefense.generate`) | minutes |

P6 Mode-2 command template (per cell; full matrix in `experiments/crossed_span_feasibility.md`):
```
python -m attacker.splitbench --model gpt2 --split START --trainable-range START,START+SPAN \
  --mode finetune --steps 100 --seq-len 32 --batch-size 1 --lr 5e-5 \
  --defense quant --quant-bits 8 --quant-grad            # D0
  # --defense quant --quant-bits 4 --quant-grad --grad-clip 1.0 --grad-noise 0.1   # D1
  --corpus-file <pinned pilot corpus> --capture-dir <fresh> --save-canonical \
  --output <cell>.json --evidence
```

## D. Explicitly NOT scheduled

- **WS1** (P0–P5 evidence release): blocked on the external programme's phase records; packaging only, no node time needed once records arrive.
- **P6 Mode-1**: requires the external surrogate stack; only Mode-1 outcomes bind the empty-set claim (protocol rule).
- **Item-11 repeated-session + adaptive tier**: design not yet frozen; draft protocols first (same discipline as p6).
- **v13-composite long-horizon**: external defence project.

## E. After-action (all queues)

1. Merge/verify artifacts with the collision-corrected merger (`experiments/merge_validation_shards.py`); run `verify_evidence` on every bundle.
2. Cluster-aware check (`experiments/analyze_clustered_gate.py`) on any new gate-relevant table before it enters a paper.
3. Analysis + paper updates only after the full test suite is green; each landing gets its own commit with the evidence pointers.
