# Crossed span×start-block design — Mode 1 / Mode 2 feasibility map

**Date:** 2026-09-16
**Protocol:** `experiments/crossed_span_protocol.p6.json` (`p6-crossed-span-1.0`, draft)
**Origin:** WS2 of `docs/open-commitments-plan-2026-09-16.md`; Paper A Table 5's P5 sweep is a 7-point diagonal in (span, start-block) space, so span, starting location, and replaced capacity co-vary.

This document maps every protocol element to (a) what the **external P0–P5 programme** must provide (Mode 1 — the design as Paper A ran P5), and (b) the **closest executable analogue in this repository** (Mode 2), and lists explicitly where Mode 2 cannot reproduce the P5 semantics. Nothing here is executed; this is the feasibility basis for the freeze decision.

---

## 1. The two modes

- **Mode 1 — external surrogate stack.** P5's actual machinery: *frozen-backbone surrogate distillation under a protected boundary* (Paper A §2). The pretrained backbone stays frozen; a trained surrogate segment of `span` blocks starting at block `start` **replaces** backbone blocks `[start, start+span)`; the surrogate interfaces at boundary width 256; controls are matched all-local and removed runs. **None of this exists in this repository** (verified: no P0–P5 artifacts, protocols, or run scripts here; the paper's referenced paths `docs/P5_DELEGATION_TRIANGLE_DECISION.md`, `outputs/delegation/…` are external records — WS1 exists to export them).
- **Mode 2 — this repo's machinery.** `attacker/splitbench.py` split finetune with the new `quant` defense, return-path hook, and `--trainable-range` (added for this protocol; see §4), attacked by the repo's seam batteries (`seam-inversion`, `label-scoring`) under `attacker/gate.py` (`frozen-gate.v2`). Qwen3-0.6B is supported by splitbench (the `qwen3` arch key reuses the verified Mistral-family splitter).

**Coverage rule (also in the protocol):** only Mode 1 gate outcomes can re-scope Paper A's empty-set claim (the pre-committed text rule). Mode 2 answers the same *statistical* question — does start explain residual variance in gate measures after conditioning on span — on local instruments, and serves as the executable rehearsal of the crossed analysis. Mode 2 cannot pass or fail the P5 gates.

---

## 2. Element-by-element map

### 2.1 The "span @ start" cell (the core mapping)

| | Mode 1 (external P5) | Mode 2 (this repo) |
|---|---|---|
| Semantics | Surrogate segment **replaces** backbone blocks `[start, start+span)`; backbone frozen; surrogate trained by distillation at the boundary | Backbone blocks `[start, start+span)` **trained in place**; every other parameter frozen; plain split finetune through the seam |
| Executable form | external programme's runner (not in this repo) | `attacker.splitbench --split START --trainable-range START,START+SPAN` |
| Capacity effect | surrogate has its *own* (reduced) capacity — the replaced region's capacity genuinely changes | the segment is the backbone's own full-capacity blocks; nothing is removed, only the *trainable set* is restricted |
| Δ_U reading | utility cost of *replacing* capacity with a distilled surrogate | utility cost of *freezing* everything except the segment, relative to full finetune |

**Gap (fundamental):** replacement ≠ in-place training. Mode 2's Δ_U is `L(segment-trained) − L(full split-finetune)`; Mode 1's is `L(trained surrogate) − L(all_local)`. The two penalties share a sign convention and a nats unit, not a mechanism. Whether Mode 1's `all_local` is the frozen pretrained model or a locally fully-trained model must be pinned from the imported P5 spec at freeze — Table 5's `Δ_U` values (0.025–0.134 nats at D0) do not say.

### 2.2 Boundary width 256

| | Mode 1 | Mode 2 |
|---|---|---|
| Semantics | surrogate interface width (P2 primary cell: "span 4 starting at block 2, boundary width 256") | **does not exist** — the splitbench seam is the model's full hidden width (768 for GPT-2) |
| Closest local knobs | — | bottleneck *defenses* (`mid-lite --bottleneck-dim`, `trained-bottleneck --tb-dim`, `pca-compress --pca-dim`) — but those are wire defenses, not a surrogate interface, and are **not used** in this protocol |

**Gap:** no boundary-width axis in Mode 2; width is fixed at the architecture's hidden size.

### 2.3 Defenses D0 / D1

| | Mode 1 | Mode 2 |
|---|---|---|
| D0 | 8-bit scalar quantization at the boundary | `--defense quant --quant-bits 8 --quant-grad` |
| D1 | 4-bit quantization + gradient clip 1.0 + noise σ=0.1 | `--defense quant --quant-bits 4 --quant-grad --grad-clip 1.0 --grad-noise 0.1` |

Mode 2 conventions (predeclared in the protocol): signed symmetric per-sequence absmax quantization — the same rule as `attacker/validation_controls.py`'s `bits_N` gradient condition, applied to the forward seam activation and (under `--quant-grad`) to the returned gradient; clip = global L2 norm of the returned gradient; noise = relative-RMS (per-sequence `RMS(g) · σ`, the `noise_σ` convention); frozen composition order **clip → noise → quantize**; the bwd capture records the post-hook gradient (what the victim receives — the return-path mirror of the established post-defense forward doctrine).

**Gaps:** (a) whether the external D0/D1 quantizes the forward latent only or both directions is not stated in Paper A's text — the local choice (both, via `--quant-grad`) is an interpretation; (b) whether the external noise σ is absolute or relative-RMS is unstated — local is relative-RMS; (c) the external composition order is unstated. All three are freeze-time import items from the P5 spec.

### 2.4 Controls

| Control | Mode 1 | Mode 2 |
|---|---|---|
| all_local | matched all-local execution per cell | full split finetune, same seam/seed/budget, `--defense none`, no `--trainable-range` — 1 per seed (8 total) |
| removed | segment removed outright (ablation) | **no analogue** — removing real backbone blocks breaks the network; not faked |
| frozen (G_R) | frozen-**random** surrogate control (`G_R = L(frozen_random) − L(trained)`, P2) | held-out loss of the **untouched pretrained** model, 1 per seed — the local frozen control measures the effect of segment fine-tuning over the pretrained baseline, *not* over a random surrogate |

**Gap:** the removed control and the random-init surrogate control exist only in Mode 1.

### 2.5 Privacy instrument

| | Mode 1 | Mode 2 |
|---|---|---|
| Families | token-inversion (P0), sequence-property-v2, structure-semantic-v2 (P4b), over forward latents and returned gradients | seam-inversion `vmi` arm (forward latents) + label-scoring (property from seam gradients), training-mode captures |
| Statistic | r-advantage = R²_attack − R²_control vs the qualified label-free control; **gate: worst per-seed advantage ≤ 0.02** | token accuracy / property accuracy excess over the majority control, Bonferroni-Wilson upper-95 under `frozen-gate.v2`; **the 0.02 threshold does not apply** |
| Role in P6 | gate measure + variance-decomposition endpoint | variance-decomposition endpoint only |

**Gap:** different instrument, units, and threshold. Mode 2 privacy outcomes feed only the "does start explain residual variance" question; they never enter the empty-set claim. The P0 property label set and decoder baselines are external (WS1 export).

### 2.6 Delegation (analytic remote FLOP share)

| | Mode 1 | Mode 2 |
|---|---|---|
| Formula | external analytic model: Table 5 shares 0.123 … 0.776 over the diagonal — position-dependent in a way homogeneous backbone blocks would not be (per-span increments are irregular), so the surrogate geometry enters; import at freeze | declared local convention: with homogeneous blocks and fwd:bwd = 1:2 FLOPs cancelling, `share = span / n_layer` — **start-independent by construction** |
| Gate | ≥ 0.15 | ≥ 0.15 ↔ `span ≥ 1.8` for GPT-2 (n=12): all grid spans {2,4,6} pass |

**Gap:** the delegation measure's start-dependence is **untestable in Mode 2** (locally the share is a pure span check). The variance decomposition for FLOP share is meaningful only in Mode 1.

### 2.7 Corpora, architectures, seeds, statistics

| | Mode 1 | Mode 2 |
|---|---|---|
| Corpora | WikiText-2 **and PTB** (all P1–P5 phases) | pinned WikiText-2 only (v3 manifest, hash-pinned in the protocol); PTB is out of scope |
| Architectures | GPT-2 124M (rev `607a30d7`), Qwen3-0.6B | same two; GPT-2 first, Qwen3-0.6B second per protocol |
| Seeds | 8 victim seeds [11..88] | identical; held-out pair {77, 88} predeclared for the secondary endpoint |
| Statistics | the protocol's two-way decomposition, crossed bootstrap, Bonferroni families | fully executable (pure python); identical analysis code path intended |

---

## 3. What only Mode 1 can do (the honest list)

1. **Surrogate replacement** — train a distilled surrogate that *replaces* backbone blocks `[start, start+span)`. Mode 2 trains the real blocks in place.
2. **Boundary width 256** (and any width axis) — the local seam is the full hidden width.
3. **The removed control** — ablating the segment outright.
4. **The P0/P4b privacy instruments and the 0.02 r-advantage gate** — sequence-property-v2, structure-semantic-v2, the qualified label-free controls, the property label set.
5. **The frozen-*random* surrogate control behind G_R** — locally only the frozen *pretrained* control exists.
6. **PTB corpus cells** — not pinned locally.
7. **The three-gate empty-set re-scope** — the pre-committed text rule binds Mode 1 outcomes only. Mode 2's utility/delegation formulas match but its privacy instrument differs, so no local cell can "pass all three gates" in the Paper A sense.
8. **Position-dependent FLOP shares** — Table 5's share model is external; locally the share is span-proportional only.

Housekeeping for the import (resolved 2026-09-20): the P5 protocol identifier is
`p5-delegation-sweep-1.0`. That is the only form recorded in a frozen artifact —
`experiments/crossed_span_protocol.p6.json` (`status: frozen`) uses it in both its
`origin` and its Mode-1 `execution` field. The short form `p5-delegation-1.0` appears
only in prose, never in a protocol file. Note that the authoritative external P5 spec
(`docs/P5_DELEGATION_TRIANGLE_DECISION.md`, `outputs/delegation/delegation_summary.md`)
is **not present in this checkout**, so this reconciliation is against the frozen P6
protocol rather than against the P5 spec itself; confirm against the exported spec if
it becomes available. Paper A still disagrees with itself (`p5-delegation-sweep-1.0` in
§5.2 and the Table 5 caption, `p5-delegation-1.0` in the Table 3 evidence-ledger row);
that is tracked as a manuscript fix and is not corrected here.

---

## 4. Repo additions that make Mode 2 executable (landed with this protocol)

All in `attacker/splitbench.py` (torch-free import preserved; `--self-test` passes with torch blocked), with tests in `tests/test_quant_defense.py` (41 tests) and updated registry assertions in `tests/test_trained_defenses.py` (12→13 defenses):

1. **`quant` seam defense** (`--defense quant --quant-bits {2..16}`, default 8): signed per-sequence absmax quantization of the seam activation (`_quantize_absmax`, mirroring `validation_controls` `bits_N`). 13th entry in `_DEFENSES`.
2. **Return-path defense hook** (finetune): `--grad-clip` (global L2, 0=off), `--grad-noise` (relative-RMS, 0=off) composable with any defense; `--quant-grad` (quant only) quantizes the returned gradient; frozen order clip → noise → quantize; **bwd capture records the post-hook gradient**. All-zero defaults are bit-identical to the historic path (tested).
3. **`--trainable-range START,END`**: freezes all parameters except blocks `[START, END)` (frozen-backbone segment training — the local span@start parameterization); bounds-checked against the model at run time; mutually exclusive with `--lora-r`; finetune-only.

Per-cell command template (GPT-2, cell span `s` @ start `b`, seed `σ`, D1 shown; D0 drops `--grad-*` and uses `--quant-bits 8`):

```
python -m attacker.splitbench --model gpt2 --split $b --mode finetune \
  --steps 128 --seq-len 32 --batch-size 1 --lr 5e-5 --seed $σ \
  --defense quant --quant-bits 4 --quant-grad --grad-clip 1.0 --grad-noise 0.1 \
  --trainable-range $b,$((b+s)) \
  --corpus-file <pinned wikitext-2 slice> --capture-dir caps/ --output run.json --evidence
```

all_local control: same, `--defense none`, no `--trainable-range`, no `--quant-grad`. Frozen control: held-out loss of the untouched pretrained model (driver-side utility eval, no training). Attacks per cell: `python -m attacker --mode training --attack seam-inversion --arm vmi …` and `--attack label-scoring …` on the cell's capture dir (the `run_long_horizon.py` pattern). The sharded driver itself is execution infrastructure and is not part of this protocol.

---

## 5. Compute budget (measured anchors)

Measured anchors in this repo:

- `results/long-horizon-v1/matrix/run-index.json` (this host's CPU, GPT-2 @6, seq_len 32, batch 1): **33.2 s per 100-step finetune capture** (`none`; 34.9 gaussian, 36.0 sparsify, 33.6 mid-lite; at-lite outlier 368 s — EMA-PCA quirk, not used here); attacks **2.4 s/cell** mean.
- `outputs/validation-v3-review-2026-09-15/cell_summary.csv`: v3 online attack cells mean **378.1 s** on the GB10 GPU hosts (336.1 h over 3,200 non-reused cells) — the measured GPU anchor for attack-scale cells; victim-training GPU wall clocks are **not** in local records.

Mode 2, GPT-2 (192 cells + 8 all_local + 8 frozen evals):

- per cell ≈ 45 s capture (128 steps, scaled from 33.2 s/100) + ~5 s attacks + I/O ≈ **~60 s** → grid ≈ **3.5–4 CPU-hours**; with 50% planning contingency ≈ **6 CPU-hours**.
- **3 CPU nodes (soula, voula, sasa):** ≈ 1.5–2 h each at the measuring host's speed class. **1 GPU node (koula):** victim-train throughput unmeasured locally — a **3-cell canary per node type at freeze** pins node factors before the full grid is scheduled (fail-closed: no full-grid scheduling without canary factors).

Mode 2, Qwen3-0.6B: ≈ 4.9× GPT-2 compute/step (0.6B vs 124M params) → **~20 CPU-hours (~30 with contingency)**, same canary rule. Mode 1: not estimable from local records; budgeted as comparable to P5's 288 runs (~1 week fleet, per the WS2 plan) at freeze.

Exclusions: model downloads, corpus prep, analysis, driver development. Not a guaranteed bound.
