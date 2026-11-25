# Aggregated Synthesis — Privacy of Distributed (Split) LLM Training

*Generated 2026-11-05 from a full read of: `Distributed Training/`, `Paper-01/`, `Paper-1-When-the-Privacy-Probe-Passes-but-the-Transcript-Leaks/`, `Paper-2-How-Much-of-an-LLM-Can-a-Protected-Boundary-Replace/`, `Paper 3 - Privacy-Constrained Remote/`, `Paper-4-Defense-Redesign/`, `Privacy-contract/`, `paper1_arxiv_submission/`, `paper-llm-attack-suite/`, `LLM-Attacker*/`, `r1-*` bundles.*

---

## 1. The program on one page

One research question runs through everything: **can you train an LLM across a split whose remote side is a fully compromised cloud, and can you *prove* what leaks?**

| Artifact | Status | Core result |
|---|---|---|
| **Paper 1** — "The Returned Gradient Nullifies the Decoys" (= `paper1_arxiv_submission`, dev pkg `Paper-1-…`) | **arXiv submitted** (+ JAIAI journal adaptation drafted) | The forward probe passed, but the **returned-gradient wire leaks**: chaff rows have exactly zero gradients, so the zero pattern discloses the real rows (4,096/4,096 frames, 9 seeds; +0.65–1.50 pp attack recovery). Gradient clip+noise closes it for ~0.01 nats. Rare-token reconstruction fails entirely (0/3,829). |
| **Attack-suite papers** — arXiv 2610.22722 (framework), 2610.04128 (per-token/per-doc measurement) | **arXiv published; journal submissions drafted (Nov 1)** | Adding returned gradients to the attacker's view **triples document recovery: 13.71% → 37.77%**; activations alone already 94.2% token recovery. |
| **`Distributed Training/` master program** ("Defense-in-Depth…" draft) | Draft + huge evidence base | Undefended boundary leaks **55–80%** of tokens; noise/quantization/top-k/LoRA are measured **non-defenses**; **E8 secret orthogonal transform → ≤3.7% at exactly zero utility cost**, accumulation budget K₅₀ ≈ 0.09·H; split FT bit-exact, async overlap 6.7–9.3×; FedAvg break-even K*≈1600. |
| **Paper 2** — "How much can a protected boundary replace?" | Manuscripts exist; **interpretation re-scoped by two independent Sept reviews** | The headline reversal: **separately trained zero-return controls match or beat the trained remote** in all 6/6 comparisons → *remote necessity was not demonstrated*. 42.2% is the clean point (47.4% demoted); 27B "refutation" quarantined (geometry bug); NLL passes (+0.034) while answer-token match collapses −16.64 pp. |
| **Paper 3** — R1 trilemma exploration | **Exploration complete: all 19 configs rejected**; confirmation NO-GO (power ≤ 26.4%) | Only **4 of 24 privacy instruments qualified** — the rejections are largely an instrument-collapse story. `span-6` failed utility by 3×10⁻⁵ nats. Deeper remote modules were *better* (depth-4 ΔU 0.0091 vs base 0.0348). |
| **Paper 4** — Defense redesign under calibrated instruments | Pilot done; **no cell pilot-eligible**; `main.tex` still a scaffold | Utility passed, causal remote gain **passed every cell**, but the privacy gate failed everywhere: **punctuation advantage 0.79–0.89 vs threshold 0.02 (~40×)**, even at q=1. TEE–FHE track: per-token CKKS ≈ 775 ms/token → FHE training not viable; TEE path (11–24% overheads) is the route. |
| **Privacy-contract** — "an explicit leakage budget" | **Complete draft + v14/v15 done; one TODO (abstract) from submission** | Certified **MINE ≈ 0 nats** on the defended boundary; MINE objective flips E2 FAIL→PASS; delegation triangle: privacy+quality @14%, privacy+delegation @28.6% (utility fails). |

Separate threads (not this program): `SLT/`, `Local Cloud/`, `Research-Paper-Writing-Skills/`, `outputs/` (CDAD), `Compare/`.

---

## 2. What we did not pay attention to before

These are the cross-paper patterns no single paper's docs headline:

### A. The measurement apparatus is now the bottleneck — and it failed *four independent times* the same way
1. **Paper 1**: a passing probe missed the gradient channel entirely (unmeasured channel).
2. **Paper 2**: the privacy evaluator was mis-specified — attacks scored against the **next** token, measuring prediction, not reconstruction.
3. **Paper 3**: 20 of 24 instruments failed calibration → 19 rejections are partly an instrument artifact, not a property of split training.
4. **Paper 4**: even the rebuilt instruments sit ~40× above threshold at the punctuation target.

**No single paper says this.** Together they are a publishable methodological thesis: *"In split-LLM privacy, the verdict is currently dominated by the measurement instrument, not by the defense."* This is the strongest candidate for the next synthesis paper (and it is written almost entirely from evidence you already have).

### B. The scientific identity hiding in plain sight: **leakage is structural, not statistical**
- Value-distortion defenses (noise, quantization, top-k, LoRA, private FT at utility-safe settings) are **measured non-defenses** (Distributed Training).
- Quantization at **q=1 does not touch** the punctuation signal (Paper 4).
- Only a **structure-changing orthogonal transform** (E8) suppresses recovery to ≤3.7% at zero utility cost.
- And Paper 1's leak is itself *structural* (exact zeros), defeated only by structure-aware fixes.

One sentence the program has earned and never printed: **attacking the values is futile; only the structure leaks, and only changing the structure defends.**

### C. Causal remote learning is the most robust result in the entire program — and it is always the runner-up
Paper 4: gain gate passed **every cell**. Paper 3: base 0.0126, and the deepest remote modules had the *lowest* utility cost and *largest* gain. Paper 2: the model demonstrably *uses* its return (PTB +0.029 nats, CI [+0.004, +0.055]) — the distinction "dependence ≠ necessity" is made but not exploited. The joint pass fails **only** on privacy measurement — so the trilemma "rejection" understates a positive: *two of three axes are consistently satisfiable.*

### D. A handful of unmined numbers
- **Noise increases remote contribution ~8× while destroying utility** (Paper 3 noise cells) — an untitled utility/leakage/response interaction.
- **Precision plateau above 8 bits** (ΔU 0.03480 vs 0.03482) — in the numbers, never in a narrative.
- **`span-6` utility margin 3×10⁻⁵ nats** — the frozen 0.10 gate itself is doing science-affecting work.
- **The 2,000-step budget is not split-invariant** → spurious privacy "failures" above ~28.6% delegation. This is a warning for the *whole field*, currently buried in a campaign report.
- **`joint_real_scaled` (+1.17 pp, z=+12.64)** — Paper 1's strongest arm is direct evidence of a cross (wire × gradient) attack term, and the cross-Gram attack itself was never run.
- Sequence-aware attackers (+0.25 pp) and MoE lookahead identity: clean nulls, never written up as nulls.

---

## 3. Consistency debts found during the audit

- `Distributed Training/README.md` cites **`REPORT.md`, `SUMMARY.md`, `PAPER_DRAFT.md`, `COVERAGE*.md` — files that don't exist** at top level; `paper-data/TABLES.md` has headers only.
- **E8 "0.00%" (README) vs "≤3.7%" (synthesis paper)** — needs one canonical statement (best vs worst across the battery).
- **FedAvg K\*≈1600** (naive transport) vs **K\*≈20/10** in `split-training/DESIGN.md` (other transport assumptions) — same symbol, different regimes, never reconciled in one place.
- Paper 3 abstract counts "11" vs "13" breaches — the same finding counted two ways.
- E8 evidence is mostly **n=1 per cell** (one victim corpus/seed); `h100-1` lost 2026-08-07 — E8/E7 JSONs there need re-verification; 35B-A3B MoE E8 cell unmeasured.
- Paper 1: E1 numbers were **RELAYED from code not in the repo** (port has since landed → a re-run converts them to VERIFIED); PLAN gate "1 of 8"; post-freeze confirmatory seeds still owed.
- 13 `LLM-Attacker*` sibling directories are snapshot worktrees of one repo — archive candidates.

---

## 4. Recommended next moves (highest value first)

1. **Finish and submit the Privacy-Contract paper.** It is a complete draft with v14/v15 done and one TODO. It is also the program's only *positive certified* result ("privacy = a number in the contract", MINE ≈ 0 nats) — the natural flagship of the set.
2. **Write the measurement paper** (§2A + §2B): *"Why everything was rejected: calibrated attacks and the structural nature of leakage in split LLM training."* Source material already exists: Paper 3's instrument collapse + `R1_CONFIRMATION_REVIEW`, Paper 4's calibration campaign + the 40× number, Paper 1's channel lesson, Paper 2's evaluator defect. This is where the program's science is genuinely strongest, and it converts three "negative" packages into one positive methodological contribution.
3. **Rescue Paper 2 as the narrow paper it has become**: "The limits of a small protected boundary — a controlled negative result on remote necessity" (the reviews literally recommend this; the NLL-vs-behavior mismatch, +0.034 vs −16.64 pp, is its best single figure). The 27B re-run is only needed if the broad claim stays on the table.
4. **One pooled confirmatory experiment instead of four per-paper ones** (the exact lesson of Paper 3's power NO-GO): ~10 documents × ≥3 seeds, covering E8 on all five models (incl. MoE), the cross-Gram attack, the post-freeze Paper 1 seeds, and a matched-intact baseline for Paper 2's economics.
5. Housekeeping that protects the record: reconcile E8's 0.00%/≤3.7% and the FedAvg K\* pair; fix the `Distributed Training` doc pointers; archive the `LLM-Attacker*` worktrees and `Paper-01`; re-verify the ex-h100-1 JSONs.

---

## 5. One-line summary

> Three papers proved that the naive split boundary leaks (55–80% token recovery), that the leak lives in the **gradient's structure**, and that only **structure-changing** defenses hold — at zero utility cost. The remaining papers did not fail: they discovered, four times over, that **the measurement instruments, not the training, set the ceiling** — and naming that discovery is the most scientifically significant result still on the table.
