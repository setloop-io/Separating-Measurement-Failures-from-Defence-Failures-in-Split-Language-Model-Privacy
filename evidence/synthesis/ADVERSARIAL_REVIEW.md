# Adversarial Review & Literature Positioning

*Companion to [SYNTHESIS.md](SYNTHESIS.md) and [PAPERS.md](PAPERS.md). Literature retrieved 2026-11-05 via alphaXiv/OpenAlex (orx discover).*

---

## Part I — Is anyone else doing this? (Yes — and that is both the risk and the opportunity)

The field around split/distributed-LLM privacy is **active and crowding fast** — a 2026 survey is already dedicated to it ([A Survey on Split Learning for LLM Fine-Tuning](https://www.alphaxiv.org/abs/2604.24468)). Mapping the closest prior art to each of your claims:

### 1. The measurement-crisis thesis (your proposed Paper 9) — **strong external validation, but you are not first to *say* it; you may be first to *demonstrate* it for split-LLM training**
- [Adaptive Model Inversion Attacks Generalize a Privacy-Robustness Tradeoff](https://www.alphaxiv.org/abs/2610.07677) — standard MIA evaluation **underestimates leakage 1.16–6.59×**; "standard MIA evaluation can mistake optimization and measurement failures for privacy." This is *exactly* your Paper 3/4 lesson, independently, for vision MIA.
- [Label-Budgeted Attribute Inference … Split Multimodal LLMs](https://www.alphaxiv.org/abs/2610.32079) — proposes a **recovery-curve** measure of feature-level privacy because native evaluations "underestimate an adaptive server's capabilities." Same diagnosis.
- [On the Privacy of LLMs: An Ablation Study](https://www.alphaxiv.org/abs/2605.02255) — unified threat model, four attacks ablated in isolation.
- **Gap you own:** these are inference-time / vision. Nobody found does what Paper 3+4 do: a *fully pre-registered, calibrated, null-controlled instrument battery for split-LLM training* where 20/24 instruments fail calibration and the failure is *reported*. Position Paper 9 as the training-side demonstration of a phenomenon the field is currently discussing only for inference.

### 2. The Privacy Contract (MINE-bounded leakage budget) — **closest prior art; sharpen the novelty claim**
- [Defense Against Prompt Inversion Attacks: An Information-Theoretic Approach](https://www.alphaxiv.org/abs/2606.11592) — learns representations by **minimizing MI between activations and prompt**, with "privacy adapters via low-dimensional information bottlenecks" and theoretical privacy–utility tradeoffs.
- [From Prompts to Responses (PIDI/ADMI)](https://www.alphaxiv.org/abs/2606.14210) — **MI-regularization defense for Split-LLM** against dual-sided inversion.
- **So MI-bottleneck privacy is no longer novel per se.** Your differentia must be: (i) **training** (not inference), (ii) a **certified** DV lower bound read as a **contract number in bits/token**, (iii) the **fully-compromised-cloud** threat model that includes the **backward channel** (which both above are not), and (iv) the honest lower-bound caveat. Cite both papers and state the delta explicitly — a reviewer will.

### 3. Paper 1 (zero-pattern decoy leak) — **appears genuinely novel for LLMs**
- In CV-FL, gradient sparsity as a leak is known (survey: [2206.07284](https://www.alphaxiv.org/abs/2206.07284); [Inversion Influence Functions](https://www.alphaxiv.org/abs/2309.13016)). For **LLM split training with decoys**, the exact-zero mechanism + the calibrated canary/null protocol looks like your own contribution.
- Corroboration that the community is converging on "gradient *structure* leaks": [TRACE](https://www.alphaxiv.org/abs/2609.30258) finds cross-time gradient correlation leaks trajectories in embodied RL and concludes "sequence-aware privacy mechanisms" are needed — echoing your sequence-aware null.

### 4. Defenses: chaff/masking is a live idea, not a relic
- [Aegis](https://www.alphaxiv.org/abs/2609.38339) defends by **superimposing masking gradients** to push effective batch beyond the attack's recovery capacity — the same family as your chaff. [HyperFL++](https://www.alphaxiv.org/abs/2611.02412) "breaks the direct connection" between shared parameters and private data via hypernetworks — same spirit as your boundary transform.

### 5. Layer/depth non-uniformity of leakage — **two 2026 papers independently found it**
- [What Does the Server See? (ActInv/PAF)](https://www.alphaxiv.org/abs/2605.23158) — "privacy vulnerability is not uniform across layers."
- [BettiSplit](https://www.alphaxiv.org/abs/2607.24556) — topology-guided split selection; inversion fidelity reaches 0.98 SSIM at deep split points.
- Your depth-gate (~12 layers) and width-bound (D=128) results should be **cross-cited** with these; they are independent support that your depth findings are not an artifact of your stack.

### 6. Also worth knowing you are not alone on
- MoE leakage as a fresh surface: [structural memorization in MoE](https://www.alphaxiv.org/abs/2610.41039) (your 35B-A3B E8 cell is unmeasured — the field is moving).
- Cloud-edge leakage audits for decoding: [CoVeil](https://www.alphaxiv.org/abs/2608.29111).
- Gradient attacks on federated LLM + PEFT: [FedSpy-LLM](https://www.alphaxiv.org/abs/2604.06297).

---

## Part II — The adversarial review

*As the harshest plausible reviewer of the program as a whole (and of the proposed synthesis paper).*

### Attack 1 — "N = one corpus, one instrument model, one budget." **Severity: high (narrowed after checking the evidence).**
**Correction (2026-11-05):** the *systems/defense* line is already multi-model — 6 victims across 2 families (Qwen3-0.6B, Mistral-7B, Qwen3-8B, Mistral-NeMo-12B, Qwen3.6-27B, Qwen3.6-35B-A3B MoE; see `EXPERIMENT_COMPARISON_MATRIX.md`), and E5 trained inversion covers 4 models. The real single-model exposure is the **measurement line**: Paper 1 and the Paper 3/4 instrument battery ran on **Qwen3-0.6B only**. And the corpus critique stands: everything rests on 2,000-step budgets and small corpora (Gutenberg/tiny-Shakespeare). The headline "leakage is structural" is demonstrated at *punctuation-level* structure on one corpus family, on one instrument model. A reviewer will ask: is "structure" a property of LLM boundaries, or of your corpus's punctuation statistics? **Required defense:** replicate the instrument battery on one already-available second family (Mistral-7B — no new hardware needed) and at least one second-domain corpus (the v14/v15 corpus work helps), or scope every claim to the tested family — which your Paper 3/4 discipline already does, but Papers 1–2 and the master draft are looser.

### Attack 2 — "Your strongest result (E8) is n=1 per cell." **Severity: high.**
E8's "≤3.7% at zero utility cost" is the program's flagship, yet it is mostly one victim corpus / one seed per cell, and the zeros are binomial upper bounds, not proofs. The h100-1 loss still blocks re-verification of the underlying JSONs. Meanwhile Paper 3's own power analysis (P(confirm) ≤ 26.4%) shows you know the standard. **Required defense:** the pooled confirmatory experiment (10 docs × 3 seeds) before any "zero-cost defense" claim is promoted.

### Attack 3 — "Your attacker class is the load-bearing assumption." **Severity: high.**
Everything is honest-but-curious, mostly passive, non-adaptive. Your own v4 result (adaptive KPA pooling breaks simple rotation) already shows the ranking of defenses changes with attacker strength. A hostile reviewer: "the entire program's positive results are relative to a declared, weak attacker — and you know it, but the README headlines don't." **Required defense:** one active-attacker evaluation (Paper 4's C5) or one sentence in every abstract that says it.

### Attack 4 — "The measurement crisis is partly self-inflicted." **Severity: medium.**
Paper 3/4 froze acceptance thresholds (0.02) that **no tested defense could pass** — Paper 4's own audit caught the 0.10-vs-0.02 confusion. A reviewer will ask whether the instruments were calibrated to be passable at all. **Required defense:** report the instrument *detection floor* alongside the gate (you now have it: ≥6.0% coordinate-visible leak), and frame the two papers as instrument development, not verdicts on split training.

### Attack 5 — "Two nodes, one topology." **Severity: medium.**
All systems claims (bit-exactness, 6.7–9.3× overlap, FedAvg K\*≈1600) come from one 2-node topology with `tc netem` WAN; ≥3-node/pool settings and one THREAT_MAP seam are unattacked; the K\* 1600-vs-20 tension is unreconciled. Fine for a systems case study; not fine for "FedAvg loses at this scale" as a general claim.

### Attack 6 — "Certified" needs a footnote every time. **Severity: low, but recurring.**
MINE/DV is a **lower bound** on MI: the contract number is conservative — the true leakage could exceed B. The direction of the error favors the customer, but "certified" must always carry "lower-bound, at 2,000-step diagnostic scale."

### Attack 7 — Provenance debt is visible to reviewers. **Severity: low.**
Missing `REPORT.md`/`SUMMARY.md`, E1 RELAYED-not-VERIFIED, E8 0.00%-vs-≤3.7%, stale `training_status.json`, 13 worktree dirs. None is scientific; all is credibility. One housekeeping pass before the synthesis paper ships.

### What survives the attack (the reviewer's own summary)
The program's **methodological discipline** — pre-registration, canary/null controls, claim ledgers, hash-verified evidence, published negatives — is *ahead of the surrounding literature* (the very papers above that call for this discipline mostly lack it). The science is real but **scope-bounded**: one architecture family, one threat model, one topology, one budget. The right posture is the one Paper 3/4 already adopted — finite, named, reproducible statements — extended to Papers 1–2 and the master draft.

---

## Part III — Actions implied

1. **Cite and differentiate** against 2606.11592, 2606.14210 (MI-bottleneck), 2610.07677, 2610.32079 (measurement), 2605.23158, 2607.24556 (layer non-uniformity), 2604.24468 (survey), 2609.38339 (masking), 2609.30258 (temporal gradients) — in the Privacy-Contract paper and the synthesis paper, before reviewers do it for you.
2. **Sharpen the Privacy-Contract novelty claim** to: *certified* (DV lower bound) *training-time* leakage budget *including the backward channel*, as a contract number.
3. **Add the "attacker class" sentence** to every abstract (it is already in Paper 1's — keep it).
4. **One second-domain corpus + one non-Qwen family** before claiming "leakage is structural" generically.
5. **Pooled confirmatory run** (E8 multi-seed, E8-MoE, D3, post-freeze seeds) — now also justified externally by the 1.16–6.59× underestimation found in [2610.07677](https://www.alphaxiv.org/abs/2610.07677): stronger/adaptive instruments will likely *raise* your measured leakage, so measure it once, properly.
