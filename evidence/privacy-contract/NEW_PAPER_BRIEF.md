# New paper brief: "The Privacy Contract — an explicit leakage budget for split LLM training on compromised cloud nodes"

Working title. Companion/sequel to `papers/arxiv-draft/main.pdf` ("Defense-in-Depth
for Split LLM Training on Fully Compromised Cloud Nodes", 23 pp, in-repo at
`../Distributed Training/papers/arxiv-draft/`).

## One-sentence thesis

Replace "empirically, the attacker finds nothing" with **a certified, explicit
leakage budget**: the encoder is trained under `Î(z; x) ≤ B` (MINE-estimated mutual
information between released latent and token identity), so privacy becomes a
**number a customer can write into a contract** — bits per token — with a measured
privacy/utility frontier the customer picks their operating point on.

## Why now (the evidence that justifies this paper)

All numbers below are measured and committed in `../Distributed Training/paper-data/`
and the campaign report (`docs/experiments/LATENT_CAMPAIGN_REPORT_V7_V9.md`):

1. **The defense works empirically** (v9.2): frozen nine-arm attacker at the
   statistical floor on 3 model families (0.6B / 12B / 35B hybrid MoE),
   excess +0.29…+0.73 pp over the label-free majority control, gate ≤ +1.0 pp;
   compromise battery (full capture / chaff-ID / partial gauge compromise / 20%
   known plaintext) flat; matching attacks (VMA, permutation reversal) below the
   no-attack control.
2. **A certified lower-bound estimator exists and behaves** (v13): the MINE
   Donsker–Varadhan estimator reads **≈0 nats on the defended boundary**
   (−1.3×10⁻⁶ on 0.6B; −0.2×10⁻³ on 35B) and **nonzero signal on undefended
   views** (validated non-blind).
3. **The MI objective flips verdicts** (re-evaluation): E2 long-frame leak
   +1.250 → +0.758 pp (seq-64), +1.364 → +0.264 pp (seq-128); noise σ=0.30
   tripwire +1.039 → +0.553 pp. The MINE penalty is the first mechanism since
   chaff that moves a FAIL to a PASS.
4. **The delegation triangle is measured** (§6 of the draft): privacy+quality
   at 14% delegation; privacy+delegation at 28.6% (utility fails); quality+delegation
   leaks 18–66%. Six independent axes (experts/depth/width/segments/function
   family/training duration) all saturate at the D=64 boundary's usable information
   (~1.5 contribution) — the ceiling is information-theoretic, not architectural.
5. **Integrity is solved empirically** (v9.5): K=3 Byzantine replicas catch a
   lying node at 100% detection / 0% false positives, utility intact under attack.

## The gap this paper closes (be honest about it)

- MINE/DV is a **lower bound** on MI: it can under-read, never over-read. A
  *certified* budget needs an **upper-bound companion** or a proof that covers
  what the estimator cannot see. Proposed formal stack:
  (a) gauge group-invariance (provable: fresh Haar rotation ⇒ uniform posterior
  over orientations; permutation ⇒ uniform over orders; log-normal scale),
  (b) MINE-certified budget on the remaining invariant channel,
  (c) Byzantine median for integrity (deterministic, K=3 tolerates one adversary).
- All evidence is **diagnostic scale** (2,000-step runs, 256-block corpus window
  of 16,896 tokens) — the paper must state scope honestly, as the first draft does.

## The proposed experiment core (v14)

Budget-parameterized training on the v9.2 winner (0.6B, D=64, chaff-48,
radial-MoE, scale gauge off):

| Cell | Budget B (bits/token) | What it measures |
|---|---|---|
| v14.0 | 0.25 | tightest certified tier; utility cost? |
| v14.1 | 0.50 | candidate standard tier |
| v14.2 | 1.00 | candidate performance tier |
| v14.3 | 2.00 | relaxed tier; does excess approach the gate? |
| v14.4 | unconstrained (β=0) | frontier's right endpoint |

Per cell: frozen gate excess, utility delta, eval ratio, MINE reading, cloud
contribution. Output: the **measured privacy/utility frontier** — the figure a
reader uses to pick their operating point.

## Related work to position against (verified in LITSEARCH docs)

- UCLA MI-bottleneck privacy adapters (arXiv:2606.11592) — formal MI theory,
  inference-only, no floor measurement, no gauges/chaff/integrity.
- Fraunhofer WACV'26 SAE dual-channel noise — closed-form MI budgets in vision;
  our LLM-side analog.
- AloePri (2603.01499) / GELO (2603.05035) — obfuscation systems; their
  overclaims cross-checked in EXTERNAL_PAPERS_PRIVACY_COMPARISON.md.
- TEE (Phala 2409.03992, ETH 2509.18886) — <7–20% overhead with hardware trust;
  we exclude the trust assumption deliberately.
- MPC/HE (SIGMA, BumbleBee, NEXUS) — 44 s to 14 min/token; 2–4 orders of
  magnitude over plaintext: the cost argument for empirical obfuscation.
- TOPLOC (2501.16007) — post-hoc integrity analog to our online K=3 vote.

## Business framing (validated with the user)

The contract budget is the product story: (1) contractible privacy — a
signable number instead of a narrative; (2) product tiers on the measured
frontier (strict/standard/performance); (3) insurable risk (actuarial input);
(4) short procurement via second-party reproducibility (the PR-101 bar).
Caveat to keep: lower-bound estimator — the sellable claim is "certified in
this estimator's terms + provable invariance + integrity," not a theorem.

## Draft skeleton (arXiv, target 10–14 pages)

1. Abstract: the contract budget; frontier measured; integrity replicated.
2. Introduction: from "trust us" to "write B in the contract."
3. Threat/system model (carried from draft §2).
4. The MINE-formalized encoder: objective, estimator, training loop.
5. Formal stack: gauge invariance proposition + budget + Byzantine median.
6. The v14 experiment: budget sweep; the frontier figure.
7. Robustness: frozen attacker + matching arms + compromise battery at each tier.
8. Business/product implications (tiers, insurability, procurement).
9. Limitations: lower-bound estimator, diagnostic scale, corpus window.
10. Related work; conclusion.

## Things to copy over at writing time

- Master ledger + campaign-map table (draft §5) as the evidence appendix base.
- The delegation triangle figure + six-axis ceiling table (draft §6).
- fig_reval.pdf (verdict flips) as the MINE-objective evidence figure.
- refs.bib (40 verified entries) — subset + the MI-estimation additions.
