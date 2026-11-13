# Paper 9 — Why Everything Was Rejected

*Calibrated Attacks and the Structural Nature of Leakage in Split-LLM Training*

Draft started 2026-11-05. Status: **full draft, all sections written, compiling** (`paper/main.pdf`, 5 pp + refs).

## What this paper is

The synthesis study of the program: four independent studies (Papers 1–4)
stopped not because a defense failed but because the **measurement
instrument** did — unmeasured gradient channel; next-token evaluator;
20/24 uncalibrated instruments; a gate sitting ~40× below the data. With
calibrated instruments the positive result is clean: leakage is
**structural, not statistical** — value-distortion defenses are measured
non-defenses; only structure-changing defenses hold (≤ 3.7 % at zero
utility cost; K₅₀ ≈ 0.09·H).

## Sources (all evidence already exists)

- `Analyse the experiments./SYNTHESIS.md` §2A–B (the thesis)
- `Analyse the experiments./ADVERSARIAL_REVIEW.md` (positioning + reviewer attacks → §6)
- Paper 1 (arXiv 2609.04382), Paper 3 (19 rejections, 4/24 instruments, ≤ 26.4 % power)
- Paper 4 (pilot: 0.79–0.89 vs 0.02 gate ≈ 40×)
- `Distributed Training` E8/E9/E-R (≤ 3.7 %, 50→0 % live wire, K₅₀)
- `paper-llm-attack-suite` (2610.04128: 13.71 → 37.77 %)
- `Privacy-contract` (MINE ≈ 0 nats; the contract framing)
- `papers-db/DB.md` curated set → `refs.bib` (TODO: real author lists)

## Open TODOs (before this draft goes to co-authors)

1. Figures: F1 the four-failures schematic; F2 values-vs-structure recovery
   comparison (55–80 % vs ≤ 3.7 %); F3 the gradient-channel gain
   (13.71 → 37.77 %).
2. `refs.bib`: replace placeholder author lists.
3. Decide title variant ("Why Everything Was Rejected: …" vs a
   measurement-first alternative) — and whether the E8/E-R detail stays
   at this depth or moves to an appendix table of numbers.
4. Cross-check every number against its source study's claim ledger before
   circulating (they were lifted from `SYNTHESIS.md` / source READMEs).
5. Depends-on note vs Papers 1/3/4: this paper cites our own unpublished
   studies — decide how to package (companion arXiv refs vs appendices).
