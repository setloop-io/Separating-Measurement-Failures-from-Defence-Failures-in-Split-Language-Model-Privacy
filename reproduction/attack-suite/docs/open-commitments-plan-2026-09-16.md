# Plan: Open Revision Commitments (Papers A & B)

## Superseding Paper A scope decision — 17 September 2026

The original plan below is retained as history, not current acceptance criteria.
The user authorised revising Paper A around verified v3 and corrected GPT-2 P6
results. P0–P5 source provenance could not be restored: recovery and arithmetic
audit are complete, but their scientific claims are withdrawn. P0 engineering
tests and Pearson-r advantage replace the incorrect qualification/R² descriptions.

Corrected P6 has 192 trained cells plus eight controls and complete analysis.
It is Mode 2 and cannot satisfy the original plan's Mode-1/P5 gate-confirmation
criterion. The paper reports all four tests and the fixed held-out-seed check.
The full corrected evidence release remains independently gated on pinned HF
verification. Qwen stopped on non-finite training; no completed replication is
claimed. BiSR fidelity and the matched comparison remain unresolved and are
explicit limitations, not work marked Done. These narrower manuscript changes
do not cancel the separately authorised Phase 2 or PR #35 executions.

See `paper/paper_a/response-to-review-2026-09-17.md` and
`docs/paper-a-closure-2026-09-17.md` for the current revision and release status.

---

**Date:** 2026-09-16
**Status:** planned — nothing here is executed yet
**Origin:** the third adversarial review of Paper A, plus commitments recorded in the revised manuscripts and the response-to-review letter.

Three commitments are currently flagged in the texts as future work:

1. **WS1 — Evidence release for P0–P5 phase records.** Paper A now promises that "the eight per-run gains will be tabulated in the released P2 summary" and that "the property label set and decoder baselines are defined in the P0 qualification protocol." Neither artifact exists in this checkout; the P0–P5 records live in the external programme's records.
2. **WS2 — Crossed span×start-block design.** Table 5's P5 sweep is a 7-point diagonal (starts 3,3,2,2,1,1,0), so span, starting location, and replaced capacity co-vary. Acknowledged in the caption; the crossed comparison is deferred.
3. **WS3 — Matched BiSR reproduction.** Paper B's `bisr-baseline` is a pilot-scale adaptation; Paper A's v3 protocol records "BiSR is not included or claimed reproduced." The review demands a direct comparison.

Ordering rationale: WS1 is days and unblocks the review response; WS3 is engineering-heavy but self-contained; WS2 is new experiments and benefits from WS3's infrastructure decisions. WS1 → (WS3 ∥ WS2).

---

## WS1 — P0–P5 evidence release package (no new experiments)

**Goal:** every number the reviewer challenged becomes independently recomputable from packaged, hash-pinned artifacts.

**Steps**

1. **Locate and export the phase records** from the programme's working repository/hosts. Required: P0's 30 qualification-test definitions and outcomes (property label set, decoder baselines for sequence-property-v2 and structure-semantic-v2); P2's per-seed paired held-out losses (frozen-random vs trained) for the 8 victim seeds [11..88]; phase decision documents referenced in Table 4 (`docs/PREREGISTRATION_P0.md`, `docs/P2_CONFIRMATORY_DECISION.md`, etc. — these paths do not exist in this checkout and must be exported or the table corrected).
2. **Tabulate the eight per-run P2 gains** and recompute the statistics from raw pairs. Must reproduce: mean 0.0167, SD 0.0032, t(7) = 5.92, p ≈ 2.9×10⁻⁴, W⁺ = 36. **This also resolves the Wilcoxon centring definitively** (margin-centred vs zero-centred — currently hedged in the text; the raw data decides, and the text is then de-hedged).
3. **Write the P0 qualification summary**: the 30 tests, the property label set, decoder baselines, and the *r*-advantage definition (R²_attack − R²_control) as now stated in the gate.
4. **Package per `docs/evidence-release-v3.md` conventions**: artifact index, per-file SHA-256, verification recipe, provenance manifest (host/date/code version). Publish as `docs/evidence-release-p0p5.md` plus the `outputs/`-tree the Table 4 paths point to.
5. **Update Paper A**: replace "will be tabulated" with the actual 8-row appendix table; point Table 4's paths at the packaged files; de-hedge the Wilcoxon sentence per step 2.

**Acceptance criteria**

- A third party recomputes the appendix statistics from the tabulated gains and lands on the printed values at displayed rounding.
- Every path in Table 4 resolves to a real, hash-pinned file.

**Effort:** 1–3 days. No GPU. **Risk:** phase records incomplete → contingency is to mark exactly what is unrecoverable in the release note; never reconstruct numbers from memory.

---

## WS2 — Crossed span×start-block delegation design (P6)

**Goal:** separate the effects of span, start block, and replaced capacity that P5's diagonal confounds; test whether a depth-only baseline predicts gate outcomes out of sample.

**Design (to be frozen before execution)**

- **Grid:** spans {2, 4, 6} × starts {0, 1, 2, 3} (feasibility: start + span ≤ block count of the surrogate segment; exact bound from the P5 surrogate spec), defenses {D0, D1}, 8 victim seeds → 3×4×2×8 = **192 trained cell-evaluations per architecture** (P5 had 112). GPT-2 first; Qwen3-0.6B second if budget allows.
- **Primary endpoint (predeclared):** after conditioning on span, does start block explain residual variance in any gate measure (Δ_U, worst advantage, FLOP share)? Two-way decomposition with crossed-bootstrap CIs, consistent with the programme's existing statistics.
- **Depth-only baseline test:** fit the gate-outcome predictor on 6 seeds, evaluate on 2 held-out seeds; report whether start adds predictive power beyond span/depth alone.
- **Discipline:** protocol `p6-crossed-span-1.0` frozen via freeze commit before any run (precedent: `064a64f`); no outcome-based retries; fail-closed error journaling; evidence-v2 sealing throughout.

**Acceptance criteria**

- Crossed ledger table (span × start × defense) in Paper A's appendix with the variance decomposition.
- A pre-committed text rule, written into the protocol before execution: if any crossed cell passes all three gates, the empty-set claim is updated to name the exception and re-scope; if none do, the caption's diagonal caveat is replaced by the crossed confirmation.

**Effort:** ~1 week including fleet runs (comparable to P5's 288 runs). **Risk:** a gate outcome flips → handled by the pre-committed text rule, not post-hoc framing.

---

## WS3 — Faithful BiSR reproduction + matched confirmatory comparison

**Goal:** replace both papers' "pilot-scale / not reproduced" disclaimers with a faithful, matched comparison against Chen et al. (CCS 2024).

**Phase A — fidelity (engineering)**

Gap analysis against `attacker/attacks/bisr_baseline.py` (current state): the SIP-style per-position inverter, bidirectional candidate *selection*, forward-consistency beam, and fail-closed aux-corpus disjointness already exist. Missing for fidelity:

1. SIP inverter trained on a **substantial** auxiliary corpus (current: pilot scale).
2. BiSR's **bidirectional optimization-based attack** (gradient-matching optimization, not only candidate selection).
3. **NaMoE** noise-robustness component.

Route decision: **clean-room re-implementation from the paper is preferred** (repo precedent: VFLAIR VMI/WBI clean-room; avoids license contamination). The upstream [SplitLLM repository](https://github.com/StupidTrees/SplitLLM) (official BiSR code, `experiments/scripts/pipeline/demo_bisr.sh`, gpt2-large/wikitext settings) is used only to *validate* — first check its LICENSE; if permissive, direct porting of the optimization attack is acceptable with attribution. **Validation gate before any of our protocols:** reproduce a published BiSR setting (their repo's gpt2-large/wikitext demo) within documented tolerance of the paper's reported numbers, recorded in the module docstring.

**Phase B — matched comparison (frozen protocol amendment)**

- Add `bisr-faithful` arms to the v3 runner under a new protocol amendment (v4 candidate; v3 stays immutable): depths {3, 6}, public and exact knowledge, k = 32-matched budget, 10 victim seeds × 2 attack seeds × 256 held-out documents — same harness, corpus, and statistics as `joint_true_k32`/`forward_k32`/SIPIT-128.
- Predeclared contrast family: BiSR vs joint-true (primary), BiSR vs forward-only and vs SIPIT-128 (secondary), paired within victim run, 100k crossed bootstrap, Bonferroni across the family. Registry additions follow the full checklist (contract, metadata, coverage pins, profile version, tests).

**Acceptance criteria**

- Validation-gate reproduction documented; matched comparison table sealed with hash-chained evidence.
- Paper B's "pilot-scale adaptation … remain outstanding" sentence and Paper A's protocol-level "not reproduced" note replaced by the matched result.

**Effort:** 1–2 weeks engineering + ~10–30 GB10 GPU-hours (≈80–160 new cells at the v3 per-cell average of 5–6 min). **Risks:** SplitLLM license restrictive → clean-room fallback (already the default); aux-corpus scale for SIP fidelity may exceed disjoint WikiText-2 supply → use a disjoint slice of C4 or The Pile with the same fail-closed disjointness assertion.

---

## Cross-cutting requirements (all workstreams)

- **Freeze before execute**, with commit anchors printed in the papers (the `064a64f` pattern).
- **Evidence-v2 sealing** on every run; crash-safe JSONL journals; no outcome-based retries.
- **Disjointness discipline**: any auxiliary or new corpus is checked fail-closed against `experiments/validation_corpus_manifest.json` and the pilot corpus.
- **Paper hygiene on landing**: each workstream ends with the corresponding manuscript text updated from "future work" to result, the response-to-review letter's status table updated, a compile check, and a commit.
- **Test suite green**: any framework code (new modules, runner changes) ships with contract/metadata/coverage/profile/tests per the established checklist; `pytest tests/ -q` must pass before each paper-text update is committed.

## Suggested timeline

| Week | Work |
|---|---|
| 1 | WS1 complete (release package + Paper A appendix table); WS3 Phase A starts |
| 2–3 | WS3 Phase A (fidelity + validation gate); WS2 protocol drafted and frozen |
| 3–4 | WS2 runs on the fleet; WS3 Phase B matched comparison |
| 4 | Both papers' texts updated; response letter statuses → Done; resubmission package |

**Owner note:** WS1 requires access to the external programme's phase records — if those records cannot be located, WS1 downgrades to an explicit "unrecoverable items" note in the release document rather than reconstructed numbers.

---

## Status addendum 2026-09-16 — fourth-review work items

Three no-new-runs items from the fourth adversarial review are **done**:

- **Cluster-aware gate re-analysis** (`experiments/analyze_clustered_gate.py`,
  `results/clustered-gate-v1/`): document-clustered paired bootstrap over the
  domain-shift and long-horizon tables. 23/24 verdicts robust; one cell
  (Gaussian σ=15, 20-step) is borderline under clustering and is reported as
  marginal in the paper. Remaining commitment: the same check for the fleet
  campaign cells (external artifacts).
- **Merged-package collision correction** (`experiments/merge_validation_shards.py`):
  fixed, verified end-to-end repo-locally; see the `docs/evidence-release-v3.md`
  addendum of the same date.
- **Positive-control enforcement** (`attacker/scorecard.py`, `scorecard-policy.v3`):
  a PASS now requires a linked positive-control artifact exceeding the gate
  threshold on an undefended reference target, else fail-closed downgrade to
  INSUFFICIENT_EVIDENCE; legacy behavior behind `--scorecard-policy
  scorecard-policy.v2`. Numeric gate (`frozen-gate.v2`) unchanged.

Still open from that review (need new runs): faithful BiSR reproduction (= WS3),
frame-retaining chaff counter-test, matched order-recovery evaluation, and the
scope/realism extensions (sequence-length sweep, independent-domain corpus,
repeated sessions, convergence probe, adaptive tier).
