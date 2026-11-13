# Paper B revision response — 17 September 2026 (updated 19 September)

This response supersedes the 16 September letter's status claims. The revised
paper separates implemented capabilities, archived real-model evidence, toy
machinery checks, and unresolved provenance or fidelity requirements. It
withdraws unsupported numerical claims rather than treating a successful
upload or a module registration as validation.

| Review issue | Revision and current status |
|---|---|
| August deployment headline evidence | Original fleet journals/manifests remain missing. Removed deployment table, noise-breakdown curve, original defence ranking, Mistral TAG negative, and corresponding abstract/taxonomy claims. No successful recovery is claimed. |
| External defence v2–v13 and PR-101 | Removed latent-era, flips, compromise, matching, integrity and reproduction numbers as active evidence. Their records are distinct from Paper A's recovered/retired P0–P5 programme. No external defence PASS is asserted. |
| Exploratory→confirmatory lineage | Withdrew the unsupported claim that the external exploratory findings became v3. The canonical v3 source freeze and primary contrast stand on their own documented provenance. |
| Scoring and control floor | Defined floor at control-rate recovery, retained token clustering and positive-control requirements, and removed unverified measured seed-variance claims. Historical versions are not asserted to share identical constants. |
| Minimum-control-rate flag | Corrected the prior “Done” statement: the generic gate does not implement a corpus-specific minimum-control-rate refusal/flag. This remains an explicit limitation; no automatic protection is claimed. |
| Domain and horizon results | Retained source-backed tables. Identified the long-horizon synthetic-calibration zero control and pooled-summary versus worst-per-step distinction. Corrected cross-epoch mid-lite recovery from uniformly zero to 0–0.5%. |
| Corrected white-box objective | Integrated the PR #35 correction and follow-up code with frozen commits preserved. Added original 36 rerun, 198 stable all-boundary cells and 99 new co-trained final-inference cells. Twenty PyTorch tests pass. |
| Layer coverage and compute | Reported every internal GPT-2 boundary 1–11, all seeds and widths, and measured head/readout work and wall time. Depth 12 and canaries excluded. No post-hoc privacy gate or universal depth claim. |
| Historical co-trained observations | The old 18 cells remain NOT_EVALUABLE. New co-trained evidence grants a final victim head and matching final encoder on newly captured held-out inference wires; it is not pooled with the public-head training tier. |
| Sequence lengths/dialogue/adaptation | Added all completed sequence-64/128 code/WikiText conditions, independent dialogue and the adaptive-whitening comparison. Reported both directions of the adaptation result descriptively. |
| Matching/order/chaff | Completed three-seed synthetic reference-bundle checks, explicitly distinct from external v13 and published VMA replication. Token and order endpoints remain separate. |
| Live planner | Updated to the new two-trial greedy run: five of ten cells UNMEASURED. Goal-hijack violations are budget violations, not demonstrated exfiltration. Deputy vulnerable cells failed parsing, preventing a live defensive-contrast conclusion. |
| 500-step extension | Execution completed; the extension remains excluded from this release pending evidence collection, independent verification and analysis. No scientific validation is inferred from execution completion. |
| BiSR | Single-seam GPT-2 implementation conformance now established on frozen fresh WikiText-103 inputs: 24 engine cells, 6,144 document–engine runs, all twelve trained mechanical gates and all 32 endpoint/stratum comparisons pass. Three paired seed differences are zero for each endpoint, yielding 90% Student-t intervals [0,0] within the prespecified ±0.01 margin, conditional on fixed documents. The original SensiReplaced/published dual-cut reproduction, superiority and matched-v4 results remain unclaimed. |
| HF evidence | Baseline and audit supplement verified; corrected original-36 package verified. Full white-box package (31 files) and Phase 2 raw supplement (four files, 7.15 GB) are verified at revisions `08ed4bd0` and `7aac5e7b`, respectively. The dataset is now public. |

Validation and immutable release anchors are recorded in
`docs/paper-b-closure-2026-09-17.md`. Completing an upload does not resolve
scientific provenance or convert an operational gate PASS into a privacy
certificate.

The 19 September conformance evidence and reproduction instructions are archived at
[the immutable fidelity release](https://huggingface.co/datasets/Setloop/llm-attacker-paper-b/tree/be59bdca280258ac82ce358cae5f1e0c529869c2/fidelity-20260919/evidence).
This closes only the declared C-A implementation-conformance requirement. The frozen v4 B1 gate requires the original published setting and remains blocked; the C-A pass does not admit that campaign.
