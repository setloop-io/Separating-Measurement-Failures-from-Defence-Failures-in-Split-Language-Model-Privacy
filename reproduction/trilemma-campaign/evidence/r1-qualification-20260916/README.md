# R1 attack qualification and exploratory admission — 2026-09-16

First scientific-stage evidence of the R1 programme. **Scoped outcome: of the 24
required target/channel/defense combinations, 4 qualified as instruments; the other
20 are unqualified, so exploratory privacy is UNDETERMINED by construction for those
channels and no scoped PASS is available under this instrument suite.**

## Chain

1. `calibration--cal001` on toula: 76/76 runs, zero failures, 26,816 s vs the
   28,800 s cap, execution root retained on toula and (in progress) on the
   orchestrator. Budget/environment reviews and the single-use admission are in
   this package.
2. Qualification analysis at analyzer commit `bc5242e` (the analyzer fix for
   bundle-root-relative manifest bindings; execution packet untouched):
   `analysis/calibration_analysis.json` and the frozen-spec proposal
   `analysis/calibration_analysis.frozen-attack-spec.json`
   (`proposal_ready_for_freeze_review`).
3. **QUALIFIED (4/24)**: `input_byte@inbound_gradients@{none,gradient-defense}`,
   `input_byte@outbound_gradients@{none,gradient-defense}` — known-plaintext
   probes on the two full-precision gradient channels pass positive-control
   floors (+0.22 to +0.40 margin) with nulls inside the 0.02 band.
4. **UNQUALIFIED (20/24)**: every `forward_values` and `returned_values` channel
   and every `next_byte_class` / `punctuation_fraction` combination — predominantly
   the null criterion (matched-null advantage 0.045–0.56 above the 0.02 band).
5. **Freeze decision (owner)**: thresholds UNCHANGED (`attack-spec-frozen.json`).
   The amendment path was rejected: qualifying more combinations would need
   `null_band >= 0.041`, admitting instruments with demonstrably invalid nulls.
   Consequence recorded in the frozen spec note: exploratory cells can only be
   REJECTED (utility/gain gates or a qualified gradient-channel breach) or left
   UNDETERMINED; unqualified channels are named in every report per protocol.
6. `exploratory--exp001` on koula: 232 runs admitted against the frozen spec
   (`547b5a93…`), started 2026-09-16 ~17:20 UTC; budget 86,400 s cap.

## Contents

- `budget-*.json`, `env-review-*.json` — the scientific budget and environment
  reviews for both stages (owned-hardware wall-clock caps, nominal USD 0.50/h).
- `admission-calibration.json`, `admission-exploratory.json` — the consumed
  (calibration) and live (exploratory) single-use admissions.
- `attack-spec-frozen.json` — the owner-signed frozen attack specification.
- `calibration_analysis_summary.json` — compact outcome summary.
- `analysis/` — full calibration analysis and the freeze-review proposal.

Full raw calibration trees (37 GB) are retained on toula and being rsynced to the
orchestrator; the retrieval hash inventory lands in the Phase 4 evidence package.
