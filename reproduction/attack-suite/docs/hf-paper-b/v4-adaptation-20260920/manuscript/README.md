# Paper B: reviewed adapted-comparison revision

This 23-page revision adds the completed single-seam adapted comparison and preserves the separate C-A validation and original-setting limitations. [Immutable evidence and reproduction instructions](https://huggingface.co/datasets/Setloop/llm-attacker-paper-b/tree/d415492614cb1eb760926bf2e01a33b7fb024124/v4-adaptation-20260920/evidence).

The prespecified depth-6/public bf-minus-joint difference is −4.17 percentage points (95% crossed-bootstrap interval −4.79 to −3.53), under distinct attack budgets. The comparison does not reproduce the original published dual-cut/SensiReplaced setting. All four strata and the secondary family are reported without a universal ranking claim.

The independent numerical review and review of the new manuscript claims found no scientific blockers. The explicit protocol-interpretation erratum resolves the review's minor clarification. All 23 rendered pages were inspected; after inserting the immutable evidence link, the final rendered pixels matched the reviewed draft exactly. LaTeX completed twice without warnings. The raw archive and full CPU analysis reproduction passed; GPU retraining or a second prediction campaign is not claimed.

`PAPER-REVIEW.md` records the review-time file identities and conditions. `RELEASE-CHECKS.json` records their final resolution and the delivered source/PDF hashes. The evidence verification and reproduction attestations are included. These reviews focus on the new adapted-comparison material, not a fresh audit of every historical experiment.

To compile the paper, use a LaTeX distribution providing the packages named in `main.tex`, change into this directory and run `pdflatex -interaction=nonstopmode -halt-on-error main.tex` twice. `main.bbl` is supplied; no bibliography regeneration is required.

Existing baseline, C-A, corrected white-box and Phase2 packages retain their immutable revision links in the paper. The separate 500-step extension remains excluded pending collection, verification and analysis.
