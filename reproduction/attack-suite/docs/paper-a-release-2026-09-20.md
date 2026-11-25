# Paper A release index — 20 September 2026

The complete corrected GPT-2 P6 raw release is verified. This additive
manuscript supplement updates the evidence links and distinguishes completed
single-seam BiSR conformance/adaptation from the unavailable original setting.
The Qwen FP64 amendment is running and contributes no results to this version.
Historical P0–P5 results remain audit-only; there is no active joint
privacy–utility–delegation feasibility claim.

## Immutable evidence

The public dataset is `Setloop/llm-attacker-paper-a`. No visibility change was
made during this closure check. Always download the revision and prefix together.

| Tier | Revision and prefix | Contents |
| --- | --- | --- |
| Baseline | `a02fb09280dae643da4f5a11abf08ca8f62d8337` | v3 campaign, inputs and historical audit; its old manuscript is superseded |
| Frozen P6 analysis and reproduction | `f7b2aef7649588e9021805f2edc25428b6f26c93`, `revision-20260917/manuscript/` | 19 verified files: analysis, frozen source, protocol, inputs and original reproduction instructions |
| Complete P6 raw evidence | `0bec5c37b960fdb5be5b2bd8fab1cdd05a4bbe4c`, `revision-20260917/p6-raw/` | 11 files, 34,703,146,309 bytes; 192 trained cells and eight controls |
| Current manuscript supplement | `closure-20260920/manuscript/` at its separately recorded immutable revision | Updated PDF, LaTeX, bibliography, figures, provisional arXiv source ZIP, metadata draft and raw-verification reports |

[Frozen source and reproduction index](https://huggingface.co/datasets/Setloop/llm-attacker-paper-a/blob/f7b2aef7649588e9021805f2edc25428b6f26c93/revision-20260917/manuscript/README.md)
and [complete raw archive](https://huggingface.co/datasets/Setloop/llm-attacker-paper-a/tree/0bec5c37b960fdb5be5b2bd8fab1cdd05a4bbe4c/revision-20260917/p6-raw).
The older reproduction index's private-access and raw-pending statements are
historical. This index supersedes those release-status statements without
altering its frozen bytes or scientific protocol.

The raw publisher verified all 200 cells against the frozen analysis anchors,
protocol, file hashes and evidence journals before archiving. It completed at
19:00:18 UTC on 20 September. A separate check queried the immutable HF tree,
matched the exact file set, verified the downloaded small-file SHA-256 hashes
and matched all large archive parts to their pinned LFS SHA-256 digests. This
second check did not download and extract the entire 34.7 GB archive again.
Raw manifest SHA-256:
`0daaabcefa94de9631f8e7f9b448a8383b0b20245d73e4ab105601b2e57e4f17`.

## Download and verify

Use `huggingface_hub.snapshot_download` with `repo_type="dataset"`, the exact
revision above and `allow_patterns=["<prefix>/**"]`. Keep each tier in a separate
directory. Within a downloaded tier, run:

```sh
shasum -a 256 -c MANIFEST.sha256
```

Verify all raw parts before concatenating them in lexical order into a fresh
extraction directory:

```sh
cat evidence.tar.part* | tar -xf -
```

The archive preserves original paths and bytes in the evidence records; the
verifier resolves moved sibling files. Failed historical runs, pilots and
engineering canaries are not final cells in this archive.

## Reproduce the analysis and execution

Obtain `source/p6-source-d5ec03e.tar.gz` and `p6/validation-input.json` from the
frozen analysis tier. Extract the source into a clean directory, install its
package and `requirements-research.lock` in an isolated environment, and place
the supplied input at `results/validation-corpus-v1/validation-input.json`.
The source commit is `d5ec03eaa561241bf71564bdf08e7fb1501dde78`; the protocol
SHA-256 is `88e2286339109d8c6356b75b5263a23647261136eb0d039ecca15075c03d533b`.

From that source snapshot, with a fresh output path:

```sh
python experiments/analyze_p6_corrected.py \
  --root /absolute/path/to/p6-corrected-gpt2-v2 \
  --protocol experiments/crossed_span_protocol.p6.v2.json \
  --output /absolute/path/to/fresh-analysis.json
```

The analyser requires the complete 200-cell matrix and validates hashes and
journals before analysis. Numerical outputs should match the frozen analysis;
absolute `evidence_anchors` paths depend on extraction location. Reanalysis
does not regenerate model training or predictions.

For full regeneration, first cache GPT-2 revision
`607a30d783dfa663caf39e06633721c8d4cfcd7e`. With the frozen input and an
appropriate PyTorch/CUDA environment, the recorded execution is:

```sh
python experiments/run_p6_corrected.py grid \
  --protocol experiments/crossed_span_protocol.p6.v2.json \
  --span 0 --controls --node reproduction --device cuda \
  --workdir /absolute/path/to/fresh-p6-grid
```

This is a GPU workload. Hardware and library differences can affect numerical
regeneration; bitwise identity across environments is not asserted. A failed
cell blocks the complete comparison and cannot be silently omitted. Future
agent-managed executions in this project must use its ORX run contract.

## Manuscript build and scope

The supplement's `source/` directory and arXiv ZIP contain the same six source
files. From `source/`, run `pdflatex main.tex`, `bibtex main`, then `pdflatex
main.tex` twice. The current 12-page PDF compiled without undefined references,
citations, overfull boxes or LaTeX warnings. All rendered pages were inspected;
the changed results, discussion and evidence sections received a closer review.
PDF, ZIP and individual source hashes matched the metadata. Figures are
unchanged from the previously reviewed v3 assets; no new statistical analysis
or figure regeneration was performed for this release-link update.

The companion Paper B's [current manuscript](https://huggingface.co/datasets/Setloop/llm-attacker-paper-b/blob/6a853d04eb50d7dd639cb0bf08354cc4a032278d/sasa-500step-20260920/manuscript/paper_b.pdf)
reports single-seam BiSR conformance and the separately frozen adaptation
comparison at distinct budgets. These do not establish original-setting
dual-cut fidelity or equal-budget superiority. Shared findings across the two
papers are not independent replications.

The arXiv material remains provisional while Qwen qualification and final
integration are pending. Author names, affiliations, category and distribution
licence require author decisions before an actual submission. No submission
has been made and no manuscript licence is selected by this package.
