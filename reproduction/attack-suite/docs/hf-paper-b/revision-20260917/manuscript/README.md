# Paper B revision — 17 September 2026

This private evidence tier contains the revised manuscript, reviewer response,
analysis, frozen protocols and source. It is not a claim that every raw-data
transfer or every Phase 2 job is complete. Source revision: `f9d378081956bb9662a268c2cdc70cd001cd80c0`.

## Artifact map

- `main.pdf`, `main.tex`, `main.bbl`: revised paper and self-contained bibliography.
- `response-to-review-2026-09-17.md`: disposition of review issues.
- `paper-b-closure-2026-09-17.md`: validation and explicit outstanding items.
- `analysis/summary.json`: recomputed descriptive tables and exclusions.
- `analysis/whitebox/`: original36 and full summaries plus source package manifest
  and PACKAGE-READY anchor. The full summary is digest-checked against that anchor;
  the complete binary package is a separate publication tier.
- `phase2-metadata.tar.gz`: completed soula/voula/koula JSON reports and journals
  under `phase2/`, plus hash-pinned input corpora at original `results/` paths.
- `protocols/`: frozen Phase 2 and corrected white-box protocols.
- `source/`: revision source, frozen Phase 2 execution source (1ee12e111f7bc6306bd0a4249b129b7c2611522f), dependency lock and license.

## Verify and reproduce the tables

From this directory, verify all files before extraction:

```sh
shasum -a 256 -c MANIFEST.sha256
mkdir reproduction
cd reproduction
tar -xzf ../source/revision-source.tar.gz
tar -xzf ../phase2-metadata.tar.gz
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/python experiments/summarize_paper_b_followups.py --phase2 phase2 --whitebox ../analysis/whitebox --output recomputed.json
cmp recomputed.json ../analysis/summary.json
```

Compile the manuscript with `pdflatex main.tex` twice in this directory.
The bibliography is supplied as main.bbl; do not regenerate it with BibTeX.

## Regenerate experiments

Use a fresh directory containing the frozen execution source, the archived
`results/` input paths and requirements-research.lock. The Phase 2 protocol
lists exact argument vectors and input hashes. Its queue runner is
`python experiments/run_phase2_pending.py --node soula` (or voula/koula).
It requires pinned GPT-2 607a30d783dfa663caf39e06633721c8d4cfcd7e and Qwen3-0.6B
c1899de289a04d12100db370d81485cdf75e47ca model snapshots in the local HF cache.
The original koula orchestration waits for a local predecessor status file;
for independent replication, execute its protocol argument vectors with the
same substitutions after preparing models, rather than inventing a predecessor
completion record. Output directories must be new. Reproduction is compute work,
not required for descriptive reanalysis above.

Corrected white-box regeneration uses the protocol and reproduction instructions
in the sibling `whitebox35-20260917` packages, which preserve original inputs,
matched checkpoints, source and measured work. The original36 package is verified
at ba37427807a29f217e6fefb565c5e468e9bdac81. The full package upload is ongoing.
The new `revision-20260917/phase2-raw` tier likewise publishes only after complete
binary digest checks. Do not infer either tier's completion from this README.

## Baseline and limitations

Paper B baseline: 9833a3df9c660a6b214336d029095c46224c6987; audit supplement:
5b2e47c61e18b1bf33ff4e70095329d58bc471f6. Both were remotely verified.
Canaries are engineering records, excluded from scientific averages. Sasa's
500-step extension is excluded while incomplete. Old missing-head results are
invalid; 18 historical co-trained observations remain unevaluable. Fresh final
co-trained captures use a separately labelled stronger access grant. Historical
external v13/PR101 and August numerical claims were withdrawn; BiSR provenance
remains unestablished. Five of ten new agentic cells are UNMEASURED. See the
paper and closure record for all limitations. Repository visibility stays private.
