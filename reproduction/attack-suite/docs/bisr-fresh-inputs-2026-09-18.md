# Corrected BiSR inputs — 18 September 2026

The reduced, reused v2 input set is superseded for new qualification work by
the fresh v3 freeze. Historical files and frozen v4 remain unchanged.

| Quantity | Frozen v3 inputs |
| --- | ---: |
| Distinct validation articles | 256 |
| Distinct auxiliary articles | 112 |
| Complete 32-token auxiliary windows | 15,604 |
| Auxiliary articles overlapping pinned historical documents/titles | 0 |
| Auxiliary articles yielding no complete windows | 0 |
| Exact validation/auxiliary token-window overlap | 0 |

The sample comes from both training shards of Salesforce/WikiText
`wikitext-103-raw-v1`, revision `b08601e04326c79dfdd32d625aee71d232d685c3`.
Validation is a new **internal article-level holdout**, not the official test
partition. The [prospective amendment](bisr-fresh-input-amendment-v3-2026-09-17.md)
defines the selection and historical exclusions; code and amendment were
committed as `9c2986e` before selection. No attack outcomes informed selection.

All WikiText-2 partitions and the pinned historical input records are excluded
by normalised document hashes and titles where available, with an additional
exact token-prefix check. Each selected article contributes at least one full
window. These are freshness checks against recorded project inputs, not proof
of absence from model pretraining or semantic near-duplicates.

The frozen manifest is `experiments/bisr_frozen_inputs.v3.json`, SHA-256
`4a4f5b76a3e86bf67eba48a35907c9381eddf34ed9e6c1ccf10662320486fe8d`.
It binds the dataset shards, prior manifests, exclusion files, tokenizer,
selection script, ordered validation IDs, selected text, windows and scorer truth.
`experiments/bisr_inputs_current.json` identifies it as the current selection.

## Verification

- Both downloaded shard digests match the LFS digests at the pinned HF revision.
- All input-source and generated-output hashes were verified.
- All selected validation and auxiliary articles were independently re-tokenised;
  every exported window matches the complete within-article token sequence.
- Counts, masks, unique article identities, nonempty auxiliary contributions,
  scorer truth and cross-group document/title/window separation were checked.
- Input and algorithm regressions: **54 passed, no skips** on koula, 23.45 seconds.

Evidence is retained locally at `results/bisr-fresh-inputs-20260918/` and in the
durable remote directory `koula:~/llm-attacker-bisr-fresh-inputs-20260918/`.
The original execution directory `/tmp/bisr-fresh-inputs-20260918/` is also
preserved. `verification.json`, `source-verification.json`, `tests.log` and the
`frozen-inputs` files record these checks; the remote frozen files live under
`results/frozen-inputs/`. Generated text/tensors remain outside git; the source
code, manifest and report are tracked. No HF upload was performed in this step.

The three seeds, algorithm settings and equivalence margin are unchanged.
Training uses 15,604 windows per epoch, so the resource estimate must use this
count rather than the old 10,918. Checkpoint preparation and the matched
three-seed validation are still pending; corrected inputs alone do not establish
fidelity or reproduce the original SensiReplaced setting.
