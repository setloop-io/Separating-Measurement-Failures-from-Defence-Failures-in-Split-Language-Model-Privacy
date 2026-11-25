# Prospective BiSR conformance amendment v2

This amendment repairs two preflight blockers before any real-data candidate
outcomes. It supersedes the affected C-A rules in the 17 September fidelity
specification; that document, its JSON companion and frozen v4 remain immutable.
The failed diagnostic at `e65dd48` remains part of the record. Neither an
engineering pass nor this amendment establishes empirical fidelity.

## Validation and auxiliary inputs

Use **all eligible distinct articles** from the pinned WikiText-2 raw test split,
whose source inventory contains 62 articles. The actual eligible count is frozen
by the input manifest before predictions. Do not count windows as independent
documents. Sort by SHA-256 of `bisr-conformance-v1` plus normalised text. Use the
first complete 32-token window per validation article; articles shorter than 32
tokens are excluded before outcomes. Retain all exclusions and their reasons.

For historical B1 WikiText inputs, conservatively exclude the complete training
partition from validation. Also check the v3 partitions, pilot, historical SIP
auxiliary corpus and historical substitute Sensi CSV. Compare normalised text
hashes and, where available, normalised article titles. Missing exclusion inputs
still block selection. This is exact-document separation, not proof of absence
from the language model's pretraining data.

Auxiliary training may reuse previously used training articles. This is an
explicit relaxation of the earlier fresh-auxiliary requirement: shared auxiliary
training is permissible for an implementation comparison, while validation
documents remain held out. No claim that auxiliary articles are outcome-unseen
is allowed. Select the first 112 train articles under the existing `bisr-aux-v1`
hash order after excluding all v3 and pilot articles and reserved validation
documents. Use all complete, non-overlapping 32-token windows within each
article; report articles producing zero windows. Pin the reused-article count.
If fewer than 112 articles remain, selection fails; do not silently reduce it.
Train fresh GRUs for the original three seeds and original 20-epoch settings.

The three seed pairs (31415, 27182, 16180), eight endpoints, equal document
weighting, 90% Student intervals and strict ±0.01 equivalence margin are unchanged.
The smaller document set narrows the empirical scope and may leave wider
uncertainty. Do not add seeds or relax the margin after outcomes. Scoring must
take the expected document count and ordered IDs from the frozen input manifest.

## Algorithm authority

For C-A, the reference is the executable SplitLLM source at
`c376590227f9718077765824af4088665bb78687`, explicitly adapted to one shared seam.
The candidate selects `splitllm-c376590`; the existing paper-equation variant is
retained as `legacy-paper-variant` and cannot qualify this upstream comparison.

- GM optimises the full sequence variable initialised as `softmax(SIP/tau)`.
  The dummy next-token loss applies a second softmax to that variable, with no
  further temperature. Use the upstream shifted soft-label cross entropy.
- Its objective is beta times the sum of squared gradient differences plus
  (1-beta) times the sum of absolute differences. GM AdamW uses betas (0.9,0.999),
  epsilon 1e-6 and weight decay 0.01. Position zero has no label-loss gradient;
  its argmax is preserved by the positive decay factor at the declared budget.
- SM uses the sum of per-sequence mean cosine losses, AdamW epsilon 1e-6 and
  declared weight decay 0.01. Decode by upstream softmax(dot product) argmax,
  not Euclidean distance. Optimiser batch size remains one.
- Preserve raw SIP initialisation, GM then SM ordering, public downstream
  weights, explicit exact/public client-head grants and scorer-only truth.

The source discrepancy is not resolved by claiming the old equations and source
are interchangeable. This amendment chooses source conformance and records that
choice. Original-setting replication still requires original SensiReplaced and
the dual-cut configuration; C-A never discharges that limitation.

## Recheck and admission

Execute upstream and candidate GM and SM on identical saved observations and
weights. Retain initial variables, gradients, updates and decoded outputs, with
the original 1e-6 absolute / 1e-5 relative tolerance. Run in a fresh output
directory; do not overwrite the failed attempt. Source extraction may omit type
annotations, simulator weight restoration on already shared frozen weights and
progress/scoring display. It must not change attack arithmetic.

A passing small fixture only resolves the discovered mechanical discrepancy.
Full admission still requires trained shared checkpoint pins, capture replay,
all remaining mechanical/metric checks, environment and resource pins, then the
complete matched real-data comparison. No full campaign is admitted by default.
