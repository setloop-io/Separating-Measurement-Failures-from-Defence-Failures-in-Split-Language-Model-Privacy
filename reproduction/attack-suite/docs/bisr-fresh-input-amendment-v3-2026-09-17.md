# BiSR fresh-input amendment v3

This prospective amendment replaces the v2 input selection before any training
or attack outcomes. It restores **256 distinct validation articles and 112
fresh auxiliary articles**. The reduced v2 set and its reused training data stay
archived; they must not be silently mixed with the new inputs.

Use Salesforce/WikiText, `wikitext-103-raw-v1`, revision
`b08601e04326c79dfdd32d625aee71d232d685c3`, both ordered training parquet shards.
Construct an **internal article-level holdout from the training partition**.
It is not the official WikiText-103 test split and does not reproduce the
published BiSR dataset. Preserve article continuity across parquet shards.

Before selection, verify the previous exclusion files against their frozen
hashes. Exclude all WikiText-2 train, validation and test articles, v3-study IDs,
pilot inputs, historical SIP auxiliary inputs, historical B1 local corpus,
both text columns of the Sensi substitute, and both groups of the v2 freeze.
Use normalised full-text hashes and normalised titles where available. Also
exclude exact 32-token historical prefixes. These checks concern the recorded
project history, not language-model pretraining exposure or semantic duplicates.

For validation, sort by SHA-256 of `bisr-conformance-v1` plus whitespace-normalised
text, breaking ties by document ID. Select the first 256 eligible unique articles,
each with at least 32 tokens, and retain its first 32 tokens. Do not admit
duplicate titles, normalised texts or token windows. Do not inflate document
counts using multiple windows from one article.

Reserve validation before choosing auxiliaries. Sort remaining articles with
salt `bisr-aux-v1`; take the first 112 passing the historical exclusions and
validation separation. Every auxiliary article must produce at least one complete
32-token window. Retain all complete non-overlapping windows within each article,
never concatenate articles. Reject an auxiliary article if any of its windows
matches a reserved validation window, a historical prefix, or a previously
selected auxiliary article's window. Repeated windows within an article do not
increase the article count. Insufficient eligible inputs cause an error, never
a smaller or reused-data fallback.

Pin both source shards, tokenizer files, exclusion files, selection code, ordered
IDs, scorer-only truth, windows and this amendment. Freshness means zero exact
document/title overlap against those pinned historical sources; no absolute
claim about all past use or pretraining is made.

Retain the upstream algorithm profile from amendment v2, the three seeds
(31415,27182,16180), GRU training settings, eight endpoints, 90% intervals and
strict ±0.01 equivalence margin. The larger source corpus changes the sampling
frame and auxiliary-window count; report both. The comparison must consume the
new manifest and all 256 identities. The 61-article freeze is superseded for new
qualification work, but its evidence remains intact. Frozen v4 is unchanged.

Input correction does not admit training or establish fidelity on its own:
checkpoint/capture pins, remaining mechanical checks and the matched three-seed
comparison are still required. Original SensiReplaced is not replaced by this
conformance dataset.

Dataset provenance and licence: [Salesforce/WikiText dataset card](https://huggingface.co/datasets/Salesforce/wikitext/tree/b08601e04326c79dfdd32d625aee71d232d685c3),
CC BY-SA 3.0 / GFDL as declared by the source. Preserve attribution on publication.
