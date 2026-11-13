# Expanded DLG++ evidence — Qwen3-0.6B

This directory contains the completed expanded DLG++ gradient-inversion run
collected from `h100-2` on 2026-08-07.

## Experimental coverage

- Model: Qwen3-0.6B
- Split depths: 1, 4, 8
- Victim documents: 10 per depth
- Attack seeds: 0, 1, 2 per document/depth
- Total cells: 90

Every expected depth × document × seed combination occurs exactly once in the
JSON artifact.

## Aggregate results

| Depth | Token recovery, mean ± sample SD | Embedding cosine, mean ± sample SD | n |
|---:|---:|---:|---:|
| 1 | 1.77% ± 0.68% | 0.0105 ± 0.0071 | 30 |
| 4 | 0.31% ± 0.64% | 0.0034 ± 0.0054 | 30 |
| 8 | 0.21% ± 0.54% | 0.0016 ± 0.0055 | 30 |

These are cell-level summaries over the ten documents and three attack seeds.
The individual observations remain in the JSON and should be used for
document-clustered inference in the paper analysis.

## Files and integrity

- `dlgpp_06b_docs10.json`: final structured result and provenance.
- `../logs/dlgpp_06b_docs10.final.log`: complete execution log.

SHA-256:

```text
b21d69dd26f822827e893ee0929bc4e9c0fdc9a9bc3f96b22564ab09b22d4439  dlgpp_06b_docs10.json
44621132b2f0fe1ce812912c723cf378f5e954eb2ba134fa9cbd8fde42bf5c9c  dlgpp_06b_docs10.final.log
```

The JSON records the model manifest hash, corpus hash and document count,
victim-document indices, repository commit, container image, hostname, attacker
configuration, threat model, all 90 results, and aggregate summaries. No model
checkpoint or credential is included.
