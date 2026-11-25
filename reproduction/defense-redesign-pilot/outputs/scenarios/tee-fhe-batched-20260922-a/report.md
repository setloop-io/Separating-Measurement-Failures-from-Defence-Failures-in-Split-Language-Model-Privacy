# TEE/FHE fleet-side engineering report: tee-fhe-batched-20260922-a

Completed 2/2 scenarios; failures: 0.
Execution revision: `0f7b17317b7a65cd73fa141f576968ca5d4107c8`. Manifests, source archives, artifact bindings,
saved outputs, and the observation-contract byte totals verified.

Engineering diagnostics only: `scientific_evidence: false`, `hardware_attested: false`,
privacy **undetermined**. No confidentiality, integrity, or attestation claim.

## Stage C — batched-matmul corrective

| Block count b | Packed/per-token ratio (measured) |
|---|---|
| 1 | 0.7256 |
| 2 | 0.8597 |
| 4 | 1.1422 |

20 repetitions retained across baseline and packed variants; no best-of selection.
Keygen-amortized serving constant: mean per-token mm 0.1183s over 32 sequential mms on one context; keygen share 0.0028s/token. The op-count
explanation (rotations and diagonal multiplies scale with the output dimension
regardless of packing) is recorded from these measurements, not asserted.

## Stage D — selective split policy costs

Per-token homomorphic mm cost per remote-linear shape: mlp_fc 0.1119s; mlp_proj 0.4508s; proj 0.1015s; qkv 0.1115s.

| Policy | Homomorphic (s) | End-to-end estimate (s) | Ciphertext bytes | Traffic bytes | Ratio to plaintext batched | Interactive conceivable |
|---|---|---|---|---|---|---|
| none | 0.000 | 0.000 | 0 | 0 | 1 | True |
| ffn-up-only | 14.317 | 14.988 | 42423910.4 | 655363 | 1.149e+04 | False |
| all-remote-linears | 99.280 | 101.957 | 169725593.6 | 2097165 | 1.43e+04 | False |

Conceivability statements are recorded per policy from these measurements only
(declared interactive reference budget; no extrapolation to other models, hardware,
backends, or layouts, and no security claim).

See [verification](verification.json), [summary](summary.json), and [source record](source.json).
