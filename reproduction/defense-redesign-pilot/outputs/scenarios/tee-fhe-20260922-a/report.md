# TEE/FHE fleet-side engineering report: tee-fhe-20260922-a

Completed 2/2 scenarios; failures: 0.
Execution revision: `d7787e0a3a834f909aae1d441700d33a7a131351`. Manifests, source archives, artifact bindings,
saved outputs, and the observation-contract byte totals verified.

Engineering diagnostics only: `scientific_evidence: false`, `hardware_attested: false`,
privacy **undetermined**. No confidentiality, integrity, or attestation claim.

## Stage A — same-checkpoint correctness (plaintext-emulated boundary)

Max |Δlogit| 0 (tolerance 1e-4); |Δloss| 0 (tolerance 1e-5); argmax agreement 1.0000 (required 1.0).
Observation contract: 704 operator calls over 16 steps, 5637181344 serialized bytes total.
This contract is the traffic inventory later attested stages must match.

## Stage B — real CKKS encrypted projection (TenSEAL)

| Tokens | Reps | Encrypt (s, mean±sd) | Homomorphic mm (s, mean±sd) | Decrypt (s) | Refresh (s) | End-to-end (s) | Plaintext CPU reference (s) | Ciphertext bytes | Max abs error |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 5 | 0.0025±0.0001 | 0.1209±0.0042 | 0.0005 | 0.0024 | 0.2221 | 0.000642 | 331139 | 4.13e-07 |
| 32 | 5 | 0.0690±0.0007 | 3.7001±0.0465 | 0.0254 | 0.0830 | 3.9659 | 0.000442 | 10602564 | 6.3e-07 |
| 128 | 5 | 0.2765±0.0020 | 14.8936±0.1827 | 0.0780 | 0.2805 | 15.6169 | 0.001481 | 42419084 | 6.99e-07 |

All repetitions retained; no best-of selection. These timings are real FHE
measurements of one encrypted projection at the boundary scale — they are not
secure-serving latencies (no TEE, no attestation), not end-to-end inference times,
and not a defense evaluation.

See [verification](verification.json), [summary](summary.json), and [source record](source.json).
