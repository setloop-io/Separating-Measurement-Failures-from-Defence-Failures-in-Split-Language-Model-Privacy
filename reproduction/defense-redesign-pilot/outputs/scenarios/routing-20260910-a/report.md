# Engineering scenario report: routing-20260910-a

Completed 6/6 scenarios; operational failures: 0.
Execution code: `a8dff2126832f5fbb2048437c1e0af6a646082a9`. Source archive and artifact hashes verified.

These records are engineering diagnostics. Privacy remains **undetermined**.

| Policy | Prompt bytes | Delegated calls | Online serialized bytes | Max logit error |
|---|---|---|---|---|
| delegate_all_phases | 8 | 96 | 3548464 | 0.0 |
| local_prefill_delegate_decode | 8 | 64 | 2496736 | 0.0 |
| local_reference | 8 | 0 | 0 | 0.0 |
| delegate_all_phases | 16 | 96 | 6694320 | 0.0 |
| local_prefill_delegate_decode | 16 | 64 | 4593952 | 0.0 |
| local_reference | 16 | 0 | 0 | 0.0 |

Setup bytes are recorded separately in each run. This is CPU plaintext loopback
with full-prefix recomputation, not a network, TEE, FHE, GPU, or KV-cache benchmark.
The model is an initialized fixture. Correct routing establishes no trained-model
utility, causal learning gain, confidentiality, or active verification result.

See [summary](summary.json), [source record](source.json), and [verification](verification.json).
