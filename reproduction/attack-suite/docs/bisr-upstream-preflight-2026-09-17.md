# BiSR upstream preflight — 17 September 2026

**Real-data validation: blocked. Matched mechanical diagnostic: failed. Fidelity
remains NOT_ESTABLISHED.** No recovery campaign or new GPU training was launched.

The diagnostic plan and executable were committed before execution as `e65dd48`.
They compare candidate `d148925` with upstream
`c376590227f9718077765824af4088665bb78687`. Pristine upstream functions were read
from git objects, avoiding the historical checkout's working-tree patches.

## Input admission

The cached, pinned Salesforce/WikiText `wikitext-2-raw-v1` revision
`b08601e04326c79dfdd32d625aee71d232d685c3` has **62 test articles**, before any
historical exclusions. The prospective specification requires 256 distinct
documents from that partition. The existing article parser in
`experiments/prepare_pilot.py` identifies document boundaries; splitting articles
into windows would change the unit of analysis and cannot fill the shortfall.

The test Arrow file is pinned by SHA-256
`2b8a3efac7b468cbe6432edba5f55c21e435d93873acc6727431f08d5ed328ea`.
The inventory records every article's title, text hash and normalised-text hash
across train, validation and test. This freezes a source inventory, **not an
admitted selection or a real-input launch manifest**. Historical exclusions,
tokenisation, auxiliary selection and fresh checkpoint preparation have not been
completed because the initial count already blocks admission.

A prospective amendment must choose either a smaller eligible test-document set
or a separately audited source supporting 256 documents. Retain the three seeds
and report the changed scope; do not replace articles with dependent windows or
relax equivalence margins after observing attack results.

## Actual upstream comparison

A CPU FP32 diagnostic used one shared frozen synthetic GPT-2 model, identical
observed activations/gradients and identical SIP logits. It executed the pinned
upstream TAG attack and gradient-loss methods, alongside the candidate's actual
GM implementation, for one AdamW step. Framework parameter restoration was a
no-op because both sides already used the same frozen public downstream model.
This isolates attack mechanics; it is not the full three-seed, real-data C gate.

| Detail | Pinned upstream | Candidate at d148925 |
| --- | --- | --- |
| Optimised initial variable | `softmax(SIP_logits / tau)` | `tau * SIP_logits`, excluding position zero |
| Soft labels inside GM | `softmax(variable)` | `softmax(variable / tau)` |
| Gradient distance | beta times **squared** L2 sum plus L1 sum | beta times L2 **norm** plus L1 norm |
| GM AdamW epsilon | 1e-6 | 1e-8 |
| GM AdamW weight decay | 0.01 | 0 |
| SM AdamW epsilon | 1e-6 | 1e-8 |
| SM embedding decode | maximum dot product | minimum Euclidean distance |

Initial variables, their gradients and updated variables fail the declared
elementwise tolerance. Maximum absolute differences were respectively
3.5610404, 0.1262200 and 3.5614321. These variables have different
parameterisations; the figures are a mechanical mismatch diagnostic, not an
effect size or recovery-performance contrast. The source-level differences
also alter the objective and optimiser trajectory.

Executing upstream `_rec_text` on the fixed embedding table
`[[1,0], [10,0], [0,1]]` and continuous input `[1,0]` selects token 1; the candidate's
Euclidean rule selects token 0. Thus these decode rules are not interchangeable.
The upstream SM optimiser epsilon was inspected in source; this diagnostic did
not execute the full upstream SM loop or certify the untested gates.

The earlier 49 passing local tests verified engineering properties and the
candidate's chosen equations. They did **not** prove agreement with upstream.
The candidate must gain an explicitly upstream-conformant path, or the
paper-equation variant must remain separately labelled. Reconcile the reference
authority and configuration deltas in an additive specification amendment before
generating real-data outcomes; preserve the present failed attempt.

## Retained evidence

Local: `results/bisr-conformance-preflight-20260917/`.
Remote: `koula:/tmp/bisr-upstream-preflight-20260917/results/`.

- `corpus-inventory.json`: all source article identities and partition digests.
- `mechanical.json`: tolerance results, optimiser values and source hashes.
- `mechanical.pt`: shared model, observations, SIP logits and both optimiser traces.
- `upstream-*.py`: pristine pinned source files used by the diagnostic.
- `environment-packages.json`: installed package versions on the CPU test host.

The tensor bundle SHA-256 is
`d35c59f0d08f6fab16b6e3641beeefca70f26a0484f799e9921d0209441da142`.
The audit script SHA-256 is
`92594f21bb4fa50e460378cd4bdea560c95cdae9112b35a1b1da22c022a128bc`.
Generated evidence remains outside git in accordance with the repository's
artifact policy; the source, plan and this report are committed. Evidence has
not been uploaded to HF by this preflight.

Frozen v4 is unchanged. Original SensiReplaced provenance and the published
dual-cut reproduction remain separate unresolved requirements.
