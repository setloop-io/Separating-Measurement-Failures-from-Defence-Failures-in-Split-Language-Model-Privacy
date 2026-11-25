# BiSR implementation and conformance tooling — 17 September 2026

The prospective [fidelity specification](bisr-fidelity-specification-2026-09-17.md)
was committed as `a941f34`. The implementation below supports its single-seam
adaptation route. It does not certify fidelity or admit a campaign.

## Implemented

- SIP: single-layer, unidirectional GRU with 256 hidden units and output dropout
  0.1. Training visits every auxiliary window once per seeded epoch, including
  the final partial batch. Select `--decoder gru --train-epochs 20 --batch-size 32
  --lr 0.001` on the existing `bisr-baseline` training command. Supply the approved
  auxiliary corpus and disjointness inputs; synthetic fixtures are tests only.
- BRE: `f` starts from raw SIP; `bf` runs gradient matching then smashed-data
  matching. Beam decoding remains separate. Eager attention is required because
  gradient matching needs second derivatives.
- Grants: an exact client head requires a SHA-256 digest and complete matching
  head parameters. A separate public downstream model prevents tied embeddings
  from accidentally granting exact downstream weights.
- Prediction: a separate process accepts only declared observations and pinned
  model inputs, exports all four arms per sample, and refuses to overwrite an
  existing prediction file. The legacy command also exports predictions, but
  still scores in-process and is not the prospective label-isolation boundary.
- Scoring: `attacker.bisr_fidelity.score_predictions` consumes exported predictions
  and scorer-only truth. It reports positional accuracy, per-token-decoded
  ROUGE-L, METEOR and token-set F1. Install the `bisr-metrics` optional extra;
  provision NLTK WordNet separately where required. Missing resources fail rather
  than silently substituting a metric.
- Comparison: `compare_scores` requires all four arms, all 256 paired documents
  and the three specified seeds, then evaluates all eight co-primary endpoints.
  Wide confidence intervals remain inconclusive. Statistical agreement alone
  never changes `launch_admitted` to true.

## Prediction entry point

```bash
python experiments/bisr_fidelity_predict.py \
  --manifest prediction-manifest.json --output predictions.json --device cpu
```

The manifest has exactly these fields:

```json
{
  "schema": "bisr.prediction_manifest.v1",
  "public_model": {"path": "public.pt", "sha256": "<digest>"},
  "inverter": {"path": "gru.pt", "sha256": "<digest>"},
  "observations": {"path": "observations.pt", "sha256": "<digest>"},
  "knowledge": "public",
  "head": null,
  "boundary": 3,
  "bre": {"gm_iters": 32, "gm_lr": 0.09, "gm_beta": 0.85,
          "gm_tau": 1.2, "sm_iters": 800, "sm_lr": 0.01,
          "sm_wd": 0.01, "opt_batch": 1}
}
```

Paths resolve relative to the manifest. Public model bundles contain `config`
and `state_dict`; GRU checkpoints use `llm_attacker.bisr_gru.v1`. For an exact
grant, replace `head: null` with a pinned `llm_attacker.seam_head.v1` bundle and
set `knowledge: exact`.

Observation bundles contain exactly `schema: bisr.observations.v1`, `sample_ids`,
FP32 tensors `h` and `g`, an all-one `mask`, and a tensor
`gradient_batch_scale` containing each sample's original capture batch size.
Only full unpadded windows are supported. Truth is a separate
`bisr.truth.v1` object with `sample_ids` and `token_ids`, used by the scorer.

## Validation and remaining gates

`tests/test_bisr_fidelity.py` checks independent GRU gate equations and gradients,
complete epoch visits, exact/public separation, exact wire replay, GM and SM
updates, stage order, metric definitions, checkpoint round trips, deterministic
prediction and separate-process label isolation. Run alongside the existing
`tests/test_bisr_baseline.py` regressions. These small fixtures establish
engineering properties, not empirical fidelity on the declared document set.

Validation on 17 September: all 49 targeted tests passed with no skips in an
isolated CPU-only checkout on koula (20.90 seconds). This includes 33 existing
baseline regressions and 16 new conformance checks. Metric dependencies were
installed in a temporary test directory; existing experiment environments and
running GPU jobs were not changed.

Before any campaign, freeze and audit the real inputs, auxiliary/validation
disjointness, captures, checkpoints, environment and source revisions; complete
the matched upstream comparison and all mechanical/statistical gates. Detailed
compute accounting still needs campaign-level validation: current new-runner
reports contain optimisation diagnostics and budgets, not a FLOP certificate.
The original SensiReplaced version and published dual-cut reproduction remain
unresolved. No new campaign has been launched by this implementation work.

The frozen `experiments/validation_protocol.v4.json` remains unchanged, SHA-256
`cc23f15d5d7597ff2e73edd45e6f77b65aaf19ce3ed6f86d121352ce1ca5c4fc`.
