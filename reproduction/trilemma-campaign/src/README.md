# Source code

The experiment package is `pcr` (`src/pcr/`), installed editable into the project
venv. It implements the P0 instrument and the P1 exploratory pilot machinery for
docs/RESEARCH_PLAN.md. Nothing produced by this package is validated evidence;
all outputs are exploratory- or calibration-cohort artifacts.

## Layout

- `model.py` — byte-level decoder-only transformer backbone (vocab 256, d_model
  256, 8 heads, 8 pre-LN blocks, d_ff 1024, learned positional embeddings,
  max_seq_len 128, no dropout; N(0, 0.02) init with residual projections scaled
  by 1/sqrt(2*n_layers)).
- `split.py` — split wrapper: trusted encoder (LayerNorm + Linear to `width`),
  boundary defense (uniform per-tensor affine quantization to `q_bits`, STE in
  training, true quantization for serialization/byte accounting), remote
  surrogate (one pre-LN block at `width`, head_dim 32, d_ff 4*width, final
  LayerNorm), trusted decoder with residual add. `AffineQuantizer`,
  `SplitBackbone`, `Surrogate`.
- `modes.py` — `all_local`, `trained`, `frozen_random` (remote frozen and
  verified), `removed_remote` (fixed identity, zero remote trainable params),
  `oracle` (split topology, width = d_model, no quantization).
- `data.py` — deterministic corpus split (first 90% train / last 10% held-out)
  and seeded batch samplers; the same seed reproduces the same batch sequence
  in every run (pairing requirement); the probe batch is identical across all
  runs (dedicated probe generator, seed 0).
- `train.py` — initialization (trusted side keyed by run seed; remote keyed by
  pairing block id), AdamW + warmup/cosine schedule, the core training loop,
  eval on fixed held-out batches, probe tensor capture, and `run_cell`.
- `transcript.py` — declared remote-visible transcript: per-step and
  accumulated byte counters for forward/returned values and outbound/inbound
  gradients via hooks, remote-state checksums, schema-validated inventory.
- `instrument.py` — analytic remote FLOP counter (`count_block_flops`,
  validated against a hand-computed case in tests).
- `privacy.py` — token-inversion probe attack family with matched
  shuffled-label null and positive-control dose curve over q_bits {4, 8, 16,
  32}; detection floor 0.05; per-run bits/token from counters.
- `manifest.py` — schema-validated run manifests (git commit, dirty-patch
  sha256, environment digest, resolved-config sha256, artifact hashes).
- `run_experiment.py` — config resolution (named factor levels to fully
  explicit resolved config), pilot enumeration, CLI:
  `.venv/bin/python -m pcr.run_experiment --config experiments/configs/pilot_matrix.yaml --run trained-short-low-s101`
  or direct flags (`--mode trained --span short --width low --seed 101
  --steps 200 --run-id smoke-trained-s101`).

## Tests

```
.venv/bin/python -m pytest tests/ -x -q
```

`tests/test_p0.py` qualifies the instrument (parameter updates, pairing
bit-identity, hand-computed FLOPs, transcript/manifest schemas, quantization
byte accounting, seed separation). `tests/test_privacy.py` qualifies the
token-inversion scorer (monotone dose response, 0.05 detection floor, null
advantage ~ 0). Both run on CPU with tiny models and inspect no pilot outcomes.

## Pilot

`scripts/run_pilot.py` enumerates the frozen matrix (48 cells + 1 calibration
run), skips runs with a completed manifest, and executes sequentially. Do not
launch it without an explicit decision to run the exploratory pilot.

Code imported from the earlier project must retain license and commit
provenance and must be audited for the historical seed-separation and
never-trained-remote defects before use.
