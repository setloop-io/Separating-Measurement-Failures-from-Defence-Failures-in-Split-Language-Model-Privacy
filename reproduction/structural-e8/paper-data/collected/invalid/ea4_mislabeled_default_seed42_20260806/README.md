# Invalid E-A4 FedAvg repeats

These artifacts are retained for provenance but must not be used as independent
training-seed repetitions. The runs were labelled as seeds 43 and 44, while the
recorded FedAvg configuration shows `seed: 42` in both client JSON files because
the launcher omitted the explicit `--seed` argument.

The split-FT seed 43 and seed 44 artifacts outside this directory are valid.
Corrected FedAvg runs explicitly passing `--seed 43` and `--seed 44` supersede
the FedAvg artifacts archived here.
