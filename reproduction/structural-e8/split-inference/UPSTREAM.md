This directory vendors https://github.com/coder903/split-inference
at upstream commit ce6dfba811ff3058da134e92a128deb6e0027b69
("Sync work in progress before vacation", MIT license).

Vendored because this reproduction modifies several files; all
modifications are marked [QWEN36] or documented in ../patches/:
- inversion_experiment.py: model-agnostic dims (_model_dims), training-status writer
- defense_experiment.py: NEW (privacy defense grid + stronger attacker)
- jacobi_server.py, cloud_server_kv.py: hybrid linear-attention (Qwen3.6) support,
  streamed meta-device loading — see ../patches/qwen36/ for diffs
- rtt_analysis.py, lookahead_ablation.py, perplexity_comparison.py: streamed loading
