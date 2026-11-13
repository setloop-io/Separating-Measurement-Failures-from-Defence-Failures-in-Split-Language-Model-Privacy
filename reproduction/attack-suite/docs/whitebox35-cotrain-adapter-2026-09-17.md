# PR35 co-trained defense adapter

`experiments/whitebox35/cotrain.py` creates a **new post-training inference experiment**. It never attacks the historical co-training wires using a final checkpoint, never relabels inference wires as training wires, and never changes the historical eighteen NOT_EVALUABLE cells.

The adapter trains a fresh GPT-2 victim with the existing interleaved trained-bottleneck implementation. It saves the final full victim, final client head, final encoder and geometry. It then records final-state `inference` / `prefill` captures on the development `utility_eval` partition. Every persisted fp16 wire must be exactly reproducible by reloading the exported head and encoder. State digests and parameter-gradient checks enforce that observation makes no model/encoder updates.

The separately labelled attacker is copied from PR35's corrected objective: optimise continuous token embeddings through **the final victim head followed by the final encoder**, then snap to vocabulary embeddings. It explicitly grants both final modules, a stronger access condition than the historical public pretrained head. Capture phase, head path, checkpoint and capture hashes are explicit in `entry.json`. The adapter rejects altered artifacts and training-phase entries. Predictions and scoring truth are retained separately; truth is used only for scoring. Per-step objectives and metered head-forward/backward work are retained.

Example production cell (fresh output required):

```bash
python -m experiments.whitebox35.cotrain \
  --output results/whitebox35-cotrain-inference/split3-seed11-dim64 \
  --depth 3 --seed 11 --dim 64 --device cuda \
  --train-steps 64 --tb-minimax-steps 200 --tb-cotrain-interval 8 \
  --attack-sequences 32 --sipit-steps 128
```

The default model revision is GPT-2 `607a30d783dfa663caf39e06633721c8d4cfcd7e`. The source corpus must match the development corpus excluded by the frozen validation manifest and remain article-disjoint from its calibration/train/evaluation partitions. Source document identities, input tokens, configuration and artifact hashes are saved. Final utility is freshly measured, not copied from historical cells. No matched utility ratio is claimed without a corresponding fresh undefended control.

`run_cell(args, base, encoded, output, partition_manifest=None)` supports orchestrator and tiny-test use. `adaptive_final_inference(args, root, entry, budgets)` is the explicit attack adapter. The historical PR35 driver is unchanged and should not be invoked through its tb-cotrain-skipping loop for these new artifacts.

Tests use a tiny GPT-2 on CPU and actually train two victim steps with two encoder updates. They verify matching final victim/head weights, exact checkpoint replay of every inference wire, unchanged observation state and gradients, finite nonzero gradients through the corrected attack, artifact-tamper refusal and rejection of training-phase inputs. Five tests pass; no GPU experiment was launched during implementation. Test/prototype failures remain in pytest's temporary output and production failures get an evidence-sealed `failure.json` once training begins.

Limitations: GPT-2 only; the adapter's default held-out windows are development data, not the untouched confirmatory corpus. This phase/access change provides additional evidence, not a reconstruction of historical capture-time encoders. Full victim checkpoints add approximately one model's storage per cell. Fresh undefended controls and production canary/budget checks belong to the surrounding orchestration.
