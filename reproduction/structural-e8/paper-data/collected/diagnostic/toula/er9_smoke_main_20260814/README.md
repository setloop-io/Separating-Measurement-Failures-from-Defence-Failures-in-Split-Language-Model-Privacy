# E-R9 corrected-budget smoke — diagnostic evidence

Toula and Koula completed the two-cell E-R9 smoke at commit `c5dfe00` using
the corrected rotation-budget remainder accounting. Both training cells, the
known-plaintext evaluator, the E-R8 approximate-alignment attacker, cleanup,
and the structural/utility validator completed.

The run is diagnostic rather than successful privacy evidence. Full1 retained
utility (`eval_loss` 2.5259 versus 4.2580 plain), but mean per-epoch recovery
was 21.74%, pooled cross-epoch recovery was 29.69%, and approximate-alignment
recovery averaged 15.21%, all above the 9.71% label-free reference. E-R8
reported no accepted solved epoch; that narrower gate does not erase the
partial-recovery measurements.

Correct `B=64` accounting triggered 164 dense cloud-weight folds. Each fold
took approximately 205–215 seconds, making this implementation unsuitable for
a larger matrix until folding is optimized or the design changes.

This directory contains aggregate JSON, the per-epoch JSONL attack journal,
the aggregate Markdown summary, and the execution log. It excludes raw tensor
captures, seeds, checkpoints, model tensors, and private corpus text.
