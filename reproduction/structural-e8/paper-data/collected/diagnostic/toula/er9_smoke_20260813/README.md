# E-R9 two-node smoke — diagnostic disposition

Toula and Koula completed the `plain` and `full1` training cells at direct RTT.
The run is **not accepted evidence** because the required E-R8
approximate-alignment attacker crashed before producing its artifact:

```text
NameError: name 'seq_len' is not defined
```

The completed known-plaintext evaluation is nevertheless useful diagnostic
information. The `full1` arm reached 164 rotation epochs, but mean per-epoch
top-1 recovery was 21.52% and pooled cross-epoch recovery was 39.06%, versus a
9.22% label-free band. This does not support the intended privacy gate. The
utility gate did not fail: full1 evaluation loss was 2.596 versus 4.258 for the
matched plain smoke cell.

Only aggregate JSON and logs are retained. Raw tensor captures, private corpus
text, checkpoints, and secret material are excluded.
