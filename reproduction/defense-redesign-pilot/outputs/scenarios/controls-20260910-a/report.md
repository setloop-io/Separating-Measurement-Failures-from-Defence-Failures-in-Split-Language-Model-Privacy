# Engineering scenario report: controls-20260910-a

Completed 36/36 scenarios; operational failures: 0.
Execution code: `a8dff2126832f5fbb2048437c1e0af6a646082a9`. Source archive and artifact hashes verified.

These records are engineering diagnostics. Privacy remains **undetermined**.

| Family | View | Null band passes | Positive floor passes |
|---|---|---|---|
| next_byte_structure | forward_and_metadata | 3/3 | 6/6 |
| next_byte_structure | forward_values | 3/3 | 6/6 |
| next_byte_structure | inbound_gradients | 3/3 | 6/6 |
| next_byte_structure | outbound_gradients | 3/3 | 6/6 |
| next_byte_structure | quantizer_metadata | 3/3 | 6/6 |
| next_byte_structure | returned_values | 1/3 | 6/6 |
| sequence_property | forward_and_metadata | 1/3 | 6/6 |
| sequence_property | forward_values | 1/3 | 6/6 |
| sequence_property | inbound_gradients | 0/3 | 6/6 |
| sequence_property | outbound_gradients | 0/3 | 6/6 |
| sequence_property | quantizer_metadata | 0/3 | 6/6 |
| sequence_property | returned_values | 1/3 | 6/6 |

Null checks are counted once per record; positive checks are counted per dose.
Every record remains unqualified, including records whose numerical checks pass.
The small P0 panels, synthetic control support, and unfrozen scientific calibration
prevent qualification. Preserve failed nulls and evaluate larger independent panels
and a justified control-domain policy before freezing; do not loosen gates post hoc.

Omitted qualification: active/adaptive attacks, state/optimizer, general joint tensors,
history/cross-run attacks, membership, timing, and unrestricted semantics.

See [summary](summary.json), [source record](source.json), and [verification](verification.json).
