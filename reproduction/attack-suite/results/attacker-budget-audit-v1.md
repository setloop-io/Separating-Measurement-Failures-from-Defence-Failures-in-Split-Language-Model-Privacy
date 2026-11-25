# Attacker budget-execution audit

schema: `llm_attacker.budget_audit.v1`
verdict: **PASS**

| tree | cells | PASS | FAIL | UNCHECKABLE |
|---|---|---|---|---|
| pilot-defended-v1 | 30 | 30 | 0 | 0 |
| bottleneck-sweep-v1 | 114 | 96 | 0 | 18 |

No budget shortfalls or counter mismatches found.

## UNCHECKABLE cells

- `bottleneck-sweep-v1:whitebox/split3/seed-11/tb-cotrain-64` — status=NOT_EVALUABLE: tb-cotrain co-trains the encoder DURING victim finetuning (splitbench interleaves a minimax encoder update every tb_cotrain_interval steps), but only the FINAL co-trained encoder checkpoint is saved. 
- `bottleneck-sweep-v1:whitebox/split3/seed-11/tb-cotrain-128` — status=NOT_EVALUABLE: tb-cotrain co-trains the encoder DURING victim finetuning (splitbench interleaves a minimax encoder update every tb_cotrain_interval steps), but only the FINAL co-trained encoder checkpoint is saved. 
- `bottleneck-sweep-v1:whitebox/split3/seed-11/tb-cotrain-256` — status=NOT_EVALUABLE: tb-cotrain co-trains the encoder DURING victim finetuning (splitbench interleaves a minimax encoder update every tb_cotrain_interval steps), but only the FINAL co-trained encoder checkpoint is saved. 
- `bottleneck-sweep-v1:whitebox/split3/seed-22/tb-cotrain-64` — status=NOT_EVALUABLE: tb-cotrain co-trains the encoder DURING victim finetuning (splitbench interleaves a minimax encoder update every tb_cotrain_interval steps), but only the FINAL co-trained encoder checkpoint is saved. 
- `bottleneck-sweep-v1:whitebox/split3/seed-22/tb-cotrain-128` — status=NOT_EVALUABLE: tb-cotrain co-trains the encoder DURING victim finetuning (splitbench interleaves a minimax encoder update every tb_cotrain_interval steps), but only the FINAL co-trained encoder checkpoint is saved. 
- `bottleneck-sweep-v1:whitebox/split3/seed-22/tb-cotrain-256` — status=NOT_EVALUABLE: tb-cotrain co-trains the encoder DURING victim finetuning (splitbench interleaves a minimax encoder update every tb_cotrain_interval steps), but only the FINAL co-trained encoder checkpoint is saved. 
- `bottleneck-sweep-v1:whitebox/split3/seed-33/tb-cotrain-64` — status=NOT_EVALUABLE: tb-cotrain co-trains the encoder DURING victim finetuning (splitbench interleaves a minimax encoder update every tb_cotrain_interval steps), but only the FINAL co-trained encoder checkpoint is saved. 
- `bottleneck-sweep-v1:whitebox/split3/seed-33/tb-cotrain-128` — status=NOT_EVALUABLE: tb-cotrain co-trains the encoder DURING victim finetuning (splitbench interleaves a minimax encoder update every tb_cotrain_interval steps), but only the FINAL co-trained encoder checkpoint is saved. 
- `bottleneck-sweep-v1:whitebox/split3/seed-33/tb-cotrain-256` — status=NOT_EVALUABLE: tb-cotrain co-trains the encoder DURING victim finetuning (splitbench interleaves a minimax encoder update every tb_cotrain_interval steps), but only the FINAL co-trained encoder checkpoint is saved. 
- `bottleneck-sweep-v1:whitebox/split6/seed-11/tb-cotrain-64` — status=NOT_EVALUABLE: tb-cotrain co-trains the encoder DURING victim finetuning (splitbench interleaves a minimax encoder update every tb_cotrain_interval steps), but only the FINAL co-trained encoder checkpoint is saved. 
- `bottleneck-sweep-v1:whitebox/split6/seed-11/tb-cotrain-128` — status=NOT_EVALUABLE: tb-cotrain co-trains the encoder DURING victim finetuning (splitbench interleaves a minimax encoder update every tb_cotrain_interval steps), but only the FINAL co-trained encoder checkpoint is saved. 
- `bottleneck-sweep-v1:whitebox/split6/seed-11/tb-cotrain-256` — status=NOT_EVALUABLE: tb-cotrain co-trains the encoder DURING victim finetuning (splitbench interleaves a minimax encoder update every tb_cotrain_interval steps), but only the FINAL co-trained encoder checkpoint is saved. 
- `bottleneck-sweep-v1:whitebox/split6/seed-22/tb-cotrain-64` — status=NOT_EVALUABLE: tb-cotrain co-trains the encoder DURING victim finetuning (splitbench interleaves a minimax encoder update every tb_cotrain_interval steps), but only the FINAL co-trained encoder checkpoint is saved. 
- `bottleneck-sweep-v1:whitebox/split6/seed-22/tb-cotrain-128` — status=NOT_EVALUABLE: tb-cotrain co-trains the encoder DURING victim finetuning (splitbench interleaves a minimax encoder update every tb_cotrain_interval steps), but only the FINAL co-trained encoder checkpoint is saved. 
- `bottleneck-sweep-v1:whitebox/split6/seed-22/tb-cotrain-256` — status=NOT_EVALUABLE: tb-cotrain co-trains the encoder DURING victim finetuning (splitbench interleaves a minimax encoder update every tb_cotrain_interval steps), but only the FINAL co-trained encoder checkpoint is saved. 
- `bottleneck-sweep-v1:whitebox/split6/seed-33/tb-cotrain-64` — status=NOT_EVALUABLE: tb-cotrain co-trains the encoder DURING victim finetuning (splitbench interleaves a minimax encoder update every tb_cotrain_interval steps), but only the FINAL co-trained encoder checkpoint is saved. 
- `bottleneck-sweep-v1:whitebox/split6/seed-33/tb-cotrain-128` — status=NOT_EVALUABLE: tb-cotrain co-trains the encoder DURING victim finetuning (splitbench interleaves a minimax encoder update every tb_cotrain_interval steps), but only the FINAL co-trained encoder checkpoint is saved. 
- `bottleneck-sweep-v1:whitebox/split6/seed-33/tb-cotrain-256` — status=NOT_EVALUABLE: tb-cotrain co-trains the encoder DURING victim finetuning (splitbench interleaves a minimax encoder update every tb_cotrain_interval steps), but only the FINAL co-trained encoder checkpoint is saved. 

## Tolerances

- `head_forward_calls`, `head_backward_calls`, `n_sequences`, `steps_per_token`, `candidate_count`, `beam_width`, `steps_per_wire`: exact integer match, zero tolerance.
- `head_forward_token_positions` and `vocabulary_dot_products` depend on prefix reuse and candidate-scoring internals; checked only for presence and positivity.
