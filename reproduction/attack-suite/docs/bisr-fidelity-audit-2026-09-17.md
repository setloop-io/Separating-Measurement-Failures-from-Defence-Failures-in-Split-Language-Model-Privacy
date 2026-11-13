# BiSR fidelity audit — 17 September 2026

**Verdict: NOT_ESTABLISHED.** Historical B1 runs completed, but they do not establish published-setting reproduction or clean-room implementation fidelity. The v4 prerequisite remains unsatisfied; matched evaluation on frozen v3 captures must not be reported as authorised by a fidelity pass. This is an evidence sufficiency verdict, not a numerical attack failure.

The machine-readable verdict, exact counts and SHA-256 anchors are in `results/phase2-status-2026-09-17/bisr-fidelity-verdict.json`. Recompute with `python3 experiments/audit_bisr_fidelity_20260917.py` (standard library only). The script independently recomputes reference positional accuracy from every saved prediction and verifies the historical aggregate to 1e-12; local accuracy is recomputed from saved cell counts. Local token predictions were not retained by the attack writer, so local ROUGE/METEOR cannot be independently recovered from those counts.

## What the records establish

| Quantity | SplitLLM reference | Local clean-room run |
|---|---:|---:|
| Attacked batches / sequences | 13 / 26 | 12 / 24 |
| Scored tokens | 3,328 | 3,072 |
| BiSR(b+f) positional correct | 3,283 | 3,071 |
| BiSR(b+f) positional accuracy | 98.647837% | 99.967448% |
| Local beam-arm positional accuracy | — | 99.023438% |
| Reference ROUGE-L F1 | 99.222118% | not saved |
| Reference token-set F1 (called `tokacc`) | 99.243659% | not the same metric |

Reference attacks occur at steps 199–203, 399–403, 599–601. Local attacks use 200–204, 400–404, 600–601. Victims were independently generated. Subtracting these rates would therefore not be a paired fidelity contrast. A privacy gate FAIL in the local JSON means successful recovery above its privacy threshold; it is not a BiSR fidelity judgement.

The reference checkout on koula is upstream commit `c376590227f9718077765824af4088665bb78687`, with six tracked files modified and an added missing noise module. Its diff and historical run scripts were copied read-only to `results/phase2-status-2026-09-17/bisr-audit-sources/`. The changes include instrumentation, import/path/GPU compatibility repairs and throttled display metrics. They are inspectable, but the old note's assertion of bit-identical training has not been experimentally verified. The local run records commit `ae37fe7f7d1dd1c6815d76286f4f6fea65204827`; v4 was frozen at `adf4cf688373b3983ebc3d6cbe2c1eebc99a97d2`.

## Why the gate cannot pass

The paper uses a GRU with hidden size 256 and dropout 0.1, three random seeds, and SensiReplaced auxiliary data. Its Table 3 specifies the 6–27 split and reports GPT2-large/WikiText BiSR ROUGE-L 99.79 ± 0.29%. Section 4.1 specifies five attacked batches every 200 fine-tuning steps, capped at 600. These are published experimental settings and seed dispersion, not a supplied replication acceptance interval. [Primary paper, v2](https://arxiv.org/html/2409.00960v2).

The historical reference instead used raw CNN/DailyMail articles as `sani_gpt4`, five SIP epochs, and seed 42 alone. Its own notes record the intended reference training budget as twenty epochs. This is a material dataset and training substitution. The clean-room run uses a per-position MLP, both forward and backward observations at seam 6, different victim training and full-length auxiliary windows. The reference has separate cuts 6 and 27 and truncated auxiliary articles. These implementation and exposure differences cannot be resolved by high accuracy alone.

No operational, predeclared tolerance was found in the frozen v4, open-commitments plan, fleet runbook, attack docstring, remote FIDELITY-NOTES, or run scripts. FIDELITY-NOTES says a tolerance band uses the paper's standard deviation but specifies no multiplier, metric family, direction, confidence method or decision rule. Its later “within ±2.9 points” statement is descriptive and follows the results. Neither that value nor v4's 0.02 practical margin for the separate matched comparison can be retrofitted as B1 acceptance. Even selecting a standard-deviation band now would be post-hoc.

There is also a v4 textual conflict: `arms.bisr_bre_bf.description` says forward then backward. The implementation docstring and historical reference describe backward then forward. Preserve the frozen file and resolve this through an additive clarification before prospective execution.

## Dataset and compute blockers

The only configured `koula:~/splitllm-ref/data/sfl/datasets/sensi/sensi.csv` is the documented raw-article substitute. No original SensiReplaced dataset was found in the configured data tree. The upstream download instructions contain no SensiReplaced download; its sanitizer directory contains code, a source-dataset link and prompts. Its README describes generating GPT-4 responses and training a sanitising model. Targeted public searches found no primary public release of the original processed dataset. This is a bounded search, not proof that no copy exists anywhere. [Upstream sanitizer instructions](https://github.com/StupidTrees/SplitLLM/tree/c376590227f9718077765824af4088665bb78687/experiments/sanitizer).

Obtaining the original processed rows and split manifest from the authors is the direct remedy. Regenerating replacements with a present-day model creates another dataset and must be labelled accordingly. No external contact or paid API call was made. The existing five-epoch training took about 2h07m according to the saved notes: a linear estimate for three twenty-epoch runs is approximately **25.4 GPU-hours of SIP training**, plus victim fine-tuning, attacks and evaluation. This is an estimate, not a measured completion promise, and can exhaust the existing 10–30 GPU-hour envelope before the matched grid.

## Prepared prospective reference configuration

Use an isolated fresh reference output and checkpoint prefix per seed, on the pinned upstream commit with the audited compatibility patch. Do not reuse `run_b1_gpt2large_wikitext.sh` unchanged: it clears old journals and its checkpoint prefix can reuse an existing inverter. Before launch, bind the original dataset, model/tokenizer revisions, dependencies and patch by hashes; record the actual attack cadence including upstream's off-by-one trigger; and adopt an operational prospective tolerance with a scientific justification independent of the observed results. Three seeds are required for the published averaging structure; 42/43/44 below are proposed prospective seeds, not asserted to be the authors' original seeds.

For each `SEED` in 42, 43 and 44, the reference command is the following, run from `~/splitllm-ref/experiments/scripts/pipeline`, with `SFL_JOURNAL_PATH` and `SFL_ATTACK_JOURNAL_PATH` set to never-before-used files and `WANDB_MODE=offline`:

```bash
~/splitllm-ref/.venv/bin/python ../py/sim_with_attacker.py \
  --noise_mode none --noise_scale 0.0 --model_name gpt2-large \
  --split_points 6-27 --global_round 1 --client_num 1 --seed "$SEED" \
  --dataset wikitext --dataset_label train \
  --data_shrink_frac 0.08 --test_data_shrink_frac 0.3 \
  --evaluate_freq 200 --client_steps 650 --client_epoch 1 --max_global_step 602 \
  --lora_at_top True --lora_at_trunk True --lora_at_bottom True \
  --collect_all_layers True --batch_size 2 \
  --exp_name B1-prospective --case_name "original-seed$SEED" \
  --sip_dataset sensireplaced --sip_prefix "b1-original-seed$SEED" \
  --sip_b2tr_enable True --sip_b2tr_layer -1 --sip_tr2t_enable False \
  --sip_train_frac 1.0 --sip_training_epochs 20 --sip_training_checkpoint_freq 5 \
  --gma_enable True --gma_lr 0.09 --gma_beta 0.85 --gma_epochs 32 --gma_init_temp 1.2 \
  --sma_enable True --sma_epochs 800 --sma_lr 0.01 --sma_wd 0.01 \
  --gsma_enable True --gsma_epochs 800 --gsma_lr 0.01 --gsma_wd 0.01 \
  --tag_enable False --eia_enable False --attacker_freq 200 --attacker_samples 5
```

This preserves the known historical pipeline's 602-step cap and thirteen attacked batches explicitly; it must be reconciled with the paper's 600-step prose in the prospective specification, not silently described as exact equivalence. GRU architecture follows upstream defaults; capture resolved arguments before training to verify GRU-256/dropout 0.1, learning rates, LoRA settings and twenty epochs. The original dataset and tolerance are missing, so this is prepared configuration, **not a launch-ready frozen fidelity protocol**.

For our implementation, first resolve whether the intended target is faithful dual-cut BiSR or a declared single-seam adaptation. Align victim snapshots and canonical sample IDs, auxiliary preprocessing, decoder/training budgets, optimiser order and metrics. Persist per-token predictions with hashes for SIP, b, f and bf. Run the prospective validation once with all failures retained, then evaluate v4's matched captures only after the fidelity gate is established. No expensive jobs, frozen protocols, manuscripts or release-stage files were changed by this audit.

## Five-node follow-up

A subsequent read-only search on toula, koula, soula, voula and sasa found no additional SensiReplaced file. The sole candidate was koula's configured CSV: all 23,665 `sani_gpt4` values exactly equal `content`, confirming the substitute (SHA-256 `5da54ac04f01937061a7aa5f5a55aec3d4f28b71375d00f6f56815667637183d`). B1 readiness is unchanged. Search boundaries and recovered external programme evidence are documented in `docs/external-p0p5-recovery-2026-09-17.md`.

## Prospective specification resolved

The additive [fidelity specification](bisr-fidelity-specification-2026-09-17.md)
now defines separate published-setting replication, implementation conformance
and single-seam adaptation routes, including a prospective equivalence criterion.
It resolves the backward-then-forward order and excludes beam initialisation
from fidelity arms. Frozen v4 and this historical NOT_ESTABLISHED verdict remain
unchanged. Specification completion is not implementation validation or launch
admission; the companion JSON records the remaining requirements.
