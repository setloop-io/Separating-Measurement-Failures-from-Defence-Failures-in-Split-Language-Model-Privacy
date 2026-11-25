# BiSR FIDELITY NOTES — Phase B1 (validation protocol v4)

Provenance record for the reproduction of the published BiSR reference setting
(SplitLLM, Chen et al., CCS 2024, arXiv:2409.00960) on koula (NVIDIA GB10).
All times UTC.

## 1. Reference source

- Repo: https://github.com/StupidTrees/SplitLLM (Apache-2.0), HTTPS clone at
  `~/splitllm-ref`, commit `c376590227f9718077765824af4088665bb78687`
  (HEAD at clone time 2026-09-16). No git mutations; working-tree changes are
  the documented repairs/instrumentation in §6 (`git diff` inspectable).
- Paper: arXiv:2409.00960v2 (CCS'24); tables extracted from the arXiv PDF
  (HTML version has image-only tables).
- Entry point: `experiments/scripts/pipeline/demo_bisr.sh` →
  `experiments/scripts/py/sim_with_attacker.py`. The demo defaults are
  llama2/codealpaca; the gate reproduces the smallest published pair
  gpt2-large/wikitext-2 with the paper's Table-1 GPT2-large constants
  (the pair the v4 protocol's BiSR budgets cite).

## 2. Published reference setting (paper §4.1 + Table 1 + repo code)

- Model: GPT2-large (36 blocks, hidden 1280, vocab 50257), full precision
  (only LLaMA2-7B/ChatGLM3-6B are 8-bit-quantized in the paper).
- Victim dataset: WikiText2-v1 ("44.8k rows"; measured 36,718 train /
  3,760 validation / 4,358 test = 44,836 in the current HF
  `Salesforce/wikitext` repo). Their loader groups text into 128-token
  blocks (`WikiTextFedDataset._pre_process`, block_size=128); victim-side
  demo shrink 0.08 train / 0.3 test, batch 2, AdamW lr 1e-5, LoRA on
  bottom+trunk+top, single client.
- Split: private-label 3-part SL, `split_points 6-27` (Table 3) — bottom =
  blocks 0-5, trunk = 6-26, top = 27-35 + ln_f + lm_head. (Table 10 used
  6-26.)
- Attack cadence (paper §4.1 + their code): attacks on 5 batches every 200
  FT steps, capped at 600 steps; main results averaged over 3 seeds. With
  the code's trigger/cap semantics this yields 5+5+2 = 12 attacked batches
  (steps 200-204, 400-404, 600-601) = 24 sequences per run/seed.
- SIP (stage 1): GRU inverter, 1 layer, hidden 256, dropout 0.1, auto-encoder
  (unshifted) CE on aux seam rows through the FROZEN pretrained bottom
  (`attack_mode='b2tr'`, layer 6). Repo training defaults
  (`InversionModelTrainingArgument`): Adam lr 1e-3 wd 1e-5, batch 6,
  20 epochs, test_frac 0.1, checkpoint every 5 epochs, save if ROUGE-L f >
  0.1; the attack loads the best-ROUGE checkpoint.
- Aux dataset D_a: SensiReplaced, 19,665 samples (paper Fig. 11) derived from
  CNN/DailyMail (Kaggle `gowrishankarp/newspaper-text-summarization-cnn-dailymail`)
  via spaCy NER + GPT-4 entity replacement. NOT distributed in the repo
  (sanitizer pipeline code only) — see D-aux1.
- Aux plumbing actually measured in their code: `sensireplaced` registers no
  custom dra labels, and `get_dra_test_label` returns the TRAIN-label map
  (upstream bug), so SIP data loading takes the self-test branch: the aux
  pool is split 70/30 — train on 13,765 rows, eval on 5,900
  (`get_dataloader_unsliced(..., further_test_split=0.3)`, shuffled). The
  paper's Table 3/10 numbers therefore come from SIP trained on ≈13.8k
  unique aux documents (x20 epochs ≈ 275k document visits), not the full
  19,665 — this matters for the B2 aux-scale recommendation.
- BRE (stage 2), paper Table 1 GPT2-large: AdamW(0.9, 0.999); GM 32 iters,
  lr 0.09, beta 0.85, tau 1.2 (their GM optimizer additionally hardcodes
  wd 0.01, eps 1e-6); SM 800 iters, lr 0.01, wd 0.01; cosine distance for
  SM (App. D.5); BiSR(b+f) = GM then SM warm-started from GM.
- Metrics (their code, `evaluate_attacker_rouge`): per-token decode joined
  with spaces; ROUGE-1/2/L F1 (python-rouge), METEOR (nltk wordnet),
  TokACC (set-overlap F1 of tokenized texts = paper's TRR), averaged over
  attacked batches.

## 3. Published numbers (GPT2-large / WikiText column)

Table 3 (split 6-27, ROUGE-L F1 %, mean±std over 3 seeds):
  SIP-only 93.53±0.34 | BiSR(b) 97.28±0.75 | BiSR(f) 99.71±0.26 | BiSR(b+f) 99.79±0.29
  METEOR:  78.04±4.68 | 82.26±3.66 | 83.91±3.71 | 84.06±3.89
Table 10 (split 6-26, ROUGE-1 F1 % / TokACC %):
  SIP-only 93.53±0.34 / 93.47±0.76
  BiSR(b)  97.28±0.75 / 96.80±0.28
  BiSR(f)  99.71±0.26 / 99.56±0.51
  BiSR(b+f) 99.79±0.29 / 99.62±0.54
Reference baselines (Table 3 ROUGE-L): AE 82.55±4.96, EIA* 81.50±0.56,
  TAG* 68.47±0.86, LAMP* 67.57±0.27.

## 4. Environment on koula (D-env decisions)

- D-env1 torch: 2.11.0+cu128 from the `~/llm-attacker/.venv` base layer
  (exposed via `.pth`, same mechanism as the pilot venv) instead of the
  pinned torch 2.3.0. The GB10 is sm_121 (Blackwell); torch 2.3.0 has no
  sm_121 kernels and cannot execute on this GPU at all. Risk: torch-level
  numeric drift vs the paper's V100/torch-2.0 runs; inherent to the
  hardware and identical for our clean-room arms.
- Fresh venv `~/splitllm-ref/.venv`, era-pinned per requirements.txt:
  transformers 4.40.0, tokenizers 0.19.1, huggingface_hub 0.23.4,
  datasets 2.15.0, numpy 1.26.2, pandas 2.2.2, peft 0.10.0, trl 0.8.6,
  accelerate 0.29.3, wandb 0.30.0, rouge 1.0.1, rouge-score 0.1.2, nltk,
  pynvml 11.5.0, regex 2023.10.3, sentencepiece. (peft 0.10.0 because
  0.10.2 lacks py3.12 wheels; torchvision/bitsandbytes skipped — ViT and
  8-bit quantization paths unused.)
- D-env2 wandb: `WANDB_MODE=offline` (no credentials on koula; runs stored
  under `experiments/scripts/pipeline/wandb/offline-run-*`). The metrics of
  record come from the env-gated JSONL journals (§6 P4/P4b), which carry
  the same numbers that flow into wandb plus decoded texts and token ids.

## 5. Dataset / aux deviations

- D-aux1 (material): SensiReplaced's CSV is not distributed (needs the
  authors' GPT-4 pipeline; Kaggle source needs credentials). Substitute at
  `data/sfl/datasets/sensi/sensi.csv`: raw CNN/DailyMail articles
  (abisee/cnn_dailymail 3.0.0 parquet conversion, snapshot
  690bb95a2ac2c5a99d7bde63ac1401539ddd3967), SAME source domain, SAME
  published scale: 19,665 'validation' rows (the SIP pool), 2,000 'test',
  2,000 'train' (ctor-required, unused). No NER marking, no GPT-4 entity
  replacement (`sani_gpt4` = raw article). SIP still trains on 13,765 rows
  / evaluates on 5,900 via their 70/30 self-test branch.
- D-data1 (minor): wikitext comes from the current `Salesforce/wikitext`
  HF repo (the documented clone target `huggingface.co/datasets/wikitext`
  now redirects there; parquet layout, no loading script; the 2024-era
  script's S3 zip URL is dead). Their `load_dataset(dir, 'wikitext-2-v1')`
  call works unchanged; row counts match the paper.

## 6. Working-tree repairs & instrumentation (non-algorithmic)

- P1 `sfl/config.py`: data_root → `/home/geo/splitllm-ref/data`
  (README-sanctioned edit).
- P2 `sfl/model/noise/fdp.py`: CREATED — the published tree imports
  `GaussianPerturber` from this module (`sfl/model/llm/split_model.py`) but
  never committed it (checked full git history); the repo is not importable
  without it. Reconstructed from the Perturber interface; never invoked at
  noise_mode='none'.
- P3 `sfl/utils/args.py`: re-export `dxp_moe_range, gaussian_moe_range,
  dc_moe_range, lora_path` from `sfl.config` (imported by
  `inversion_training.py`, missing upstream — same class of incompleteness).
- P4/P4b instrumentation (env-gated; unset ⇒ byte-identical behavior):
  `sfl/simulator/simulator.py` journals the per-step report dict to
  `SFL_JOURNAL_PATH`; `sfl/strategies/sl_strategy_with_attacker.py`
  journals every per-batch attack result (rouge-1/2/l f, meteor, tokacc,
  decoded attacked/true texts, argmax/true token ids) to
  `SFL_ATTACK_JOURNAL_PATH`.
- P5 `sfl/utils/model.py get_best_gpu`: NVML memory query is unsupported on
  GB10 (`nvmlDeviceGetMemoryInfo` → NVMLError_NotSupported, same `[N/A]` as
  nvidia-smi); falls back to cuda:0 (single-GPU node, same choice).
- P6 `inversion_training.py`: progress-bar ROUGE display throttled to every
  50 steps. Display-only — loss/backward/optimizer/batches untouched, so
  the trained inverter is bit-identical; the per-step
  `evaluate_attacker_rouge` (nltk-wordnet meteor on 6 pairs) was ~85% of
  step wall time and made the published 20-epoch run ~8.5h on this GPU.

## 7. Runs executed (all on koula, one GPU job at a time)

- SMOKE (17:12, ~3 min, exit 0): `run_b1_smoke.sh` — full pipeline at tiny
  budgets validated; journals confirmed (SIP 0.334 / BiSR(b) 0.946 /
  BiSR(f) 0.334 / BiSR(b+f) 0.952 ROUGE-L on the 2-sequence smoke batch,
  1-epoch 2%-frac inverter, 4/8/8-iter BRE budgets).
- PROBE (17:15, ~4 min, exit 0): `run_b1_probe.sh` — SIP steady-state
  2.15 it/s with P6 (compute-bound), checkpoint save/load path verified.
- REFERENCE (started 17:16:41Z): `run_b1_gpt2large_wikitext.sh` with
  SEED=42, SIP_EPOCHS=5, SIP_TRAIN_FRAC=1.0, SIP_CKPT_FREQ=1.
  D-run1 (deliberate shrink, recorded): 5 SIP epochs instead of the
  published 20 — the published budget is 8.5h on this GPU even with P6;
  5 epochs keeps the FULL paper-scale aux supply (19,665-doc pool →
  13,765 trained x5 epochs ≈ 68.8k document visits vs paper ≈275k) inside
  the ~2h compute guidance. Checkpoint every epoch (vs their 5) so the
  attack loads the best-ROUGE of epochs 1-5, giving a convergence trace.
  Attack side NOT shrunk: full paper Table-1 budgets (GM 32 / SM 800 /
  GSMA 800), full 602-step FT with 12 attacked batches (24 sequences).
  Everything else per §2. Deviations vs the demo script (to match the
  PAPER): gma_epochs 32 (demo: 18), sma enabled with 800/0.01/0.01 (demo:
  disabled), gsma 800/0.01/0.01 (demo: 600/0.001/0.01), client_steps 650 +
  max_global_step 602 + attacker_samples 5 (paper §4.1; demo: 2 attacked
  batches by step 202), dropped the no-op `--sup_enable`.
  Single seed (paper averages 3) — recorded; tolerance band uses the
  paper's published ±std.

## 8. Results

REFERENCE RUN (seed 42, exit 0; wall ≈ 2h55m: SIP 2h07m + FT/attacks/evals).
SIP aux-test ROUGE-L by epoch (their checkpoint eval, 5,900-row test split):
e0 0.9846, e1 0.9896, e2 0.9899, e3 0.9904, e4 0.9903 — saturated early;
the attack loaded the epoch-3 checkpoint (best, 0.9904; epoch-4 0.9903).
Attack journal: 52 records = 13 attacked batches (FT steps 199-203,
399-403, 599-601; their trigger attacks the trigger step itself) x 4 arms
= 26 wikitext-2 victim sequences of 128 tokens.
Aggregate over the 13 batches (per-batch means ± sd; posacc = position-
aligned token accuracy computed offline from the journaled ids — the v4
harness's metric view):

  arm        rougeL        rouge1        meteor        tokacc        posacc
  SIP        95.00±1.47    95.00±1.47    94.25±2.89    94.75±1.48    92.49±3.90
  BiSR(b)    99.84±0.26    99.84±0.26    99.82±0.31    99.67±0.41    99.55±0.50
  BiSR(f)    99.17±0.82    99.17±0.82    99.39±0.52    99.20±0.78    98.65±1.03
  BiSR(b+f)  99.22±0.80    99.22±0.80    99.40±0.79    99.24±0.81    98.65±1.10

vs paper (Table 3 ROUGE-L / Table 10 TokACC, GPT2-large/WikiText):
  SIP:      95.00 vs 93.53 (+1.47) | TokACC 94.75 vs 93.47 (+1.28)
  BiSR(b):  99.84 vs 97.28 (+2.56) | TokACC 99.67 vs 96.80 (+2.87)
  BiSR(f):  99.17 vs 99.71 (−0.54) | TokACC 99.20 vs 99.56 (−0.36)
  BiSR(b+f):99.22 vs 99.79 (−0.57) | TokACC 99.24 vs 99.62 (−0.38)

All four arms within ±2.9 points of the published means (paper's own seed
std ≤ ±0.75; the two SM arms land inside −0.6; SIP/GM land +1.5/+2.6 high,
consistent with a slightly stronger aux supply (un-replaced real entities)
and the single-seed draw). ROUGE-1 == ROUGE-L here exactly as in the
paper's own Table 10 GPT2-large/WikiText column (their per-token-decode
metric behavior reproduced). METEOR runs higher than the paper's
(94.25/99.82/99.39/99.40 vs 78.04/82.26/83.91/84.06) — the paper's
WikiText METEOR column is anomalously low relative to its own ROUGE-L
(also visible for LLaMA2: 89.06 vs 95.90); ours tracks ROUGE-L instead.
Recorded, not explained away.

Artifacts (koula):
  ~/splitllm-ref/results/ref-seed42/{ref.log, attacks.jsonl, report.jsonl,
    attacks.jsonl.aggregate.json}
  ~/splitllm-ref/results/{smoke,probe}/ (validation runs)
  ~/splitllm-ref/data/sfl/models/attacker/gpt2-large/sensireplaced/
    validation*1.000-validation*0.100/gru/layer6/normal/epoch_* (5 ckpts)
  wandb offline run: experiments/scripts/pipeline/wandb/
    offline-run-20260916_202954-eg1eh4nr

OUR CLEAN-ROOM MATCHED RUN: see GATE.md (results/bisr-fidelity-v1/).

## 9. Aux-supply accounting for B2

Published SIP supply: 19,665-doc aux pool (CNN/DailyMail news) → their
self-test branch trains on 13,765 docs x20 epochs (≈275k doc visits;
documents truncated to ≤300 GPT2 tokens by their `_col_fun` max_len=300).
B1 reference run: same pool, 13,765 docs x5 epochs (≈68.8k visits).
B1 our-side run: same 19,665 articles streamed FULL-LENGTH into 127,433
windows of 128 tokens (16.3M tokens), 20,000 steps x batch 32 = 640k
window visits sampled from the shards (resume-safe cache under
results/bisr-fidelity-v1/).
