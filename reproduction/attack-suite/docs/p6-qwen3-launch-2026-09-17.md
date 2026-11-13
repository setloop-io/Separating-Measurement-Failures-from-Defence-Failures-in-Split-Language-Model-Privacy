# Qwen3 P6 launch record — 17 September 2026

The corrected Qwen3-0.6B grid is running on **koula**, isolated checkout `/home/geo/llm-attacker-p6-qwen-20260917`. Queue PID **562873**, grid-driver PID **565319**. The full matrix started **14:23:15 UTC** after both excluded canaries passed full cell/evidence verification.

- Trained span6/start1/D1 canary: **51.125 s**.
- All-local control canary: **90.569 s**.
- Full matrix: **192 trained cells + eight controls**.
- Conservative projection: **4.392 GPU-hours**, using 1.5× measured cell times; maximum queue duration **24 hours**, including canaries and analysis.

The source/protocol freeze is commit `875cb2c0dc0cbec50e41bc3a2f4edc0a478cd2db`. Protocol SHA-256 `7996bfe38d0a4000796cf6ca51eabd62f0a3198384fd09801df2098a230fae49`; retokenized input SHA-256 `44e67d3f140ffe324126132a0c4a510c229169c76b3042bec105a1a8be6c86ac`. Qwen uses its own tokenizer on the exact original source-document partitions; calibration selected Qwen token properties 16, 320 and 264 with document counts 16, 15 and 18 of 32. No GPT-2 token IDs were reused.

Eight corrected-P6 tests pass. Qwen analyzer differs from the tested generic analyzer only in its module-root path and the analytic layer-fraction denominator (28 instead of 12). Actual Qwen canaries additionally verify the real-model splitter, training, held-out observation, VMI and property paths. Environment: Python 3.12, torch 2.11.0+cu128, transformers 5.15.0, NVIDIA GB10.

Local launch evidence is retained in `results/p6-qwen-launch-20260917/`, including both canary cell hash inventories, excluded-canary completion marker, budget projection and environment. These are **launch-time snapshots**, not full-grid completion records. The active status on koula is `results/qwen-queue-status.json`; the full-grid marker is `results/qwen-grid-v1/grid-complete.json`. A complete grid triggers strict verification and frozen analysis automatically, producing `results/qwen-grid-v1.analysis.json`. Failed cells stop the queue and remain preserved; no automatic retries. No result or paper conclusion is claimed before completion.

Any following koula GPU job must wait until the queue's terminal status (`complete`, `failed` or `budget_blocked`) and process exit. This prevents overlap with grid execution or its postprocessing.
