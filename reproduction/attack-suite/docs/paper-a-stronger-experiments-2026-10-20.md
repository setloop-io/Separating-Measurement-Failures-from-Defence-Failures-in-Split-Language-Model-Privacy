# Paper A stronger experiments — 20 October 2026

The user authorised Toula/Koula and a ceiling of 48 aggregate GPU-hours, including canaries and failures. Both hosts were reachable and idle, each with an NVIDIA GB10. The [prospective study plan](../artifacts/paper-a-stronger-20261020/study-plan.md) specifies a retained-capture search expansion without victim retraining.

Tracked resource experiment: `bc4345b0-5103-4955-8fc0-ff08a42f63dc`, branch `orx/paper-a-stronger-search-resource-canary`, source `e0015336af5d8e03f43052cf20dee4ceca19c7e5`.

- Toula canary: `42b5eadd-eee4-4326-9785-45d2e2f8f685`.
- Koula canary: `f3631218-0e31-400e-a044-00210ed3913b`.

Both calibration-only resource checks completed all ten cases and passed local evidence-chain verification. Their model-stage elapsed times were 171.65 seconds on Toula and 169.10 seconds on Koula, with maximum allocated memory 4.72 GiB. Extrapolating those public-calibration timings to all100 executions gives14.61–14.66 aggregateGPU-hours before overhead; this is not a worst-case bound. See [resource admission](../artifacts/paper-a-stronger-20261020/resource-admission.json).

Original-anchor replay is now running under child experiment `54a58e19-5214-46b5-aa40-79df19187005`, branch `orx/paper-a-stronger-search-anchor-replay`, source `96108c7081f3d2d722a87e426241a293b6d89d0b`.

- Toula anchor run: `1aca3804-38aa-43b7-8ef2-94fc09ba004e`.
- Koula current anchor run: `084c0e74-ee83-44f3-9021-d80b06c42b96` (the initial failed setup is retained below).

Each host has five paired victim anchors (ten physical executions), capped at one hour. Both original attack-seed transcripts are hash-verified references; only one deterministic execution is counted. All40 reference transcripts and their16 source-report bindings passed local preflight. Expansion remains blocked until both complete anchor sets pass; inspect residual-difference flags as well as discrete predictions/candidate lists.

Budget reservation: reserve 25 aggregate GPU-hours for the first batch, including source staging and the retained failed setup attempt; keep 23 hours provisionally available for follow-up. Script limits are 20 minutes per canary, one hour per anchor worker and at most11 hours per expansion worker. Actual allocations and failures are recorded separately; reservations are not measured costs.

Original protocols, published evidence and the40-page candidate remain unchanged. No expanded-search result is claimed yet.

Koula's first anchor attempt `534edc49-610d-4413-8058-ce43e0b55b01` failed after26seconds during reference-file discovery, before decoding. The tracked source archive does not transport historical results. The unchanged source is being supplied with the88required original reports, seals/journals and40anchor prediction files in a separate retained-input staging directory. The attempt is retained and charged to the budget. Toula's first forward anchor completed with exact tokens, candidates and residuals against both original17/29transcripts.

The learned-baseline audit also corrected the reviewer response: the completed single-seam SIP/BRE adaptation already uses these same captures. Avoid rerunning it merely to obtain a learned baseline; original published-setting BiSR, Vec2Text and matched-compute questions remain distinct.

Koula repair run: `084c0e74-ee83-44f3-9021-d80b06c42b96`, same source `96108c7`. Its public heads/observations and original reference reports/transcripts were staged from the Mac; hashes remain unchanged. Both anchor workers are running.

## Bounded unattended continuation

The local controller is running, with live state in [controller-status.json](../artifacts/paper-a-stronger-20261020/controller-status.json). Its source and both CPU analyses were committed before expanded outcomes on isolated branch `research/paper-a-stronger-orchestration-20261020`, commit `dcffc4905ff19a59023583cdc73f6c0df22aaeb8`. [The source lock](../artifacts/paper-a-stronger-20261020/orchestration-source.json) records their hashes and preflight checks. The manuscript working tree and frozen experiment branches were preserved.

The controller waits for both full anchor sets, then requires exact tokens, candidates and residuals against all 40 original references. Only that complete admission permits the fixed 80-cell expansion. All model and CPU jobs launch through OpenResearch under the unchanged entry command. GNU `timeout` enforces each remote phase limit, including termination of its process group. A failure blocks subsequent phases; no automatic retry or outcome-selected variant is permitted.

Each collected report is bound to its ORX run directory, committed plan, replay-helper hash, input hash, binary grant inventory and exact per-host cell assignment. Collection uses fresh run-ID directories. All 100 completed predictions are independently rescored before the fixed ten-contrast token family is analysed with 100,000 crossed victim/document bootstrap draws. Completion also requires verifying the analysis seal and every supporting output hash.

A separate CPU sensitivity step uses the retained GPT-2 tokenizer JSON, revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`, SHA-256 `8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6`. It compares token-ID exactness, decoded exactness, ASCII left-space normalisation and Unicode-whitespace normalisation. Its passage intervals are descriptive and pointwise, outside the ten-contrast token family. No model or download is required. The core report's `decoded_normalisation: NOT_COMPUTED` field refers to that core script alone; the companion sensitivity report must be checked separately before claiming this planned item is complete.

Final outputs, if all stages pass, will be collected under `artifacts/paper-a-stronger-20261020/analysis/` and `normalisation/`. On error, the status file records the reason and known runs; committed remote caps remain enforced. The Mac is kept awake while the bounded controller runs. There is no chat wake-up service in this session; the controller writes durable results without depending on another user prompt. No paper, arXiv, GitHub or Hugging Face publication is part of this batch.


## Completed — 21 October 2026

The fixed batch completed at **04:13 London time on 21 October 2026** (03:13 UTC). Both expanded-search workers completed 40 cells; all 20 original anchors and 80 new search cells were collected. The separately tracked CPU analysis and decoded-normalisation sensitivity both returned COMPLETE. Both analysis seals and all fifteen supporting-output hashes were checked again after completion.

[Complete results](../artifacts/paper-a-stronger-20261020/analysis/REPORT.md) and [decoded-normalisation sensitivity](../artifacts/paper-a-stronger-20261020/normalisation/REPORT.md) retain all configurations and the declared contrast families.

At k=128, beam=2, activation-only token recovery is 98.0981%, compared with 98.1567% for joint decoding: joint-minus-activation 0.0586 points, ten-contrast Bonferroni interval [-0.1501, 0.2673]. At k=64, beam=2 the corresponding difference is 0.9167 points [0.5164, 1.3416]. The original k32/beam2 gain remains 3.1714 points. Original joint-k32 minus activation-k128 is -0.7227 points [-1.1731, -0.3162]; the larger activation search costs more, so this is not an equal-compute comparison. The gradient benefit is budget-dependent; the k128 interval does not establish equivalence.

The 47-page manuscript (`paper/paper_a/main.pdf`, Section 4.7, Table 8; Appendix G, Tables 28–29) and the second-round reviewer response (`docs/paper-a-review-v2-response-2026-10-20.md`) now incorporate these completed results. Original protocols, predictions and test families are unchanged. The completed adaptation and candidate-list reporting fixes remain separate from this 80-cell expansion.

## Step-resolved 500-step live training trajectory paired replay — Completed 21 October 2026

Tracked experiment: `e34bc2f4-3d9a-4d0c-a198-e051ac0db306`, branch `orx/paper-a-step-resolved-500-step-live-training-tra`, source `a35a0c2e04c53d3857a89ef7145568be92629456` (parent `9686b319-17fd-4618-9d08-5f47b71d7a7c`).

- Toula run (`steps 0..247`, `248` steps, `9` cells): `a076ccbe-8831-4773-a80a-dc499c3b3d7a` (`1,292.31 s`, `status: TRAJECTORY_COMPLETE`).
- Koula run (`steps 248..499`, `252` steps, `9` cells): `03403770-d515-4b67-9492-75260823104a` (`1,320.96 s`, `status: TRAJECTORY_COMPLETE`).

This follow-up evaluates paired `backward`, `forward_k32` ($B=2$) and `joint_true_k32` ($B=2$) reconstruction across all 500 live fine-tuning steps of the three retained undefended GPT-2 split-layer-six trajectory captures (`convergence-none-{11,22,33}`, `3 × 500 = 1,500` unique 32-token training steps, `48,000` token positions, executed in `train()` mode with active dropout $p=0.1$). Calibration uses Paper A's frozen 32-document `input.json['calibration']` partition (`sha256: d79ef4fc...`) and `public-101-6.pt`, which has zero exact overlap and zero five-gram near-duplicate overlap ($\text{Jaccard} \ge 0.8$) with all 1,500 trajectory steps.

See [complete trajectory analysis](../artifacts/paper-a-stronger-20261020/trajectory-analysis/REPORT.md). Across all 1,500 steps:

- `backward`: `53.95%` token recovery (`0.00%` exact passages; `55.55%` on positions 2–32, `4.33%` on position 1).
- `forward_k32` ($B=2$): `88.31%` token recovery (`0.33%` exact passages, `0.53%` whitespace-normalised; `91.00%` on positions 2–32, `4.93%` on position 1; `432.7 s` per 500 steps).
- `joint_true_k32` ($B=2$): `94.45%` token recovery (`2.40%` exact passages, `3.40%` whitespace-normalised; `97.34%` on positions 2–32, `5.00%` on position 1; `433.4 s` per 500 steps).
- Paired gradient gain (`joint_true_k32 - forward_k32`): **`+6.15 pp`** (step-paired 95% CI `[+5.88, +6.42]`, three-contrast Bonferroni CI `[+5.82, +6.48]`, crossed three-seed-by-500-step 95% CI `[+5.46, +6.91]`, seed means `+5.92 / +5.58 / +6.94 pp`). Across 100-step bins, the paired gain is `+7.20 pp` (steps 1–100), `+6.71 pp` (steps 101–200), `+5.45 pp` (steps 201–300), `+5.40 pp` (steps 301–400), and `+5.99 pp` (steps 401–500).

These results are incorporated into `paper/paper_a/main.pdf` (Section 4.8, Table 9) and `docs/paper-a-review-v2-response-2026-10-20.md`.

## Defended joint and public-grant adaptive attack evaluation — Completed 21 October 2026

Tracked experiment: `4fc1ac9c-5c95-4042-b3ec-0cdc08b2bc81`, branch `orx/paper-a-defended-joint-and-adaptive-attack-evalu`, source `e4775c3ca76335eb011e85be5814b6fa2cf02b9b` (parent `e34bc2f4-3d9a-4d0c-a198-e051ac0db306`).

- Toula run (`seeds 101..505`, `mid-lite steps 0..247`, `70` held-out cells + `9` trajectory cells): `ec13ebcf-abbb-4f1b-9113-fea60336e84d` (`10,947.28 s`, `status: DEFENDED_REPLAY_COMPLETE`).
- Koula run (`seeds 606..1010`, `mid-lite steps 248..499`, `70` held-out cells + `9` trajectory cells): `aac62ef7-5148-4519-8a29-bc2fb6015114` (`11,028.23 s`, `status: DEFENDED_REPLAY_COMPLETE`).

This follow-up evaluates backward-only, activation-only (`forward_k32`), joint (`joint_true_k32`), and strictly public-grant adaptive decoders at $L=6$, public client copy, $k=32, B=2$ across:
1. `replica-median` (`10` victim seeds `101..1010`, `2,560` held-out passages; PPL `36.62`),
2. `secret-mixup` ($\lambda=0.75$, `10` victim seeds `101..1010`, `2,560` held-out passages; PPL `45.75`),
3. `trained-bottleneck` ($768\to 64$ learned adversarial/MINE bottleneck encoder with public cloud decoder, `10` victim seeds `101..1010`, `2,560` held-out passages; PPL `98.92`), and
4. `mid-lite` ($768\to 64$ random-projection bottleneck across `3 × 500 = 1,500` live training steps, seeds `11, 22, 33`, paired step-for-step with `convergence-none-{11,22,33}`).

All 20 freshly captured `replica-median` and `secret-mixup` forward wires matched the retained `observations-{defense}-{seed}-6.pt` tensors with `max_abs_diff = 0.0`, and `forward_k32` on anchor seeds `101` and `606` matched the frozen `v3` transcripts with 100% bit-exact predictions and candidates.

See [complete defended analysis](../artifacts/paper-a-stronger-20261020/defended-analysis/REPORT.md). Across the 11-contrast Bonferroni family ($\alpha = 0.05/11$, 100,000 crossed victim/document bootstrap draws, seed `20261020`):

- **`replica-median`**: `backward` `47.61%`, `forward_k32` `94.36%` (`14.57%` exact passages), `joint_true_k32` **`97.49%`** (`41.72%` exact passages, `53.09%` whitespace-normalised); paired gain **`+3.13 pp`** (Bonferroni 95% CI `[+2.49, +3.82] pp`, `10/10` seeds).
- **`secret-mixup`**: `backward` `52.10%` (unattenuated wire gradient before client decoy subtraction), `forward_k32` `86.41%` (`0.04%` exact), `joint_true_k32` **`91.79%`** (`0.20%` exact; paired gain **`+5.38 pp`**, `[+4.60, +6.19] pp`), `adaptive_demix_forward_k32` **`88.53%`** (`1.56%` exact; **`+2.12 pp`** over `forward_k32`, `[+1.57, +2.70] pp`), and `adaptive_demix_joint_k32` **`93.09%`** (**`5.47%`** exact passages, `8.16%` whitespace-normalised; **`+6.68 pp`** over `forward_k32`, `[+5.86, +7.53] pp`; **`+4.56 pp`** over `adaptive_demix_forward_k32`, `[+3.87, +5.31] pp`; **`+1.31 pp`** over `joint_true_k32`, `[+0.77, +1.90] pp`).
- **`trained-bottleneck`**: raw 64-D wire attacks fail (`backward` `0.00%`, `forward_k32` `0.05%`, `joint_true_k32` `0.04%`), whereas strictly public-grant cloud-decoder adaptation (with **zero queries to `tb.encode`**) recovers **`62.13%`** for `adaptive_decode_forward_k32` (**`+62.08 pp`**, `[+60.08, +64.08] pp`) and **`65.23%`** for `adaptive_decode_joint_k32` (**`+65.18 pp`** over raw wire, `[+63.34, +67.04] pp`; **`+3.10 pp`** over `adaptive_decode_forward_k32`, `[+2.36, +3.85] pp`).
- **`mid-lite` (500-step live trajectory, 1,500 steps)**: `forward_k32` `0.50%`, `backward` **`20.62%`** (`33.43%` in steps 1–100, `33.88%` in steps 101–200), `joint_true_k32` `5.95%` (candidate recall `33.18%`; `+5.45 pp` `[+5.19, +5.71]` over `forward_k32`).

These results are incorporated into `paper/paper_a/main.pdf` (Section 4.9, Table 10) and `docs/paper-a-review-v2-response-2026-10-20.md`.

## Fixed-boundary placement, Qwen3-0.6B D1 sub-component isolation, and independent 64/128-token passages — Completed 21 October 2026

Tracked experiment: `6904cc4e-c11c-4da0-9826-aa80e162a423`, branch `orx/paper-a-exp3-fixed-boundary-placement-qwen3-d1-i`, source `923d16e54b14d9f17ae07edbbbc642f483e048b9` (parent `4fc1ac9c-5c95-4042-b3ec-0cdc08b2bc81`).

- Toula run (`seeds 11..44` Part A, `seeds 66, 11` Part B, `seeds 101..505` Part C; `162` cells): `80b28684-57c6-491b-a308-6434fd73db14` (`6,874.17 s`, `status: EXP3_COMPLETE`).
- Koula run (`seeds 55..88` Part A, `seeds 22, 33, 66` Part B, `seeds 606..1010` Part C; `160` cells): `e7ce0f2f-0d1c-47f4-99f3-6cf76f2a6845` (`6,885.55 s`, `status: EXP3_COMPLETE`).

See [complete Experiment #3 analysis](../artifacts/paper-a-stronger-20261020/exp3-analysis/REPORT.md). All 322 cell `.evidence.json` chains and sidecar SHA-256 digests passed verification:

1. **Part A — GPT-2 fixed-boundary training-placement intervention (`R4 Major 3` & `R6 Major 1`; `128` cells + `8` `all_local` controls across seeds `11..88`, $s=4$)**:
   - **Regime A (Fixed $L_{\text{obs}} = 1$, $b_{\text{train}} \in \{1, 3, 5, 7\}$)**: Under `D0` (8-bit seam wire), holding the observed seam fixed at block 1 and shifting the four-block server trainable segment deeper from $b_{\text{train}} = 1 \to 3 \to 5 \to 7$ *reduces* the excess validation loss penalty $\Delta_U$ from `+0.2342 → +0.1804 → +0.1471 → +0.1686 nats` (`partial_r2 = 0.9152`, `within_seed_r2 = 0.9786`, $p = 0.0001$, held-out seed RMSE `-25.3%`). Forward seam leakage is bit-exact invariant across $b_{\text{train}}$ ($\text{VMI}_{\text{excess}} = 92.27\%$, `forward_k32` $= 99.77\%$, `partial_r2 = 0.0`), while `backward` token recovery declines from `11.29% → 9.30%` (`partial_r2 = 0.8930`, $p = 0.0001$, held-out RMSE `-57.9%`). This directly resolves the original coupled P6 confound: the $+4.0686\text{ nats}$ utility spike and $1.70\%$ VMI collapse at $(s=4, b=3)$ in the original coupled P6 grid were caused by moving per-sequence `absmax` quantization to deeper GPT-2 residual-stream layers ($L_{\text{obs}} \ge 3$), not by training middle transformer blocks.
   - **Regime B (Fixed $L_{\text{obs}} = 6$, $b_{\text{train}} \in \{0, 2, 4, 6\}$)**: Under `D0`, allowing client blocks before the $L_{\text{obs}} = 6$ seam to adapt ($b_{\text{train}} = 0$) lowers $\Delta_U$ from `+3.8608 nats` ($b_{\text{train}} = 6$) to `+2.7928 nats` (`partial_r2 = 0.9863`, $p = 0.0001$, held-out RMSE `-82.2%`) while increasing `forward_k32` recovery from `3.55% → 35.26%` and `joint_true_k32` from `6.30% → 54.15%` (`+18.88 pp` paired gradient gain at $b_{\text{train}} = 0$). Under `D1` (4-bit + `0.1 × RMS` gradient noise), backpropagating quantized noisy gradients into client blocks ($b_{\text{train}} \in \{0, 2\}$) severely degrades utility ($\Delta_U = +9.31\text{ nats}$ vs. $+4.30\text{ nats}$ when client blocks before $L_{\text{obs}} = 6$ remain frozen at $b_{\text{train}} = 6$; `partial_r2 = 0.9999`, $p = 0.0001$).
2. **Part B — Qwen3-0.6B FP32 `D1` 8-arm sub-component isolation factorial (`R6 Minor 10`; $s=4, b=0$, seeds `66, 11, 22, 33` + cross-host verification on `seed 66`, `34` cells)**:
   - Adding gradient clipping (`1.0`) and `0.1 × RMS` gradient noise to 8-bit forward quantization (`quant8_clip1_noise01`) is **100% finite (4/4 seeds)** with mean held-out $\text{PPL} = 31.08$ (`seed 66` $\text{PPL} = 34.48$), ruling out clipping and noise as the primary cause of FP32 overflow.
   - Conversely, 4-bit forward quantization *alone* with raw unclipped, noiseless backward gradients (`quant4_fwd_only`) overflows in **2/4 seeds** (`seed 22` at step 47, `seed 33` at step 38) and collapses the remaining finite seeds to mean $\text{PPL} = 1335.14$ (`seed 66` $\text{PPL} = 2663.61$).
   - Root cause and fix: at $b = 0$ (`split = 1`), trainable block `0` sits *before* the quantized wire `h_sent`. In `_quantize_absmax(bits=4)`, `.round()` has zero derivative almost everywhere in PyTorch autograd, so `h_star` receives gradient only through `scale = amax / 7.0`, zeroing out all $32{,}767$ non-extremal coordinates and routing the entire inner product $\langle q/7, g \rangle$ onto the single `amax` coordinate of `h_star`. Replacing direct rounding autograd with a straight-through estimator (`D1_ste_fix`: `h_sent = h_star + (h_q - h_star).detach()`) under the full `D1` defence makes **all 4/4 seeds 100% finite in FP32** with mean held-out $\text{PPL} = 46.30$ (`seed 66` $\text{PPL} = 49.1795$, bit-exact across `toula` and `koula`).
3. **Part C — Independent 32-, 64-, and 128-token passage evaluation (`R6 Major 3`; `255` held-out WikiText-2 articles $\ge 128$ tokens $\times$ `10` primary victim seeds `101..1010`, $L = 6$, public client copy, $B = 2$, `150` cells)**:
   - **Zero prefix-error compounding**: At $T = 128$, late-position token recovery (`pos 64..127`) exceeds early-position recovery (`pos 0..31`) by `+0.43 pp` (`[-0.25, +1.14]`, `94.63%` vs. `94.20%`) for `forward_k32` and by `+0.90 pp` (`[+0.36, +1.46]`, `97.93%` vs. `97.03%`) for `joint_true_k32`, confirming that local transformer attention and step-wise activation residual scoring prevent beam drift across 128-token sequences.
   - **At $k = 32, B = 2$**: The paired gradient token gain is stable across sequence lengths at **`+3.17 pp`** ($T = 32$, `[+2.73, +3.65]`), **`+2.91 pp`** ($T = 64$, `[+2.59, +3.24]`), and **`+3.02 pp`** ($T = 128$, `[+2.76, +3.28]`), while exact full-passage recovery increases **`2.75×`** at $T = 32$ (`13.76% → 37.92%`, `+24.16 pp`), **`4.96×`** at $T = 64$ (`4.35% → 21.57%`, `+17.22 pp` `[+11.61, +22.94]`; `6.94% → 30.24%` whitespace-normalised), and **`23.5×`** at $T = 128$ (`0.31% → 7.37%`, `+7.06 pp` `[+4.23, +10.43]`; `0.55% → 12.16%` whitespace-normalised). At $T = 128$, `joint_true_k32` also lifts contiguous 8-token exact span recovery from `66.92% → 84.86%`, longest contiguous exact token run from `49.23 → 76.94 tokens`, and multi-token entity/name span exact match from `68.68% → 84.47%`.
   - **At $k = 128, B = 2$**: Expanding the activation-only candidate budget to $k = 128$ (`forward_k128`) raises token recovery to **`98.51%`** at $T = 64$ (`+3.95 pp` over `forward_k32`, exact match `36.39%`) and **`98.70%`** at $T = 128$ (`+4.04 pp` over `forward_k32`, exact match `20.04%`, longest run `97.44 tokens`), narrowing the paired gradient token gain (`joint_true_k128 - forward_k128`) to **`+0.05 pp`** (`[-0.09, +0.18]`) at $T = 64$ and **`+0.06 pp`** (`[-0.05, +0.17]`) at $T = 128$ at `4.0×` online search time.

These results are incorporated into the 57-page manuscript `paper/paper_a/main.pdf` (`Section 4.10`, `Table 11`; `Section 5`; `Appendix C.2`, `Table 16`; `Appendix C.4`, `Tables 20–21`) and `docs/paper-a-review-v2-response-2026-10-20.md`.

## Cross-architecture Qwen3-0.6B controlled joint text-reconstruction benchmark — Completed 21 October 2026

Tracked experiment: `f5e39ef0-80c6-46a2-afef-25f5841d675c`, branch `orx/paper-b-qwen3-recon-benchmark-20261020`, source `b18a256` (parent `6904cc4e-c11c-4da0-9826-aa80e162a423`).

- Toula run (`seeds 11, 22, 33, 44`, `152` cells, `32` prediction sidecars): `5bc66b03-3d8d-4d20-a19f-6bf45334617b` (`3,795.4 s`, `status: QWEN3_RECON_COMPLETE`).
- Koula run (`seeds 55, 66, 77, 88`, `152` cells, `32` prediction sidecars): `85534cfb-a8ba-4813-a843-a42f109d04f4` (`3,794.0 s`, `status: QWEN3_RECON_COMPLETE`).

See [complete Qwen3-0.6B reconstruction report](../output/paper-b-qwen3-recon-20261020/qwen3_recon_report.md) and [JSON summary](../output/paper-b-qwen3-recon-20261020/qwen3_recon_summary.json). All 304 evaluation cells (`152` per host) and 64 prediction sidecars passed `EvidenceLogger` SHA-256 hash-chain and sidecar verification (`kv_cache_verified_cells = 304`). The benchmark evaluates `Qwen/Qwen3-0.6B` (28 layers, $d=1024$, $V=151{,}936$, GQA 16Q/8KV, QK-Norm, RoPE, SwiGLU, RMSNorm) across split depths $L \in \{1, 7, 14, 21\}$ (4%, 25%, 50%, 75% depth), client layer grants (`public`, `exact`), and eight fine-tuned victim seeds (`11..88`, eval NLL `3.8404 → 3.7931 ± 0.1221`, PPL `46.54 → 44.70 ± 5.88`) on 256 held-out 32-token WikiText-2 passages (`2,048` documents = `65,536` token positions per cell; 100,000-replicate two-way crossed bootstrap CIs, seed `93281`):

- **$L = 1$ (4% depth)**: `forward_k32` is already near-saturated on `Qwen3-0.6B` (`97.99%` token / `60.01%` exact doc under `public`; `98.25%` / `64.99%` under `exact`), whereas `backward` is `0.23%` / `0.24%` because a single client block provides almost no token-aligned shifted gradient signal. Replacing $k/2 = 16$ forward candidates with uninformative backward proposals slightly reduces recovery at $k=32$ (`joint_true_k32 = 97.36%`, `-0.63 pp` `[-0.78, -0.49]` under `public`, matching `joint_shuffled_k32 = 97.34%`), while at $k=128$ `forward_k128` (`98.40%`, `65.48%` exact doc) and `joint_true_k128` (`98.27%`, `63.53%` exact doc) both exceed `98.2%`.
- **$L = 7$ (25% depth)**: `backward` rises to `5.66%` (`public`) and `5.75%` (`exact`), while `forward_k32` drops to `63.10%` (`public`, candidate recall `70.61%`) and `75.10%` (`exact`, recall `76.07%`). `joint_true_k32` lifts token recovery to **`65.96%`** (`+2.87 pp` `[+2.25, +3.50]` over `forward_k32`; `+4.65 pp` `[+4.03, +5.30]` over `joint_shuffled_k32 = 61.32%`) under `public` and **`77.11%`** (`+2.02 pp` `[+1.48, +2.59]` over `forward_k32`; `+4.49 pp` `[+3.96, +5.02]` over `joint_shuffled_k32 = 72.63%`) under `exact`. At $k = 128$ (`public`), `joint_true_k128` reaches **`69.84%`** (`+5.43 pp` `[+4.70, +6.16]` over `forward_k128 = 64.40%`), and `joint_true_k32` (`65.96%`) beats `forward_k128` (`64.40%`) by **`+1.56 pp`** `[+0.77, +2.35]` at one-fourth the candidate budget.
- **$L = 14$ (50% depth, midpoint split analogous to $L=6/12$ in GPT-2)**: `backward` reaches `26.09%` (`public`) and `26.56%` (`exact`). Under `public`, `forward_k32` recovers `55.20%` (recall `72.57%`) while `joint_true_k32` reaches **`69.05%`** (recall `86.64%`), a **`+13.85 pp`** (`[+12.38, +15.27]`) gain over `forward_k32` and **`+15.27 pp`** (`[+13.81, +16.71]`) over `joint_shuffled_k32` (`53.78%`). Under `exact`, `forward_k32` recovers `72.79%` (`0.05%` exact doc, `19.21%` span-8) while `joint_true_k32` reaches **`86.87%`** (**`5.27%`** `[2.39%, 8.98%]` exact doc, `45.46%` span-8), a **`+14.08 pp`** (`[+12.82, +15.41]`) gain over `forward_k32` and **`+16.51 pp`** (`[+15.23, +17.81]`) over `joint_shuffled_k32` (`70.36%`). Crucially, unlike GPT-2 at $L=6$, quadrupling the `public` activation-only candidate budget to $k = 128$ raises `forward_k128` by only `+1.98 pp` (`55.20% → 57.18%`, recall `76.70%`) at `4.4×` online search time (`87.9 s` vs. `19.9 s`), so **`joint_true_k32` (`69.05%`) beats `forward_k128` (`57.18%`) by `+11.88 pp` `[+10.30, +13.43]`** at one-fourth the candidate budget, while **`joint_true_k128` reaches `71.35%`** (recall `90.90%`, **`+14.18 pp` `[+12.84, +15.49]`** over `forward_k128`).
- **$L = 21$ (75% depth)**: `backward` reaches `73.67%` (`public`) and `73.84%` (`exact`). Under `public`, `forward_k32` recovers `64.56%` (`8.48%` span-8, recall `76.46%`), whereas `joint_true_k32` reaches **`82.35%`** (`31.29%` span-8, recall `96.97%`), beating `forward_k32` by **`+17.79 pp`** (`[+16.40, +19.21]`), `joint_shuffled_k32` (`62.68%`) by **`+19.66 pp`** (`[+18.25, +21.13]`), and `backward` by **`+8.67 pp`** (`[+5.41, +11.44]`). Under `exact`, `forward_k32` recovers `75.48%` (`0.00%` exact doc, `23.42%` span-8), whereas `joint_true_k32` reaches **`96.85%`** (**`44.58%`** `[38.92%, 50.34%]` exact 32-token passages, `88.68%` span-8, recall `97.82%`), beating `forward_k32` by **`+21.37 pp`** (`[+19.87, +22.98]`), `joint_shuffled_k32` (`72.47%`) by **`+24.38 pp`** (`[+22.72, +26.16]`), and `backward` by **`+23.01 pp`** (`[+21.15, +24.85]`).

These results are incorporated into both Paper B (`paper/paper_b/arxiv/` and `paper/paper_b/eurosp2027/`, `Table S3`) and the 57-page Paper A manuscript `paper/paper_a/main.pdf` (`Section 4.11`, `Table 12`) and `docs/paper-a-review-v2-response-2026-10-20.md`.


