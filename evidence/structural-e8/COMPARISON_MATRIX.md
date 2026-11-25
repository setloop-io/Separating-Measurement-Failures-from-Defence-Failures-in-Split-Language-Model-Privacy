# COMPARISON_MATRIX.md — v2 (post clean-slate)

Every row below is backed by a committed JSON artifact under `paper-data/`
(synced from the nodes). "Legacy" rows exist only under git tag `legacy-v1`
and are **not** carried into v2 claims. n=3 seeds (privacy) or n=3 reps
(inference) unless noted.

Nodes: **toula/koula** = 2× DGX Spark GB10 (ARM64, 128GB unified, bond0 +
tc-netem WAN emulation) · **h100-1/h100-2** = H100 NVL 94GB (x86).

## 1. Inference: split decoding throughput × RTT (n=3 reps per cell)

| Model | Split (local/cloud/local) | Nodes | Profiles | Status | Artifact |
|---|---|---|---|---|---|
| Mistral-7B | 2 / 28 / 2 layers | toula↔koula | 0, 20, 80, hostile(~460ms) | ✅ done | `results/v2/7b/rtt_*` |
| Mistral-NeMo-12B | 2 / 36 / 2 layers | toula↔koula | 0, 20, 80, hostile | ✅ done | `results/v2/12b/rtt_*` |

Headline (7B): lookahead gain +15% (0ms), +17% (20ms), +15% (80ms), +11%
(hostile); acceptance 1.195 tok/step; cloud compute flat at 58.8ms across
profiles — RTT dominance confirmed. 12B numbers in artifacts.

## 2. Identity: lookahead == sequential greedy (200 tokens, 4 prompts)

| Model | Result | Status | Artifact |
|---|---|---|---|
| Mistral-7B @80ms | 0/4 diverge | ✅ | `results/v2/7b/identity_80ms.json` |
| Mistral-NeMo-12B @80ms | 0/4 diverge | ✅ (verified post-fix) | `results/v2/12b/identity_80ms_rep*.json` |

## 3. Trained-inversion privacy (MLP attacker, lora-local FT, utility-tracked)

Recovery top-1 at FT step 0 (flat across 0/10/100 steps everywhere — the
"no adaptation defense at utility-safe LR" finding):

| Model | d1 | d4 | d8 | Utility intact | Node | Status |
|---|---|---|---|---|---|---|
| Mistral-7B | 64.2% | — | 56.4% | ✅ (2.72→2.88) | h100-1 | ✅ |
| Qwen3-8B | 69.9% | — | 64.7% | ✅ | h100-1 | ✅ |
| Mistral-NeMo-12B | 65.1% | 61.8% | 59.0% | ✅ (3.05→2.62) | h100-1 | ✅ |
| Qwen3.6-27B | 69.9% | 67.4% | *running* | ✅ (2.41→2.13) | h100-2 | ⏳ d8 |

Artifacts: `results-h100-*/v2/privacy/ti_*.json`. Trend: larger model ⇒
leakier boundary; depth helps modestly; fine-tuning drift does not defend.

## 4. Exact inversion: SipIt v3 hybrid (MLP proposes, exact verifier certifies)

| Model | d1 | d4 | d8 | Node | Artifact |
|---|---|---|---|---|---|
| Qwen3-0.6B | 97.9% | 76.6–78.1% | 76.6–78.1% | h100-1 | `sipit_06b.json` |

Geometric proxy provably rank-deficient at depth ≥4 (recall@4096 ≈ 0–25%);
the hybrid proposer+verifier is the paper's "MLP numbers are lower bounds"
evidence (issue #8).

## 5. E8 obfuscation defense (trained attacker, in-process)

| Model | Baseline d1 | Obfuscated (max over depths) | W-leaked control | Node | Artifact |
|---|---|---|---|---|---|
| Mistral-7B | 64.2% | 0.59% | = baseline | h100-2 | `e8_7b.json` |
| Qwen3-8B | 69.8% | 0.33% | = baseline | h100-1 | `e8_8b.json` |
| Mistral-NeMo-12B | 65.1% | 1.30% | = baseline | h100-1 | `e8_12b.json` |

## 5b. E8 robustness battery (attacking the defense; issue #24)

Robustness sweep against the E8 obfuscation defense from Section 5: passive
label-free attacks, session-rotation pooling, an oracle/chosen-plaintext
attack swept over K labeled pairs, and partial leakage of W's rows. Model:
Qwen3-0.6B. Node: h100-1. n=3 seeds unless noted. Two runs exist —
`e8_robustness_06b_fp32_rep2.json` (fp32, 36 rows, quoted below) and
`e8_robustness_06b.json` (bf16, 33 rows, used only for the dtype-sensitivity
cross-check in the caveats).

**Control check** (harness sanity, issue #24): at K=5000 the oracle has
recovered W essentially completely, so `3_oracle_pairs K=5000` must equal
`0_baseline_no_defense`/`plain_h` at every depth — full key recovery leaves
the defense contributing nothing to top-1. Confirmed exactly in the fp32
run: depth 1 68.55% = 68.55%, depth 4 67.38% = 67.38%, depth 8 65.76% =
65.76%.

| Setting | Depth 1 | Depth 4 | Depth 8 |
|---|---|---|---|
| `0_baseline_no_defense` | 68.55 ± 0.20% | 67.38 ± 0.39% | 65.76 ± 0.57% |
| `0_obfuscated_static_W` | 0.20 ± 0.20% | 0.65 ± 0.96% | 0.20 ± 0.20% |
| `1a_whiten_nn` | 0.00 ± 0.00% | 0.00 ± 0.00% | 0.00 ± 0.00% |
| `1b_moment_align` | 1.63 ± 0.56% | 0.20 ± 0.00% | 0.20 ± 0.00% |
| `2_session_rotation` | 2.73 ± 0.00% | 0.20 ± 0.00% | 0.39 ± 0.00% |
| `3_oracle_pairs K=10` | 21.55 ± 2.80% | 3.32 ± 2.58% | 1.76 ± 0.39% |
| `3_oracle_pairs K=100` | 49.55 ± 1.33% | 28.97 ± 2.84% | 22.01 ± 3.23% |
| `3_oracle_pairs K=1000` | 68.55 ± 0.20% | 67.38 ± 0.39% | 65.69 ± 0.30% |
| `3_oracle_pairs K=1024` | 68.55 ± 0.20% | 67.32 ± 0.30% | 65.82 ± 0.52% |
| `3_oracle_pairs K=5000` | 68.55 ± 0.20% | 67.38 ± 0.39% | 65.76 ± 0.57% |
| `4_partial_leak rows_leaked=50%` | 65.04 ± 0.85% | 56.32 ± 0.60% | 53.39 ± 2.26% |
| `4_partial_leak rows_leaked=90%` | 68.42 ± 0.30% | 66.93 ± 0.30% | 65.17 ± 0.49% |

Caveats:

- Quote the fp32 run. The `4_partial_leak rows_leaked=50%` row is
  dtype-sensitive and must NOT be quoted from bf16: it shifts +10.68pp
  (depth 4), +6.97pp (depth 8), +5.60pp (depth 1) from bf16 to fp32 — 6.6× /
  17.8× / 3.1× its own fp32 seed std (depths 1/4/8), i.e. a systematic shift,
  not noise. The low-K
  oracle points (K=10, K=100) also move by up to 4.69pp between dtypes
  (depth 8, K=100), but that sits within ~1–2× their own reported std
  (1.3–3.3pp); every other row moves under 1.6pp.
- The oracle "security parameter" is **K₅₀ ≈ 88–94 ≈ 0.09·H at depth 1**,
  measured — see the sub-grid below, and note it is a depth-1 figure: depths
  4 and 8 cross at 352.3 and 293.8 (≈0.34·H and ≈0.29·H).
- Do NOT quote the `verdict` string of any artifact committed before issue
  #30 — including this one, whose verdict reads "crosses 50% at K=1000". That
  was the first *sampled* K with top-1 ≥ 50, never a crossing, and this run's
  grid jumps 100 → 1000: K=100 read 49.55%, a whisker under, so it published
  1000, a ~10× overstatement. Two runs with identical inputs disagreed by 10×
  on that line (`rep2` says K=1000, `anchorcheck` says K=100) purely because
  one landed either side of 50. `crossing_k()` in `e8_robustness.py` replaced
  the read-off: it interpolates linearly in K between the bracketing measured
  points, or declines when the bracket is wider than 3×, and emits
  `oracle_k50` as a real artifact field. On this run's grid it correctly
  declines rather than naming a K.
- `1a_whiten_nn` and `2_session_rotation` report n_seeds=1, not 3 — those
  rows have no error bars (the ±0.00 in the table reflects a single run,
  not zero variance).

### 5b.1 Oracle sub-grid — where the defense actually breaks (depths 1/4/8, fp32)

Three independent W seeds at depth 1, n=3 decoder seeds per point; depths 4 and
8 were swept by seed 123 alone, so those two tables carry a single column.
`K₅₀` is linearly interpolated between the bracketing measured points (linear
in K, not log K). One table, one `K₅₀` line and one `Depth N:` heading per
depth, in the order 1, 4, 8: `bin/render_matrix_5b.py --check` pairs the i-th
of each with the i-th depth and fails if a block is added, dropped, reordered
or relabelled. The headings are part of what is checked because the pairing is
positional — swapping two of them would otherwise publish one depth's evidence
under another depth's name with every table byte-identical.

**Read the pool column.** `max(k_list)` sizes the oracle token pool
(`k_max`/`frequent_tokens` in `e8_robustness.py`), which feeds `h0` — so it is a result-determining
input, not a reporting grid. Only **seed 123** was swept at pool 5000, matching
the primary `rep2` run; seeds 42 and 99 drew smaller pools and are therefore not
directly comparable to `rep2` row-for-row.

Depth 1:

| K | seed 42 (pool 100) | seed 99 (pool 1000) | seed 123 (pool 5000) |
|---|---|---|---|
| 10 | 21.29 ± 1.09% | 21.55 ± 2.80% | 19.46 ± 2.05% |
| 20 | 25.13 ± 0.63% | — | — |
| 30 | 26.69 ± 1.56% | 29.69 ± 1.36% | 29.23 ± 1.15% |
| 40 | 34.38 ± 0.85% | — | — |
| 50 | 38.41 ± 1.49% | 37.37 ± 1.01% | 37.89 ± 0.85% |
| 60 | 44.14 ± 0.85% | — | — |
| 70 | 43.75 ± 2.72% | 44.53 ± 1.22% | 43.16 ± 1.22% |
| 80 | 48.11 ± 1.99% | — | — |
| 90 | 49.61 ± 1.03% | 50.72 ± 0.30% | 51.04 ± 0.60% |
| 100 | 50.52 ± 1.64% | 52.34 ± 0.51% | 52.34 ± 1.41% |
| 200 | — | 62.17 ± 1.00% | 61.91 ± 0.20% |
| 300 | — | 66.54 ± 0.23% | 67.06 ± 0.81% |
| 500 | — | 68.29 ± 0.49% | 69.14 ± 1.22% |
| 700 | — | 68.68 ± 0.12% | 69.59 ± 0.92% |
| 1000 | — | 68.62 ± 0.30% | 69.73 ± 1.22% |
| 1024 | — | — | 69.66 ± 1.33% |
| 2000 | — | — | 69.66 ± 1.33% |
| 5000 | — | — | 69.66 ± 1.33% |

**K₅₀ = 94.3 (seed 42), 87.7 (seed 99), 87.4 (seed 123).** With H=1024 that is
**~9% of the pairs a determined system would need at depth 1**, replicated
across three independent secrets. Seeds 99 and 123 agree to 0.3 (87.7 vs 87.4)
despite drawing from differently sized pools — 1000 and 5000 — so pool size does
not visibly move the depth-1 crossing even though it is a result-determining
input. Only seed 123 was swept at pool 5000, so it is the one directly
comparable to `rep2` row-for-row. Depths 4 and 8 cost the attacker ~4× more —
see below.

Two-regime result worth stating explicitly, **at depth 1**:
`W_recovery_rel_err` stays at ~1.34 through K=100 and only collapses near K=H
(0.22 at K=1000, 1e-4 at K=1024 in the main run). Token recovery crosses 50%
while the recovered W is still ~134% wrong in Frobenius norm — **reading the
tokens does not require recovering the key.** The algebraic threshold is at
K=H as theory predicts; the security threshold is an order of magnitude below
it.

The collapse at K=H is depth-1-specific and softens with depth: at K=1024 the
same run gives 0.0377 at depth 4 and 0.0442 at depth 8, ~400× the depth-1
figure, and only reaches ~4e-4 by K=5000. The two-regime shape survives; the
sharp algebraic threshold does not.

**Depth is the other axis of the security parameter.** Seed 123 swept the same
K grid at depths 4 and 8; seeds 42 and 99 are depth-1 runs, so the two tables
below have a single column and no cross-seed replication.

Depth 4:

| K | seed 123 (pool 5000) |
|---|---|
| 10 | 2.28 ± 1.52% |
| 30 | 14.91 ± 1.57% |
| 50 | 14.39 ± 1.14% |
| 70 | 14.45 ± 4.74% |
| 90 | 31.51 ± 0.57% |
| 100 | 32.62 ± 0.68% |
| 200 | 40.17 ± 3.67% |
| 300 | 45.96 ± 2.15% |
| 500 | 61.39 ± 1.38% |
| 700 | 66.54 ± 0.63% |
| 1000 | 66.99 ± 0.33% |
| 1024 | 67.06 ± 0.12% |
| 2000 | 67.06 ± 0.12% |
| 5000 | 67.06 ± 0.12% |

**K₅₀ = 352.3 (seed 123).**

Depth 8:

| K | seed 123 (pool 5000) |
|---|---|
| 10 | 2.60 ± 0.74% |
| 30 | 9.51 ± 2.66% |
| 50 | 12.57 ± 1.75% |
| 70 | 17.97 ± 3.05% |
| 90 | 23.37 ± 5.54% |
| 100 | 28.71 ± 1.37% |
| 200 | 38.22 ± 1.56% |
| 300 | 50.78 ± 1.70% |
| 500 | 60.29 ± 2.66% |
| 700 | 65.36 ± 0.30% |
| 1000 | 65.75 ± 0.79% |
| 1024 | 65.56 ± 0.81% |
| 2000 | 65.69 ± 0.79% |
| 5000 | 65.69 ± 0.79% |

**K₅₀ = 293.8 (seed 123).**

**Depth raises the attacker's cost ~4×.** Against the same secret and the same
pool (seed 123, pool 5000) the crossing moves from 87.4 pairs at depth 1 to
352.3 at depth 4 (4.03×) and 293.8 at depth 8 (3.36×) — ~0.34·H and ~0.29·H
against depth 1's ~0.09·H. The ~0.09·H security parameter quoted above is
therefore a **depth-1** figure, not a depth-independent property of the
defense. It is the right number to quote against an attacker who can choose
where the boundary sits — it is that attacker's cheapest case — but reading it
as *the* security parameter understates what a depth-4 or depth-8 boundary
costs by roughly 4×.

Three caveats on the depth-4/8 numbers, all readable off the tables above.
One secret only: the depth-1 estimate is replicated across three independent W
seeds (87.4–94.3) and these are not, so there is no equivalent bound on the
seed-to-seed spread. Wider brackets: the crossings sit in [300, 500] (1.67×)
and [200, 300] (1.50×) against depth 1's [70, 90] (1.29×), all inside the 3×
interpolation limit but correspondingly softer point estimates. And `K₅₀` is
not monotone in depth — depth 8 crosses *earlier* than depth 4 (293.8 vs
352.3), and at K=300 the depth-8 curve is above the depth-4 one (50.78% vs
45.96%). Read "~4×" as the scale of the depth effect, not as a trend in depth.

**What the ± near K₅₀ does and does not cover.** In every artifact in this
table `W_hat = polar(lstsq(...))` was computed once per K *outside* the seed
loop, so the reported ± varies only decoder init, never the key estimate. At
K≈100 the least-squares is rank-100 in a 1024-dim space, leaving ~924 null
directions for `polar()` to complete arbitrarily.
`e8_robustness_06b_fp32_rep2.json` and `..._anchorcheck.json` share every
result-determining input — `dtraining_commit`, `corpus_sha256`,
`model_manifest_sha256`, container image, `W_seed` 99, victim doc indices,
**and `max(k_list)` = 5000** — yet produced K=100 top-1 of **49.55 vs 51.24**,
a **1.69pp** spread against reported stds of **1.33 and 0.63**.

(Corrected: this sentence previously read "0.51 and 0.63". `0.51` is
`..._kvoid_seed99.json`'s K=100 std — the run the very next paragraph excludes
as a non-replicate. `rep2`'s own std is `1.3293`, printed as `49.55 ± 1.33%` in
the §5b table above. The spread is therefore ~1.3× the larger std, not ~3×, so
it is weaker evidence of instability than the original sentence implied.
Re-derive with
`python3 -c "import json;d=json.load(open('paper-data/results-h100-1/v2/privacy/e8_robustness_06b_fp32_rep2.json'));print([r for r in d['summary'] if r['attack']=='3_oracle_pairs' and r['setting']=='K=100' and r['depth']==1])"`.)

The primary `top1_mean`/`top1_std` deliberately keep the top-K-most-frequent
selection and its decoder-only ±: attack 3 is chosen-plaintext, so the security
parameter belongs to the best pair choice the attacker has, and a random draw
is a weaker attacker whose curve would move K₅₀ up and overstate the defense.

Pair-selection sensitivity is measured separately, by
`e8_robustness.py --d3-repetitions` (`d3_oracle_repeatability` in the output),
which nests independent probe draws around the decoder seeds rather than
adding a field to every oracle row. Decoder seeds alone are not independent
oracle repeats — see `REPRODUCTION_V2.md` §E8 D3. None of the artifacts in
this table carry `d3_oracle_repeatability`: it is off by default and was not
requested for these runs.

Do not widen that to the 2.80pp implied by also including
`..._kvoid_seed99.json` (52.34): `k_max = max(args.k_list)` sizes the oracle
token pool (`k_max`/`frequent_tokens` in `e8_robustness.py`), so that run drew
from 1000 tokens rather than 5000 and is not an identical-input replicate.
`k_list` is a result-determining input, not just a reporting grid. Runs now
record `provenance.oracle_pool_sha256` and `provenance.oracle_h0_sha256` per
depth, so "same pool?" and "same activations?" are answerable across runs
instead of inferred — the artifacts above predate both fields. Full-rank
points (K ≥ 1024) agree to the 4 decimals these files record; near-threshold
points are where the ± above is doing the least work.

### 5b.2 fp32 round-trip exactness — established (issue #30 defect A)

`e8_obfuscation.py`'s `interpretation` asked for a `--dtype fp32` re-run to
verify the W round-trip is exact. The functional half always held
(**`token_identity` = 1.0 at 6/6 cells**); the exactness half did not, because
`--dtype` never controlled the quantity under test. `make_secret` returns a CPU
float32 tensor, both call sites move it to the device with no dtype cast, and
the round-trip casts with `h.float()` — so both operands were fp32 in *every*
run regardless of the flag, while the NGC container's
`float32_matmul_precision="high"` made every one of those matmuls execute as
TF32. `--dtype` set the model dtype and nothing else.

`--matmul-precision {highest,high,medium}` now controls it, and every run
records what actually executed — `matmul_precision_effective`,
`cuda_matmul_allow_tf32`, `cudnn_allow_tf32`, `deterministic_algorithms`,
`cublas_workspace_config`, `torch_version` — read back out of torch rather than
echoed from the flag. Echoing is how the wrong precision went unnoticed for the
whole run family.

Measured on h100-3, two runs differing only in that flag
(`e8_obfuscation_06b_fp32_tf32.json` and `..._highest.json`, both `W_seed` 42):

- **Relative deviation falls from 9.34e-05 – 3.33e-04 to 2.24e-07 – 3.83e-07**,
  i.e. 2.56–3.17 orders tighter and below 1e-6 at all six cells. That settles
  the exactness claim.
- **Quote the relative figure, not the absolute one.** Under `highest` the
  *absolute* deviation at depths 4/8 is still 1.46e-03 – 1.95e-03, because
  `max_abs_h` there is 5.75e3 – 6.54e3 against 7.84 – 9.95 at depth 1. Any
  absolute target that fits depth 1 is unreachable at four of the six cells by
  three orders. `max_abs_h` is now recorded per cell, so this is checkable from
  the artifact instead of asserted.
- **`token_identity` stays 1.0 at 6/6 under `highest`** — tightening precision
  did not cost utility.
- **The default path did not move.** With the flag absent, all six
  `max_roundtrip_deviation` values are bit-identical to the h100-1 artifact
  (0.0023221969604492188, 0.0028057098388671875, 2.17822265625, 1.033203125,
  1.38134765625, 0.53662109375). That also stands in for a model-identity check
  that is otherwise impossible: rep2 carries a **v1** `model_manifest_sha256`,
  which hashed the HuggingFace download rather than the model (#36) and cannot
  be compared against a v2 hash, and h100-1 no longer exists.

Overhead, re-measured under every mode at a recorded `H`=1024, `seq_len`=64:
**highest 0.0544, high 0.0541, medium 0.0508 ms** per matmul. Single timings
with no repetitions, and the two committed runs disagree with each other by
more than the highest-vs-high gap (0.0529/0.0512 in the other), so the honest
reading is that **the cost of true fp32 is below what this measurement can
resolve at this size** — not that it is zero. That also settles the older
reading of `overhead_ms_per_matmul`
(0.017 for 0.6B, 0.0285 for 8B) as a dtype comparison — it is a
**hidden-dimension** difference, and these h100-3 timings are not comparable
with h100-1's at all.

Artifacts: `e8_robustness_06b_fp32_rep2.json` (fp32, primary quoted run),
`e8_robustness_06b.json` (bf16, dtype cross-check only),
`e8_robustness_06b_fp32_kvoid{,_seed99,_seed123,_sub100_seed42}.json` (sub-grid),
`e8_robustness_06b_fp32_anchorcheck.json` (reproduction check),
`e8_obfuscation_06b_fp32.json` (round-trip) — all h100-1. Under
`results-h100-3/v2/privacy/`: `e8_obfuscation_06b_fp32_{tf32,highest}.json`,
the precision pair quoted above.

## 6. E8 deployment reality (E9, three legs, live wire toula↔koula)

| Leg | Config | Expected | Status |
|---|---|---|---|
| A | baseline, no W | reference | ✅ done (throughput ~1.5 tok/s) |
| B | W local-only (E8 as specified) | broken output (not equivariant) | ✅ done (mismatch at token 0) |
| C | shared-key (cloud unwraps) | identity restored | ✅ done (identical sequence) |
| eval | trained attacker on captured wire (n=3 sessions) | base ~23–50%, obf 0.0% | ✅ done (results in `results/e9/`) |

Script: `bin/run_e9.sh` (rewritten per issue #27). Probe:
`probes/probe_e9_equivariance.py` (numpy proof, PR #28).

## 7. E8 known-prefix break (issue #3, realistic Procrustes)

| Model | Known prefix | Sessions to full break (measured) | Artifact |
|---|---|---|---|
| Qwen3-0.6B | 32 tok | 13 | `e8kp_06b.json` |
| Qwen3-0.6B | 64 tok | 4 | 〃 |
| Qwen3-0.6B | 128 tok | 4 | 〃 |
| Qwen3-0.6B | 256 tok | 2 | 〃 |
| Mistral-7B | 32 tok | 85 (final recovery 62.4% = baseline) | `e8kp_7b.json` |

## 8. Output-side boundary attack (issue #21, next-token recovery)

| Model | Tail depth | Baseline top-1/top-5 | Obf-passive | Obf-retrained | Artifact |
|---|---|---|---|---|---|
| Qwen3-0.6B | 2 | 23.1 / 38.8% | 3.1% | 21.3% | `output_inv_06b.json` |
| Qwen3-0.6B | 4 | 22.7 / 38.1% | 1.2% | 23.0% | 〃 |
| Mistral-7B | 2 | 21.8 / 36.3% | 1.9% | 23.1% | `output_inv_7b.json` |
| Mistral-7B | 4 | 20.7 / 35.0% | 1.5% | 21.0% | 〃 |

## 9. Pending / queued

| Experiment | Models | Nodes | Blocked by | Issue |
|---|---|---|---|---|
| TI-27B depth 8 | 27B | h100-2 | OOM fix in flight (agent) | — |
| DLG++ gradient inversion | 0.6B | h100-2 | chained after TI-27B | — |
| E-D1 identity divergences (5/24, 4 unprobed) | 7B | h100-1 | 12B case fixed in 4f108fa; same lockstep treatment pending | #10 |
| V2-C: real WAN vs netem validation | 7B | toula↔h100-1 | E9 completion | — |
| V2-D: training matrix n=3 (E4/FedAvg) | 0.6B/7B | Spark pair | E9 completion | — |
| E-D2: two-node draft-spec WAN run | 2B→27B | cross-WAN | FLA kernels + #5 scoping | #5, #19 |
| FLA kernels (GB10) | — | Sparks | not started | #19 |

## Node assignment summary

| Node | Done (v2) | Running now |
|---|---|---|
| toula+koula | 7B sweep ×3, 12B sweep ×3, identity both | E9 three-leg |
| h100-1 | TI-7B/8B/12B, E8-8B/12B, SipIt-0.6B, OI-0.6B/7B, KP-0.6B/7B, E8-robustness-0.6B (bf16+fp32+sub-grid) | idle |
| h100-2 | E8-7B, TI-27B (d1/d4) | TI-27B d8 → DLG++ |

*Last updated: E8 robustness battery + oracle sub-grid (issue #24). Regenerate from artifacts; do not hand-edit numbers.*
