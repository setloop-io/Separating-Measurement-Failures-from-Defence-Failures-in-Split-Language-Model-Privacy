# BiSR fidelity progress — 18 September 2026

**Real-model engineering checks passed; full empirical conformance remains
NOT_ESTABLISHED.** The full run has not launched because its conservative
resource projection exceeds the committed 30 GPU-hour ceiling. Approval to
increase that ceiling to 36 GPU-hours has been requested and is pending.

The target is route C-A: GPT-2 single-seam implementation agreement with pinned
SplitLLM, using 256 fresh validation articles, 112 auxiliary articles, 15,604
auxiliary windows, three seeds, depths 3/6 and public/exact grants. It does not
reproduce original SensiReplaced or the published dual-cut setting.

## Completed on koula

- Real pretrained GPT-2, FP32 eager attention, TF32 disabled and deterministic
  operations, at both depths under both grants.
- Candidate SIP logits against the upstream GRU class on identical weights.
- Exact saved-wire replay with no victim update.
- Candidate/upstream GM and SM initial variables, gradients and one AdamW update
  within the original 1e-6 absolute / 1e-5 relative tolerance.
- Full GM32 / SM800(f) / SM800(bf) resource canaries for both implementations.
- Fixed-string ROUGE-L, METEOR, token-set F1 and positional-accuracy parity at
  1e-12, using pinned upstream metric functions. The declared whitespace-only
  hypothesis canonicalisation is applied to both sides.
- Excluded GRU training timing canary, with two warm-up steps discarded and
  sixteen measured steps at the intended 32-window batch size.

Canaries used only frozen auxiliary inputs, seeds 81273/81274, untrained GRUs
and an explicitly perturbed exact client head. They are not trained-victim
results or heldout recovery evidence. Trained-checkpoint mechanical checks and
the complete matched statistical comparison remain required.

## Measured timing and resource gate

| Depth / grant | Candidate BRE seconds per document | Upstream BRE seconds per document |
| --- | ---: | ---: |
| 3 / public | 8.038 | 9.397 |
| 3 / exact | 8.167 | 9.096 |
| 6 / public | 11.865 | 13.175 |
| 6 / exact | 11.801 | 17.358 |

These timings cover BRE after SIP logits are available; they are not total
cold-start costs. Retained peak allocated GPU memory was at most about 3.04 GB
in these canaries; this excludes other processes and is not a whole-device bound.

| Projection | GPU-hours |
| --- | ---: |
| Online BRE, 256 documents × 3 seeds × both depths/grants/implementations | 18.965 |
| Six GRU training runs, 58,560 steps | 1.297 |
| Combined, before margin and other overhead | 20.261 |
| Combined with frozen 1.5× margin | **30.392** |

Victim training, public activation caching, capture/replay, checkpoint I/O,
model loading, SIP replay and canary costs are additional. The frozen resource
rule therefore does not admit the full run under 30 hours. The requested 36-hour
ceiling preserves every document, seed and attack iteration. No timing-based
reduction of scientific coverage or relaxation of fidelity tolerance was made.

## Prepared work and evidence

The [execution plan](bisr-execution-plan-2026-09-18.md) now specifies fresh shared
victim/GRU checkpoints, training budgets, observation captures and grant exports.
`experiments/bisr_checkpoint_prepare.py` implements preparation and replay without
attacking heldout inputs. After preparation, checkpoint hashes still need to be
bound into the committed prediction manifest before heldout attack generation.

Source commits: `f513ee4` (real-model canary), `f63aeeb` (checkpoint preparation
and resource plan), `9349144` (metric gate). These are local commits, not a
published fidelity certificate.

Remote evidence: `koula:~/llm-attacker-bisr-fidelity-20260918/`, including
`canary-v1/`, `training-canary-v1/`, `metric-gate-v1.json`, package inventory and
regression logs. Local evidence: `results/bisr-fidelity-canary-20260918/`.
Canary tensor traces, checkpoint perturbations and per-engine predictions are
retained; none are included in the confirmatory validation analysis.

Metric resources: rouge 1.0.1, nltk 3.9.1, transformers 5.15.0; WordNet ZIP SHA-256
`cbda5ea6eef7f36a97a43d4a75f85e07fccbb4f23657d27b4ccbc93e2646ab59`.
No manuscripts or HF evidence releases have been relabelled as fidelity-qualified.
