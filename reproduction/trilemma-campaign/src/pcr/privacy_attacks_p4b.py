"""P4b redesigned attack families (docs/PREREGISTRATION_P4B.md, frozen 2026-09-07).

Redesigns trace to the P4a failure diagnoses (docs/P4A_PRIVACY_BROADENING_DECISION.md).
Each family exposes qualify(...) (positive control + null band, frozen floors)
and scoring helpers. Floors are frozen and never tuned to P4b outcomes.

  1. sequence_property_v2: ridge regression from the mean-pooled quantized
     latent to CONTINUOUS targets (vowel fraction, punctuation fraction,
     newline count per sequence); metric = Pearson r, advantage = r - r_null
     (shuffled-label null); gate per target: r-advantage >= 0.10 at q=32 AND
     |null pipeline advantage| <= 0.02 (P0-style alignment-broken pipeline);
     family qualifies iff >= 1 target passes (all three always reported).
     Ridge lambda = 1.0 (fixed). Punctuation set: ASCII punctuation
     !"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ (fixed). Newline = byte 10 only (fixed).
  2. structure_semantic_v2: same next-byte 5-class channel as P4a but with the
     stronger probe (InversionProbe 2-hidden-layer MLP, 200 epochs, lr 3e-3)
     trained on the full dense-capture probe set; gate: q=32 advantage >= 0.02
     AND |null pipeline advantage| <= 0.02.
  3. history_accumulation_v2: frozen saturation rule — the evaluation step is
     the EARLIEST captured step whose single-step inversion accuracy is <= 0.95
     at q=32 on the calibration capture (computed from the capture, not tuned);
     K=5 consecutive captured steps; advantage = history - single; null =
     step-order-shuffled history; gate >= 0.02 and |null| <= 0.02; if no
     captured step satisfies the rule the family is DISQUALIFIED with the
     cause recorded.
  4. canary_membership_v2: 64 canaries vs 64 non-members (32 bytes, 10
     repeats, stream seed 9790); per-candidate features = concat(per-position
     probe max-probability, per-position latent mean) (64 dims); scalar score =
     mean of the 64 features (fixed reducer — the preregistration freezes the
     features; the reduction is documented here). Null band: B=50 bootstrap
     AUCs from the matched no-canary run's scores, band = mean +/- 3 SD.
     Gate: canary AUC >= 0.75 at q=32 (min over seeds) AND each null AUC
     within its derived band.
"""

from __future__ import annotations

import numpy as np
import torch

from .privacy import train_probe
from .privacy_attacks import (  # P4a machinery, reused unchanged
    _train_eval_probe,
    auc,
    build_history_features,
    next_byte_class,
    quantize_latents,
    shuffle_history_windows,
)
from .utils import derive_seed

# P5: channel axis — attacks run on forward latents AND inbound gradients
# (grad wrt remote output). Gradient features are quantized to the run's q
# before probing, matching the serialized-transcript byte accounting.
CHANNELS = ("forward_values", "inbound_gradients")
INVERSION_FLOOR = 0.05   # P0 detection floor
INVERSION_BAND = 0.02

FLOORS_P4B = {
    "sequence_property_v2": {"r_advantage": 0.10, "null_band": 0.02},
    "structure_semantic_v2": {"advantage": 0.02, "null_band": 0.02},
    "history_accumulation_v2": {"advantage": 0.02, "null_band": 0.02,
                                "saturation_ceiling": 0.95},
    "canary_membership_v2": {"auc": 0.75, "bootstrap_nulls": 50, "band_sd": 3.0},
}
RIDGE_LAMBDA = 1.0
PUNCTUATION_BYTES = set(b"!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
PROBE_EPOCHS_STRONG = 200
PROBE_LR_STRONG = 3e-3


# --------------------------------------------------------------------------- #
# family 1 v2: ridge probes on continuous sequence properties
# --------------------------------------------------------------------------- #
def sequence_targets(tokens: torch.Tensor) -> dict[str, np.ndarray]:
    """Continuous per-sequence targets (B,): vowel fraction, punctuation
    fraction, newline count."""
    tok = tokens.numpy()
    return {
        "vowel_fraction": np.isin(tok, list(b"aeiouAEIOU")).mean(axis=1),
        "punctuation_fraction": np.isin(tok, list(PUNCTUATION_BYTES)).mean(axis=1),
        "newline_count": (tok == 10).sum(axis=1).astype(np.float64),
    }


def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float = RIDGE_LAMBDA) -> dict:
    """Closed-form ridge with intercept (centered). Deterministic."""
    x_mean = X.mean(axis=0)
    y_mean = float(y.mean())
    xc = X - x_mean
    yc = y - y_mean
    d = X.shape[1]
    w = np.linalg.solve(xc.T @ xc + lam * np.eye(d), xc.T @ yc)
    return {"x_mean": x_mean, "y_mean": y_mean, "w": w}


def ridge_predict(model: dict, X: np.ndarray) -> np.ndarray:
    return (X - model["x_mean"]) @ model["w"] + model["y_mean"]


def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _split(n: int, seed: int, frac: float = 0.8):
    gen = np.random.default_rng(derive_seed("p4b-split", seed, n))
    perm = gen.permutation(n)
    cut = int(n * frac)
    return perm[:cut], perm[cut:]


def _ridge_r_and_null(feats: np.ndarray, y: np.ndarray, seed: int
                      ) -> tuple[float, float]:
    """r on the held-out split for the ridge probe and its shuffled-label null."""
    tr, te = _split(len(y), seed)
    probe = ridge_fit(feats[tr], y[tr])
    r = pearson_r(ridge_predict(probe, feats[te]), y[te])
    shuffle = np.random.default_rng(derive_seed("p4b-null", seed))
    null = ridge_fit(feats[tr], shuffle.permutation(y[tr]))
    r_null = pearson_r(ridge_predict(null, feats[te]), y[te])
    return r, r_null


def _ridge_r_and_null_matched(feats: np.ndarray, y: np.ndarray, seed: int
                              ) -> tuple[float, float]:
    """P5 variant (gradient-channel qualification): the shuffled-label null is
    evaluated on test pairs whose feature<->label alignment is ALSO broken
    (test-time permutation). Without this, a label-free ridge's noise weights
    correlate with the labels through signal-carrying test features — an
    artifact of r metrics, not of the channel. The P4b forward-channel numbers
    used the unshuffled-test null (conservative: inflates r_null, shrinks
    r_advantage; family verdicts unchanged)."""
    tr, te = _split(len(y), seed)
    probe = ridge_fit(feats[tr], y[tr])
    r = pearson_r(ridge_predict(probe, feats[te]), y[te])
    shuffle = np.random.default_rng(derive_seed("p4b-null", seed))
    null = ridge_fit(feats[tr], shuffle.permutation(y[tr]))
    te_perm = np.random.default_rng(derive_seed("p5-null-te", seed)).permutation(len(te))
    r_null = pearson_r(ridge_predict(null, feats[te][te_perm]), y[te])
    return r, r_null


def family1_v2_features(events: list[dict], q_bits: int) -> tuple[np.ndarray, dict]:
    feats, targets = [], {"vowel_fraction": [], "punctuation_fraction": [],
                          "newline_count": []}
    for event in events:
        lat = quantize_latents(event["latents"], q_bits)
        feats.append(lat.mean(dim=1).numpy())
        for name, values in sequence_targets(event["tokens"]).items():
            targets[name].append(values)
    return np.concatenate(feats), {k: np.concatenate(v) for k, v in targets.items()}


def family1_v2_gate(per_target: dict, floor: float, band: float) -> bool:
    """Frozen qualification gate: at least one target passes the r-advantage
    floor AND its null band. Pure; unit-tested directly."""
    return any(v["passes_floor"] and v["null_within_band"] for v in per_target.values())


def family1_v2_qualify(events: list[dict], seed: int, matched_test_null: bool = False
                       ) -> dict:
    """Positive control at q=32 (unquantized) + P0-style broken-pipeline null.

    matched_test_null=False reproduces the P4b forward-channel construction
    exactly; True (P5 gradient-channel qualification) additionally breaks the
    feature<->label alignment at test time for the null (see
    _ridge_r_and_null_matched).
    """
    feats, targets = family1_v2_features(events, 32)
    floor = FLOORS_P4B["sequence_property_v2"]["r_advantage"]
    band = FLOORS_P4B["sequence_property_v2"]["null_band"]
    r_fn = _ridge_r_and_null_matched if matched_test_null else _ridge_r_and_null
    per_target = {}
    for name, y in targets.items():
        r, r_null = r_fn(feats, y, seed)
        broken_gen = np.random.default_rng(derive_seed("p4b-alignment-break", seed, name))
        r_broken, r_broken_null = r_fn(feats[broken_gen.permutation(len(y))], y, seed)
        null_adv = r_broken - r_broken_null
        per_target[name] = {
            "r": r, "r_null": r_null, "r_advantage": r - r_null,
            "null_pipeline_advantage": null_adv,
            "passes_floor": bool(r - r_null >= floor),
            "null_within_band": bool(abs(null_adv) <= band),
        }
    qualified = family1_v2_gate(per_target, floor, band)
    return {"family": "sequence_property_v2", "floor": floor, "null_band": band,
            "ridge_lambda": RIDGE_LAMBDA, "matched_test_null": matched_test_null,
            "per_target": per_target,
            "qualified": bool(qualified)}


def family1_v2_train(events: list[dict], q_bits: int, seed: int) -> dict:
    """Ridge probes + shuffled nulls on calibration events (shared per cell)."""
    feats, targets = family1_v2_features(events, q_bits)
    out = {}
    for name, y in targets.items():
        shuffle = np.random.default_rng(derive_seed("p4b-null-score", seed, name))
        out[name] = {"probe": ridge_fit(feats, y),
                     "null": ridge_fit(feats, shuffle.permutation(y))}
    return out


def family1_v2_apply(probes: dict, run_latents: torch.Tensor, run_tokens: torch.Tensor,
                     q_bits: int, matched_test_null: bool = False) -> dict:
    feats = quantize_latents(run_latents, q_bits).mean(dim=1).numpy()
    targets = sequence_targets(run_tokens)
    out = {}
    for name, pair in probes.items():
        y = targets[name]
        r = pearson_r(ridge_predict(pair["probe"], feats), y)
        if matched_test_null:
            # P5: null evaluated on alignment-broken test pairs (see
            # _ridge_r_and_null_matched for the rationale)
            perm = np.random.default_rng(derive_seed("p5-null-te-score", name,
                                                     len(y))).permutation(len(y))
            r_null = pearson_r(ridge_predict(pair["null"], feats[perm]), y)
        else:
            r_null = pearson_r(ridge_predict(pair["null"], feats), y)
        out[name] = {"r": r, "r_null": r_null, "r_advantage": r - r_null}
    return out


# --------------------------------------------------------------------------- #
# family 2 v2: stronger probe on dense captures
# --------------------------------------------------------------------------- #
def dense_samples_to_events(samples: list[dict], tokens_per_batch: list[torch.Tensor],
                            channel: str = "forward_values",
                            q_bits: int | None = None) -> list[dict]:
    """Convert dense transcript samples to {latents, tokens} events. Each
    sample may carry multiple probe batches (probe_batch_index). P5: `channel`
    selects the transcript tensor (forward_values or inbound_gradients);
    q_bits quantizes channel tensors on the fly (matching serialized
    accounting)."""
    events = []
    by_step: dict[int, dict[int, dict]] = {}
    for s in samples:
        by_step.setdefault(s["step"], {})[s.get("probe_batch_index", 0)] = s
    for step in sorted(by_step):
        for batch_index in sorted(by_step[step]):
            latents = by_step[step][batch_index][channel]
            if q_bits is not None:
                latents = quantize_latents(latents, q_bits)
            events.append({"step": step, "latents": latents,
                           "tokens": tokens_per_batch[batch_index]})
    return events


def family2_v2_dataset(events: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Dense captures are already quantized at the run's q; use as-is."""
    feats, labels = [], []
    for event in events:
        lat = event["latents"]
        bsz, seq, width = lat.shape
        nxt = np.roll(event["tokens"].numpy(), -1, axis=1)
        feats.append(lat.reshape(bsz * seq, width).numpy())
        labels.append(next_byte_class(nxt).reshape(-1))
    return np.concatenate(feats), np.concatenate(labels)


def _train_eval_strong(features: np.ndarray, labels: np.ndarray, n_classes: int,
                       seed: int, device: torch.device) -> tuple[float, float]:
    """The frozen stronger probe: 200 epochs, lr 3e-3."""
    tr, te = _split(len(labels), seed)
    probe = train_probe(features[tr], labels[tr], features.shape[1], PROBE_EPOCHS_STRONG,
                        4096, PROBE_LR_STRONG, seed, device)
    shuffle = np.random.default_rng(derive_seed("p4b-null", seed))
    null = train_probe(features[tr], shuffle.permutation(labels[tr]), features.shape[1],
                       PROBE_EPOCHS_STRONG, 4096, PROBE_LR_STRONG, seed, device)
    probe.eval()
    null.eval()
    with torch.no_grad():
        acc = (probe(torch.from_numpy(features[te]).to(device)).argmax(-1).cpu().numpy()
               == labels[te]).mean()
        null_acc = (null(torch.from_numpy(features[te]).to(device)).argmax(-1).cpu().numpy()
                    == labels[te]).mean()
    return float(acc), float(null_acc)


def family2_v2_qualify(dense_events: list[dict], seed: int, device: torch.device) -> dict:
    feats, labels = family2_v2_dataset(dense_events)
    acc, null_acc = _train_eval_strong(feats, labels, 5, seed, device)
    broken_gen = np.random.default_rng(derive_seed("p4b-alignment-break", seed))
    broken_acc, broken_null = _train_eval_strong(feats[broken_gen.permutation(len(labels))],
                                                 labels, 5, seed, device)
    null_adv = broken_acc - broken_null
    floor = FLOORS_P4B["structure_semantic_v2"]["advantage"]
    band = FLOORS_P4B["structure_semantic_v2"]["null_band"]
    return {"family": "structure_semantic_v2", "floor": floor, "null_band": band,
            "accuracy": acc, "null_accuracy": null_acc, "advantage": acc - null_acc,
            "null_pipeline_advantage": null_adv,
            "passes_floor": bool(acc - null_acc >= floor),
            "null_within_band": bool(abs(null_adv) <= band),
            "qualified": bool(acc - null_acc >= floor and abs(null_adv) <= band)}


def family2_v2_train(events_919: list[dict], q_bits: int, seed: int,
                     device: torch.device) -> dict:
    """Stronger probe trained on the width-matched 919 calibration, quantized
    on the fly to the cell's q (scoring instrument)."""
    feats, labels = [], []
    for event in events_919:
        lat = quantize_latents(event["latents"], q_bits)
        bsz, seq, width = lat.shape
        nxt = np.roll(event["tokens"].numpy(), -1, axis=1)
        feats.append(lat.reshape(bsz * seq, width).numpy())
        labels.append(next_byte_class(nxt).reshape(-1))
    feats = np.concatenate(feats)
    labels = np.concatenate(labels)
    # train_probe builds InversionProbe(width) with the fixed 256-way head
    # (pcr.privacy); labels use 5 classes within it — consistent with P4a
    probe = train_probe(feats, labels, feats.shape[1], PROBE_EPOCHS_STRONG, 4096,
                        PROBE_LR_STRONG, seed, device)
    shuffle = np.random.default_rng(derive_seed("p4b-null-score", seed))
    null = train_probe(feats, shuffle.permutation(labels), feats.shape[1],
                       PROBE_EPOCHS_STRONG, 4096, PROBE_LR_STRONG, seed, device)
    return {"probe": probe, "null": null}


def family2_v2_apply(probes: dict, run_latents: torch.Tensor, run_tokens: torch.Tensor,
                     q_bits: int, device: torch.device) -> dict:
    run_lat = quantize_latents(run_latents, q_bits)
    bsz, seq, width = run_lat.shape
    run_feats = run_lat.reshape(bsz * seq, width).numpy()
    run_labels = next_byte_class(np.roll(run_tokens.numpy(), -1, axis=1)).reshape(-1)
    probe, null = probes["probe"], probes["null"]
    probe.eval()
    null.eval()
    with torch.no_grad():
        acc = (probe(torch.from_numpy(run_feats).to(device)).argmax(-1).cpu().numpy()
               == run_labels).mean()
        null_acc = (null(torch.from_numpy(run_feats).to(device)).argmax(-1).cpu().numpy()
                    == run_labels).mean()
    return {"accuracy": float(acc), "null_accuracy": float(null_acc),
            "advantage": float(acc - null_acc)}


# --------------------------------------------------------------------------- #
# family 3 v2: saturation-ruled history accumulation
# --------------------------------------------------------------------------- #
def single_step_accuracy_by_step(dense_latents: torch.Tensor, tokens: np.ndarray,
                                 seed: int, device: torch.device,
                                 epochs: int = 20) -> dict[int, float]:
    """Per captured step: single-step inversion accuracy (80/20 split)."""
    out = {}
    n_steps = dense_latents.shape[0]
    labels = np.tile(tokens.reshape(-1), 1)
    for i in range(n_steps):
        feats = dense_latents[i].reshape(-1, dense_latents.shape[-1]).numpy()
        tr, te = _split(len(labels), seed + i)
        probe = train_probe(feats[tr], labels[tr], feats.shape[1], epochs, 4096, 1e-3,
                            seed, device)
        probe.eval()
        with torch.no_grad():
            acc = (probe(torch.from_numpy(feats[te]).to(device)).argmax(-1).cpu().numpy()
                   == labels[te]).mean()
        out[i] = float(acc)
    return out


def saturation_step(accuracies: dict[int, float], captured_steps: list[int],
                    ceiling: float = 0.95) -> tuple[int | None, str | None]:
    """Frozen rule: earliest captured step with single-step accuracy <= ceiling."""
    for idx, step in enumerate(captured_steps):
        if accuracies[idx] <= ceiling:
            return idx, None
    return None, (f"no captured step with single-step accuracy <= {ceiling} "
                  f"(min observed {min(accuracies.values()):.4f})")


def family3_v2_evaluate_at(dense_latents: torch.Tensor, tokens: np.ndarray,
                           step_index: int, seed: int, device: torch.device,
                           k: int = 5, epochs: int = 20) -> dict:
    """History (K consecutive captured steps) vs single-step at step_index."""
    n_steps, bsz, seq, width = dense_latents.shape
    history_all, single_all = build_history_features(dense_latents, k)
    n_windows = n_steps - k + 1
    labels = np.tile(tokens.reshape(-1), n_windows)
    per_window = bsz * seq
    sl = slice(step_index * per_window, (step_index + 1) * per_window)
    history = history_all[sl]
    single = single_all[sl]
    y = labels[sl]
    tr, te = _split(len(y), seed)
    probe_h = train_probe(history[tr], y[tr], history.shape[1], epochs, 4096, 1e-3,
                          seed, device)
    probe_s = train_probe(single[tr], y[tr], width, epochs, 4096, 1e-3, seed, device)
    shuffled = shuffle_history_windows(history, k, width, seed)
    probe_n = train_probe(shuffled[tr], y[tr], history.shape[1], epochs, 4096, 1e-3,
                          seed, device)
    for probe in (probe_h, probe_s, probe_n):
        probe.eval()
    with torch.no_grad():
        acc_h = (probe_h(torch.from_numpy(history[te]).to(device)).argmax(-1).cpu().numpy()
                 == y[te]).mean()
        acc_s = (probe_s(torch.from_numpy(single[te]).to(device)).argmax(-1).cpu().numpy()
                 == y[te]).mean()
        acc_n = (probe_n(torch.from_numpy(shuffled[te]).to(device)).argmax(-1).cpu().numpy()
                 == y[te]).mean()
    return {"accuracy_history": float(acc_h), "accuracy_single": float(acc_s),
            "accuracy_null_shuffled": float(acc_n),
            "advantage": float(acc_h - acc_s),
            "null_advantage": float(acc_n - acc_s)}


def family3_v2_qualify(dense_by_run: dict[str, torch.Tensor], tokens: np.ndarray,
                       seed: int, device: torch.device,
                       captured_steps: list[int]) -> dict:
    """Frozen saturation rule per run; gate on min advantage / max |null|."""
    floor = FLOORS_P4B["history_accumulation_v2"]["advantage"]
    band = FLOORS_P4B["history_accumulation_v2"]["null_band"]
    ceiling = FLOORS_P4B["history_accumulation_v2"]["saturation_ceiling"]
    per_run = {}
    causes = []
    for run_id, latents in dense_by_run.items():
        accuracies = single_step_accuracy_by_step(latents, tokens, seed, device)
        idx, cause = saturation_step(accuracies, captured_steps, ceiling)
        entry = {"accuracies_by_capture": accuracies, "saturation_index": idx,
                 "cause": cause}
        if idx is not None:
            entry.update(family3_v2_evaluate_at(latents, tokens, idx, seed, device))
            entry["passes_floor"] = bool(entry["advantage"] >= floor)
            entry["null_within_band"] = bool(abs(entry["null_advantage"]) <= band)
        per_run[run_id] = entry
        if cause:
            causes.append(f"{run_id}: {cause}")
    evaluated = [r for r in per_run.values() if r["saturation_index"] is not None]
    if not evaluated:
        return {"family": "history_accumulation_v2", "floor": floor, "null_band": band,
                "saturation_ceiling": ceiling, "per_run": per_run,
                "qualified": False, "disqualify_cause": "; ".join(causes)}
    min_adv = min(r["advantage"] for r in evaluated)
    max_null = max(abs(r["null_advantage"]) for r in evaluated)
    qualified = bool(len(evaluated) == len(per_run) and min_adv >= floor and max_null <= band)
    return {"family": "history_accumulation_v2", "floor": floor, "null_band": band,
            "saturation_ceiling": ceiling, "per_run": per_run,
            "min_advantage": min_adv, "max_abs_null_advantage": max_null,
            "qualified": qualified,
            "disqualify_cause": "; ".join(causes) if causes else None}


# --------------------------------------------------------------------------- #
# family 4 v2: canary membership with bootstrap null band
# --------------------------------------------------------------------------- #
def canary_features(probe, latents: torch.Tensor, device: torch.device) -> np.ndarray:
    """Per candidate: concat(per-position probe max-prob, per-position latent
    mean) -> (N, 2T)."""
    probe.eval()
    with torch.no_grad():
        probs = torch.softmax(probe(latents.to(device)), dim=-1)
        maxprob = probs.max(dim=-1).values.cpu().numpy()     # (N, T)
    latent_mean = latents.mean(dim=-1).cpu().numpy()          # (N, T)
    return np.concatenate([maxprob, latent_mean], axis=1)


def canary_scores(probe, latents: torch.Tensor, device: torch.device) -> np.ndarray:
    """Fixed scalar reducer: mean of the concatenated feature vector."""
    return canary_features(probe, latents, device).mean(axis=1)


def bootstrap_null_aucs(null_member_scores: np.ndarray, null_non_member_scores: np.ndarray,
                        n_boot: int = 50, seed: int = 0) -> dict:
    """B bootstrap AUCs from the no-canary run's scores (resample with
    replacement within each label); band = mean +/- 3 SD."""
    gen = np.random.default_rng(derive_seed("p4b-bootstrap", seed))
    aucs = []
    n_m, n_n = len(null_member_scores), len(null_non_member_scores)
    for _ in range(n_boot):
        m = null_member_scores[gen.integers(0, n_m, n_m)]
        nn = null_non_member_scores[gen.integers(0, n_n, n_n)]
        aucs.append(auc(m.tolist(), nn.tolist()))
    arr = np.asarray(aucs)
    return {"bootstrap_aucs": arr.tolist(), "mean": float(arr.mean()),
            "sd": float(arr.std(ddof=0)),
            "band_low": float(arr.mean() - 3.0 * arr.std(ddof=0)),
            "band_high": float(arr.mean() + 3.0 * arr.std(ddof=0))}


def family4_v2_score_run(probe, canaries: dict, null_canaries: dict,
                         seed: int, device: torch.device) -> dict:
    member_scores = canary_scores(probe, canaries["member_latents"], device)
    non_member_scores = canary_scores(probe, canaries["non_member_latents"], device)
    null_m = canary_scores(probe, null_canaries["member_latents"], device)
    null_n = canary_scores(probe, null_canaries["non_member_latents"], device)
    band = bootstrap_null_aucs(null_m, null_n,
                               n_boot=FLOORS_P4B["canary_membership_v2"]["bootstrap_nulls"],
                               seed=seed)
    null_auc = auc(null_m.tolist(), null_n.tolist())
    return {"auc": auc(member_scores.tolist(), non_member_scores.tolist()),
            "null_auc": null_auc, "null_band": band,
            "null_within_band": bool(band["band_low"] <= null_auc <= band["band_high"]),
            "mean_score_members": float(member_scores.mean()),
            "mean_score_non_members": float(non_member_scores.mean())}


# --------------------------------------------------------------------------- #
# P5: token-inversion on an arbitrary channel (gradient-channel attacks)
# --------------------------------------------------------------------------- #
def inversion_channel_dataset(events: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Per-position (features, token labels) from channel events."""
    feats, labels = [], []
    for event in events:
        lat = event["latents"]
        bsz, seq, width = lat.shape
        feats.append(lat.reshape(bsz * seq, width).numpy())
        labels.append(event["tokens"].numpy().reshape(-1).astype(np.int64))
    return np.concatenate(feats), np.concatenate(labels)


def inversion_channel_qualify(events: list[dict], seed: int, device: torch.device,
                              epochs: int = 20, batch_size: int = 4096) -> dict:
    """P0 token-inversion instrument applied to a channel (e.g. inbound
    gradients). Floor 0.05 (P0 detection floor), |null pipeline| <= 0.02."""
    feats, labels = inversion_channel_dataset(events)
    acc, null_acc = _train_eval_probe(feats, labels, 256, seed, epochs, device,
                                      batch_size=batch_size)
    broken_gen = np.random.default_rng(derive_seed("p5-alignment-break", seed))
    broken_acc, broken_null = _train_eval_probe(
        feats[broken_gen.permutation(len(labels))], labels, 256, seed, epochs, device,
        batch_size=batch_size)
    null_adv = broken_acc - broken_null
    return {"family": "token_inversion", "floor": INVERSION_FLOOR,
            "null_band": INVERSION_BAND,
            "accuracy": acc, "null_accuracy": null_acc, "advantage": acc - null_acc,
            "null_pipeline_advantage": null_adv,
            "passes_floor": bool(acc - null_acc >= INVERSION_FLOOR),
            "null_within_band": bool(abs(null_adv) <= INVERSION_BAND),
            "qualified": bool(acc - null_acc >= INVERSION_FLOOR
                              and abs(null_adv) <= INVERSION_BAND)}


def inversion_channel_train(events: list[dict], seed: int, device: torch.device,
                            epochs: int = 20, batch_size: int = 4096) -> dict:
    """Train inversion probe + shuffled null on channel events (shared per
    cell); P0 budget (20 epochs, lr 1e-3)."""
    feats, labels = inversion_channel_dataset(events)
    probe = train_probe(feats, labels, feats.shape[1], epochs, batch_size, 1e-3,
                        seed, device)
    shuffle = np.random.default_rng(derive_seed("p5-null-score", seed))
    null = train_probe(feats, shuffle.permutation(labels), feats.shape[1], epochs,
                     batch_size, 1e-3, seed, device)
    return {"probe": probe, "null": null}


def inversion_channel_apply(bank: dict, latents: torch.Tensor, tokens: torch.Tensor,
                            device: torch.device) -> dict:
    feats = latents.reshape(-1, latents.shape[-1]).numpy()
    labels = tokens.numpy().reshape(-1).astype(np.int64)
    probe, null = bank["probe"], bank["null"]
    probe.eval()
    null.eval()
    with torch.no_grad():
        acc = (probe(torch.from_numpy(feats).to(device)).argmax(-1).cpu().numpy()
               == labels).mean()
        null_acc = (null(torch.from_numpy(feats).to(device)).argmax(-1).cpu().numpy()
                    == labels).mean()
    return {"accuracy": float(acc), "null_accuracy": float(null_acc),
            "advantage": float(acc - null_acc)}
