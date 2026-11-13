"""P4a attack families (docs/PREREGISTRATION_P4A.md, frozen 2026-09-07).

Four families, all operating on the declared remote-visible transcript
(quantized boundary latents). Each exposes `qualify(...)` (positive control +
matched null band, frozen floors) and `score(...)`. Reuses the P0
inversion-probe machinery (pcr.privacy.InversionProbe / train_probe).

  1. sequence_property: predict a sequence-level property of the probe input
     (vowel-fraction bucket, 4 classes; digit presence, balanced accuracy)
     from the mean-pooled quantized latent. Null: latents shuffled w.r.t.
     sequences. Floor: advantage >= 0.05 on unquantized (q=32) calibration
     latents. Vowel-bucket edges are fixed absolute thresholds
     [0, 0.25, 0.35, 0.45, 1.0] (the preregistration leaves the bucketization
     unspecified; this choice is documented here and in the outputs).
  2. structure_semantic: predict the class of the NEXT byte after each
     position (newline / space / letter / digit / punctuation; 5 classes)
     from the quantized latent at that position. Accuracy; shuffled null;
     floor advantage >= 0.05 at q=32.
  3. history_accumulation: token-inversion probe given K=5 consecutive dense
     captures (latents at steps t-K+1..t concatenated) vs single-step.
     Metric: per-position top-1 accuracy advantage of history over
     single-step. Null: history windows with step order shuffled (advantage
     of shuffled history over single-step). Floor: >= 0.02 at q=32; null band
     |null advantage| <= 0.02.
  4. canary_membership: score candidate strings by mean per-position
     max-probability of a token-inversion probe (trained on dense calibration
     captures at the matching q); AUC members vs non-members. Null: same
     scoring on latents from matched no-canary runs. Floor: canary AUC > 0.9
     at q=32; null |AUC - 0.5| <= 0.02.

No estimator output here is a mutual-information bound or a privacy
certificate; each family is one measured advantage over a matched null.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .privacy import InversionProbe, train_probe
from .split import AffineQuantizer
from .utils import derive_seed

FLOORS = {
    "sequence_property": {"positive_control_advantage": 0.05, "null_band": 0.02},
    "structure_semantic": {"positive_control_advantage": 0.05, "null_band": 0.02},
    "history_accumulation": {"positive_control_advantage": 0.02, "null_band": 0.02},
    "canary_membership": {"positive_control_auc": 0.9, "null_auc_band": 0.02},
}
HISTORY_K = 5
VOWEL_BUCKET_EDGES = (0.0, 0.25, 0.35, 0.45, 1.0)  # fixed; documented
NEXT_BYTE_CLASSES = ("newline", "space", "letter", "digit", "punctuation")


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def _to_numpy(events_latents: torch.Tensor) -> np.ndarray:
    return events_latents.detach().to("cpu", torch.float32).numpy()


def quantize_latents(latents: torch.Tensor, q_bits: int) -> torch.Tensor:
    return AffineQuantizer(q_bits).quantize_dequantize(latents)


def _split(n: int, seed: int, frac: float = 0.8):
    gen = np.random.default_rng(derive_seed("p4a-split", seed, n))
    perm = gen.permutation(n)
    cut = int(n * frac)
    return perm[:cut], perm[cut:]


def _train_eval_probe(features: np.ndarray, labels: np.ndarray, n_classes: int,
                      seed: int, epochs: int, device: torch.device,
                      batch_size: int = 4096, lr: float = 1e-3) -> tuple[float, float]:
    """Train probe + shuffled-label null; return (accuracy, null_accuracy)."""
    width = features.shape[1]
    tr, te = _split(len(labels), seed)
    probe = train_probe(features[tr], labels[tr], width, epochs, batch_size, lr,
                        seed, device)
    shuffle = np.random.default_rng(derive_seed("p4a-null", seed))
    null = train_probe(features[tr], shuffle.permutation(labels[tr]), width,
                       epochs, batch_size, lr, seed, device)
    probe.eval()
    null.eval()
    with torch.no_grad():
        acc = (probe(torch.from_numpy(features[te]).to(device)).argmax(-1).cpu().numpy()
               == labels[te]).mean()
        null_acc = (null(torch.from_numpy(features[te]).to(device)).argmax(-1).cpu().numpy()
                    == labels[te]).mean()
    return float(acc), float(null_acc)


def balanced_accuracy(labels: np.ndarray, preds: np.ndarray) -> float:
    classes = np.unique(labels)
    recalls = []
    for c in classes:
        mask = labels == c
        recalls.append(float((preds[mask] == c).mean()) if mask.any() else 0.0)
    return float(np.mean(recalls))


def auc(member_scores: list[float], non_member_scores: list[float]) -> float:
    """Mann-Whitney AUC with tie-averaged ranks."""
    scores = list(member_scores) + list(non_member_scores)
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n_pos, n_neg = len(member_scores), len(non_member_scores)
    sum_pos = sum(ranks[:n_pos])
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


# --------------------------------------------------------------------------- #
# family 1: sequence property
# --------------------------------------------------------------------------- #
def sequence_property_labels(tokens: torch.Tensor) -> dict[str, np.ndarray]:
    """Per-sequence labels from input tokens (B, T): vowel-fraction bucket
    (fixed edges) and digit presence."""
    tok = tokens.numpy()
    vowels = np.isin(tok, list(b"aeiouAEIOU")).mean(axis=1)
    bucket = np.digitize(vowels, VOWEL_BUCKET_EDGES[1:-1])  # 0..3
    digit = np.isin(tok, list(b"0123456789")).any(axis=1).astype(np.int64)
    return {"vowel_fraction_bucket": bucket.astype(np.int64), "digit_presence": digit}


def family1_features(events: list[dict], q_bits: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Mean-pooled latents per sequence + labels. events carry
    (latents (B,T,W), tokens (B,T)); quantized on the fly to q_bits."""
    feats, labels = [], {"vowel_fraction_bucket": [], "digit_presence": []}
    for event in events:
        lat = quantize_latents(event["latents"], q_bits)
        feats.append(lat.mean(dim=1).numpy())  # (B, W)
        for name, values in sequence_property_labels(event["tokens"]).items():
            labels[name].append(values)
    return (np.concatenate(feats),
            {k: np.concatenate(v) for k, v in labels.items()})


def family1_gate(per_property: dict, floor: float, band: float) -> bool:
    """Frozen qualification gate for family 1: at least one sub-property passes
    the positive-control floor AND its null band. Pure; unit-tested directly."""
    return any(v["passes_floor"] and v["null_within_band"] for v in per_property.values())


def family1_qualify(events: list[dict], seed: int, epochs: int,
                    device: torch.device, batch_size: int = 4096) -> dict:
    """Positive control on UNQUANTIZED (q=32) calibration latents + null band.

    Per sub-property: advantage = accuracy - shuffled-null accuracy (floor
    0.05); the null band mirrors P0's matched-null construction: run the same
    probe/null pipeline on alignment-broken data (sequence-level permutation
    of features w.r.t. labels); the broken pipeline's advantage must satisfy
    |advantage| <= 0.02. The preregistration's floor speaks of "a
    sequence-level property ... must be detectable", so the family is
    QUALIFIED iff AT LEAST ONE sub-property passes both gates; a degenerate
    sub-property (e.g. one class on the measured data) is named and still
    reported. Both sub-properties are always visible.
    """
    feats, labels = family1_features(events, 32)
    floor = FLOORS["sequence_property"]["positive_control_advantage"]
    band = FLOORS["sequence_property"]["null_band"]
    per_prop = {}
    for name, y in labels.items():
        n_classes = int(y.max()) + 1
        acc, null_acc = _train_eval_probe(feats, y, n_classes, seed, epochs, device,
                                          batch_size=batch_size)
        broken_gen = np.random.default_rng(derive_seed("p4a-alignment-break", seed, name))
        broken_feats = feats[broken_gen.permutation(len(y))]
        broken_acc, broken_null = _train_eval_probe(
            broken_feats, y, n_classes, seed, epochs, device, batch_size=batch_size)
        null_adv = broken_acc - broken_null
        per_prop[name] = {
            "accuracy": acc, "null_accuracy": null_acc, "advantage": acc - null_acc,
            "majority_rate": float(max(np.bincount(y)) / len(y)),
            "null_pipeline_advantage": null_adv,
            "passes_floor": bool(acc - null_acc >= floor),
            "null_within_band": bool(abs(null_adv) <= band),
        }
    qualified = family1_gate(per_prop, floor, band)
    return {"family": "sequence_property", "floor": floor, "null_band": band,
            "qualification_rule": "at least one sub-property passes floor and null band",
            "per_property": per_prop, "qualified": bool(qualified)}


def family1_train(events: list[dict], q_bits: int, seed: int, epochs: int,
                  device: torch.device, batch_size: int = 4096) -> dict:
    """Train probe + null per sub-property on calibration events (shared across
    runs: the calibration data and q depend only on the cell, not the seed)."""
    feats, labels = family1_features(events, q_bits)
    probes = {}
    for name, y in labels.items():
        n_classes = int(y.max()) + 1
        probe = train_probe(feats, y, n_classes, epochs, batch_size, 1e-3, seed, device)
        shuffle = np.random.default_rng(derive_seed("p4a-null-score", seed, name))
        null = train_probe(feats, shuffle.permutation(y), n_classes, epochs,
                           batch_size, 1e-3, seed, device)
        probes[name] = {"probe": probe, "null": null}
    return probes


def family1_apply(probes: dict, run_latents: torch.Tensor, run_tokens: torch.Tensor,
                  q_bits: int, device: torch.device) -> dict:
    run_feats = quantize_latents(run_latents, q_bits).mean(dim=1).numpy()
    run_labels = sequence_property_labels(run_tokens)
    out = {}
    for name, pair in probes.items():
        probe, null = pair["probe"], pair["null"]
        probe.eval()
        null.eval()
        with torch.no_grad():
            p_acc = (probe(torch.from_numpy(run_feats).to(device)).argmax(-1).cpu().numpy()
                     == run_labels[name]).mean()
            n_acc = (null(torch.from_numpy(run_feats).to(device)).argmax(-1).cpu().numpy()
                     == run_labels[name]).mean()
        out[name] = {"accuracy": float(p_acc), "null_accuracy": float(n_acc),
                     "advantage": float(p_acc - n_acc)}
    return out


def family1_score(events: list[dict], q_bits: int, run_latents: torch.Tensor,
                  run_tokens: torch.Tensor, seed: int, epochs: int,
                  device: torch.device) -> dict:
    """Score a run's final-step latents: probe trained on calibration events
    (quantized on the fly to q_bits), applied to the run's sequences."""
    probes = family1_train(events, q_bits, seed, epochs, device)
    return family1_apply(probes, run_latents, run_tokens, q_bits, device)


# --------------------------------------------------------------------------- #
# family 2: structure / semantic (next-byte class per position)
# --------------------------------------------------------------------------- #
def next_byte_class(next_bytes: np.ndarray) -> np.ndarray:
    """5 classes: newline / space / letter / digit / punctuation."""
    out = np.full(next_bytes.shape, 4, dtype=np.int64)  # punctuation
    out[np.isin(next_bytes, list(b"\n\r"))] = 0
    out[np.isin(next_bytes, list(b" \t"))] = 1
    out[np.isin(next_bytes, list(b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"))] = 2
    out[np.isin(next_bytes, list(b"0123456789"))] = 3
    return out


def family2_dataset(events: list[dict], q_bits: int):
    """(features (N,W), labels (N,)) — label of position t is class(next byte)."""
    feats, labels = [], []
    for event in events:
        lat = quantize_latents(event["latents"], q_bits)
        bsz, seq, width = lat.shape
        nxt = np.roll(event["tokens"].numpy(), -1, axis=1)  # next byte at each position
        feats.append(lat.reshape(bsz * seq, width).numpy())
        labels.append(next_byte_class(nxt).reshape(-1))
    return np.concatenate(feats), np.concatenate(labels)


def family2_qualify(events: list[dict], seed: int, epochs: int,
                    device: torch.device, batch_size: int = 4096) -> dict:
    feats, labels = family2_dataset(events, 32)
    acc, null_acc = _train_eval_probe(feats, labels, len(NEXT_BYTE_CLASSES),
                                      seed, epochs, device, batch_size=batch_size)
    # matched-null pipeline (P0 construction): position-level alignment break,
    # broken pipeline's advantage must be ~0
    broken_gen = np.random.default_rng(derive_seed("p4a-alignment-break", seed))
    broken_acc, broken_null = _train_eval_probe(
        feats[broken_gen.permutation(len(labels))], labels, len(NEXT_BYTE_CLASSES),
        seed, epochs, device, batch_size=batch_size)
    null_adv = broken_acc - broken_null
    floor = FLOORS["structure_semantic"]["positive_control_advantage"]
    band = FLOORS["structure_semantic"]["null_band"]
    return {"family": "structure_semantic", "floor": floor, "null_band": band,
            "accuracy": acc, "null_accuracy": null_acc, "advantage": acc - null_acc,
            "null_pipeline_advantage": null_adv,
            "null_within_band": bool(abs(null_adv) <= band),
            "passes_floor": bool(acc - null_acc >= floor),
            "qualified": bool(acc - null_acc >= floor and abs(null_adv) <= band)}


def family2_train(events: list[dict], q_bits: int, seed: int, epochs: int,
                  device: torch.device, batch_size: int = 4096) -> dict:
    """Train probe + null on calibration events (shared across runs of a cell)."""
    feats, labels = family2_dataset(events, q_bits)
    probe = train_probe(feats, labels, len(NEXT_BYTE_CLASSES), epochs, batch_size,
                        1e-3, seed, device)
    shuffle = np.random.default_rng(derive_seed("p4a-null-score", seed))
    null = train_probe(feats, shuffle.permutation(labels), len(NEXT_BYTE_CLASSES),
                       epochs, batch_size, 1e-3, seed, device)
    return {"probe": probe, "null": null}


def family2_apply(probes: dict, run_latents: torch.Tensor, run_tokens: torch.Tensor,
                  q_bits: int, device: torch.device) -> dict:
    run_lat = quantize_latents(run_latents, q_bits)
    bsz, seq, width = run_lat.shape
    run_feats = run_lat.reshape(bsz * seq, width).numpy()
    nxt = np.roll(run_tokens.numpy(), -1, axis=1)
    run_labels = next_byte_class(nxt).reshape(-1)
    probe, null = probes["probe"], probes["null"]
    probe.eval()
    null.eval()
    with torch.no_grad():
        p_acc = (probe(torch.from_numpy(run_feats).to(device)).argmax(-1).cpu().numpy()
                 == run_labels).mean()
        n_acc = (null(torch.from_numpy(run_feats).to(device)).argmax(-1).cpu().numpy()
                 == run_labels).mean()
    return {"accuracy": float(p_acc), "null_accuracy": float(n_acc),
            "advantage": float(p_acc - n_acc)}


def family2_score(events: list[dict], q_bits: int, run_latents: torch.Tensor,
                  run_tokens: torch.Tensor, seed: int, epochs: int,
                  device: torch.device) -> dict:
    probes = family2_train(events, q_bits, seed, epochs, device)
    return family2_apply(probes, run_latents, run_tokens, q_bits, device)


# --------------------------------------------------------------------------- #
# family 3: history accumulation (K=5 consecutive dense captures)
# --------------------------------------------------------------------------- #
def build_history_features(dense_latents: torch.Tensor, k: int = HISTORY_K,
                           ) -> tuple[np.ndarray, np.ndarray]:
    """dense_latents: (S, B, T, W) stacked consecutive-step latents.
    Returns (history (N, k*W), single (N, W)) with N = (S-k+1)*B*T."""
    n_steps = dense_latents.shape[0]
    lat = dense_latents.numpy()
    hist, single = [], []
    for t in range(k - 1, n_steps):
        window = lat[t - k + 1: t + 1]                       # (k, B, T, W)
        hist.append(np.concatenate(list(window), axis=-1))   # (B, T, k*W)
        single.append(lat[t])                                # (B, T, W)
    h = np.concatenate(hist)          # (N, k*W) with B,T folded
    s = np.concatenate(single)
    bsz, seq = dense_latents.shape[1], dense_latents.shape[2]
    h = h.reshape(-1, h.shape[-1])
    s = s.reshape(-1, s.shape[-1])
    return h, s


def shuffle_history_windows(history: np.ndarray, k: int, width: int,
                            seed: int) -> np.ndarray:
    """Same latents, step order within each window shuffled (seeded)."""
    n = history.shape[0]
    gen = np.random.default_rng(derive_seed("p4a-history-shuffle", seed))
    windows = history.reshape(n, k, width)
    order = np.argsort(gen.random((n, k)), axis=1)
    shuffled = np.take_along_axis(windows, order[:, :, None], axis=1)
    return shuffled.reshape(n, k * width)


def family3_evaluate(dense_latents: torch.Tensor, tokens: np.ndarray, seed: int,
                     epochs: int, device: torch.device,
                     k: int = HISTORY_K) -> dict:
    """Accuracy of history vs single-step features vs shuffled-history null."""
    n_steps, bsz, seq, width = dense_latents.shape
    history, single = build_history_features(dense_latents, k)
    labels = np.tile(tokens.reshape(-1), n_steps - k + 1)
    tr, te = _split(len(labels), seed)
    probe_h = train_probe(history[tr], labels[tr], history.shape[1], epochs, 4096,
                          1e-3, seed, device)
    probe_s = train_probe(single[tr], labels[tr], width, epochs, 4096, 1e-3, seed,
                          device)
    shuffled = shuffle_history_windows(history, k, width, seed)
    probe_n = train_probe(shuffled[tr], labels[tr], history.shape[1], epochs, 4096,
                          1e-3, seed, device)
    for probe in (probe_h, probe_s, probe_n):
        probe.eval()
    with torch.no_grad():
        acc_h = (probe_h(torch.from_numpy(history[te]).to(device)).argmax(-1).cpu().numpy()
                 == labels[te]).mean()
        acc_s = (probe_s(torch.from_numpy(single[te]).to(device)).argmax(-1).cpu().numpy()
                 == labels[te]).mean()
        acc_n = (probe_n(torch.from_numpy(shuffled[te]).to(device)).argmax(-1).cpu().numpy()
                 == labels[te]).mean()
    return {"accuracy_history": float(acc_h), "accuracy_single": float(acc_s),
            "accuracy_null_shuffled": float(acc_n),
            "advantage": float(acc_h - acc_s),
            "null_advantage": float(acc_n - acc_s)}


def family3_qualify(dense_events_by_run: dict[str, torch.Tensor],
                    tokens: np.ndarray, seed: int, epochs: int,
                    device: torch.device) -> dict:
    """Positive control on the q=32 dense captures (unquantized): history must
    beat single-step by >= 0.02 and the shuffled-window null must be within
    |null advantage| <= 0.02. Evaluated per run; gated on the minimum."""
    floor = FLOORS["history_accumulation"]["positive_control_advantage"]
    band = FLOORS["history_accumulation"]["null_band"]
    per_run = {}
    for run_id, latents in dense_events_by_run.items():
        per_run[run_id] = family3_evaluate(latents, tokens, seed, epochs, device)
    min_adv = min(r["advantage"] for r in per_run.values())
    max_null = max(abs(r["null_advantage"]) for r in per_run.values())
    qualified = bool(min_adv >= floor and max_null <= band)
    return {"family": "history_accumulation", "floor": floor, "null_band": band,
            "k": HISTORY_K, "per_run": per_run, "min_advantage": min_adv,
            "max_abs_null_advantage": max_null, "qualified": qualified}


# --------------------------------------------------------------------------- #
# family 4: canary membership
# --------------------------------------------------------------------------- #
def canary_confidence(probe: InversionProbe, latents: torch.Tensor,
                      device: torch.device) -> np.ndarray:
    """Mean per-position max-probability of the inversion probe (N strings)."""
    probe.eval()
    with torch.no_grad():
        logits = probe(latents.to(device))
        probs = F.softmax(logits, dim=-1)
        per_pos = probs.max(dim=-1).values  # (N, T)
    return per_pos.mean(dim=1).cpu().numpy()


def train_inversion_probe_on_dense(dense_latents: torch.Tensor,
                                   tokens: np.ndarray, seed: int, epochs: int,
                                   device: torch.device) -> InversionProbe:
    """Inversion probe on dense captures (already quantized at the run's q)."""
    n_steps, bsz, seq, width = dense_latents.shape
    feats = dense_latents.reshape(-1, width).numpy()
    labels = np.tile(tokens.reshape(-1), n_steps)
    return train_probe(feats, labels, width, epochs, 4096, 1e-3, seed, device)


def family4_score_run(probe: InversionProbe, canaries: dict,
                      device: torch.device) -> dict:
    """AUC of member vs non-member confidence on one run's captured latents."""
    member_scores = canary_confidence(probe, canaries["member_latents"], device)
    non_member_scores = canary_confidence(probe, canaries["non_member_latents"], device)
    return {"auc": auc(member_scores.tolist(), non_member_scores.tolist()),
            "mean_confidence_members": float(member_scores.mean()),
            "mean_confidence_non_members": float(non_member_scores.mean())}
