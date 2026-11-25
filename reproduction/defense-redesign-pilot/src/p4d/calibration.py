"""Scoped Paper 4 probe adapters; no inherited qualification verdicts."""
import numpy as np
import torch
from torch import nn
from pcr.privacy_attacks_p4b import ridge_fit, ridge_predict, pearson_r, sequence_targets, next_byte_class

PROPERTY_FLOOR = 0.10
STRUCTURE_FLOOR = 0.02
NULL_BAND = 0.02
PRIVACY_ACCEPTANCE = 0.02
PRIMARY_PROPERTY = "punctuation_fraction"
INSTRUMENT_VERSION = "p4-probe-adapters-0.1"


def estimate_property_directions(latents, targets, method="ridge", alpha=1.0):
    """Centered ridge followed by SVD rank reduction/orthonormalization."""
    x = np.asarray(latents, dtype=np.float64)
    if (method != "ridge" or x.ndim != 2 or len(x) < 3 or not targets
            or not np.isfinite(x).all() or alpha <= 0):
        raise ValueError("Need finite features, varying targets, and positive ridge penalty")
    directions = []
    for y in targets.values():
        y = np.asarray(y, dtype=np.float64)
        if y.shape != (len(x),) or not np.isfinite(y).all() or np.std(y) == 0:
            raise ValueError("Calibration target must vary and align with feature rows")
        directions.append(ridge_fit(x, y, lam=alpha)["w"])
    _, singular, vh = np.linalg.svd(np.stack(directions), full_matrices=False)
    rank = int(np.sum(singular > max(1e-10, singular[0] * 1e-6)))
    if rank == 0:
        raise ValueError("Calibration produced no usable property direction")
    return torch.tensor(vh[:rank].copy(), dtype=torch.float32)


def _features(events, structure=False):
    if not events:
        raise ValueError("Empty capture panel")
    features, targets = [], []
    for e in events:
        z = e["latents"].detach().cpu().float()
        if not torch.isfinite(z).all():
            raise ValueError("Nonfinite captured features")
        if structure:
            # Actual next-byte labels, including final position; never np.roll.
            y = e["next_tokens"].detach().cpu()
            if tuple(y.shape) != tuple(z.shape[:2]):
                raise ValueError("Next-token labels do not align with capture")
            features.append(z.reshape(-1, z.shape[-1]).numpy())
            targets.append(next_byte_class(y.numpy()).reshape(-1))
        else:
            features.append(z.mean(dim=1).numpy())
            targets.append(sequence_targets(e["tokens"].detach().cpu()))
    if structure:
        return np.concatenate(features), np.concatenate(targets)
    return np.concatenate(features), {k: np.concatenate([t[k] for t in targets])
                                     for k in targets[0]}


def score_properties(train_events, test_events, seed=0):
    """Separate text panels, same model coordinates, released values used as-is."""
    x, ys = _features(train_events)
    xt, yst = _features(test_events)
    rng = np.random.default_rng(seed)
    out = {}
    for name, y in ys.items():
        yt = yst[name]
        if min(len(y), len(yt)) < 4 or np.std(y) == 0 or np.std(yt) == 0:
            out[name] = {"status": "undetermined", "advantage": None,
                         "reason": "insufficient rows or constant target"}
            continue
        fitted = ridge_fit(x, y)
        null = ridge_fit(x, rng.permutation(y))
        r = pearson_r(ridge_predict(fitted, xt), yt)
        rn = pearson_r(ridge_predict(null, xt[rng.permutation(len(xt))]), yt)
        out[name] = {"status": "scored", "r": r, "r_null": rn, "advantage": r - rn}
    return out


def score_structure(train_events, test_events, seed=0, epochs=100):
    """Five-class MLP adaptation; this version requires fresh qualification."""
    x, y = _features(train_events, True)
    xt, yt = _features(test_events, True)
    if len(np.unique(y)) < 2 or len(np.unique(yt)) < 2:
        return {"status": "undetermined", "advantage": None, "reason": "constant target"}
    mean, std = x.mean(0), np.maximum(x.std(0), 1e-6)
    x, xt = torch.tensor((x - mean) / std), torch.tensor((xt - mean) / std)
    labels = torch.tensor(y, dtype=torch.long)
    rng = np.random.default_rng(seed)
    accuracies = []
    for targets in (labels, labels[rng.permutation(len(labels))]):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            probe = nn.Sequential(nn.Linear(x.shape[1], 64), nn.ReLU(), nn.Linear(64, 5))
            optimizer = torch.optim.Adam(probe.parameters(), lr=3e-3)
            for _ in range(epochs):
                optimizer.zero_grad(set_to_none=True)
                nn.functional.cross_entropy(probe(x), targets).backward()
                optimizer.step()
            with torch.no_grad():
                pred = probe(xt).argmax(-1).numpy()
        accuracies.append(float(np.mean(pred == yt)))
    return {"status": "scored", "accuracy": accuracies[0],
            "null_accuracy": accuracies[1], "advantage": accuracies[0] - accuracies[1]}


def qualification_from_controls(positive, null, family, binding):
    """Target/channel/defense binding required; floors are not acceptance gates."""
    if family not in ("sequence_property", "next_byte_structure") or not binding:
        raise ValueError("Unknown family or missing qualification binding")
    floor = PROPERTY_FLOOR if family == "sequence_property" else STRUCTURE_FLOOR
    valid = all(v is not None and np.isfinite(v) for v in (positive, null))
    qualified = bool(valid and positive >= floor and abs(null) <= NULL_BAND)
    return {"instrument_version": INSTRUMENT_VERSION, "binding": binding,
            "family": family, "positive_advantage": positive,
            "null_pipeline_advantage": null, "positive_floor": floor,
            "null_band": NULL_BAND, "acceptance_threshold": PRIVACY_ACCEPTANCE,
            "qualified": qualified}


def requalify_instrument(defense_name, events_null, events_positive,
                        instrument_family, binding, seed=0, epochs=100):
    """Controls are {train: [...], test: [...]} separate text panels.

    Positive signal is injected into observer input in a documented calibration
    experiment. Pre-defense injection alone is not detector qualification.
    """
    if binding.get("defense") != defense_name:
        raise ValueError("Defense does not match qualification binding")
    if instrument_family == "sequence_property":
        target = binding.get("target")
        def score(panel):
            return score_properties(panel["train"], panel["test"], seed)[target]["advantage"]
    elif instrument_family == "next_byte_structure":
        def score(panel):
            return score_structure(panel["train"], panel["test"], seed, epochs)["advantage"]
    else:
        raise ValueError("Unknown instrument family")
    return qualification_from_controls(score(events_positive), score(events_null),
                                       instrument_family, binding)


def qualified_gate(scores, required):
    """Missing/unqualified scores never become zero leakage or privacy passes."""
    any_undetermined = False
    for key in required:
        result = scores.get(key, {})
        advantage = result.get("advantage")
        if not result.get("qualified") or advantage is None or not np.isfinite(advantage):
            any_undetermined = True
        elif advantage > PRIVACY_ACCEPTANCE:
            return "reject"
    return "undetermined" if any_undetermined or not required else "within_qualified_probe_gates"
