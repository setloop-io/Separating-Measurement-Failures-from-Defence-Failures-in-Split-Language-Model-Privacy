"""Control-only matched-null estimators and support-valid quantized signals."""
import math
import numpy as np
import torch
from pcr.privacy_attacks_p4b import sequence_targets
from pcr.utils import derive_seed
from .experiment import measurement
from .warmup_scenarios import TARGET, derangement, panel_batches, tensor_hash

ADAPTER = "p4-ensemble-property-control-0.1"


def _permutation(order, n):
    order = np.asarray(order)
    if order.shape != (n,) or not np.issubdtype(order.dtype, np.integer) or not np.array_equal(np.sort(order), np.arange(n)):
        raise ValueError("Expected a complete row permutation")
    return order


def correlations(predictions, target):
    p, y = np.asarray(predictions, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if p.ndim != 2 or y.shape != (len(p),) or not np.isfinite(p).all() or not np.isfinite(y).all() or np.std(y) == 0:
        raise ValueError("Invalid prediction matrix or constant target")
    p, y = p - p.mean(0), y - y.mean()
    norm = np.linalg.norm(p, axis=0) * np.linalg.norm(y)
    # Same constant-prediction convention as the reference pearson_r helper.
    return np.divide(y @ p, norm, out=np.zeros(p.shape[1]), where=norm != 0).clip(-1, 1)


class RidgeDesign:
    """Cache the ridge projection; whole-row permutations do not change its Gram matrix."""
    def __init__(self, fit, test, penalty=1.0):
        x, xt = np.asarray(fit, dtype=np.float64), np.asarray(test, dtype=np.float64)
        if (x.ndim != 2 or xt.ndim != 2 or x.shape[1] != xt.shape[1]
                or min(len(x), len(xt)) < 4 or not np.isfinite(x).all() or not np.isfinite(xt).all()
                or not math.isfinite(penalty) or penalty <= 0):
            raise ValueError("Need finite aligned feature matrices and positive ridge penalty")
        mean = x.mean(0)
        xc = x - mean
        self.projection = np.linalg.solve(xc.T @ xc + penalty * np.eye(x.shape[1]), xc.T)
        self.test = xt - mean
        self.n = len(x)

    def score(self, y, yt, seed, counts=(1, 16, 64), fit_donors=None, test_donors=None):
        y, yt = np.asarray(y, dtype=np.float64), np.asarray(yt, dtype=np.float64)
        if (y.shape != (self.n,) or yt.shape != (len(self.test),) or np.std(y) == 0 or np.std(yt) == 0
                or not np.isfinite(y).all() or not np.isfinite(yt).all()):
            raise ValueError("Need aligned, finite, varying targets")
        if not counts or list(counts) != sorted(set(counts)) or any(type(k) is not int or k < 1 for k in counts):
            raise ValueError("Matched-null counts must be increasing positive integers")
        df = _permutation(np.arange(self.n) if fit_donors is None else fit_donors, self.n)
        dt = _permutation(np.arange(len(yt)) if test_donors is None else test_donors, len(yt))
        rng = np.random.default_rng(seed)
        labels, orders = [], []
        for _ in range(max(counts)):
            labels.append(rng.permutation(y))
            orders.append(rng.permutation(len(yt)))
        target_matrix = np.column_stack([y, *labels])
        means = target_matrix.mean(0)
        weights = self.projection[:, df] @ (target_matrix - means)
        predictions = self.test[dt] @ weights + means
        observed = float(correlations(predictions[:, :1], yt)[0])
        shuffled_predictions = predictions[np.array(orders).T, np.arange(1, max(counts) + 1)]
        nulls = correlations(shuffled_predictions, yt).tolist()
        return {"scorer_seed": seed, "r": observed, "r_null": nulls,
                "estimators": [{"matched_null_count": k, "mean_r_null": float(np.mean(nulls[:k])),
                                "advantage": observed - float(np.mean(nulls[:k]))} for k in counts]}


def no_signal_trials(design, y, yt, repeats, counts, seed):
    if type(repeats) is not int or repeats < 1:
        raise ValueError("Need positive outer-null repetition count")
    rows = []
    for i in range(repeats):
        seeds = {k: derive_seed(ADAPTER, seed, len(y), i, k) for k in ("fit", "test", "scorer")}
        rows.append({"repetition": i, "seeds": seeds,
                     **design.score(y, yt, seeds["scorer"], counts,
                                    derangement(len(y), seeds["fit"]), derangement(len(yt), seeds["test"]))})
    return rows


def prequant_capture(model, panel, batch_size, seed, device=None):
    """Evaluator-only full tensors at the post-defense/pre-final-quantization point."""
    captures = []
    with measurement(model, seed), torch.no_grad():
        for x, _ in panel_batches(panel, batch_size):
            if device is not None:
                x = x.to(device)
            h = model.tok_emb(x) + model.pos_emb(torch.arange(x.shape[1], device=x.device))
            for block in model.blocks[:model.split.start]:
                h = block(h)
            path = model.remote_path
            z = path.defense(path.encoder(path.encoder_ln(h)))
            captures.append(z.detach().clone())
    return torch.cat(captures)


def signal_spec(calibration_means, calibration_tokens, seed):
    x = torch.as_tensor(calibration_means, dtype=torch.float32)
    y = sequence_targets(calibration_tokens)[TARGET]
    if (x.ndim != 2 or len(x) != len(y) or not torch.isfinite(x).all()
            or len(y) < 4 or np.std(y) == 0):
        raise ValueError("Need finite, varying calibration features/targets")
    generator = torch.Generator().manual_seed(seed)
    direction = torch.randn(x.shape[1], generator=generator)
    direction /= torch.linalg.vector_norm(direction)
    scale = float(x.std(unbiased=False)) * math.sqrt(x.shape[1])
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Degenerate calibration feature scale")
    return {"seed": seed, "direction": direction, "feature_scale": scale,
            "target_mean": float(y.mean()), "target_std": float(y.std()),
            "calibration_features_sha256": tensor_hash(x),
            "calibration_tokens_sha256": tensor_hash(calibration_tokens)}


def quantized_control(prequant, target, donors, spec, dose, quantizer, batch_size=32):
    if not math.isfinite(dose) or dose < 0 or quantizer.q_bits == 32:
        raise ValueError("Need a nonnegative dose and quantized release")
    if (prequant.ndim != 3 or not torch.isfinite(prequant).all() or batch_size < 1
            or spec["direction"].shape != (prequant.shape[-1],)
            or not torch.isfinite(spec["direction"]).all()
            or not math.isfinite(spec["feature_scale"]) or spec["feature_scale"] <= 0
            or not math.isfinite(spec["target_mean"]) or not math.isfinite(spec["target_std"]) or spec["target_std"] <= 0):
        raise ValueError("Invalid quantized-control geometry")
    donors = _permutation(donors, len(prequant))
    target = torch.as_tensor(target, dtype=prequant.dtype)
    if target.shape != (len(prequant),) or not torch.isfinite(target).all():
        raise ValueError("Control labels must be finite and align with donor rows")
    ztarget = (target - spec["target_mean"]) / spec["target_std"]
    means, minima, scales = [], [], []
    maximum_error, zero_equal = 0.0, True
    with torch.no_grad():
        for i in range(0, len(prequant), batch_size):
            donor = prequant[donors[i:i + batch_size]]
            injected = donor + dose * spec["feature_scale"] * ztarget[i:i + batch_size, None, None] * spec["direction"]
            values, meta = quantizer(injected)
            lo, scale = meta["minimum"], meta["scale"]
            codes = (values - lo) / scale
            error = float((codes - codes.round()).abs().max())
            maximum_error = max(maximum_error, error)
            if (not torch.isfinite(values).all() or not torch.isfinite(lo).all()
                    or not torch.isfinite(scale).all() or not (scale > 0).all()
                    or float(codes.min()) < -1e-3 or float(codes.max()) > 2 ** quantizer.q_bits - 1 + 1e-3
                    or error > 1e-3):
                raise RuntimeError("Invalid quantized control support")
            if dose == 0:
                zero_equal = zero_equal and torch.equal(values, quantizer(donor)[0])
            means.append(values.mean(1))
            minima.append(lo)
            scales.append(scale)
    if not zero_equal:
        raise RuntimeError("Zero-dose control differs from ordinary donor release")
    return torch.cat(means), {"q_bits": quantizer.q_bits, "dose": dose,
        "maximum_normalized_rounding_error": maximum_error, "support_valid": True,
        "zero_dose_exact": zero_equal if dose == 0 else None,
        "minimum_sha256": tensor_hash(torch.cat(minima)), "scale_sha256": tensor_hash(torch.cat(scales))}
