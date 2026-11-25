"""Observer-input control construction; input captures are never modified."""
from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np
import torch

from pcr.privacy_attacks_p4b import sequence_targets, next_byte_class

VIEWS = ("forward_values", "returned_values", "inbound_gradients", "outbound_gradients",
         "quantizer_metadata", "forward_and_metadata")


@dataclass(frozen=True)
class Panel:
    values: torch.Tensor
    tokens: torch.Tensor
    next_tokens: torch.Tensor

    def __post_init__(self):
        z, x, y = self.values, self.tokens, self.next_tokens
        if (z.ndim != 3 or z.shape[0] < 2 or z.shape[-1] < 1
                or x.shape != z.shape[:2] or y.shape != x.shape
                or not torch.isfinite(z).all()):
            raise ValueError("Need finite sequence features and aligned input/next-token labels")
        for t in (x, y):
            if t.dtype not in (torch.int64, torch.int32, torch.uint8) or t.min() < 0 or t.max() > 255:
                raise ValueError("Labels must be integer bytes")

    def events(self):
        return [{"latents": self.values, "tokens": self.tokens, "next_tokens": self.next_tokens}]

    def sha256(self):
        h = hashlib.sha256()
        for tensor in (self.values, self.tokens, self.next_tokens):
            value = tensor.detach().cpu().contiguous().numpy()
            h.update(json.dumps([list(value.shape), str(value.dtype)]).encode())
            h.update(value.tobytes())
        return h.hexdigest()


def observation_panel(events, view):
    if view not in VIEWS or not events:
        raise ValueError("Unknown view or empty captures")
    zs, xs, ys = [], [], []
    for event in events:
        if view in ("quantizer_metadata", "forward_and_metadata"):
            metadata = event["quantizer_metadata"]
            z = torch.cat([metadata["minimum"], metadata["scale"]], -1)
            if view == "forward_and_metadata":
                z = torch.cat([event["forward_values"], z], -1)
        else:
            z = event[view]
        zs.append(z.detach().cpu().float().clone())
        xs.append(event["tokens"].detach().cpu().clone())
        ys.append(event["next_tokens"].detach().cpu().clone())
    return Panel(torch.cat(zs), torch.cat(xs), torch.cat(ys))


def donor_null(panel, seed):
    """Derange whole sequences; finite-sample residual associations may remain."""
    order = np.random.default_rng(seed).permutation(len(panel.values))
    donors = np.empty_like(order)
    donors[order] = np.roll(order, 1)
    result = Panel(panel.values[donors].clone(), panel.tokens.clone(), panel.next_tokens.clone())
    return result, donors.tolist()


def signal_spec(fit, family, target, seed):
    """All scaling/geometry is determined on the fit panel or an independent seed."""
    width = fit.values.shape[-1]
    generator = torch.Generator().manual_seed(seed)
    basis = torch.randn(5 if family == "next_byte_structure" else 1, width, generator=generator)
    basis = basis / basis.norm(dim=1, keepdim=True).clamp_min(1e-12)
    scale = float(fit.values.std(unbiased=False))
    if scale <= 1e-12:
        raise ValueError("Constant observation has no usable scale for this control")
    result = {"family": family, "target": target, "basis": basis,
              "feature_scale": scale * math.sqrt(width), "geometry_seed": seed}
    if family == "sequence_property":
        targets = sequence_targets(fit.tokens)
        if target not in targets:
            raise ValueError("Unknown property target")
        y = targets[target]
        if np.std(y) <= 1e-12:
            raise ValueError("Constant calibration target")
        result.update(target_mean=float(np.mean(y)), target_std=float(np.std(y)))
    elif family != "next_byte_structure" or target != "next_byte_class":
        raise ValueError("Unknown family/target binding")
    return result


def inject_signal(null_panel, spec, dose):
    """Diagnostic post-release injection; never quantized or called normal traffic."""
    if not math.isfinite(dose) or dose < 0:
        raise ValueError("Dose must be finite and nonnegative")
    if spec["family"] == "sequence_property":
        y = sequence_targets(null_panel.tokens)[spec["target"]]
        target = torch.tensor((y - spec["target_mean"]) / spec["target_std"], dtype=torch.float32)
        signal = target[:, None, None] * spec["basis"][0][None, None, :]
    else:
        labels = next_byte_class(null_panel.next_tokens.numpy())
        signal = spec["basis"][torch.as_tensor(labels, dtype=torch.long)]
    return Panel(null_panel.values + dose * spec["feature_scale"] * signal,
                 null_panel.tokens.clone(), null_panel.next_tokens.clone())
