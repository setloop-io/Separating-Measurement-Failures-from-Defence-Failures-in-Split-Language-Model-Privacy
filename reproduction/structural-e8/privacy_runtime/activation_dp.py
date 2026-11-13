"""Formal Gaussian privacy mechanisms for split-compute boundaries.

The accountant uses replace-one adjacency.  Per-sample vectors are clipped to
L2 norm C, giving sensitivity 2C; Gaussian noise has standard deviation
``noise_multiplier * C``.  Composition is reported through zCDP and converted
to (epsilon, delta)-DP.  Forward and return directions are accounted
separately and jointly.
"""

from __future__ import annotations

import math


def rho_for_epsilon(epsilon: float, delta: float) -> float:
    """Invert epsilon = rho + 2*sqrt(rho*log(1/delta))."""
    if epsilon <= 0 or not 0 < delta < 1:
        raise ValueError("epsilon must be positive and delta must be in (0,1)")
    root = math.sqrt(math.log(1.0 / delta) + epsilon) - math.sqrt(
        math.log(1.0 / delta))
    return root * root


def noise_for_rho(releases: int, rho: float,
                  adjacency: str = "replace_one") -> float:
    if releases <= 0 or rho <= 0:
        raise ValueError("releases and rho must be positive")
    sensitivity = 2.0 if adjacency == "replace_one" else 1.0
    return sensitivity * math.sqrt(releases / (2.0 * rho))


class BoundaryDPAccountant:
    def __init__(self, delta: float, adjacency: str = "replace_one"):
        if not 0.0 < delta < 1.0:
            raise ValueError("delta must be in (0, 1)")
        if adjacency not in ("replace_one", "add_remove"):
            raise ValueError("unsupported adjacency")
        self.delta = float(delta)
        self.adjacency = adjacency
        self.releases = {"forward": 0, "return": 0}
        self.rho = {"forward": 0.0, "return": 0.0}

    @property
    def sensitivity_multiplier(self) -> float:
        return 2.0 if self.adjacency == "replace_one" else 1.0

    def record(self, direction: str, noise_multiplier: float,
               releases: int = 1) -> None:
        if direction not in self.releases:
            raise ValueError("direction must be forward or return")
        if noise_multiplier <= 0 or releases <= 0:
            raise ValueError("noise_multiplier and releases must be positive")
        sensitivity = self.sensitivity_multiplier
        self.releases[direction] += int(releases)
        self.rho[direction] += releases * sensitivity ** 2 / (
            2.0 * noise_multiplier ** 2)

    def epsilon(self, direction: str | None = None) -> float:
        rho = sum(self.rho.values()) if direction is None else self.rho[direction]
        if rho == 0:
            return 0.0
        return rho + 2.0 * math.sqrt(rho * math.log(1.0 / self.delta))

    def report(self) -> dict:
        return {
            "method": "zcdp_gaussian_replace_one" if self.adjacency == "replace_one"
                      else "zcdp_gaussian_add_remove",
            "adjacency": self.adjacency,
            "delta": self.delta,
            "releases": dict(self.releases),
            "rho": dict(self.rho),
            "epsilon": {
                "forward": self.epsilon("forward"),
                "return": self.epsilon("return"),
                "composed": self.epsilon(),
            },
        }


def clip_and_noise(tensor, max_norm: float, noise_multiplier: float,
                   accountant: BoundaryDPAccountant, direction: str,
                   generator=None):
    """Clip each batch item and add calibrated Gaussian noise.

    Each final-dimension row is one token-level privacy vector. This matches
    the labelled token-pair attacker and prevents a sequence from being
    counted as one release while exposing many independently labelled rows.
    The function requires torch at call time but keeps this module importable
    in lightweight gates.
    """
    import torch

    if tensor.ndim < 2:
        raise ValueError("boundary tensor must include batch and feature dimensions")
    if max_norm <= 0 or noise_multiplier <= 0:
        raise ValueError("max_norm and noise_multiplier must be positive")
    flat = tensor.float().reshape(-1, tensor.shape[-1])
    norms = flat.norm(2, dim=1, keepdim=True)
    scales = (max_norm / norms.clamp_min(1e-12)).clamp(max=1.0)
    clipped = (flat * scales).reshape_as(tensor.float())
    noise = torch.randn(clipped.shape, dtype=clipped.dtype,
                        device=clipped.device, generator=generator)
    protected = clipped + noise * (noise_multiplier * max_norm)
    releases = int(flat.shape[0])
    accountant.record(direction, noise_multiplier, releases=releases)
    return protected.to(dtype=tensor.dtype), {
        "max_preclip_norm": float(norms.max().item()),
        "mean_clip_scale": float(scales.mean().item()),
        "noise_std": float(noise_multiplier * max_norm),
        "token_releases": releases,
        "adjacency_unit": "one_boundary_token_row",
    }


class BidirectionalBoundaryDP:
    def __init__(self, forward_clip: float, forward_noise: float,
                 return_clip: float, return_noise: float, delta: float,
                 adjacency: str = "replace_one"):
        self.forward_clip = float(forward_clip)
        self.forward_noise = float(forward_noise)
        self.return_clip = float(return_clip)
        self.return_noise = float(return_noise)
        self.accountant = BoundaryDPAccountant(delta, adjacency)

    def protect_forward(self, tensor, generator=None):
        return clip_and_noise(tensor, self.forward_clip, self.forward_noise,
                              self.accountant, "forward", generator)

    def protect_return(self, tensor, generator=None):
        return clip_and_noise(tensor, self.return_clip, self.return_noise,
                              self.accountant, "return", generator)

    def report(self) -> dict:
        result = self.accountant.report()
        result["parameters"] = {
            "forward_clip": self.forward_clip,
            "forward_noise_multiplier": self.forward_noise,
            "return_clip": self.return_clip,
            "return_noise_multiplier": self.return_noise,
        }
        return result
