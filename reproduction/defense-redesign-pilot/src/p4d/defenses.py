"""Paper 4 candidate mechanisms. No module makes a privacy guarantee."""
from dataclasses import asdict, dataclass, field
from contextlib import contextmanager
import math
import torch
from torch import nn


@dataclass
class DefenseConfig:
    name: str
    strength: str = "medium"
    params: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


# Proposal grid; scientific execution requires a separately frozen protocol.
STRENGTH_GRIDS = {
    "property_noise": {k: {"noise_scale": v} for k, v in
                       zip(("low", "medium", "high"), (0.1, 0.5, 1.0))},
    "aggressive_rate": {k: {"q_bits": v} for k, v in
                        zip(("low", "medium", "high"), (3, 2, 1))},
}


class PropertyNoiseDefense(nn.Module):
    """Orthonormal subspace noise shared across sequence positions.

    Directions must be calibrated in this model's coordinates. A private CPU
    generator isolates noise from model/data RNG. Evaluation keeps defense active.
    """
    def __init__(self, width, noise_scale=0.5, property_directions=None, seed=0):
        super().__init__()
        if not math.isfinite(noise_scale) or noise_scale < 0:
            raise ValueError("noise_scale must be finite and nonnegative")
        if property_directions is None:
            raise ValueError("Property noise requires calibrated directions; no silent passthrough")
        basis = torch.as_tensor(property_directions, dtype=torch.float32).clone()
        if (basis.ndim != 2 or basis.shape[1] != width or not basis.shape[0]
                or not torch.isfinite(basis).all()):
            raise ValueError("Expected a finite, nonempty (rank, width) basis")
        if not torch.allclose(basis @ basis.T, torch.eye(len(basis)), atol=1e-5):
            raise ValueError("Property directions must be orthonormal")
        self.register_buffer("directions", basis)
        self.noise_scale = float(noise_scale)
        self.generator = torch.Generator(device="cpu").manual_seed(seed)

    def forward(self, z):
        if self.noise_scale == 0:
            return z
        eps = torch.randn(z.shape[0], 1, len(self.directions), generator=self.generator)
        delta = eps.to(device=z.device, dtype=z.dtype) @ self.directions.to(z.dtype)
        return z + self.noise_scale * delta

    def get_extra_state(self):
        return self.generator.get_state()

    def set_extra_state(self, state):
        self.generator.set_state(state.cpu())


@contextmanager
def noise_stream(module, seed=None):
    """Measurement never advances training's defense-noise stream."""
    streams = [(m.generator, m.generator.get_state()) for m in module.modules()
               if isinstance(m, (PropertyNoiseDefense, SequenceShuffleDefense))]
    try:
        if seed is not None:
            for generator, _ in streams:
                generator.manual_seed(seed)
        yield
    finally:
        for generator, state in streams:
            generator.set_state(state)


class CausalAffineQuantizer(nn.Module):
    """Per-token min/scale over width only; metadata is part of the observer view.

    Byte accounting is analytic packed-code size, not measured network traffic.
    Scorers must not re-quantize the resulting released values.
    """
    def __init__(self, q_bits):
        super().__init__()
        if isinstance(q_bits, bool) or not isinstance(q_bits, int) or not 1 <= q_bits <= 32:
            raise ValueError("q_bits must be an integer from 1 to 32")
        self.q_bits = q_bits

    def quantize(self, x):
        if not torch.isfinite(x).all():
            raise ValueError("Nonfinite boundary representation")
        if self.q_bits == 32:
            return x, {}
        lo = x.detach().amin(dim=-1, keepdim=True)
        hi = x.detach().amax(dim=-1, keepdim=True)
        scale = ((hi - lo) / (2 ** self.q_bits - 1)).clamp_min(1e-12)
        codes = ((x.detach() - lo) / scale).round().clamp(0, 2 ** self.q_bits - 1)
        values = codes * scale + lo
        return values, {"minimum": lo, "scale": scale}

    def forward(self, x):
        values, metadata = self.quantize(x)
        return x + (values - x).detach(), metadata

    def serialized_nbytes(self, x):
        if self.q_bits == 32:
            return x.numel() * 4
        return math.ceil(x.numel() * self.q_bits / 8) + x.shape[0] * x.shape[1] * 8


class SequenceShuffleDefense(nn.Module):
    """D5 diagnostic control only: preserves means and can violate causal order."""
    def __init__(self, shuffle_fraction=1.0, seed=0):
        super().__init__()
        if not 0 <= shuffle_fraction <= 1:
            raise ValueError("shuffle_fraction must be between zero and one")
        self.shuffle_fraction = shuffle_fraction
        self.generator = torch.Generator().manual_seed(seed)

    def forward(self, z):
        result = z.clone()
        n = int(z.shape[1] * self.shuffle_fraction)
        for b in range(z.shape[0]):
            positions = torch.randperm(z.shape[1], generator=self.generator)[:n]
            perm = torch.randperm(n, generator=self.generator)
            result[b, positions.to(z.device)] = z[b, positions[perm].to(z.device)]
        return result
