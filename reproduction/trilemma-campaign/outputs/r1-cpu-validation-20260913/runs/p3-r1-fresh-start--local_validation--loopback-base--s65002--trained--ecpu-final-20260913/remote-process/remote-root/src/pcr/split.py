"""Split wrapper: replace a contiguous block span with a remote path.

Path for span [start, start+span):
    residual = x
    z_pre    = Linear(d_model->width)(LayerNorm(x))      # trusted encoder
    z        = affine_quantize(z_pre, q_bits)            # boundary defense, STE in training
    y        = surrogate(z)                              # remote module (untrusted side)
    out      = residual + Linear(width->d_model)(y)      # trusted decoder, residual add

The surrogate is ONE pre-LN transformer block at dim=width (head_dim 32, so
n_heads = width/32, d_ff = 4*width) plus a final LayerNorm. It is the only
component whose updating differs between trained and frozen modes.

Modes:
  trained        — surrogate params require grad and are optimized.
  frozen_random  — surrogate params are frozen at their paired init.
  removed_remote — surrogate replaced by a fixed identity; encoder/decoder
                   remain and train; zero remote trainable params.
  oracle         — same topology but width == d_model and no quantization.
all_local is handled by using a plain Backbone (no split at all).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

from .model import Backbone, BackboneConfig, Block

METADATA_BYTES = 8  # fp32 scale + int32 zero_point per serialized tensor


@dataclass
class SplitConfig:
    start: int
    span: int
    width: int
    q_bits: int
    remote_depth: int = 1  # additive (P5): stacked surrogate blocks; 1 = original behavior
    grad_defense: dict | None = None  # additive (P5): {clip_norm, noise_std} or None

    def to_dict(self) -> dict:
        return asdict(self)


class AffineQuantizer:
    """Uniform per-tensor affine quantization to `q_bits`.

    Training uses a straight-through estimator; serialization/byte accounting
    uses the true quantized integer codes. q_bits >= 32 disables quantization
    (identity; fp32 passthrough) — used by the oracle mode and the 32-bit dose
    of the privacy positive control.
    """

    def __init__(self, q_bits: int):
        assert 1 <= q_bits, f"q_bits must be >= 1, got {q_bits}"
        self.q_bits = q_bits

    @property
    def disabled(self) -> bool:
        return self.q_bits >= 32

    def _affine_params(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        qmax = float(2**self.q_bits - 1)
        xmin = x.detach().amin()
        xmax = x.detach().amax()
        scale = (xmax - xmin) / qmax
        constant_scale = torch.where(xmin.abs() > 0, xmin.abs() / qmax,
                                     torch.ones_like(xmin))
        scale = torch.where(scale > 0, scale, constant_scale)
        constant_zero = torch.where(xmin < 0, torch.full_like(xmin, qmax),
                                    torch.zeros_like(xmin))
        zero_point = torch.where(xmax == xmin, constant_zero,
                                 torch.round(-xmin / scale).clamp(0.0, qmax))
        return scale, zero_point

    def quantize_dequantize(self, x: torch.Tensor) -> torch.Tensor:
        """True quantization (detached); used for serialization and probes."""
        if self.disabled:
            return x.detach().clone()
        scale, zero_point = self._affine_params(x)
        qmax = float(2**self.q_bits - 1)
        q = torch.round(x.detach() / scale + zero_point).clamp(0.0, qmax)
        return (q - zero_point) * scale

    def ste(self, x: torch.Tensor) -> torch.Tensor:
        """Straight-through estimator: forward value is quantized, grad passes through."""
        if self.disabled:
            return x
        return x + (self.quantize_dequantize(x) - x.detach())

    def integer_codes(self, x: torch.Tensor) -> torch.Tensor:
        """True quantized integer codes (what would actually cross the boundary)."""
        if self.disabled:
            raise ValueError("integer codes undefined for disabled quantizer")
        scale, zero_point = self._affine_params(x)
        qmax = float(2**self.q_bits - 1)
        q = torch.round(x.detach() / scale + zero_point).clamp(0.0, qmax)
        if self.q_bits <= 8:
            return q.to(torch.uint8)
        if self.q_bits <= 16:
            return q.to(torch.uint16)
        return q.to(torch.int32)

    def serialized_nbytes(self, x: torch.Tensor) -> int:
        """Measured serialized size: packed codes plus affine metadata.

        Payload is ceil(numel * q_bits / 8) bytes (packed integer codes); for
        q_bits >= 32 the payload is raw fp32. Metadata is 8 bytes per tensor
        (fp32 scale + int32 zero_point), included for honest accounting.
        """
        if self.disabled:
            return x.numel() * 4
        return math.ceil(x.numel() * self.q_bits / 8) + METADATA_BYTES

    def payload_nbytes(self, x: torch.Tensor) -> int:
        if self.disabled:
            return x.numel() * 4
        return math.ceil(x.numel() * self.q_bits / 8)


class Surrogate(nn.Module):
    """Remote module: `depth` pre-LN transformer blocks plus a final LayerNorm.

    depth=1 reproduces the original single-block surrogate exactly (same
    parameter order for init; same math).
    """

    def __init__(self, width: int, depth: int = 1):
        super().__init__()
        assert width % 32 == 0, f"width {width} must give head_dim 32"
        assert depth >= 1, f"depth must be >= 1, got {depth}"
        self.blocks = nn.ModuleList(
            Block(dim=width, n_heads=width // 32, d_ff=4 * width) for _ in range(depth)
        )
        self.ln_f = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x)


class IdentityRemote(nn.Module):
    """Fixed identity bypass for removed_remote mode; zero trainable params."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class RemovedRemote(nn.Module):
    """Additive (P5): stacked identity bypass matching the surrogate's depth
    (`identity per block`); zero trainable params."""

    def __init__(self, depth: int = 1):
        super().__init__()
        self.blocks = nn.ModuleList(IdentityRemote() for _ in range(depth))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class _GradDefense(torch.autograd.Function):
    """Additive (P5): identity forward; backward clips per-position gradient
    vectors to `clip_norm` and adds Gaussian noise with std `noise_std`.

    The noise stream is a dedicated trusted-side generator
    (derive_seed("grad-defense", seed), attached by build_initialized_model);
    drawn on CPU and moved to the gradient's device (MPS generators cannot be
    shared across devices), so same seed+config gives identical losses across
    executions on the same device.
    """

    @staticmethod
    def forward(ctx, x, clip_norm, noise_std, generator):
        ctx.clip_norm = clip_norm
        ctx.noise_std = noise_std
        ctx.generator = generator
        return x

    @staticmethod
    def backward(ctx, grad_output):
        g = grad_output
        if ctx.clip_norm is not None:
            norm = g.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            scale = (ctx.clip_norm / norm).clamp(max=1.0)
            g = g * scale
        if ctx.noise_std:
            noise = torch.randn(g.shape, generator=ctx.generator, dtype=torch.float32)
            g = g + noise.to(g.device, g.dtype) * ctx.noise_std
        return g, None, None, None


class RemotePath(nn.Module):
    """Encoder -> boundary quantizer -> remote module -> decoder (residual add)."""

    def __init__(self, d_model: int, cfg: SplitConfig, removed_remote: bool):
        super().__init__()
        self.cfg = cfg
        self.encoder_ln = nn.LayerNorm(d_model)
        self.encoder = nn.Linear(d_model, cfg.width)
        self.quantizer = AffineQuantizer(cfg.q_bits)
        self.remote = (
            RemovedRemote(cfg.remote_depth) if removed_remote
            else Surrogate(cfg.width, cfg.remote_depth)
        )
        self.decoder = nn.Linear(cfg.width, d_model)
        self._grad_defense_gen = None  # attached by train.build_initialized_model

    def _defend_grad(self, t: torch.Tensor) -> torch.Tensor:
        """Additive (P5): identity unless cfg.grad_defense is set; applies
        clip+noise to the gradient crossing the boundary in the backward
        graph only (forward values and byte accounting unchanged)."""
        if self.cfg.grad_defense is None:
            return t
        return _GradDefense.apply(t, self.cfg.grad_defense["clip_norm"],
                                  self.cfg.grad_defense["noise_std"],
                                  self._grad_defense_gen)

    def forward(self, x: torch.Tensor, capture=None) -> torch.Tensor:
        z_pre = self.encoder(self.encoder_ln(x))
        if capture is not None and getattr(capture, "wants_raw_latents", False):
            capture.on_raw_latents(z_pre)
        z = self.quantizer.ste(z_pre)
        if capture is not None:
            capture.on_forward_values(z)  # its grad hook sees the defended gradient
        z = self._defend_grad(z)
        y = self.remote(z)
        if capture is not None:
            capture.on_returned_values(y)  # its grad hook sees the defended gradient
        y = self._defend_grad(y)
        return x + self.decoder(y)


class SplitBackbone(nn.Module):
    """Backbone whose blocks[start:start+span] are replaced by one RemotePath."""

    def __init__(self, cfg: BackboneConfig, split: SplitConfig, removed_remote: bool = False):
        super().__init__()
        self.cfg = cfg
        self.split = split
        assert 0 <= split.start and split.start + split.span <= cfg.n_layers
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        local_blocks = [Block(cfg.d_model, cfg.n_heads, cfg.d_ff) for _ in range(cfg.n_layers)]
        ordered = (
            local_blocks[: split.start]
            + [RemotePath(cfg.d_model, split, removed_remote)]
            + local_blocks[split.start + split.span :]
        )
        self.blocks = nn.ModuleList(ordered)
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    @property
    def remote_path(self) -> RemotePath:
        return self.blocks[self.split.start]

    @property
    def _remote_prefix(self) -> str:
        return f"blocks.{self.split.start}.remote."

    def backbone_state_dict(self) -> dict:
        """Trusted-side state (backbone + encoder/decoder), excluding the remote module."""
        return {k: v for k, v in self.state_dict().items() if not k.startswith(self._remote_prefix)}

    def trusted_split_parameters(self) -> list[nn.Parameter]:
        return (
            list(self.remote_path.encoder_ln.parameters())
            + list(self.remote_path.encoder.parameters())
            + list(self.remote_path.decoder.parameters())
        )

    def remote_parameters(self) -> list[nn.Parameter]:
        return list(self.remote_path.remote.parameters())

    def forward(self, idx: torch.Tensor, capture=None) -> torch.Tensor:
        _, seq = idx.shape
        assert seq <= self.cfg.max_seq_len, f"seq {seq} > max_seq_len {self.cfg.max_seq_len}"
        pos = torch.arange(seq, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            if isinstance(block, RemotePath):
                x = block(x, capture=capture)
            else:
                x = block(x)
        return self.head(self.ln_f(x))


def build_model(
    backbone_cfg: BackboneConfig,
    mode: str,
    split: SplitConfig | None,
) -> nn.Module:
    """Construct (uninitialized) model for a mode. Init happens in train.build_initialized_model."""
    if mode == "all_local":
        return Backbone(backbone_cfg)
    assert split is not None, f"mode {mode} requires a split config"
    return SplitBackbone(backbone_cfg, split, removed_remote=(mode == "removed_remote"))
