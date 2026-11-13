"""Byte-level decoder-only transformer backbone.

Fixed pilot backbone: vocab 256, d_model 256, 8 heads (head_dim 32), 8 pre-LN
blocks, d_ff 1024, learned positional embeddings, max_seq_len 128, no dropout.
Init: N(0, 0.02); residual projections (attention output proj, MLP out proj)
scaled by 1/sqrt(2*n_layers). All init draws come from an explicit generator so
runs sharing a seed get bit-identical backbones.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BackboneConfig:
    vocab_size: int = 256
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 8
    d_ff: int = 1024
    max_seq_len: int = 128
    init_std: float = 0.02

    def to_dict(self) -> dict:
        return asdict(self)


class Block(nn.Module):
    """Pre-LN transformer block. `proj` and `mlp_proj` are residual projections."""

    def __init__(self, dim: int, n_heads: int, d_ff: int):
        super().__init__()
        assert dim % n_heads == 0, f"dim {dim} not divisible by n_heads {n_heads}"
        self.n_heads = n_heads
        self.ln1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp_fc = nn.Linear(dim, d_ff)
        self.mlp_proj = nn.Linear(d_ff, dim)
        self.proj._pcr_residual_projection = True
        self.mlp_proj._pcr_residual_projection = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq, dim = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).view(bsz, seq, 3, self.n_heads, dim // self.n_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        att = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], is_causal=True)
        att = att.transpose(1, 2).contiguous().view(bsz, seq, dim)
        x = x + self.proj(att)
        h = self.ln2(x)
        x = x + self.mlp_proj(F.gelu(self.mlp_fc(h)))
        return x


def init_module_(module: nn.Module, generator: torch.Generator, n_layers: int, std: float) -> None:
    """Standard GPT init drawn entirely from `generator` (CPU tensors only)."""
    residual_std = std / math.sqrt(2 * n_layers)
    for mod in module.modules():
        if isinstance(mod, nn.Linear):
            proj_std = residual_std if getattr(mod, "_pcr_residual_projection", False) else std
            nn.init.normal_(mod.weight, mean=0.0, std=proj_std, generator=generator)
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)
        elif isinstance(mod, nn.Embedding):
            nn.init.normal_(mod.weight, mean=0.0, std=std, generator=generator)
        elif isinstance(mod, nn.LayerNorm):
            nn.init.ones_(mod.weight)
            nn.init.zeros_(mod.bias)


class Backbone(nn.Module):
    def __init__(self, cfg: BackboneConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList(
            Block(cfg.d_model, cfg.n_heads, cfg.d_ff) for _ in range(cfg.n_layers)
        )
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, capture=None) -> torch.Tensor:
        # `capture` is accepted for interface parity with SplitBackbone and ignored.
        _, seq = idx.shape
        assert seq <= self.cfg.max_seq_len, f"seq {seq} > max_seq_len {self.cfg.max_seq_len}"
        pos = torch.arange(seq, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))
