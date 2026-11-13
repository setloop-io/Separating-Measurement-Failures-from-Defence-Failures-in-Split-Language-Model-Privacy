"""Analytic remote FLOP counter.

Convention (multiply-accumulate = 2 FLOPs), per token, forward pass of one
pre-LN transformer block at width `dim` with MLP hidden `d_ff`:

  attention projections (qkv + out):  4 linear maps x 2*dim*dim  = 8*dim^2
  attention scores + values:          2*seq_len*dim + 2*seq_len*dim = 4*seq_len*dim
  MLP (fc + proj):                    2 x 2*dim*d_ff              = 4*dim*d_ff

LayerNorm/softmax/GELU elementwise terms are excluded (second-order). The
surrogate's final LayerNorm is likewise excluded. Training cost uses the
standard approximation: backward = 2x forward when parameters update, 1x when
only input gradients are computed (frozen remote). removed_remote and
all_local perform zero remote FLOPs.
"""

from __future__ import annotations


def count_block_flops(dim: int, d_ff: int, seq_len: int) -> int:
    """Forward FLOPs per token for one pre-LN transformer block."""
    projections = 8 * dim * dim
    attention = 4 * seq_len * dim
    mlp = 4 * dim * d_ff
    return projections + attention + mlp


def remote_step_flops(width: int, seq_len: int, tokens: int, mode: str,
                      remote_depth: int = 1) -> dict:
    """FLOPs attributable to the remote module for one training step.

    Returns fwd_only and fwd_bwd (the training-step total). Backward factors:
    trained/oracle = 2x forward (input + parameter grads), frozen_random = 1x
    forward (input grads only), removed_remote/all_local = 0. remote_depth
    (additive, P5) multiplies the per-block count; depth 1 = original values.
    """
    if mode in ("removed_remote", "all_local"):
        return {"fwd_only": 0, "fwd_bwd": 0}
    fwd_per_token = count_block_flops(dim=width, d_ff=4 * width, seq_len=seq_len) * remote_depth
    fwd_only = fwd_per_token * tokens
    bwd_factor = 1 if mode == "frozen_random" else 2
    return {"fwd_only": fwd_only, "fwd_bwd": fwd_only * (1 + bwd_factor)}


def model_step_flops(n_layers: int, d_model: int, d_ff: int, seq_len: int,
                     tokens: int, mode: str, split_width: int | None = None,
                     split_span: int = 0, remote_depth: int = 1) -> dict:
    """Additive (P5): analytic TOTAL-model FLOPs for one training step, under
    the same MAC=2-FLOPs, second-order-excluded convention as count_block_flops.

    Components (forward + backward):
      local blocks    (n_layers - span when split, else n_layers): bwd = 2x fwd
      encoder/decoder linears when split (2 x 2*d_model*width per token fwd):
                      bwd = 2x fwd (trusted side always trains)
      remote blocks   per remote_step_flops (mode-dependent backward)
    Embeddings, LayerNorms, softmax/GELU, and the head are excluded
    (second-order, matching the module convention).
    """
    bwd2 = 3  # fwd + 2x bwd
    n_local = n_layers - split_span if split_width is not None else n_layers
    local = count_block_flops(dim=d_model, d_ff=d_ff, seq_len=seq_len) * n_local * tokens * bwd2
    encdec = 0
    if split_width is not None:
        encdec = 2 * (2 * d_model * split_width) * tokens * bwd2
    remote = remote_step_flops(split_width or 0, seq_len, tokens, mode, remote_depth)
    total = local + encdec + remote["fwd_bwd"]
    return {"local_blocks": local, "encoder_decoder": encdec,
            "remote_fwd_bwd": remote["fwd_bwd"], "total_fwd_bwd": total}


class RemoteInstrument:
    """Accumulates remote FLOPs across a run; additive: total-model counter."""

    def __init__(self, width: int, seq_len: int, mode: str, remote_depth: int = 1,
                 model_shape: dict | None = None):
        self.width = width
        self.seq_len = seq_len
        self.mode = mode
        self.remote_depth = remote_depth
        self.model_shape = model_shape  # {n_layers, d_model, d_ff, span} when split
        self.cumulative = {"fwd_only": 0, "fwd_bwd": 0}
        self.total_model = 0

    def record_step(self, tokens: int) -> None:
        step = remote_step_flops(self.width, self.seq_len, tokens, self.mode,
                                 self.remote_depth)
        self.cumulative["fwd_only"] += step["fwd_only"]
        self.cumulative["fwd_bwd"] += step["fwd_bwd"]
        if self.model_shape is not None:
            shape = self.model_shape
            self.total_model += model_step_flops(
                shape["n_layers"], shape["d_model"], shape["d_ff"], self.seq_len,
                tokens, self.mode,
                split_width=self.width if shape.get("span") else None,
                split_span=shape.get("span", 0), remote_depth=self.remote_depth,
            )["total_fwd_bwd"]

    def snapshot(self) -> dict:
        out = dict(self.cumulative)
        if self.model_shape is not None:
            out["total_model_fwd_bwd"] = self.total_model
            out["remote_share"] = (self.cumulative["fwd_bwd"] / self.total_model
                                   if self.total_model else 0.0)
        return out
