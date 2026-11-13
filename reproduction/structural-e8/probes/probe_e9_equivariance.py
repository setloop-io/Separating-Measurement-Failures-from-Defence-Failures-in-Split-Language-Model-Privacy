#!/usr/bin/env python3
"""E9 boundary-rotation probe: W^T . middle(h @ W) is NOT middle(h).

WHAT THIS PROVES
----------------
E8 multiplies the boundary activation by a secret orthogonal W before it leaves
the trusted node.  E9 wires that into the real split server:

    split-inference/jacobi_server.py:883-896
        883  # [E8] Secret obfuscation at the wire boundary (defense experiment):
        885  if self.obf_W is not None:
        886      hidden = einsum('bsh,hd->bsd', hidden.float(), self.obf_W)   # h -> h @ W
        890  hidden = self._send_to_cloud(...)                                # cloud middle()
        895  if self.obf_W is not None:
        896      hidden = einsum('bsh,hd->bsd', hidden.float(), self.obf_Wt)  # -> ... @ W^T

    split-inference/cloud_server_kv.py
        contains ZERO obfuscation logic (`grep -c -i obf` -> 0), so the cloud
        evaluates its UNMODIFIED PUBLIC middle layers on the rotated tensor.

So the deployed composition is  W^T . middle(h @ W)  against an unmodified
middle().  Transformer blocks are not equivariant under an orthogonal change of
basis of the residual stream, so that composition does not recover middle(h).
This probe demonstrates it with numbers, on a minimal pre-norm residual block

    middle(x) = x + rmsnorm(x, g) @ P

and shows exactly which algebraic condition is violated:

    W^T . middle(h @ W) = h + ((n @ W) * g) @ P @ W^T          (n = h / rms(h))
    middle(h)           = h +  (n * g) @ P

  1. learned gain g       -> fails (order-1 relative error)
  2. constant gain g = 1  -> STILL fails.  The RMSNorm gain is a red herring on
     its own: with g = 1 the requirement collapses to W P W^T = P, i.e. W must
     COMMUTE with the projection.  A random orthogonal W does not.
  3. signed-permutation W -> still fails.  The QuaRot/SpinQuant-style structured
     choice commutes with diag(g) up to a permutation, which fixes the gain term
     and nothing else; W P W^T != P remains.
  4. folding is required AND sufficient: conjugate the projection as
     P' = W^T diag(g) P W and retire the gain (g -> 1); error drops to ~1e-15.

WHY IT MATTERS
--------------
(a) Correctness: any E9 accuracy / acceptance / utility number produced by the
    deployed path is measuring a different function than the model, not a
    defended version of it.
(b) Security: the fix in (4) is what makes the composition exact, and it is
    self-defeating.  Folding puts W-transformed copies of the middle weights on
    the untrusted cloud while their public originals stay downloadable, so the
    attacker solves  public @ W = folded  by least squares and reads W off in
    closed form.  Section 5 below demonstrates that recovery, together with a
    rank-deficient control where it correctly FAILS.

NOTES
-----
float64 throughout, so the ~1e-15 residuals are float64 rounding and nothing
else; the deployment runs bf16/fp16, which is strictly worse.  Deterministic
(seeded).  numpy only: no torch, no GPU, no network, runs in ~1s at H=512.

Usage:
    python3 probes/probe_e9_equivariance.py [--seed N] [--dim H] [--tokens T]

Exit code 0 if every check holds, 1 otherwise (CI-wireable).
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

Arr = np.ndarray

# An "order-1" relative error: the deployed path is not approximately right.
FAIL_FLOOR = 0.5
# Exactness bar for the folded path, in float64.
EXACT_CEIL = 1e-12


# ----------------------------------------------------------------- primitives


def rms_norm(x: Arr, g: Arr, eps: float = 1e-6) -> Arr:
    """Pre-norm RMSNorm with per-channel gain: (x / rms(x)) * g."""
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return (x / rms) * g


def middle(x: Arr, g: Arr, P: Arr) -> Arr:
    """One pre-norm residual sublayer -- stands in for the cloud's public stack."""
    return x + rms_norm(x, g) @ P


def deployed(h: Arr, W: Arr, g: Arr, P: Arr) -> Arr:
    """Exactly jacobi_server.py:883-896 over an unmodified cloud: W^T . middle(h @ W)."""
    return middle(h @ W, g, P) @ W.T


def fold(W: Arr, g: Arr, P: Arr) -> Arr:
    """QuaRot/SpinQuant-style fold: conjugate the projection and absorb the gain.

    Used with gain = ones, this makes middle'(h @ W) == middle(h) @ W exactly.
    """
    return W.T @ (g[:, None] * P) @ W


def rel_err(got: Arr, want: Arr) -> float:
    return float(np.linalg.norm(got - want) / np.linalg.norm(want))


def random_orthogonal(rng: np.random.Generator, dim: int) -> Arr:
    """Haar-distributed orthogonal matrix (QR with the standard sign fix)."""
    q, r = np.linalg.qr(rng.standard_normal((dim, dim)))
    return q * np.sign(np.diag(r))


def quantize_bf16(x: Arr) -> Arr:
    """Round float32 values to bf16 precision and return them as float32."""
    x32 = np.asarray(x, dtype=np.float32)
    bits = x32.view(np.uint32)
    rounded = bits + np.uint32(0x7fff) + ((bits >> 16) & 1)
    return (rounded & np.uint32(0xffff0000)).view(np.float32)


def signed_permutation(rng: np.random.Generator, dim: int) -> Arr:
    """Orthogonal matrix with exactly one +-1 per row and column."""
    w = np.zeros((dim, dim))
    w[np.arange(dim), rng.permutation(dim)] = rng.choice([-1.0, 1.0], size=dim)
    return w


# -------------------------------------------------------------------- checker


class Checker:
    """Collects named numeric checks and turns them into a PASS/FAIL exit code."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, float, str, bool]] = []

    def check(self, name: str, measured: float, expect: str, ok: bool) -> None:
        self.rows.append((name, measured, expect, ok))
        print(f"  {'ok ' if ok else 'BAD'}  {name:<68s} {measured:>11.4e}  expect {expect}")

    def broken(self, name: str, measured: float) -> None:
        """Expect the composition to be catastrophically wrong."""
        self.check(name, measured, f"> {FAIL_FLOOR}", measured > FAIL_FLOOR)

    def exact(self, name: str, measured: float, tol: float = EXACT_CEIL) -> None:
        """Expect the composition to be exact up to float64 rounding."""
        self.check(name, measured, f"< {tol:.0e}", measured < tol)

    def summary(self) -> int:
        bad = [r[0] for r in self.rows if not r[3]]
        print("\n" + "=" * 78)
        if not bad:
            print(f"PASS  ({len(self.rows)}/{len(self.rows)} checks held)")
            return 0
        print(f"FAIL  ({len(bad)}/{len(self.rows)} checks did not hold)")
        for name in bad:
            print(f"      - {name}")
        return 1


# ---------------------------------------------------------------------- cases


def run_equivariance(ck: Checker, rng: np.random.Generator, dim: int, tokens: int) -> None:
    """Claims 1-4: the deployed path is wrong; only folding fixes it."""
    h = rng.standard_normal((tokens, dim))
    g = rng.uniform(0.5, 2.0, size=dim)          # learned per-channel gain
    ones = np.ones(dim)
    proj = rng.standard_normal((dim, dim)) / np.sqrt(dim)   # standard 1/sqrt(H) init
    w_rand = random_orthogonal(rng, dim)
    w_perm = signed_permutation(rng, dim)

    print("\n[0] setup / harness sanity")
    ck.exact("||W^T W - I||_F  (random orthogonal)",
             float(np.linalg.norm(w_rand.T @ w_rand - np.eye(dim))))
    ck.exact("||W^T W - I||_F  (signed permutation)",
             float(np.linalg.norm(w_perm.T @ w_perm - np.eye(dim))))
    r_h = np.sqrt(np.mean(h * h, axis=-1))
    r_hw = np.sqrt(np.mean((h @ w_rand) ** 2, axis=-1))
    ck.exact("max |rms(hW) - rms(h)| / rms(h)   [norm IS preserved]",
             float(np.max(np.abs(r_hw - r_h) / r_h)))
    ck.exact("control: W = I reproduces middle(h)",
             rel_err(deployed(h, np.eye(dim), g, proj), middle(h, g, proj)))

    print("\n[1] deployed path, LEARNED gain -- jacobi_server.py:883-896 as shipped")
    ck.broken("rel_err( W^T middle(hW) , middle(h) )   g learned",
              rel_err(deployed(h, w_rand, g, proj), middle(h, g, proj)))

    print("\n[2] same, CONSTANT gain g = 1 -- the gain is not the culprit")
    ck.broken("rel_err( W^T middle(hW) , middle(h) )   g = 1",
              rel_err(deployed(h, w_rand, ones, proj), middle(h, ones, proj)))
    ck.broken("||W P W^T - P||_F / ||P||_F            [the real condition]",
              rel_err(w_rand @ proj @ w_rand.T, proj))

    print("\n[3] structured W (signed permutation) does not rescue it")
    ck.broken("rel_err( W^T middle(hW) , middle(h) )   perm, g learned",
              rel_err(deployed(h, w_perm, g, proj), middle(h, g, proj)))
    ck.broken("rel_err( W^T middle(hW) , middle(h) )   perm, g = 1",
              rel_err(deployed(h, w_perm, ones, proj), middle(h, ones, proj)))
    permuted_g = np.abs(w_perm) @ g   # g[sigma(i)]; |W| drops the sign, which cancels
    ck.exact("sanity: ||W diag(g) - diag(g_perm) W||_F  [perm DOES commute w/ gain]",
             float(np.linalg.norm(w_perm @ np.diag(g) - np.diag(permuted_g) @ w_perm)))

    print("\n[4] folding the weights (P' = W^T diag(g) P W, gain -> 1) is exact")
    for label, w_mat in (("random orthogonal", w_rand), ("signed permutation", w_perm)):
        ck.exact(f"rel_err( W^T middle'(hW) , middle(h) )  {label}",
                 rel_err(deployed(h, w_mat, ones, fold(w_mat, g, proj)), middle(h, g, proj)))


def run_key_recovery(ck: Checker, rng: np.random.Generator, dim: int) -> None:
    """Claim 5: folding leaks W, because the folded weights' originals are public.

    A cloud linear layer stores its weight as [d_out, H] and computes x @ Wq.T.
    To stay correct on the rotated stream x' = x @ W it must hold Wq' = Wq @ W
    (equivalently Wq'.T = W.T @ Wq.T).  Wq is public and downloadable, Wq' sits
    on the untrusted node, so W = argmin ||Wq @ X - Wq'|| is a least-squares
    solve -- unique exactly when rank(Wq) = H.

    Tolerance is conditioning-aware: the forward error of a least-squares solve
    is bounded by ~cond(Wq) * eps, and a SQUARE gaussian Wq is ill-conditioned
    (cond ~1e3-1e4 here), so a flat 1e-12 bar is a numerics artifact, not a
    statement about recoverability.  The condition number is printed alongside.
    The overdetermined stacked case -- what a real attacker uses -- has cond ~10
    and lands at ~1e-15 regardless.
    """
    w_secret = random_orthogonal(rng, dim)
    print("\n[5] security consequence: the fold leaks the secret key W")

    cases = (
        ("q_proj-like  [d_out=H]     full rank", dim, True),
        ("k_proj-like  [d_out=H/4]   RANK-DEFICIENT control", dim // 4, False),
        ("stacked q|k|v [d_out=1.5H] full rank", dim + 2 * (dim // 4), True),
    )
    for label, d_out, recoverable in cases:
        assert d_out > 0, f"degenerate d_out for {label}"
        public = rng.standard_normal((d_out, dim)) / np.sqrt(dim)
        folded = public @ w_secret                      # what the cloud would hold
        solved, _, rank, svals = np.linalg.lstsq(public, folded, rcond=None)
        assert rank == min(d_out, dim), f"unexpected rank {rank} for {label}"
        cond = float(svals.max() / svals.min())
        err = rel_err(solved, w_secret)
        name = f"rel_err(W_rec, W)  {label}  cond={cond:.1e}"
        if recoverable:
            ck.exact(name, err, tol=max(EXACT_CEIL, 64 * float(np.finfo(np.float64).eps) * cond))
        else:
            ck.broken(name, err)

    # Storage precision does not rescue the fold.  A real cloud normally has
    # bf16 public originals and bf16 folded weights, so quantize both sides and
    # solve an overdetermined stacked projection exactly as an observer would.
    d_out = dim + 2 * (dim // 4)
    public_bf16 = quantize_bf16(
        rng.standard_normal((d_out, dim)) / np.sqrt(dim))
    folded_bf16 = quantize_bf16(public_bf16 @ w_secret)
    solved, _, rank, _ = np.linalg.lstsq(
        public_bf16.astype(np.float64), folded_bf16.astype(np.float64),
        rcond=None)
    err = rel_err(solved, w_secret)
    ck.check("bf16 folded snapshot recovers W (stacked q|k|v)", err,
             "< 1e-2", rank == dim and err < 1e-2)


# ----------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Prove that E9's deployed boundary rotation (jacobi_server.py:883-896 "
                    "over an unmodified cloud_server_kv.py) does not compute middle(h), "
                    "that only weight folding fixes it, and that folding leaks the key.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--seed", type=int, default=0, help="RNG seed (deterministic)")
    ap.add_argument("--dim", type=int, default=512, help="hidden width H")
    ap.add_argument("--tokens", type=int, default=64, help="number of token rows")
    args = ap.parse_args()
    assert args.dim >= 8 and args.dim % 4 == 0, "--dim must be a multiple of 4 and >= 8"
    assert args.tokens >= 1, "--tokens must be >= 1"

    print("=" * 78)
    print("E9 EQUIVARIANCE PROBE   W^T . middle(h @ W)  vs  middle(h)")
    print("=" * 78)
    print("source   split-inference/jacobi_server.py:883-896   applies W then W^T at the wire")
    print("         split-inference/cloud_server_kv.py         no obfuscation code at all")
    print("model    middle(x) = x + rmsnorm(x, g) @ P          (minimal pre-norm block)")
    print(f"config   H={args.dim}  tokens={args.tokens}  seed={args.seed}  dtype=float64")
    print("         h ~ N(0,1)   P ~ N(0,1/H)   g ~ U(0.5,2.0)")

    rng = np.random.default_rng(args.seed)
    ck = Checker()
    run_equivariance(ck, rng, args.dim, args.tokens)
    run_key_recovery(ck, rng, args.dim)

    print("\n" + "-" * 78)
    print("CONSEQUENCE")
    print("  As deployed, E9 does not compute the model: the cloud's public middle layers")
    print("  are not equivariant under W, so W^T . middle(h @ W) != middle(h).  Any utility")
    print("  or acceptance number measured through that path describes a different function.")
    print("  The only composition that IS exact folds W into the cloud's weights, which puts")
    print("  W-transformed copies of PUBLIC weights on the untrusted node -- and section [5]")
    print("  recovers W from that pair by least squares to float64 precision.  E8's secrecy")
    print("  and E9's correctness cannot both hold.")
    print("-" * 78)
    return ck.summary()


if __name__ == "__main__":
    sys.exit(main())
