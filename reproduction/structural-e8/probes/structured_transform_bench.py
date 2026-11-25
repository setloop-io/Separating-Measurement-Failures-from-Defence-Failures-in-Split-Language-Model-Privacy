#!/usr/bin/env python3
"""Benchmark: dense orthogonal W vs structured alternatives for the E8
boundary transform (local: h' = h·W; cloud-side tail: h = h'·W^T).

e8_obfuscation.py uses a dense H×H orthogonal W (QR of seeded randn): one
O(H^2) matmul per direction and H^2 floats of secret state. This probe
measures cheaper structured LINEAR ISOMETRIES that could replace it:

  dense            : full orthogonal W (the E8 baseline)
  perm_sign        : permutation + random sign flips — O(H), H params
  block_rot(B)     : block-diagonal orthogonal, block size B in {64, 128} —
                     O(H·B), H·B params
  walsh_hadamard   : fast Walsh-Hadamard butterfly — O(H log H), no stored
                     matrix. For non-power-of-2 H, the butterfly runs per
                     block of size P = largest power of two dividing H, then
                     a small dense orthogonal mixes across the H/P blocks
                     (keeps the transform an exact isometry).

Per (H, transform): ms per transform, secret-state memory, and the
inversion (round-trip) error max|x·T·T^{-1} - x|.

SECURITY NOTE: every candidate here is a linear isometry, exactly like the
dense W. Perm/sign and block-diagonal structure shrink the secret's entropy
(a perm+sign secret has only H!·2^H possibilities, and a passive observer
who sees h·W can read off |h| up to permutation — coordinate magnitudes
leak). The E8 defense claim ("a secret linear isometry at the boundary
collapses the public-weights trained attacker") is unchanged either way;
this bench only quantifies the speed/memory/error trade-off, not security.

Usage:
    python structured_transform_bench.py --help     # works without torch
    python structured_transform_bench.py            # full bench (CPU ok)
    python structured_transform_bench.py --quick    # H=1024 only, fewer reps
"""

import argparse
import json
import os
import time

# Guarded heavy import: `--help` must work on torch-less hosts (repo rule).
try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def largest_pow2_divisor(n):
    p = 1
    while n % 2 == 0:
        n //= 2
        p *= 2
    return p


def fwht_(x):
    """In-place unnormalized Walsh-Hadamard butterfly along the last dim.
    Last-dim size must be a power of two. O(H log H)."""
    h = 1
    n = x.shape[-1]
    while h < n:
        x = x.reshape(*x.shape[:-1], n // (2 * h), 2, h)
        a, b = x[..., 0, :], x[..., 1, :]
        x = torch.stack([a + b, a - b], dim=-2)
        x = x.reshape(*x.shape[:-3], n)
        h *= 2
    return x


class Transform:
    """name, secret-state bytes, forward(x), inverse(y). All exact isometries."""

    def __init__(self, name, H, seed, block=64):
        self.name = name
        g = torch.Generator().manual_seed(seed)
        if name == "dense":
            q, _ = torch.linalg.qr(torch.randn(H, H, generator=g))
            self.W = q
            self.Wt = q.T.contiguous()
            self.nbytes = 2 * H * H * 4
            self.fwd = lambda x: x @ self.W
            self.inv = lambda y: y @ self.Wt
        elif name == "perm_sign":
            self.perm = torch.randperm(H, generator=g)
            self.signs = (torch.randint(0, 2, (H,), generator=g) * 2 - 1).float()
            self.inv_perm = torch.argsort(self.perm)
            self.nbytes = H * 4 + H * 4
            self.fwd = lambda x: x[..., self.perm] * self.signs
            self.inv = lambda y: (y * self.signs)[..., self.inv_perm]
        elif name == "block_rot":
            B = block
            assert H % B == 0, f"H={H} not divisible by block={B}"
            nb = H // B
            q, _ = torch.linalg.qr(torch.randn(nb, B, B, generator=g))
            self.Wb = q  # [nb, B, B]
            self.nbytes = 2 * nb * B * B * 4
            self.fwd = lambda x: torch.einsum(
                "...nb,nbk->...nk",
                x.reshape(*x.shape[:-1], nb, B), self.Wb).reshape(*x.shape)
            self.inv = lambda y: torch.einsum(
                "...nk,nbk->...nb",
                y.reshape(*y.shape[:-1], nb, B), self.Wb).reshape(*y.shape)
        elif name == "walsh_hadamard":
            P = largest_pow2_divisor(H)
            nb = H // P
            self.P, self.nb = P, nb
            self.scale = 1.0 / (P ** 0.5)
            if nb > 1:
                # small dense orthogonal mixing across blocks (isometry)
                q, _ = torch.linalg.qr(torch.randn(nb, nb, generator=g))
                self.M = q
                self.nbytes = 2 * nb * nb * 4
            else:
                self.M = None
                self.nbytes = 0

            def fwd(x):
                y = fwht_(x.reshape(*x.shape[:-1], nb, P)) * self.scale
                if self.M is not None:
                    y = torch.einsum("...bp,bc->...cp", y, self.M)
                return y.reshape(*x.shape)

            def inv(y):
                z = y.reshape(*y.shape[:-1], nb, P)
                if self.M is not None:
                    z = torch.einsum("...cp,bc->...bp", z, self.M)
                z = fwht_(z) * self.scale  # butterfly is its own inverse (up to scale)
                return z.reshape(*y.shape)

            self.fwd, self.inv = fwd, inv
        else:
            raise ValueError(name)


def bench(H, seq_len, reps, seed, device):
    x = torch.randn(1, seq_len, H, device=device)
    rows = []
    for name, block in [("dense", None), ("perm_sign", None),
                        ("block_rot", 64), ("block_rot", 128),
                        ("walsh_hadamard", None)]:
        tag = name if block is None else f"{name}(B={block})"
        t = Transform(name, H, seed, block or 64)
        if device.startswith("cuda"):
            t.fwd = _wrap_dev(t.fwd, device)  # transforms are CPU-built; move lazily
            t.inv = _wrap_dev(t.inv, device)
        # warmup
        for _ in range(3):
            t.inv(t.fwd(x))
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            y = t.fwd(x)
            z = t.inv(y)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / reps
        err = (z - x).abs().max().item()
        rows.append({"H": H, "transform": tag,
                     "ms_per_transform": round(dt / 2 * 1000, 4),
                     "secret_bytes": t.nbytes,
                     "roundtrip_max_abs_err": float(f"{err:.3e}")})
        print(f"  H={H} {tag:22s} {dt / 2 * 1000:8.4f} ms/transform  "
              f"secret={t.nbytes:>12d} B  roundtrip_err={err:.2e}")
    return rows


def _wrap_dev(fn, device):
    """CPU-built weights used from CUDA: fall back to CPU round-trip.
    (The bench is indicative; GPU deployments would store weights on-device.)"""
    def wrapped(x):
        return fn(x.cpu()).to(device)
    return wrapped


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", type=int, nargs="+", default=[1024, 4096, 5120],
                    help="hidden sizes H to bench")
    ap.add_argument("--seq-len", type=int, default=64,
                    help="sequence positions per transform call")
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device",
                    default="cuda" if torch is not None and torch.cuda.is_available() else "cpu")
    ap.add_argument("--quick", action="store_true",
                    help="H=1024 only, 20 reps (machinery check)")
    ap.add_argument("--output", default="structured_transform_bench.json")
    args = ap.parse_args()

    if torch is None:
        ap.error("torch not installed; install it or run --help only")

    if args.quick:
        args.sizes = [1024]
        args.reps = 20

    results = []
    for H in args.sizes:
        print(f"[bench] H={H} seq={args.seq_len} reps={args.reps} "
              f"device={args.device}")
        results.extend(bench(H, args.seq_len, args.reps, args.seed, args.device))

    out = {
        "config": {"sizes": args.sizes, "seq_len": args.seq_len,
                   "reps": args.reps, "seed": args.seed,
                   "device": args.device, "quick": args.quick,
                   "hostname": os.uname().nodename},
        "security_note": ("all candidates are linear isometries, like the "
                          "dense W; the E8 defense claim is unchanged. "
                          "Structured secrets carry less entropy than a dense "
                          "orthogonal matrix (perm+sign leaks coordinate "
                          "magnitudes) — this bench measures the "
                          "speed/memory/error trade-off only."),
        "results": results,
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
