"""Paper 4: Defense redesign under calibrated privacy instruments.

This package extends `pcr` (Paper 3's privacy-constrained-remote package) with
new boundary defense modules. It does NOT duplicate Paper 3 infrastructure —
it imports and extends it.

Defense candidates:
  D1 — property-targeted noise injection
  D2 — adversarial encoder training (minimax)
  D3 — aggressive rate reduction (q < 4 bits/dim)
  D4 — dimension-selective quantization
  D5 — sequence-level shuffling / permutation defense
"""

__version__ = "0.0.0"
