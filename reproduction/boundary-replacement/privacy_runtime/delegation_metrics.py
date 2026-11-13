"""Definitions for Paper 2 delegation measurements.

The functions in this module deliberately keep model coverage, surrogate work,
and observed transport separate.  A covered layer fraction is not a measured
remote-compute fraction.
"""

from __future__ import annotations

import math


def nll_to_perplexity(nll: float) -> float:
    """Convert mean token negative log-likelihood to perplexity."""
    if not math.isfinite(nll):
        raise ValueError("NLL must be finite")
    return math.exp(nll)


def nominal_covered_layer_share(total_layers: int, split_after: int,
                                resume_after: int) -> float:
    """Fraction of backbone layers omitted from the trusted path."""
    if total_layers <= 0:
        raise ValueError("total_layers must be positive")
    if not 0 <= split_after < resume_after <= total_layers:
        raise ValueError("invalid protected-boundary span")
    return (resume_after - split_after - 1) / total_layers


def make_utility_metrics(nll: float, reference_nll: float) -> dict[str, float]:
    """Return NLL-first utility values; perplexity is a derived reporting view."""
    return {
        "mean_token_nll": nll,
        "perplexity": nll_to_perplexity(nll),
        "nll_delta_vs_no_replacement": nll - reference_nll,
    }


def make_transport_metrics(*, forward_outbound_wire_bytes: int,
                           forward_inbound_wire_bytes: int,
                           backward_outbound_wire_bytes: int,
                           backward_inbound_wire_bytes: int,
                           control_outbound_wire_bytes: int,
                           control_inbound_wire_bytes: int,
                           remote_forward_compute_ms: float,
                           remote_backward_compute_ms: float,
                           actual_remote_flops: int | None = None,
                           actual_remote_flops_status: str = "not_profiled") -> dict[str, float | int | None]:
    """Summarise observed protocol traffic without inventing FLOP counts."""
    values = (
        forward_outbound_wire_bytes, forward_inbound_wire_bytes,
        backward_outbound_wire_bytes, backward_inbound_wire_bytes,
        control_outbound_wire_bytes, control_inbound_wire_bytes,
    )
    if any(value < 0 for value in values):
        raise ValueError("wire byte counts must be non-negative")
    return {
        "forward_outbound_wire_bytes": forward_outbound_wire_bytes,
        "forward_inbound_wire_bytes": forward_inbound_wire_bytes,
        "backward_outbound_wire_bytes": backward_outbound_wire_bytes,
        "backward_inbound_wire_bytes": backward_inbound_wire_bytes,
        "control_outbound_wire_bytes": control_outbound_wire_bytes,
        "control_inbound_wire_bytes": control_inbound_wire_bytes,
        "total_wire_bytes": sum(values),
        "remote_forward_compute_ms": remote_forward_compute_ms,
        "remote_backward_compute_ms": remote_backward_compute_ms,
        "actual_remote_flops": actual_remote_flops,
        "actual_remote_flops_status": actual_remote_flops_status,
    }
