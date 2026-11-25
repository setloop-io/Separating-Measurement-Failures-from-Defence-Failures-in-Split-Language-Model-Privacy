"""Stable identities for joining delegation and privacy measurements."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


SCHEMA = "dtraining.privacy_cell_identity.v1"

# These are the knobs that can change the protected-boundary view or the
# delegated computation. Paths are retained because a local model/corpus path
# is the runner's only immutable identifier until content hashes are available.
RUNNER_FIELDS = (
    "model", "corpus", "seed", "split_after", "resume_after",
    "sequence_length", "steps", "warmup_steps", "lr", "train_blocks",
    "eval_blocks", "latent_dim", "noise_multiplier", "clip_norm",
    "adversary_strength", "cloud_kind", "cloud_experts", "cloud_layers",
    "cloud_channels", "fragment_channels", "public_steps",
    "byzantine_verify", "byzantine_threshold", "secret_wire_rotation",
    "secret_token_permutation", "secret_token_gauge", "token_scale_sigma",
    "chaff_tokens", "mine_penalty", "bundle_canonical_fraction",
    "wire_quant", "gram_flatten", "remote_grad_clip", "surrogate_mode",
    "surrogate_placement", "delegate_sublayer",
    "analysis_stage", "frozen_grid_id", "frozen_grid_cell_id",
)

OPTIONAL_DEFAULTS = {
    # Older runner artifacts omitted this default-valued field deliberately.
    "delegate_sublayer": "full",
    "analysis_stage": "exploratory",
    "frozen_grid_id": None,
    "frozen_grid_cell_id": None,
}


def runner_cell_payload(runner: Mapping[str, Any]) -> dict[str, Any]:
    """Return the canonical experimental coordinates from a runner artifact."""
    missing = [field for field in RUNNER_FIELDS
               if field not in runner and field not in OPTIONAL_DEFAULTS]
    if missing:
        raise ValueError("runner artifact lacks privacy identity fields: "
                         + ", ".join(missing))
    return {"schema": SCHEMA,
            "coordinates": {field: runner.get(field, OPTIONAL_DEFAULTS.get(field))
                            for field in RUNNER_FIELDS}}


def privacy_cell_id(runner: Mapping[str, Any]) -> str:
    """Hash canonical runner coordinates without depending on JSON key order."""
    encoded = json.dumps(runner_cell_payload(runner), sort_keys=True,
                         separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()
