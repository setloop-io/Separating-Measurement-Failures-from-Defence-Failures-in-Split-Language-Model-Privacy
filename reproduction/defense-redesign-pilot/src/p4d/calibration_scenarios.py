"""Exercise controls on hash-bound P0 captures; never qualify scientific attacks."""
from pathlib import Path
import json
import math

import jsonschema
import yaml
import torch

from pcr.utils import derive_seed
from .calibration import score_properties, score_structure, qualification_from_controls
from .controls import VIEWS, observation_panel, donor_null, signal_spec, inject_signal
from .experiment import ROOT, digest, write_json
from .scenario_io import Campaign, require_engineering, validate_seed_namespace


def validate_config(cfg):
    require_engineering(cfg, "calibration")
    validate_seed_namespace(cfg["control_seeds"], "calibration_engineering")
    if not cfg["source_manifests"] or len(set(cfg["source_manifests"])) != len(cfg["source_manifests"]):
        raise ValueError("Need unique source manifests")
    if not cfg["views"] or len(set(cfg["views"])) != len(cfg["views"]) or not set(cfg["views"]) <= set(VIEWS):
        raise ValueError("Need unique implemented observation views")
    if cfg["targets"] != [{"family": "sequence_property", "target": "punctuation_fraction"},
                           {"family": "next_byte_structure", "target": "next_byte_class"}]:
        raise ValueError("Initial engineering matrix fixes the two primary targets")
    if not cfg["doses"] or any(not math.isfinite(v) or v <= 0 for v in cfg["doses"]):
        raise ValueError("Positive doses must be finite and nonzero")
    if len(set(cfg["doses"])) != len(cfg["doses"]) or cfg["probe_epochs"] < 1:
        raise ValueError("Duplicate doses or invalid attack budget")
    return len(cfg["source_manifests"]) * len(cfg["views"]) * len(cfg["targets"]) * len(cfg["control_seeds"])


def load_bound_captures(manifest_path):
    path = ROOT / manifest_path
    manifest = json.loads(path.read_text())
    schema = json.loads((ROOT / "experiments/schemas/run_manifest.schema.json").read_text())
    jsonschema.validate(manifest, schema)
    if manifest["status"] != "completed" or manifest["mode"] == "all_local":
        raise ValueError("Need completed split captures")
    artifacts = {}
    for artifact in manifest["artifacts"]:
        p = ROOT / artifact["path"]
        if not p.is_file() or digest(p) != artifact["sha256"]:
            raise ValueError(f"Missing or changed source artifact: {artifact['path']}")
        artifacts[p.name] = p
    resolved = json.loads(artifacts["resolved.json"].read_text())
    if (resolved["cell"]["id"] != manifest["cell"] or resolved["mode"] != manifest["mode"]
            or resolved["seed"] != manifest["seed"]):
        raise ValueError("Manifest/config treatment mismatch")
    capture_path = artifacts["captures.pt"]
    panels = torch.load(capture_path, weights_only=True)
    binding = {"source_run": manifest["run_id"], "source_manifest_sha256": digest(path),
               "capture_sha256": digest(capture_path), "source_tree_sha256": manifest["source_tree_sha256"],
               "corpus_sha256": manifest["corpus_sha256"], "training_seed": manifest["seed"],
               "mode": manifest["mode"], "cell": manifest["cell"],
               "defense": resolved["cell"]["defense"], "q_bits": resolved["cell"]["q_bits"],
               "basis_policy": resolved["matrix"]["basis_policy"],
               "attacker_side_information": "same-model paired public-text fit/labels; diagnostic P0 captures"}
    return panels, binding


def score(panel_fit, panel_test, family, target, seed, epochs):
    if family == "sequence_property":
        return score_properties(panel_fit.events(), panel_test.events(), seed)[target]
    return score_structure(panel_fit.events(), panel_test.events(), seed, epochs)


def exercise_controls(events, binding, view, family, target, seed, doses, epochs):
    fit, test = (observation_panel(events[k], view) for k in ("fit", "test"))
    if fit.sha256() == test.sha256():
        raise ValueError("Identical fit/evaluation panel")
    nf, fit_donors = donor_null(fit, derive_seed("p4-null-fit", seed))
    nt, test_donors = donor_null(test, derive_seed("p4-null-test", seed))
    spec = signal_spec(fit, family, target, derive_seed("p4-control-geometry", seed))
    attack_seed = derive_seed("p4-control-attacker", seed)
    exact_binding = {**binding, "channel": view, "target": target, "family": family,
                     "fit_sha256": fit.sha256(), "test_sha256": test.sha256()}
    normal = score(fit, test, family, target, attack_seed, epochs)
    null = score(nf, nt, family, target, attack_seed, epochs)
    rows = []
    for dose in doses:
        pf, pt = inject_signal(nf, spec, dose), inject_signal(nt, spec, dose)
        positive = score(pf, pt, family, target, attack_seed, epochs)
        check = qualification_from_controls(positive["advantage"], null["advantage"], family, exact_binding)
        rows.append({"dose": dose, "positive_score": positive,
                     "positive_fit_sha256": pf.sha256(), "positive_test_sha256": pt.sha256(),
                     "control_checks_passed": check.pop("qualified"), "thresholds": check})
    return {"schema_version": "p4-control-diagnostic-0.1", "binding": exact_binding,
            "cohort": "engineering", "scientific_evidence": False, "qualified": False,
            "qualification_status": "engineering_only", "privacy_outcome": "undetermined",
            "normal_score": normal, "null_score": null, "doses": rows,
            "control_seed": seed, "attacker_seed": attack_seed, "probe_epochs": epochs,
            "null_fit_sha256": nf.sha256(), "null_test_sha256": nt.sha256(),
            "fit_donors": fit_donors, "test_donors": test_donors,
            "signal_spec": {k: v.tolist() if isinstance(v, torch.Tensor) else v for k, v in spec.items()},
            "scope": "Post-release control copies; no pre-defense removal test, no re-quantization; finite donor null can retain association."}


def run(config_path, label=None, validate_only=False):
    cfg = yaml.safe_load(Path(config_path).read_text())
    count = validate_config(cfg)
    if validate_only:
        return {"scenario_count": count, "positive_control_scores": count * len(cfg["doses"]),
                "scientific_execution_enabled": False}
    # Resolve all inputs before creating a campaign, so missing ignored tensors are clear.
    sources = [load_bound_captures(p) for p in cfg["source_manifests"]]
    campaign = Campaign(cfg, config_path, label)
    for events, binding in sources:
        for view in cfg["views"]:
            for target in cfg["targets"]:
                for seed in cfg["control_seeds"]:
                    ident = f"{binding['cell']}-{binding['mode']}-{view}-{target['target']}-s{seed}"
                    out, manifest = campaign.start(ident, binding)
                    error = None
                    try:
                        result = exercise_controls(events, binding, view, **target, seed=seed,
                                                   doses=cfg["doses"], epochs=cfg["probe_epochs"])
                        schema = json.loads((ROOT / "experiments/schemas/control_diagnostic.schema.json").read_text())
                        jsonschema.validate(result, schema)
                        write_json(out / "controls.json", result)
                    except Exception as exc:
                        error = exc
                    campaign.finish(out, manifest, error)
    return campaign.summarize({"expected_scenarios": count, "qualification_status": "engineering_only",
        "omitted": ["scientific instrument qualification", "active/adaptive attacks", "state/optimizer attacks",
                    "general joint tensors", "cross-run/history attacks", "membership", "timing", "unrestricted semantics"]})
