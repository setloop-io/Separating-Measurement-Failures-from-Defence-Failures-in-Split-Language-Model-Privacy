"""Shared provenance for engineering scenarios, with no scientific execution path."""
from datetime import datetime, timezone
from pathlib import Path
import json
import re

import jsonschema
import torch

from .experiment import ROOT, digest, source_provenance, verify_inherited_source, write_json


def require_engineering(cfg, kind):
    if (cfg.get("kind") != kind or cfg.get("cohort") != "engineering"
            or cfg.get("status") != "engineering_ready"
            or not cfg.get("protocol_id", "").startswith("p4-")):
        raise ValueError("Only explicitly identified engineering scenarios may execute")


def validate_seed_namespace(seeds, namespace):
    registry = json.loads((ROOT / "experiments/seed_registry.json").read_text())
    if not seeds or len(seeds) != len(set(seeds)) or not set(seeds) <= set(registry[namespace]):
        raise ValueError(f"Seeds must be unique reserved {namespace} IDs")


class Campaign:
    def __init__(self, cfg, config_path, label=None):
        require_engineering(cfg, cfg["kind"])
        label = label or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", label):
            raise ValueError("Invalid campaign label")
        verify_inherited_source()
        torch.set_num_threads(cfg.get("cpu_threads", 2))
        torch.use_deterministic_algorithms(not cfg.get("allow_nondeterministic_kernels", False))
        self.cfg, self.label = cfg, label
        self.path = ROOT / "outputs/scenarios" / label
        self.path.mkdir(parents=True, exist_ok=False)
        self.source = source_provenance(self.path, device=cfg.get("device", "cpu"),
                                        allow_nondeterministic=cfg.get("allow_nondeterministic_kernels", False))
        self.config_hash = digest(config_path)
        self.records = []
        self.schema = json.loads((ROOT / "experiments/schemas/scenario_manifest.schema.json").read_text())
        write_json(self.path / "config.json", cfg)

    def start(self, scenario_id, inputs):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", scenario_id):
            raise ValueError("Invalid scenario ID")
        run_id = f"p4-{self.cfg['kind']}-{self.label}-{scenario_id}"
        out = self.path / run_id
        out.mkdir(exist_ok=False)
        record = {"schema_version": "p4-scenario-0.1", "run_id": run_id,
                  "kind": self.cfg["kind"], "protocol_id": self.cfg["protocol_id"],
                  "cohort": "engineering", "scientific_evidence": False,
                  "source_tree_sha256": self.source["source_tree_sha256"],
                  "git_commit": self.source["git_commit"],
                  "source_record": str((self.path / "source.json").relative_to(ROOT)),
                  "config_sha256": self.config_hash, "inputs": inputs,
                  "started_at": datetime.now(timezone.utc).isoformat()}
        return out, record

    def finish(self, out, record, error=None):
        record.update(status="failed" if error else "completed",
                      finished_at=datetime.now(timezone.utc).isoformat(),
                      artifacts=[{"path": str(p.relative_to(ROOT)), "sha256": digest(p)}
                                 for p in sorted(out.iterdir()) if p.is_file()])
        if error:
            record["error"] = f"{type(error).__name__}: {error}"
        jsonschema.validate(record, self.schema)
        write_json(ROOT / "experiments/manifests" / f"{record['run_id']}.json", record)
        self.records.append(record)
        print(f"{record['run_id']}: {record['status']}", flush=True)

    def summarize(self, extra=None):
        result = {"protocol_id": self.cfg["protocol_id"], "label": self.label,
                  "kind": self.cfg["kind"], "cohort": "engineering", "scientific_evidence": False,
                  "completed": sum(r["status"] == "completed" for r in self.records),
                  "failed": sum(r["status"] == "failed" for r in self.records),
                  "privacy_outcome": "undetermined", "run_ids": [r["run_id"] for r in self.records],
                  **(extra or {})}
        write_json(self.path / "summary.json", result)
        return result
