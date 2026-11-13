"""Read run artifacts only after verifying their source and manifest bindings.

Run-manifest analogue of scenario_verification.verify_scenario for the
outputs/runs campaigns (p4-run-0.1 / p4-run-0.2 manifests bound to a campaign
source archive). scenario_verification is left unchanged for scenario manifests.
"""
import hashlib
import json
import tarfile
import jsonschema
import yaml
from .experiment import ROOT, digest


def verify_run(manifest_path):
    path = ROOT / manifest_path
    record = json.loads(path.read_text())
    schema = json.loads((ROOT / "experiments/schemas/run_manifest.schema.json").read_text())
    jsonschema.validate(record, schema)
    if path.stem != record["run_id"]:
        raise ValueError("Manifest filename/identity mismatch")
    source_path = ROOT / record["source_record"]
    source = json.loads(source_path.read_text())
    tree_hash = hashlib.sha256(json.dumps(source["files"], sort_keys=True).encode()).hexdigest()
    if (tree_hash != source["source_tree_sha256"] or tree_hash != record["source_tree_sha256"]
            or source["git_commit"] != record["git_commit"]):
        raise ValueError("Manifest/source identity mismatch")
    config = None
    with tarfile.open(source_path.with_name("source.tar.gz"), "r:gz") as archive:
        for name, expected in source["files"].items():
            member = archive.extractfile(name)
            if member is None:
                raise ValueError(f"Missing source: {name}")
            data = member.read()
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f"Source archive mismatch: {name}")
            if name.endswith(".yaml") and expected == record["matrix_sha256"]:
                config = yaml.safe_load(data)
    if config is None or config["protocol_id"] != record["protocol_id"]:
        raise ValueError("Configuration is absent from the bound source archive")
    artifacts = {}
    for artifact in record["artifacts"]:
        p = ROOT / artifact["path"]
        if not p.is_file() or digest(p) != artifact["sha256"]:
            raise ValueError(f"Missing or changed artifact: {artifact['path']}")
        if p.name in artifacts:
            raise ValueError("Ambiguous artifact name")
        artifacts[p.name] = p
    return record, artifacts, config
