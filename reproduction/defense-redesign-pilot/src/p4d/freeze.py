"""Freeze-record generation and validation for the exploratory pilot freeze."""
from pathlib import Path
from datetime import datetime, timezone
import json
import subprocess

import yaml

from .experiment import ROOT, digest, write_json

FREEZE_ID = "p4-defense-pilot-1.0"

# Paths whose content the freeze pins: any change after the record's code_commit
# requires a numbered amendment and a new freeze record.
SOURCE_PATHS = ("src", "scripts", "tests", "experiments/configs", "docs", "vendor",
                "pyproject.toml", "uv.lock", "AGENTS.md")


def _git(args):
    return subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)


def git_head(root=ROOT):
    result = _git(["rev-parse", "HEAD"])
    return result.stdout.strip() if result.returncode == 0 else None


def make_freeze_record(config_path, protocol_doc_path, seed_audit_path,
                       qualification_verification_path, out_path, freeze_id=FREEZE_ID):
    """Write a freeze record binding the config, protocol document, corpus, audit, and code."""
    config_path, protocol_doc_path = Path(config_path), Path(protocol_doc_path)
    seed_audit_path, verification_path = Path(seed_audit_path), Path(qualification_verification_path)
    audit = json.loads(seed_audit_path.read_text())
    if audit.get("passed") is not True:
        raise ValueError("Seed audit did not pass; refusing to freeze")
    json.loads(verification_path.read_text())
    cfg = yaml.safe_load(config_path.read_text())
    record = {"freeze_id": freeze_id, "protocol_id": cfg["protocol_id"],
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "config_sha256": digest(config_path),
              "protocol_doc_sha256": digest(protocol_doc_path),
              "corpus_sha256": cfg["data"]["sha256"],
              "documents_sha256": digest(ROOT / cfg["data"]["documents"]),
              "seed_audit": {"passed": True, "sha256": digest(seed_audit_path)},
              "qualification_verification_sha256": digest(verification_path),
              "seeds": {"pilot": list(cfg["seeds"])},
              "code_commit": git_head(),
              "amendments": []}
    write_json(out_path, record)
    return record


def require_freeze_record(cfg, config_path):
    """Refuse execution unless the committed freeze record matches this exact context."""
    path = cfg.get("freeze_record")
    if not path:
        raise ValueError("The frozen pilot config must name its freeze record")
    record_path = ROOT / path
    if not record_path.is_file():
        raise ValueError(f"Freeze record {path} is absent; the exploratory pilot is disabled until freeze")
    record = json.loads(record_path.read_text())
    if record.get("config_sha256") != digest(config_path):
        raise ValueError("Pilot config differs from the freeze record")
    if record.get("protocol_id") != cfg.get("protocol_id"):
        raise ValueError("Freeze record protocol differs from the config")
    if record.get("seeds", {}).get("pilot") != list(cfg["seeds"]):
        raise ValueError("Freeze record seeds differ from the config")
    if record.get("corpus_sha256") != cfg["data"]["sha256"]:
        raise ValueError("Freeze record corpus differs from the config")
    if git_head() != record.get("code_commit"):
        base = record.get("code_commit")
        if not base or _git(["merge-base", "--is-ancestor", base, "HEAD"]).returncode != 0:
            raise ValueError("Code commit differs from the freeze record")
        if _git(["diff", "--quiet", base, "HEAD", "--", *SOURCE_PATHS]).returncode != 0:
            raise ValueError("Source files changed after the freeze; a numbered amendment is required")
    return record
