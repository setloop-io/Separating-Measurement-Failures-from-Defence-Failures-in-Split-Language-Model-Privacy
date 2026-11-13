#!/usr/bin/env python3
"""Deterministic external collector fixture; production manifests must reject it."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p4d.cc_external_collector import (
    AUTH_SCHEMA,
    PROTOCOL_SCHEMA,
    canonical_bytes,
    canonical_sha256,
    external_auth_payload,
)


def send(value) -> None:
    sys.stdout.buffer.write(canonical_bytes(value) + b"\n")
    sys.stdout.buffer.flush()


def receive() -> dict:
    line = sys.stdin.buffer.readline()
    if not line:
        raise EOFError("controller closed input")
    value = json.loads(line)
    if not isinstance(value, dict):
        raise ValueError("request must be an object")
    return value


def sign(payload: bytes, private_key: Path) -> bytes:
    with tempfile.TemporaryDirectory(prefix="p4-cc-test-collector-") as temporary:
        payload_path = Path(temporary) / "payload.bin"
        signature_path = Path(temporary) / "signature.bin"
        payload_path.write_bytes(payload)
        result = subprocess.run(
            ["openssl", "pkeyutl", "-sign", "-inkey", str(private_key),
             "-rawin", "-in", str(payload_path), "-out", str(signature_path)],
            capture_output=True, check=False, timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError("fixture signing failed")
        return signature_path.read_bytes()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-key", type=Path, required=True)
    args = parser.parse_args()
    start = receive()
    start_fields = {
        "protocol_schema", "type", "collector_id", "session_id",
        "admission_sha256", "workload_config_sha256",
        "collector_manifest_sha256", "collector_executable_sha256",
        "collector_public_key_sha256",
    }
    if (set(start) != start_fields or start["protocol_schema"] != PROTOCOL_SCHEMA
            or start["type"] != "start"):
        raise ValueError("invalid start request")
    send({
        "protocol_schema": PROTOCOL_SCHEMA,
        "type": "ready",
        "ready": True,
        "collector_id": start["collector_id"],
        "collector_manifest_sha256": start["collector_manifest_sha256"],
        "collector_executable_sha256": start["collector_executable_sha256"],
        "collector_public_key_sha256": start["collector_public_key_sha256"],
        "session_id": start["session_id"],
        "admission_sha256": start["admission_sha256"],
        "workload_config_sha256": start["workload_config_sha256"],
    })
    journal_id = None
    lines: list[bytes] = []
    while True:
        request = receive()
        if request.get("protocol_schema") != PROTOCOL_SCHEMA:
            raise ValueError("protocol mismatch")
        if request.get("type") == "begin_journal":
            if journal_id is not None or set(request) != {
                    "protocol_schema", "type", "journal_id"}:
                raise ValueError("invalid begin request")
            journal_id = request["journal_id"]
            lines = []
            send({"protocol_schema": PROTOCOL_SCHEMA, "type": "journal_ready",
                  "journal_id": journal_id})
        elif request.get("type") == "observe":
            if (journal_id is None or request.get("journal_id") != journal_id
                    or set(request) != {"protocol_schema", "type", "journal_id",
                                       "trigger", "trigger_sha256"}
                    or canonical_sha256(request["trigger"]) != request["trigger_sha256"]):
                raise ValueError("invalid observe request")
            trigger = request["trigger"]
            sequence = len(lines)
            event = {
                "schema_version": "p4-cc-external-event-0.2",
                "correlation_id": trigger["correlation_id"],
                "sequence": trigger["sequence"],
                "view": "external_attacker",
                "session_id": trigger["session_id"],
                "request_id": trigger["request_id"],
                "step_id": trigger["step_id"],
                "phase": trigger["phase"],
                "operation": trigger["operation"],
                "collector_id": start["collector_id"],
                "collector_measurement_sha256": start["collector_executable_sha256"],
                "collector_sequence": sequence,
                "trigger_sha256": request["trigger_sha256"],
                "status": "observed",
                "observation": {
                    "collector_id": start["collector_id"],
                    "collector_measurement_sha256": start[
                        "collector_executable_sha256"],
                    "metadata_fields": ["length", "timing", "direction", "phase"],
                    "wire_bytes": 128 + sequence,
                    "packet_count": 1,
                    "elapsed_ns": 1000 + sequence,
                    "status_code": "fixture-observed",
                    "observation_sha256": request["trigger_sha256"],
                    "limitation": "Deterministic CPU fixture; no host traffic was measured.",
                },
                "reason": "Deterministic test-only live collector observation.",
            }
            encoded = canonical_bytes(event)
            lines.append(encoded)
            send({
                "protocol_schema": PROTOCOL_SCHEMA,
                "type": "observation",
                "journal_id": journal_id,
                "trigger_sha256": request["trigger_sha256"],
                "event": event,
            })
        elif request.get("type") == "finish_journal":
            if (journal_id is None or request.get("journal_id") != journal_id
                    or set(request) != {"protocol_schema", "type", "journal_id",
                                       "external_journal_sha256", "event_count"}):
                raise ValueError("invalid finish request")
            journal_bytes = b"".join(line + b"\n" for line in lines)
            if (hashlib.sha256(journal_bytes).hexdigest()
                    != request["external_journal_sha256"]
                    or request["event_count"] != len(lines)):
                raise ValueError("controller/collector journal bytes differ")
            payload = external_auth_payload(
                session_id=start["session_id"], journal_id=journal_id,
                journal_sha256=request["external_journal_sha256"],
                event_count=len(lines), admission_sha256=start["admission_sha256"],
                workload_config_sha256=start["workload_config_sha256"],
                collector_manifest_sha256=start["collector_manifest_sha256"],
                collector_executable_sha256=start["collector_executable_sha256"],
                collector_public_key_sha256=start["collector_public_key_sha256"],
            )
            encoded = canonical_bytes(payload)
            auth = {
                "schema_version": AUTH_SCHEMA,
                "payload": payload,
                "payload_sha256": hashlib.sha256(encoded).hexdigest(),
                "signature_base64": base64.b64encode(
                    sign(encoded, args.private_key)).decode("ascii"),
            }
            send({"protocol_schema": PROTOCOL_SCHEMA, "type": "journal_auth",
                  "journal_id": journal_id, "auth": auth})
            journal_id = None
            lines = []
        elif request.get("type") == "shutdown":
            if journal_id is not None or set(request) != {"protocol_schema", "type"}:
                raise ValueError("invalid shutdown request")
            send({"protocol_schema": PROTOCOL_SCHEMA, "type": "stopped"})
            return 0
        else:
            raise ValueError("unknown collector request")


if __name__ == "__main__":
    raise SystemExit(main())
