"""Live, separately signed external-view collection for CC engineering runs.

The collector is an independent process speaking newline-delimited JSON over
stdin/stdout.  It must be ready before any accelerator work begins.  Each
trusted event is synchronously correlated with one external observation while
the workload is running; there is no supported post-hoc template-filling path
for production.  The collector signs the exact final JSONL with a distinct
Ed25519 key whose public half, executable bytes, and manifest are admission
bound.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import subprocess
import tempfile
from typing import Any, Callable, Mapping, Sequence

from .cc_io import (read_json_snapshot, read_snapshot, require_unchanged,
                    write_json_exclusive)


MANIFEST_SCHEMA = "p4-cc-external-collector-manifest-0.1"
PROTOCOL_SCHEMA = "p4-cc-external-collector-protocol-0.1"
AUTH_SCHEMA = "p4-cc-external-journal-auth-0.1"
AUTH_PAYLOAD_SCHEMA = "p4-cc-external-journal-auth-payload-0.1"
IDENTIFIER = re.compile(r"^[A-Za-z0-9_.:-]+$")
SHA256 = re.compile(r"^[a-f0-9]{64}$")
HEX_IDENTIFIER = re.compile(r"^[a-f0-9]{64}$")
CLAIM_SCOPE = (
    "Live external metadata collection for an engineering workload; not a "
    "privacy verdict or evidence for an unobserved channel."
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_ed25519_public_key(path: str | Path) -> None:
    """Reject keys that can verify a signature but are not Ed25519 keys."""
    try:
        result = subprocess.run(
            ["openssl", "pkey", "-pubin", "-in", str(path), "-text_pub", "-noout"],
            capture_output=True, check=False, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"cannot inspect collector public key: {exc}") from exc
    first = result.stdout.splitlines()[0] if result.stdout else b""
    if result.returncode != 0 or first.strip() != b"ED25519 Public-Key:":
        raise PermissionError("collector public key must be Ed25519")


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise ValueError(f"invalid {label}")
    # A digest-shaped string is not accepted as a human/domain identifier.  It
    # prevents callers from satisfying identity fields with invented hex.
    if HEX_IDENTIFIER.fullmatch(value):
        raise ValueError(f"{label} cannot be a fabricated digest-shaped identifier")
    return value


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise ValueError(f"invalid {label} SHA-256")
    return value


def _resolve_manifest_file(manifest_path: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"collector {label} path is required")
    raw = Path(value)
    path = raw if raw.is_absolute() else manifest_path.parent / raw
    path = path.resolve()
    if not path.is_file() or path.is_symlink() or path.stat().st_size < 1:
        raise ValueError(f"collector {label} must be a nonempty non-symlink file")
    return path


def load_collector_manifest(
        manifest_path: str | Path, *, production: bool,
        expected_argv: Sequence[str] | None = None,
        expected_public_key_path: str | Path | None = None) -> dict[str, Any]:
    """Validate a collector manifest against actual executable/key bytes."""
    manifest_path = Path(manifest_path).resolve()
    try:
        manifest, manifest_snapshot = read_json_snapshot(manifest_path)
    except (OSError, ValueError) as exc:
        raise ValueError(f"collector manifest cannot be read: {exc}") from exc
    fields = {
        "schema_version", "collector_id", "protocol_schema", "test_only",
        "argv", "executable_path", "executable_sha256", "public_key_path",
        "public_key_sha256", "claim_scope",
    }
    if (not isinstance(manifest, Mapping) or set(manifest) != fields
            or manifest.get("schema_version") != MANIFEST_SCHEMA
            or manifest.get("protocol_schema") != PROTOCOL_SCHEMA
            or manifest.get("claim_scope") != CLAIM_SCOPE):
        raise ValueError("collector manifest fields or protocol differ")
    _identifier(manifest["collector_id"], "collector_id")
    if type(manifest["test_only"]) is not bool:
        raise ValueError("collector test_only flag must be boolean")
    if production and manifest["test_only"]:
        raise PermissionError("test-only collector cannot satisfy a production workload")
    argv = manifest["argv"]
    if (not isinstance(argv, list) or not argv
            or any(not isinstance(item, str) or not item for item in argv)):
        raise ValueError("collector argv must be a nonempty string list")
    if expected_argv is not None and list(expected_argv) != argv:
        raise PermissionError("explicit collector argv differs from its manifest")
    executable = _resolve_manifest_file(manifest_path, manifest["executable_path"],
                                        "executable")
    argv_executable = Path(argv[0]).resolve()
    if argv_executable != executable:
        raise ValueError("collector argv[0] differs from the measured executable")
    if not os.access(executable, os.X_OK):
        raise PermissionError("collector executable is not executable")
    executable_snapshot = read_snapshot(
        executable, expected_sha256=_sha(manifest["executable_sha256"], "executable"))
    if production and executable_snapshot.data.startswith(b"#!"):
        raise PermissionError(
            "production collector must be a native measured executable, not a script")
    public_key = _resolve_manifest_file(manifest_path, manifest["public_key_path"],
                                       "public key")
    if expected_public_key_path is not None \
            and Path(expected_public_key_path).resolve() != public_key:
        raise PermissionError("explicit collector public key differs from its manifest")
    public_key_snapshot = read_snapshot(
        public_key, expected_sha256=_sha(manifest["public_key_sha256"], "public key"))
    require_ed25519_public_key(public_key)
    require_unchanged(public_key_snapshot)
    return {
        **dict(manifest),
        "manifest_path": manifest_path,
        "manifest_sha256": manifest_snapshot.sha256,
        "resolved_executable_path": executable,
        "resolved_public_key_path": public_key,
        "_manifest_snapshot": manifest_snapshot,
        "_executable_snapshot": executable_snapshot,
        "_public_key_snapshot": public_key_snapshot,
    }


def trigger_from_trusted_event(event: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "correlation_id", "sequence", "session_id", "request_id", "step_id",
        "phase", "operation", "exercise_status",
    )
    if any(key not in event for key in fields):
        raise ValueError("trusted event lacks collector correlation fields")
    trigger = {key: event[key] for key in fields}
    _identifier(trigger["session_id"], "session_id")
    _identifier(trigger["request_id"], "request_id")
    if (trigger["correlation_id"] != f"event-{trigger['sequence']:08d}"
            or type(trigger["sequence"]) is not int or trigger["sequence"] < 0
            or type(trigger["step_id"]) is not int or trigger["step_id"] < 0):
        raise ValueError("trusted event collector correlation differs")
    return trigger


def external_auth_payload(
        *, session_id: str, journal_id: str, journal_sha256: str,
        event_count: int, admission_sha256: str, workload_config_sha256: str,
        collector_manifest_sha256: str, collector_executable_sha256: str,
        collector_public_key_sha256: str) -> dict[str, Any]:
    _identifier(session_id, "session_id")
    _identifier(journal_id, "journal_id")
    for value, label in (
        (journal_sha256, "external journal"),
        (admission_sha256, "admission"),
        (workload_config_sha256, "workload config"),
        (collector_manifest_sha256, "collector manifest"),
        (collector_executable_sha256, "collector executable"),
        (collector_public_key_sha256, "collector public key"),
    ):
        _sha(value, label)
    if type(event_count) is not int or event_count < 1:
        raise ValueError("external journal event count must be positive")
    return {
        "schema_version": AUTH_PAYLOAD_SCHEMA,
        "session_id": session_id,
        "journal_id": journal_id,
        "external_journal_sha256": journal_sha256,
        "event_count": event_count,
        "admission_sha256": admission_sha256,
        "workload_config_sha256": workload_config_sha256,
        "collector_manifest_sha256": collector_manifest_sha256,
        "collector_executable_sha256": collector_executable_sha256,
        "collector_public_key_sha256": collector_public_key_sha256,
        "signature_algorithm": "Ed25519",
    }


def _verify_ed25519(payload: bytes, signature: bytes, public_key: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="p4-cc-ext-verify-") as temporary:
        payload_path = Path(temporary) / "payload.bin"
        signature_path = Path(temporary) / "signature.bin"
        payload_path.write_bytes(payload)
        signature_path.write_bytes(signature)
        try:
            result = subprocess.run(
                ["openssl", "pkeyutl", "-verify", "-pubin", "-inkey",
                 str(public_key), "-rawin", "-in", str(payload_path),
                 "-sigfile", str(signature_path)],
                capture_output=True, check=False, timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"external journal signature verification failed: {exc}") from exc
    if result.returncode != 0:
        raise ValueError("external journal signature verification failed")


def verify_external_journal_auth(
        auth_path: str | Path, journal_path: str | Path,
        collector_manifest_path: str | Path, collector_public_key_path: str | Path, *,
        expected_session_id: str, expected_journal_id: str,
        expected_admission_sha256: str, expected_workload_config_sha256: str,
        expected_collector_manifest_sha256: str,
        expected_collector_executable_sha256: str,
        expected_collector_public_key_sha256: str,
        event_validator: Callable[[Mapping[str, Any]], None] | None = None,
        production: bool = False) -> dict[str, Any]:
    """Verify exact external bytes, collector measurement, bindings and signature."""
    manifest = load_collector_manifest(
        collector_manifest_path, production=production,
        expected_public_key_path=collector_public_key_path,
    )
    if manifest["manifest_sha256"] != expected_collector_manifest_sha256:
        raise ValueError("collector manifest/admission binding mismatch")
    if manifest["executable_sha256"] != expected_collector_executable_sha256:
        raise ValueError("collector executable/admission binding mismatch")
    if manifest["public_key_sha256"] != expected_collector_public_key_sha256:
        raise ValueError("collector public key/admission binding mismatch")
    journal_path = Path(journal_path)
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(journal_path.read_text().splitlines(), start=1):
        if not line:
            raise ValueError(f"external journal row {number} is blank")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"external journal row {number} is invalid JSON") from exc
        if not isinstance(row, Mapping):
            raise ValueError("external journal row must be an object")
        if event_validator is not None:
            event_validator(row)
        if row.get("collector_id") != manifest["collector_id"]:
            raise ValueError("external event collector identity differs from manifest")
        if row.get("collector_measurement_sha256") != manifest["executable_sha256"]:
            raise ValueError("external event collector measurement differs from manifest")
        if row.get("collector_sequence") != number - 1:
            raise ValueError("external collector sequence is not contiguous")
        rows.append(dict(row))
    if not rows:
        raise ValueError("external journal is empty")
    try:
        record = json.loads(Path(auth_path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"external journal authentication cannot be read: {exc}") from exc
    if (not isinstance(record, Mapping)
            or set(record) != {"schema_version", "payload", "payload_sha256",
                               "signature_base64"}
            or record.get("schema_version") != AUTH_SCHEMA):
        raise ValueError("external journal authentication fields differ")
    expected = external_auth_payload(
        session_id=expected_session_id,
        journal_id=expected_journal_id,
        journal_sha256=file_sha256(journal_path),
        event_count=len(rows),
        admission_sha256=expected_admission_sha256,
        workload_config_sha256=expected_workload_config_sha256,
        collector_manifest_sha256=expected_collector_manifest_sha256,
        collector_executable_sha256=expected_collector_executable_sha256,
        collector_public_key_sha256=expected_collector_public_key_sha256,
    )
    if record.get("payload") != expected:
        raise ValueError("external journal authentication binding mismatch")
    encoded = canonical_bytes(expected)
    if record.get("payload_sha256") != hashlib.sha256(encoded).hexdigest():
        raise ValueError("external journal authentication payload digest mismatch")
    try:
        signature = base64.b64decode(record.get("signature_base64", ""), validate=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("external journal signature is not canonical base64") from exc
    _verify_ed25519(encoded, signature, manifest["resolved_public_key_path"])
    for key in ("_manifest_snapshot", "_executable_snapshot", "_public_key_snapshot"):
        require_unchanged(manifest[key])
    return dict(record)


class ExternalCollectorClient:
    """Synchronous process client; one process may sign several run journals."""

    def __init__(
            self, manifest_path: str | Path, argv: Sequence[str],
            public_key_path: str | Path, *, session_id: str,
            admission_sha256: str, workload_config_sha256: str,
            stderr_path: str | Path, production: bool,
            response_timeout_seconds: float = 15.0,
            event_validator: Callable[[Mapping[str, Any]], None] | None = None):
        self.manifest = load_collector_manifest(
            manifest_path, production=production, expected_argv=argv,
            expected_public_key_path=public_key_path,
        )
        _identifier(session_id, "session_id")
        _sha(admission_sha256, "admission")
        _sha(workload_config_sha256, "workload config")
        if response_timeout_seconds <= 0:
            raise ValueError("collector response timeout must be positive")
        self.session_id = session_id
        self.admission_sha256 = admission_sha256
        self.workload_config_sha256 = workload_config_sha256
        self.response_timeout_seconds = float(response_timeout_seconds)
        self.event_validator = event_validator
        self.stderr_path = Path(stderr_path)
        self.stderr_path.parent.mkdir(parents=True, exist_ok=True)
        self._stderr = self.stderr_path.open("xb")
        for key in ("_manifest_snapshot", "_executable_snapshot", "_public_key_snapshot"):
            require_unchanged(self.manifest[key])
        try:
            collector_env = {
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "LANG": "C",
                "LC_ALL": "C",
            }
            self.process = subprocess.Popen(
                list(argv), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=self._stderr, text=True, encoding="utf-8", bufsize=1,
                shell=False, close_fds=True, start_new_session=True,
                cwd=self.manifest["manifest_path"].parent, env=collector_env,
            )
        except Exception:
            self._stderr.close()
            raise
        try:
            for key in ("_manifest_snapshot", "_executable_snapshot",
                        "_public_key_snapshot"):
                require_unchanged(self.manifest[key])
        except Exception:
            self.terminate()
            raise
        if self.process.stdin is None or self.process.stdout is None:
            self.terminate()
            raise RuntimeError("collector process pipes are unavailable")
        self._journal: dict[str, Any] | None = None
        self._send({
            "protocol_schema": PROTOCOL_SCHEMA,
            "type": "start",
            "collector_id": self.manifest["collector_id"],
            "session_id": session_id,
            "admission_sha256": admission_sha256,
            "workload_config_sha256": workload_config_sha256,
            "collector_manifest_sha256": self.manifest["manifest_sha256"],
            "collector_executable_sha256": self.manifest["executable_sha256"],
            "collector_public_key_sha256": self.manifest["public_key_sha256"],
        })
        ready = self._receive()
        expected_ready = {
            "protocol_schema": PROTOCOL_SCHEMA,
            "type": "ready",
            "ready": True,
            "collector_id": self.manifest["collector_id"],
            "collector_manifest_sha256": self.manifest["manifest_sha256"],
            "collector_executable_sha256": self.manifest["executable_sha256"],
            "collector_public_key_sha256": self.manifest["public_key_sha256"],
            "session_id": session_id,
            "admission_sha256": admission_sha256,
            "workload_config_sha256": workload_config_sha256,
        }
        if ready != expected_ready:
            self.terminate()
            raise PermissionError("collector readiness handshake or binding differs")

    def _send(self, value: Mapping[str, Any]) -> None:
        if self.process.poll() is not None:
            raise RuntimeError("collector process exited before request")
        assert self.process.stdin is not None
        self.process.stdin.write(canonical_bytes(value).decode("utf-8") + "\n")
        self.process.stdin.flush()

    def _receive(self) -> dict[str, Any]:
        assert self.process.stdout is not None
        selector = selectors.DefaultSelector()
        try:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            if not selector.select(self.response_timeout_seconds):
                self.terminate()
                raise TimeoutError("collector process response timed out")
            line = self.process.stdout.readline()
        finally:
            selector.close()
        if not line:
            code = self.process.poll()
            raise RuntimeError(f"collector process closed its response stream (exit={code})")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError("collector process returned invalid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError("collector process response must be an object")
        return value

    def begin_journal(self, journal_id: str, journal_path: str | Path,
                      auth_path: str | Path) -> None:
        if self._journal is not None:
            raise RuntimeError("collector already has an active journal")
        _identifier(journal_id, "journal_id")
        journal_path = Path(journal_path)
        auth_path = Path(auth_path)
        if journal_path.exists() or auth_path.exists():
            raise FileExistsError("collector output already exists")
        journal_path.parent.mkdir(parents=True, exist_ok=True)
        self._send({
            "protocol_schema": PROTOCOL_SCHEMA, "type": "begin_journal",
            "journal_id": journal_id,
        })
        if self._receive() != {
                "protocol_schema": PROTOCOL_SCHEMA, "type": "journal_ready",
                "journal_id": journal_id}:
            raise PermissionError("collector did not acknowledge the exact journal")
        self._journal = {
            "journal_id": journal_id, "journal_path": journal_path,
            "auth_path": auth_path, "count": 0,
            "stream": journal_path.open("xb"),
        }

    def observe(self, trusted_event: Mapping[str, Any]) -> dict[str, Any]:
        if self._journal is None:
            raise RuntimeError("collector has no active journal")
        trigger = trigger_from_trusted_event(trusted_event)
        request = {
            "protocol_schema": PROTOCOL_SCHEMA,
            "type": "observe",
            "journal_id": self._journal["journal_id"],
            "trigger": trigger,
            "trigger_sha256": canonical_sha256(trigger),
        }
        self._send(request)
        response = self._receive()
        if (set(response) != {"protocol_schema", "type", "journal_id",
                              "trigger_sha256", "event"}
                or response.get("protocol_schema") != PROTOCOL_SCHEMA
                or response.get("type") != "observation"
                or response.get("journal_id") != self._journal["journal_id"]
                or response.get("trigger_sha256") != request["trigger_sha256"]
                or not isinstance(response.get("event"), Mapping)):
            raise ValueError("collector observation response differs")
        event = dict(response["event"])
        if self.event_validator is not None:
            self.event_validator(event)
        expected_identity = {key: trigger[key] for key in (
            "correlation_id", "sequence", "session_id", "request_id", "step_id",
            "phase", "operation",
        )}
        if any(event.get(key) != value for key, value in expected_identity.items()):
            raise ValueError("live collector event is not correlated to its trigger")
        if event.get("collector_id") != self.manifest["collector_id"]:
            raise ValueError("live collector identity differs from manifest")
        if event.get("collector_measurement_sha256") != self.manifest["executable_sha256"]:
            raise ValueError("live collector measurement differs from executable bytes")
        if event.get("collector_sequence") != self._journal["count"]:
            raise ValueError("live collector sequence differs")
        if event.get("trigger_sha256") != request["trigger_sha256"]:
            raise ValueError("live collector trigger binding differs")
        encoded = canonical_bytes(event)
        stream = self._journal["stream"]
        stream.write(encoded + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
        self._journal["count"] += 1
        return event

    def finish_journal(self) -> tuple[Path, Path]:
        if self._journal is None or self._journal["count"] < 1:
            raise RuntimeError("collector cannot finish an empty/inactive journal")
        journal = self._journal
        journal["stream"].close()
        journal_sha = file_sha256(journal["journal_path"])
        self._send({
            "protocol_schema": PROTOCOL_SCHEMA, "type": "finish_journal",
            "journal_id": journal["journal_id"],
            "external_journal_sha256": journal_sha,
            "event_count": journal["count"],
        })
        response = self._receive()
        if (set(response) != {"protocol_schema", "type", "journal_id", "auth"}
                or response.get("protocol_schema") != PROTOCOL_SCHEMA
                or response.get("type") != "journal_auth"
                or response.get("journal_id") != journal["journal_id"]
                or not isinstance(response.get("auth"), Mapping)):
            raise ValueError("collector final authentication response differs")
        auth_path = journal["auth_path"]
        write_json_exclusive(auth_path, response["auth"])
        os.chmod(journal["journal_path"], 0o600)
        os.chmod(auth_path, 0o600)
        self._journal = None
        verify_external_journal_auth(
            auth_path, journal["journal_path"], self.manifest["manifest_path"],
            self.manifest["resolved_public_key_path"],
            expected_session_id=self.session_id,
            expected_journal_id=journal["journal_id"],
            expected_admission_sha256=self.admission_sha256,
            expected_workload_config_sha256=self.workload_config_sha256,
            expected_collector_manifest_sha256=self.manifest["manifest_sha256"],
            expected_collector_executable_sha256=self.manifest["executable_sha256"],
            expected_collector_public_key_sha256=self.manifest["public_key_sha256"],
            event_validator=self.event_validator,
            production=not self.manifest["test_only"],
        )
        return journal["journal_path"], auth_path

    def abort_journal(self) -> None:
        if self._journal is not None:
            stream = self._journal.get("stream")
            if stream is not None and not stream.closed:
                stream.close()
        self._journal = None
        self.terminate()

    def close(self) -> None:
        if self._journal is not None:
            raise RuntimeError("cannot close collector with an unfinished journal")
        if self.process.poll() is None:
            self._send({"protocol_schema": PROTOCOL_SCHEMA, "type": "shutdown"})
            response = self._receive()
            if response != {"protocol_schema": PROTOCOL_SCHEMA, "type": "stopped"}:
                self.terminate()
                raise RuntimeError("collector shutdown acknowledgement differs")
            assert self.process.stdin is not None
            self.process.stdin.close()
            try:
                code = self.process.wait(timeout=self.response_timeout_seconds)
            except subprocess.TimeoutExpired as exc:
                self.terminate()
                raise TimeoutError("collector did not exit after shutdown") from exc
            if code != 0:
                raise RuntimeError(f"collector exited unsuccessfully: {code}")
        self._stderr.close()

    def terminate(self) -> None:
        journal = getattr(self, "_journal", None)
        if journal is not None:
            stream = journal.get("stream")
            if stream is not None and not stream.closed:
                stream.close()
            self._journal = None
        process = getattr(self, "process", None)
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        stderr = getattr(self, "_stderr", None)
        if stderr is not None and not stderr.closed:
            stderr.close()

    def __enter__(self) -> "ExternalCollectorClient":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is None:
            self.close()
        else:
            self.terminate()

    def __del__(self) -> None:
        # Last-resort process hygiene only; successful workloads close
        # explicitly and never rely on finalisation for evidence integrity.
        try:
            self.terminate()
        except Exception:
            pass
