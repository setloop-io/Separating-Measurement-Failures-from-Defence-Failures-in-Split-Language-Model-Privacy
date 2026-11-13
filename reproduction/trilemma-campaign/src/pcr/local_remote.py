"""Lifecycle helper for real loopback R1 remote-service subprocesses."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .distributed_split import RemoteClient
from .r1_packets import prepare_remote_service_packet, verify_process_packet
from .utils import sha256_file


class LocalRemoteService:
    def __init__(self, root: Path, work_dir: Path, session_id: str,
                 q_bits: int, timeout: float = 30.0,
                 resume_checkpoint: Path | None = None,
                 read_timeout: float | None = None) -> None:
        self.root = root.resolve()
        self.work_dir = work_dir.resolve()
        self.session_id = session_id
        self.q_bits = q_bits
        self.timeout = timeout
        self.read_timeout = timeout if read_timeout is None else read_timeout
        if self.read_timeout <= 0:
            raise ValueError("remote read timeout must be positive")
        self.work_dir.mkdir(parents=True, exist_ok=False)
        self.remote_root = self.work_dir / "remote-root"
        prepared = self.root / "process-packets/remote-service"
        if prepared.is_dir():
            verify_process_packet(prepared, "remote-service")
            shutil.copytree(prepared, self.remote_root)
        else:
            prepare_remote_service_packet(self.root, self.remote_root)
        verify_process_packet(self.remote_root, "remote-service")
        self.resume_root = self.work_dir / "resume"
        self.resume_root.mkdir()
        self.resume_checkpoint_id = None
        self.resume_checkpoint_sha256 = None
        if resume_checkpoint is not None:
            if resume_checkpoint.is_symlink():
                raise ValueError("remote resume checkpoint is unavailable")
            source = resume_checkpoint.resolve()
            if not source.is_file():
                raise ValueError("remote resume checkpoint is unavailable")
            self.resume_checkpoint_id = source.name
            target = self.resume_root / self.resume_checkpoint_id
            shutil.copyfile(source, target)
            self.resume_checkpoint_sha256 = sha256_file(str(target))
        self.secret = os.urandom(32)
        self.secret_path = self.work_dir / "wire.secret"
        self.secret_path.write_bytes(self.secret)
        self.secret_path.chmod(0o600)
        self.ready_path = self.work_dir / "ready.json"
        self.output_dir = self.work_dir / "remote"
        self.process = self._start()
        self.port = self._wait_ready()

    def _start(self) -> subprocess.Popen[str]:
        command = [sys.executable, str(self.remote_root / "scripts/run_remote_service.py"),
                   "--secret-file", str(self.secret_path),
                   "--output-dir", str(self.output_dir),
                   "--ready-file", str(self.ready_path),
                   "--timeout", str(self.read_timeout),
                   "--resume-root", str(self.resume_root)]
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(self.remote_root / "src")
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        return subprocess.Popen(command, cwd=self.remote_root, env=environment,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def _wait_ready(self) -> int:
        deadline = time.monotonic() + self.timeout
        while not self.ready_path.is_file() and time.monotonic() < deadline:
            if self.process.poll() is not None:
                stdout, stderr = self.process.communicate()
                raise RuntimeError(f"remote service exited before ready: {stdout} {stderr}")
            time.sleep(0.01)
        if not self.ready_path.is_file():
            self.process.terminate()
            raise TimeoutError("remote service did not become ready")
        return int(json.loads(self.ready_path.read_text())["port"])

    def connect(self) -> RemoteClient:
        return RemoteClient.connect("127.0.0.1", self.port, self.secret,
                                    self.session_id, self.q_bits, self.timeout)

    def wait(self) -> None:
        try:
            stdout, stderr = self.process.communicate(timeout=self.timeout)
        except subprocess.TimeoutExpired as exc:
            self.terminate()
            raise TimeoutError("remote service did not exit after session completion") from exc
        finally:
            self.secret_path.unlink(missing_ok=True)
        if self.process.returncode != 0:
            raise RuntimeError(f"remote service failed: {stdout} {stderr}")

    def terminate(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=self.timeout)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=self.timeout)
        self.secret_path.unlink(missing_ok=True)

    def __enter__(self) -> "LocalRemoteService":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is not None:
            self.terminate()
            return
        self.wait()
