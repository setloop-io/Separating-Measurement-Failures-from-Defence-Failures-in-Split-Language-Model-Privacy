#!/usr/bin/env python3
"""Run one authenticated corpus-free R1 remote-surrogate session on loopback."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pcr.remote_service import serve_once  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--secret-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--resume-root", type=Path)
    args = parser.parse_args(argv)
    try:
        secret = args.secret_file.read_bytes()
        if len(secret) < 32:
            raise ValueError("secret file must contain at least 32 bytes")
        serve_once(args.host, args.port, secret, args.output_dir, args.ready_file,
                   args.timeout, args.resume_root)
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"remote service failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
