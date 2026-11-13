#!/usr/bin/env python3
"""Copy a pinned public corpus locally after verifying its digest."""
import argparse
from pathlib import Path
from p4d.experiment import ROOT, CORPUS_SHA256, digest

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--path", type=Path, default=Path("data/raw/tinyshakespeare.txt"))
    parser.add_argument("--sha256", default=CORPUS_SHA256)
    args = parser.parse_args()
    if digest(args.source) != args.sha256:
        raise SystemExit("Source corpus does not match the pinned digest")
    dst = ROOT / args.path
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(args.source.read_bytes())
    print(f"Prepared {dst.name}: {args.sha256}")
