#!/usr/bin/env python3
"""Generate MANIFEST.sha256 for a staged release tree.

Usage: .venv/bin/python experiments/release_manifest.py <staged-repo-dir>

Writes <staged-repo-dir>/MANIFEST.sha256 with lines "<sha256>  <relative path>"
for every file in the tree except MANIFEST.sha256 itself. Paths are sorted.
"""
import hashlib
import sys
from pathlib import Path

CHUNK = 1 << 20


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> int:
    root = Path(sys.argv[1])
    files = sorted(
        q for q in root.rglob("*")
        if q.is_file() and q.name != "MANIFEST.sha256"
    )
    lines = []
    total = 0
    for q in files:
        digest = sha256_file(q)
        total += q.stat().st_size
        lines.append(f"{digest}  {q.relative_to(root).as_posix()}")
    (root / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")
    print(f"{root}: files={len(files)} bytes={total} GiB={total/2**30:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
