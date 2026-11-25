#!/usr/bin/env python3
"""Recover and hash-check the 860 merged evidence references from execution hosts.

Requires reviewed staging trees under build/hf-release-2026-09-17.
Does not upload files, delete source data, or regenerate evidence.
"""
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "build/hf-release-2026-09-17"


def log(*args):
    print(time.strftime("%H:%M:%S"), *args, flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)


def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_dir():
        dst.rmdir()  # Empty placeholders from the original uploader.
    elif dst.exists():
        dst.unlink()
    os.link(src, dst)


def add_file(stage, src, rel):
    link(src, stage / rel)


def recover_binary_tier():
    """Read expected hashes from canonical merged evidence and original hosts."""
    import concurrent.futures
    merged = json.loads((ROOT / "build/merge-fix-v1/merged/validation.json").read_text())
    expected = dict(merged["artifacts"])
    for item in merged["cells"]:
        ref = item.get("defended_observations")
        if ref:
            if ref["path"] in expected and expected[ref["path"]] != ref["sha256"]:
                raise ValueError("Conflicting evidence hash: " + ref["path"])
            expected[ref["path"]] = ref["sha256"]
    for item in merged["defended_victims"]:
        ref = item["provenance"]
        expected[ref["path"]] = ref["sha256"]
    write(STAGE / "binary-tier-expected.json", json.dumps(expected, indent=2))
    dst = STAGE / "paper-a/binary-tier"
    groups = {}
    for rel, want in expected.items():
        local = dst / rel
        if local.is_file() and sha256(local) == want:
            continue
        if local.is_dir():
            local.rmdir()  # Only remove empty directories left by the old path bug.
        if rel.startswith("shard"):
            shard, tail = rel.split("/", 1)
            host = "toula" if int(shard[5:]) < 8 else "koula"
            target = dst / shard
        else:
            shard, tail, host, target = "shard8", rel, "koula", dst
        groups.setdefault((host, shard, str(target)), []).append(tail)
    def pull(item):
        (host, shard, target), paths = item
        suffix = "shared" if Path(target) == dst else shard
        listing = STAGE / ("recover-" + suffix + ".txt")
        listing.write_text("\n".join(paths) + "\n")
        Path(target).mkdir(parents=True, exist_ok=True)
        subprocess.run(["rsync", "-a", "--timeout=300", "--files-from=" + str(listing),
            f"{host}:llm-attacker-pilots/gpt2-wikitext2-v01/code/results/validation-heldout-v3-r3-{shard}/",
            target + "/"], check=True)
        log("recovered", host, suffix, len(paths))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(pull, groups.items()))
    missing, mismatch = [], []
    for rel, want in expected.items():
        p = dst / rel
        if not p.is_file(): missing.append(rel)
        elif sha256(p) != want: mismatch.append(rel)
    report = dict(expected=len(expected), missing=missing, mismatch=mismatch,
                  ok=not missing and not mismatch)
    write(STAGE / "binary-tier-verification.json", json.dumps(report, indent=2))
    if not report["ok"]:
        raise RuntimeError("Binary reference verification failed")
    for rel in expected:
        link(dst / rel, STAGE / "paper-b/binary-tier" / rel)
    for paper in ("paper-a", "paper-b"):
        add_file(STAGE / paper, STAGE / "binary-tier-verification.json", "BINARY-VERIFICATION.json")
    log("BINARY_VERIFIED", len(expected))


if __name__ == "__main__":
    # Existing staged paper trees must be reviewed before upload. This command
    # repairs their binary tier only, without rebuilding cards or live snapshots.
    recover_binary_tier()
