#!/usr/bin/env python3
"""Fetch and assemble the Paper 4 public-domain text corpus (engineering tooling).

Fetches Project Gutenberg UTF-8 plain-text files, strips Gutenberg header/footer
boilerplate, normalizes line endings, records per-document digests and identities,
assigns documents to regions at AUTHOR granularity (all works by one author stay
in one region), and writes the concatenated corpus plus a committed manifest.

This script only prepares bytes and metadata. It is not a scientific runner and
produces no evidence. The assembled corpus digest must be pinned in any config
that references it. See docs/CALIBRATION_TEXT_PROTOCOL.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) research-corpus-fetch"}

# Candidate Project Gutenberg IDs (public-domain English prose; no Shakespeare).
CANDIDATE_IDS = [
    98, 1400, 766, 730, 1023, 580, 46,           # Dickens
    1342, 158, 141, 105, 121, 161,               # Austen
    74, 76, 1837, 86,                            # Twain
    1260, 768, 767, 969,                         # Brontes
    122, 153, 107, 110,                          # Hardy
    2701,                                        # Melville
    33, 77,                                      # Hawthorne
    145, 550,                                    # Eliot
    1661, 244, 2852, 2097,                       # Conan Doyle
    174,                                         # Wilde
    345,                                         # Stoker
    84,                                          # Shelley
    43, 120,                                     # Stevenson
    35, 36, 5230,                                # Wells
    164, 103, 83,                                # Verne
    11, 12,                                      # Carroll
    829,                                         # Swift
    521, 370,                                    # Defoe
    28054, 2554, 2638,                           # Dostoevsky
    2600, 1399,                                  # Tolstoy
    135,                                         # Hugo
    1184, 1257,                                  # Dumas
    514,                                         # Alcott
    45,                                          # Montgomery
    215, 910,                                    # London
    236, 2226,                                   # Kipling
    599,                                         # Thackeray
    209, 2833,                                   # James
    113, 479,                                    # Burnett
    3155, 2166,                                  # Haggard
    55,                                          # Baum
    16,                                          # Barrie
    289,                                         # Grahame
    219,                                         # Conrad
]

REGION_TARGETS = {"train": 0.60, "calibration": 0.15, "probe_fit": 0.125, "evaluation": 0.125}
ASSEMBLY_SEED = 20260917
MIN_DOC_BYTES = 100_000

START_RE = re.compile(r"^\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*$", re.M)
END_RE = re.compile(r"^\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*$", re.M)
TITLE_RE = re.compile(r"^Title:\s*(.+)$", re.M)
AUTHOR_RE = re.compile(r"^Author:\s*(.+)$", re.M)


def fetch_url(pg_id: int) -> bytes | None:
    urls = [
        f"https://www.gutenberg.org/cache/epub/{pg_id}/pg{pg_id}.txt",
        f"https://www.gutenberg.org/files/{pg_id}/{pg_id}.txt",
    ]
    for url in urls:
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            if len(data) > 10_000:
                return data
        except Exception as exc:  # noqa: BLE001 - record and try next mirror
            print(f"    {url} failed: {exc}")
        time.sleep(1.0)
    return None


def clean(pg_id: int, raw: bytes) -> tuple[str, str, str] | None:
    text = raw.decode("utf-8-sig", errors="strict").replace("\r\n", "\n").replace("\r", "\n")
    start = START_RE.search(text)
    end = END_RE.search(text)
    if not start or not end or end.start() <= start.end():
        print(f"    pg{pg_id}: missing START/END markers, skipped")
        return None
    header = text[: start.start()]
    body = text[start.end():end.start()].strip("\n") + "\n"
    title_m = TITLE_RE.search(header)
    author_m = AUTHOR_RE.search(header)
    title = title_m.group(1).strip() if title_m else f"pg{pg_id}"
    author = author_m.group(1).strip() if author_m else "Unknown"
    if len(body.encode("utf-8")) < MIN_DOC_BYTES:
        print(f"    pg{pg_id}: only {len(body.encode('utf-8'))} bytes after cleaning, skipped")
        return None
    return body, title, author


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def assign_regions(docs: list[dict]) -> None:
    """Author-level greedy assignment; deterministic under ASSEMBLY_SEED."""
    by_author: dict[str, list[dict]] = {}
    for d in docs:
        by_author.setdefault(d["author"], []).append(d)
    groups = [
        {"author": a, "bytes": sum(d["bytes"] for d in ds), "docs": ds}
        for a, ds in by_author.items()
    ]
    rng = random.Random(ASSEMBLY_SEED)
    rng.shuffle(groups)
    groups.sort(key=lambda g: -g["bytes"])  # largest first, stable after shuffle
    total = sum(g["bytes"] for g in groups)
    filled = {r: 0 for r in REGION_TARGETS}
    for g in groups:
        deficits = {r: REGION_TARGETS[r] * total - filled[r] for r in REGION_TARGETS}
        region = max(deficits, key=deficits.get)
        for d in g["docs"]:
            d["region"] = region
        filled[region] += g["bytes"]


def overlap_check(corpus: bytes, reference: bytes, sample: int = 512, ngram: int = 32) -> dict:
    """Fraction of sampled reference n-grams found verbatim in the new corpus."""
    rng = random.Random(ASSEMBLY_SEED)
    if len(reference) < ngram:
        return {"sampled": 0, "found": 0, "fraction": 0.0}
    starts = rng.sample(range(0, len(reference) - ngram), min(sample, len(reference) - ngram))
    found = sum(1 for s in starts if reference[s:s + ngram] in corpus)
    return {
        "sampled": len(starts),
        "ngram_bytes": ngram,
        "found": found,
        "fraction": found / len(starts),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/raw", help="corpus output directory (gitignored)")
    ap.add_argument("--manifest", default="data/gutenberg_p4.manifest.json", help="committed manifest path")
    ap.add_argument("--reference", default="data/raw/tinyshakespeare.txt", help="historical corpus for overlap check")
    ap.add_argument("--src-cache", default="data/raw/gutenberg_src")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    src_dir = Path(args.src_cache)
    out_dir.mkdir(parents=True, exist_ok=True)
    src_dir.mkdir(parents=True, exist_ok=True)

    docs: list[dict] = []
    for pg_id in CANDIDATE_IDS:
        print(f"pg{pg_id}:")
        raw = fetch_url(pg_id)
        if raw is None:
            print("    download failed, skipped")
            continue
        try:
            cleaned = clean(pg_id, raw)
        except UnicodeDecodeError as exc:
            print(f"    decode failed: {exc}, skipped")
            continue
        if cleaned is None:
            continue
        body, title, author = cleaned
        blob = body.encode("utf-8")
        digest = sha256_bytes(blob)
        (src_dir / f"pg{pg_id}.txt").write_bytes(blob)
        docs.append({
            "pg_id": pg_id,
            "title": title,
            "author": author,
            "bytes": len(blob),
            "sha256": digest,
            "source_urls": [
                f"https://www.gutenberg.org/cache/epub/{pg_id}/pg{pg_id}.txt",
                f"https://www.gutenberg.org/files/{pg_id}/{pg_id}.txt",
            ],
        })
        print(f"    {title!r} by {author}: {len(blob)} bytes")
        time.sleep(0.5)

    if not docs:
        print("no documents fetched", file=sys.stderr)
        return 1

    assign_regions(docs)
    region_order = ["train", "calibration", "probe_fit", "evaluation"]
    docs.sort(key=lambda d: (region_order.index(d["region"]), d["author"], d["pg_id"]))

    corpus_path = out_dir / "gutenberg_p4.txt"
    offset = 0
    with corpus_path.open("wb") as fh:
        for d in docs:
            blob = (src_dir / f"pg{d['pg_id']}.txt").read_bytes()
            d["offset"] = offset
            fh.write(blob)
            offset += len(blob)
    corpus_bytes = corpus_path.read_bytes()
    corpus_digest = sha256_bytes(corpus_bytes)

    reference = Path(args.reference).read_bytes() if Path(args.reference).exists() else b""
    overlap = overlap_check(corpus_bytes, reference) if reference else None

    region_stats = {}
    for r in region_order:
        region_docs = [d for d in docs if d["region"] == r]
        region_stats[r] = {
            "bytes": sum(d["bytes"] for d in region_docs),
            "documents": len(region_docs),
            "authors": sorted({d["author"] for d in region_docs}),
        }

    manifest = {
        "corpus_id": "gutenberg_p4",
        "assembled_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "assembly_seed": ASSEMBLY_SEED,
        "preprocessing": {
            "encoding": "utf-8-sig decode, CRLF/CR -> LF",
            "boilerplate": "Gutenberg START/END marker lines and everything outside them removed",
            "min_document_bytes": MIN_DOC_BYTES,
            "normalization": "none beyond the above",
        },
        "region_policy": {
            "granularity": "author-level greedy assignment toward byte targets",
            "targets": REGION_TARGETS,
            "realized": {r: region_stats[r]["bytes"] for r in region_order},
        },
        "corpus_sha256": corpus_digest,
        "corpus_bytes": len(corpus_bytes),
        "documents": docs,
        "region_stats": region_stats,
        "tinyshakespeare_overlap": overlap,
        "license_note": (
            "All listed works are public domain in the USA per Project Gutenberg "
            "license headers; Gutenberg boilerplate was removed. Verify per-document "
            "headers at the recorded source URLs."
        ),
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print("\n=== summary ===")
    print(f"documents: {len(docs)}  corpus bytes: {len(corpus_bytes)}")
    print(f"corpus sha256: {corpus_digest}")
    for r in region_order:
        s = region_stats[r]
        print(f"  {r:12s} {s['bytes']:>10,} bytes  {s['documents']:>3d} docs  {len(s['authors'])} authors")
    if overlap:
        print(f"tiny-Shakespeare overlap: {overlap['found']}/{overlap['sampled']} sampled "
              f"{overlap['ngram_bytes']}-byte n-grams found ({overlap['fraction']:.4f})")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
