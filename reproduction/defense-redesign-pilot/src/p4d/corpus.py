"""Pinned Paper 4 public-domain corpus access with per-document integrity checks."""
import hashlib
import json

import numpy as np
import torch
from pcr.data import load_corpus

from .experiment import ROOT, digest
from .warmup_scenarios import panel_record

REGIONS = ("train", "calibration", "probe_fit", "evaluation")


def load_pinned_corpus(data_cfg):
    path = ROOT / data_cfg["path"]
    if digest(path) != data_cfg["sha256"]:
        raise ValueError("Pinned corpus digest mismatch")
    return load_corpus(path)


def load_document_index(data_cfg):
    """Region -> verified document list; any mismatch is an error, never a substitute."""
    index = json.loads((ROOT / data_cfg["documents"]).read_text())
    if index["corpus_sha256"] != data_cfg["sha256"]:
        raise ValueError("Document index does not match the pinned corpus digest")
    raw = (ROOT / data_cfg["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != data_cfg["sha256"]:
        raise ValueError("Pinned corpus digest mismatch")
    if index.get("corpus_bytes", len(raw)) != len(raw):
        raise ValueError("Document index corpus length mismatch")
    regions = {}
    for doc in index["documents"]:
        if doc["region"] not in REGIONS:
            raise ValueError(f"Unknown document region: {doc['region']}")
        chunk = raw[doc["offset"]:doc["offset"] + doc["bytes"]]
        if len(chunk) != doc["bytes"] or hashlib.sha256(chunk).hexdigest() != doc["sha256"]:
            raise ValueError(f"Document {doc.get('pg_id')} content mismatch")
        regions.setdefault(doc["region"], []).append(
            {"pg_id": doc["pg_id"], "title": doc["title"], "author": doc["author"],
             "offset": doc["offset"], "bytes": doc["bytes"]})
    return {region: sorted(docs, key=lambda d: d["offset"]) for region, docs in regions.items()}


def contiguous_region(corpus, documents):
    """Region bytes only when the document assignment is one contiguous span."""
    ordered = sorted(documents, key=lambda d: d["offset"])
    if not ordered:
        raise ValueError("Empty document region")
    for a, b in zip(ordered, ordered[1:]):
        if a["offset"] + a["bytes"] != b["offset"]:
            raise ValueError("Document region is not a contiguous span")
    return corpus[ordered[0]["offset"]:ordered[-1]["offset"] + ordered[-1]["bytes"]]


def document_text_panel(corpus, documents, count, sequence_length, seed, region=None):
    """Nonoverlapping within-document windows include the final next-byte target."""
    stride = sequence_length + 1
    if not documents or count < 1 or sequence_length < 1:
        raise ValueError("Invalid panel documents, size, or sequence length")
    grids, ids = [], []
    for doc in documents:
        if doc["offset"] < 0 or doc["offset"] + doc["bytes"] > len(corpus):
            raise ValueError("Document range outside the corpus")
        starts = np.arange(doc["offset"], doc["offset"] + doc["bytes"] - stride + 1,
                           stride, dtype=np.int64)
        grids.append(starts)
        ids.extend([doc["pg_id"]] * len(starts))
    grid = np.concatenate(grids)
    if count > len(grid):
        raise ValueError("Insufficient disjoint windows in panel region")
    chosen = np.random.default_rng(seed).permutation(len(grid))[:count]
    offsets = torch.tensor(grid[chosen], dtype=torch.int64)
    windows = corpus[offsets[:, None] + torch.arange(stride)].long()
    return {"offsets": offsets, "tokens": windows[:, :-1].contiguous(),
            "next_tokens": windows[:, 1:].contiguous(),
            "document_ids": [ids[i] for i in chosen], "region": region}


def document_panel_record(panel):
    """panel_record plus per-window document identity."""
    return {**panel_record(panel), "document_ids": list(panel["document_ids"])}
