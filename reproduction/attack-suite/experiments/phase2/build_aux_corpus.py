#!/usr/bin/env python
"""Build a disjoint BiSR auxiliary corpus from wikitext-2 train.

Reuses the exact parsing/id conventions of experiments/prepare_pilot.py
(articles) and the exclusion doctrine of experiments/curate_validation.py
(pilot identity by text sha256 AND normalized title, plus near-duplicate
5-gram Jaccard >= 0.8 against pilot prefixes), and additionally excludes
every document id selected into the frozen validation corpus manifest
(any partition). Output is gated locally by
attacker.attacks.bisr_baseline.assert_aux_disjoint before it may ship.

Usage: .venv/bin/python build/phase2/build_aux_corpus.py [N_ARTICLES]
"""
import hashlib
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

from experiments.prepare_pilot import articles
from attacker.attacks.bisr_baseline import assert_aux_disjoint

REPO = Path(__file__).resolve().parents[2]
INPUTS = REPO / "results" / "pilot-gpt2-wikitext2" / "inputs"
MANIFEST = REPO / "experiments" / "validation_corpus_manifest.json"
OUT = REPO / "build" / "phase2" / "aux-corpus.jsonl"


def grams(tokens):
    return {tuple(tokens[i:i + 5]) for i in range(len(tokens) - 4)}


def title_key(title):
    import re
    return re.sub(r"\s+", " ", title).strip().casefold()


def main():
    n_want = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(INPUTS / "tokenizer",
                                              use_fast=True)

    pilot = [json.loads(line)
             for line in (INPUTS / "pilot-corpus.jsonl").read_text().splitlines()]
    excluded_text = {hashlib.sha256(r["text"].encode()).hexdigest()
                     for r in pilot}
    excluded_titles = {title_key(r.get("title", "")) for r in pilot}
    pilot_grams = [grams(tokenizer.encode(r["text"],
                                          add_special_tokens=False,
                                          truncation=True, max_length=32))
                   for r in pilot]

    manifest = json.loads(MANIFEST.read_text())
    frozen_ids = set()
    for ids in manifest.get("selected_ids", {}).values():
        frozen_ids.update(ids)

    kept, discarded = [], 0
    seen_text = set()
    for title, text in articles(pq.read_table(INPUTS / "train.parquet")["text"].to_pylist()):
        digest = hashlib.sha256(text.encode()).hexdigest()
        doc_id = "wikitext2-train-" + digest[:20]
        if digest in seen_text:
            discarded += 1
            continue
        if doc_id in frozen_ids or digest in excluded_text:
            discarded += 1
            continue
        if title_key(title) in excluded_titles:
            discarded += 1
            continue
        grams_ = grams(tokenizer.encode(text, add_special_tokens=False,
                                        truncation=True, max_length=32))
        if any(len(grams_ & g) / len(grams_ | g) >= 0.8 for g in pilot_grams):
            discarded += 1
            continue
        seen_text.add(digest)
        kept.append({"id": doc_id, "text": text})
        if len(kept) >= n_want:
            break

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for row in kept:
            f.write(json.dumps(row) + "\n")

    audit = assert_aux_disjoint(OUT, MANIFEST, INPUTS / "pilot-corpus.jsonl")
    audit["kept"] = len(kept)
    audit["discarded"] = discarded
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
