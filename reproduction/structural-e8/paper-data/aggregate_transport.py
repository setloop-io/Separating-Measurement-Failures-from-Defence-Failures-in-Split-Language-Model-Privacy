#!/usr/bin/env python3
"""Aggregate repeated direct/tunnel transport benchmark artifacts."""

import argparse
import hashlib
import json
import random
import re
import statistics
from pathlib import Path


PATTERN = re.compile(r"transport_(direct|tunnel)_(0|80)ms_rep([1-9][0-9]*)\.json$")
METRICS = ("avg_measured_tok_s", "avg_measured_rtt_ms",
           "avg_per_step_wall_ms", "avg_acceptance_rate")


def percentile(values, q):
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def summarize(values, seed):
    rng = random.Random(seed)
    boot = [statistics.fmean(rng.choices(values, k=len(values)))
            for _ in range(10_000)]
    return {"values": values, "mean": statistics.fmean(values),
            "sample_std": statistics.stdev(values),
            "bootstrap_mean_95ci": [percentile(boot, 0.025),
                                     percentile(boot, 0.975)]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("output")
    args = parser.parse_args()
    groups = {}
    inputs = []
    for path in sorted(Path(args.input_dir).glob("transport_*_rep*.json")):
        match = PATTERN.match(path.name)
        if not match:
            continue
        transport, rtt, rep = match.groups()
        data = json.loads(path.read_text())
        key = f"{transport}_{rtt}ms"
        groups.setdefault(key, {})[int(rep)] = data["summary"]
        inputs.append({"path": str(path),
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    expected = {f"{transport}_{rtt}ms" for transport in ("direct", "tunnel")
                for rtt in (0, 80)}
    if set(groups) != expected or any(set(reps) != {1, 2, 3}
                                      for reps in groups.values()):
        parser.error("expected repetitions 1,2,3 for direct/tunnel at 0/80 ms")

    cells = {}
    for cell_index, (key, reps) in enumerate(sorted(groups.items())):
        cells[key] = {}
        for mode_index, mode in enumerate(("sequential", "lookahead")):
            cells[key][mode] = {}
            for metric_index, metric in enumerate(METRICS):
                values = [reps[rep][mode][metric] for rep in (1, 2, 3)]
                cells[key][mode][metric] = summarize(
                    values, 20260807 + 100 * cell_index + 10 * mode_index + metric_index)
    result = {"schema": "dtraining.transport.aggregate.v1", "repetitions": 3,
              "ci_method": "10,000-replicate percentile bootstrap over run means",
              "inputs": inputs, "cells": cells}
    output = Path(args.output)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
