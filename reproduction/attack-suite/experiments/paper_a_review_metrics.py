"""Exploratory, CPU-only review analyses of retained Paper A predictions.

No model is loaded, no attack is rerun, and source evidence is never modified.
Run from the repository root with .venv/bin/python experiments/paper_a_review_metrics.py.
The GPT-2 tokenizer must already exist in the local Hugging Face cache.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import itertools
import json
from pathlib import Path
import platform
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
MAIN_ARMS = ("forward_k32", "joint_true_k32")
REPLICA_ARMS = ("sipit_steps128", "sipit_steps128@replica-median")
SPAN_ARMS = ("forward_k8", "forward_k16", "forward_k32", "sipit_steps32",
             "sipit_steps64", "sipit_steps128")
KINDS = ("name", "number", "identifier")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"empty table: {path}")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_sources(results: Path):
    """Hash/check source bindings and align selected prediction cells by document ID."""
    sources, main, replica, spans, seen = [], {}, {}, defaultdict(dict), set()
    docs, protocol = None, None
    sys.path.insert(0, str(ROOT))
    from attacker.reconstruction_metrics import reconstruction_metrics
    from attacker.evidence import verify_evidence

    def record(path, role, expected=None):
        actual = digest(path)
        if expected is not None and actual != expected:
            raise ValueError(f"SHA256 mismatch: {path}")
        sources.append({"path": str(path.relative_to(ROOT)), "role": role,
                        "sha256": actual, "recorded_sha256": expected,
                        "bytes": path.stat().st_size})
        return actual

    shards = sorted(results.glob("validation-heldout-v3-r3-shard*"))
    if not shards:
        raise ValueError("no retained campaign shards")
    for shard in shards:
        report = json.loads((shard / "validation.json").read_text())
        evidence_path = shard / "validation.json.evidence.json"
        ok, problems = verify_evidence(str(evidence_path))
        if not ok:
            raise ValueError(f"source evidence chain failed: {shard}: {problems}")
        evidence = json.loads(evidence_path.read_text())
        record(evidence_path, "verified_source_evidence_manifest")
        journal_path = shard / "validation.json.evidence.jsonl"
        record(journal_path, "verified_hash_chained_measurement_journal")
        journal = {}
        for line in journal_path.read_text().splitlines():
            event = json.loads(line)
            if event["kind"] == "cell_complete":
                payload = event["payload"]
                if payload["cell_id"] in journal:
                    raise ValueError(f"duplicate journal cell: {shard}")
                journal[payload["cell_id"]] = payload
        record(shard / "validation.json", "journal_bound_retained_cell_index", evidence["artifact"]["sha256"])
        if report["status"] != "COMPLETE":
            raise ValueError(f"incomplete shard: {shard}")
        p = json.loads((shard / "protocol-source.json").read_text())
        record(shard / "protocol-source.json", "frozen_source_protocol", report["protocol_sha256"])
        record(shard / "protocol-effective.json", "effective_shard_protocol",
               report["effective_protocol_sha256"])
        effective = json.loads((shard / "protocol-effective.json").read_text())
        if effective != report["protocol"]:
            raise ValueError(f"effective protocol differs from retained report: {shard}")
        record(shard / "input.json", "frozen_input", report["input_sha256"])
        if report["input_sha256"] != p["frozen_inputs"]["tokenized_sha256"]:
            raise ValueError(f"input not bound to frozen protocol: {shard}")
        data = json.loads((shard / "input.json").read_text())
        if docs is None:
            docs, protocol = data["evaluation"], p
        elif docs != data["evaluation"] or p != protocol:
            raise ValueError("source shards disagree on evaluation inputs or source protocol")
        ids = [d["id"] for d in docs]
        if len(ids) != 256 or len(set(ids)) != 256:
            raise ValueError("expected 256 distinct evaluation documents")
        truth = np.asarray([d["tokens"] for d in docs], dtype=np.int64)
        if truth.shape != (256, 32):
            raise ValueError("expected exactly 32 retained tokens per document")
        for cell in report["cells"]:
            if cell["mode"] != "training":
                continue
            arm = cell["arm"]
            base = arm.removesuffix("@secret-mixup")
            is_span = base in SPAN_ARMS and arm in (base, base + "@secret-mixup")
            is_main = cell["depth"] == 6 and cell["knowledge"] == "public" and arm in MAIN_ARMS
            is_replica = cell["depth"] == 3 and cell["knowledge"] == "public" and arm in REPLICA_ARMS
            if not (is_span or is_main or is_replica):
                continue
            key = (cell["depth"], cell["knowledge"], arm,
                   cell["victim_seed"], cell["attack_seed"])
            if key in seen:
                raise ValueError(f"duplicate cell: {key}")
            seen.add(key)
            path = shard / cell["prediction_file"]
            if path.parent != shard:
                raise ValueError("prediction must be directly inside its retained shard")
            logged = journal.get(cell["cell_id"])
            fields = ("prediction_file", "prediction_sha256", "arm", "mode", "depth", "knowledge",
                      "victim_seed", "attack_seed", "cell_id", "execution_id")
            if logged is None or any(logged[f] != cell[f] for f in fields):
                raise ValueError(f"prediction/index not bound to measurement journal: {path}")
            record(path, "retained_prediction", logged["prediction_sha256"])
            prediction = json.loads(path.read_text())
            if prediction["document_ids"] != ids:
                raise ValueError(f"unaligned prediction IDs: {path}")
            tokens = np.asarray(prediction["predictions"], dtype=np.int64)
            if tokens.shape != truth.shape or np.any(tokens < 0):
                raise ValueError(f"invalid prediction shape/IDs: {path}")
            rescored = [{"document_id": doc["id"], **reconstruction_metrics(
                doc["tokens"], pred, doc.get("annotations", []))}
                for doc, pred in zip(docs, prediction["predictions"], strict=True)]
            if rescored != cell["metrics"]:
                raise ValueError(f"retained metrics do not reproduce: {path}")
            correct = tokens == truth
            if is_main:
                main[(arm, cell["victim_seed"], cell["attack_seed"])] = tokens
            if is_replica:
                replica[(arm, cell["victim_seed"], cell["attack_seed"])] = tokens
            if is_span:
                recovered = [bool(correct[i, a["start"]:a["end"]].all())
                             for i, d in enumerate(docs) for a in d["annotations"]]
                spans[key[:3]][key[3:]] = np.asarray(recovered, dtype=bool)
        print(f"Checked {shard.name}; {len(seen)} selected prediction files", flush=True)
    assert docs is not None and protocol is not None
    runs = list(itertools.product(protocol["victim_seeds"], protocol["attack_seeds"]))
    expected = {(arm, v, a) for arm, (v, a) in itertools.product(MAIN_ARMS, runs)}
    if set(main) != expected:
        raise ValueError("main paired prediction grid is incomplete")
    if set(replica) != {(arm, v, a) for arm, (v, a) in itertools.product(REPLICA_ARMS, runs)}:
        raise ValueError("selected replica-median paired prediction grid is incomplete")
    expected_span = set(itertools.product((3, 6), ("public", "exact"),
                        [a + suffix for suffix in ("", "@secret-mixup") for a in SPAN_ARMS]))
    if set(spans) != expected_span or any(set(cells) != set(runs) for cells in spans.values()):
        raise ValueError("span condition grid is incomplete")
    annotations = ROOT / protocol["frozen_inputs"]["name_annotations"]
    record(annotations, "pre_outcome_name_annotations",
           protocol["frozen_inputs"]["name_annotations_sha256"])
    arrays = np.asarray([[main[arm, v, a] for a in protocol["attack_seeds"]]
                         for arm in MAIN_ARMS for v in protocol["victim_seeds"]])
    arrays = arrays.reshape(2, len(protocol["victim_seeds"]), len(protocol["attack_seeds"]), 256, 32)
    replica_arrays = np.asarray([[replica[arm, v, a] for a in protocol["attack_seeds"]]
                                 for arm in REPLICA_ARMS for v in protocol["victim_seeds"]])
    replica_arrays = replica_arrays.reshape(2, 10, 2, 256, 32)
    return docs, protocol, arrays, replica_arrays, spans, sources


def load_tokenizer(path: Path | None):
    """Only a cached pinned tokenizer is allowed; model weights are never opened."""
    import transformers
    from transformers import AutoTokenizer
    if path is None:
        path = Path.home() / ".cache/huggingface/hub/models--openai-community--gpt2/snapshots" / REVISION
    if path.name != REVISION:
        raise ValueError("tokenizer directory must identify the pinned GPT-2 revision")
    required = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
    if any(not (path / f).is_file() for f in required):
        raise FileNotFoundError("pinned GPT-2 tokenizer is not locally cached; no download attempted")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    hashes = {f: digest(path / f) for f in required}
    return tokenizer, {"model": "openai-community/gpt2", "revision": REVISION,
                       "files_sha256": hashes, "transformers_version": transformers.__version__,
                       "local_files_only": True, "clean_up_tokenization_spaces": False}


def crossed_bootstrap(cubes, *, replicates, seed, batch_size):
    """Paired resampling of victim and document axes; attack seeds averaged first.

    Each replicate uses the original analysis's RNG order (victims, then documents).
    Multiplicity weights avoid allocating replicates x victims x documents x outcomes.
    """
    names = list(cubes)
    matrices = np.stack([cubes[n].mean(axis=2) for n in names])  # metric, arm, victim, doc
    if matrices.shape[1:] != (2, 10, 256):
        raise ValueError(f"invalid paired bootstrap grid: {matrices.shape}")
    rng = np.random.default_rng(seed)
    draws = np.empty((replicates, len(names), 2))
    for start in range(0, replicates, batch_size):
        size = min(batch_size, replicates - start)
        victim_weights = np.empty((size, 10))
        doc_weights = np.empty((size, 256))
        for j in range(size):
            victim_weights[j] = np.bincount(rng.integers(10, size=10), minlength=10) / 10
            doc_weights[j] = np.bincount(rng.integers(256, size=256), minlength=256) / 256
        by_doc = np.einsum("bv,mavd->bmad", victim_weights, matrices, optimize=True)
        draws[start:start+size] = np.einsum("bmad,bd->bma", by_doc, doc_weights, optimize=True)
    summaries = {}
    for i, name in enumerate(names):
        values = cubes[name].mean(axis=(1, 2, 3))
        difference = draws[:, i, 1] - draws[:, i, 0]
        if np.any(draws[:, i, 0] <= 0):
            raise ValueError(f"undefined bootstrap ratio for {name}")
        ratio = draws[:, i, 1] / draws[:, i, 0]
        summaries[name] = {
            "activation_percent": float(100*values[0]), "joint_percent": float(100*values[1]),
            "difference_pp": float(100*(values[1]-values[0])),
            "difference_ci95_pp": (100*np.quantile(difference, [.025, .975])).tolist(),
            "ratio": float(values[1]/values[0]),
            "ratio_ci95": np.quantile(ratio, [.025, .975]).tolist(),
            "activation_ci95_percent": (100*np.quantile(draws[:, i, 0], [.025, .975])).tolist(),
            "joint_ci95_percent": (100*np.quantile(draws[:, i, 1], [.025, .975])).tolist(),
        }
    return summaries, names, draws


def span_tables(docs, spans):
    annotations = [{"document_id": d["id"], "kind": a["kind"], "start": a["start"],
                    "end": a["end"], "length_tokens": a["end"]-a["start"]}
                   for d in docs for a in d["annotations"]]
    distributions = []
    for kind in KINDS:
        selected = [a for a in annotations if a["kind"] == kind]
        lengths = [a["length_tokens"] for a in selected]
        distributions.append({"kind": kind, "distinct_documents": len({a["document_id"] for a in selected}),
            "distinct_spans": len(selected), "mean_length_tokens": float(np.mean(lengths)),
            "median_length_tokens": float(np.median(lengths)), "min_length_tokens": min(lengths),
            "max_length_tokens": max(lengths), "length_histogram": dict(sorted(Counter(lengths).items()))})
    rows = []
    for (depth, copy, arm), cells in sorted(spans.items()):
        recovered = np.asarray([cells[k] for k in sorted(cells)])
        for kind in KINDS:
            for length in [None] + sorted({a["length_tokens"] for a in annotations if a["kind"] == kind}):
                mask = np.asarray([a["kind"] == kind and (length is None or a["length_tokens"] == length)
                                   for a in annotations])
                rows.append({"depth": depth, "copy": copy, "arm": arm.removesuffix("@secret-mixup"),
                    "defence": "secret_mixup" if "@" in arm else "none", "kind": kind,
                    "length_tokens": "all" if length is None else length,
                    "distinct_spans": int(mask.sum()),
                    "distinct_documents": len({a["document_id"] for a, selected in zip(annotations, mask) if selected}),
                    "evaluations_per_span": len(cells), "recovery_percent": float(100*recovered[:, mask].mean())})
    return annotations, distributions, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/paper-a-review-20261020/metrics")
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=100000)
    parser.add_argument("--bootstrap-seed", type=int, default=93281)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.bootstrap_replicates < 100000 or args.batch_size < 1:
        raise ValueError("at least 100000 bootstrap replicates and positive batch size required")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("output must be new or empty; never overwrite retained results")
    docs, protocol, predictions, replica_predictions, spans, sources = load_sources(args.results)
    tokenizer, tokenizer_provenance = load_tokenizer(args.tokenizer)
    truth = np.asarray([d["tokens"] for d in docs])
    correct = predictions == truth
    decoded_truth = tokenizer.batch_decode(truth.tolist(), clean_up_tokenization_spaces=False,
                                           skip_special_tokens=False)
    decoded = np.asarray(tokenizer.batch_decode(predictions.reshape(-1, 32).tolist(),
        clean_up_tokenization_spaces=False, skip_special_tokens=False), dtype=object).reshape(2, 10, 2, 256)
    cubes = {"token": correct.mean(axis=-1), "document_token_id": correct.all(axis=-1)}
    cubes["document_decoded_exact"] = decoded == np.asarray(decoded_truth, dtype=object)
    for label, normalize in [("document_leading_space_normalized", lambda s: s.lstrip(" ")),
                             ("document_whitespace_normalized", lambda s: re.sub(r"\s+", " ", s).strip())]:
        truths = np.asarray([normalize(s) for s in decoded_truth], dtype=object)
        values = np.asarray([normalize(s) for s in decoded.reshape(-1)], dtype=object).reshape(decoded.shape)
        cubes[label] = values == truths
    for length in (8, 16, 24, 32):
        cubes[f"prefix_{length}_token_id"] = correct[..., :length].all(axis=-1)
    summary, bootstrap_names, bootstrap_draws = crossed_bootstrap(cubes,
        replicates=args.bootstrap_replicates, seed=args.bootstrap_seed, batch_size=args.batch_size)
    replica_summary, _, replica_draws = crossed_bootstrap({"replica": (replica_predictions == truth).all(axis=-1)},
        replicates=args.bootstrap_replicates, seed=args.bootstrap_seed, batch_size=args.batch_size)
    replica_stat = replica_summary["replica"]
    replica_result = {"depth": 3, "copy": "public", "attack": "SipIt", "iterations_per_token": 128,
        "undefended_percent": replica_stat["activation_percent"],
        "replica_median_percent": replica_stat["joint_percent"],
        "difference_defended_minus_undefended_pp": replica_stat["difference_pp"],
        "difference_ci95_pp": replica_stat["difference_ci95_pp"],
        "drop_pp": -replica_stat["difference_pp"],
        "drop_ci95_pp": [-replica_stat["difference_ci95_pp"][1], -replica_stat["difference_ci95_pp"][0]],
        "ratio_defended_over_undefended": replica_stat["ratio"], "ratio_ci95": replica_stat["ratio_ci95"],
        "selection": "Largest observed whole-document drop selected from the original replica-median conditions; pointwise exploratory interval, not a simultaneous or selection-adjusted interval."}
    expected_rounding = {"token": (94.20, 97.38), "document_token_id": (13.71, 37.77)}
    for label, expected in expected_rounding.items():
        got = tuple(round(summary[label][k], 2) for k in ("activation_percent", "joint_percent"))
        if got != expected:
            raise ValueError(f"headline rates do not reproduce: {label}: {got}")
    act, joint = correct
    transitions = {"both_correct": int((act & joint).sum()),
                   "corrected_activation_error": int((~act & joint).sum()),
                   "introduced_error": int((act & ~joint).sum()),
                   "both_wrong": int((~act & ~joint).sum()), "total_token_evaluations": int(act.size)}
    act_errors = transitions["corrected_activation_error"] + transitions["both_wrong"]
    transitions.update({"activation_errors": act_errors,
        "gross_correction_percent_of_activation_errors": 100*transitions["corrected_activation_error"]/act_errors,
        "introduced_errors_pp": 100*transitions["introduced_error"]/act.size,
        "gross_corrections_pp": 100*transitions["corrected_activation_error"]/act.size,
        "net_error_reduction_percent": 100*(transitions["corrected_activation_error"]-transitions["introduced_error"])/act_errors})
    positional, independence = [], []
    for i, arm in enumerate(MAIN_ARMS):
        p = correct[i].mean(axis=(0, 1, 2))
        positional.extend({"arm": arm, "position_one_based": j+1, "token_recovery_percent": 100*float(x)}
                          for j, x in enumerate(p))
        independence.append({"arm": arm, "observed_document_percent": float(100*correct[i].all(axis=-1).mean()),
            "homogeneous_independence_percent": float(100*correct[i].mean()**32),
            "position_specific_independence_percent": float(100*np.prod(p)),
            "first_position_percent": float(100*p[0]), "remaining_positions_mean_percent": float(100*p[1:].mean()),
            "first_and_homogeneous_remainder_independence_percent": float(100*p[0]*p[1:].mean()**31)})
    first_token = []
    for i, arm in enumerate(MAIN_ARMS):
        predicted = predictions[i, :, :, :, 0].reshape(-1)
        actual = np.broadcast_to(truth[:, 0], (10, 2, 256)).reshape(-1)
        wrong = predicted != actual
        left = tokenizer.batch_decode(predicted[:, None].tolist(), clean_up_tokenization_spaces=False)
        right = tokenizer.batch_decode(actual[:, None].tolist(), clean_up_tokenization_spaces=False)
        space_only = sum(bool(w and a.lstrip(" ") == b.lstrip(" ")) for w, a, b in zip(wrong, left, right))
        first_token.append({"arm": arm, "first_token_errors": int(wrong.sum()),
                            "leading_space_only_first_token_errors": space_only})
    annotations, distributions, span_rows = span_tables(docs, spans)
    span_ranges = []
    for length, kind in itertools.product((1, 2), ("name", "number")):
        values = [row["recovery_percent"] for row in span_rows
                  if row["defence"] == "secret_mixup" and row["kind"] == kind
                  and row["length_tokens"] == length]
        if not values:
            raise ValueError(f"missing secret-mixup span stratum: {kind}/{length}")
        span_ranges.append({"defence": "secret_mixup", "kind": kind, "length_tokens": length,
                            "conditions": len(values), "minimum_percent": min(values),
                            "maximum_percent": max(values)})
    source_sha = {s["path"]: s["sha256"] for s in sources}
    if any(digest(ROOT / p) != sha for p, sha in source_sha.items()):
        raise ValueError("a source changed during analysis")
    report = {"analysis_status": "exploratory_post_hoc_not_preregistered",
        "setting": {"model": "openai-community/gpt2", "depth": 6, "copy": "public", "candidate_budget": 32,
                    "victim_seeds": protocol["victim_seeds"], "attack_seeds": protocol["attack_seeds"],
                    "distinct_documents": 256, "tokens_per_document": 32},
        "bootstrap": {"replicates": args.bootstrap_replicates, "seed": args.bootstrap_seed,
            "method": "paired percentile crossed bootstrap; average two attack seeds first; resample ten victim seeds and 256 documents independently",
            "ci_level": .95, "rng": "numpy.random.default_rng PCG64",
            "rng_order": "for every replicate: 10 victim indices, then 256 document indices",
            "multiplicity_weighted": True},
        "normalization": {"decoded": "GPT-2 decode with clean_up_tokenization_spaces=False and skip_special_tokens=False",
            "leading_space": "strip ASCII spaces from left boundary of each complete decoded sequence only",
            "whitespace": "collapse every Unicode whitespace run to one ASCII space, then strip both boundaries"},
        "metrics": summary, "paired_token_transitions": transitions,
        "selected_replica_median_comparison": replica_result,
        "independence_comparisons": independence, "first_token_diagnostics": first_token,
        "span_distributions": distributions, "span_length_ranges": span_ranges,
        "tokenizer": tokenizer_provenance,
        "software": {"python": platform.python_version(), "numpy": np.__version__},
        "limits": ["No fresh training, inference, attack execution, or longer-document experiment was performed.",
            "All new analyses and intervals are exploratory and unadjusted for multiple comparisons.",
            "Bootstrap uncertainty conditions on one fixed training corpus, one evaluation corpus source, the observed ten victim seeds, and the two averaged attack seeds.",
            "Prefix results truncate already reconstructed 32-token outputs; attacks were not rerun at shorter lengths and outputs may use full-window information.",
            "Independence products are descriptive plug-in baselines; neither establish independence nor identify the sign of error correlation.",
            "Whitespace normalization intentionally discards text differences; exact token-ID results remain separately reported.",
            "Span annotations are fixed AI-assisted semantic names and deterministic lexical number/identifier proxies, not independently adjudicated human gold.",
            "Only two distinct identifier spans occur in two documents; repeated model/attack evaluations do not support class-level identifier generalization.",
            "Span length strata remain descriptive and sparse at some lengths; they do not isolate a causal semantic-category effect.",
            "All source journal chains and their retained report SHA bindings, selected prediction SHA bindings to journal cell_complete events, paired completeness and selected stored per-document metrics were verified; these internal bindings are not an independent external authenticity anchor.",
            "The selected largest replica-median drop has an exploratory pointwise interval; no maximum-selection or simultaneous-comparison adjustment was performed."]}
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "review-metrics.json", report)
    write_json(args.output / "source-manifest.json", {"sources": sources,
        "all_sources_rehashed_after_analysis": True, "script_sha256": digest(Path(__file__))})
    write_csv(args.output / "position-accuracy.csv", positional)
    write_csv(args.output / "span-annotations-token-lengths.csv", annotations)
    write_csv(args.output / "span-length-strata.csv", span_rows)
    write_csv(args.output / "paired-metric-summary.csv", [{"metric": k, **{f:v for f,v in value.items() if not isinstance(v,list)}}
                                                         for k,value in summary.items()])
    np.savez_compressed(args.output / "paired-bootstrap-draws.npz", names=np.asarray(bootstrap_names),
                        arm_rates=bootstrap_draws, selected_replica_arm_rates=replica_draws[:, 0])
    write_json(args.output / "alignment.json", {"document_ids": [d["id"] for d in docs],
        "prediction_grid_shape_arms_victims_attacks_documents_tokens": list(predictions.shape),
        "selected_prediction_files": sum(s["role"] == "retained_prediction" for s in sources),
        "paired_main_prediction_files": 40, "paired_selected_replica_prediction_files": 40,
        "span_conditions": len(spans), "source_journal_chains_verified": True,
        "selected_prediction_hashes_match_journal_events": True,
        "all_selected_retained_metrics_reconstructed_exactly": True})
    command = ".venv/bin/python experiments/paper_a_review_metrics.py"
    lines = ["# Paper A exploratory review metrics", "", "These post-hoc analyses reuse retained predictions and do not change the preregistered evidence.", "",
             f"Reproduce from the repository root: `{command}` (use a fresh `--output` directory).", "",
             "| Metric | Activation (%) | Joint (%) | Paired gain, pp [95% CI] | Ratio [95% CI] |", "|---|---:|---:|---:|---:|"]
    for name, value in summary.items():
        lo, hi = value["difference_ci95_pp"]; rlo, rhi = value["ratio_ci95"]
        lines.append(f"| {name} | {value['activation_percent']:.5f} | {value['joint_percent']:.5f} | {value['difference_pp']:.5f} [{lo:.5f}, {hi:.5f}] | {value['ratio']:.5f} [{rlo:.5f}, {rhi:.5f}] |")
    lines += ["", "Prefix rows score truncated existing outputs, not independently attacked shorter or longer documents.", "",
        f"Of {act_errors:,} activation-only token errors, the joint attacker corrects {transitions['corrected_activation_error']:,} ({transitions['gross_correction_percent_of_activation_errors']:.5f}%) and introduces {transitions['introduced_error']:,} errors elsewhere; the net reduction is {transitions['net_error_reduction_percent']:.5f}%.", "",
        "Complete numerical results and limitations: [review-metrics.json](review-metrics.json). Source bindings: [source-manifest.json](source-manifest.json). Pairing verification: [alignment.json](alignment.json).", "",
        "Span length data: [annotations](span-annotations-token-lengths.csv), [condition-by-length recovery](span-length-strata.csv). Position rates: [position-accuracy.csv](position-accuracy.csv).", "",
        "## Limits", ""] + [f"- {limit}" for limit in report["limits"]]
    (args.output / "README.md").write_text("\n".join(lines) + "\n")
    # Every reported statistic comes from the same objects written to JSON/CSV.
    # Keep only the design choices (the selected lengths and conditions) literal.
    position_products = " / ".join(f"{row['position_specific_independence_percent']:.5f}%"
                                   for row in independence)
    pooled_products = " / ".join(f"{row['homogeneous_independence_percent']:.5f}%"
                                 for row in independence)
    prefix_lengths = (8, 16, 24, 32)
    prefix_labels = "/".join(str(length) for length in prefix_lengths)
    prefix_ratios = "/".join(f"{summary[f'prefix_{length}_token_id']['ratio']:.5f}"
                            for length in prefix_lengths)
    by_kind = {row["kind"]: row for row in distributions}
    by_stratum = {(row["kind"], row["length_tokens"]): row for row in span_ranges}
    span_sentences = []
    for length in (1, 2):
        name_range = by_stratum["name", length]
        number_range = by_stratum["number", length]
        if name_range["conditions"] != number_range["conditions"]:
            raise ValueError("class range summaries use different condition counts")
        overlap = max(name_range["minimum_percent"], number_range["minimum_percent"]) <= min(
            name_range["maximum_percent"], number_range["maximum_percent"])
        relationship = "the ranges overlap" if overlap else "the ranges do not overlap"
        span_sentences.append(
            f"At {length} token(s), across {name_range['conditions']} secret-mixup conditions, "
            f"name recovery ranges {name_range['minimum_percent']:.2f}–{name_range['maximum_percent']:.2f}% "
            f"and number recovery {number_range['minimum_percent']:.2f}–{number_range['maximum_percent']:.2f}%; "
            f"{relationship}. These are descriptive ranges, not a test of a class effect.")
    span_description = (
        f"Names average {by_kind['name']['mean_length_tokens']:.5f} tokens "
        f"({by_kind['name']['distinct_spans']} spans), versus "
        f"{by_kind['number']['mean_length_tokens']:.5f} for numbers "
        f"({by_kind['number']['distinct_spans']} spans). " + " ".join(span_sentences) +
        f" The {by_kind['identifier']['distinct_spans']} distinct identifier spans "
        f"in {by_kind['identifier']['distinct_documents']} documents support no class-level generalization.")
    setting = report["setting"]
    short = ["# Exploratory retained-prediction reanalysis", "",
        f"Main setting: {setting['model']}, split depth {setting['depth']}, {setting['copy']} client copy, "
        f"k={setting['candidate_budget']}; {len(setting['victim_seeds'])} victim seeds, "
        f"{len(setting['attack_seeds'])} attack seeds and {setting['distinct_documents']} distinct "
        f"{setting['tokens_per_document']}-token windows. All new analyses are post-hoc. "
        "No training, inference or attack was rerun.", "",
        f"Exact document recovery: {summary['document_token_id']['activation_percent']:.5f}% to {summary['document_token_id']['joint_percent']:.5f}%; paired gain {summary['document_token_id']['difference_pp']:.5f} points (95% CI {summary['document_token_id']['difference_ci95_pp']}), ratio {summary['document_token_id']['ratio']:.6f} (95% CI {summary['document_token_id']['ratio_ci95']}).", "",
        f"Removing leading ASCII spaces gives {summary['document_leading_space_normalized']['activation_percent']:.5f}% to {summary['document_leading_space_normalized']['joint_percent']:.5f}%, ratio {summary['document_leading_space_normalized']['ratio']:.6f}. Collapsing whitespace runs and trimming both boundaries gives {summary['document_whitespace_normalized']['activation_percent']:.5f}% to {summary['document_whitespace_normalized']['joint_percent']:.5f}%, ratio {summary['document_whitespace_normalized']['ratio']:.6f}.", "",
        f"The joint attacker corrects {transitions['corrected_activation_error']:,} of {act_errors:,} activation errors ({transitions['gross_correction_percent_of_activation_errors']:.5f}%) and introduces {transitions['introduced_error']:,} errors; net error reduction is {transitions['net_error_reduction_percent']:.5f}%.", "",
        f"The position-specific independence product is {position_products}, compared with pooled "
        f"homogeneous p^{setting['tokens_per_document']} values {pooled_products}. These descriptive "
        f"baselines do not identify error correlation. Prefix ratios at {prefix_labels} positions are "
        f"{prefix_ratios}; these truncate existing full-window reconstructions.", "",
        span_description, "",
        f"The selected largest replica-median document drop is {replica_result['undefended_percent']:.5f}% "
        f"to {replica_result['replica_median_percent']:.5f}% "
        f"({replica_result['attack']}, {replica_result['iterations_per_token']} iterations per token, "
        f"{replica_result['copy']} copy, depth {replica_result['depth']}), a {replica_result['drop_pp']:.5f}-point "
        f"drop with 95% pointwise CI {replica_result['drop_ci95_pp']}. This interval does not adjust "
        "for selecting the maximum.", "",
        f"Intervals use {args.bootstrap_replicates:,} paired crossed-bootstrap replicates with seed {args.bootstrap_seed}, averaging attack seeds before independent document/victim resampling. They condition on the fixed training corpus. All source journal chains, report SHA bindings and selected prediction SHA bindings to journal events were verified; this supplies internal consistency, not independent external authenticity.", "",
        "See [review-metrics.json](review-metrics.json), [source-manifest.json](source-manifest.json), [alignment.json](alignment.json), and [span-length-strata.csv](span-length-strata.csv). The reproducible script is `experiments/paper_a_review_metrics.py`. All limitations appear in [README.md](README.md)."]
    (args.output / "report.md").write_text("\n".join(short) + "\n")
    (args.output / "SOURCE-MANIFEST.sha256").write_text("".join(
        f"{sha}  {path}\n" for path, sha in sorted(source_sha.items()))
        + f"{digest(Path(__file__))}  {Path(__file__).relative_to(ROOT)}\n")
    (args.output / "MANIFEST.sha256").write_text("".join(
        f"{digest(path)}  {path.name}\n" for path in sorted(args.output.iterdir())
        if path.is_file() and path.name != "MANIFEST.sha256"))
    print(json.dumps({"output": str(args.output), "metrics": summary,
                      "paired_token_transitions": transitions, "independence": independence,
                      "span_distributions": distributions}, indent=2), flush=True)


if __name__ == "__main__":
    main()
