"""Development pilot over the trained splitbench defenders.

Matrix: defenses {none, gaussian, trained-bottleneck, secret-mixup,
replica-median} x splits {3, 6} x victim seeds {11, 22, 33} on the 96
development wikitext-2 articles of the earlier pretrained pilot
(results/pilot-gpt2-wikitext2/inputs/pilot-corpus.jsonl). Every defended
cell is paired with the undefended (none) control at the identical seed
and split, and every cell carries the SAME attacker-access spec
(matched access): public calibration partition, pretrained public head
exported before victim training, no final victim head, fixed query
budgets for the ridge/beam/SIPIT baselines.

The frozen 256-document evaluation corpus stays untouched: the runner
and the analyzer both assert (fail closed) that the development corpus
is exactly the pilot corpus named by
experiments/validation_corpus_manifest.json source.excluded_pilot_sha256
and that no development article id appears among the frozen corpus's
selected ids.

Per cell (split{N}/seed-{S}/{defense}/): splitbench training captures
(captures/), a sealed run artifact (run.json + evidence), cell.json with
wall_seconds / attack_seconds / cuda_peak_allocated_bytes /
utility_before/after / provenance pointer / access spec, the common
seam-inversion forward + SIPIT baselines against the post-defense wire
(forward-baseline.json, sipit-baseline.json), and the defense-specific
consumers:

  trained-bottleneck: trained_defenses convert -> representation-privacy
                      --bundle (bottleneck + gram_flattening arms)
  secret-mixup:       convert-mixup -> view-privacy --bundle (mixup arm)
  replica-median:     convert-replica -> replica-integrity
                      --captures/--truth; the one-malicious and
                      two-colluding arms inject their perturbation at
                      the aggregation point (replica_integrity.perturb
                      replaces channel outputs before aggregate())

--dry-run writes the full matrix directory structure (status PLANNED
cells) without torch; --quick runs the whole matrix on CPU with a tiny
random GPT-2 and synthetic documents (engineering smoke tier only).
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attacker.evidence import file_sha256  # noqa: E402

SCHEMA = "llm_attacker.defended_pilot.v1"
CELL_SCHEMA = "llm_attacker.defended_pilot_cell.v1"
DEFENSES = ("none", "gaussian", "trained-bottleneck", "secret-mixup",
            "replica-median")
TRAINED_DEFENSES = ("trained-bottleneck", "secret-mixup", "replica-median")
DEFAULT_SPLITS = (3, 6)
DEFAULT_SEEDS = (11, 22, 33)
DEFAULT_CORPUS = "results/pilot-gpt2-wikitext2/inputs/pilot-corpus.jsonl"
DEFAULT_MANIFEST = "experiments/validation_corpus_manifest.json"

PROVENANCE_NAMES = {
    "trained-bottleneck": "trained_bottleneck_provenance.json",
    "secret-mixup": "secret_mixup_provenance.json",
    "replica-median": "replica_median_provenance.json",
}


# ---------------------------------------------------------------------------
# Frozen-corpus separation (fail closed)


def load_frozen_article_ids(manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get("schema") != "llm_attacker.validation_corpus.v1":
        raise ValueError("unexpected frozen-corpus manifest schema")
    ids = set()
    selected = manifest.get("selected_ids", {})
    for partition, values in selected.items():
        ids.update(values)
    excluded = manifest.get("source", {}).get("excluded_pilot_sha256")
    if not ids or not isinstance(excluded, str) or len(excluded) != 64:
        raise ValueError("frozen-corpus manifest lacks selected ids or "
                         "excluded_pilot_sha256")
    return ids, excluded


def assert_development_corpus_disjoint(corpus_path, manifest_path,
                                       require_excluded_pilot=True):
    """The defended pilot may only touch the 96 development articles.

    Fail closed: the corpus file must BE the pilot corpus the frozen
    manifest excludes (require_excluded_pilot), and no article id may
    appear in any frozen-corpus partition (calibration/train/evaluation).
    """
    docs = [json.loads(line) for line in Path(corpus_path).read_text().splitlines()
            if line.strip()]
    ids = {d["id"] for d in docs}
    if len(ids) != len(docs):
        raise ValueError("duplicate article ids in the development corpus")
    frozen, excluded = load_frozen_article_ids(manifest_path)
    overlap = sorted(ids & frozen)
    if overlap:
        raise ValueError(
            "development corpus overlaps the frozen evaluation corpus: "
            f"{len(overlap)} shared article ids, e.g. {overlap[:3]}")
    digest = file_sha256(corpus_path)
    if require_excluded_pilot and digest != excluded:
        raise ValueError(
            "development corpus is not the excluded pilot corpus named by "
            f"the frozen manifest ({digest} != {excluded}); refusing to "
            "run against anything but the 96 development articles")
    return {"corpus_sha256": digest, "articles": len(docs),
            "frozen_manifest": str(manifest_path),
            "excluded_pilot_sha256": excluded,
            "article_ids_disjoint_from_frozen": True,
            "identity_checked": require_excluded_pilot}


# ---------------------------------------------------------------------------
# Matched attacker access (identical across defense conditions)


def matched_access_spec(args):
    return {
        "schema": "llm_attacker.defended_pilot_access.v1",
        "matched_across_defenses": True,
        "calibration": {"documents": args.calibration_seqs,
                        "source": "public calibration partition of the "
                                  "96-article development corpus",
                        "distinct_from_scoring_truth": True},
        "public_head": "pretrained public head exported before victim "
                       "training (split{N}/public-head-{seed}.pt)",
        "victim_final_head_granted": False,
        "public_defense_decoder": "granted when the defense exposes a cloud decoder (bottleneck v2)",
        "observations": "post-defense training-phase fwd wire captures "
                        "plus canonical scoring truth (scoring only)",
        "query_budgets": {
            "ridge_fit": "one pass over the public calibration partition",
            "forward_beam": {"candidate_count": args.forward_candidates,
                             "beam_width": args.beam_width},
            "sipit": {"steps_per_token": args.sipit_steps},
            "scored_sequences": args.attack_sequences},
    }


# ---------------------------------------------------------------------------
# Cell execution (torch path)


def _tiny_model(seed):
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel
    torch.manual_seed(seed)
    return GPT2LMHeadModel(GPT2Config(
        vocab_size=64, n_embd=32, n_layer=8, n_head=4, n_positions=64,
        resid_pdrop=0., embd_pdrop=0., attn_pdrop=0.))


def _synthetic_docs(n=18, seq_len=8, data_seed=2026):
    docs = []
    for i in range(n):
        rng = random.Random(i * 131 + data_seed)
        docs.append({"id": f"synthetic-{i}",
                     "tokens": [rng.randrange(64) for _ in range(seq_len)],
                     "text": " ".join(str(rng.randrange(64))
                                      for _ in range(seq_len))})
    return docs


def _partition(docs, data_seed):
    docs = sorted(docs, key=lambda d: d["id"])
    random.Random(data_seed).shuffle(docs)
    n = len(docs) // 3
    return {"calibration": docs[:n], "victim_train": docs[n:2 * n],
            "utility_eval": docs[2 * n:]}


def _bench_config(args, split_depth, seed, defense, cell_dir):
    from attacker import splitbench as sb
    kw = {}
    if defense == "trained-bottleneck":
        kw = dict(tb_dim=args.tb_dim, tb_hidden=args.tb_hidden,
                  tb_probe_hidden=args.tb_probe_hidden,
                  tb_minimax_steps=args.tb_minimax_steps,
                  tb_allow_cpu=(args.device == "cpu"))
    cfg = sb.BenchConfig(
        split=str(split_depth), mode="finetune", steps=args.train_steps,
        seq_len=args.seq_len, batch_size=1, lr=args.lr, defense=defense,
        sigma=args.sigma if defense == "gaussian" else 0.0, seed=seed,
        save_canonical=True, device=args.device,
        capture_dir=str(cell_dir / "captures"),
        output=str(cell_dir / "run.json"), **kw)
    return cfg.validate()


def _utility(split, cfg, batches):
    from attacker.experiment import utility
    return utility(split, cfg, batches)


def _provenance_pointer(cell_dir, defense):
    if defense in PROVENANCE_NAMES:
        path = cell_dir / "captures" / PROVENANCE_NAMES[defense]
    else:
        path = cell_dir / "run.json"
    if not path.is_file():
        raise RuntimeError(f"provenance sidecar missing: {path}")
    return {"path": str(path.relative_to(cell_dir)),
            "sha256": file_sha256(path)}


def seam_baselines(args, cell_dir, split_depth, seed, head_pt, cal):
    """Common leakage scale: forward beam + SIPIT against the cell's
    post-defense fwd wire, using only the matched-access head/calibration."""
    import torch
    from attacker import captures
    from attacker.attacks import seam_inversion as si
    from attacker.reconstruction_baselines import (
        MeteredHead, beam_reconstruct, candidate_ids, forward_scores,
        sipit_reconstruct)
    split, embedding = si._load_head_pt(str(head_pt), args.device)
    split.model.requires_grad_(False)
    embedding = embedding.detach()
    head = MeteredHead(split)
    with torch.no_grad():
        head(ids=cal[:1])
        hc = head(ids=cal).reshape(-1, embedding.shape[-1])
        decoder, _ = si._ridge_fit(hc, embedding[cal.flatten()], 0)
    cap_dir = cell_dir / "captures"
    records = [(m, p) for m, p in captures.scan_captures(str(cap_dir), "training")
               if m.get("phase") == "fwd"][:args.attack_sequences]
    if not records:
        raise RuntimeError(f"no fwd captures in {cap_dir}")
    hidden = torch.cat([captures.load_tensor(p).reshape(1, -1, embedding.shape[-1])
                        for _, p in records]).to(args.device)
    public_decoder = cell_dir / 'public-bottleneck-decoder.pt'
    if public_decoder.is_file():
        geometry = torch.load(public_decoder, map_location=args.device, weights_only=True)
        basis, mean = geometry['basis'], geometry['mean']
        hidden = hidden[..., :basis.shape[1]] @ basis.T + mean
    out = {}
    with head.measure() as work, torch.no_grad():
        fwd = forward_scores(hidden, embedding, decoder, head)
        cand = candidate_ids(fwd, None, args.forward_candidates, "forward")
        pred, info = beam_reconstruct(head, hidden, cand, args.beam_width)
    (cell_dir / 'forward-predictions.json').write_text(json.dumps(pred.cpu().tolist()))
    truth = torch.load(cap_dir / 'canonical.pt', map_location=args.device, weights_only=True)[:len(records)]
    if truth.shape != pred.shape:
        raise RuntimeError('canonical truth is not row-aligned to predictions')
    correct = (pred == truth)
    out["forward"] = {
        "accuracy": correct.float().mean().item(),
        "n_tokens": int(truth.numel()), "n_sequences": int(truth.shape[0]),
        "candidate_count": args.forward_candidates,
        "beam_width": args.beam_width,
        "work": {k: v for k, v in vars(work).items()},
        "artifact": "forward-baseline.json"}
    (cell_dir / "forward-baseline.json").write_text(json.dumps(
        {"schema": "llm_attacker.defended_pilot_forward_baseline.v1",
         "split": split_depth, "seed": seed,
         "accuracy": out["forward"]["accuracy"],
         "per_sequence_accuracy": correct.float().mean(-1).tolist(),
         "residuals": info,
         "budgets": {"candidate_count": args.forward_candidates,
                     "beam_width": args.beam_width}},
        indent=2, allow_nan=False))
    with head.measure() as work:
        pred, info = sipit_reconstruct(head, hidden, embedding,
                                       steps=args.sipit_steps, seed=seed)
    (cell_dir / 'sipit-predictions.json').write_text(json.dumps(pred.cpu().tolist()))
    correct = (pred == truth)
    out["sipit"] = {
        "accuracy": correct.float().mean().item(),
        "n_tokens": int(truth.numel()), "n_sequences": int(truth.shape[0]),
        "steps_per_token": args.sipit_steps,
        "work": {k: v for k, v in vars(work).items()},
        "artifact": "sipit-baseline.json"}
    (cell_dir / "sipit-baseline.json").write_text(json.dumps(
        {"schema": "llm_attacker.defended_pilot_sipit_baseline.v1",
         "split": split_depth, "seed": seed,
         "accuracy": out["sipit"]["accuracy"],
         "per_sequence_accuracy": correct.float().mean(-1).tolist(),
         "diagnostics": {k: v for k, v in info.items()
                         if k != "iterations_by_position"},
         "budgets": {"steps_per_token": args.sipit_steps}},
        indent=2, allow_nan=False))
    del split, head, embedding, decoder, hidden
    return out


def defense_consumers(args, defense, cell_dir, seed):
    """Defense-specific attack consumers on the converted releases."""
    from attacker import trained_defenses as td
    out = {}
    if defense == "trained-bottleneck":
        from attacker.attacks import representation_privacy as rp
        bundle = cell_dir / "representation-bundle.json"
        if not bundle.is_file():
            raise ValueError("public auxiliary representation bundle required; legacy victim-label conversion forbidden")
        artifact = cell_dir / "representation-privacy.json"
        rp.run(rp.build_parser().parse_args(
            ["--bundle", str(bundle), "--output", str(artifact),
             "--arms", "bottleneck", "--nonlinear-probe", "--evidence",
             "--seed", str(seed)]))
        raw = json.loads(artifact.read_text())
        out["representation-privacy"] = {
            "bundle": bundle.name, "artifact": artifact.name,
            "access": "public labeled auxiliary releases; post-training victim scoring releases",
            "arms": {a: {"probes": r["results"]}
                     for a, r in raw["results"]["arms"].items()
                     if isinstance(r, dict)}}
    elif defense == "secret-mixup":
        from attacker.attacks import view_privacy as vp
        bundle = cell_dir / "view-bundle.json"
        td.capture_to_view_privacy_bundle(str(cell_dir / "captures"),
                                          output=str(bundle))
        artifact = cell_dir / "view-privacy.json"
        vp.run(vp.build_parser().parse_args(
            ["--bundle", str(bundle), "--output", str(artifact),
             "--seed", str(seed)]))
        raw = json.loads(artifact.read_text())
        out["view-privacy"] = {
            "bundle": bundle.name, "artifact": artifact.name,
            "rows": [{"arm": r["arm"], "mse": r.get("mse")}
                     for r in raw["results"] if r.get("case") == "mixup"]}
    elif defense == "replica-median":
        from attacker.attacks import replica_integrity as ri
        observations = cell_dir / "replica-observations.json"
        truth = cell_dir / "replica-truth.json"
        td.capture_to_replica_bundle(str(cell_dir / "captures"),
                                     observations=str(observations),
                                     truth=str(truth),
                                     max_blocks=args.attack_sequences)
        artifact = cell_dir / "replica-integrity.json"
        ri.run(ri.build_parser().parse_args(
            ["--captures", str(observations), "--truth", str(truth),
             "--output", str(artifact),
             "--arms", "clean", "one-malicious", "two-colluding",
             "--shift", str(args.replica_shift),
             "--tolerance", str(args.replica_tolerance),
             "--error-tolerance", str(args.replica_error_tolerance),
             "--seed", str(seed)]))
        raw = json.loads(artifact.read_text())
        out["replica-integrity"] = {
            "observations": observations.name, "truth": truth.name,
            "artifact": artifact.name,
            "arms": raw["summary"],
            "perturbation_point": "channel outputs replaced before the "
                                  "coordinate-median aggregation point "
                                  "(replica_integrity.perturb)"}
    return out


def run_cell(args, base, split_depth, seed, defense, encoded, cell_dir,
             access):
    import torch
    from attacker import splitbench as sb
    from attacker.attacks import seam_inversion as si
    from attacker.evidence import EvidenceLogger
    cell_dir.mkdir(parents=True)
    torch.manual_seed(seed)
    model = copy.deepcopy(base).to(args.device)
    split = sb.build_split(
        model, sb.SplitPlan.parse(str(split_depth),
                                  n_layers=model.config.n_layer))
    cfg = _bench_config(args, split_depth, seed, defense, cell_dir)
    from experiments.pilot_measurements import utility as matched_utility, representation_bundle
    measured = {}
    def measurement(stage, current_split, current_cfg, state):
        measured[stage] = matched_utility(current_split, current_cfg, state,
                                          encoded['utility_eval'], encoded['calibration'])
        measured[stage]['stage'] = stage + '_victim_training'
        if stage == 'after' and defense == 'trained-bottleneck':
            geometry = state['tb'].geometry
            if geometry is not None:
                torch.save({k: geometry[k] for k in ('mean', 'basis')},
                           cell_dir / 'public-bottleneck-decoder.pt')
            ids = json.loads((cell_dir.parents[2] / 'partitions.json').read_text())
            bundle = representation_bundle(current_split, state, encoded,
                                            {k: [d['id'] for d in v] for k, v in ids.items()})
            (cell_dir / 'representation-bundle.json').write_text(json.dumps(bundle))
    order = list(range(len(encoded["victim_train"])))
    random.Random(seed).shuffle(order)
    batches = [encoded["victim_train"][i] for i in order]
    writer = sb.CaptureWriter(str(cell_dir / "captures"), "training",
                              f"defended-pilot-s{split_depth}-{seed}-{defense}")
    began = time.perf_counter()
    try:
        with EvidenceLogger(str(cell_dir / "run.json"),
                            node_role=f"defended-pilot/{defense}",
                            extra={"cell": CELL_SCHEMA}) as ev:
            stats = sb.execute(cfg, split, writer, batches, evidence=ev,
                               measurement=measurement, capture_replica_input=True)
    finally:
        writer.close()
    if args.device == "cuda":
        torch.cuda.synchronize()
    wall = time.perf_counter() - began
    before, after = measured['before'], measured['after']
    head_pt = cell_dir.parents[1] / f"public-head-{seed}.pt"
    cal = torch.cat(encoded["calibration"][:args.calibration_seqs]).to(args.device)
    attack_began = time.perf_counter()
    attacks = seam_baselines(args, cell_dir, split_depth, seed, head_pt, cal)
    attacks.update(defense_consumers(args, defense, cell_dir, seed))
    if args.device == "cuda":
        torch.cuda.synchronize()
    attack_seconds = time.perf_counter() - attack_began
    cell = {
        "schema": CELL_SCHEMA, "status": "COMPLETE",
        "measurement_version": 2,
        "observation_location": "body_input_seam",
        "baseline_scope": "public-head reconstruction; uses public cloud decoder for bottleneck v2",
        "split": split_depth, "seed": seed, "defense": defense,
        "paired_control": ("../none/cell.json" if defense != "none" else None),
        "wall_seconds": wall, "attack_seconds": attack_seconds,
        "cuda_peak_allocated_bytes": (
            torch.cuda.max_memory_allocated() if args.device == "cuda"
            else None),
        "resources_scope": "victim wall includes capture I/O and matched utility measurements; attack time is separate; cuda peak is "
                           "process-wide (experiment.json convention)",
        "utility_before": before, "utility_after": after,
        "training": {"steps_run": stats["steps_run"],
                     "final_loss": stats.get("final_loss"),
                     "mean_loss": stats.get("mean_loss")},
        "access": access,
        "provenance": _provenance_pointer(cell_dir, defense),
        "run_artifact": "run.json",
        "evidence": "run.json.evidence.json",
        "captures_dir": "captures",
        "attacks": attacks,
    }
    if (cell_dir / 'public-bottleneck-decoder.pt').is_file():
        cell['public_decoder'] = {'path': 'public-bottleneck-decoder.pt', 'sha256': file_sha256(cell_dir / 'public-bottleneck-decoder.pt')}
    cell['artifact_hashes'] = {p.name: file_sha256(p) for p in cell_dir.glob('*.json')
                               if not p.name.endswith('.evidence.json')}
    with EvidenceLogger(str(cell_dir / 'cell.json')):
        (cell_dir / "cell.json").write_text(json.dumps(cell, indent=2, allow_nan=False))
    print(f"[defended-pilot] split={split_depth} seed={seed} "
          f"defense={defense} wall={wall:.1f}s", flush=True)
    del split, model
    return cell


def write_planned_cell(cell_dir, split_depth, seed, defense, access):
    cell_dir.mkdir(parents=True, exist_ok=True)
    cell = {"schema": CELL_SCHEMA, "status": "PLANNED",
            "split": split_depth, "seed": seed, "defense": defense,
            "paired_control": ("../none/cell.json" if defense != "none"
                               else None),
            "access": access,
            "provenance": {"path": PROVENANCE_NAMES.get(defense,
                                                        "run.json")},
            "run_artifact": "run.json", "captures_dir": "captures"}
    (cell_dir / "cell.json").write_text(json.dumps(cell, indent=2))


def run(args):
    output = Path(args.output)
    if output.exists():
        raise ValueError("output directory already exists; use a fresh one")
    output.mkdir(parents=True)
    access = matched_access_spec(args)
    (output / "access.json").write_text(json.dumps(access, indent=2))
    tier = "engineering_smoke" if args.quick else "development_pilot"
    if args.quick:
        docs = _synthetic_docs(seq_len=args.seq_len,
                               data_seed=args.data_seed)
        corpus_path = output / "inputs" / "synthetic-corpus.jsonl"
        corpus_path.parent.mkdir(parents=True, exist_ok=True)
        corpus_path.write_text("".join(
            json.dumps({"id": d["id"], "text": d["text"]}) + "\n"
            for d in docs))
        disjointness = {"article_ids_disjoint_from_frozen": True,
                        "identity_checked": False,
                        "corpus_sha256": file_sha256(corpus_path),
                        "articles": len(docs),
                        "frozen_manifest": str(args.manifest),
                        "note": "synthetic smoke corpus; the fail-closed "
                                "pilot-corpus identity check applies to "
                                "development_pilot runs only"}
    else:
        corpus_path = Path(args.corpus)
        if not corpus_path.is_file():
            raise ValueError(f"development corpus not found: {corpus_path}")
        disjointness = assert_development_corpus_disjoint(
            corpus_path, args.manifest, require_excluded_pilot=True)
        docs = [json.loads(line) for line in corpus_path.read_text().splitlines()
                if line.strip()]
    plan = {"schema": SCHEMA, "tier": tier,
            "measurement_version": 2,
            "purpose": "development pilot over the trained splitbench "
                       "defenders; the frozen 256-document evaluation "
                       "corpus stays untouched",
            "defenses": list(args.defenses), "splits": list(args.splits),
            "seeds": list(args.seeds),
            "expected_cells": (len(args.defenses) * len(args.splits)
                               * len(args.seeds)),
            "corpus": {"path": str(corpus_path),
                       "sha256": file_sha256(corpus_path)},
            "disjointness": disjointness,
            "access_spec_sha256": hashlib.sha256(
                json.dumps(access, sort_keys=True).encode()).hexdigest(),
            "budgets": {"seq_len": args.seq_len,
                        "train_steps": args.train_steps,
                        "calibration_seqs": args.calibration_seqs,
                        "forward_candidates": args.forward_candidates,
                        "beam_width": args.beam_width,
                        "sipit_steps": args.sipit_steps,
                        "attack_sequences": args.attack_sequences,
                        "lr": args.lr, "sigma": args.sigma,
                        "data_seed": args.data_seed,
                        "tb_minimax_steps": args.tb_minimax_steps},
            "runner": str(Path(__file__).name),
            "runner_sha256": file_sha256(__file__),
            "limits": [
                "development pilot for effect sizes and plumbing; not a "
                "confirmatory security claim",
                "three victim seeds; intervals are conditional on this "
                "fixed article sample",
                "trained defenders are in-repo re-implementations, not "
                "the original defense projects' code or weights",
                "the MINE reading in bottleneck provenance is a "
                "diagnostic, never a certificate",
                "split depth and defense are confounded with cell "
                "ordering on a single host"],
            "cells": []}
    for split_depth in args.splits:
        for seed in args.seeds:
            for defense in args.defenses:
                rel = f"split{split_depth}/seed-{seed}/{defense}"
                plan["cells"].append({"path": rel, "split": split_depth,
                                      "seed": seed, "defense": defense})
                if args.dry_run:
                    write_planned_cell(output / rel, split_depth, seed,
                                       defense, access)
    (output / "plan.json").write_text(json.dumps(plan, indent=2))
    print(f"[defended-pilot] plan: {plan['expected_cells']} cells, "
          f"tier={tier} -> {output}", flush=True)
    if args.dry_run:
        return 0

    import torch
    from attacker.attacks import seam_inversion as si
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(args.threads)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    partitions = _partition(docs, args.data_seed)
    encoded = {}
    tokenizer = None
    if not args.quick:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    for name, part in partitions.items():
        rows = []
        for doc in part:
            if args.quick:
                ids = doc["tokens"][:args.seq_len]
            else:
                ids = tokenizer.encode(doc["text"], add_special_tokens=False)
            if len(ids) < args.seq_len:
                raise ValueError(f"document {doc['id']} shorter than seq_len")
            rows.append(torch.tensor([ids[:args.seq_len]], dtype=torch.long))
        encoded[name] = rows
    manifest = {k: [{"id": d["id"],
                     "text_sha256": hashlib.sha256(d["text"].encode()).hexdigest()}
                    for d in partitions[k]] for k in partitions}
    (output / "partitions.json").write_text(json.dumps(manifest, indent=2))
    for seed in args.seeds:
        if args.quick:
            base = _tiny_model(seed)
        else:
            torch.manual_seed(seed)
            base = AutoModelForCausalLM.from_pretrained(args.model)
            if base.config.model_type != "gpt2":
                raise ValueError("defended pilot currently supports the "
                                 "GPT-2 family (public-head export)")
        for split_depth in args.splits:
            head_dir = output / f"split{split_depth}"
            head_dir.mkdir(parents=True, exist_ok=True)
            si.save_head_pt(str(head_dir / f"public-head-{seed}.pt"),
                            base, split_depth)
        for split_depth in args.splits:
            for defense in args.defenses:
                run_cell(args, base, split_depth, seed, defense, encoded,
                         output / f"split{split_depth}/seed-{seed}/{defense}",
                         access)
        del base
    plan["status"] = "COMPLETE"
    (output / "plan.json").write_text(json.dumps(plan, indent=2,
                                                 allow_nan=False))
    print(f"[defended-pilot] COMPLETE: {output}", flush=True)
    return 0


def build_parser(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True)
    ap.add_argument("--corpus", default=DEFAULT_CORPUS,
                    help="the 96-article development corpus (must hash to "
                         "the frozen manifest's excluded_pilot_sha256)")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--model", default="openai-community/gpt2")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--defenses", nargs="+", choices=DEFENSES,
                    default=list(DEFENSES))
    ap.add_argument("--splits", nargs="+", type=int,
                    default=list(DEFAULT_SPLITS))
    ap.add_argument("--seeds", nargs="+", type=int,
                    default=list(DEFAULT_SEEDS))
    ap.add_argument("--data-seed", type=int, default=2026)
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--train-steps", type=int, default=64,
                    help="victim optimizer steps; must cover >= 2 epochs "
                         "over the 32 victim documents (the view-privacy "
                         "mixup converter requires repeated targets), so "
                         "the default is 64, not the pilot's 32")
    ap.add_argument("--calibration-seqs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--forward-candidates", type=int, default=8)
    ap.add_argument("--beam-width", type=int, default=2)
    ap.add_argument("--sipit-steps", type=int, default=64)
    ap.add_argument("--attack-sequences", type=int, default=32)
    ap.add_argument("--tb-dim", type=int, default=64)
    ap.add_argument("--tb-hidden", type=int, default=256)
    ap.add_argument("--tb-probe-hidden", type=int, default=128)
    ap.add_argument("--tb-minimax-steps", type=int, default=200)
    ap.add_argument("--replica-shift", type=float, default=0.5)
    ap.add_argument("--replica-tolerance", type=float, default=0.02)
    ap.add_argument("--replica-error-tolerance", type=float, default=0.05)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--quick", action="store_true",
                    help="CPU smoke: tiny random GPT-2, synthetic "
                         "documents, minimal budgets (engineering tier)")
    ap.add_argument("--dry-run", action="store_true",
                    help="write the full matrix structure (PLANNED cells) "
                         "without torch")
    args = ap.parse_args(argv)
    if args.quick:
        args.device = "cpu"
        args.seq_len, args.calibration_seqs = 8, 6
        args.train_steps = 12  # 2 epochs over the 6 synthetic victim docs
        args.forward_candidates, args.beam_width = 4, 2
        args.sipit_steps, args.attack_sequences = 4, 4
        args.tb_dim, args.tb_hidden, args.tb_probe_hidden = 8, 16, 16
        args.tb_minimax_steps = 2
    if args.train_steps < 2 * (args.calibration_seqs if args.quick else 32):
        ap.error("train-steps must cover >= 2 victim epochs (mixup "
                 "converter requires repeated targets)")
    if len(set(args.seeds)) != len(args.seeds):
        ap.error("seeds must be unique")
    if "none" not in args.defenses:
        ap.error("the undefended control (none) is required for pairing")
    if args.dry_run and args.quick:
        ap.error("--dry-run already avoids torch; drop --quick")
    return args


def main(argv=None):
    return run(build_parser(argv))


if __name__ == "__main__":
    raise SystemExit(main())
