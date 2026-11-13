"""Bottleneck dimension/method development sweep.

Motivation: the defended development pilot found the frozen-encoder
trained-bottleneck configuration utility-destroying (PPL 48.3 undefended
-> 38M-124M defended, CE 4.44 -> 17.46) at 0.00% token recovery, and the
mechanism-v2 repair (public PCA geometry + bounded residual + public
decoder) still leaves a large utility cost. This sweep varies the
bottleneck DIMENSION and METHOD on development data:

  methods   pca-compress        compression-only control (no adversarial
                                training; isolates what the minimax adds)
            tb-frozen           trained-bottleneck, encoder frozen after
                                the calibration pre-phase
            tb-cotrain          trained-bottleneck, co-trained encoder
                                (interleaved minimax rounds every
                                --tb-cotrain-interval victim steps)
  dims      {64, 128, 256} (the wire's released coordinate count k)
  splits    {3, 6}; dev seeds {11, 22, 33}

on the 96 development wikitext-2 articles (the pilot corpus named by the
frozen manifest's source.excluded_pilot_sha256). Every defended cell is
paired with the undefended (none) control at the identical seed and
split (run once per (split, seed), shared across dims), and every cell
carries the SAME attacker-access spec (matched access, byte-identical
access.json convention): public calibration partition, pretrained public
head exported before victim training, no final victim head, fixed query
budgets. Attacker arms per cell: the common forward-beam + SIPIT
baselines against the post-defense wire, plus a DEFENSE-AWARE arm
(defense class + k + public calibration known; a PCA surrogate
encoder+inverse fit on public calibration is the learned prior over the
observed wire — BPDA-style approximation of the straight-through
gradient; reconstruction_baselines.fit_pca_surrogate). Raised budgets
over the pilot: candidates 32, beam 4, SIPIT 128 steps/token.

Utility gate per defended cell: ppl_after <= 1.5x and <= 2x the paired
undefended ppl_after are BOTH reported (utility_gate in cell.json).

The frozen 256-document evaluation corpus stays untouched: the runner
and the analyzer fail closed on the same corpus-identity + disjointness
assertion as the defended pilot (run_defended_pilot.
assert_development_corpus_disjoint).

--dry-run writes the full matrix directory structure (status PLANNED
cells) without torch; --quick runs the matrix on CPU with a tiny random
GPT-2 and synthetic documents (engineering smoke tier only; dims fall
back to {4, 8} for the 32-wide tiny seam).
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from attacker.evidence import file_sha256  # noqa: E402
from run_defended_pilot import (  # noqa: E402
    DEFAULT_CORPUS, DEFAULT_MANIFEST, _partition, _synthetic_docs,
    _tiny_model, assert_development_corpus_disjoint)

SCHEMA = "llm_attacker.bottleneck_sweep.v1"
CELL_SCHEMA = "llm_attacker.bottleneck_sweep_cell.v1"
METHODS = ("pca-compress", "tb-frozen", "tb-cotrain")
TRAINED_METHODS = ("tb-frozen", "tb-cotrain")
# "none" is accepted by --methods to run ONLY the undefended controls
# (positive-control tier); it is never a defended cell method.
CONTROL_ONLY = "none"
DEFAULT_DIMS = (64, 128, 256)
DEFAULT_SPLITS = (3, 6)
DEFAULT_SEEDS = (11, 22, 33)

PROVENANCE_NAMES = {
    "pca-compress": "pca_compress_provenance.json",
    "tb-frozen": "trained_bottleneck_provenance.json",
    "tb-cotrain": "trained_bottleneck_provenance.json",
}

GATE_RATIOS = (1.5, 2.0)


# ---------------------------------------------------------------------------
# Matched attacker access (identical across methods/dims/splits/seeds)


def matched_access_spec(args):
    return {
        "schema": "llm_attacker.bottleneck_sweep_access.v1",
        "matched_across_defenses": True,
        "calibration": {"documents": args.calibration_seqs,
                        "source": "public calibration partition of the "
                                  "96-article development corpus",
                        "distinct_from_scoring_truth": True},
        "public_head": "pretrained public head exported before victim "
                       "training (split{N}/public-head-{seed}.pt)",
        "victim_final_head_granted": False,
        "public_defense_decoder": "granted when the defense exposes a "
                                  "cloud decoder (bottleneck v2 geometry, "
                                  "pca-compress basis)",
        "defense_aware_arm": {"knowledge": "defense CLASS and released "
                                           "width k only (no defense "
                                           "weights or randomness)",
                              "surrogate": "PCA encoder+inverse fit on "
                                           "the public calibration "
                                           "partition at the same k",
                              "use": "learned prior over the observed "
                                     "wire for the forward/beam/SIPIT "
                                     "kernels (BPDA-style straight-"
                                     "through approximation)"},
        "observations": "post-defense training-phase fwd wire captures "
                        "plus canonical scoring truth (scoring only)",
        "query_budgets": {
            "ridge_fit": "one pass over the public calibration partition",
            "forward_beam": {"candidate_count": args.forward_candidates,
                             "beam_width": args.beam_width},
            "sipit": {"steps_per_token": args.sipit_steps},
            "scored_sequences": args.attack_sequences,
        },
    }


# ---------------------------------------------------------------------------
# Cell execution (torch path)


def method_defense_kwargs(args, method, dim):
    """Sweep (method, dim) -> splitbench defense + knob overrides."""
    if method == "pca-compress":
        return "pca-compress", {"pca_dim": dim}
    kw = {"tb_dim": dim, "tb_hidden": args.tb_hidden,
          "tb_probe_hidden": args.tb_probe_hidden,
          "tb_minimax_steps": args.tb_minimax_steps,
          "tb_allow_cpu": args.device == "cpu",
          "tb_cotrain": method == "tb-cotrain",
          "tb_cotrain_interval": args.tb_cotrain_interval}
    return "trained-bottleneck", kw


def _bench_config(args, split_depth, seed, method, dim, cell_dir):
    from attacker import splitbench as sb
    defense, kw = (("none", {}) if method == "none"
                   else method_defense_kwargs(args, method, dim))
    cfg = sb.BenchConfig(
        split=str(split_depth), mode="finetune", steps=args.train_steps,
        seq_len=args.seq_len, batch_size=1, lr=args.lr, defense=defense,
        seed=seed, save_canonical=True, device=args.device,
        capture_dir=str(cell_dir / "captures"),
        output=str(cell_dir / "run.json"), **kw)
    return cfg.validate()


def _provenance_pointer(cell_dir, method):
    if method in PROVENANCE_NAMES:
        path = cell_dir / "captures" / PROVENANCE_NAMES[method]
    else:
        path = cell_dir / "run.json"
    if not path.is_file():
        raise RuntimeError(f"provenance sidecar missing: {path}")
    return {"path": str(path.relative_to(cell_dir)),
            "sha256": file_sha256(path)}


def _export_public_decoder(cell_dir, state, method):
    """Export the public cloud decoder the matched access grants."""
    import torch
    if method in TRAINED_METHODS:
        geometry = state["tb"].geometry
        if geometry is not None:
            payload = {k: geometry[k] for k in ("mean", "basis")}
            torch.save(payload, cell_dir / "public-bottleneck-decoder.pt")
            return "public-bottleneck-decoder.pt"
    if method == "pca-compress":
        pca = state["pca"]
        torch.save({"mean": pca.mean, "basis": pca.basis},
                   cell_dir / "public-pca-decoder.pt")
        return "public-pca-decoder.pt"
    return None


def _load_public_decoder(cell_dir, device):
    import torch
    for name in ("public-bottleneck-decoder.pt", "public-pca-decoder.pt"):
        path = cell_dir / name
        if path.is_file():
            return torch.load(path, map_location=device,
                              weights_only=True)
    return None


def seam_baselines(args, cell_dir, split_depth, seed, head_pt, cal,
                   method, dim):
    """Forward beam + SIPIT against the post-defense wire (matched-access
    head/calibration only), plus the defense-aware surrogate-inverse arm
    for defended cells."""
    import torch
    from attacker import captures
    from attacker.attacks import seam_inversion as si
    from attacker.reconstruction_baselines import (
        MeteredHead, beam_reconstruct, candidate_ids, fit_pca_surrogate,
        forward_scores, sipit_reconstruct, surrogate_decode,
        surrogate_encode, surrogate_self_consistency)
    split, embedding = si._load_head_pt(str(head_pt), args.device)
    split.model.requires_grad_(False)
    embedding = embedding.detach()
    head = MeteredHead(split)
    with torch.no_grad():
        head(ids=cal[:1])
        hc = head(ids=cal).reshape(-1, embedding.shape[-1])
        decoder, _ = si._ridge_fit(hc, embedding[cal.flatten()], 0)
    cap_dir = cell_dir / "captures"
    records = [(m, p) for m, p in captures.scan_captures(str(cap_dir),
                                                         "training")
               if m.get("phase") == "fwd"][:args.attack_sequences]
    if not records:
        raise RuntimeError(f"no fwd captures in {cap_dir}")
    hidden = torch.cat(
        [captures.load_tensor(p).reshape(1, -1, embedding.shape[-1])
         for _, p in records]).to(args.device)
    wire_raw = hidden
    geometry = _load_public_decoder(cell_dir, args.device)
    if geometry is not None:
        # matched access grants the public cloud decoder
        basis, mean = geometry["basis"], geometry["mean"]
        hidden = hidden[..., :basis.shape[1]] @ basis.T + mean
    truth = torch.load(cap_dir / "canonical.pt", map_location=args.device,
                       weights_only=True)[:len(records)]

    def score(pred):
        if truth.shape != pred.shape:
            raise RuntimeError("canonical truth is not row-aligned to "
                               "predictions")
        correct = (pred == truth)
        return {"accuracy": correct.float().mean().item(),
                "n_tokens": int(truth.numel()),
                "n_sequences": int(truth.shape[0])}

    out = {}
    with head.measure() as work, torch.no_grad():
        fwd = forward_scores(hidden, embedding, decoder, head)
        cand = candidate_ids(fwd, None, args.forward_candidates,
                             "forward")
        pred, info = beam_reconstruct(head, hidden, cand, args.beam_width)
    out["forward"] = {**score(pred),
                      "candidate_count": args.forward_candidates,
                      "beam_width": args.beam_width,
                      "work": {k: v for k, v in vars(work).items()},
                      "artifact": "forward-baseline.json"}
    (cell_dir / "forward-baseline.json").write_text(json.dumps(
        {"schema": "llm_attacker.bottleneck_sweep_forward_baseline.v1",
         "split": split_depth, "seed": seed, "method": method, "dim": dim,
         "accuracy": out["forward"]["accuracy"],
         "budgets": {"candidate_count": args.forward_candidates,
                     "beam_width": args.beam_width}},
        indent=2, allow_nan=False))
    with head.measure() as work:
        pred, info = sipit_reconstruct(head, hidden, embedding,
                                       steps=args.sipit_steps, seed=seed)
    out["sipit"] = {**score(pred), "steps_per_token": args.sipit_steps,
                    "work": {k: v for k, v in vars(work).items()},
                    "artifact": "sipit-baseline.json"}
    (cell_dir / "sipit-baseline.json").write_text(json.dumps(
        {"schema": "llm_attacker.bottleneck_sweep_sipit_baseline.v1",
         "split": split_depth, "seed": seed, "method": method, "dim": dim,
         "accuracy": out["sipit"]["accuracy"],
         "budgets": {"steps_per_token": args.sipit_steps}},
        indent=2, allow_nan=False))
    if method != "none":
        # Defense-aware arm: the attacker knows the defense class + k and
        # fits a PCA surrogate encoder+inverse on the PUBLIC calibration
        # partition; the surrogate inverse is the learned prior over the
        # observed wire (BPDA-style approximation of the straight-through
        # gradient). No defense weights or randomness are used.
        surrogate = fit_pca_surrogate(hc, dim)
        with torch.no_grad():
            cal_wire = torch.zeros_like(hc)
            cal_wire[:, :dim] = surrogate_encode(surrogate, hc)
            hc_hat = surrogate_decode(surrogate, cal_wire)
            da_decoder, _ = si._ridge_fit(hc_hat,
                                          embedding[cal.flatten()], 0)
            hidden_hat = surrogate_decode(surrogate, wire_raw)
        with head.measure() as work, torch.no_grad():
            fwd = forward_scores(hidden_hat, embedding, da_decoder, head)
            cand = candidate_ids(fwd, None, args.forward_candidates,
                                 "forward")
            pred, info = beam_reconstruct(head, hidden_hat, cand,
                                          args.beam_width)
        da = {"defense_aware": True, "defense_class": method,
              "k": dim,
              "surrogate": "pca_encoder_inverse_on_public_calibration",
              "surrogate_explained_variance_at_k":
                  surrogate["explained_variance_at_k"],
              "forward": {**score(pred),
                          "candidate_count": args.forward_candidates,
                          "beam_width": args.beam_width}}
        with head.measure() as work:
            pred, info = sipit_reconstruct(head, hidden_hat, embedding,
                                           steps=args.sipit_steps,
                                           seed=seed)
        da["sipit"] = {**score(pred),
                       "steps_per_token": args.sipit_steps}
        da["artifact"] = "defense-aware-baseline.json"
        payload = {"schema":
                   "llm_attacker.bottleneck_sweep_defense_aware_baseline.v1",
                   "split": split_depth, "seed": seed, "method": method,
                   "dim": dim, "defense_aware": True,
                   "forward_accuracy": da["forward"]["accuracy"],
                   "sipit_accuracy": da["sipit"]["accuracy"],
                   "surrogate_explained_variance_at_k":
                       da["surrogate_explained_variance_at_k"],
                   "budgets": {"candidate_count": args.forward_candidates,
                               "beam_width": args.beam_width,
                               "steps_per_token": args.sipit_steps}}
        if getattr(args, "da_self_consistency", False):
            # Diagnostic only (off by default; completed-sweep cells are
            # unaffected): how well the surrogate reconstructs its OWN
            # calibration distribution, end-to-end and in tensor terms.
            # A high value here alongside low wire accuracy isolates
            # cross-basis/distribution mismatch from within-distribution
            # fit (see experiments/notes/surrogate_arm_analysis.md).
            with torch.no_grad():
                cal_h = hc_hat.reshape(cal.shape[0], cal.shape[1], -1)
                sc_fwd = forward_scores(cal_h, embedding, da_decoder, head)
                sc_cand = candidate_ids(sc_fwd, None,
                                        args.forward_candidates, "forward")
                sc_pred, _ = beam_reconstruct(head, cal_h, sc_cand,
                                              args.beam_width)
            self_consistency = {
                "calibration_loop_forward_accuracy":
                    (sc_pred == cal).float().mean().item(),
                **surrogate_self_consistency(surrogate, hc)}
            da["surrogate_self_consistency"] = self_consistency
            payload["surrogate_self_consistency"] = self_consistency
        (cell_dir / "defense-aware-baseline.json").write_text(json.dumps(
            payload, indent=2, allow_nan=False))
        out["defense_aware"] = da
    del split, head, embedding, decoder, hidden
    return out


def utility_gate(cell_after_ppl, control_after_ppl):
    """Dual utility gate vs the paired undefended control: defended
    ppl_after must stay within 1.5x and 2x of the control's ppl_after."""
    ratio = cell_after_ppl / control_after_ppl
    return {"ppl_after": cell_after_ppl,
            "control_ppl_after": control_after_ppl,
            "ratio": ratio,
            "pass_1_5x": bool(ratio <= GATE_RATIOS[0]),
            "pass_2x": bool(ratio <= GATE_RATIOS[1]),
            "thresholds": list(GATE_RATIOS)}


def run_cell(args, base, split_depth, seed, method, dim, encoded,
             cell_dir, access, control_ppl_after=None):
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
    cfg = _bench_config(args, split_depth, seed, method, dim, cell_dir)
    from experiments.pilot_measurements import utility as matched_utility
    measured = {}

    def measurement(stage, current_split, current_cfg, state):
        measured[stage] = matched_utility(current_split, current_cfg,
                                          state, encoded["utility_eval"],
                                          encoded["calibration"])
        measured[stage]["stage"] = stage + "_victim_training"
        if stage == "after":
            decoder = _export_public_decoder(cell_dir, state, method)
            if decoder:
                measured["decoder"] = decoder

    order = list(range(len(encoded["victim_train"])))
    random.Random(seed).shuffle(order)
    batches = [encoded["victim_train"][i] for i in order]
    writer = sb.CaptureWriter(str(cell_dir / "captures"), "training",
                              f"bottleneck-sweep-s{split_depth}-{seed}-"
                              f"{method}-{dim}")
    began = time.perf_counter()
    try:
        with EvidenceLogger(str(cell_dir / "run.json"),
                            node_role=f"bottleneck-sweep/{method}",
                            extra={"cell": CELL_SCHEMA}) as ev:
            stats = sb.execute(cfg, split, writer, batches, evidence=ev,
                               measurement=measurement)
    finally:
        writer.close()
    if args.device == "cuda":
        torch.cuda.synchronize()
    wall = time.perf_counter() - began
    before, after = measured["before"], measured["after"]
    # heads live at split{N}/public-head-{seed}.pt (exported pre-training)
    head_pt = Path(args.output) / f"split{split_depth}" / \
        f"public-head-{seed}.pt"
    cal = torch.cat(encoded["calibration"][:args.calibration_seqs]).to(
        args.device)
    attack_began = time.perf_counter()
    attacks = seam_baselines(args, cell_dir, split_depth, seed, head_pt,
                             cal, method, dim)
    if args.device == "cuda":
        torch.cuda.synchronize()
    attack_seconds = time.perf_counter() - attack_began
    cell = {
        "schema": CELL_SCHEMA, "status": "COMPLETE",
        "measurement_version": 2,
        "observation_location": "body_input_seam",
        "baseline_scope": "public-head reconstruction; uses public cloud "
                          "decoder when the defense exposes one; the "
                          "defense-aware arm uses only the defense class, "
                          "k and the public calibration partition",
        "split": split_depth, "seed": seed, "method": method, "dim": dim,
        "paired_control": ("../../none/cell.json"
                           if method != "none" else None),
        "wall_seconds": wall, "attack_seconds": attack_seconds,
        "cuda_peak_allocated_bytes": (
            torch.cuda.max_memory_allocated() if args.device == "cuda"
            else None),
        "resources_scope": "victim wall includes capture I/O and matched "
                           "utility measurements; attack time is "
                           "separate; cuda peak is process-wide",
        "utility_before": before, "utility_after": after,
        "training": {"steps_run": stats["steps_run"],
                     "final_loss": stats.get("final_loss"),
                     "mean_loss": stats.get("mean_loss")},
        "access": access,
        "provenance": _provenance_pointer(cell_dir, method),
        "run_artifact": "run.json",
        "evidence": "run.json.evidence.json",
        "captures_dir": "captures",
        "attacks": attacks,
    }
    if measured.get("decoder"):
        cell["public_decoder"] = {
            "path": measured["decoder"],
            "sha256": file_sha256(cell_dir / measured["decoder"])}
    if method != "none":
        if control_ppl_after is None:
            raise RuntimeError("defended cells need the paired control's "
                               "ppl_after for the utility gate")
        cell["utility_gate"] = utility_gate(
            after["perplexity"], control_ppl_after)
    cell["artifact_hashes"] = {
        p.name: file_sha256(p) for p in cell_dir.glob("*.json")
        if not p.name.endswith(".evidence.json")}
    with EvidenceLogger(str(cell_dir / "cell.json")):
        (cell_dir / "cell.json").write_text(json.dumps(cell, indent=2,
                                                       allow_nan=False))
    print(f"[bottleneck-sweep] split={split_depth} seed={seed} "
          f"method={method} dim={dim} wall={wall:.1f}s", flush=True)
    del split, model
    return cell


def write_planned_cell(cell_dir, split_depth, seed, method, dim, access):
    cell_dir.mkdir(parents=True, exist_ok=True)
    cell = {"schema": CELL_SCHEMA, "status": "PLANNED",
            "split": split_depth, "seed": seed, "method": method,
            "dim": dim,
            "paired_control": ("../../none/cell.json"
                               if method != "none" else None),
            "access": access,
            "provenance": {"path": PROVENANCE_NAMES.get(method,
                                                       "run.json")},
            "run_artifact": "run.json", "captures_dir": "captures"}
    (cell_dir / "cell.json").write_text(json.dumps(cell, indent=2))


def cell_path(split_depth, seed, method, dim):
    if method == "none":
        return f"split{split_depth}/seed-{seed}/none"
    return f"split{split_depth}/seed-{seed}/dim-{dim}/{method}"


def run(args):
    output = Path(args.output)
    if output.exists():
        raise ValueError("output directory already exists; use a fresh one")
    output.mkdir(parents=True)
    access = matched_access_spec(args)
    (output / "access.json").write_text(json.dumps(access, indent=2))
    tier = "engineering_smoke" if args.quick else "development_sweep"
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
                                "development_sweep runs only"}
    else:
        corpus_path = Path(args.corpus)
        if not corpus_path.is_file():
            raise ValueError(f"development corpus not found: {corpus_path}")
        disjointness = assert_development_corpus_disjoint(
            corpus_path, args.manifest, require_excluded_pilot=True)
        docs = [json.loads(line) for line in
                corpus_path.read_text().splitlines() if line.strip()]
    defended_methods = [m for m in args.methods if m != CONTROL_ONLY]
    cells = []
    for split_depth in args.splits:
        for seed in args.seeds:
            cells.append({"path": cell_path(split_depth, seed, "none", 0),
                          "split": split_depth, "seed": seed,
                          "method": "none", "dim": 0})
            for dim in args.dims:
                for method in defended_methods:
                    cells.append({"path": cell_path(split_depth, seed,
                                                    method, dim),
                                  "split": split_depth, "seed": seed,
                                  "method": method, "dim": dim})
    plan = {"schema": SCHEMA, "tier": tier,
            "measurement_version": 2,
            "purpose": "bottleneck dimension/method development sweep: "
                       "pca-compress control x frozen/co-trained "
                       "trained-bottleneck x dims, with a defense-aware "
                       "attacker arm and dual 1.5x/2x utility gates; the "
                       "frozen 256-document evaluation corpus stays "
                       "untouched",
            "methods": list(args.methods), "dims": list(args.dims),
            "splits": list(args.splits), "seeds": list(args.seeds),
            "expected_cells": len(cells),
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
                        "lr": args.lr, "data_seed": args.data_seed,
                        "tb_minimax_steps": args.tb_minimax_steps,
                        "tb_cotrain_interval": args.tb_cotrain_interval},
            "utility_gate": {"thresholds": list(GATE_RATIOS),
                             "definition": "ppl_after <= threshold x the "
                                           "paired undefended ppl_after"},
            "runner": str(Path(__file__).name),
            "runner_sha256": file_sha256(__file__),
            "limits": [
                "development sweep for method/dim selection; not a "
                "confirmatory security claim",
                "three victim seeds; intervals are conditional on this "
                "fixed article sample",
                "trained defenders are in-repo re-implementations, not "
                "the original defense projects' code or weights",
                "the defense-aware surrogate is a PCA stand-in for the "
                "true defense map; it bounds a class-and-calibration "
                "adversary, not a defense-internals adversary",
                "the MINE reading in bottleneck provenance is a "
                "diagnostic, never a certificate",
                "split depth, dim and method are confounded with cell "
                "ordering on a single host"],
            "cells": cells}
    if args.dry_run:
        for c in cells:
            write_planned_cell(output / c["path"], c["split"], c["seed"],
                               c["method"], c["dim"], access)
    (output / "plan.json").write_text(json.dumps(plan, indent=2))
    print(f"[bottleneck-sweep] plan: {plan['expected_cells']} cells, "
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
                ids = tokenizer.encode(doc["text"],
                                       add_special_tokens=False)
            if len(ids) < args.seq_len:
                raise ValueError(f"document {doc['id']} shorter than "
                                 "seq_len")
            rows.append(torch.tensor([ids[:args.seq_len]],
                                     dtype=torch.long))
        encoded[name] = rows
    manifest = {k: [{"id": d["id"],
                     "text_sha256": hashlib.sha256(
                         d["text"].encode()).hexdigest()}
                    for d in partitions[k]] for k in partitions}
    (output / "partitions.json").write_text(json.dumps(manifest,
                                                       indent=2))
    controls = []
    for seed in args.seeds:
        if args.quick:
            base = _tiny_model(seed)
        else:
            torch.manual_seed(seed)
            base = AutoModelForCausalLM.from_pretrained(args.model)
            if base.config.model_type != "gpt2":
                raise ValueError("the bottleneck sweep currently supports "
                                 "the GPT-2 family (public-head export)")
        for split_depth in args.splits:
            head_dir = output / f"split{split_depth}"
            head_dir.mkdir(parents=True, exist_ok=True)
            si.save_head_pt(str(head_dir / f"public-head-{seed}.pt"),
                            base, split_depth)
        for split_depth in args.splits:
            control = run_cell(args, base, split_depth, seed, "none", 0,
                               encoded,
                               output / cell_path(split_depth, seed,
                                                  "none", 0), access)
            controls.append(control)
            control_ppl = control["utility_after"]["perplexity"]
            for dim in args.dims:
                for method in defended_methods:
                    run_cell(args, base, split_depth, seed, method, dim,
                             encoded,
                             output / cell_path(split_depth, seed, method,
                                                dim), access,
                             control_ppl_after=control_ppl)
        del base
    plan["status"] = "COMPLETE"
    (output / "plan.json").write_text(json.dumps(plan, indent=2,
                                                 allow_nan=False))
    print(f"[bottleneck-sweep] COMPLETE: {output}", flush=True)
    if args.min_control_accuracy is not None:
        failing = []
        for cell in controls:
            best = max(cell["attacks"]["forward"]["accuracy"],
                       cell["attacks"]["sipit"]["accuracy"])
            if best < args.min_control_accuracy:
                failing.append(f"split{cell['split']}/seed-{cell['seed']} "
                               f"best={best:.4f}")
        if failing:
            raise SystemExit(
                f"[bottleneck-sweep] positive-control gate FAILED: "
                f"position-aligned recovery < {args.min_control_accuracy} "
                f"in {len(failing)}/{len(controls)} control cells: "
                + "; ".join(failing))
        print(f"[bottleneck-sweep] positive-control gate PASSED: all "
              f"{len(controls)} control cells >= "
              f"{args.min_control_accuracy}", flush=True)
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
    ap.add_argument("--methods", nargs="+",
                    choices=list(METHODS) + [CONTROL_ONLY],
                    default=list(METHODS),
                    help="defended methods to sweep; \"none\" selects the "
                         "undefended controls only (positive-control tier: "
                         "no defended cells are run)")
    ap.add_argument("--min-control-accuracy", type=float, default=None,
                    help="when set, fail (exit 1) after completion unless "
                         "every undefended control cell reaches this "
                         "position-aligned recovery under the best of the "
                         "forward-beam / SIPIT baselines (positive-control "
                         "gate)")
    ap.add_argument("--dims", nargs="+", type=int,
                    default=list(DEFAULT_DIMS))
    ap.add_argument("--splits", nargs="+", type=int,
                    default=list(DEFAULT_SPLITS))
    ap.add_argument("--seeds", nargs="+", type=int,
                    default=list(DEFAULT_SEEDS))
    ap.add_argument("--data-seed", type=int, default=2026)
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--train-steps", type=int, default=64)
    ap.add_argument("--calibration-seqs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--forward-candidates", type=int, default=32)
    ap.add_argument("--beam-width", type=int, default=4)
    ap.add_argument("--sipit-steps", type=int, default=128)
    ap.add_argument("--attack-sequences", type=int, default=32)
    ap.add_argument("--tb-hidden", type=int, default=256)
    ap.add_argument("--tb-probe-hidden", type=int, default=128)
    ap.add_argument("--tb-minimax-steps", type=int, default=200)
    ap.add_argument("--tb-cotrain-interval", type=int, default=8)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--da-self-consistency", action="store_true",
                    help="diagnostic: also record how well the defense-aware "
                         "surrogate reconstructs its own calibration "
                         "distribution (end-to-end calibration-loop accuracy "
                         "and relative reconstruction MSE) in "
                         "defense-aware-baseline.json")
    ap.add_argument("--quick", action="store_true",
                    help="CPU smoke: tiny random GPT-2, synthetic "
                         "documents, minimal budgets (engineering tier); "
                         "dims fall back to {4, 8}")
    ap.add_argument("--dry-run", action="store_true",
                    help="write the full matrix structure (PLANNED "
                         "cells) without torch")
    args = ap.parse_args(argv)
    if not args.dims or any(d < 1 for d in args.dims):
        ap.error("dims must be positive integers")
    if args.quick:
        args.device = "cpu"
        args.seq_len, args.calibration_seqs = 8, 6
        args.train_steps = 12
        args.forward_candidates, args.beam_width = 4, 2
        args.sipit_steps, args.attack_sequences = 4, 4
        args.dims = [4, 8]
        args.tb_hidden, args.tb_probe_hidden = 16, 16
        args.tb_minimax_steps = 2
        args.tb_cotrain_interval = 4
    if len(set(args.seeds)) != len(args.seeds):
        ap.error("seeds must be unique")
    if args.dry_run and args.quick:
        ap.error("--dry-run already avoids torch; drop --quick")
    return args


def main(argv=None):
    return run(build_parser(argv))


if __name__ == "__main__":
    raise SystemExit(main())
