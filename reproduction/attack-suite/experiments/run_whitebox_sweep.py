"""White-box-aware attacker addendum for the bottleneck sweep.

POST-HOC ADDENDUM pass over an existing run_bottleneck_sweep.py output
directory: it does not modify, re-run, or re-label any base cell. For
every defended cell (pca-compress / tb-frozen / tb-cotrain at each
dim/split/seed) it GRANTS THE ATTACKER THE ACTUAL TRAINED DEFENSE
CHECKPOINT — the encoder weights + provenance sidecar for tb-*
(captures/trained_bottleneck_encoder.pt), the fitted PCA basis for
pca-compress (captures/pca_compress_encoder.pt) — mirroring the defense
framework's own assumption that "the current transform is necessarily
visible to the executor". This is a STRICTLY STRONGER, clearly labeled
access condition than the base matrix's defense-aware surrogate arm
(defense class + k + public calibration only); results land in a
separate whitebox tier under <root>/whitebox/ so the matched-access
identity of the original matrix is untouched.

Arms per cell (budgets matched to the sweep's raised budgets: candidates
32 / beam 4 / SIPIT 128):

  whitebox-inverter      a small MLP inverter (k -> hidden -> H) trained
                         on public-calibration head-forward activations
                         pushed through the TRUE defense map (encoder
                         forward, then learn the inverse), under
                         fork_rng + manual_seed; the learned inverse
                         reconstructs the cell's evaluation wires, then
                         the standard forward-beam and SIPIT kernels run
                         on the reconstruction
  whitebox-optimization  gradient-based input optimization against the
                         true encoder map: candidate continuous seam
                         inputs are optimized (Adam) so the frozen true
                         encoder's output matches each released z; the
                         optimized map is decoded via the public head
                         (nearest-embedding snap readout plus the
                         forward-beam kernel over its scores)

Scoring is position-aligned token accuracy against the cell's canonical
truth, exactly like the base sweep's baselines. Utility is NOT
recomputed: the attack cannot change the defense's training-time
utility, so each whitebox cell carries the base cell's utility_after /
utility_gate verbatim.

Fail-closed: a defended cell whose checkpoint or provenance sidecar is
missing/hash-mismatched is written as a FAILED whitebox cell (never
dropped) and the run exits nonzero after the remaining cells finish.

Co-trained encoders (tb-cotrain): the encoder is updated during victim
finetuning while only the FINAL checkpoint is saved, and the base sweep
captures training-phase wires only — so no capture exists for which the
saved checkpoint is the capture-time encoder. Those cells are recorded
NOT_EVALUABLE with the reason (a planned exclusion, never silently
evaluated against a mismatched encoder, never counted as FAILED).

--adaptive-e2e (opt-in) runs a THIRD, separate arm instead of the two
pipeline arms above: the adaptive end-to-end white-box attack. It
optimizes candidate token sequences (as continuous embeddings, snapped
to the vocabulary at readout) STRAIGHT THROUGH the victim's frozen
client head and then the TRUE defense transform against the
training-phase wire end-to-end — minimizing
||encoder(head(x)) - z_released|| over the whole sequence jointly, with
no intermediate decode stages (no learned inverter, no ridge map, no
beam or SIPIT head kernels; the head forward pass is the victim's own
client-side stack, not a decode stage). Budget mapping: steps_per_wire
= the sweep's SIPIT step count (128), spent as joint full-sequence Adam
steps through the client head and the true encoder (vs SIPIT's
steps-per-token head-side budget) — one full-sequence head forward and
one head backward per step — plus one vocabulary-wide Euclidean snap
readout per wire. Cells land in a separate whitebox-e2e tier under
<root>/whitebox-e2e/ with their own access record (strictly stronger,
same labeling conventions); without the flag the runner's behavior is
bit-identical. Fail-closed per cell
(never dropped); tb-cotrain cells remain NOT_EVALUABLE under the same
encoder-version mismatch rule.

--dry-run enumerates the planned whitebox cells from the root without
torch (checkpoint presence is checked from the filesystem only).
--quick runs the torch path on CPU against a sweep directory produced by
run_bottleneck_sweep.py --quick (tiny model, synthetic corpus; the
corpus is re-encoded from the root's recorded corpus file, so no model
weights or tokenizer downloads are needed in either mode).
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from attacker.evidence import file_sha256  # noqa: E402
from run_bottleneck_sweep import SCHEMA as BASE_SCHEMA  # noqa: E402
from run_defended_pilot import _partition  # noqa: E402

SCHEMA = "llm_attacker.whitebox_sweep.v1"
CELL_SCHEMA = "llm_attacker.whitebox_sweep_cell.v1"
ACCESS_SCHEMA = "llm_attacker.whitebox_sweep_access.v1"

E2E_SCHEMA = "llm_attacker.whitebox_sweep_e2e.v1"
E2E_CELL_SCHEMA = "llm_attacker.whitebox_sweep_e2e_cell.v1"
E2E_ACCESS_SCHEMA = "llm_attacker.whitebox_sweep_e2e_access.v1"

CHECKPOINT_NAMES = {
    "pca-compress": "pca_compress_encoder.pt",
    "tb-frozen": "trained_bottleneck_encoder.pt",
    "tb-cotrain": "trained_bottleneck_encoder.pt",
}

_INVERTER_SEED_XOR = 0xB07  # inverter training stream, distinct per cell
_E2E_SEED_XOR = 0xE2E  # adaptive-e2e init stream, distinct per cell

COTRAIN_NOT_EVALUABLE_REASON = (
    "tb-cotrain co-trains the encoder DURING victim finetuning "
    "(splitbench interleaves a minimax encoder update every "
    "tb_cotrain_interval steps), but only the FINAL co-trained encoder "
    "checkpoint is saved. The base sweep captures training-phase fwd "
    "wires only, so every capture was produced by an EARLIER encoder "
    "version than the saved checkpoint and no post-training/inference "
    "capture exists for which the saved checkpoint is the authoritative "
    "encoder. Evaluating these captures against the final checkpoint "
    "would silently attack with a mismatched encoder, so the cell is "
    "NOT_EVALUABLE. A future capture rig that stores per-update "
    "encoder versions (or post-training inference captures) can lift "
    "this exclusion.")
COTRAIN_METHOD = "tb-cotrain"


# ---------------------------------------------------------------------------
# Torch-free cell enumeration / access spec


def cell_checkpoint(cell_dir, method):
    return cell_dir / "captures" / CHECKPOINT_NAMES[method]


def whitebox_cell_relpath(split_depth, seed, method, dim):
    return f"whitebox/split{split_depth}/seed-{seed}/{method}-{dim}"


def whitebox_access_spec(base_access, base_cell_rel, checkpoint_rel):
    """The surrogate arm's access PLUS the true defense checkpoint.

    A strictly stronger, clearly labeled condition; kept structurally
    separate (own schema + whitebox_extension) so it can never be
    confused with the base matrix's matched access."""
    spec = copy.deepcopy(base_access)
    spec["schema"] = ACCESS_SCHEMA
    spec["tier"] = "whitebox"
    spec["whitebox_extension"] = {
        "granted": "the defended cell's ACTUAL trained defense checkpoint "
                   "and provenance sidecar (encoder weights for tb-*, "
                   "fitted PCA basis for pca-compress), in addition to "
                   "everything the defense-aware surrogate arm holds",
        "strength": "strictly stronger than the surrogate arm's "
                    "defense-class + k + public-calibration knowledge",
        "rationale": "mirrors the defense framework's assumption that "
                     "the current transform is necessarily visible to "
                     "the executor",
        "base_cell": base_cell_rel,
        "defense_checkpoint": checkpoint_rel,
        "arms": {
            "whitebox_inverter": "MLP inverse of the TRUE defense map "
                                 "trained on public-calibration "
                                 "activations pushed through the true "
                                 "encoder; forward-beam + SIPIT on the "
                                 "reconstruction",
            "whitebox_optimization": "Adam input optimization against "
                                     "the frozen true encoder's "
                                     "input-output map (steps_per_wire = "
                                     "the sweep's SIPIT steps budget), "
                                     "decoded via the public head"},
        "caveat": "white-box recovery is measured under strictly "
                  "stronger access; it does not constrain black-box "
                  "recovery of the same wire, nor the surrogate arm or "
                  "any defense-internals-hidden adversary"}
    return spec


def whitebox_e2e_cell_relpath(split_depth, seed, method, dim):
    return f"whitebox-e2e/split{split_depth}/seed-{seed}/{method}-{dim}"


def whitebox_e2e_access_spec(base_access, base_cell_rel, checkpoint_rel):
    """Access record for the adaptive end-to-end arm: the same true
    defense checkpoint grant as the whitebox tier, PLUS the explicitly
    adaptive condition that gradients flow through the TRUE
    training-phase transform itself (no surrogate, no decode pipeline).
    Structurally separate (own schema + tier) like the whitebox tier."""
    spec = copy.deepcopy(base_access)
    spec["schema"] = E2E_ACCESS_SCHEMA
    spec["tier"] = "whitebox-e2e"
    spec["whitebox_extension"] = {
        "granted": "the defended cell's ACTUAL trained defense checkpoint "
                   "and provenance sidecar (encoder weights for tb-*, "
                   "fitted PCA basis for pca-compress), used as the exact "
                   "training-phase transform the attack differentiates "
                   "through, in addition to everything the defense-aware "
                   "surrogate arm holds",
        "strength": "strictly stronger than the surrogate arm's "
                    "defense-class + k + public-calibration knowledge, "
                    "and adaptive where the whitebox pipeline arms are "
                    "not: the attack is customized end-to-end against "
                    "the true transform",
        "rationale": "mirrors the defense framework's assumption that "
                     "the current transform is necessarily visible to "
                     "the executor, and evaluates the adaptive adversary "
                     "who optimizes straight through it",
        "base_cell": base_cell_rel,
        "defense_checkpoint": checkpoint_rel,
        "arms": {
            "adaptive_e2e": "joint Adam optimization of each wire's full "
                            "continuous token-embedding sequence through "
                            "the victim's frozen client head and then "
                            "the TRUE encoder against the released z "
                            "(minimize ||encoder(head(x)) - z_released||), "
                            "snapped to the vocabulary by a single "
                            "Euclidean nearest-embedding readout; no "
                            "inverter, ridge map, beam or SIPIT stages"},
        "budget_mapping": "steps_per_wire = the sweep's SIPIT step count "
                          "(128 at the raised budgets): exactly "
                          "sipit_steps joint full-sequence Adam steps "
                          "through the client head and the true encoder "
                          "per wire, vs SIPIT's steps-per-TOKEN head-side "
                          "budget; each step costs one full-sequence head "
                          "forward and one head backward, and the single "
                          "terminal snap costs n_sequences * seq_len * "
                          "vocab vocabulary dot products with no further "
                          "head forward calls",
        "caveat": "adaptive white-box recovery is measured under "
                  "strictly stronger access; it does not constrain "
                  "black-box recovery of the same wire, nor the "
                  "surrogate arm or any defense-internals-hidden "
                  "adversary"}
    return spec


def enumerate_cells(root):
    """Torch-free: defended cells of the base plan + checkpoint presence.

    Returns (plan, cells) where each cell dict carries path/split/seed/
    method/dim plus base_status, checkpoint (relative path) and
    checkpoint_present."""
    root = Path(root)
    plan = json.loads((root / "plan.json").read_text())
    if plan.get("schema") != BASE_SCHEMA:
        raise ValueError(f"{root}: not a {BASE_SCHEMA} sweep directory")
    cells = []
    for entry in plan["cells"]:
        if entry["method"] == "none":
            continue
        cell_dir = root / entry["path"]
        base_status = None
        cell_json = cell_dir / "cell.json"
        if cell_json.is_file():
            try:
                base_status = json.loads(cell_json.read_text()).get("status")
            except ValueError:
                base_status = "INVALID"
        ckpt = cell_checkpoint(cell_dir, entry["method"])
        cells.append({"path": entry["path"], "split": entry["split"],
                      "seed": entry["seed"], "method": entry["method"],
                      "dim": entry["dim"], "base_status": base_status,
                      "checkpoint": str(ckpt.relative_to(root)),
                      "checkpoint_present": ckpt.is_file(),
                      "whitebox_path": whitebox_cell_relpath(
                          entry["split"], entry["seed"], entry["method"],
                          entry["dim"])})
    return plan, cells


# ---------------------------------------------------------------------------
# Torch path


def _load_true_defense(cell_dir, method):
    """The cell's ACTUAL trained defense map (fail closed on absence)."""
    from attacker import trained_defenses as td
    ckpt = cell_checkpoint(cell_dir, method)
    if not ckpt.is_file():
        raise RuntimeError(f"true defense checkpoint missing: {ckpt}")
    if method == "pca-compress":
        defense = td.PcaCompress.load(ckpt)
    else:
        defense = td.TrainedBottleneck.load(ckpt)
    if defense.bottleneck_dim <= 0:
        raise RuntimeError(f"{ckpt}: malformed checkpoint")
    return defense


def _train_inverter(z_cal, h_cal, hidden, steps, seed, device):
    """Small MLP inverse of the TRUE defense map, fit on public
    calibration only (fork_rng + manual_seed; deterministic on CPU)."""
    import torch
    k = z_cal.shape[-1]
    width = h_cal.shape[-1]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed ^ _INVERTER_SEED_XOR)
        inverter = torch.nn.Sequential(
            torch.nn.Linear(k, hidden), torch.nn.GELU(),
            torch.nn.Linear(hidden, width)).to(device)
        opt = torch.optim.Adam(inverter.parameters(), lr=1e-2)
        for _ in range(steps):
            opt.zero_grad()
            loss = (inverter(z_cal) - h_cal).square().mean()
            loss.backward()
            opt.step()
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite whitebox-inverter training")
    for p in inverter.parameters():
        p.requires_grad_(False)
    return inverter.eval()


def whitebox_attacks(args, root, entry, cal, budgets):
    """Both white-box-aware arms against one defended cell's wires."""
    import torch
    from attacker import captures
    from attacker.attacks import seam_inversion as si
    from attacker.reconstruction_baselines import (
        MeteredHead, beam_reconstruct, candidate_ids, forward_scores,
        sipit_reconstruct)
    method, dim = entry["method"], entry["dim"]
    split_depth, seed = entry["split"], entry["seed"]
    cell_dir = root / entry["path"]
    defense = _load_true_defense(cell_dir, method)
    encode = defense.encode
    device = args.device
    head_pt = root / f"split{split_depth}" / f"public-head-{seed}.pt"
    split, embedding = si._load_head_pt(str(head_pt), device)
    split.model.requires_grad_(False)
    embedding = embedding.detach()
    head = MeteredHead(split)
    with torch.no_grad():
        head(ids=cal[:1])
        hc = head(ids=cal).reshape(-1, embedding.shape[-1])
        decoder, _ = si._ridge_fit(hc, embedding[cal.flatten()], 0)
        z_cal = encode(hc)
    cap_dir = cell_dir / "captures"
    records = [(m, p) for m, p in captures.scan_captures(str(cap_dir),
                                                         "training")
               if m.get("phase") == "fwd"][:budgets["attack_sequences"]]
    if not records:
        raise RuntimeError(f"no fwd captures in {cap_dir}")
    wire = torch.cat(
        [captures.load_tensor(p).reshape(1, -1, embedding.shape[-1])
         for _, p in records]).to(device)
    truth = torch.load(cap_dir / "canonical.pt", map_location=device,
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
    # -- whitebox-inverter arm -------------------------------------------
    inverter = _train_inverter(z_cal, hc, args.inverter_hidden,
                               args.inverter_steps, seed, device)
    with torch.no_grad():
        cal_hat = inverter(z_cal)
        inv_decoder, _ = si._ridge_fit(cal_hat,
                                       embedding[cal.flatten()], 0)
        hidden_hat = inverter(wire[..., :dim])
    with head.measure() as work, torch.no_grad():
        fwd = forward_scores(hidden_hat, embedding, inv_decoder, head)
        cand = candidate_ids(fwd, None, budgets["forward_candidates"],
                             "forward")
        pred, _ = beam_reconstruct(head, hidden_hat, cand,
                                   budgets["beam_width"])
    inv = {"true_encoder_granted": True,
           "inverter": {"architecture": f"mlp {dim}->"
                                        f"{args.inverter_hidden}->"
                                        f"{hc.shape[-1]} gelu",
                        "trained_on": "public calibration pushed through "
                                      "the TRUE defense map",
                        "steps": args.inverter_steps,
                        "final_mse": float(loss_mse(inverter, z_cal, hc))},
           "forward": {**score(pred),
                       "candidate_count": budgets["forward_candidates"],
                       "beam_width": budgets["beam_width"],
                       "work": {k: v for k, v in vars(work).items()}}}
    with head.measure() as work:
        pred, _ = sipit_reconstruct(head, hidden_hat, embedding,
                                    steps=budgets["sipit_steps"], seed=seed)
    inv["sipit"] = {**score(pred), "steps_per_token": budgets["sipit_steps"],
                    "work": {k: v for k, v in vars(work).items()}}
    out["whitebox_inverter"] = inv
    # -- whitebox-optimization arm ----------------------------------------
    z_wire = wire[..., :dim].detach()
    with torch.no_grad():
        init = defense.decode(wire)
    variable = init.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([variable], lr=args.optimize_lr)
    for _ in range(budgets["sipit_steps"]):
        opt.zero_grad()
        loss = (encode(variable) - z_wire).square().mean()
        loss.backward()
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite whitebox-optimization")
        opt.step()
    optimized = variable.detach()
    with head.measure() as work, torch.no_grad():
        norm_e = embedding.square().sum(-1)
        distance = (norm_e[None, None]
                    + optimized.square().sum(-1, keepdim=True)
                    - 2 * optimized @ embedding.T)
        head.work.vocabulary_dot_products += int(
            optimized.shape[0] * optimized.shape[1] * len(embedding))
        snap = distance.argmin(-1)
        fwd = forward_scores(optimized, embedding, decoder, head)
        cand = candidate_ids(fwd, None, budgets["forward_candidates"],
                             "forward")
        pred, _ = beam_reconstruct(head, optimized, cand,
                                   budgets["beam_width"])
    work_dict = {k: v for k, v in vars(work).items()}
    out["whitebox_optimization"] = {
        "true_encoder_granted": True,
        "objective": "match the released z by optimizing continuous seam "
                     "inputs through the frozen true encoder (Adam)",
        "steps_per_wire": budgets["sipit_steps"],
        "budget_match": "one encoder-side optimization budget equal to "
                        "the sweep's SIPIT step count, replacing "
                        "per-token head optimization",
        "final_encoder_mse": float(loss.item()),
        "snap": {**score(snap)},
        "forward": {**score(pred),
                    "candidate_count": budgets["forward_candidates"],
                    "beam_width": budgets["beam_width"],
                    "work": work_dict}}
    del split, head, embedding
    return out


def loss_mse(inverter, z_cal, hc):
    import torch
    with torch.no_grad():
        return (inverter(z_cal) - hc).square().mean().item()


def adaptive_e2e_attack(args, root, entry, budgets):
    """Adaptive end-to-end arm against one defended cell's wires.

    Optimizes the whole continuous token-embedding sequence of each wire
    jointly, straight through the victim's frozen client head and then
    the TRUE defense transform, minimizing
    ||encoder(head(x)) - z_released||; readout is a single Euclidean
    nearest-embedding snap to the vocabulary. The head forward pass is
    the victim's own client-side stack, which is what makes the variable
    a token-embedding quantity the snap can read; it is not a decode
    stage (no inverter, ridge map, beam or SIPIT kernels)."""
    import torch
    from attacker import captures
    from attacker.attacks import seam_inversion as si
    from attacker.reconstruction_baselines import MeteredHead
    method, dim = entry["method"], entry["dim"]
    split_depth, seed = entry["split"], entry["seed"]
    cell_dir = root / entry["path"]
    defense = _load_true_defense(cell_dir, method)
    encode = defense.encode
    device = args.device
    head_pt = root / f"split{split_depth}" / f"public-head-{seed}.pt"
    split, embedding = si._load_head_pt(str(head_pt), device)
    split.model.requires_grad_(False)
    embedding = embedding.detach()
    head = MeteredHead(split)
    cap_dir = cell_dir / "captures"
    records = [(m, p) for m, p in captures.scan_captures(str(cap_dir),
                                                         "training")
               if m.get("phase") == "fwd"][:budgets["attack_sequences"]]
    if not records:
        raise RuntimeError(f"no fwd captures in {cap_dir}")
    wire = torch.cat(
        [captures.load_tensor(p).reshape(1, -1, embedding.shape[-1])
         for _, p in records]).to(device)
    truth = torch.load(cap_dir / "canonical.pt", map_location=device,
                       weights_only=True)[:len(records)]
    z_wire = wire[..., :dim].detach()
    n_seq, length, _ = wire.shape
    with torch.random.fork_rng(devices=[]), head.measure() as opt_work:
        torch.manual_seed(seed ^ _E2E_SEED_XOR)
        init_ids = torch.randint(len(embedding), (n_seq, length))
        variable = embedding[init_ids.to(device)].detach().clone()
        variable.requires_grad_(True)
        opt = torch.optim.Adam([variable], lr=args.optimize_lr)
        for _ in range(budgets["sipit_steps"]):
            opt.zero_grad()
            # token embeddings -> client head -> seam -> true encoder:
            # encode() takes the seam hidden state, never an embedding
            loss = (encode(head(embeddings=variable))
                    - z_wire).square().mean()
            loss.backward()
            head.backward(n_seq, length)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite adaptive-e2e optimization")
            opt.step()
        if variable.grad is None:
            raise RuntimeError(
                f"true encoder of {entry['path']} is not "
                f"input-differentiable; the adaptive-e2e arm requires "
                f"true input gradients through the transform")
    opt_work_dict = {k: v for k, v in vars(opt_work).items()}
    optimized = variable.detach()
    with head.measure() as work, torch.no_grad():
        norm_e = embedding.square().sum(-1)
        distance = (norm_e[None, None]
                    + optimized.square().sum(-1, keepdim=True)
                    - 2 * optimized @ embedding.T)
        head.work.vocabulary_dot_products += int(
            n_seq * length * len(embedding))
        pred = distance.argmin(-1)
    if truth.shape != pred.shape:
        raise RuntimeError("canonical truth is not row-aligned to "
                           "predictions")
    correct = (pred == truth)
    snap = {"accuracy": correct.float().mean().item(),
            "n_tokens": int(truth.numel()),
            "n_sequences": int(truth.shape[0]),
            "work": {k: v for k, v in vars(work).items()}}
    del split, head, embedding
    return {"adaptive_e2e": {
        "true_encoder_granted": True,
        "adaptive": True,
        "objective": "minimize ||encoder(head(x)) - z_released|| over "
                     "the wire's full continuous token-embedding "
                     "sequence, straight through the victim's frozen "
                     "client head and then the TRUE defense transform "
                     "(Adam); no intermediate decode stages",
        "init": "uniform random vocabulary embeddings (seeded per-cell "
                "stream under fork_rng + manual_seed)",
        "steps_per_wire": budgets["sipit_steps"],
        "budget_match": "one joint full-sequence optimization budget "
                        "equal to the sweep's SIPIT step count, vs "
                        "SIPIT's steps-per-token head-side budget; each "
                        "step spends one full-sequence head forward and "
                        "one head backward",
        "final_encoder_mse": float(loss.item()),
        "optimization": {"steps": budgets["sipit_steps"],
                         "n_sequences": n_seq,
                         "work": opt_work_dict},
        "snap": snap}}


def run_e2e_cell(args, root, entry, budgets):
    import torch
    began = time.perf_counter()
    wb_dir = root / entry["whitebox_e2e_path"]
    wb_dir.mkdir(parents=True, exist_ok=True)
    base_cell = json.loads((root / entry["path"] / "cell.json").read_text())
    access = whitebox_e2e_access_spec(base_cell["access"], entry["path"],
                                      entry["checkpoint"])
    torch.manual_seed(entry["seed"])
    attacks = adaptive_e2e_attack(args, root, entry, budgets)
    if args.device == "cuda":
        torch.cuda.synchronize()
    ckpt = root / entry["checkpoint"]
    cell = {
        "schema": E2E_CELL_SCHEMA, "status": "COMPLETE",
        "tier": "whitebox-e2e",
        "access_condition": "adaptive white-box-aware (true encoder "
                            "granted, differentiated through end-to-end) "
                            "— strictly stronger than the base matrix's "
                            "matched access; NOT comparable as a "
                            "matched-access row",
        "split": entry["split"], "seed": entry["seed"],
        "method": entry["method"], "dim": entry["dim"],
        "base_cell": entry["path"],
        "defense_checkpoint": {"path": entry["checkpoint"],
                               "sha256": file_sha256(ckpt)},
        "wall_seconds": time.perf_counter() - began,
        "attack_seconds": time.perf_counter() - began,
        "cuda_peak_allocated_bytes": (
            torch.cuda.max_memory_allocated() if args.device == "cuda"
            else None),
        "utility_after": base_cell["utility_after"],
        "utility_gate": base_cell.get("utility_gate"),
        "utility_source": "carried verbatim from the base cell; the "
                          "attack cannot change training-time utility",
        "access": access,
        "attacks": attacks,
        "evidence": "cell.json.evidence.json",
    }
    from attacker.evidence import EvidenceLogger
    with EvidenceLogger(str(wb_dir / "cell.json"),
                        node_role=f"whitebox-e2e-sweep/{entry['method']}",
                        extra={"cell": E2E_CELL_SCHEMA}):
        (wb_dir / "cell.json").write_text(json.dumps(cell, indent=2,
                                                     allow_nan=False))
    print(f"[whitebox-e2e] split={entry['split']} seed={entry['seed']} "
          f"method={entry['method']} dim={entry['dim']} "
          f"wall={cell['wall_seconds']:.1f}s", flush=True)
    return cell


E2E_WRITER_KW = {"path_key": "whitebox_e2e_path",
                 "access_fn": whitebox_e2e_access_spec,
                 "tier": "whitebox-e2e", "cell_schema": E2E_CELL_SCHEMA}


def run_e2e(args, root, plan, cells, budgets):
    """The adaptive end-to-end tier: separate whitebox-e2e tree, own
    access records, same fail-closed / NOT_EVALUABLE conventions."""
    for entry in cells:
        entry["whitebox_e2e_path"] = whitebox_e2e_cell_relpath(
            entry["split"], entry["seed"], entry["method"], entry["dim"])
    e2e_plan = {"schema": E2E_SCHEMA,
                "tier": "whitebox-e2e",
                "addendum_to": str(root),
                "base_plan_schema": BASE_SCHEMA,
                "purpose": "post-hoc ADAPTIVE white-box attacker addendum: "
                           "the attacker is granted each defended cell's "
                           "ACTUAL trained defense checkpoint and "
                           "optimizes candidate token sequences straight "
                           "through the TRUE training-phase transform "
                           "end-to-end (no decode pipeline); a strictly "
                           "stronger, clearly labeled access condition, "
                           "reported as a separate tier",
                "budgets": budgets,
                "budget_mapping": ("steps_per_wire = sipit_steps (the "
                                   "sweep's SIPIT step count): joint "
                                   "full-sequence Adam steps through the "
                                   "client head and the true encoder per "
                                   "wire, one head forward and one head "
                                   "backward each, plus one "
                                   "vocabulary-wide snap readout"),
                "arms": ["adaptive_e2e"],
                "runner": str(Path(__file__).name),
                "runner_sha256": file_sha256(__file__),
                "limits": [
                    "adaptive white-box recovery is a strictly stronger "
                    "access measurement; it does not constrain black-box "
                    "recovery of the same wire, nor the reverse",
                    "the addendum reuses the base cells' captured wires, "
                    "canonical truth and utility; nothing about victim "
                    "training is re-run",
                    "the arm's steps-per-wire budget is matched to the "
                    "sweep's SIPIT step count, but it is spent as "
                    "full-sequence gradient steps through the client "
                    "head and the encoder (SIPIT spends the same count "
                    "per token position at the head)"],
                "cells": []}
    for entry in cells:
        e2e_plan["cells"].append({
            "path": entry["whitebox_e2e_path"], "base_cell": entry["path"],
            "split": entry["split"], "seed": entry["seed"],
            "method": entry["method"], "dim": entry["dim"],
            "base_status": entry["base_status"],
            "checkpoint": entry["checkpoint"],
            "checkpoint_present": entry["checkpoint_present"]})
    if args.dry_run:
        for entry in cells:
            wb_dir = root / entry["whitebox_e2e_path"]
            wb_dir.mkdir(parents=True, exist_ok=True)
            if entry["method"] == COTRAIN_METHOD:
                status, extra = "NOT_EVALUABLE", {
                    "not_evaluable_reason": COTRAIN_NOT_EVALUABLE_REASON}
            else:
                status = ("PLANNED" if entry["checkpoint_present"]
                          and entry["base_status"] == "COMPLETE"
                          else "BLOCKED")
                extra = {}
            (wb_dir / "cell.json").write_text(json.dumps(
                {"schema": E2E_CELL_SCHEMA, "status": status,
                 "tier": "whitebox-e2e",
                 "split": entry["split"], "seed": entry["seed"],
                 "method": entry["method"], "dim": entry["dim"],
                 "base_cell": entry["path"],
                 "checkpoint": entry["checkpoint"],
                 "checkpoint_present": entry["checkpoint_present"],
                 **extra},
                indent=2))
        (root / "whitebox-e2e").mkdir(exist_ok=True)
        e2e_plan["not_evaluable"] = [
            {"path": e["whitebox_e2e_path"], "method": e["method"],
             "reason": COTRAIN_NOT_EVALUABLE_REASON}
            for e in cells if e["method"] == COTRAIN_METHOD]
        (root / "whitebox-e2e" / "plan.json").write_text(
            json.dumps(e2e_plan, indent=2))
        n_ready = sum(1 for e in cells
                      if e["checkpoint_present"]
                      and e["base_status"] == "COMPLETE"
                      and e["method"] != COTRAIN_METHOD)
        print(f"[whitebox-e2e] dry-run: {len(cells)} defended cells "
              f"({n_ready} ready, {len(e2e_plan['not_evaluable'])} "
              f"not evaluable) -> {root / 'whitebox-e2e'}", flush=True)
        return 0

    import torch
    torch.set_num_threads(args.threads)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    failures = []
    not_evaluable = []
    completed = 0
    for entry in cells:
        base_cell = None
        cell_json = root / entry["path"] / "cell.json"
        if cell_json.is_file():
            try:
                base_cell = json.loads(cell_json.read_text())
            except ValueError:
                base_cell = None
        if entry["method"] == COTRAIN_METHOD:
            # never silently attack training-phase captures with the
            # FINAL co-trained encoder (a mismatched version)
            write_not_evaluable_cell(root, entry,
                                     COTRAIN_NOT_EVALUABLE_REASON,
                                     base_cell, **E2E_WRITER_KW)
            not_evaluable.append(entry["whitebox_e2e_path"])
            continue
        if entry["base_status"] != "COMPLETE":
            write_failed_cell(root, entry,
                              f"base cell status "
                              f"{entry['base_status']!r}", base_cell,
                              **E2E_WRITER_KW)
            failures.append(entry["whitebox_e2e_path"])
            continue
        if not entry["checkpoint_present"]:
            write_failed_cell(root, entry,
                              f"true defense checkpoint missing: "
                              f"{entry['checkpoint']}", base_cell,
                              **E2E_WRITER_KW)
            failures.append(entry["whitebox_e2e_path"])
            continue
        try:
            run_e2e_cell(args, root, entry, budgets)
            completed += 1
        except Exception as exc:
            write_failed_cell(root, entry, f"{type(exc).__name__}: {exc}",
                              base_cell, **E2E_WRITER_KW)
            failures.append(entry["whitebox_e2e_path"])
    (root / "whitebox-e2e").mkdir(exist_ok=True)
    e2e_plan["status"] = "FAILED_CELLS" if failures else "COMPLETE"
    e2e_plan["not_evaluable"] = [
        {"path": e["whitebox_e2e_path"], "method": e["method"],
         "reason": COTRAIN_NOT_EVALUABLE_REASON}
        for e in cells if e["whitebox_e2e_path"] in set(not_evaluable)]
    (root / "whitebox-e2e" / "plan.json").write_text(
        json.dumps(e2e_plan, indent=2, allow_nan=False))
    print(f"[whitebox-e2e] {'FAILED_CELLS' if failures else 'COMPLETE'}: "
          f"{completed} ok, {len(failures)} failed, "
          f"{len(not_evaluable)} not evaluable -> "
          f"{root / 'whitebox-e2e'}", flush=True)
    if failures:
        raise ValueError(f"{len(failures)} whitebox-e2e cells FAILED "
                         f"(never dropped): {failures}")
    return 0


def run_whitebox_cell(args, root, entry, cal, budgets):
    import torch
    began = time.perf_counter()
    wb_dir = root / entry["whitebox_path"]
    wb_dir.mkdir(parents=True, exist_ok=True)
    base_cell = json.loads((root / entry["path"] / "cell.json").read_text())
    access = whitebox_access_spec(base_cell["access"], entry["path"],
                                  entry["checkpoint"])
    torch.manual_seed(entry["seed"])
    attacks = whitebox_attacks(args, root, entry, cal, budgets)
    if args.device == "cuda":
        torch.cuda.synchronize()
    ckpt = root / entry["checkpoint"]
    cell = {
        "schema": CELL_SCHEMA, "status": "COMPLETE",
        "tier": "whitebox",
        "access_condition": "white-box-aware (true encoder granted) — "
                            "strictly stronger than the base matrix's "
                            "matched access; NOT comparable as a "
                            "matched-access row",
        "split": entry["split"], "seed": entry["seed"],
        "method": entry["method"], "dim": entry["dim"],
        "base_cell": entry["path"],
        "defense_checkpoint": {"path": entry["checkpoint"],
                               "sha256": file_sha256(ckpt)},
        "wall_seconds": time.perf_counter() - began,
        "attack_seconds": time.perf_counter() - began,
        "cuda_peak_allocated_bytes": (
            torch.cuda.max_memory_allocated() if args.device == "cuda"
            else None),
        "utility_after": base_cell["utility_after"],
        "utility_gate": base_cell.get("utility_gate"),
        "utility_source": "carried verbatim from the base cell; the "
                          "attack cannot change training-time utility",
        "access": access,
        "attacks": attacks,
        "evidence": "cell.json.evidence.json",
    }
    from attacker.evidence import EvidenceLogger
    with EvidenceLogger(str(wb_dir / "cell.json"),
                        node_role=f"whitebox-sweep/{entry['method']}",
                        extra={"cell": CELL_SCHEMA}):
        (wb_dir / "cell.json").write_text(json.dumps(cell, indent=2,
                                                     allow_nan=False))
    print(f"[whitebox-sweep] split={entry['split']} seed={entry['seed']} "
          f"method={entry['method']} dim={entry['dim']} "
          f"wall={cell['wall_seconds']:.1f}s", flush=True)
    return cell


def write_failed_cell(root, entry, reason, base_cell=None, *,
                      path_key="whitebox_path",
                      access_fn=whitebox_access_spec,
                      tier="whitebox", cell_schema=CELL_SCHEMA):
    wb_dir = root / entry[path_key]
    wb_dir.mkdir(parents=True, exist_ok=True)
    access = access_fn(
        (base_cell or {}).get("access",
                              {"schema": "llm_attacker."
                                         "bottleneck_sweep_access.v1"}),
        entry["path"], entry["checkpoint"])
    cell = {"schema": cell_schema, "status": "FAILED", "tier": tier,
            "split": entry["split"], "seed": entry["seed"],
            "method": entry["method"], "dim": entry["dim"],
            "base_cell": entry["path"], "failure_reason": reason,
            "access": access}
    (wb_dir / "cell.json").write_text(json.dumps(cell, indent=2))


def write_not_evaluable_cell(root, entry, reason, base_cell=None, *,
                             path_key="whitebox_path",
                             access_fn=whitebox_access_spec,
                             tier="whitebox", cell_schema=CELL_SCHEMA):
    """A PLANNED exclusion, not a failure: the cell is recorded
    NOT_EVALUABLE with the reason, never evaluated against a mismatched
    encoder and never counted among the failed cells."""
    wb_dir = root / entry[path_key]
    wb_dir.mkdir(parents=True, exist_ok=True)
    access = access_fn(
        (base_cell or {}).get("access",
                              {"schema": "llm_attacker."
                                         "bottleneck_sweep_access.v1"}),
        entry["path"], entry["checkpoint"])
    cell = {"schema": cell_schema, "status": "NOT_EVALUABLE",
            "tier": tier,
            "split": entry["split"], "seed": entry["seed"],
            "method": entry["method"], "dim": entry["dim"],
            "base_cell": entry["path"],
            "not_evaluable_reason": reason,
            "access": access}
    (wb_dir / "cell.json").write_text(json.dumps(cell, indent=2))


def _encoded_calibration(root, plan, args):
    """Re-encode the public calibration partition from the root's
    recorded corpus (synthetic 'tokens' rows need no tokenizer)."""
    import torch
    corpus = Path(plan["corpus"]["path"])
    if not corpus.is_absolute():
        candidate = root / corpus
        corpus = candidate if candidate.is_file() else corpus
    docs = [json.loads(line) for line in
            corpus.read_text().splitlines() if line.strip()]
    budgets = plan["budgets"]
    seq_len = budgets["seq_len"]
    part = _partition(docs, budgets["data_seed"])["calibration"]
    tokenizer = None
    rows = []
    for doc in part:
        if "tokens" in doc:
            ids = doc["tokens"]
        elif args.quick:
            # the base sweep's quick corpus writes synthetic token ids as
            # space-joined text; no tokenizer needed
            ids = [int(t) for t in doc["text"].split()]
        else:
            if tokenizer is None:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(args.model)
            ids = tokenizer.encode(doc["text"], add_special_tokens=False)
        if len(ids) < seq_len:
            raise ValueError(f"document {doc['id']} shorter than seq_len")
        rows.append(torch.tensor([ids[:seq_len]], dtype=torch.long))
    return torch.cat(rows[:budgets["calibration_seqs"]]).to(args.device)


def run(args):
    root = Path(args.root)
    if not (root / "plan.json").is_file():
        raise ValueError(f"no plan.json under {root}")
    plan, cells = enumerate_cells(root)
    budgets = {"seq_len": plan["budgets"]["seq_len"],
               "data_seed": plan["budgets"]["data_seed"],
               "calibration_seqs": plan["budgets"]["calibration_seqs"],
               "attack_sequences": plan["budgets"]["attack_sequences"],
               "forward_candidates": (args.forward_candidates or
                                      plan["budgets"]["forward_candidates"]),
               "beam_width": (args.beam_width or
                              plan["budgets"]["beam_width"]),
               "sipit_steps": (args.sipit_steps or
                               plan["budgets"]["sipit_steps"])}
    if args.adaptive_e2e:
        return run_e2e(args, root, plan, cells, budgets)
    wb_plan = {"schema": SCHEMA,
               "tier": "whitebox",
               "addendum_to": str(root),
               "base_plan_schema": BASE_SCHEMA,
               "purpose": "post-hoc white-box-aware attacker addendum: the "
                          "attacker is granted each defended cell's ACTUAL "
                          "trained defense checkpoint; a strictly stronger "
                          "access condition, reported as a separate tier — "
                          "the base matrix's matched-access identity is "
                          "untouched",
               "budgets": budgets,
               "arms": ["whitebox_inverter", "whitebox_optimization"],
               "runner": str(Path(__file__).name),
               "runner_sha256": file_sha256(__file__),
               "limits": [
                   "white-box recovery is a strictly stronger access "
                   "measurement; it does not constrain black-box "
                   "recovery of the same wire, nor the reverse",
                   "the addendum reuses the base cells' captured wires, "
                   "canonical truth and utility; nothing about victim "
                   "training is re-run",
                   "the optimization arm's steps-per-wire budget is "
                   "matched to the sweep's SIPIT step count, but the two "
                   "arms spend different compute mixes (encoder-side vs "
                   "head-side)"],
               "cells": []}
    failures = []
    not_evaluable = []
    for entry in cells:
        wb_plan["cells"].append({
            "path": entry["whitebox_path"], "base_cell": entry["path"],
            "split": entry["split"], "seed": entry["seed"],
            "method": entry["method"], "dim": entry["dim"],
            "base_status": entry["base_status"],
            "checkpoint": entry["checkpoint"],
            "checkpoint_present": entry["checkpoint_present"]})
    if args.dry_run:
        for entry in cells:
            wb_dir = root / entry["whitebox_path"]
            wb_dir.mkdir(parents=True, exist_ok=True)
            if entry["method"] == COTRAIN_METHOD:
                status, extra = "NOT_EVALUABLE", {
                    "not_evaluable_reason": COTRAIN_NOT_EVALUABLE_REASON}
            else:
                status = ("PLANNED" if entry["checkpoint_present"]
                          and entry["base_status"] == "COMPLETE"
                          else "BLOCKED")
                extra = {}
            (wb_dir / "cell.json").write_text(json.dumps(
                {"schema": CELL_SCHEMA, "status": status, "tier": "whitebox",
                 "split": entry["split"], "seed": entry["seed"],
                 "method": entry["method"], "dim": entry["dim"],
                 "base_cell": entry["path"],
                 "checkpoint": entry["checkpoint"],
                 "checkpoint_present": entry["checkpoint_present"],
                 **extra},
                indent=2))
        (root / "whitebox").mkdir(exist_ok=True)
        wb_plan["not_evaluable"] = [
            {"path": e["whitebox_path"], "method": e["method"],
             "reason": COTRAIN_NOT_EVALUABLE_REASON}
            for e in cells if e["method"] == COTRAIN_METHOD]
        (root / "whitebox" / "plan.json").write_text(
            json.dumps(wb_plan, indent=2))
        n_ready = sum(1 for e in cells
                      if e["checkpoint_present"]
                      and e["base_status"] == "COMPLETE"
                      and e["method"] != COTRAIN_METHOD)
        print(f"[whitebox-sweep] dry-run: {len(cells)} defended cells "
              f"({n_ready} ready, {len(wb_plan['not_evaluable'])} "
              f"not evaluable) -> {root / 'whitebox'}", flush=True)
        return 0

    import torch
    torch.set_num_threads(args.threads)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    cal = _encoded_calibration(root, plan, args)
    completed = 0
    for entry in cells:
        base_cell = None
        cell_json = root / entry["path"] / "cell.json"
        if cell_json.is_file():
            try:
                base_cell = json.loads(cell_json.read_text())
            except ValueError:
                base_cell = None
        if entry["method"] == COTRAIN_METHOD:
            # never silently attack training-phase captures with the
            # FINAL co-trained encoder (a mismatched version)
            write_not_evaluable_cell(root, entry,
                                     COTRAIN_NOT_EVALUABLE_REASON,
                                     base_cell)
            not_evaluable.append(entry["whitebox_path"])
            continue
        if entry["base_status"] != "COMPLETE":
            write_failed_cell(root, entry,
                              f"base cell status "
                              f"{entry['base_status']!r}", base_cell)
            failures.append(entry["whitebox_path"])
            continue
        if not entry["checkpoint_present"]:
            write_failed_cell(root, entry,
                              f"true defense checkpoint missing: "
                              f"{entry['checkpoint']}", base_cell)
            failures.append(entry["whitebox_path"])
            continue
        try:
            run_whitebox_cell(args, root, entry, cal, budgets)
            completed += 1
        except Exception as exc:
            write_failed_cell(root, entry, f"{type(exc).__name__}: {exc}",
                              base_cell)
            failures.append(entry["whitebox_path"])
    (root / "whitebox").mkdir(exist_ok=True)
    wb_plan["status"] = "FAILED_CELLS" if failures else "COMPLETE"
    wb_plan["not_evaluable"] = [
        {"path": e["whitebox_path"], "method": e["method"],
         "reason": COTRAIN_NOT_EVALUABLE_REASON}
        for e in cells if e["whitebox_path"] in set(not_evaluable)]
    (root / "whitebox" / "plan.json").write_text(
        json.dumps(wb_plan, indent=2, allow_nan=False))
    print(f"[whitebox-sweep] {'FAILED_CELLS' if failures else 'COMPLETE'}: "
          f"{completed} ok, {len(failures)} failed, "
          f"{len(not_evaluable)} not evaluable -> "
          f"{root / 'whitebox'}", flush=True)
    if failures:
        raise ValueError(f"{len(failures)} whitebox cells FAILED (never "
                         f"dropped): {failures}")
    return 0


def build_parser(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True,
                    help="existing run_bottleneck_sweep.py output dir")
    ap.add_argument("--model", default="openai-community/gpt2",
                    help="tokenizer source when the corpus lacks pre-"
                         "tokenized rows (never needed for --quick roots)")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--forward-candidates", type=int, default=None,
                    help="override the root plan's beam budget")
    ap.add_argument("--beam-width", type=int, default=None)
    ap.add_argument("--sipit-steps", type=int, default=None)
    ap.add_argument("--inverter-hidden", type=int, default=256)
    ap.add_argument("--inverter-steps", type=int, default=512)
    ap.add_argument("--optimize-lr", type=float, default=0.05)
    ap.add_argument("--adaptive-e2e", action="store_true",
                    help="run the adaptive end-to-end white-box arm "
                         "INSTEAD of the pipeline arms: optimize token "
                         "sequences straight through the true defense "
                         "transform against the training-phase wire; "
                         "cells land in <root>/whitebox-e2e/ (without "
                         "this flag behavior is unchanged)")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--quick", action="store_true",
                    help="CPU path against a run_bottleneck_sweep.py "
                         "--quick fixture root (tiny budgets)")
    ap.add_argument("--dry-run", action="store_true",
                    help="torch-free: enumerate planned whitebox cells")
    args = ap.parse_args(argv)
    if args.quick:
        args.device = "cpu"
        args.inverter_hidden = 16
        args.inverter_steps = 8
    if args.dry_run and args.quick:
        ap.error("--dry-run already avoids torch; drop --quick")
    return args


def main(argv=None):
    return run(build_parser(argv))


if __name__ == "__main__":
    raise SystemExit(main())
