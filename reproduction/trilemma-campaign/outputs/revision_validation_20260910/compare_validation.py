"""Compare newly computed verification outputs against untouched retained results."""
from pathlib import Path
import hashlib
import json
import sys
from datetime import datetime, timezone

import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from pcr.data import load_corpus, make_probe_batch, split_corpus
from pcr.utils import derive_seed


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def load(path):
    return json.loads(path.read_text())


original_known_path = ROOT / "outputs/adversarial_review_2026-09-08/within_run_probe.json"
original_known = load(original_known_path)
recomputed_known = load(OUT / "within_run_probe.json")
execution = load(OUT / "within_run_execution.json")
assert original_known["records"] == recomputed_known["records"]
assert sha(original_known_path) == execution["original_result_sha256_before"]
index = {r["path"]: r for r in load(ROOT / "evidence/release-20260910-v1.json")["files"]}
captures = {}
for row in recomputed_known["records"]:
    rel = f"outputs/rate_sweep/trained-medium-high-q{row['q']}-s{row['seed']}/transcript_tensors.pt"
    captures[rel] = sha(ROOT / rel)
    assert captures[rel] == row["capture_sha256"] == index[rel]["sha256"]

original_p5_path = ROOT / "outputs/delegation/delegation_analysis.json"
original_p5 = load(original_p5_path)
new_p5 = load(OUT / "p5_nonprivacy_analysis.json")
assert sha(original_p5_path) == index[str(original_p5_path.relative_to(ROOT))]["sha256"]
checks = {}
for key, cell in original_p5["cells"].items():
    new_cell = new_p5["cells"][key]
    checks[key] = {field: cell[field] == new_cell[field]
                   for field in ("span", "start", "defense", "remote_depth", "delta_u_by_seed", "gr_mean", "gr_per_seed")}
    checks[key].update({field: cell["gates"][field] == new_cell["gates"][field]
                       for field in ("utility", "delegated_share", "gr_mean", "du_mean", "share_mean")})
assert all(all(v.values()) for v in checks.values())
assert not new_p5["operational_failures"]

corpus = load_corpus(ROOT / "data/raw/tinyshakespeare.txt")
_, heldout = split_corpus(corpus)
generator = torch.Generator().manual_seed(derive_seed(0, "probe", 0))
starts = torch.randint(0, len(heldout) - 128, (8,), generator=generator).tolist()
assert torch.equal(torch.stack([heldout[s:s + 128].long() for s in starts]), make_probe_batch(heldout, 8, 128)[0])
nonoverlap = all(abs(a - b) >= 128 for i, a in enumerate(starts) for b in starts[i + 1:])
assert nonoverlap

prop = load(OUT / "p5_property_refit.json")
assert all(c["aggregation_matches_at_1e-10"] for c in prop["cells"].values())
assert all(prop["source_unchanged"].values())
assert prop["input_hashes_unchanged_after_refit"]
comparison = {
    "status": "post-hoc verification only; original observations and historical classifications preserved",
    "checked_at": datetime.now(timezone.utc).isoformat(),
    "known_plaintext": {
        "all_16_seed_records_exactly_match": True,
        "all_128_fold_accuracies_exactly_match": True,
        "original_result_unchanged": True,
        "original_extra_top_level_keys": sorted(set(original_known) - set(recomputed_known)),
        "fixed_window_check": {"window_length": 128, "heldout_relative_start_offsets": starts, "corpus_absolute_start_offsets": [int(len(corpus) * 0.9) + s for s in starts], "nonoverlapping": nonoverlap, "reconstructed_windows_match_make_probe_batch": True, "scope": "These are eight windows of the same held-out corpus region, not eight independent new documents."},
        "captures_sha256_matching_original_results_and_release_index": captures,
        "means": {str(q): {field: sum(r[field] for r in recomputed_known["records"] if r["q"] == q) / 8
                           for field in ("accuracy", "null_accuracy", "majority_baseline")}
                  for q in (4, 32)},
    },
    "p5_nonprivacy": {
        "all_selected_fields_match_exactly": True,
        "checks_by_cell": checks,
        "operational_failures": new_p5["operational_failures"],
        "privacy_and_joint_verdicts_excluded": "The analyzer was run with --no-privacy-scoring. Its missing-score privacy flags and resulting triangle verdict are NOT recomputed privacy evidence.",
    },
    "p5_property": {
        "records": len(prop["records"]),
        "maximum_absolute_difference_from_original_all_target_per_seed_scores": max(c["maximum_absolute_aggregation_difference"] for c in prop["cells"].values()),
        "qualified_target": prop["qualification"]["qualified_targets"],
        "all_14_cells_punctuation_worst_seed_above_002": all(not c["qualified_punctuation_gate_pass"] for c in prop["cells"].values()),
        "original_qualification_reused_not_refitted": True,
        "structure_and_gradient_banks_not_refitted": True,
    },
    "generated_files_sha256": {p.name: sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name not in {"comparison.json", "README.md"}},
}
(OUT / "comparison.json").write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
print("Exact agreement: known-plaintext records, all selected P5 nonprivacy fields, and all 112 P5 property aggregates.")
