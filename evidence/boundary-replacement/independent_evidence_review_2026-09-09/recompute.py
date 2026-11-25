"""Read-only numerical audit; run from the repository root with Python 3."""
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {}


def read(path):
    p = ROOT / path
    raw = p.read_bytes()
    SOURCES[str(p.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def summary(values):
    n = len(values)
    m = statistics.mean(values)
    t = {3: 4.302652730, 5: 2.776445105}[n]
    half = t * statistics.stdev(values) / math.sqrt(n)
    return dict(n=n, values=values, mean=m,
                seed_t95=[m-half, m+half])


out = {"scope": "Recomputed from cell JSON, not paper prose. Seed intervals condition on the fixed corpus window; no model reruns.",
       "historical_confirmation": {}, "clean_controls": {}}
for corpus, splits in [("wikitext2", [19, 21]), ("penn_treebank", [14, 17])]:
    for split in splits:
        paths = sorted((ROOT / f"outputs/phase06/{corpus}_confirmation_rerun").glob(f"confirm_split{split}_s*.json"))
        cells = [read(p) for p in paths]
        out["historical_confirmation"][f"{corpus}_split{split}"] = summary([c["eval_loss_delta"] for c in cells])
    modes = {}
    for mode in ["trained_remote", "trained_local", "frozen_random_remote", "zero_output"]:
        paths = sorted((ROOT / f"outputs/phase08_clean_endpoints/clean_controls/{corpus}").glob(f"{mode}_s*.json"))
        modes[mode] = {c["seed"]: c for c in [read(p) for p in paths]}
    out["clean_controls"][corpus] = {
        "mode_delta": {mode: summary([cells[s]["eval_loss_delta"] for s in sorted(cells)]) for mode, cells in modes.items()},
        "paired_remote_minus_control_nll": {
            mode: summary([modes["trained_remote"][s]["candidate_eval_loss"]-cells[s]["candidate_eval_loss"] for s in sorted(cells)])
            for mode, cells in modes.items() if mode != "trained_remote"}}

intact = read("outputs/phase09_economics_quality/quality_intact.json")["results"]
composed = read("outputs/phase09_economics_quality/quality_composed.json")["results"]
out["quality"] = {
    "nll_delta": composed["utility"]["mean_nll"]-intact["utility"]["mean_nll"],
    "perplexity_relative_increase": composed["utility"]["perplexity"]/intact["utility"]["perplexity"]-1,
    "answer_top1_delta_pp": 100*(composed["answer_top1"]["token_match_rate"]-intact["answer_top1"]["token_match_rate"]),
    "intact": intact, "composed": composed}
out["economics"] = {}
for name, path in [("original", "outputs/phase09_economics_quality/clean21_econ.json"),
                   ("reprofile", "outputs/phase09_economics_quality/energy_reprofile/clean21/clean21_econ.json")]:
    c = read(path)
    t = c["transport"]
    out["economics"][name] = {
        "remote_operator_count_whole_session": t["actual_remote_flops"],
        "trusted_operator_count_one_training_step": t["trusted_flops_per_step"],
        "remote_total_over_extrapolated_trusted_training_percent": 100*t["actual_remote_flops"]/(t["trusted_flops_per_step"]*c["steps"]),
        "ratio_limitation": "Mismatched scopes: remote whole session, trusted representative step extrapolated across training; not actual workload share.",
        "wire_bytes": t["total_wire_bytes"], "reported_eval_time_ratio": c["eval_time_ratio"],
        "decoder_zero_minus_candidate_nll": c["decoder_zero_eval_loss"]-c["candidate_eval_loss"]}
out["large_model_geometry"] = {}
for p in sorted((ROOT / "outputs/phase06/qwen35_27b_reproduction").glob("isolation*.json")):
    c = read(p)
    plan = next(e for e in c["log"] if e["event"] == "split_plan")
    out["large_model_geometry"][p.name] = {"split_after": c["args"]["split_after"],
        "resume_after": c["args"]["resume_after"], "plan": plan}
out["clean_audit_support"] = {}
for p in sorted((ROOT / "paper-data/collected/diagnostic/clean_endpoint_audits").glob("*_bundles.json")):
    out["clean_audit_support"][p.name] = read(p)["support_leak"]
out["source_sha256"] = SOURCES
target = Path(__file__).with_name("recomputed_evidence.json")
target.write_text(json.dumps(out, indent=2) + "\n")
print(f"Wrote {target}; {len(SOURCES)} source artifacts hashed.")
