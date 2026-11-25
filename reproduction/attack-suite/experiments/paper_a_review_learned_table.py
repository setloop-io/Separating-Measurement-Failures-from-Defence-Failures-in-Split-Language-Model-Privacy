"""Verify and tabulate the retained single-seam learned comparison.

This is an offline table export. It imports no model code, executes no attack,
and does not rerun the bootstrap. Existing document metrics are used only to
check the twenty retained means and the already computed contrast estimates.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "build/v4-release-20260920"
VICTIMS = (101, 202, 303, 404, 505, 606, 707, 808, 909, 1010)
DEPTHS = (3, 6)
GRANTS = ("public", "exact")
ATTACKS = (17, 29)
ARMS = ("forward_k32", "sipit_steps128", "joint_true_k32", "bisr_sip", "bisr_bre_bf")
LABELS = ("Activation k32", "SipIt-128", "Joint k32", "Learned SIP", "Adapted BRE-bf")
EVIDENCE_REVISION = "d415492614cb1eb760926bf2e01a33b7fb024124"
EVIDENCE_MANIFEST = "881bdf02d6777e29449ac3bbef9f2112725fd75dda78052757e18baf0d221e42"
EVIDENCE_URL = (
    "https://huggingface.co/datasets/Setloop/llm-attacker-paper-b/tree/"
    + EVIDENCE_REVISION + "/v4-adaptation-20260920/evidence"
)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=ROOT / "artifacts/paper-a-review-round2-20261020/learned-comparison")
    ap.add_argument("--tex", type=Path, default=ROOT / "paper/paper_a/review-learned.tex")
    args = ap.parse_args()
    args.output = args.output.resolve()
    args.tex = args.tex.resolve()
    require(not args.output.exists() or not any(args.output.iterdir()), "Output must be new or empty")
    sources = []

    def record(path, role, expected=None):
        digest = sha(path)
        require(expected is None or digest == expected, "Source digest mismatch: " + str(path))
        sources.append({"path": str(path.relative_to(ROOT)), "sha256": digest,
                        "role": role, "expected_sha256": expected})
        return digest

    attestation_path = RELEASE / "REPRODUCTION-VERIFIED.json"
    integrity_path = RELEASE / "support/INTEGRITY.json"
    publication_path = RELEASE / "evidence-status-verification.json"
    attestation, integrity, publication = map(read, (attestation_path, integrity_path, publication_path))
    for path, role in ((attestation_path, "retained_cpu_reproduction_attestation"),
                       (integrity_path, "retained_analysis_digest_bindings"),
                       (publication_path, "historical_immutable_publication_verification"),
                       (RELEASE / "support/REPRODUCE.md", "retained_reproduction_instructions"),
                       (RELEASE / "support/PROTOCOL-ERRATUM.md", "retained_protocol_interpretation_erratum")):
        record(path, role)
    require(attestation["ok"] is True and attestation["analysis_reproduced"] is True
            and attestation["gpu_prediction_replayed"] is False, "Wrong reproduction attestation")
    require(integrity["ok"] is True and integrity["analysis_status"] == "COMPLETE"
            and integrity["shared_trained_grus"] == 4
            and integrity["candidate_joint_cells"] == 80, "Incomplete retained analysis")
    require(publication["ok"] is True and publication["errors"] == []
            and publication["revision"] == EVIDENCE_REVISION
            and publication["manifest_sha256"] == EVIDENCE_MANIFEST, "Wrong immutable release record")
    for name, digest in integrity["analysis_files"].items():
        record(RELEASE / "analysis" / name, "locally_reverified_published_analysis", digest)
    require(set(attestation["analysis"]["deterministic_sha256"]) == set(integrity["analysis_files"]) - {"result.json"},
            "Reproduction file coverage differs")
    for name, digest in attestation["analysis"]["deterministic_sha256"].items():
        require(digest == integrity["analysis_files"][name], "Reproduction digest differs: " + name)

    # Read recipe bytes directly from the retained execution archive; never import them.
    archive = RELEASE / "support/execution-source-2d0f58b.tar.gz"
    record(archive, "retained_execution_source_archive")
    archive_names = ("experiments/v4_adaptation.v1.json", "experiments/v4_sip_preparation_plan.json",
                     "experiments/v4-preparation-pins/inputs.json", "experiments/v4_sip_prepare.py",
                     "experiments/v4_adaptation_predict.py", "attacker/bisr_fidelity.py",
                     "experiments/v4-final-admission/launch.json")
    with tarfile.open(archive) as tf:
        members = {name: tf.extractfile(name).read() for name in archive_names}
    for name, data in members.items():
        record(ROOT / name, "byte_equal_to_retained_execution_source", hashlib.sha256(data).hexdigest())
    plan = json.loads(members[archive_names[0]])
    preparation = json.loads(members[archive_names[1]])
    auxiliary = json.loads(members[archive_names[2]])
    launch = json.loads(members[archive_names[-1]])
    require(plan["matrix"]["victim_seeds"] == list(VICTIMS)
            and plan["matrix"]["depths"] == list(DEPTHS)
            and plan["matrix"]["knowledge"] == list(GRANTS)
            and plan["matrix"]["attack_seeds"] == list(ATTACKS), "Unexpected protocol matrix")
    require(plan["algorithm"]["training_epochs"] == 20
            and plan["algorithm"]["training_batch_size"] == 32
            and plan["algorithm"]["training_lr"] == .001
            and plan["algorithm"]["stage_order"] == ["SIP", "GM32", "SM800"]
            and plan["algorithm"]["opt_batch"] == 1
            and plan["gradient_conversion"]["multiplier"] == 256, "Recipe differs")
    require(preparation["epochs"] == 20 and preparation["batch_size"] == 32
            and auxiliary["auxiliary_articles"] == 112 and auxiliary["auxiliary_windows"] == 15604,
            "Auxiliary preparation differs")
    require(launch["gradient_multiplier"] == 256 and len(launch["cells"]) == 80,
            "Admitted launch differs")
    require(hashlib.sha256(members[archive_names[-1]]).hexdigest() == integrity["launch_sha256"],
            "Launch digest does not bind retained analysis")

    summaries = read(RELEASE / "analysis/arm-summaries.json")
    result = read(RELEASE / "analysis/result.json")
    contrasts = read(RELEASE / "analysis/contrasts.json")
    analysis_integrity = read(RELEASE / "analysis/integrity.json")
    require(result["status"] == "COMPLETE" and result["published_setting_reproduced"] is False,
            "Wrong result scope")
    require(summaries == result["arm_summaries"], "Summary copies disagree")
    require(result["secondary_family_size"] == 15 and len(contrasts) == 16,
            "Original contrast family differs")
    require(result["bootstrap"]["replicates"] == 100000 and result["bootstrap"]["seed"] == 93281,
            "Original bootstrap specification differs")
    expected_groups = set(itertools.product(DEPTHS, GRANTS, ARMS))
    by_group = {(r["depth"], r["knowledge"], r["arm"]): r for r in summaries}
    require(len(summaries) == len(by_group) == 20 and set(by_group) == expected_groups,
            "Incomplete twenty-mean table")

    input_path = ROOT / "experiments/v4-analysis-inputs/v3-input.json"
    record(input_path, "same_original_evaluation_inputs", analysis_integrity["input_pins"]["v3-input.json"])
    docs = read(input_path)["evaluation"]
    doc_ids = {doc["id"] for doc in docs}
    require(len(doc_ids) == len(docs) == 256 and all(len(d["tokens"]) == 32 for d in docs), "Wrong evaluation set")
    expected_cells = set(itertools.product(VICTIMS, DEPTHS, GRANTS, ATTACKS, ARMS))
    seen, correct = defaultdict(set), defaultdict(int)
    rows_checked = 0
    with (RELEASE / "analysis/document-metrics.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            key = (row["victim_seed"], row["depth"], row["knowledge"], row["attack_seed"], row["arm"])
            require(key in expected_cells and row["document_id"] in doc_ids, "Unexpected metric identity")
            require(row["document_id"] not in seen[key], "Duplicate metric row")
            require(row["token_count"] == 32 and type(row["correct_tokens"]) is int
                    and 0 <= row["correct_tokens"] <= 32
                    and row["token_accuracy"] == row["correct_tokens"] / 32, "Invalid retained token metric")
            seen[key].add(row["document_id"])
            correct[key] += row["correct_tokens"]
            rows_checked += 1
    require(set(seen) == expected_cells and all(ids == doc_ids for ids in seen.values())
            and rows_checked == 102400, "Retained document metrics incomplete")
    for (depth, grant, arm), row in by_group.items():
        vm = {str(v): sum(correct[(v, depth, grant, a, arm)] for a in ATTACKS) / (2 * 256 * 32)
              for v in VICTIMS}
        require(vm == row["victim_means"], "Per-victim summary mismatch")
        require(sum(vm.values()) / len(VICTIMS) == row["token_accuracy"], "Arm mean mismatch")
    primaries = [c for c in contrasts if c["primary"]]
    require(len(primaries) == 1 and primaries[0] == result["primary"], "Primary copies disagree")
    for c in contrasts:
        left = by_group[(c["depth"], c["knowledge"], c["left"])]
        right = by_group[(c["depth"], c["knowledge"], c["right"])]
        require(math.isclose(left["token_accuracy"] - right["token_accuracy"], c["difference"], abs_tol=1e-15),
                "Original contrast estimate disagrees with means")
        require(c["lower"] <= c["difference"] <= c["upper"] and c["alpha"] == (.05 if c["primary"] else .05 / 15),
                "Original interval or family membership invalid")
    primary = primaries[0]
    require((primary["depth"], primary["knowledge"], primary["left"], primary["right"])
            == (6, "public", "bisr_bre_bf", "joint_true_k32"), "Wrong primary contrast")
    identity = result["sip_grant_pair_identity"]
    require(identity["pairs_checked"] == identity["pairs_with_all_256_predictions_identical"] == 40
            and identity["independent_attack_replications_added_by_identical_grant_pairs"] == 0,
            "Wrong SIP grant-pair interpretation")
    for depth in DEPTHS:
        require(by_group[(depth, "public", "bisr_sip")]["victim_means"]
                == by_group[(depth, "exact", "bisr_sip")]["victim_means"], "SIP grant means differ")

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "arm-means.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["depth", "client_copy", "method", "source_arm", "token_recovery_percent"])
        for depth, grant in itertools.product(DEPTHS, GRANTS):
            for arm, label in zip(ARMS, LABELS):
                writer.writerow([depth, "trained" if grant == "exact" else grant, label, arm,
                                 100 * by_group[(depth, grant, arm)]["token_accuracy"]])
    for name in ("arm-summaries.json", "contrasts.json", "result.json"):
        shutil.copyfile(RELEASE / "analysis" / name, args.output / name)
    for path in (attestation_path, integrity_path, publication_path, RELEASE / "support/REPRODUCE.md",
                 RELEASE / "support/PROTOCOL-ERRATUM.md"):
        shutil.copyfile(path, args.output / path.name)

    table_rows = []
    for depth, grant in itertools.product(DEPTHS, GRANTS):
        numbers = " & ".join(f"{100 * by_group[(depth, grant, arm)]['token_accuracy']:.2f}" for arm in ARMS)
        table_rows.append(f"{depth} & {'Trained' if grant == 'exact' else 'Public'} & {numbers} " + r"\\")
    tex = r"""% Generated by experiments/paper_a_review_learned_table.py from retained published results.
\subsection{Existing learned-inversion comparison}
\label{sec:learnedcomparison}
The companion study's separately pre-specified single-seam GPT-2 adaptation \cite{llmattacker2026framework} already evaluates learned inversion on these same observations. We tabulate its retained results here to make that comparison inspectable. It reuses all twenty original undefended activation/gradient captures, the same 256 evaluation passages and ten victim seeds, at depths three and six with public and trained client copies. Two attack seeds (17 and 29) give 80 executions, each retaining learned SIP and backward--forward refinement (BRE-bf); the original activation, SipIt and joint controls are reused without regeneration.

Learned SIP is a causal GRU inverter with hidden width 256 and output dropout 0.1, distinct from the sequential SipIt decoder. Four GRUs (two depths by two attack seeds) are trained for twenty epochs, batch size 32 and AdamW learning rate $10^{-3}$, on 15{,}604 non-overlapping 32-token windows from 112 separate WikiText-103 articles. These auxiliary articles are disjoint from the evaluation inputs and differ from the 32-document calibration set of the ridge decoder. BRE-bf applies 32 gradient-matching steps followed by 800 activation-matching steps, optimising one passage at a time. The original captured gradients are multiplied by 256 to reverse their full-evaluation-set loss normalisation; their original bytes are preserved. BRE uses publicly released pretrained downstream weights as a surrogate; the trained-copy grant supplies only the client prefix, not the victim's trained server weights. The adaptation uses eager attention, while exact-head activation replay uses the original SDPA path. The replay does not establish full downstream-gradient regeneration.

\begin{table}[H]
\centering\footnotesize
\begin{tabular}{@{}llrrrrr@{}}
\toprule
Depth & Copy & Activation & SipIt-128 & Joint & Learned SIP & BRE-bf \\
 & & $k{=}32$ & & $k{=}32$ & & \\
\midrule
""" + "\n".join(table_rows) + r"""
\bottomrule
\end{tabular}
\caption{Existing single-seam adapted comparison: token recovery (\%). Each stratum averages two attack seeds, ten victim runs and the same 256 passages. The first three methods are frozen original controls; the final two are the separately evaluated learned adaptation. Budgets differ. Means alone do not establish statistical differences or a uniform ranking.}
\label{tab:learnedcomparison}
\end{table}

"""
    diff, lower, upper = (100 * primary[k] for k in ("difference", "lower", "upper"))
    tex += (
        "The adaptation's original primary contrast, BRE-bf minus joint at depth six/public, is "
        f"${diff:.4f}$ percentage points (original 95\\% interval $[{lower:.4f},{upper:.4f}]$). "
        "This interval uses 100,000 crossed document/victim bootstrap draws with seed 93281, after averaging attack seeds. "
        "It belongs to the adaptation's separate pre-specified family (one primary and fifteen Bonferroni-corrected secondaries), not the original 47-secondary family of this paper. "
        "The four GRUs are shared across victim runs and grants. Learned SIP predictions are identical across public/trained grants in all forty paired cells, adding no independent attacker replications. "
        "The intervals condition on those fixed inverters and the auxiliary corpus, and do not cover auxiliary-data selection or attacker retraining. "
        "This is a comparison at distinct declared budgets, rather than equal total compute, original published-setting BiSR, or Vec2Text. "
        "It supplies no universal attack ranking. The retained "
        f"\\href{{{EVIDENCE_URL}}}{{immutable evidence and reproduction instructions}} "
        "preserve all means, original intervals, source and input bindings. No new model execution or bootstrap was needed to add this table.\n"
    )
    args.tex.write_text(tex)
    audit = {
        "status": "VERIFIED_RETAINED_RESULT_TABLE_EXPORTED", "new_model_execution": False,
        "new_bootstrap": False, "new_inferential_contrasts": False,
        "twenty_means_reconstructed_from_retained_metrics": True, "document_metric_rows_checked": rows_checked,
        "complete_metric_cells_checked": len(expected_cells), "original_contrast_estimates_checked": len(contrasts),
        "primary_contrast_unchanged": primary, "secondary_family_size_unchanged": 15,
        "execution_archive_recipe_matches_local_files": list(archive_names),
        "fresh_local_checks": ["All five analysis-file SHA-256 bindings", "Four deterministic output hashes match retained CPU reproduction attestation",
                               "Complete 102400-row retained metric coverage and twenty means", "Original sixteen contrast point estimates and original family membership",
                               "Frozen execution archive recipe, launch and auxiliary declarations"],
        "historical_release_verification": {"record": str(publication_path.relative_to(ROOT)),
                                           "revision": EVIDENCE_REVISION, "manifest_sha256": EVIDENCE_MANIFEST,
                                           "manifest_reverified_in_this_export": False,
                                           "note": "The release MANIFEST and full raw archive are not re-downloaded or reverified here; the retained successful publication and CPU reproduction records are preserved separately."},
        "sources": sources, "tex": {"path": str(args.tex.relative_to(ROOT)), "sha256": sha(args.tex)},
        "limitations": result["limitations"],
    }
    record(Path(__file__), "table_export_source")
    dump(args.output / "table-verification.json", audit)
    (args.output / "SOURCE-MANIFEST.sha256").write_text("".join(f"{s['sha256']}  {s['path']}\n" for s in sources))
    (args.output / "README.md").write_text(
        "# Retained learned-comparison table\n\n"
        "This export adds the existing single-seam learned SIP/BRE-bf comparison to Paper A without model execution, "
        "a new bootstrap or new inferential contrasts. All twenty arm means were checked against all 102,400 retained document-metric rows. "
        "All five analysis outputs match their retained published digest bindings. The sixteen existing contrast estimates and family membership were checked; their bootstrap intervals were copied unchanged.\n\n"
        "[Table verification](table-verification.json) records the exact source paths and hashes. [Arm means](arm-means.csv) distinguishes SipIt from learned SIP. "
        "[Original contrasts](contrasts.json), [result summary](result.json), [CPU reproduction attestation](REPRODUCTION-VERIFIED.json), "
        "[recipe/reproduction instructions](REPRODUCE.md) and [protocol interpretation erratum](PROTOCOL-ERRATUM.md) preserve the evidence and its scope.\n\n"
        f"[Immutable published evidence]({EVIDENCE_URL}); original release manifest SHA-256 `{EVIDENCE_MANIFEST}`. "
        "The successful release-manifest and full-archive verification are historical retained records, not new checks by this table exporter. "
        "No download or model dependency is needed for the local table export.\n"
    )
    (args.output / "MANIFEST.sha256").write_text("".join(
        f"{sha(p)}  {p.name}\n" for p in sorted(args.output.iterdir()) if p.is_file() and p.name != "MANIFEST.sha256"))
    print(json.dumps({"status": audit["status"], "means": len(summaries), "metric_rows": rows_checked,
                      "tex": str(args.tex), "tex_sha256": sha(args.tex)}))


if __name__ == "__main__":
    main()
