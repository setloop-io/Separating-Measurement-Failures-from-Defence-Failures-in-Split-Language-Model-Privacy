"""Optional separately versioned CodeBLEU from the exact retained decoded texts.

This never generates predictions. It does not replace the original native score.
Runtime scoring belongs to a separately recorded ORX postprocessing node.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path


METRIC_VERSION = "codebleu-default-whitespace.v1-from-saved-generations"


def extract_saved_texts(payload):
    if not isinstance(payload, list) or len(payload) != 1:
        raise ValueError("Require exactly one complete native utility evaluation")
    row = payload[0]
    if row.get("dataset") != "CodeAlpaca" or row.get("max_new_tokens") != 200:
        raise ValueError("Require the original CodeAlpaca generation scope")
    n = row.get("test_rows")
    calls = row.get("scoring_calls")
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0 or not isinstance(calls, list) or len(calls) != 10 * n:
        raise ValueError("Native utility does not retain ten exact calls per test row")
    if len(row.get("predicted_ids", [])) != n or len(row.get("target_ids", [])) != n:
        raise ValueError("Native generation/target coverage differs from scoring calls")
    documents = []
    for index in range(n):
        group = calls[index * 10:(index + 1) * 10]
        first = group[0]
        reference, prediction = first.get("references"), first.get("predictions")
        if (not isinstance(reference, list) or len(reference) != 1 or not isinstance(reference[0], str)
                or not isinstance(prediction, list) or len(prediction) != 1 or not isinstance(prediction[0], str)):
            raise ValueError("Each retained native call must identify one exact reference/prediction text")
        if any(call.get("references") != reference or call.get("predictions") != prediction
               or call.get("lang") != "python" or call.get("weights") != [0.25, 0.25, 0.25, 0.25] for call in group):
            raise ValueError("Native per-document call texts/language do not agree")
        native = max(float(call["result"]["codebleu"]) for call in group)
        if not math.isfinite(native):
            raise ValueError("Non-finite native utility score")
        documents.append({"ordinal": index, "reference": reference[0], "prediction": prediction[0], "native_codebleu": native})
    expected = sum(document["native_codebleu"] for document in documents) / n
    if abs(expected - float(row["native_result"]["acc"])) > 1e-12:
        raise ValueError("Native utility aggregation differs from retained call results")
    return documents


def rescore(path):
    from codebleu import calc_codebleu
    pins = {"codebleu": "0.7.0", "tree-sitter": "0.22.3", "tree-sitter-python": "0.21.0"}
    if {key: importlib.metadata.version(key) for key in pins} != pins:
        raise RuntimeError("Corrected scorer does not match the admitted metric stack")
    raw = Path(path).read_bytes()
    documents = extract_saved_texts(json.loads(raw))
    rows = []
    for document in documents:
        score = calc_codebleu([document["reference"]], [document["prediction"]], lang="python",
                             weights=(0.25, 0.25, 0.25, 0.25), tokenizer=None)
        rows.append({"ordinal": document["ordinal"], "native_codebleu": document["native_codebleu"],
                     "corrected_components": score,
                     "reference_text_sha256": hashlib.sha256(document["reference"].encode()).hexdigest(),
                     "prediction_text_sha256": hashlib.sha256(document["prediction"].encode()).hexdigest()})
    return {"metric_version": METRIC_VERSION, "status": "RESCORED_SAME_SAVED_GENERATIONS", "versions": pins,
            "source_utility_path": str(path), "source_utility_sha256": hashlib.sha256(raw).hexdigest(),
            "scorer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "documents": len(rows),
            "native_codebleu_macro": sum(row["native_codebleu"] for row in rows) / len(rows),
            "corrected_codebleu_macro": sum(row["corrected_components"]["codebleu"] for row in rows) / len(rows),
            "rows": rows, "tokenizer_convention": "CodeBLEU default whitespace split; Python grammar; equal component weights",
            "native_score_replaced": False, "new_model_generation_or_attack": False,
            "threshold_or_acceptance_policy_selected": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--utility", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        documents = extract_saved_texts(json.loads(args.utility.read_text()))
        print(json.dumps({"status": "SAVED_TEXT_CONTRACT_VALID", "documents": len(documents), "metric_executed": False}))
        return 0
    if not os.environ.get("ORX_RUN_ID") or args.output is None:
        raise RuntimeError("Actual rescoring requires a recorded ORX postprocessing node and --output")
    result = rescore(args.utility)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("metric_version", "status", "documents", "native_codebleu_macro", "corrected_codebleu_macro")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
